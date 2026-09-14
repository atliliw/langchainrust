#!/usr/bin/env bash
# 发布 workspace 全部 crate 到 crates.io —— tag 驱动发布(.github/workflows/cd.yml)的唯一入口。
#
# 为什么需要这个脚本:
#   根 Cargo.toml 是虚拟 workspace,根目录直接 `cargo publish` 会报
#   "current directory is part of a virtual workspace";23 个 crate 之间用
#   path+version 依赖,crates.io 要求依赖版本已存在于索引中,必须按拓扑序
#   「发一个 → 等索引 → 发下一个」交错发布(0.22.0~0.22.4 手工发布的教训)。
#
# 保证:
#   1. 发布顺序由 `cargo metadata` 的 workspace 内部依赖拓扑算出(Kahn),
#      facade crate(langchainrust)必然最后;不依赖人手维护成员列表顺序。
#   2. 幂等/可断点续跑:目标版本已存在于 crates.io sparse index 的 crate
#      自动跳过——crates.io 不允许覆盖已发布版本,中途失败后重跑本脚本即可。
#   3. 索引延迟兜底:依赖刚上传、对端索引尚未可见时,publish 校验会报
#      "no matching package",按指数退避重试,而不是把整批发稿留在半成态。
#
# 环境变量:
#   CARGO_REGISTRY_TOKEN  crates.io token(cd.yml 由 secret 注入)
#   PUBLISH_DRY_RUN=1     只跑 cargo publish --dry-run,不实际上传
#   PUBLISH_ARGS          透传给 cargo publish 的额外参数
#                         (本机镜像源场景可传 --registry crates-io;GitHub runner 默认即官方源)
#   PUBLISH_RETRIES       单 crate 发布重试次数(默认 6)
#   PUBLISH_WAIT_BASE     重试基础等待秒数(默认 20)
#
# 参考:crates.io sparse index 路径规则
#   1 字符名  /1/<name>;2 字符 /2/<name>;3 字符 /3/<首字符>/<name>;
#   更长      /<前2字符>/<第3-4字符>/<name>('-'/'_' 原样保留)。
set -euo pipefail

cd "$(dirname "$0")/.."

RETRIES="${PUBLISH_RETRIES:-6}"
WAIT_BASE="${PUBLISH_WAIT_BASE:-20}"
DRY_RUN="${PUBLISH_DRY_RUN:-0}"
PUBLISH_ARGS="${PUBLISH_ARGS:-}"
INDEX_BASE="https://index.crates.io"

# 选一个「真能运行」的 python3:Git Bash 里 WindowsApps/python3 可能是商店桩文件
# (command -v 能找到但执行即失败),需要用真实 -c 探测,失败再回退 python。
PYTHONBIN=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys' >/dev/null 2>&1; then
    PYTHONBIN="$cand"
    break
  fi
done

# T2(v0.23)发布纪律硬保证:真实发布只允许发生在 tag 触发的 GitHub Actions
# (cd.yml:pre-publish release-gate 全绿 → publish 调用本脚本)。0.22.4 绕过
# release-gate 手工发布,fastembed feature 编译错随包上线、docs.rs 自 0.20 起
# 一直红。本地仅允许 dry-run;确需在 CI 外真实发布(应急),必须显式
# ALLOW_LOCAL_PUBLISH=1——故意双保险,不写进常规文档。
if [[ "$DRY_RUN" != "1" && "${GITHUB_ACTIONS:-}" != "true" && "${ALLOW_LOCAL_PUBLISH:-}" != "1" ]]; then
  echo "ERROR: 禁止在 CI 之外真实发布(0.22.4 事故后收口,T2 v0.23)。" >&2
  echo "       正确流程:打 tag 推送 → cd.yml 的 release-gate --all-features 全绿 → CI 自动拓扑发布。" >&2
  echo "       本地验证请用 PUBLISH_DRY_RUN=1 bash scripts/publish-crates.sh" >&2
  exit 2
fi

# 输出:每行 "名称<TAB>版本<TAB>sparse index 相对路径",按依赖拓扑排序(被依赖者在前)。
topo_plan() {
  cargo metadata --format-version 1 --no-deps | "$PYTHONBIN" -c '
import json, os, sys
from collections import deque

meta = json.load(sys.stdin)

# cargo 各版本 pkgid 形态不一(新版 "name version (path+..)",旧版 "path+..#ver",
# workspace_members 与 package.id 还可能不一致),改用「manifest 在 workspace 根下」
# 判定成员,绕开 id 字符串匹配。
ws_root = os.path.normcase(os.path.abspath(meta["workspace_root"]))

def under_root(path):
    p = os.path.normcase(os.path.abspath(path))
    return p == ws_root or p.startswith(ws_root + os.sep)

pkgs = {p["id"]: p for p in meta["packages"]
        if under_root(os.path.dirname(p["manifest_path"]))}

# 包目录 -> id;旧版 metadata 的内部依赖只给目录(path 字段),不给 pkgid。
dir_to_id = {}
for pid, p in pkgs.items():
    d = os.path.normcase(os.path.abspath(os.path.dirname(p["manifest_path"])))
    dir_to_id[d] = pid

def index_path(name):
    if len(name) == 1:
        return "1/" + name
    if len(name) == 2:
        return "2/" + name
    if len(name) == 3:
        return "3/" + name[0] + "/" + name
    return name[0:2] + "/" + name[2:4] + "/" + name

# 邻接表:edge dep -> dependent(Kahn 排序后被依赖者先出队)。
# normal/build/dev 三类依赖全部计入:交错发布时 dev-dependency(如 lc-testkit)
# 也必须已在 crates.io 上,保守排序只会让被依赖者更早发布。
deps_of = {pid: set() for pid in pkgs}
indegree = {pid: 0 for pid in pkgs}
for pid, p in pkgs.items():
    for d in p["dependencies"]:
        dep_pid = None
        if d.get("pkg"):                      # 新版 cargo:直接给依赖 pkgid
            dep_pid = d["pkg"] if d["pkg"] in pkgs else None
        elif d.get("path"):                   # 旧版 cargo:按目录匹配
            dep_pid = dir_to_id.get(os.path.normcase(os.path.abspath(d["path"])))
        if dep_pid and pid not in deps_of[dep_pid]:
            deps_of[dep_pid].add(pid)
            indegree[pid] += 1

queue = deque(sorted(
    (pid for pid in pkgs if indegree[pid] == 0),
    key=lambda pid: pkgs[pid]["name"],
))
order = []
while queue:
    pid = queue.popleft()
    order.append(pid)
    for nxt in sorted(deps_of[pid], key=lambda p: pkgs[p]["name"]):
        indegree[nxt] -= 1
        if indegree[nxt] == 0:
            queue.append(nxt)

if len(order) != len(pkgs):
    sys.stderr.write("fatal: workspace internal dependency cycle detected\n")
    sys.exit(1)

for pid in order:
    p = pkgs[pid]
    name, version = p["name"], p["version"]
    print("\t".join([name, version, index_path(name)]))
'
}

# 版本是否已在 crates.io(稀疏索引每行一个 JSON 发布记录,按版本升序)。
# 注意不能直接 `curl | grep -q`:grep 命中即退、curl 收 SIGPIPE,pipefail 下
# 整条流水线随机返回非零(时序相关,实测 23 个只认出 1 个)。先取 body 再匹配。
# 网络失败时返回"未发布"也安全:publish 阶段 cargo 自己会再校验一次。
version_published() {
  local idx_path="$1" version="$2" body
  body="$(curl -fsS --http1.1 --max-time 45 --retry 4 --retry-all-errors --retry-delay 2 \
    "${INDEX_BASE}/${idx_path}" 2>/dev/null || true)"
  # 必须用 herestring 而不是 printf|grep -q:pipefail 下 grep 命中即退、写端收
  # SIGPIPE,整个函数随机返回非零(实测 23 个全 miss,而 facade 因 JSON 较短侥幸命中)。
  [ -n "$body" ] && grep -q "\"vers\":\"${version}\"" <<< "$body"
}

publish_one() {
  local name="$1" version="$2"
  local attempt=1 wait_s logf rc
  logf="$(mktemp)"
  while :; do
    echo ">>> [${name} ${version}] publish attempt ${attempt}/${RETRIES}"
    if [ "$DRY_RUN" = "1" ]; then
      # shellcheck disable=SC2086
      cargo publish -p "$name" --dry-run $PUBLISH_ARGS
      rm -f "$logf"
      return 0
    fi
    set +e
    # shellcheck disable=SC2086
    cargo publish -p "$name" $PUBLISH_ARGS 2>&1 | tee "$logf"
    rc="${PIPESTATUS[0]}"
    set -e
    if [ "$rc" -eq 0 ]; then
      rm -f "$logf"
      return 0
    fi
    # cargo 的权威判定:该版本已存在 → 按已发布处理(稀疏索引预判漏网/断点续跑)。
    # 绝不能重试:already-exists 永远不会自愈,重试只会把整批发布卡死在第一个 crate。
    if grep -q "already exists" "$logf"; then
      echo "--- [${name} ${version}] cargo 报告该版本已存在,按已发布跳过"
      rm -f "$logf"
      return 42  # 42 = 与调用方约定的"已存在跳过",区别于真发布成功(0)
    fi
    if [ "$attempt" -ge "$RETRIES" ]; then
      echo "!!! [${name} ${version}] failed after ${RETRIES} attempts" >&2
      rm -f "$logf"
      return 1
    fi
    wait_s=$(( WAIT_BASE * attempt ))
    echo "... [${name}] publish 失败(可能是依赖刚上传、索引尚未可见),${wait_s}s 后重试"
    sleep "$wait_s"
    attempt=$(( attempt + 1 ))
  done
}

main() {
  [ -n "$PYTHONBIN" ] || { echo "fatal: 需要 python3(或 python)计算发布拓扑" >&2; exit 1; }

  local plan
  plan="$(topo_plan)"
  local total
  total=$(printf '%s\n' "$plan" | grep -c .)
  echo "=== 发布计划:${total} 个 workspace crate(拓扑序)==="
  printf '%s\n' "$plan" | cut -f1 | nl -ba
  echo

  local n=0 skipped=0
  while IFS=$'\t' read -r name version idx_path; do
    # Windows 版 Python 文本模式 stdout 会把 \n 转 CRLF,末字段(idx_path)带 \r;
    # 统一剥掉,Linux(LF)上是空操作。
    name="${name%$'\r'}"; version="${version%$'\r'}"; idx_path="${idx_path%$'\r'}"
    n=$(( n + 1 ))
    if version_published "$idx_path" "$version"; then
      skipped=$(( skipped + 1 ))
      echo "--- [${n}/${total}] ${name} ${version} 已存在于 crates.io,跳过"
      continue
    fi
    echo "--- [${n}/${total}] 发布 ${name} ${version}"
    if publish_one "$name" "$version"; then
      :
    elif [ "$?" -eq 42 ]; then
      skipped=$(( skipped + 1 ))
    else
      echo "fatal: ${name} ${version} 发布失败,已中止;修复后重跑本脚本,已上传的 crate 会自动跳过" >&2
      exit 1
    fi
  done <<< "$plan"

  local published=$(( total - skipped ))
  if [ "$DRY_RUN" != "1" ] && [ "$published" -eq 0 ]; then
    echo "fatal: ${total} 个 crate 的当前版本全部已存在于 crates.io——发布前是否忘记统一 bump 版本号?" >&2
    exit 1
  fi
  echo "=== 完成:${total} 个 crate,跳过已发布 ${skipped} 个,新发布 ${published} 个 ==="
}

main "$@"
