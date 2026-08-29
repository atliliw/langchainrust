#!/usr/bin/env bash
# ============================================================================
# security-scan.sh — 危险模式静态扫描(advisory gate)
#
# 背景(0.18.1 ⏳ 待办):门禁补安全/质量静态分析。正则基线(非 semgrep),
# 覆盖:子进程命令拼接 / SQL 字符串拼接 / 路径拼接+读取 / URL 内嵌 secret /
# 硬编码 key 字面量。先落高危面:lc-tools(subprocess/ssrf/sql/url_fetch)、
# lc-providers(key 处理)、lc-mcp(transport/sandbox)。
#
# 语义:
#   BLOCK    —— 高风险模式;--fail-if-found 下任一命中 → exit 1(CI 门禁)
#   ADVISORY —— 需人工复核;只报告不阻塞
#
# 用法:
#   scripts/security-scan.sh                  # 打印报告;有 BLOCK 也 exit 0
#   scripts/security-scan.sh --fail-if-found  # 有 BLOCK 则 exit 1(CI 门禁)
#
# 设计约束:
#   - 纯 POSIX grep,无 cargo 依赖;本地(Git Bash)与 CI(ubuntu)均可运行
#   - 不含 --all-features 等环境相关行为,不受 windows-gnu / ort-sys 盲区影响
#   - allowlist 是「已人工复核接受」的已知模式,改动需更新注释里的原因
# ============================================================================
set -u

FAIL_IF_FOUND=false
[ "${1:-}" = "--fail-if-found" ] && FAIL_IF_FOUND=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 2

# 扫描目录(存在才扫;target/.git 不在其中)
SCAN_DIRS=()
for d in crates examples docs; do
  [ -d "$d" ] && SCAN_DIRS+=("$d")
done
if [ "${#SCAN_DIRS[@]}" -eq 0 ]; then
  echo "security-scan: 未找到 crates/examples/docs 目录,退出" >&2
  exit 2
fi

GITGREP=(grep -rn --exclude-dir=target --exclude-dir=.git)

# ---------------------------------------------------------------------------
# A5 已知接受 key(memory:用户认定这 6 处硬编码真实 key「没事」,非阻塞项;
# 含测试夹具里的假 key)。值级过滤:A5 命中行包含下列任一值则标 allowlisted。
# ---------------------------------------------------------------------------
KNOWN_ACCEPTED_KEYS=(
  "sk-6eb65fcf5d17491ca10b984efe1f43e7"   # 用户接受的真实 key(例程/测试夹具)
  "sk-abcdefghijklmnopqrstuvwxyz123456"   # 测试夹具假 key
)

# ---------------------------------------------------------------------------
# A2 已接受文件(人工复核:表名来自配置 self.table,非用户输入,SQL 其余值
# 走 $1 参数化;若未来表名可用户可控,须加白名单校验并移出此列表)。
# ---------------------------------------------------------------------------
A2_ALLOW_FILES=(
  "crates/lc-vector-stores/src/pgvector.rs"
)

# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
fail_count=0

report() { # <id> <severity> <label> <hits:multi-line>
  local id="$1" sev="$2" label="$3" hits="$4" n=0
  [ -n "$hits" ] && n="$(printf '%s\n' "$hits" | grep -c .)"
  if [ "$n" -eq 0 ]; then
    printf '  [OK] %-4s %-46s (0)\n' "$id" "$label"
  else
    printf '  [%s] %-4s %-46s (%d)\n' "$sev" "$id" "$label" "$n"
    printf '%s\n' "$hits" | sed 's/^/        /'
  fi
  if [ "$sev" = "BLOCK" ]; then
    # 仅统计未 allowlisted 的 BLOCK 行
    fail_count=$((fail_count + $(printf '%s\n' "$hits" | grep -v '\[allowlisted\]' | grep -c .)))
  fi
}

filter_a5() { # 过滤已知接受 key
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    local allowed=false
    for k in "${KNOWN_ACCEPTED_KEYS[@]}"; do
      case "$line" in *"$k"*) allowed=true ;; esac
    done
    if $allowed; then printf '%s  [allowlisted]\n' "$line"; else printf '%s\n' "$line"; fi
  done
}

filter_a2() { # 过滤已接受文件
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    local f="${line%%:*}" allowed=false
    for a in "${A2_ALLOW_FILES[@]}"; do [ "$f" = "$a" ] && allowed=true; done
    if $allowed; then printf '%s  [allowlisted]\n' "$line"; else printf '%s\n' "$line"; fi
  done
}

# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------
echo "== security-scan:危险模式静态扫描 =="
echo "   范围:${SCAN_DIRS[*]}  |  模式:$( [ "$FAIL_IF_FOUND" = true ] && echo '--fail-if-found' || echo 'advisory' )"
echo

# ---- A1 (ADVISORY) 子进程命令拼接 ----------------------------------------
# 仅覆盖真子进程面(std::process::Command)。redis-rs 的 cmd().arg() 是客户端
# 命令构建器,非子进程,不做拼接告警。当前 3 个文件即完整子进程面。
CMD_FILES="$(grep -rln 'Command::new' "${SCAN_DIRS[@]}" --include='*.rs' 2>/dev/null || true)"
A1_HITS=""
if [ -n "$CMD_FILES" ]; then
  A1_HITS="$(printf '%s\n' "$CMD_FILES" | while read -r f; do
    grep -HnE '\.arg\((format!|&)|\.args\((&|format!)' "$f" 2>/dev/null
  done | grep -v ': *0:' || true)"
fi
printf '  [..] A1   子进程命令参数拼接(ADVISORY)\n'
if [ -n "$CMD_FILES" ]; then
  printf '        Command::new 文件(子进程面,%d 个,需人工复核):\n' "$(printf '%s\n' "$CMD_FILES" | grep -c .)"
  printf '%s\n' "$CMD_FILES" | sed 's/^/          /'
else
  echo '        Command::new 文件:无'
fi
if [ -n "$A1_HITS" ]; then
  printf '%s\n' "$A1_HITS" | sed 's/^/          !/'
else
  echo '        参数拼接:无'
fi
echo

# ---- A2 (BLOCK) SQL 字符串拼接 -------------------------------------------
# format!/concat! 直接把值拼进 SQL 语句体(非参数化)。命中即审查。
A2_HITS="$("${GITGREP[@]}" -E 'format!\s*\(\s*"(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|MERGE)|concat!\s*\(.*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)' --include='*.rs' "${SCAN_DIRS[@]}" 2>/dev/null | filter_a2 || true)"
report A2 BLOCK 'SQL 字符串拼接(应参数化)' "$A2_HITS"
echo

# ---- A3 (BLOCK) 路径拼接 + 读取 ------------------------------------------
# 文件操作直接吃 format!/拼接路径,无 normalize/沙箱校验 → 路径穿越面。
A3_HITS="$("${GITGREP[@]}" -E '(read_to_string|fs::read|fs::write|fs::File::open|File::open|std::fs::)[[:space:]]*\([[:space:]]*(format!|concat!)|PathBuf::from\([^)]*\)[[:space:]]*\.join\(' --include='*.rs' "${SCAN_DIRS[@]}" 2>/dev/null || true)"
report A3 BLOCK '路径拼接后文件操作(需 normalize/沙箱)' "$A3_HITS"
echo

# ---- A4 (ADVISORY) URL query 内嵌 secret ---------------------------------
# URL 字符串里带 ?key= / &token= 等查询参数(secret 应走 header)。
A4_HITS="$("${GITGREP[@]}" -E '"https?://[^" ]*\?[^" ]*(key|token|secret|password|api_key|apikey)=' --include='*.rs' "${SCAN_DIRS[@]}" 2>/dev/null || true)"
report A4 ADVISORY 'URL query 内嵌 secret(应走 header)' "$A4_HITS"
echo

# ---- A5 (BLOCK) 硬编码 key 字面量 ----------------------------------------
A5_HITS="$("${GITGREP[@]}" -E 'sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}' --include='*.rs' --include='*.md' "${SCAN_DIRS[@]}" 2>/dev/null | filter_a5 || true)"
report A5 BLOCK '硬编码 key 字面量(新增 key 需移除)' "$A5_HITS"
echo

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
echo "== 汇总 =="
if [ "$fail_count" -eq 0 ]; then
  echo "  PASS:无未处理 BLOCK 命中"
  exit 0
else
  echo "  FAIL:${fail_count} 个未处理 BLOCK 命中(见上)"
  exit 1
fi
