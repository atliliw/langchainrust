#!/usr/bin/env bash
# ============================================================================
# release-gate.sh — 发布门禁(把 RELEASE_CHECKLIST 手工步骤固化为可执行脚本)
#
# 背景(0.18.2 ⏳ 待办):发布门禁是手工清单,靠人逐项勾,易遗漏。本脚本把
# RELEASE_CHECKLIST 的「构建/测试门禁」一节固化为串联流程,失败即停。
# 本地(Git Bash)与 CI(cd.yml pre-publish)共用同一脚本。
#
# 步骤(磁盘预检是前置检查,不算进 step 序号;run() 共 7 步):
#   0. 磁盘空间预检(全量构建前 df,阈值 MIN_FREE_GB 默认 10G)
#   1. fmt --check
#   2. clippy --workspace --all-targets [--all-features]
#   3. test --workspace --no-fail-fast
#   4. rustdoc -D warnings(--no-deps [--all-features])
#   5. semver-checks --workspace --baseline-version <BASELINE>(feature 选择:本地 --default-features / CI --all-features)
#   6. security-scan.sh --fail-if-found
#   7. 版本一致性:全 workspace 22 个 crates/*/Cargo.toml 的 [package].version 一致
#
# 用法:
#   scripts/release-gate.sh                # 本地门禁(不带 --all-features)
#   scripts/release-gate.sh --all-features # CI 门禁(cd.yml pre-publish 使用)
#
# env 覆盖:
#   BASELINE=0.19.0    semver 基线版本(默认 0.19.0)
#   SKIP_SEMVER=1      跳过 semver-checks(本地未装 cargo-semver-checks 时)
#   MIN_FREE_GB=10     磁盘空间告警阈值(单位 GB)
#   JOBS=4             cargo -j 并行度(默认 4;本机 16 核 + 16G 内存下
#                      `cargo test --workspace` 默认 16 路并行编译会 OOM——
#                      handle_alloc_error / E0463 级联,实测 2026-08-28)
#
# 设计约束(盲区说明,见 RELEASE_CHECKLIST S2.3):
#   - 本地默认不带 --all-features:windows-gnu + rust-toolchain 1.85 下,
#     lc-embeddings/local-embeddings → ort-sys 2.0.0-rc.13 需 rustc≥1.88
#     无法编译。clippy/doc 的 --all-features 组合是 CI 责任,本地不强跑。
#   - set -euo pipefail;每步失败即退出,便于快速定位。
# ============================================================================
set -euo pipefail

ALL_FEATURES=false
[ "${1:-}" = "--all-features" ] && ALL_FEATURES=true

BASELINE="${BASELINE:-0.19.0}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
JOBS="${JOBS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FEATURES=()
if [ "$ALL_FEATURES" = true ]; then FEATURES=(--all-features); fi

step=0
total=7
run() { # <name> <cmd...> — 顺序执行,失败即停
  local name="$1"; shift
  step=$((step + 1))
  printf '  [..] %2d/%d  %s\n' "$step" "$total" "$name"
  if "$@"; then
    printf '  [PASS] %2d/%d  %s\n' "$step" "$total" "$name"
  else
    printf '  [FAIL] %2d/%d  %s\n' "$step" "$total" "$name"
    echo "release-gate: 门禁失败于「$name」,终止(修复后重跑)。" >&2
    exit 1
  fi
}
warn() { printf '  [WARN]       %s\n' "$1"; }

echo "== release-gate =="
echo "   模式:$([ "$ALL_FEATURES" = true ] && echo 'CI(--all-features)' || echo '本地(无 all-features,ort-sys 盲区见 RELEASE_CHECKLIST S2.3)')"
echo "   semver 基线:$BASELINE   特性:${FEATURES[*]:-(无)}"
echo

# ---- 1. 磁盘空间预检 ------------------------------------------------------
# memory:全量构建缓存可达 20G,C 盘紧张;构建缓存在同步目录外(target-dir
# = D:/rust-target)。这里检查当前工作盘(仓库所在盘,与 target 同盘)。
FREE_GB=$(( $(df -Pk . | tail -1 | awk '{print $4}') / 1024 / 1024 ))
printf '  磁盘空间:可用 %dG(阈值 %dG)\n' "$FREE_GB" "$MIN_FREE_GB"
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
  warn "空间低于阈值,全量构建可能失败;建议先清理 rust-target 或降低 MIN_FREE_GB"
fi
echo

# ---- 2. fmt ---------------------------------------------------------------
run "cargo fmt --check" cargo fmt --all -- --check

# ---- 3. clippy ------------------------------------------------------------
run "clippy --workspace --all-targets" cargo clippy --workspace --all-targets -j "$JOBS" "${FEATURES[@]}" -- -D warnings

# ---- 4. test --------------------------------------------------------------
run "cargo test --workspace --no-fail-fast" cargo test --workspace --no-fail-fast -j "$JOBS"

# ---- 5. rustdoc -----------------------------------------------------------
run "rustdoc -D warnings --no-deps" env RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps -j "$JOBS" "${FEATURES[@]}"

# ---- 6. semver-checks -----------------------------------------------------
# 本地模式用 --default-features:cargo-semver-checks 会在仓库外的临时工程里
# 重新解析依赖(脱离 Cargo.lock),feature 门控的 sqlx 可能升到 0.9.0(要求
# rustc≥1.94),而本机默认工具链是 1.93 编不了。--default-features 不拉
# pgvector-storage→sqlx 与 local-embeddings→ort-sys,绕开本地工具链盲区;
# CI(cd.yml pre-publish,ubuntu + 新 stable)跑 --all-features 兜底 feature 门控的 API。
SEMVER_FEATURES=(--default-features)
if [ "$ALL_FEATURES" = true ]; then SEMVER_FEATURES=(--all-features); fi

if [ "${SKIP_SEMVER:-0}" = "1" ]; then
  warn "SKIP_SEMVER=1,跳过 semver-checks"
elif command -v cargo-semver-checks >/dev/null 2>&1; then
  run "semver-checks --baseline ${BASELINE}" cargo semver-checks --workspace --baseline-version "$BASELINE" "${SEMVER_FEATURES[@]}"
else
  warn "cargo-semver-checks 未安装,本地跳过(CI pre-publish 会跑);安装后或 SKIP_SEMVER=1 显式确认"
fi

# ---- 7. security-scan -----------------------------------------------------
run "security-scan --fail-if-found" bash scripts/security-scan.sh --fail-if-found

# ---- 8. 版本一致性 --------------------------------------------------------
check_version_consistency() {
  local expected="" bad=0 files=0 v f
  while IFS= read -r f; do
    v="$(awk -F'"' '/^\[package\]/{p=1} p && /^version *=/{print $2; exit}' "$f")"
    files=$((files + 1))
    if [ -z "$expected" ]; then expected="$v"; fi
    if [ -z "$v" ] || [ "$v" != "$expected" ]; then
      printf '        ✗ %s: version=%s(期望 %s)\n' "$f" "$v" "$expected"
      bad=1
    fi
  done < <(printf '%s\n' crates/*/Cargo.toml)
  printf '        检查 %d 个 Cargo.toml,期望版本 %s\n' "$files" "$expected"
  [ "$bad" -eq 0 ]
}
run "版本一致性(全 workspace)" check_version_consistency

echo
echo "== release-gate:全部通过 =="
