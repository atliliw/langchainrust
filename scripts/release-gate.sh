#!/usr/bin/env bash
# ============================================================================
# release-gate.sh — release gate (turns RELEASE_CHECKLIST manual steps into an
# executable script)
#
# Background (0.18.2 backlog): the release gate used to be a manual checklist,
# easy to miss an item. This script turns the "build/test gate" section of
# RELEASE_CHECKLIST into a sequential pipeline; a failure stops it.
# Shared by local (Git Bash) and CI (cd.yml pre-publish).
#
# Steps (the disk preflight is a precondition and is not counted; run() has 7):
#   0. Disk preflight (df before a full build; threshold MIN_FREE_GB, default 10G)
#   1. fmt --check                                          [soft gate]
#   2. clippy --workspace --all-targets [--all-features]    [soft gate]
#   3. test --workspace --no-fail-fast
#   4. rustdoc -D warnings (--no-deps [--all-features])
#   5. semver-checks --workspace --baseline-version <BASELINE>
#      (features: local --default-features / CI --all-features)
#   6. security-scan.sh --fail-if-found
#   7. Version consistency: all 23 crates/*/Cargo.toml [package].version match
#
# Layered gate (2026-11-14):
#   - Hard gates: test / rustdoc / semver / security / version consistency —
#     fail fast.
#   - Soft gates: fmt / clippy — print WARN only, do not block the release.
#     They mostly reflect formatting and new lints, not crate functionality;
#     they repeatedly blocked whole releases when existing code went red under
#     a new stable toolchain.
#   - Set STRICT_LINT=1 to restore strict mode.
#
# Usage:
#   scripts/release-gate.sh                # local gate (no --all-features)
#   scripts/release-gate.sh --all-features # CI gate (used by cd.yml pre-publish)
#
# Env overrides:
#   BASELINE=0.19.0    semver baseline (default 0.19.0)
#   SKIP_SEMVER=1      skip semver-checks (when cargo-semver-checks is not installed locally)
#   STRICT_LINT=1      make fmt/clippy blocking again (default 0, soft gates)
#   MIN_FREE_GB=10     disk warning threshold in GB
#   JOBS=4             cargo -j parallelism (default 4; on a 16-core/16G machine
#                      `cargo test --workspace` with 16-way parallelism OOMs —
#                      handle_alloc_error / E0463 cascade, observed 2026-08-28)
#
# Design constraints (known blind spots, see RELEASE_CHECKLIST S2.3):
#   - Local runs omit --all-features: under windows-gnu + rust-toolchain 1.85,
#     lc-embeddings/local-embeddings → ort-sys 2.0.0-rc.13 requires rustc ≥ 1.88
#     and cannot build. clippy/doc with --all-features are CI's responsibility;
#     local does not force them.
#   - set -euo pipefail; a hard-gate failure exits; a soft-gate failure continues.
# ============================================================================
set -euo pipefail

ALL_FEATURES=false
[ "${1:-}" = "--all-features" ] && ALL_FEATURES=true

BASELINE="${BASELINE:-0.19.0}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
JOBS="${JOBS:-4}"
STRICT_LINT="${STRICT_LINT:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FEATURES=()
if [ "$ALL_FEATURES" = true ]; then FEATURES=(--all-features); fi

step=0
total=7
run() { # <name> <cmd...> — run sequentially; a failure stops the gate
  local name="$1"; shift
  step=$((step + 1))
  printf '  [..] %2d/%d  %s\n' "$step" "$total" "$name"
  if "$@"; then
    printf '  [PASS] %2d/%d  %s\n' "$step" "$total" "$name"
  else
    printf '  [FAIL] %2d/%d  %s\n' "$step" "$total" "$name"
    echo "release-gate: gate failed at \"$name\"; aborting (fix and re-run)." >&2
    exit 1
  fi
}
run_soft() { # <name> <cmd...> — soft gate: WARN and continue; STRICT_LINT=1 makes it blocking
  local name="$1"; shift
  step=$((step + 1))
  printf '  [..] %2d/%d  %s\n' "$step" "$total" "$name"
  if "$@"; then
    printf '  [PASS] %2d/%d  %s\n' "$step" "$total" "$name"
  elif [ "$STRICT_LINT" = "1" ]; then
    printf '  [FAIL] %2d/%d  %s (STRICT_LINT=1)\n' "$step" "$total" "$name"
    echo "release-gate: strict mode: soft gate failed at \"$name\"; aborting." >&2
    exit 1
  else
    printf '  [WARN] %2d/%d  %s (non-blocking)\n' "$step" "$total" "$name"
    soft_fail=$((soft_fail + 1))
    soft_names+=("$name")
  fi
}
warn() { printf '  [WARN]       %s\n' "$1"; }

soft_fail=0
soft_names=()

echo "== release-gate =="
echo "   mode: $([ "$ALL_FEATURES" = true ] && echo 'CI(--all-features)' || echo 'local(no all-features; ort-sys blind spot, see RELEASE_CHECKLIST S2.3)')"
echo "   semver baseline: $BASELINE   features: ${FEATURES[*]:-(none)}"
echo

# ---- 1. Disk space preflight ----------------------------------------------
# memory: a full build cache can reach 20G and C: is tight; the build cache is
# outside the sync folder (target-dir = D:/rust-target). Check the current work
# drive (the repo drive, same drive as target).
FREE_GB=$(( $(df -Pk . | tail -1 | awk '{print $4}') / 1024 / 1024 ))
printf '  disk space: %dG free (threshold %dG)\n' "$FREE_GB" "$MIN_FREE_GB"
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
  warn "free space is below the threshold; a full build may fail — clean rust-target or lower MIN_FREE_GB"
fi
echo

# ---- 2. fmt (soft gate) ---------------------------------------------------
run_soft "cargo fmt --check" cargo fmt --all -- --check

# ---- 3. clippy (soft gate) ------------------------------------------------
run_soft "clippy --workspace --all-targets" cargo clippy --workspace --all-targets -j "$JOBS" "${FEATURES[@]}" -- -D warnings

# ---- 4. test --------------------------------------------------------------
run "cargo test --workspace --no-fail-fast" cargo test --workspace --no-fail-fast -j "$JOBS"

# ---- 5. rustdoc -----------------------------------------------------------
# sqlx-postgres 0.8.0's rustdoc fails on recent rustc (E0277 `!: Decode`), and
# sqlx cannot be pinned higher without a libsqlite3-sys links conflict with
# rusqlite (see crates/lc/Cargo.toml). Only pgvector-storage /
# checkpoint-postgres pull it in, forwarded through the facade's --all-features.
# So in CI mode the facade and the two sqlx-pg owners are excluded from the
# all-features workspace doc and documented separately with default features —
# matching the docs.rs feature surface, which also excludes pgvector-storage.
rustdoc_gate() {
  if [ "$ALL_FEATURES" = true ]; then
    env RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps -j "$JOBS" --all-features \
      --exclude langchainrust --exclude lc-vector-stores --exclude lc-langgraph
    env RUSTDOCFLAGS="-D warnings" cargo doc -p langchainrust -p lc-vector-stores -p lc-langgraph \
      --no-deps -j "$JOBS"
  else
    env RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps -j "$JOBS"
  fi
}
run "rustdoc -D warnings --no-deps" rustdoc_gate

# ---- 6. semver-checks -----------------------------------------------------
# Local mode uses --default-features: cargo-semver-checks re-resolves deps in a
# temporary project outside the repo (ignoring Cargo.lock), so the
# feature-gated sqlx may jump to 0.9.0 (rustc ≥ 1.94) and fail to build on the
# local 1.93 toolchain. --default-features avoids pgvector-storage→sqlx and
# local-embeddings→ort-sys, sidestepping the local toolchain blind spot.
# CI (cd.yml pre-publish, ubuntu + new stable) runs --all-features to cover
# feature-gated APIs.
SEMVER_FEATURES=(--default-features)
if [ "$ALL_FEATURES" = true ]; then SEMVER_FEATURES=(--all-features); fi

if [ "${SKIP_SEMVER:-0}" = "1" ]; then
  warn "SKIP_SEMVER=1: skipping semver-checks"
elif command -v cargo-semver-checks >/dev/null 2>&1; then
  run "semver-checks --baseline ${BASELINE}" cargo semver-checks --workspace --baseline-version "$BASELINE" "${SEMVER_FEATURES[@]}"
else
  warn "cargo-semver-checks is not installed; skipping locally (CI pre-publish runs it). Install it or confirm with SKIP_SEMVER=1"
fi

# ---- 7. security-scan -----------------------------------------------------
run "security-scan --fail-if-found" bash scripts/security-scan.sh --fail-if-found

# ---- 8. Version consistency -----------------------------------------------
check_version_consistency() {
  local expected="" bad=0 files=0 v f
  while IFS= read -r f; do
    v="$(awk -F'"' '/^\[package\]/{p=1} p && /^version *=/{print $2; exit}' "$f")"
    files=$((files + 1))
    if [ -z "$expected" ]; then expected="$v"; fi
    if [ -z "$v" ] || [ "$v" != "$expected" ]; then
      printf '        ✗ %s: version=%s (expected %s)\n' "$f" "$v" "$expected"
      bad=1
    fi
  done < <(printf '%s\n' crates/*/Cargo.toml)
  printf '        checked %d Cargo.toml files, expected version %s\n' "$files" "$expected"
  [ "$bad" -eq 0 ]
}
run "version consistency (whole workspace)" check_version_consistency

echo
if [ "$soft_fail" -gt 0 ]; then
  echo "== release-gate: all hard gates passed; ${soft_fail} soft gate(s) failed (non-blocking) =="
  printf '   soft failures: %s\n' "${soft_names[*]}"
  echo "   set STRICT_LINT=1 to make fmt/clippy blocking again."
else
  echo "== release-gate: all passed =="
fi
