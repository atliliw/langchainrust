#!/usr/bin/env bash
# Publish every workspace crate to crates.io — the only entry point for
# tag-driven releases (.github/workflows/cd.yml).
#
# Why this script exists:
#   The root Cargo.toml is a virtual workspace, so running `cargo publish` at
#   the root fails with "current directory is part of a virtual workspace".
#   The 23 crates depend on each other via path+version, and crates.io requires
#   a dependency version to exist in the index first, so publishing must be
#   interleaved in topological order: publish one -> wait for the index ->
#   publish the next (the lesson from the manual 0.22.0–0.22.4 releases).
#
# Guarantees:
#   1. The order comes from the workspace-internal dependency topology in
#      `cargo metadata` (Kahn); the facade crate (langchainrust) is always last.
#      No hand-maintained member list.
#   2. Idempotent / resumable: a crate whose target version already exists in
#      the crates.io sparse index is skipped — crates.io never allows
#      overwriting a published version, so re-running after a failure resumes.
#   3. Index-lag fallback: when a dependency was just uploaded and the peer
#      index has not caught up yet, publish verification reports
#      "no matching package"; retry with exponential backoff instead of leaving
#      the batch half-published.
#
# Environment variables:
#   CARGO_REGISTRY_TOKEN  crates.io token (injected from the secret in cd.yml)
#   PUBLISH_DRY_RUN=1     run cargo publish --dry-run only, upload nothing
#   PUBLISH_ARGS          extra args passed through to cargo publish
#                         (for local mirrors, pass --registry crates-io; the
#                         GitHub runner defaults to the official registry)
#   PUBLISH_RETRIES       per-crate publish attempts (default 6)
#   PUBLISH_WAIT_BASE     base retry wait in seconds (default 20)
#
# Reference: crates.io sparse index path rules
#   1-char name  /1/<name>; 2-char /2/<name>; 3-char /3/<first>/<name>;
#   longer       /<first 2>/<chars 3-4>/<name> ('-'/'_' kept as-is).
set -euo pipefail

cd "$(dirname "$0")/.."

RETRIES="${PUBLISH_RETRIES:-6}"
WAIT_BASE="${PUBLISH_WAIT_BASE:-20}"
DRY_RUN="${PUBLISH_DRY_RUN:-0}"
PUBLISH_ARGS="${PUBLISH_ARGS:-}"
INDEX_BASE="https://index.crates.io"

# Pick a python3 that actually runs: under Git Bash, WindowsApps/python3 may be
# a Store stub (command -v finds it but execution fails); probe with a real -c
# and fall back to python.
PYTHONBIN=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys' >/dev/null 2>&1; then
    PYTHONBIN="$cand"
    break
  fi
done

# T2 (v0.23) hard release discipline: real publishing may only happen in the
# tag-triggered GitHub Actions workflow (cd.yml: pre-publish release-gate green
# -> publish calls this script). 0.22.4 bypassed release-gate and published
# manually; a fastembed feature compile error shipped and docs.rs has been red
# since 0.20. Locally, only dry-run is allowed; if a real publish outside CI is
# truly required (emergency), set ALLOW_LOCAL_PUBLISH=1 explicitly — a
# deliberate double safety, not part of the normal docs.
if [[ "$DRY_RUN" != "1" && "${GITHUB_ACTIONS:-}" != "true" && "${ALLOW_LOCAL_PUBLISH:-}" != "1" ]]; then
  echo "ERROR: real publishing outside CI is forbidden (closed after the 0.22.4 incident, T2 v0.23)." >&2
  echo "       Correct flow: push a tag -> cd.yml release-gate --all-features green -> CI publishes in topological order." >&2
  echo "       For local validation use: PUBLISH_DRY_RUN=1 bash scripts/publish-crates.sh" >&2
  exit 2
fi

# Output: one line per crate "name<TAB>version<TAB>sparse-index path", ordered
# by dependency topology (dependencies first).
topo_plan() {
  cargo metadata --format-version 1 --no-deps | "$PYTHONBIN" -c '
import json, os, sys
from collections import deque

meta = json.load(sys.stdin)

# cargo pkgid formats vary across versions (new: "name version (path+..)",
# old: "path+..#ver"; workspace_members and package.id may also disagree).
# Decide membership by "manifest lives under the workspace root" instead of
# matching id strings.
ws_root = os.path.normcase(os.path.abspath(meta["workspace_root"]))

def under_root(path):
    p = os.path.normcase(os.path.abspath(path))
    return p == ws_root or p.startswith(ws_root + os.sep)

pkgs = {p["id"]: p for p in meta["packages"]
        if under_root(os.path.dirname(p["manifest_path"]))}

# Package directory -> id; older metadata only gives a directory (path field)
# for internal dependencies, not a pkgid.
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

# Adjacency: edge dep -> dependent (after Kahn ordering, dependencies are
# dequeued first). Count normal/build/dev dependencies: during interleaved
# publishing a dev-dependency (e.g. lc-testkit) must also already be on
# crates.io; conservative ordering only publishes dependencies earlier.
deps_of = {pid: set() for pid in pkgs}
indegree = {pid: 0 for pid in pkgs}
for pid, p in pkgs.items():
    for d in p["dependencies"]:
        dep_pid = None
        if d.get("pkg"):                      # newer cargo: dependency pkgid directly
            dep_pid = d["pkg"] if d["pkg"] in pkgs else None
        elif d.get("path"):                   # older cargo: match by directory
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

# Whether the version is already on crates.io (one JSON release record per line
# in the sparse index, ascending by version).
# Do not use `curl | grep -q`: grep exits on the first match, curl gets SIGPIPE,
# and under pipefail the pipeline randomly returns non-zero (timing-dependent;
# observed 23 crates resolving only 1). Read the body first, then match.
# Returning "not published" on a network failure is also safe: cargo verifies
# again during the publish phase.
version_published() {
  local idx_path="$1" version="$2" body
  body="$(curl -fsS --http1.1 --max-time 45 --retry 4 --retry-all-errors --retry-delay 2 \
    "${INDEX_BASE}/${idx_path}" 2>/dev/null || true)"
  # Must use a herestring rather than printf|grep -q: under pipefail, grep exits
  # on match and the writer gets SIGPIPE, making the function randomly return
  # non-zero (observed 23 misses; the facade only matched by luck because its
  # JSON is shorter).
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
    # cargo's authoritative verdict: the version already exists -> treat as
    # published (sparse-index prediction missed / resuming a batch).
    # Never retry: already-exists never self-heals; retrying would wedge the
    # whole batch on the first crate.
    if grep -q "already exists" "$logf"; then
      echo "--- [${name} ${version}] cargo reports this version already exists; treating it as published and skipping"
      rm -f "$logf"
      return 42  # 42 = "already exists, skipped" agreed with the caller; distinct from a real publish (0)
    fi
    if [ "$attempt" -ge "$RETRIES" ]; then
      echo "!!! [${name} ${version}] failed after ${RETRIES} attempts" >&2
      rm -f "$logf"
      return 1
    fi
    wait_s=$(( WAIT_BASE * attempt ))
    echo "... [${name}] publish failed (possibly the dependency was just uploaded and the index is not visible yet); retrying in ${wait_s}s"
    sleep "$wait_s"
    attempt=$(( attempt + 1 ))
  done
}

main() {
  [ -n "$PYTHONBIN" ] || { echo "fatal: python3 (or python) is required to compute the publish topology" >&2; exit 1; }

  local plan
  plan="$(topo_plan)"
  local total
  total=$(printf '%s\n' "$plan" | grep -c .)
  echo "=== Publish plan: ${total} workspace crates (topological order) ==="
  printf '%s\n' "$plan" | cut -f1 | nl -ba
  echo

  local n=0 skipped=0
  while IFS=$'\t' read -r name version idx_path; do
    # Windows Python text-mode stdout converts \n to CRLF, so the last field
    # (idx_path) carries \r; strip it uniformly (a no-op on Linux/LF).
    name="${name%$'\r'}"; version="${version%$'\r'}"; idx_path="${idx_path%$'\r'}"
    n=$(( n + 1 ))
    if version_published "$idx_path" "$version"; then
      skipped=$(( skipped + 1 ))
      echo "--- [${n}/${total}] ${name} ${version} already exists on crates.io; skipping"
      continue
    fi
    echo "--- [${n}/${total}] publishing ${name} ${version}"
    if publish_one "$name" "$version"; then
      :
    elif [ "$?" -eq 42 ]; then
      skipped=$(( skipped + 1 ))
    else
      echo "fatal: ${name} ${version} failed; aborting. Fix and re-run this script; already-uploaded crates will be skipped." >&2
      exit 1
    fi
  done <<< "$plan"

  local published=$(( total - skipped ))
  if [ "$DRY_RUN" != "1" ] && [ "$published" -eq 0 ]; then
    echo "fatal: all ${total} crates already exist at the current version on crates.io — did you forget to bump versions before publishing?" >&2
    exit 1
  fi
  echo "=== Done: ${total} crates, skipped ${skipped} already-published, newly published ${published} ==="
}

main "$@"
