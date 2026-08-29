#!/usr/bin/env bash
# ============================================================================
# verify-doc-refs.sh — 文档「能力 ↔ 实现接线」半自动核对
#
# 背景(0.18.2 ⏳ 待办):README/USAGE 声称的能力没有自动校验,重构后文档里的
# `langchainrust::X` / `lc_xxx::Y` 引用可能已经失效。本脚本把文档中的 crate
# 路径引用抽取出来,检查目标符号在对应 crate 里仍以 pub 形式存在。
#
# 用法:
#   scripts/verify-doc-refs.sh [FILE...]     # 默认扫 README.md docs/USAGE*.md
#   scripts/verify-doc-refs.sh --quiet        # 只输出 REVIEW 摘要
#
# 语义:
#   PASS   符号在目标 crate 找到 pub 声明/再导出
#   REVIEW 未找到 pub 声明(需人工确认:可能已删、改了名、或只是文本提及)
#
# 已知局限(半自动化,按计划设计):
#   - 只查「crate 路径的第一段符号」,不解析嵌套模块/泛型/method
#   - pub 存在 = 目标 crate src 里有 `pub use/struct/trait/enum/fn/type/
#     mod/const/static` 与该符号同行的行;模块内部细节靠人工
#   - 宏(tool 等)与 derive 不在此检查范围(cargo check 覆盖)
#   - 本检查不含 --all-features,不受 ort-sys 盲区影响
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 2

QUIET=false
FILES=()
for a in "$@"; do
  case "$a" in
    --quiet) QUIET=true ;;
    -*) echo "未知参数: $a" >&2; exit 2 ;;
    *) FILES+=("$a") ;;
  esac
done
if [ "${#FILES[@]}" -eq 0 ]; then
  FILES=(README.md docs/USAGE.md docs/USAGE_EN.md)
fi

# crate 名 → 源码目录
crate_dir() { # <crate>
  case "$1" in
    langchainrust)   echo "crates/lc" ;;
    lc_tools_derive) echo "crates/lc-tools-derive" ;;
    *) echo "crates/${1//_/-}" ;;
  esac
}

# 检查符号在 crate 里是否 pub 存在(三级启发式,见脚本头「已知局限」)
pub_exists() { # <crate-dir> <symbol>
  local dir="$1" sym="$2"
  [ -d "$dir/src" ] || return 1
  # 1) 显式 pub 声明 / 单行 pub use 再导出(任意 src 文件)
  grep -rnE '^[[:space:]]*pub (use|struct|trait|enum|fn|type|mod|const|static|extern)' "$dir/src" \
    | grep -E '\b'"$sym"'\b' >/dev/null 2>&1 && return 0
  # 2) lib.rs 再导出区(含多行 `pub use x::{ ... }` 块)——facade/各 crate 的
  #    公开面都在 lib.rs 收口,符号出现在 lib.rs 即几乎必然可经根级访问
  grep -E '\b'"$sym"'\b' "$dir/src/lib.rs" >/dev/null 2>&1 && return 0
  # 3) 各子模块 mod.rs 的再导出块
  grep -rE '\b'"$sym"'\b' "$dir/src"/*/mod.rs >/dev/null 2>&1 && return 0
  return 1
}

extract_refs() { # <file> → 每行 "crate symbol" (brace 组展开 + 只取首段符号)
  local file="$1" crate sym brace
  # 1) brace 组:crate::{A, B::C, D} → A, B, D(取首段;B::C 的 C 不单独校验)
  while read -r crate brace; do
    [ -z "$crate" ] && continue
    echo "$brace" | tr ', ' '\n' | sed '/^$/d; s/^{//; s/}$//' | while read -r sym; do
      [ -n "$sym" ] && printf '%s %s\n' "$crate" "${sym%%::*}"
    done
  done < <(grep -oE '\b(langchainrust|lc_[a-z_]+)::\{[^}]+\}' "$file" | \
           sed -E 's/::\{/ {/' )
  # 2) 单个:crate::Symbol::Method → crate Symbol(只取首段)
  grep -oE '\b(langchainrust|lc_[a-z_]+)::[A-Za-z_][A-Za-z0-9_]*' "$file" \
    | sed -E 's/::/ /' | sort -u
}

pass=0; review=0; seen=""
check_one() { # <file> <crate> <symbol>
  local file="$1" crate="$2" sym="$3" key="$crate::$sym"
  case "|$seen|" in *"|$key|"*) return ;; esac
  seen="$seen|$key|"
  local dir; dir="$(crate_dir "$crate")"
  if [ -d "$dir" ] && pub_exists "$dir" "$sym"; then
    pass=$((pass + 1))
    $QUIET || printf '  [PASS]   %s::%s  (%s)\n' "$crate" "$sym" "$file"
  else
    review=$((review + 1))
    printf '  [REVIEW] %s::%s  (%s)  <%s 未找到 pub 声明>\n' "$crate" "$sym" "$file" "$dir"
  fi
}

echo "== verify-doc-refs:文档 crate 路径引用核对 =="
echo "   文件:${FILES[*]}"
echo
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "  (跳过不存在: $f)" >&2; continue; }
  while read -r crate sym; do
    [ -z "$crate" ] && continue
    check_one "$f" "$crate" "$sym"
  done < <(extract_refs "$f")
done

echo
echo "== 汇总 =="
echo "  PASS=$pass  REVIEW=$review"
if [ "$review" -gt 0 ]; then
  echo "  ${review} 个引用需人工确认(见上;含文档历史/文本提及等非失效场景)"
else
  echo "  全部文档引用均可解析到 pub 声明"
fi
