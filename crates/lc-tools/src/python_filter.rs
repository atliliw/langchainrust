// lc-tools/src/python_filter.rs
//! Shared Python "dangerous code" noise-filter.
//!
//! **Single source of truth (T2, lc-tools §5):** both [`crate::sandbox::LocalSandbox`] and
//! [`crate::python_repl::PythonREPLTool`] used to carry their own Python blacklists —
//! [`crate::sandbox::local::BLOCKED_PYTHON_IMPORTS`] and `BLOCKED_IMPORTS` / `DANGEROUS_BUILTIN_CALLS`
//! in `python_repl.rs` — which had drifted (the REPL added `antigravity`, the sandbox added
//! `__builtins__`). They now resolve to the consts and functions here, so a module/call either
//! is blocked in both or in neither.
//!
//! **Security positioning**: every check in this module is a *noise filter*, **not a security
//! boundary** — line-by-line substring/regex matching is bypassed by unicode obfuscation,
//! `"o"+"s"` concatenation, `().__class__` reflection, etc., and has false positives on string
//! literals. Untrusted code must run in a real sandbox (container / VM / WASM), not through
//! these filters. The filters only reduce noise reaching an execution backend.

use std::sync::LazyLock;

use regex::Regex;

/// Dangerous Python modules blocked by name (noise-filter substring blacklist).
///
/// Union of the two previously-duplicated lists: sandbox's `BLOCKED_PYTHON_IMPORTS` (20) plus
/// the REPL's `antigravity`.
pub const BLOCKED_PYTHON_IMPORTS: &[&str] = &[
    "os",
    "subprocess",
    "sys",
    "shutil",
    "signal",
    "ctypes",
    "multiprocessing",
    "socket",
    "http.server",
    "xmlrpc",
    "pickle",
    "shelve",
    "importlib",
    "code",
    "codeop",
    "compileall",
    "pty",
    "commands",
    "pdb",
    "webbrowser",
    "antigravity",
];

/// Dangerous builtin calls that bypass the import blacklist (word-boundary + function-call form).
///
/// `open` / `breakpoint` / `input` are file/interactive escape vectors requiring no import;
/// `__import__` / `import_module` are dynamic-import routes; `eval` / `exec` / `execfile` /
/// `compile` run raw code. `__builtins__` is handled separately (bare object access, not a call).
pub const DANGEROUS_BUILTIN_CALLS: &[&str] = &[
    "__import__",
    "import_module",
    "eval",
    "exec",
    "execfile",
    "compile",
    "open",
    "breakpoint",
    "input",
];

/// Matches dangerous calls (`__import__(` / `import_module(` / `eval(` / ...) plus the bare
/// `__builtins__` object access that needs no parens (`__builtins__.__dict__['open']`).
static BLOCKED_PYTHON_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    // INVARIANT: every entry in DANGEROUS_BUILTIN_CALLS is a word-only identifier (letters /
    // underscores), so join("|") introduces no regex metacharacters and the template compiles.
    let calls = DANGEROUS_BUILTIN_CALLS.join("|");
    Regex::new(&format!(r"\b(?:{calls})\s*\(|\b__builtins__\b"))
        .expect("static Python call regex literal must compile")
});

/// Check if Python code contains a dangerous top-level `import`/`from` of a blocked module.
///
/// Whitespace runs are collapsed to a single space first so `import   os` / `import\tos` (which
/// lack the exact `import os` substring) don't slip past. Substring matching keeps module names
/// with `.` (e.g. `http.server`) literal, unlike a regex `join("|")` where `.` would match any char.
fn contains_dangerous_python_import(code: &str) -> Option<String> {
    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        let code_part = trimmed.split('#').next().unwrap_or(trimmed);
        if code_part.contains("import") {
            let collapsed = code_part.split_whitespace().collect::<Vec<_>>().join(" ");
            for blocked in BLOCKED_PYTHON_IMPORTS {
                if collapsed.contains(&format!("import {}", blocked))
                    || collapsed.contains(&format!("from {} ", blocked))
                    || collapsed.contains(&format!("from {}.", blocked))
                    || collapsed.contains(&format!("from {}import", blocked))
                {
                    return Some(blocked.to_string());
                }
            }
        }
    }
    None
}

/// Check if Python code reaches a dangerous builtin call (dynamic import / exec / open / ...).
fn contains_dangerous_python_call(code: &str) -> Option<String> {
    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        let code_part = trimmed.split('#').next().unwrap_or(trimmed);
        if let Some(m) = BLOCKED_PYTHON_CALL_REGEX.find(code_part) {
            return Some(m.as_str().to_string());
        }
    }
    None
}

/// Combined import + builtin-call check, matching the REPL's historical `contains_dangerous_code`
/// semantics (imports checked first, then dangerous calls).
pub fn contains_dangerous_python(code: &str) -> Option<String> {
    contains_dangerous_python_import(code).or_else(|| contains_dangerous_python_call(code))
}

/// The dangerous-import half, exposed so a caller can name the *category* of the block.
pub fn dangerous_python_import(code: &str) -> Option<String> {
    contains_dangerous_python_import(code)
}

/// The dangerous-call half, exposed so a caller can name the *category* of the block.
pub fn dangerous_python_call(code: &str) -> Option<String> {
    contains_dangerous_python_call(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_imports_unified_with_repl_antigravity() {
        assert!(contains_dangerous_python_import("import os").is_some());
        assert!(contains_dangerous_python_import("import subprocess").is_some());
        assert!(contains_dangerous_python_import("from sys import path").is_some());
        assert!(contains_dangerous_python_import("from os.path import join").is_some());
        assert!(contains_dangerous_python_import("# import os").is_none());
        // repl-only module is now in the shared list too.
        assert!(contains_dangerous_python_import("import antigravity").is_some());
    }

    #[test]
    fn detects_calls_and_bare_builtins() {
        assert!(contains_dangerous_python_call("__import__('os').system('id')").is_some());
        assert!(contains_dangerous_python_call("importlib.import_module('os')").is_some());
        assert!(contains_dangerous_python_call("exec('print(1)')").is_some());
        assert!(contains_dangerous_python_call("eval('1+1')").is_some());
        assert!(contains_dangerous_python_call("open('/etc/shadow').read()").is_some());
        // false-positive guard
        assert!(contains_dangerous_python_call("evaluate(fn).run()").is_none());
        assert!(contains_dangerous_python_call("x = 1 + evalue").is_none());
    }

    #[test]
    fn combined_check_matches_previous_repl_behavior() {
        assert!(contains_dangerous_python("import os").is_some());
        assert!(contains_dangerous_python("__builtins__.__dict__['open']").is_some());
        assert!(contains_dangerous_python("import math").is_none());
        assert!(contains_dangerous_python("print(len([1, 2, 3]))").is_none());
    }
}
