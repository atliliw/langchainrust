// lc-tools/src/sandbox/local.rs
//! Local process sandbox backend.

use std::sync::LazyLock;
use std::time::Instant;

use async_trait::async_trait;
use regex::Regex;
use tokio::process::Command;

use super::{CodeSandbox, Language, RunResult, SandboxError};

/// The Python dangerous-code noise filter is the shared source of truth in
/// [`crate::python_filter`] (T2 §5): `LocalSandbox` and `PythonREPLTool` previously carried
/// duplicated import/call blacklists that had drifted (the REPL added `antigravity`, this
/// module added `__builtins__`). Gone; both now read the same consts/functions, so a module or
/// call is blocked in both or neither.
///
/// The JS branch below carries an equivalent JS-only noise filter ([`BLOCKED_JS_MODULES`] +
/// [`BLOCKED_JS_CALL_REGEX`], H1) so it is not left completely open where Python has a guard.
/// Backward-compatible names for this module's own call sites and tests.
use crate::python_filter::dangerous_python_call as contains_dangerous_python_call;
use crate::python_filter::dangerous_python_import as contains_dangerous_python_import;

/// Dangerous Node.js modules for the JS branch, mirroring [`BLOCKED_PYTHON_IMPORTS`].
///
/// H1 (0.25.0): the JS branch previously executed `node -e <code>` with *no* guard while Python
/// had at least the noise-filter blacklist — strictly worse, and it made the "sandbox" name even
/// more misleading for a local raw process. This is the same **noise filter, not a security
/// boundary**: a substring scan is trivially bypassed (`require('child'+'process')`,
/// `globalThis['fs']`, unicode obfuscation...). Untrusted code must not run through
/// [`LocalSandbox`] unless the process itself sits inside a real sandbox (container / VM / WASM).
const BLOCKED_JS_MODULES: &[&str] = &[
    "fs",
    "child_process",
    "net",
    "dgram",
    "tls",
    "http",
    "https",
    "http2",
    "worker_threads",
    "vm",
    "cluster",
    "repl",
    "readline",
    "os",
];

/// `require('<blocked>')` with optional whitespace anywhere the JS grammar allows a token gap
/// (`require ( 'fs' )`, `require\n('fs')`), so space/tab/newline padding can't hide a blocked
/// module (T2: the old literal-only `require('fs')` scan missed `require( 'fs' )`).
///
/// Concatenation obfuscation (`require('f'+'s')`, `globalThis['fs']`) is *not* covered — this is
/// a noise filter, not a security boundary (see [`BLOCKED_JS_MODULES`]).
static BLOCKED_JS_REQUIRE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    let mods = BLOCKED_JS_MODULES.join("|");
    Regex::new(&format!(r#"\brequire\s*\(\s*['"](?:{})['"]\s*\)"#, mods))
        .expect("static JS require regex literal must compile")
});

/// Common dangerous Node.js **calls** not gated by an import: `process.env`/`process.exit`, the
/// `eval`/`Function` constructors, `globalThis` dynamic reach, and `spawn`/`exec` (which a
/// `node -e` one-liner can also reach via `child_process`, already blocked above). These are the
/// JS analogues of [`BLOCKED_PYTHON_IMPORTS`]/builtin calls — word-boundary guarded so ordinary
/// identifiers like `evaluate(`/`spawner(` do not false-positive.
///
/// T2: `process\.` became `process\s*\.` (member access tolerates whitespace in JS) and
/// `globalThis` got a trailing `\b`, closing the space/comment-split bypasses.
static BLOCKED_JS_CALL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:process\s*\.|eval\s*\(|Function\s*\(|globalThis\b|spawn\s*\(|exec\s*\()")
        .expect("static JS regex literal must compile")
});

/// Remove `//` line and `/* */` block comments from JS source (noise-filter helper).
///
/// Both comment forms are stripped first so a block comment can neither hide a real match
/// reasonably reachable without it (T2: previously only `//` was removed and `/* */` was left
/// opaque) nor produce a false positive for commented-out detector hits. A tokenizer-precise
/// scan is unnecessary here — string-literal `"//"`/`"/*"` inside the payload are out of scope
/// for a filter that is explicitly not a security boundary.
fn strip_js_comments(code: &str) -> String {
    let mut out = String::with_capacity(code.len());
    let mut rest = code;
    // Earliest of the two comment openers; `Option::min` treats `None` as "smaller" and would
    // suppress a real match, so fold the pair here instead.
    let earliest = |rest: &str| match (rest.find("//"), rest.find("/*")) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    while let Some(i) = earliest(rest) {
        let (head, tail) = rest.split_at(i);
        out.push_str(head);
        if tail.starts_with("//") {
            // Line comment: drop through end of line, keep the newline so `stmt // note\nnext`
            // still scans `next` on its own line.
            match tail.find('\n') {
                Some(nl) => {
                    out.push('\n');
                    rest = &tail[nl + 1..];
                }
                None => rest = "",
            }
        } else {
            // Block comment: drop through the closing `*/`.
            match tail.find("*/") {
                Some(end) => rest = &tail[end + 2..],
                None => rest = "",
            }
        }
    }
    out.push_str(rest);
    out
}

/// Check if JavaScript code reaches dangerous Node.js APIs (noise-filter, see above).
fn contains_dangerous_javascript(code: &str) -> Option<String> {
    // Strip `//` and `/* */` comments *before* scanning (T2): comment-padding around a blocked
    // token then can't shadow a real match, and commented-out occurrences are no match either.
    let stripped = strip_js_comments(code);
    // Scan the whole (comment-free) text, not line-by-line: `\s*` in the regexes then spans
    // newlines, so `require\n('fs')` and a `eval\n(` split across lines are caught as such
    // instead of being hidden because `require` and its `(...)` land on different lines.
    if let Some(m) = BLOCKED_JS_REQUIRE_REGEX.find(&stripped) {
        return Some(m.as_str().to_string());
    }
    // statement-level dangerous access: process.* / eval( / Function( / globalThis / spawn / exec.
    if let Some(m) = BLOCKED_JS_CALL_REGEX.find(&stripped) {
        return Some(m.as_str().to_string());
    }
    None
}

/// Local process sandbox using `tokio::process::Command`.
///
/// By default dangerous operations (Python imports of os/subprocess/..., the
/// `__import__`/`importlib`/exec family, Node.js `require`/process/eval) are **denied**.
/// An explicit [`LocalSandbox::with_dangerously_allow`] opt-in turns them loose —
/// that should only be done when the process itself already sits inside a real sandbox
/// (container / VM / WASM), since the static scan is a noise filter, not a security boundary.
pub struct LocalSandbox {
    python_path: String,
    node_path: String,
    /// Opt-in that relaxes the dangerous-op noise filters. Off (deny) by default.
    allow_dangerous: bool,
}

impl LocalSandbox {
    /// Create a new local sandbox with auto-detected interpreter paths.
    pub fn new() -> Self {
        Self {
            python_path: Self::find_python(),
            node_path: "node".to_string(),
            allow_dangerous: false,
        }
    }

    /// Use a custom Python interpreter path.
    pub fn with_python_path(mut self, path: impl Into<String>) -> Self {
        self.python_path = path.into();
        self
    }

    /// Use a custom Node.js runtime path.
    pub fn with_node_path(mut self, path: impl Into<String>) -> Self {
        self.node_path = path.into();
        self
    }

    /// Opt in to running dangerous operations (default denies them). Only set this
    /// when the subprocess is itself confined by a real sandbox; the static scan is
    /// a noise filter, not a security boundary.
    pub fn with_dangerously_allow(mut self, allow: bool) -> Self {
        self.allow_dangerous = allow;
        self
    }

    /// Auto-detect the Python interpreter on the system.
    fn find_python() -> String {
        for candidate in &["python3", "python"] {
            if std::process::Command::new(candidate)
                .arg("--version")
                .output()
                .is_ok()
            {
                return candidate.to_string();
            }
        }
        "python3".to_string()
    }

    /// Build the command for the given language and code.
    fn build_command(&self, code: &str, language: Language) -> Result<Command, SandboxError> {
        match language {
            Language::Python => {
                // Dangerous ops are denied by default; only the explicit
                // with_dangerously_allow(true) opt-in relaxes the noise filter.
                if !self.allow_dangerous {
                    if let Some(blocked) = contains_dangerous_python_import(code) {
                        return Err(SandboxError::Runtime(format!(
                            "Code contains dangerous import: '{}'. \
                             Blocked by the noise-filter blacklist (note: this is not a \
                             security boundary; untrusted code must run in a real sandbox).",
                            blocked
                        )));
                    }
                    if let Some(blocked) = contains_dangerous_python_call(code) {
                        return Err(SandboxError::Runtime(format!(
                            "Code contains dangerous Python call: '{}'. \
                             Blocked by the noise-filter blacklist (note: this is not a \
                             security boundary; untrusted code must run in a real sandbox).",
                            blocked
                        )));
                    }
                }
                let mut cmd = Command::new(&self.python_path);
                cmd.arg("-c").arg(code);
                Ok(cmd)
            }
            Language::JavaScript => {
                if !self.allow_dangerous {
                    if let Some(dangerous) = contains_dangerous_javascript(code) {
                        return Err(SandboxError::Runtime(format!(
                            "Code contains dangerous Node.js API: '{}'. \
                             Blocked by the noise-filter blacklist (note: this is not a \
                             security boundary; untrusted code must run in a real sandbox).",
                            dangerous
                        )));
                    }
                }
                let mut cmd = Command::new(&self.node_path);
                cmd.arg("-e").arg(code);
                Ok(cmd)
            }
            Language::Rust => Err(SandboxError::UnsupportedLanguage(
                "Rust compilation is not supported by LocalSandbox".to_string(),
            )),
            // WASI modules are stripped to a single program counter *inside* the wasmi
            // interpreter; a subprocess has nothing to hand to `python`/`node`. This is the
            // `WasmSandbox`'s job (feature `sandbox-wasm`).
            Language::Wasm => Err(SandboxError::UnsupportedLanguage(
                "WASM execution is not supported by LocalSandbox; use WasmSandbox".to_string(),
            )),
        }
    }
}

impl Default for LocalSandbox {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CodeSandbox for LocalSandbox {
    async fn run(
        &self,
        code: &str,
        language: Language,
        timeout_ms: u64,
    ) -> Result<RunResult, SandboxError> {
        let mut cmd = self.build_command(code, language)?;

        let start = Instant::now();

        let result =
            tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), cmd.output())
                .await
                .map_err(|_| SandboxError::Timeout(timeout_ms))?
                .map_err(|e| {
                    SandboxError::Runtime(format!("failed to execute subprocess: {}", e))
                })?;

        let execution_time_ms = start.elapsed().as_millis() as u64;

        let stdout = String::from_utf8_lossy(&result.stdout).to_string();
        let stderr = String::from_utf8_lossy(&result.stderr).to_string();
        let exit_code = result.status.code().unwrap_or(-1);

        Ok(RunResult {
            stdout,
            stderr,
            exit_code,
            execution_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_sandbox_default() {
        let sandbox = LocalSandbox::default();
        assert!(!sandbox.python_path.is_empty());
        assert_eq!(sandbox.node_path, "node");
        // Dangerous ops denied by default (H4 opt-in gate off).
        assert!(!sandbox.allow_dangerous);
    }

    #[test]
    fn test_local_sandbox_custom_paths() {
        let sandbox = LocalSandbox::new()
            .with_python_path("/usr/bin/python3.11")
            .with_node_path("/usr/local/bin/node");
        assert_eq!(sandbox.python_path, "/usr/bin/python3.11");
        assert_eq!(sandbox.node_path, "/usr/local/bin/node");
    }

    #[test]
    fn test_rust_unsupported() {
        let sandbox = LocalSandbox::new();
        let result = sandbox.build_command("fn main(){}", Language::Rust);
        assert!(result.is_err());
        match result.unwrap_err() {
            SandboxError::UnsupportedLanguage(msg) => {
                assert!(msg.contains("Rust"));
            }
            other => panic!("expected UnsupportedLanguage, got: {:?}", other),
        }
    }

    #[test]
    fn test_wasm_unsupported() {
        // #201: a subprocess sandbox has nothing to hand wasm to; `WasmSandbox` owns it.
        let sandbox = LocalSandbox::new();
        let result = sandbox.build_command("(module)", Language::Wasm);
        assert!(result.is_err());
        match result.unwrap_err() {
            SandboxError::UnsupportedLanguage(msg) => {
                assert!(msg.contains("WasmSandbox"), "unexpected msg: {msg}");
            }
            other => panic!("expected UnsupportedLanguage, got: {:?}", other),
        }
    }

    #[test]
    fn test_dangerous_python_import_detection() {
        assert!(contains_dangerous_python_import("import os").is_some());
        assert!(contains_dangerous_python_import("from sys import path").is_some());
        assert!(contains_dangerous_python_import("import subprocess").is_some());
        assert!(contains_dangerous_python_import("import math").is_none());
        assert!(contains_dangerous_python_import("import json").is_none());
        assert!(contains_dangerous_python_import("from datetime import datetime").is_none());
        assert!(contains_dangerous_python_import("# import os").is_none());
    }

    #[test]
    fn test_dangerous_python_call_detection() {
        // Explicit bypass routes that defeat the import-statement scan (H4).
        assert!(contains_dangerous_python_call("__import__('os').system('id')").is_some());
        assert!(contains_dangerous_python_call("importlib.import_module('os')").is_some());
        assert!(contains_dangerous_python_call("exec('print(1)')").is_some());
        assert!(contains_dangerous_python_call("eval('1+1')").is_some());
        assert!(contains_dangerous_python_call("code = compile('a=1', '<s>', 'exec')").is_some());
        assert!(contains_dangerous_python_call("__builtins__.__dict__").is_some());
        // Benign identifiers must not false-positive.
        assert!(contains_dangerous_python_call("evaluate(fn).run()").is_none());
        assert!(contains_dangerous_python_call("x = 1 + evalue").is_none());
    }

    #[test]
    fn test_dangerous_javascript_detection() {
        // module requires (H1)
        assert!(
            contains_dangerous_javascript("require('fs').readFileSync('/etc/passwd')").is_some()
        );
        assert!(contains_dangerous_javascript("require(\"child_process\")").is_some());
        assert!(contains_dangerous_javascript("const net = require('net')").is_some());
        // statement-level dangerous calls
        assert!(contains_dangerous_javascript("process.env").is_some());
        assert!(contains_dangerous_javascript("eval('console.log(1)')").is_some());
        assert!(contains_dangerous_javascript("new Function('x', 'return x')").is_some());
        assert!(contains_dangerous_javascript("globalThis.foo = 1").is_some());
        // benign code and comment lines pass
        assert!(contains_dangerous_javascript("console.log(1 + 2)").is_none());
        assert!(contains_dangerous_javascript("JSON.parse(x).map(v => v * 2)").is_none());
        assert!(contains_dangerous_javascript("// require('fs')").is_none());
        assert!(contains_dangerous_javascript("/* blocked? no: require('fs') */").is_none());
        assert!(contains_dangerous_javascript("evaluate(fn).execute()").is_none());
    }

    /// T2: whitespace padding and comment-splits must not hide a blocked module / call.
    #[test]
    fn test_dangerous_javascript_bypass_attempts_blocked() {
        // require(...) with whitespace between tokens.
        assert!(contains_dangerous_javascript("require( 'fs' )").is_some());
        assert!(contains_dangerous_javascript("require\n('child_process')").is_some());
        assert!(contains_dangerous_javascript("const x = require('net')\n;doNothing()").is_some());
        // member-access whitespace (`process . env`) and call splitting.
        assert!(contains_dangerous_javascript("process .env").is_some());
        assert!(contains_dangerous_javascript("eval (\"1\")").is_some());
        // block-comment padding around a real call is still caught.
        assert!(contains_dangerous_javascript("/* note */ eval('1')").is_some());
        // line + block comments are fully stripped before scanning.
        assert!(
            contains_dangerous_javascript("const a = 1; // require('fs')\nprocess.exit()")
                .is_some()
        );
    }

    #[tokio::test]
    async fn test_javascript_dangerous_api_blocked() {
        let sandbox = LocalSandbox::new();
        let result = sandbox
            .run(
                "require('fs').readFileSync('/etc/passwd')",
                Language::JavaScript,
                10_000,
            )
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("dangerous Node.js API"),
            "Expected dangerous Node.js API error, got: {}",
            err
        );
    }

    #[test]
    fn dangerous_ops_denied_by_default_and_opt_in_required() {
        let sandbox = LocalSandbox::new();
        // Default: dangerous bypass routes are denied.
        assert!(sandbox
            .build_command("__import__('os').system('id')", Language::Python)
            .is_err());
        assert!(sandbox
            .build_command("importlib.import_module('os')", Language::Python)
            .is_err());
        assert!(sandbox
            .build_command("exec('import os')", Language::Python)
            .is_err());
        assert!(sandbox
            .build_command("require('child_process').exec('id')", Language::JavaScript)
            .is_err());
        // Without opt-in, benign code still passes even with the call filter on.
        assert!(sandbox
            .build_command("print('hi')", Language::Python)
            .is_ok());
        // Explicit opt-in relaxes the filter (command builds).
        let allowed = LocalSandbox::new().with_dangerously_allow(true);
        assert!(allowed
            .build_command("__import__('os').system('id')", Language::Python)
            .is_ok());
        assert!(allowed
            .build_command("exec('import os')", Language::Python)
            .is_ok());
        assert!(allowed
            .build_command("require('child_process').exec('id')", Language::JavaScript)
            .is_ok());
    }

    #[tokio::test]
    async fn test_python_dangerous_import_blocked() {
        let sandbox = LocalSandbox::new();
        let result = sandbox
            .run("import os; print(os.getcwd())", Language::Python, 10_000)
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("dangerous import"),
            "Expected dangerous import error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_python_execution_if_available() {
        let sandbox = LocalSandbox::new();
        let result = sandbox.run("print(1 + 2)", Language::Python, 10_000).await;

        match result {
            Ok(run_result) => {
                if run_result.exit_code == 0 {
                    assert!(
                        run_result.stdout.trim() == "3",
                        "expected '3', got '{}'",
                        run_result.stdout.trim()
                    );
                }
            }
            Err(SandboxError::Runtime(msg)) => {
                eprintln!("Python not available (expected in some CI): {}", msg);
            }
            Err(other) => panic!("unexpected error: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_javascript_execution_if_available() {
        let sandbox = LocalSandbox::new();
        let result = sandbox
            .run("console.log(1 + 2)", Language::JavaScript, 10_000)
            .await;

        match result {
            Ok(run_result) => {
                if run_result.exit_code == 0 {
                    assert!(
                        run_result.stdout.trim() == "3",
                        "expected '3', got '{}'",
                        run_result.stdout.trim()
                    );
                }
            }
            Err(SandboxError::Runtime(msg)) => {
                eprintln!("Node.js not available (expected in some CI): {}", msg);
            }
            // Warm Windows runners occasionally let `node -e` blow the 10s
            // budget (Defender scan / host load); this is an "if available"
            // smoke test, so treat an environmental timeout like no runtime
            // rather than a code regression (observed 2026-09-14, passed on
            // the same code the two previous runs).
            Err(SandboxError::Timeout(ms)) => {
                eprintln!("Node.js present but too slow in this CI environment ({ms}ms)");
            }
            Err(other) => panic!("unexpected error: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_rust_execution_unsupported() {
        let sandbox = LocalSandbox::new();
        let result = sandbox.run("fn main(){}", Language::Rust, 10_000).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            SandboxError::UnsupportedLanguage(_) => {}
            other => panic!("expected UnsupportedLanguage, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_execution_timeout() {
        let sandbox = LocalSandbox::new();
        let result = sandbox
            .run("import time; time.sleep(10)", Language::Python, 100)
            .await;

        match result {
            Err(SandboxError::Timeout(ms)) => {
                assert_eq!(ms, 100);
            }
            Ok(_) => {
                // Python not available, code didn't run — acceptable
            }
            Err(other) => panic!("expected Timeout, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_execution_time_is_recorded() {
        let sandbox = LocalSandbox::new();
        let result = sandbox.run("print('hi')", Language::Python, 10_000).await;

        if let Ok(run_result) = result {
            assert!(run_result.execution_time_ms < 10_000);
        }
    }

    #[tokio::test]
    async fn test_stderr_captured() {
        let sandbox = LocalSandbox::new();
        let result = sandbox
            .run(
                "import sys; print('error', file=sys.stderr)",
                Language::Python,
                10_000,
            )
            .await;

        if let Ok(run_result) = result {
            if run_result.exit_code == 0 {
                assert!(
                    run_result.stderr.contains("error"),
                    "stderr should contain 'error', got: '{}'",
                    run_result.stderr
                );
            }
        }
    }
}
