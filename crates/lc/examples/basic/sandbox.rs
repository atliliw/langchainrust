//! Code sandbox example
//!
//! Demonstrates LocalSandbox (subprocess + timeout). Note: LocalSandbox is a plain
//! subprocess with **no OS-level isolation** — the bundled import blacklist is a noise
//! filter, not a security boundary. It is disabled by default when wrapped in
//! [`SandboxTool`]; untrusted code must run in a real sandbox (container / VM / WASM).
//!
//! # Run
//! ```bash
//! cargo run --example basic_sandbox
//! ```

use langchainrust::tools::sandbox::{CodeSandbox, Language, LocalSandbox, SandboxTool};
use langchainrust::BaseTool;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create a LocalSandbox
    let sandbox = LocalSandbox::new();

    // 2. Run Python directly (timeout is specified in the run() call)
    println!("=== Running Python ===");
    let result = sandbox
        .run("print(2 + 2)", Language::Python, 10_000)
        .await?;
    println!("stdout: {}", result.stdout.trim());
    println!("exit_code: {}", result.exit_code);
    println!("elapsed: {}ms", result.execution_time_ms);

    // 3. Run JavaScript
    println!("\n=== Running JavaScript ===");
    let result = sandbox
        .run(
            "console.log(Math.floor(Math.PI * 100))",
            Language::JavaScript,
            10_000,
        )
        .await?;
    println!("stdout: {}", result.stdout.trim());

    // 4. Timeout test (3s timeout, code sleeps 30s)
    println!("\n=== Timeout test ===");
    let result = sandbox
        .run("import time; time.sleep(30)", Language::Python, 3_000)
        .await;
    match result {
        Err(e) => println!("Expected timeout: {}", e),
        Ok(r) => println!("Unexpected success: {:?}", r),
    }

    // 5. Dangerous imports are blocked
    println!("\n=== Dangerous import blocked ===");
    let result = sandbox
        .run("import os; print(os.getcwd())", Language::Python, 10_000)
        .await;
    match result {
        Err(e) => println!("Expected block: {}", e),
        Ok(r) => println!("Unexpected success: {:?}", r),
    }

    // 6. Use it as an Agent tool (wrapped in SandboxTool)
    println!("\n=== As an Agent tool ===");
    let tool = SandboxTool::new(LocalSandbox::new(), Language::Python)
        .with_timeout(10_000)
        .with_dangerously_allow(true);
    let input = r#"{"code": "import json\nprint(json.dumps({'result': 42}))"}"#;
    let output = tool.run(input.to_string()).await?;
    println!("tool output: {}", output);

    Ok(())
}
