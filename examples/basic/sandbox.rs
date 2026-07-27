//! 代码沙箱示例
//!
//! 展示 LocalSandbox 的安全代码执行:子进程隔离 + 超时杀进程。
//!
//! # 运行
//! ```bash
//! cargo run --example basic_sandbox
//! ```

use langchainrust::tools::sandbox::{CodeSandbox, Language, LocalSandbox, SandboxTool};
use langchainrust::BaseTool;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 创建 LocalSandbox
    let sandbox = LocalSandbox::new();

    // 2. 直接执行 Python 代码（超时在 run() 调用时指定）
    println!("=== 执行 Python ===");
    let result = sandbox
        .run("print(2 + 2)", Language::Python, 10_000)
        .await?;
    println!("stdout: {}", result.stdout.trim());
    println!("exit_code: {}", result.exit_code);
    println!("耗时: {}ms", result.execution_time_ms);

    // 3. 执行 JavaScript
    println!("\n=== 执行 JavaScript ===");
    let result = sandbox
        .run(
            "console.log(Math.floor(Math.PI * 100))",
            Language::JavaScript,
            10_000,
        )
        .await?;
    println!("stdout: {}", result.stdout.trim());

    // 4. 超时测试（3 秒超时,代码要睡 30 秒）
    println!("\n=== 超时测试 ===");
    let result = sandbox
        .run("import time; time.sleep(30)", Language::Python, 3_000)
        .await;
    match result {
        Err(e) => println!("预期超时: {}", e),
        Ok(r) => println!("意外成功: {:?}", r),
    }

    // 5. 危险 import 被拦截
    println!("\n=== 危险 import 拦截 ===");
    let result = sandbox
        .run("import os; print(os.getcwd())", Language::Python, 10_000)
        .await;
    match result {
        Err(e) => println!("预期拦截: {}", e),
        Ok(r) => println!("意外成功: {:?}", r),
    }

    // 6. 作为 Agent 工具使用（SandboxTool 包装）
    println!("\n=== 作为 Agent 工具 ===");
    let tool = SandboxTool::new(LocalSandbox::new(), Language::Python).with_timeout(10_000);
    let input = r#"{"code": "import json\nprint(json.dumps({'result': 42}))"}"#;
    let output = tool.run(input.to_string()).await?;
    println!("工具输出: {}", output);

    Ok(())
}
