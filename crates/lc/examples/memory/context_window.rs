//! ContextWindow 示例
//!
//! 展示如何使用 ContextWindow 管理长上下文对话。
//!
//! # 运行
//! ```bash
//! cargo run --example memory_context_window
//! ```

use langchainrust::{ContextWindow, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ContextWindow 示例 ===\n");

    // 1. Truncate 策略: 超限时丢弃最旧消息
    //    ContextWindow<OpenAIChat> 是默认泛型,Truncate 策略不需要 LLM
    let cw: ContextWindow<langchainrust::OpenAIChat> = ContextWindow::new(4096)?;

    let messages = vec![
        Message::system("你是一个助手"),
        Message::human("第一个问题"),
        Message::ai("第一个回答"),
        Message::human("第二个问题"),
        Message::ai("第二个回答"),
        Message::human("最新问题"),
    ];

    let fitted = cw.fit(messages.clone()).await?;
    println!(
        "Truncate 策略: {} 条消息 → {} 条消息(在 4096 token 内)",
        messages.len(),
        fitted.len()
    );

    // 2. Summarize 策略: 超限时用 LLM 压缩旧消息
    println!("\nSummarize 策略:");
    println!("  let llm = OpenAIChat::new(config);");
    println!("  let cw = ContextWindow::with_strategy(4096, Strategy::summarize(llm));");
    println!("  let fitted = cw.fit(messages).await?;");
    println!("\n工作流程:");
    println!("  1. 统计消息总 token 数");
    println!("  2. 若超限,找到保留最新消息的分割点");
    println!("  3. 用 LLM 将旧消息压缩为摘要");
    println!("  4. 返回: system + [摘要] + 最新消息");

    // 3. 自定义 prompt 的 Summarize
    println!("\n自定义摘要 prompt:");
    println!("  let cw = ContextWindow::with_strategy(");
    println!("      4096,");
    println!(
        "      Strategy::summarize_with_prompt(llm, \"请用中文总结: {{conversation}}\\n摘要:\"),"
    );
    println!("  );");

    Ok(())
}
