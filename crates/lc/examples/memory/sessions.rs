//! Sessions 示例
//!
//! 展示如何使用 SessionManager 管理多轮对话会话。
//!
//! # 运行
//! ```bash
//! cargo run --example memory_sessions
//! ```

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sessions 示例 ===\n");

    // SessionManager 管理多个独立的对话会话
    println!("Sessions 功能:");
    println!("1. 创建/恢复对话会话");
    println!("2. 每个会话独立维护对话历史");
    println!("3. 支持会话持久化(可选)");
    println!("4. 支持会话过期自动清理");

    println!("\n使用场景:");
    println!("- 多用户聊天机器人");
    println!("- 长期对话上下文管理");
    println!("- 需要恢复之前的对话");

    println!("\nSessionManager 使用方式:");
    println!("  let manager = SessionManager::new(llm, memory);");
    println!("  let session_id = manager.create_session().await?;");
    println!("  let response = manager.chat(&session_id, \"你好\").await?;");
    println!("  // 同一会话保持上下文");
    println!("  let response2 = manager.chat(&session_id, \"我刚才说了什么?\").await?;");

    println!("\n提示: 需要设置 OPENAI_API_KEY 才能进行真实调用。");
    Ok(())
}
