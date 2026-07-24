//! Streaming Agent 集成测试 - 需要 API Key
//!
//! 手动运行:
//! ```bash
//! cargo test --test integration_streaming -- --ignored
//! ```

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use futures_util::StreamExt;
use langchainrust::agents::streaming::{AgentStreamEvent, StreamingFunctionCallingAgent};

#[tokio::test]
#[ignore = "需要 API Key"]
async fn test_invoke_stream() {
    let llm = TestConfig::get().openai_chat();
    let agent = StreamingFunctionCallingAgent::new(llm);
    let mut stream = agent.invoke_stream("一句话介绍 Rust".to_string()).await;

    let mut full = String::new();
    let mut had_text = false;
    let mut had_final = false;
    while let Some(event) = stream.next().await {
        match event {
            AgentStreamEvent::Text { content } => {
                full.push_str(&content);
                had_text = true;
            }
            AgentStreamEvent::FinalAnswer { content } => {
                had_final = true;
                assert_eq!(content, full, "FinalAnswer 应等于拼接的文本");
            }
            _ => {}
        }
    }
    assert!(had_text, "应有 Text 事件");
    assert!(had_final, "应有 FinalAnswer 事件");
    assert!(!full.is_empty(), "内容不应为空");
    println!("流式输出: {}", full);
}
