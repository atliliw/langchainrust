//! Sessions 集成测试 - 需要 API Key
//!
//! 测试多轮对话的会话记忆。
//!
//! 手动运行:
//! ```bash
//! cargo test --test integration_sessions -- --ignored
//! ```

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::sessions::{MemorySessionStore, SessionManager};
use std::sync::Arc;

#[tokio::test]
#[ignore = "需要 API Key"]
async fn test_session_multi_turn_memory() {
    let manager = SessionManager::new(Arc::new(MemorySessionStore::new()));
    let llm = TestConfig::get().openai_chat();
    let id = manager.create_session().await.expect("创建会话失败");

    let r1 = manager
        .chat(&id, &llm, "请记住我叫张三".to_string())
        .await
        .expect("第一轮失败");
    println!("回复1: {}", r1);

    let r2 = manager
        .chat(&id, &llm, "我叫什么名字?".to_string())
        .await
        .expect("第二轮失败");
    println!("回复2: {}", r2);
    assert!(r2.contains("张三"), "会话应记住名字");

    // 2 轮对话 = 4 条消息(2 human + 2 ai)
    let history = manager.history(&id).await.expect("获取历史失败");
    assert_eq!(history.len(), 4);
}
