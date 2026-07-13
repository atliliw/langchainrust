//! Handoffs 集成测试 - 需要 API Key
//!
//! 手动运行:
//! ```bash
//! cargo test --test integration_handoffs -- --ignored
//! ```

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::agents::handoffs::HandoffManager;
use langchainrust::{AgentExecutor, BaseAgent, FunctionCallingAgent};
use std::sync::Arc;

#[tokio::test]
#[ignore = "需要 API Key"]
async fn test_handoff_manager_run_primary() {
    let llm = TestConfig::get().openai_chat();
    let agent = FunctionCallingAgent::new(llm, vec![], None);
    let executor = Arc::new(AgentExecutor::new(
        Arc::new(agent) as Arc<dyn BaseAgent>,
        vec![],
    ));

    let mgr = HandoffManager::new();
    mgr.register_agent("assistant", executor).unwrap();
    mgr.set_primary("assistant").unwrap();

    let result = mgr.run("一句话介绍 Rust".to_string()).await.unwrap();
    println!("结果: {}", result);
    assert!(!result.is_empty());
}
