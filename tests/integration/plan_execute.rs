//! Plan-Execute Agent 集成测试 - 需要 API Key
//!
//! 测试规划-执行-重规划流程。
//!
//! 手动运行:
//! ```bash
//! cargo test --test integration_plan_execute -- --ignored
//! ```

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::tools::Calculator;
use langchainrust::{BaseTool, PlanExecuteAgent};
use std::sync::Arc;

#[tokio::test]
#[ignore = "需要 API Key"]
async fn test_plan_execute_agent_run() {
    let llm = TestConfig::get().openai_chat();
    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
    let agent = PlanExecuteAgent::new(llm, tools).with_max_replans(1);

    let result = agent.run("计算 25 + 17 并解释结果").await;
    println!("结果: {:?}", result);
    assert!(result.is_ok(), "Plan-Execute 应成功");
    assert!(!result.unwrap().is_empty());
}
