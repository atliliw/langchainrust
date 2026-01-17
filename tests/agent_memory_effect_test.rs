#[path = "common.rs"]
mod common;

use langchainrust::agent::{Agent, ReActAgent};
use langchainrust::llms::LLM;
use langchainrust::memory::SimpleMemory;
use langchainrust::tools::{Calculator, Tool};
use std::sync::Arc;

#[tokio::test]
async fn test_agent_memory_accumulates_outputs() {
    let llm = LLM::new(common::create_test_llm_config_streaming());
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Calculator)];

    let agent = ReActAgent::new(llm, tools, Some(Box::new(SimpleMemory::default())));

    agent.add_memory("step1", "第一次输出：1+2=3");
    agent.add_memory("step2", "第二次输出：3*4=12");

    let mem_ctx = agent.memory_context();
    assert!(mem_ctx.contains("第一次输出：1+2=3"));
    assert!(mem_ctx.contains("第二次输出：3*4=12"));
}
