#[path = "common.rs"]
mod common;

use crate::common::llm_Config;
use langchainrust::agent::{AgentExecutor, ReActAgent};
use langchainrust::llms::LLM;
use langchainrust::memory::SimpleMemory;
use langchainrust::tools::{Calculator, Tool};
use langchainrust::messages::Message;
use langchainrust::prompts::ChatPromptTemplate;
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test]
async fn test_agent_with_memory_and_template_like_chain() {
    let config = llm_Config();
    let llm = LLM::new(config);

    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Calculator)];

    let template = ChatPromptTemplate::new(vec![
        Message::system("你是数学家{name}，回答风格是{style}。"),
        Message::human("请详细计算：{input}。"),
        Message::human("请把结果乘{multiplier}后给出最终答案。"),
        Message::human("回答时附带你的名字：{name}。"),
    ]);

    let agent = ReActAgent::with_template(llm, tools.clone(), Some(Box::new(SimpleMemory::default())), template);
    let executor = AgentExecutor::new(Box::new(agent), tools).with_max_iterations(3);

    let vars1: HashMap<String, String> = HashMap::from([
        ("name".to_string(), "小李".to_string()),
        ("style".to_string(), "简洁".to_string()),
        ("multiplier".to_string(), "1".to_string()),
    ]);

    let result1 = executor.run_with_vars("1+3", vars1).await;
    assert!(result1.is_ok(), "第一次执行失败: {:?}", result1.err());
    println!("第一次结果: {}", result1.unwrap());

    let vars2: HashMap<String, String> = HashMap::from([
        ("name".to_string(), "小李".to_string()),
        ("style".to_string(), "简洁".to_string()),
        ("multiplier".to_string(), "100".to_string()),
    ]);

    let result2 = executor
        .run_with_vars("上一步结果", vars2)
        .await;
    assert!(result2.is_ok(), "第二次执行失败: {:?}", result2.err());
    println!("第二次结果: {}", result2.unwrap());
}

