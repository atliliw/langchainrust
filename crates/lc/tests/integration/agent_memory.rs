//! 整合测试 - Agent + Tools + Memory

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::schema::Message;
use langchainrust::BaseChatModel;
use langchainrust::{
    AgentExecutor, BaseAgent, BaseTool, Calculator, ChatMessageHistory, ReActAgent, SimpleMathTool,
};
use std::sync::Arc;

/// 测试 Agent 多轮对话 + 工具调用
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_agent_with_memory_multi_turn() {
    let config = TestConfig::get();
    let llm = config.openai_chat();
    let mut history = ChatMessageHistory::new();

    let tools: Vec<Arc<dyn BaseTool>> =
        vec![Arc::new(Calculator::new()), Arc::new(SimpleMathTool::new())];

    let agent = ReActAgent::new(config.openai_chat(), tools.clone(), None);
    let executor =
        AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools).with_max_iterations(5);

    println!("=== 第一轮：介绍名字 ===");
    let user_msg1 = "My name is Alice.";

    let mut messages1 = vec![Message::system("You are a helpful assistant.")];
    messages1.extend(history.messages().iter().cloned());
    messages1.push(Message::human(user_msg1));

    let response1 = llm.chat(messages1, None).await.unwrap();
    println!("User: {}", user_msg1);
    println!("Assistant: {}", response1.content);

    history.add_message(Message::human(user_msg1));
    history.add_message(Message::ai(&response1.content));

    println!("\n=== 第二轮：问名字 ===");
    let user_msg2 = "What's my name?";

    let mut messages2 = vec![Message::system("You are a helpful assistant.")];
    messages2.extend(history.messages().iter().cloned());
    messages2.push(Message::human(user_msg2));

    let response2 = llm.chat(messages2, None).await.unwrap();
    println!("User: {}", user_msg2);
    println!("Assistant: {}", response2.content);

    assert!(response2.content.to_lowercase().contains("alice"));

    history.add_message(Message::human(user_msg2));
    history.add_message(Message::ai(&response2.content));

    println!("\n=== 第三轮：使用工具计算 ===");
    let user_msg3 = "Calculate 37 + 58.";

    println!("User: {}", user_msg3);
    let response3 = executor.invoke(user_msg3.to_string()).await.unwrap();
    println!("Assistant: {}", response3);

    assert!(response3.contains("95"));

    history.add_message(Message::human(user_msg3));
    history.add_message(Message::ai(&response3));

    println!("\n=== 第四轮：问计算结果 ===");
    let user_msg4 = "What was the result?";

    let mut messages4 = vec![Message::system("Remember previous conversations.")];
    messages4.extend(history.messages().iter().cloned());
    messages4.push(Message::human(user_msg4));

    let response4 = llm.chat(messages4, None).await.unwrap();
    println!("User: {}", user_msg4);
    println!("Assistant: {}", response4.content);

    assert!(response4.content.contains("95"));
}
