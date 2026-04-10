// tests/integration/langsmith_connection.rs
//! LangSmith 连接测试

use langchainrust::{
    OpenAIChat, OpenAIConfig, BaseChatModel,
    CallbackManager, LangSmithHandler, RunTree, RunType,
    RunnableConfig,
};
use langchainrust::schema::Message;
use std::sync::Arc;

#[tokio::test]
async fn test_langsmith_connection() {
    println!("\n=== LangSmith 连接测试 ===\n");
    
    // 创建 LangSmith 处理器
    let handler = match LangSmithHandler::from_env() {
        Ok(h) => {
            println!("✅ LangSmith 配置成功");
            Arc::new(h)
        }
        Err(e) => {
            println!("❌ LangSmith 配置失败: {}", e);
            println!("\n请设置以下环境变量:");
            println!("  export LANGSMITH_API_KEY=lsv2_xxx");
            println!("  export LANGSMITH_PROJECT=my-project");
            return;
        }
    };
    
    let callbacks = Arc::new(CallbackManager::new().add_handler(handler));
    
    // 创建 LLM
    let config = OpenAIConfig::from_env();
    println!("API Base URL: {}", config.base_url);
    println!("Model: {}", config.model);
    
    let llm = OpenAIChat::new(config);
    
    // 执行调用
    let run_config = RunnableConfig::new()
        .with_callbacks(callbacks)
        .with_run_name("langsmith_test");
    
    let messages = vec![
        Message::human("说 hello，什么是rust"),
    ];
    
    println!("\n正在调用 LLM...");
    let result = llm.chat(messages, Some(run_config)).await;
    
    match result {
        Ok(response) => {
            println!("✅ LLM 响应: {}", response.content);
            println!("\n=== 查看追踪 ===");
            println!("打开浏览器访问: https://smith.langchain.com");
            println!("项目: pr-potable-dime-18");
        }
        Err(e) => {
            println!("❌ LLM 调用失败: {}", e);
        }
    }
}