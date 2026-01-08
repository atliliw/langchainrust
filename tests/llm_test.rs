#[path = "common.rs"]
mod common;
use common::create_test_llm;


use langchainrust::llms::{OpenAIConfig, LLM};
use langchainrust::messages::Message;
pub use langchainrust::prompts::ChatPromptTemplate;


#[tokio::test]
async fn test_llm_generate() {
    let llm = create_test_llm();
    let result = llm.generate("Hello, world!").await;
    println!("Result: {:?}", result);
    assert!(result.is_ok(), "LLM call failed: {:?}", result.err());
}

#[tokio::test]
async fn test_llm_generate11() {
    let llm = create_test_llm();
    let result = llm.generate("Hello, world!").await;

    match &result {
        Ok(text) => println!("✅ Success: {}", text),
        Err(e) => eprintln!("❌ Error: {}", e),
    }

    assert!(result.is_ok(), "LLM call failed: {:?}", result.err());
}

#[tokio::test]
async fn test_with_template() {
    let llm = create_test_llm();

    let template = ChatPromptTemplate::new(vec![
        Message::system("你是由{name}开发的AI助手，专精于{field}领域。请用清晰易懂的方式回答问题。"),
        Message::human("请向初学者解释{topic}是什么。")
    ]);

    let mut values = std::collections::HashMap::new();
    values.insert("name", "阿里云");
    values.insert("field", "人工智能");
    values.insert("topic", "大语言模型");

    let result = llm.invoke_chat_template(&template, &values).await;
    println!("{:?}", result);
    assert!(result.is_ok(), "Template invocation failed: {:?}", result.err());
}