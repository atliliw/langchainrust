// tests/prompts_test.rs

use langchainrust::prompts::{ChatPromptTemplate, PromptTemplate};
use std::collections::HashMap;
use langchainrust::messages::Message;



#[test]
fn test_prompt_from_user_perspective() {
    let template = "Hi, {user}! You have {count} messages.";
    let prompt = PromptTemplate::new(template);

    let mut input = HashMap::new();
    input.insert("user", "Bob");
    input.insert("count", "5");

    let result = prompt.format(&input).unwrap();
    assert_eq!(result, "Hi, Bob! You have 5 messages.");
}


#[test]
fn test_llm_generatePto() {
    let template = PromptTemplate::new(
        "请用{tone}风格解释{topic}。",
    );

    // ✅ 直接用字面量构造 HashMap<&str, &str>
    let values = HashMap::from([
        ("tone", "简洁"),
        ("topic", "Rust的所有权"),
    ]);

    let prompt = template.format(&values).unwrap();
    assert_eq!(prompt, "请用简洁风格解释Rust的所有权。");
    println!("{}", prompt);
}


#[test]
fn test_chat_prompt_format() {
    let chat_prompt = ChatPromptTemplate::new(vec![
        Message::system("你是一个{role}助手。".to_string()),
        Message::human("你好，{name}！".to_string()),
    ]);

    let values = HashMap::from([
        ("role", "编程"),
        ("name", "Alice"),
    ]);

    let messages = chat_prompt.format(&values).unwrap();

    assert_eq!(messages.len(), 2);

}
