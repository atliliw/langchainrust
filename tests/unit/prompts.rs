//! 单元测试 - Prompts (PromptTemplate)

use langchainrust::prompts::{ChatPromptTemplate, PromptTemplate};
use langchainrust::schema::Message;
use std::collections::HashMap;

#[test]
fn test_prompt_template_basic() {
    let template = PromptTemplate::new("Hello, {name}! Today is {day}.");

    let mut vars = HashMap::new();
    vars.insert("name", "Alice");
    vars.insert("day", "Monday");

    let result = template.format(&vars).unwrap();
    assert_eq!(result, "Hello, Alice! Today is Monday.");
}

#[test]
fn test_prompt_template_multiple_vars() {
    let template = PromptTemplate::new("{greeting}, {name}! Welcome to {place}.");

    let mut vars = HashMap::new();
    vars.insert("greeting", "Hi");
    vars.insert("name", "Bob");
    vars.insert("place", "Rust");

    let result = template.format(&vars).unwrap();
    assert_eq!(result, "Hi, Bob! Welcome to Rust.");
}

#[test]
fn test_chat_prompt_template() {
    let template = ChatPromptTemplate::new(vec![
        Message::system("You are a {role}."),
        Message::human("Hello, I'm {name}."),
    ]);

    let mut vars = HashMap::new();
    vars.insert("role", "Rust expert");
    vars.insert("name", "Bob");

    let messages = template.format(&vars).unwrap();
    assert_eq!(messages.len(), 2);
    assert!(messages[0].content.contains("Rust expert"));
}

#[test]
fn test_chat_prompt_template_multi_turn() {
    let template = ChatPromptTemplate::new(vec![
        Message::system("You are a {role}."),
        Message::human("Hello"),
        Message::ai("Hi!"),
        Message::human("{question}"),
    ]);

    let mut vars = HashMap::new();
    vars.insert("role", "assistant");
    vars.insert("question", "What is Rust?");

    let messages = template.format(&vars).unwrap();
    assert_eq!(messages.len(), 4);
}
