//! 单元测试 - Schema (Message 类型)

use langchainrust::schema::{Message, MessageType};

#[test]
fn test_message_system() {
    let msg = Message::system("You are a helpful assistant");
    assert!(matches!(msg.message_type, MessageType::System));
    assert_eq!(msg.content, "You are a helpful assistant");
}

#[test]
fn test_message_human() {
    let msg = Message::human("What is Rust?");
    assert!(matches!(msg.message_type, MessageType::Human));
    assert_eq!(msg.content, "What is Rust?");
}

#[test]
fn test_message_ai() {
    let msg = Message::ai("Rust is a systems programming language");
    assert!(matches!(msg.message_type, MessageType::AI));
    assert_eq!(msg.content, "Rust is a systems programming language");
}

#[test]
fn test_message_tool() {
    let msg = Message::tool("call_123", "Result: 42");
    assert!(matches!(msg.message_type, MessageType::Tool { .. }));
    assert_eq!(msg.content, "Result: 42");
}
