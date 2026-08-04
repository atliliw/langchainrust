// src/language_models/providers/anthropic/tests.rs
//! Tests for the Anthropic module.

use super::chat::AnthropicChat;
use super::config::{AnthropicConfig, ThinkingConfig, ThinkingType};
use super::types::{AnthropicContent, AnthropicDelta};

#[test]
fn test_thinking_type_default() {
    assert_eq!(ThinkingType::default(), ThinkingType::Disabled);
}

#[test]
fn test_thinking_config_enabled() {
    let config = ThinkingConfig::enabled(10000);
    assert!(config.is_enabled());
    assert_eq!(config.budget_tokens, 10000);
}

#[test]
fn test_thinking_config_disabled() {
    let config = ThinkingConfig::disabled();
    assert!(!config.is_enabled());
}

#[test]
fn test_anthropic_config_with_thinking() {
    let config = AnthropicConfig::new("test-key").with_thinking(ThinkingConfig::enabled(5000));
    assert!(config.thinking.is_enabled());
    assert_eq!(config.thinking.budget_tokens, 5000);
}

#[test]
fn test_anthropic_config_default_no_thinking() {
    let config = AnthropicConfig::default();
    assert!(!config.thinking.is_enabled());
}

#[test]
fn test_anthropic_chat_with_thinking() {
    let config = AnthropicConfig::new("test-key");
    let chat = AnthropicChat::new(config).with_thinking(8000);
    assert!(chat.thinking_config().is_enabled());
    assert_eq!(chat.thinking_config().budget_tokens, 8000);
}

#[test]
fn test_build_request_body_without_thinking() {
    let config = AnthropicConfig::new("test-key");
    let chat = AnthropicChat::new(config);
    let body = chat.build_request_body(vec![], false);
    assert!(body.get("thinking").is_none());
}

#[test]
fn test_build_request_body_with_thinking() {
    let config = AnthropicConfig::new("test-key").with_thinking(ThinkingConfig::enabled(10000));
    let chat = AnthropicChat::new(config);
    let body = chat.build_request_body(vec![], false);

    let thinking = body.get("thinking").expect("thinking should be present");
    assert_eq!(thinking["type"], "enabled");
    assert_eq!(thinking["budget_tokens"], 10000);
}

#[test]
fn test_anthropic_content_deserialize_thinking() {
    let json = r#"{"type": "thinking", "thinking": "Let me analyze..."}"#;
    let content: AnthropicContent = serde_json::from_str(json).unwrap();
    assert_eq!(content.content_type, "thinking");
    assert_eq!(content.thinking, "Let me analyze...");
}

#[test]
fn test_anthropic_content_deserialize_text() {
    let json = r#"{"type": "text", "text": "The answer is 42."}"#;
    let content: AnthropicContent = serde_json::from_str(json).unwrap();
    assert_eq!(content.content_type, "text");
    assert_eq!(content.text, "The answer is 42.");
}

#[test]
fn test_anthropic_delta_deserialize_thinking_delta() {
    let json = r#"{"type": "thinking_delta", "thinking": "Hmm..."}"#;
    let delta: AnthropicDelta = serde_json::from_str(json).unwrap();
    assert_eq!(delta.type_field, "thinking_delta");
    assert_eq!(delta.thinking, "Hmm...");
}

#[test]
fn test_anthropic_delta_deserialize_text_delta() {
    let json = r#"{"type": "text_delta", "text": "Hello"}"#;
    let delta: AnthropicDelta = serde_json::from_str(json).unwrap();
    assert_eq!(delta.type_field, "text_delta");
    assert_eq!(delta.text, "Hello");
}
