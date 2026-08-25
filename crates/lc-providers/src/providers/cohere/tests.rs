// lc-providers/src/providers/cohere/tests.rs

use super::*;
use crate::ENV_TEST_LOCK;

fn save_and_set(key: &str, value: &str) -> Option<String> {
    let old = std::env::var(key).ok();
    std::env::set_var(key, value);
    old
}

fn restore(key: &str, old: Option<String>) {
    match old {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }
}

#[test]
fn test_config_new() {
    let config = CohereConfig::new("test-key");
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.base_url, COHERE_BASE_URL);
    assert_eq!(config.model, "command-r-plus");
}

#[test]
fn test_config_builder() {
    let config = CohereConfig::new("key")
        .with_model("command-r")
        .with_base_url("https://custom.cohere.com/v2")
        .with_temperature(0.5)
        .with_max_tokens(1024)
        .with_preamble("You are a helpful assistant.");
    assert_eq!(config.model, "command-r");
    assert_eq!(config.base_url, "https://custom.cohere.com/v2");
    assert_eq!(config.temperature, Some(0.5));
    assert_eq!(config.max_tokens, Some(1024));
    assert_eq!(
        config.preamble,
        Some("You are a helpful assistant.".to_string())
    );
}

#[test]
fn test_config_from_env_result_ok() {
    let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let old = save_and_set("COHERE_API_KEY", "env-key");
    let result = CohereConfig::from_env_result();
    assert!(result.is_ok());
    assert_eq!(result.unwrap().api_key, "env-key");
    restore("COHERE_API_KEY", old);
}

#[test]
fn test_config_from_env_result_err_when_missing() {
    let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let old = std::env::var("COHERE_API_KEY").ok();
    std::env::remove_var("COHERE_API_KEY");
    let result = CohereConfig::from_env_result();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("COHERE_API_KEY"));
    restore("COHERE_API_KEY", old);
}

#[test]
fn test_chat_new() {
    let config = CohereConfig::new("test-key");
    let _chat = CohereChat::new(config);
}

#[test]
fn test_model_name() {
    let config = CohereConfig::new("key").with_model("command-r");
    let chat = CohereChat::new(config);
    assert_eq!(chat.model_name(), "command-r");
}

#[test]
fn test_build_request_body() {
    let config = CohereConfig::new("key").with_preamble("System prompt");
    let chat = CohereChat::new(config);
    let body = chat.build_request_body(vec![Message::human("hello")], false);
    assert_eq!(body["model"], "command-r-plus");
    assert!(body.get("messages").is_some());
    assert_eq!(body["preamble"], "System prompt");
}

#[test]
fn test_error_display() {
    let err = CohereError::Http("timeout".to_string());
    assert!(err.to_string().contains("HTTP error"));
    let err = CohereError::Api("rate limit".to_string());
    assert!(err.to_string().contains("API error"));
    let err = CohereError::Parse("bad json".to_string());
    assert!(err.to_string().contains("parse error"));
}

#[test]
fn test_message_to_cohere_format_human() {
    let msg = Message::human("Hello");
    let formatted = CohereChat::message_to_cohere_format(&msg);
    assert_eq!(formatted["role"], "user");
    assert_eq!(formatted["content"], "Hello");
}

#[test]
fn test_message_to_cohere_format_system() {
    let msg = Message::system("You are helpful");
    let formatted = CohereChat::message_to_cohere_format(&msg);
    assert_eq!(formatted["role"], "system");
}
