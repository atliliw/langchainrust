// lc-providers/src/providers/azure/tests.rs

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
    let config = AzureOpenAIConfig::new(
        "https://myresource.openai.azure.com",
        "gpt-4-deployment",
        "test-key",
    );
    assert_eq!(config.endpoint, "https://myresource.openai.azure.com");
    assert_eq!(config.deployment_name, "gpt-4-deployment");
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.api_version, AZURE_DEFAULT_API_VERSION);
}

#[test]
fn test_config_builder() {
    let config = AzureOpenAIConfig::new("https://res.openai.azure.com", "deploy", "key")
        .with_api_version("2024-06-01")
        .with_model("gpt-4o")
        .with_temperature(0.5)
        .with_max_tokens(2048)
        .with_top_p(0.9);
    assert_eq!(config.api_version, "2024-06-01");
    assert_eq!(config.model, Some("gpt-4o".to_string()));
    assert_eq!(config.temperature, Some(0.5));
    assert_eq!(config.max_tokens, Some(2048));
    assert_eq!(config.top_p, Some(0.9));
}

#[test]
fn test_config_from_env_result_ok() {
    let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let old_ep = save_and_set("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com");
    let old_dn = save_and_set("AZURE_OPENAI_DEPLOYMENT_NAME", "my-deploy");
    let old_key = save_and_set("AZURE_OPENAI_API_KEY", "azure-key-123");
    let result = AzureOpenAIConfig::from_env_result();
    assert!(result.is_ok());
    let config = result.unwrap();
    assert_eq!(config.endpoint, "https://test.openai.azure.com");
    assert_eq!(config.deployment_name, "my-deploy");
    assert_eq!(config.api_key, "azure-key-123");
    restore("AZURE_OPENAI_ENDPOINT", old_ep);
    restore("AZURE_OPENAI_DEPLOYMENT_NAME", old_dn);
    restore("AZURE_OPENAI_API_KEY", old_key);
}

#[test]
fn test_config_from_env_result_err_when_missing() {
    let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let old_ep = std::env::var("AZURE_OPENAI_ENDPOINT").ok();
    std::env::remove_var("AZURE_OPENAI_ENDPOINT");
    let result = AzureOpenAIConfig::from_env_result();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("AZURE_OPENAI_ENDPOINT"));
    restore("AZURE_OPENAI_ENDPOINT", old_ep);
}

#[test]
fn test_chat_url() {
    let config = AzureOpenAIConfig::new("https://myresource.openai.azure.com", "gpt4", "key");
    let url = config.chat_url();
    assert!(url.contains("myresource.openai.azure.com"));
    assert!(url.contains("/openai/deployments/gpt4/chat/completions"));
    assert!(url.contains("api-version="));
}

#[test]
fn test_chat_url_trailing_slash() {
    let config = AzureOpenAIConfig::new("https://myresource.openai.azure.com/", "gpt4", "key");
    let url = config.chat_url();
    // Should not have double slashes
    assert!(!url.contains("//openai"));
}

#[test]
fn test_effective_model() {
    let config_no_model = AzureOpenAIConfig::new("https://ep", "deploy", "key");
    assert_eq!(config_no_model.effective_model(), "deploy");

    let config_with_model =
        AzureOpenAIConfig::new("https://ep", "deploy", "key").with_model("gpt-4o");
    assert_eq!(config_with_model.effective_model(), "gpt-4o");
}

#[test]
fn test_chat_new() {
    let config = AzureOpenAIConfig::new("https://ep", "deploy", "key");
    let _chat = AzureOpenAIChat::new(config);
}

#[test]
fn test_model_name() {
    let config = AzureOpenAIConfig::new("https://ep", "deploy", "key").with_model("gpt-4o");
    let chat = AzureOpenAIChat::new(config);
    assert_eq!(chat.model_name(), "gpt-4o");
}

#[test]
fn test_build_request_body_no_model() {
    let config = AzureOpenAIConfig::new("https://ep", "deploy", "key");
    let chat = AzureOpenAIChat::new(config);
    let body = chat.build_request_body(vec![Message::human("hello")], false);
    // Azure request body should NOT contain "model" field
    assert!(body.get("model").is_none());
    assert!(body.get("messages").is_some());
}

#[test]
fn test_error_display() {
    let err = AzureOpenAIError::Http("timeout".to_string());
    assert!(err.to_string().contains("HTTP error"));
    let err = AzureOpenAIError::Api("rate limit".to_string());
    assert!(err.to_string().contains("API error"));
    let err = AzureOpenAIError::Parse("bad json".to_string());
    assert!(err.to_string().contains("parse error"));
}
