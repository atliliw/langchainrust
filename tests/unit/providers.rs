// tests/unit/providers.rs
//! Unit tests for LLM providers

use langchainrust::{
    AnthropicConfig, DeepSeekConfig, MoonshotConfig, QwenConfig, ThinkingConfig, ThinkingType,
    ZhipuConfig,
};

#[test]
fn test_deepseek_config_default() {
    let config = DeepSeekConfig::default();
    assert_eq!(config.base_url, "https://api.deepseek.com/v1");
    assert_eq!(config.model, "deepseek-chat");
}

#[test]
fn test_deepseek_config_new() {
    let config = DeepSeekConfig::new("test-api-key");
    assert_eq!(config.api_key, "test-api-key");
}

#[test]
fn test_deepseek_config_with_model() {
    let config = DeepSeekConfig::new("test-key").with_model("deepseek-coder");
    assert_eq!(config.model, "deepseek-coder");
}

#[test]
fn test_deepseek_config_with_temperature() {
    let config = DeepSeekConfig::new("test-key").with_temperature(0.5);
    assert_eq!(config.temperature, Some(0.5));
}

#[test]
fn test_moonshot_config_default() {
    let config = MoonshotConfig::default();
    assert_eq!(config.base_url, "https://api.moonshot.cn/v1");
    assert_eq!(config.model, "moonshot-v1-8k");
}

#[test]
fn test_moonshot_config_with_model() {
    let config = MoonshotConfig::new("test-key").with_model("moonshot-v1-128k");
    assert_eq!(config.model, "moonshot-v1-128k");
}

#[test]
fn test_zhipu_config_default() {
    let config = ZhipuConfig::default();
    assert_eq!(config.base_url, "https://open.bigmodel.cn/api/paas/v4");
    assert_eq!(config.model, "glm-4-flash");
}

#[test]
fn test_zhipu_config_with_model() {
    let config = ZhipuConfig::new("test-key").with_model("glm-4");
    assert_eq!(config.model, "glm-4");
}

#[test]
fn test_qwen_config_default() {
    let config = QwenConfig::default();
    assert_eq!(
        config.base_url,
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    );
    assert_eq!(config.model, "qwen-plus");
}

#[test]
fn test_qwen_config_with_model() {
    let config = QwenConfig::new("test-key").with_model("qwen-max");
    assert_eq!(config.model, "qwen-max");
}

#[test]
fn test_anthropic_config_default() {
    let config = AnthropicConfig::default();
    assert_eq!(config.base_url, "https://api.anthropic.com/v1");
    assert_eq!(config.model, "claude-3-5-sonnet-20241022");
    assert_eq!(config.max_tokens, 4096);
}

#[test]
fn test_anthropic_config_new() {
    let config = AnthropicConfig::new("test-api-key");
    assert_eq!(config.api_key, "test-api-key");
}

#[test]
fn test_anthropic_config_with_model() {
    let config = AnthropicConfig::new("test-key").with_model("claude-3-opus-20240229");
    assert_eq!(config.model, "claude-3-opus-20240229");
}

#[test]
fn test_anthropic_config_with_max_tokens() {
    let config = AnthropicConfig::new("test-key").with_max_tokens(8192);
    assert_eq!(config.max_tokens, 8192);
}

#[test]
fn test_anthropic_config_with_temperature() {
    let config = AnthropicConfig::new("test-key").with_temperature(0.7);
    assert_eq!(config.temperature, Some(0.7));
}

#[test]
fn test_anthropic_config_with_system_prompt() {
    let config =
        AnthropicConfig::new("test-key").with_system_prompt("You are a helpful assistant.");
    assert_eq!(
        config.system_prompt,
        Some("You are a helpful assistant.".to_string())
    );
}

// ============================================================================
// Extended Thinking tests
// ============================================================================

#[test]
fn test_thinking_type_default() {
    assert_eq!(ThinkingType::default(), ThinkingType::Disabled);
}

#[test]
fn test_thinking_config_enabled() {
    let config = ThinkingConfig::enabled(10000);
    assert!(config.is_enabled());
    assert_eq!(config.budget_tokens, 10000);
    assert_eq!(config.r#type, ThinkingType::Enabled);
}

#[test]
fn test_thinking_config_disabled() {
    let config = ThinkingConfig::disabled();
    assert!(!config.is_enabled());
    assert_eq!(config.budget_tokens, 0);
    assert_eq!(config.r#type, ThinkingType::Disabled);
}

#[test]
fn test_thinking_config_default() {
    let config = ThinkingConfig::default();
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
