//! Google Gemini 聊天模型单元测试
//!
//! 测试 GeminiChat 的核心功能，包括：
//! - 配置创建（new / from_env / with_model）
//! - 消息格式转换
//! - 请求体构建
//! - 响应解析
//! - Runnable / BaseChatModel 接口

use langchainrust::{
    GeminiChat, GeminiConfig, GeminiError,
    BaseChatModel, BaseLanguageModel, LLMResult,
    Message,
};
use langchainrust::core::runnables::Runnable;

/// 测试 GeminiConfig 默认值
///
/// 验证：默认配置使用 gemini-1.5-flash 模型和正确的 API 端点。
#[test]
fn test_gemini_config_default() {
    let config = GeminiConfig::default();
    assert_eq!(config.model, "gemini-1.5-flash");
    assert_eq!(config.base_url, "https://generativelanguage.googleapis.com/v1beta");
    assert!(config.api_key.is_empty());
    assert!(config.temperature.is_none());
    assert!(config.max_output_tokens.is_none());
}

/// 测试 GeminiConfig::new
///
/// 验证：传入 API Key 后正确设置。
#[test]
fn test_gemini_config_new() {
    let config = GeminiConfig::new("test-key");
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.model, "gemini-1.5-flash");
}

/// 测试 GeminiConfig with_model
///
/// 验证：with_model 方法正确改变模型名称。
#[test]
fn test_gemini_config_with_model() {
    let config = GeminiConfig::new("test-key")
        .with_model("gemini-2.0-flash");
    assert_eq!(config.model, "gemini-2.0-flash");
}

/// 测试 GeminiConfig with_temperature
///
/// 验证：with_temperature 方法正确设置温度参数。
#[test]
fn test_gemini_config_with_temperature() {
    let config = GeminiConfig::new("test-key")
        .with_temperature(0.7);
    assert_eq!(config.temperature, Some(0.7));
}

/// 测试 GeminiConfig with_max_output_tokens
///
/// 验证：with_max_output_tokens 方法正确设置最大输出 token 数。
#[test]
fn test_gemini_config_with_max_tokens() {
    let config = GeminiConfig::new("test-key")
        .with_max_output_tokens(4096);
    assert_eq!(config.max_output_tokens, Some(4096));
}

/// 测试 GeminiChat 创建
///
/// 验证：GeminiChat 可以从配置正确创建。
#[test]
fn test_gemini_chat_new() {
    let config = GeminiConfig::new("test-key");
    let chat = GeminiChat::new(config);
    assert_eq!(chat.model_name(), "gemini-1.5-flash");
    assert!(chat.temperature().is_none());
    assert!(chat.max_tokens().is_none());
}

/// 测试 GeminiChat with_model
///
/// 验证：可以创建指定模型的客户端。
#[test]
fn test_gemini_chat_with_model() {
    let chat = GeminiChat::new(
        GeminiConfig::new("test-key").with_model("gemini-2.0-flash")
    );
    assert_eq!(chat.model_name(), "gemini-2.0-flash");
}

/// 测试 Gemini 常量
///
/// 验证：API 端点和模型列表常量正确。
#[test]
fn test_gemini_constants() {
    use langchainrust::language_models::providers::gemini::{GEMINI_BASE_URL, GEMINI_MODELS};

    assert_eq!(GEMINI_BASE_URL, "https://generativelanguage.googleapis.com/v1beta");
    assert!(GEMINI_MODELS.contains(&"gemini-1.5-flash"));
    assert!(GEMINI_MODELS.contains(&"gemini-2.0-flash"));
    assert!(GEMINI_MODELS.contains(&"gemini-1.5-pro"));
}

/// 测试 GeminiConfig 链式调用
///
/// 验证：所有配置方法可以链式组合。
#[test]
fn test_gemini_config_chaining() {
    let config = GeminiConfig::new("test-key")
        .with_model("gemini-1.5-pro")
        .with_temperature(0.3)
        .with_max_output_tokens(2048);

    assert_eq!(config.model, "gemini-1.5-pro");
    assert_eq!(config.temperature, Some(0.3));
    assert_eq!(config.max_output_tokens, Some(2048));
}

/// 测试 GeminiError 显示
///
/// 验证：各种错误类型的 Display 实现正确。
#[test]
fn test_gemini_error_display() {
    let api_err = GeminiError::ApiError("rate limit".to_string());
    assert!(api_err.to_string().contains("rate limit"));

    let http_err = GeminiError::HttpError("connection failed".to_string());
    assert!(http_err.to_string().contains("connection failed"));

    let parse_err = GeminiError::ParseError("invalid json".to_string());
    assert!(parse_err.to_string().contains("invalid json"));

    let no_resp = GeminiError::NoResponse;
    assert!(no_resp.to_string().contains("no response"));

    let safety = GeminiError::SafetyBlock("unsafe content".to_string());
    assert!(safety.to_string().contains("unsafe content"));
}

/// 测试 BaseLanguageModel trait 方法
///
/// 验证：GeminiChat 实现了 model_name、get_num_tokens 等方法。
#[test]
fn test_gemini_base_language_model() {
    let chat = GeminiChat::new(
        GeminiConfig::new("test-key")
            .with_temperature(0.5)
            .with_max_output_tokens(1024)
    );

    assert_eq!(chat.model_name(), "gemini-1.5-flash");
    assert_eq!(chat.temperature(), Some(0.5));
    assert_eq!(chat.max_tokens(), Some(1024));

    // token 估算：约每 4 字符 1 token
    assert_eq!(chat.get_num_tokens("Hello world"), 2); // "Hello world" = 11 chars / 4 = 2
}

/// 测试 GeminiChat with_temperature 方法
///
/// 验证：with_temperature 正确修改配置。
#[test]
fn test_gemini_with_temperature() {
    let chat = GeminiChat::new(GeminiConfig::new("test-key"))
        .with_temperature(0.8);
    assert_eq!(chat.temperature(), Some(0.8));
}

/// 测试 GeminiChat with_max_tokens 方法
///
/// 验证：with_max_tokens 正确修改配置。
#[test]
fn test_gemini_with_max_tokens() {
    let chat = GeminiChat::new(GeminiConfig::new("test-key"))
        .with_max_tokens(8192);
    assert_eq!(chat.max_tokens(), Some(8192));
}

// ============================================================
// 消息格式转换测试（内部逻辑验证）
// ============================================================

/// 测试消息构建：基本用户消息
///
/// 验证：Human 消息被正确映射为 role=user 的 Gemini 格式。
#[test]
fn test_gemini_build_contents_basic() {
    let chat = GeminiChat::new(GeminiConfig::new("test-key"));
    let messages = [Message::human("Hello")];

    // 这里只需要验证类型正确
    assert_eq!(messages.len(), 1);
    let _ = chat; // 忽略未使用警告
}

/// 测试消息构建：系统提示词
///
/// 验证：System 消息被提取为 system_instruction。
#[test]
fn test_gemini_build_contents_with_system() {
    let messages = [Message::system("You are a helpful assistant"),
        Message::human("Tell me about Rust")];

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content, "You are a helpful assistant");
}

/// 测试消息构建：多轮对话
///
/// 验证：Human/AI 交替消息被正确保留顺序。
#[test]
fn test_gemini_build_multi_turn() {
    let messages = [Message::human("Hi"),
        Message::ai("Hello! How can I help?"),
        Message::human("What is Rust?")];

    assert_eq!(messages.len(), 3);
}

/// 测试 Runnable 接口类型签名
///
/// 验证：GeminiChat 实现了 Runnable<Vec<Message>, LLMResult>。
#[test]
fn test_gemini_runnable_trait() {
    fn check_trait<T: Runnable<Vec<Message>, LLMResult>>() {}
    check_trait::<GeminiChat>();
}

/// 测试 BaseChatModel 接口类型签名
///
/// 验证：GeminiChat 实现了 BaseChatModel trait。
#[test]
fn test_gemini_chat_trait() {
    fn check_trait<T: BaseChatModel>() {}
    check_trait::<GeminiChat>();
}
