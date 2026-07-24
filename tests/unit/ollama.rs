// tests/unit/ollama.rs
// Ollama 配置和结构的单元测试，不调用实际 API

use langchainrust::language_models::openai::sse::{SSEEvent, SSEParser};
use langchainrust::schema::Message;
use langchainrust::{BaseLanguageModel, OllamaChat, OllamaConfig};

// 测试默认配置值
#[test]
fn test_ollama_config_default() {
    let config = OllamaConfig::default();
    assert_eq!(config.base_url, "http://localhost:11434/v1");
    assert_eq!(config.model, "");
    assert_eq!(config.temperature, None);
    assert_eq!(config.max_tokens, None);
}

// 测试指定模型创建配置
#[test]
fn test_ollama_config_new() {
    let config = OllamaConfig::new("mistral");
    assert_eq!(config.model, "mistral");
    assert_eq!(config.base_url, "http://localhost:11434/v1");
}

// 测试链式配置方法
#[test]
fn test_ollama_config_with_methods() {
    let config = OllamaConfig::new("qwen2.5:7b")
        .with_base_url("http://192.168.1.100:11434/v1")
        .with_temperature(0.5)
        .with_max_tokens(1000);

    assert_eq!(config.base_url, "http://192.168.1.100:11434/v1");
    assert_eq!(config.temperature, Some(0.5));
    assert_eq!(config.max_tokens, Some(1000));
}

// 测试直接创建 OllamaChat
#[test]
fn test_ollama_chat_new() {
    let chat = OllamaChat::new("qwen2.5:7b");
    assert_eq!(chat.model_name(), "qwen2.5:7b");
}

// 测试用配置创建 OllamaChat
#[test]
fn test_ollama_chat_with_config() {
    let config = OllamaConfig::new("qwen2.5:7b").with_temperature(0.7);
    let chat = OllamaChat::with_config(config);
    assert_eq!(chat.model_name(), "qwen2.5:7b");
    assert_eq!(chat.temperature(), Some(0.7));
}

// 测试 BaseLanguageModel trait 实现
#[test]
fn test_ollama_base_language_model_traits() {
    let chat = OllamaChat::new("qwen2.5:7b");
    assert_eq!(chat.model_name(), "qwen2.5:7b");

    let token_count = chat.get_num_tokens("hello world");
    assert_eq!(token_count, 2);

    let chat_with_temp = chat.with_temperature(0.8);
    assert_eq!(chat_with_temp.temperature(), Some(0.8));

    let chat_with_max = chat_with_temp.with_max_tokens(500);
    assert_eq!(chat_with_max.max_tokens(), Some(500));
}

// 测试消息构建，不调用 API
#[test]
fn test_ollama_message_format() {
    let messages = [
        Message::system("You are a helpful assistant"),
        Message::human("Hello"),
    ];

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content, "You are a helpful assistant");
    assert_eq!(messages[1].content, "Hello");
}

// 测试 SSEParser 解析 Ollama 格式数据
#[test]
fn test_ollama_sse_parser_basic() {
    let mut parser = SSEParser::new();

    let chunk = r#"data: {"id":"chat-123","object":"chat.completion.chunk","created":123,"model":"qwen2.5:7b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}


"#;
    let events = parser.parse(chunk);

    assert_eq!(events.len(), 1);
    assert!(!events[0].is_done());
}

// 测试 SSEParser 解析 Ollama 结束事件
#[test]
fn test_ollama_sse_done_event() {
    let mut parser = SSEParser::new();

    let chunk = "data: [DONE]\n\n";
    let events = parser.parse(chunk);

    assert_eq!(events.len(), 1);
    assert!(events[0].is_done());
}

// 测试 SSEParser 解析多个事件
#[test]
fn test_ollama_sse_multiple_events() {
    let mut parser = SSEParser::new();

    let chunk = r#"data: {"choices":[{"delta":{"content":"Hi"}}]}


data: {"choices":[{"delta":{"content":"!"}}]}


"#;
    let events = parser.parse(chunk);

    assert_eq!(events.len(), 2);
}

// 测试 SSEParser buffer 处理跨 chunk 数据
#[test]
fn test_ollama_sse_buffer_incomplete_chunk() {
    let mut parser = SSEParser::new();

    let chunk1 = r#"data: {"choices":"#;
    let events1 = parser.parse(chunk1);
    assert_eq!(events1.len(), 0);

    let chunk2 = r#"[{"delta":{"content":"Hello"}}]}


"#;
    let events2 = parser.parse(chunk2);
    assert_eq!(events2.len(), 1);
}

// 测试 SSEEvent 解析 Ollama 流式 chunk
#[test]
fn test_ollama_sse_parse_chunk_content() {
    let event = SSEEvent {
        event: None,
        data: r#"{"id":"chat-123","object":"chat.completion.chunk","created":123,"model":"qwen2.5:7b","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}"#.to_string(),
    };

    let chunk = event.parse_openai_chunk().unwrap().unwrap();
    assert_eq!(chunk.choices[0].delta.content, Some("你好".to_string()));
    assert_eq!(chunk.model, "qwen2.5:7b");
}

// 测试 SSEEvent 解析空 content
#[test]
fn test_ollama_sse_parse_chunk_empty_content() {
    let event = SSEEvent {
        event: None,
        data: r#"{"id":"chat-123","object":"chat.completion.chunk","created":123,"model":"qwen2.5:7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#.to_string(),
    };

    let chunk = event.parse_openai_chunk().unwrap().unwrap();
    assert_eq!(chunk.choices[0].delta.content, None);
    assert_eq!(chunk.choices[0].delta.role, Some("assistant".to_string()));
}
