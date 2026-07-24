// tests/integration/ollama_chat.rs
// Ollama 实际 API 调用测试，需要启动 Ollama 服务

use langchainrust::schema::Message;
use langchainrust::{BaseChatModel, OllamaChat, OllamaConfig};

// 测试基本聊天功能，需要 Ollama 服务运行
// 运行前确保: ollama serve
#[tokio::test]
#[ignore = "需要启动 Ollama 服务: ollama serve"]
async fn test_ollama_chat_basic() {
    let llm = OllamaChat::new("qwen2.5:7b");

    let messages = vec![Message::human("什么是rust")];

    let result = llm.chat(messages, None).await;

    let response = match result {
        Ok(r) => r,
        Err(err) => {
            if err.to_string().contains("Connection refused")
                || err.to_string().contains("HTTP 404")
            {
                println!("Ollama service not available, skipping test");
                return;
            }
            panic!("Ollama chat failed: {}", err);
        }
    };
    assert!(!response.content.is_empty());
    println!("Response: {}", response.content);
}

// 测试多轮对话，需要 Ollama 服务运行
#[tokio::test]
#[ignore = "需要启动 Ollama 服务: ollama serve"]
async fn test_ollama_chat_multi_turn() {
    let llm = OllamaChat::new("qwen2.5:7b");

    let messages = vec![
        Message::system("You are a helpful assistant. Answer briefly."),
        Message::human("What is 2+2?"),
    ];

    let result = llm.chat(messages, None).await;

    let response = match result {
        Ok(r) => r,
        Err(err) => {
            if err.to_string().contains("Connection refused")
                || err.to_string().contains("HTTP 404")
            {
                println!("Ollama service not available, skipping test");
                return;
            }
            panic!("Ollama chat failed: {}", err);
        }
    };
    assert!(!response.content.is_empty());
    println!("Response: {}", response.content);
}

// 测试自定义配置，需要 Ollama 服务运行
#[tokio::test]
#[ignore = "需要启动 Ollama 服务: ollama serve"]
async fn test_ollama_chat_with_custom_config() {
    let config = OllamaConfig::new("qwen2.5:7b")
        .with_temperature(0.1)
        .with_max_tokens(50);

    let llm = OllamaChat::with_config(config);

    let messages = vec![Message::human("Count from 1 to 5")];

    let result = llm.chat(messages, None).await;

    let response = match result {
        Ok(r) => r,
        Err(err) => {
            if err.to_string().contains("Connection refused")
                || err.to_string().contains("HTTP 404")
            {
                println!("Ollama service not available, skipping test");
                return;
            }
            panic!("Ollama chat failed: {}", err);
        }
    };
    assert!(!response.content.is_empty());
    println!("Response: {}", response.content);
    println!("Token usage: {:?}", response.token_usage);
}

// 测试流式输出，需要 Ollama 服务运行
#[tokio::test]
#[ignore = "需要启动 Ollama 服务: ollama serve"]
async fn test_ollama_stream_chat() {
    use futures_util::StreamExt;

    let llm = OllamaChat::new("qwen2.5:7b");

    let messages = vec![Message::human("Say hello")];

    let result = llm.stream_chat(messages, None).await;

    match result {
        Ok(stream) => {
            let tokens: Vec<String> = stream
                .take(10)
                .filter_map(|t| async move { t.ok() })
                .collect()
                .await;

            assert!(!tokens.is_empty());
            println!("Streamed tokens: {:?}", tokens);
        }
        Err(e) => {
            if e.to_string().contains("Connection refused") || e.to_string().contains("HTTP 404") {
                println!("Ollama service not available, skipping test");
                return;
            }
            panic!("Ollama stream failed: {}", e);
        }
    }
}
