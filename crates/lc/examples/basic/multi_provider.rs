//! 多 LLM Provider 示例
//!
//! 展示如何在 OpenAI / Ollama / DeepSeek 之间切换,
//! 用相同的消息调用不同后端。
//!
//! # 运行
//! ```bash
//! # 默认用 openai
//! cargo run --example basic_multi_provider
//! # 切换到 ollama(需本地运行 Ollama)
//! $env:PROVIDER="ollama"; cargo run --example basic_multi_provider
//! # 切换到 deepseek
//! $env:PROVIDER="deepseek"; cargo run --example basic_multi_provider
//! ```
//!
//! # 环境变量
//! - `PROVIDER`:openai(默认)/ ollama / deepseek
//! - `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`:对应 provider 的密钥

use langchainrust::schema::Message;
use langchainrust::{
    BaseChatModel, DeepSeekChat, DeepSeekConfig, OllamaChat, OllamaConfig, OpenAIChat, OpenAIConfig,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = std::env::var("PROVIDER").unwrap_or_else(|_| "openai".to_string());

    let messages = vec![
        Message::system("你是一个 helpful assistant,一句话回答。"),
        Message::human("用一句话介绍你自己。"),
    ];

    let answer = match provider.as_str() {
        "openai" => {
            let api_key = std::env::var("OPENAI_API_KEY").expect("请设置 OPENAI_API_KEY");
            let base_url = std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
            let llm = OpenAIChat::new(OpenAIConfig {
                api_key,
                base_url,
                model: "gpt-4o-mini".to_string(),
                ..Default::default()
            });
            llm.chat(messages, None).await?.content
        }
        "ollama" => {
            let llm = OllamaChat::with_config(OllamaConfig {
                base_url: std::env::var("OLLAMA_BASE_URL")
                    .unwrap_or_else(|_| "http://localhost:11434/v1".to_string()),
                model: "llama3.2".to_string(),
                ..Default::default()
            });
            llm.chat(messages, None).await?.content
        }
        "deepseek" => {
            let api_key = std::env::var("DEEPSEEK_API_KEY").expect("请设置 DEEPSEEK_API_KEY");
            let llm = DeepSeekChat::new(DeepSeekConfig {
                api_key,
                model: "deepseek-chat".to_string(),
                ..Default::default()
            });
            llm.chat(messages, None).await?.content
        }
        other => return Err(format!("未知 provider: {other}(可选:openai/ollama/deepseek)").into()),
    };

    println!("[{provider}] 回答:\n{answer}");
    Ok(())
}
