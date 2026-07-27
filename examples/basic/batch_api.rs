//! Batch API 示例
//!
//! 展示 BatchClient 的批量推理流程:提交 → 轮询 → 取结果。
//! 适合离线评估、批量翻译/摘要,成本降 50%。
//!
//! # 运行
//! ```bash
//! cargo run --example basic_batch_api
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)
//! - `OPENAI_BASE_URL`:API 基址(可选)

use langchainrust::core::batch::{BatchClient, BatchProvider, BatchRequest};
use langchainrust::Message;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("请设置 OPENAI_API_KEY 环境变量");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    // 1. 创建 Batch 客户端
    let client = BatchClient::new(BatchProvider::OpenAI, &api_key).with_base_url(&base_url);

    // 2. 准备批量请求
    let requests = vec![
        BatchRequest {
            custom_id: "translate-1".into(),
            messages: vec![Message::human("将以下英文翻译为中文: Hello, World!")],
            model: "gpt-4o-mini".into(),
            temperature: Some(0.3),
            max_tokens: None,
        },
        BatchRequest {
            custom_id: "translate-2".into(),
            messages: vec![Message::human("将以下英文翻译为中文: Rust is awesome!")],
            model: "gpt-4o-mini".into(),
            temperature: Some(0.3),
            max_tokens: None,
        },
        BatchRequest {
            custom_id: "summarize-1".into(),
            messages: vec![Message::human(
                "用一句话总结: Rust 是一门系统编程语言,注重内存安全和并发性能。",
            )],
            model: "gpt-4o-mini".into(),
            temperature: Some(0.3),
            max_tokens: None,
        },
    ];

    println!("提交 {} 个批量请求...", requests.len());

    // 3. 提交并等待结果（自动轮询,每 5 秒检查一次,最多等 5 分钟）
    let results = client.submit_and_wait(requests, 5_000, 300_000).await?;

    // 4. 输出结果
    println!("\n=== 批量结果 ===");
    for result in &results {
        match &result.result {
            Ok(llm_result) => {
                println!("[{}] {}", result.custom_id, llm_result.content);
            }
            Err(e) => {
                println!("[{}] 错误: {}", result.custom_id, e);
            }
        }
    }

    Ok(())
}
