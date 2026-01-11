#[path = "common.rs"]
mod common;

use std::io::Write;
use std::time::Duration;
use futures_util::StreamExt;
use common::create_test_llm_config_streaming;
use langchainrust::llms::{OpenAIConfig, LLM};

#[tokio::test]
async fn test_streaming_generate() {
    // 创建一个启用流式输出的 LLM 实例
    let llm = LLM::new(create_test_llm_config_streaming());
    // 调用 generate 方法，请求模型从 1 数到 5
    let result = llm.generate("Count from 1 to 5.").await;
    // 检查结果
    match &result {
        Ok(text) => {
            println!("✅ 流式生成成功: {}", text);
            // 验证返回内容不为空
            assert!(!text.is_empty(), "流式响应内容不应为空");
        }
        Err(e) => eprintln!("❌ 流式生成出错: {}", e),
    }

    // 确保调用成功，若失败则打印错误详情
    assert!(result.is_ok(), "流式 LLM 调用失败: {:?}", result.err());
}




#[tokio::test]
async fn test_streaming_LLM() {
    // 创建一个启用流式输出的 LLM 实例
    let llm = LLM::new(create_test_llm_config_streaming());
    // 调用 generate 方法，请求模型从 1 数到 5
    let mut stream = llm.stream_generate("生成500字的春天的诗").await.unwrap();

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                print!("{}", token);
                tokio::time::sleep(Duration::from_millis(50)).await;
                std::io::stdout().flush().unwrap();
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }
    println!(); // 换行
}


use tokio::time;
// cargo test test_streaming_LLM1 -- --nocapture
#[tokio::test]
async fn test_streaming_LLM1() {
    let llm = LLM::new(create_test_llm_config_streaming());
    println!("开始流式输出：");

    let mut stream = llm.stream_generate("生成一段春天的散文3000字").await.unwrap();

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                // 等效于 Python: print(token, end="", flush=True)
                print!("{}", token);
                std::io::stdout().flush().unwrap();

            }
            Err(e) => {
                eprintln!("\nError: {}", e);
                break;
            }
        }
    }

    println!("\n流式输出结束");
}


#[tokio::test]
async fn test_streaming_chat() {
    // Create LLM with streaming enabled
    let config = OpenAIConfig {
        api_key: "".to_string(),
        base_url: "https://api.openai-proxy.org/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: true, // Enable streaming
    };
    let llm = LLM::new(config);

    let result = llm.chat(Some("You are a concise assistant."), "Say 'Hello, streaming!'").await;

    match &result {
        Ok(text) => {
            println!("✅ Streaming chat success: {}", text);
            assert!(!text.is_empty(), "Streaming chat response should not be empty");
        }
        Err(e) => eprintln!("❌ Streaming chat error: {}", e),
    }

    assert!(result.is_ok(), "Streaming chat call failed: {:?}", result.err());
}