//! What: OpenAI-compatible endpoint that returns `tool_calls` on the wire.
//! Run: cargo run -p langchainrust --example openai_server_tools --features openai-server
//! Env: none (deterministic demo backend); optional LC_ADDR

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use langchainrust::agents::{openai_compat_router, ChatBackend};
use langchainrust::core::language_models::{LLMResult, StreamChunk};
use langchainrust::core::tools::ToolCall;
use langchainrust::schema::Message;

/// Deterministic backend: asks for a tool call when the user mentions weather.
struct ToolDemoBackend;

#[async_trait]
impl ChatBackend for ToolDemoBackend {
    async fn complete(&self, messages: Vec<Message>) -> Result<LLMResult, String> {
        let prompt = messages
            .last()
            .map(|message| message.content.to_lowercase())
            .unwrap_or_default();

        if prompt.contains("weather") {
            let call = ToolCall::builder("call_1")
                .name("get_weather")
                .arguments(r#"{"city":"beijing"}"#.to_string())
                .build();
            Ok(LLMResult {
                content: String::new(),
                model: "tool-demo".to_string(),
                token_usage: None,
                tool_calls: Some(vec![call]),
                thinking_content: None,
            })
        } else {
            Ok(LLMResult {
                content: "Say \"weather\" to receive a tool call.".to_string(),
                model: "tool-demo".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    async fn stream_chat(
        &self,
        _messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, String>> + Send>>, String> {
        Ok(Box::pin(futures_util::stream::iter(vec![Ok(
            StreamChunk::new("streaming demo"),
        )])))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = openai_compat_router(Arc::new(ToolDemoBackend), "tool-demo");
    let addr = std::env::var("LC_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    println!("OpenAI-compatible tool demo on http://{addr}");
    println!(
        "curl -s http://{addr}/v1/chat/completions \
         -H 'content-type: application/json' \
         -d '{{\"model\":\"tool-demo\",\"messages\":[{{\"role\":\"user\",\"content\":\"weather?\"}}]}}'"
    );

    axum::serve(listener, app).await?;
    Ok(())
}
