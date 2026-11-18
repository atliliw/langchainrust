//! `lc serve` — serve an OpenAI-compatible endpoint backed by a model or an agent.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use axum::Router;
use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use futures_util::stream::once;
use futures_util::Stream;
use std::collections::HashMap;

use langchainrust::agents::{openai_compat_router, ChatBackend, ModelBackend};
use langchainrust::chains::{BaseChain, LLMChain};
use langchainrust::core::language_models::{LLMResult, StreamChunk};
use langchainrust::mcp::MCPServer;
use langchainrust::prelude::*;
use langchainrust::schema::{Message, MessageType};
use langchainrust::A2AServer;
use serde_json::Value;

/// Adapts an [`AgentExecutor`] into a [`ChatBackend`] so an agent can be served
/// behind `/v1/chat/completions`. The agent's answer becomes the completion.
struct AgentBackend {
    executor: Arc<AgentExecutor>,
}

#[async_trait]
impl ChatBackend for AgentBackend {
    async fn complete(&self, messages: Vec<Message>) -> Result<LLMResult, String> {
        let input = last_human_text(&messages);
        let text = self.executor.invoke(input).await.map_err(|e| e.to_string())?;
        Ok(LLMResult {
            content: text,
            model: "langchainrust-agent".to_string(),
            token_usage: None,
            tool_calls: None,
            thinking_content: None,
        })
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamChunk, String>> + Send>>,
        String,
    > {
        let text = self.complete(messages).await?.content;
        let chunk = StreamChunk {
            text,
            thinking_content: None,
            token_usage: None,
            tool_calls: None,
        };
        Ok(Box::pin(once(async move { Ok(chunk) })))
    }
}

/// Last human message's text, or empty string.
fn last_human_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.message_type == MessageType::Human)
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

/// Adapts an [`LLMChain`] into a [`ChatBackend`] so a chain can be served.
struct ChainBackend<C> {
    chain: Arc<C>,
}

#[async_trait]
impl<C> ChatBackend for ChainBackend<C>
where
    C: BaseChain + Send + Sync + 'static,
{
    async fn complete(&self, messages: Vec<Message>) -> Result<LLMResult, String> {
        let input = last_human_text(&messages);
        let mut inputs: HashMap<String, Value> = HashMap::new();
        inputs.insert("question".to_string(), Value::String(input));
        let outputs = self.chain.invoke(inputs).await.map_err(|e| e.to_string())?;
        let text = outputs
            .get("output")
            .and_then(Value::as_str)
            .or_else(|| outputs.get("result").and_then(Value::as_str))
            .or_else(|| outputs.values().find_map(Value::as_str))
            .unwrap_or_default()
            .to_string();
        Ok(LLMResult {
            content: text,
            model: "langchainrust-chain".to_string(),
            token_usage: None,
            tool_calls: None,
            thinking_content: None,
        })
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamChunk, String>> + Send>>,
        String,
    > {
        let text = self.complete(messages).await?.content;
        let chunk = StreamChunk {
            text,
            thinking_content: None,
            token_usage: None,
            tool_calls: None,
        };
        Ok(Box::pin(once(async move { Ok(chunk) })))
    }
}

/// Optional Bearer-token gate on the whole router.
async fn bearer_auth(State(token): State<String>, req: Request, next: Next) -> Response {
    let authorized = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| v == format!("Bearer {token}"))
        .unwrap_or(false);
    if authorized {
        next.run(req).await
    } else {
        StatusCode::UNAUTHORIZED.into_response()
    }
}

pub async fn run(
    port: u16,
    model: Option<&str>,
    backend: &str,
    auth: Option<&str>,
    mcp_port: Option<u16>,
    a2a_port: Option<u16>,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_id = model
        .map(str::to_string)
        .unwrap_or_else(|| std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into()));

    let mut app: Router = match backend {
        "agent" => {
            let executor = Arc::new(
                AgentBuilder::new()
                    .llm(OpenAIChat::new(OpenAIConfig::from_env_result()?))
                    .system("You are a concise assistant.")
                    .tool(Calculator::new())
                    .max_iterations(5)
                    .build_executor()?,
            );
            openai_compat_router(Arc::new(AgentBackend { executor }), model_id)
        }
        "chain" => {
            let llm = OpenAIChat::new(OpenAIConfig::from_env_result()?);
            let chain = Arc::new(LLMChain::new(
                llm,
                "Answer concisely in one sentence:\n{question}",
            ));
            openai_compat_router(Arc::new(ChainBackend { chain }), model_id)
        }
        _ => {
            let llm = Arc::new(OpenAIChat::new(OpenAIConfig::from_env_result()?));
            openai_compat_router(Arc::new(ModelBackend::new(llm)), model_id)
        }
    };

    if let Some(token) = auth {
        app = app.layer(middleware::from_fn_with_state(
            token.to_string(),
            bearer_auth,
        ));
    }

    // Optional sidecar: MCP stateless HTTP server on a separate port.
    if let Some(mcp_port) = mcp_port {
        let mcp = Arc::new(
            MCPServer::new()
                .with_server_info("lc-cli", env!("CARGO_PKG_VERSION"))
                .with_tool(Arc::new(Calculator::new())),
        );
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", mcp_port)).await?;
        let url = mcp.serve_http(listener);
        println!("MCP stateless server: {url}");
    }

    // Optional sidecar: A2A agent server on a separate port (LLMChain skill).
    if let Some(a2a_port) = a2a_port {
        let llm = OpenAIChat::new(OpenAIConfig::from_env_result()?);
        let chain = Arc::new(LLMChain::new(
            llm,
            "Answer concisely in one sentence:\n{question}",
        ));
        let a2a = A2AServer::new(chain as Arc<dyn BaseChain>);
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", a2a_port)).await?;
        tokio::spawn(async move {
            let _ = a2a.serve_on(listener).await;
        });
        println!("A2A agent server: http://127.0.0.1:{a2a_port}");
    }

    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("OpenAI-compatible endpoint: http://{addr}/v1");
    println!("  GET  /v1/models");
    println!("  POST /v1/chat/completions");
    if auth.is_some() {
        println!("  auth: Bearer token required");
    }
    axum::serve(listener, app).await?;
    Ok(())
}