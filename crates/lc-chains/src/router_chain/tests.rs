// lc-chains/src/router_chain/tests.rs
//! Unit tests for the router chain.

use super::*;
use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{LLMResult, StreamChunk};
use lc_core::runnables::RunnableConfig;
use lc_core::tools::ToolDefinition;
use lc_core::BaseChatModel;
use lc_core::{BaseLanguageModel, Runnable};
use lc_providers::ProviderError;
use lc_schema::Message;
use serde_json::json;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use crate::base::{BaseChain, ChainError, ChainResult};

/// Mock chat model that returns a canned `LLMResult`. `supports_tools`
/// controls whether `bind_tools` yields a bound copy (tool-call path) or
/// `None` (plain-text fallback path) — so both P2-5 branches are testable.
#[derive(Clone)]
struct MockRouterLLM {
    response: LLMResult,
    supports_tools: bool,
}

impl MockRouterLLM {
    fn tools(response: LLMResult) -> Self {
        Self {
            response,
            supports_tools: true,
        }
    }
    fn plain(response: LLMResult) -> Self {
        Self {
            response,
            supports_tools: false,
        }
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for MockRouterLLM {
    type Error = ProviderError;
    async fn invoke(
        &self,
        _input: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        Ok(self.response.clone())
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for MockRouterLLM {
    fn model_name(&self) -> &str {
        "mock"
    }
    fn get_num_tokens(&self, t: &str) -> usize {
        t.len()
    }
    fn with_temperature(self, _: f32) -> Self {
        self
    }
    fn with_max_tokens(self, _: usize) -> Self {
        self
    }
}

#[async_trait]
impl BaseChatModel for MockRouterLLM {
    async fn chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        Ok(self.response.clone())
    }
    async fn stream_chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        let tokens = [Ok(StreamChunk::new(self.response.content.clone()))];
        Ok(Box::pin(futures_util::stream::iter(tokens)))
    }
    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        if self.supports_tools {
            Some(Box::new(self.clone()))
        } else {
            None
        }
    }
}

/// Simple destination chain that echoes the input under `output`.
struct EchoChain;

#[async_trait]
impl BaseChain for EchoChain {
    fn input_keys(&self) -> Vec<&str> {
        vec!["input"]
    }
    fn output_keys(&self) -> Vec<&str> {
        vec!["output"]
    }
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let mut result = HashMap::new();
        if let Some(v) = inputs.get("input") {
            result.insert("output".to_string(), v.clone());
        }
        Ok(result)
    }
}

fn router_with(llm: MockRouterLLM) -> LLMRouterChain {
    LLMRouterChain::new(llm)
        .add_route(
            "math",
            "handles mathematical questions",
            Arc::new(EchoChain),
        )
        .add_route(
            "science",
            "handles scientific questions",
            Arc::new(EchoChain),
        )
}

fn decision_response(destination: &str, reason: &str) -> LLMResult {
    LLMResult {
        content: String::new(),
        model: "mock".to_string(),
        token_usage: None,
        tool_calls: Some(vec![lc_core::tools::ToolCall::builder("call_1")
            .name("route_to_destination")
            .arguments(format!(
                r#"{{"destination": "{}", "reason": "{}"}}"#,
                destination, reason
            ))
            .build()]),
        thinking_content: None,
    }
}

fn text_response(content: &str) -> LLMResult {
    LLMResult {
        content: content.to_string(),
        model: "mock".to_string(),
        token_usage: None,
        tool_calls: None,
        thinking_content: None,
    }
}

/// P2-5: the routing LLM's `route_to_destination` tool-call arguments (the
/// structured `{destination, reason}` object) drive the route.
#[tokio::test]
async fn test_llm_router_routes_via_tool_call() {
    let chain = router_with(MockRouterLLM::tools(decision_response(
        "math",
        "user asked a calculation",
    )));
    let inputs = HashMap::from([("input".to_string(), json!("what is 2 + 2?"))]);

    let result = chain.invoke(inputs).await.unwrap();
    assert_eq!(
        result.get("output").unwrap(),
        &json!("what is 2 + 2?"),
        "routed to the math destination"
    );
}

/// P2-5: a provider without tool binding falls back to the same call's
/// text, parsed as a JSON `{destination, reason}` object.
#[tokio::test]
async fn test_llm_router_json_text_fallback() {
    let chain = router_with(MockRouterLLM::plain(text_response(
        r#"{"destination": "science", "reason": "explains a phenomenon"}"#,
    )));
    let inputs = HashMap::from([("input".to_string(), json!("why is the sky blue?"))]);

    let result = chain.invoke(inputs).await.unwrap();
    assert_eq!(
        result.get("output").unwrap(),
        &json!("why is the sky blue?"),
        "routed to the science destination"
    );
}

/// P2-5: a bare destination name still works (lenient text fallback).
#[tokio::test]
async fn test_llm_router_bare_name_fallback() {
    let chain = router_with(MockRouterLLM::plain(text_response("math")));
    let inputs = HashMap::from([("input".to_string(), json!("calculate something"))]);

    let result = chain.invoke(inputs).await.unwrap();
    assert_eq!(result.get("output").unwrap(), &json!("calculate something"));
}

/// P2-5: an unknown destination from the LLM does not abort routing — the
/// real routing diagnostics stay in the error (P1-6).
#[tokio::test]
async fn test_llm_router_unknown_destination_reports_diagnostics() {
    let chain = router_with(MockRouterLLM::tools(decision_response(
        "physics", "closest",
    )));
    let inputs = HashMap::from([("input".to_string(), json!("what is 2+2"))]);

    let err = match chain.invoke(inputs).await {
        Ok(_) => panic!("expected a routing failure"),
        Err(e) => e,
    };
    let msg = format!("{err:?}");
    assert!(
        msg.contains("unknown route destination"),
        "expected the LLM diagnostics in the error, got: {msg}"
    );
}

/// P2-5: when the LLM names a nonexistent destination but a keyword
/// matches, the keyword fallback still lands a destination.
#[tokio::test]
async fn test_llm_router_unknown_destination_keyword_still_routes() {
    let chain = LLMRouterChain::new(MockRouterLLM::tools(decision_response(
        "physics", "nonsense",
    )))
    .add_route_with_keywords(
        "math",
        "handles mathematical questions",
        Arc::new(EchoChain),
        vec!["calculate", "2+2"],
    )
    .add_route(
        "science",
        "handles scientific questions",
        Arc::new(EchoChain),
    );

    let inputs = HashMap::from([("input".to_string(), json!("please calculate 2+2"))]);
    let result = chain.invoke(inputs).await.unwrap();
    assert_eq!(
        result.get("output").unwrap(),
        &json!("please calculate 2+2")
    );
}

#[test]
fn test_route_decision_from_text_json() {
    let d = RouteDecision::from_text(r#"{"destination": "math", "reason": "why"}"#);
    assert_eq!(d.destination, "math");
    assert_eq!(d.reason.as_deref(), Some("why"));
}

#[test]
fn test_route_decision_from_text_bare_name() {
    let d = RouteDecision::from_text("  science  ");
    assert_eq!(d.destination, "science");
    assert!(d.reason.is_none());
}

#[test]
fn test_route_tool_schema() {
    let tool = route_tool();
    assert_eq!(tool.function.name, "route_to_destination");
    assert!(tool.function.parameters.is_some());
}
