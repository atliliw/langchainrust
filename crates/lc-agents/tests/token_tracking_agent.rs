//! Integration test: `TokenTrackingLLM` plugged into `FunctionCallingAgent`
//! (v0.20.1).
//!
//! Proves the "framework counting + framework agent" path end to end: a tracked
//! LLM is passed directly into the agent, the agent loop's `chat` calls
//! accumulate into the shared usage `Arc`, and afterwards the cumulative usage
//! (`get_usage`) and cost (`estimate_cost`) are readable from the same tracked
//! handle — plus the agent's own `last_token_usage` is populated.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use lc_agents::types::AgentOutput;
use lc_agents::{BaseAgent, FunctionCallingAgent};
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::Runnable;
use lc_core::token_counter::{CharRatioCounter, ModelPricing, TokenTrackingLLM};
use lc_core::tools::ToolDefinition;
use lc_core::RunnableConfig;
use lc_providers::ProviderError;
use lc_schema::Message;

/// Mock provider reporting a fixed token usage on every call.
///
/// Uses `ProviderError` directly as its error type so `TokenTrackingLLM<Mock>` is
/// `BaseChatModel<Error = ProviderError>` and coerces to the agent's unified
/// trait-object type without a wrapper.
#[derive(Debug, Clone)]
struct MockProvider {
    usage: TokenUsage,
    content: String,
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for MockProvider {
    type Error = ProviderError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, config).await
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for MockProvider {
    fn model_name(&self) -> &str {
        "mock-provider"
    }

    fn get_num_tokens(&self, _text: &str) -> usize {
        0
    }

    fn with_temperature(self, _temp: f32) -> Self
    where
        Self: Sized,
    {
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self
    where
        Self: Sized,
    {
        self
    }
}

#[async_trait]
impl BaseChatModel for MockProvider {
    async fn chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        Ok(LLMResult {
            content: self.content.clone(),
            model: "mock-provider".to_string(),
            token_usage: Some(self.usage.clone()),
            tool_calls: None,
            thinking_content: None,
        })
    }

    async fn stream_chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        let chunk = StreamChunk {
            text: self.content.clone(),
            token_usage: Some(self.usage.clone()),
            tool_calls: None,
        };
        Ok(Box::pin(futures_util::stream::iter(vec![Ok(chunk)])))
    }

    fn bind_tools(
        &self,
        _tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        Some(Box::new(self.clone()))
    }
}

#[tokio::test]
async fn tracked_llm_plugs_into_function_calling_agent_and_counts() {
    let mock = MockProvider {
        usage: TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 25,
            total_tokens: 125,
        },
        content: "final answer".to_string(),
    };
    let tracked = TokenTrackingLLM::new(mock, Arc::new(CharRatioCounter::new(4)));

    // Keep a concrete handle (for `get_usage`/`estimate_cost`), and coerce a
    // clone to the unified trait-object type the agent expects.
    let concrete: Arc<TokenTrackingLLM<MockProvider>> = Arc::new(tracked);
    let llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync> = concrete.clone();
    let agent = FunctionCallingAgent::from_arc(llm, vec![], None);

    let output = agent
        .plan(
            &[],
            &HashMap::from([("input".to_string(), "hi".to_string())]),
        )
        .await
        .expect("plan should succeed");
    assert!(
        matches!(output, AgentOutput::Finish(_)),
        "mock returns text, so the agent should finish, got {output:?}"
    );

    // The agent loop called `chat` through the tracked LLM, so the shared usage
    // `Arc` now reflects the agent-run tokens.
    let usage = concrete.get_usage().await;
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 25);
    assert_eq!(usage.total_tokens, 125);

    // Cost estimation works off the cumulative usage.
    let cost = concrete.estimate_cost(&ModelPricing::gpt4o_mini()).await;
    assert!(cost > 0.0, "cumulative usage must price above zero");

    // The agent's own last-call usage is populated too.
    let last = agent.last_token_usage().expect("agent records last usage");
    assert_eq!(last.prompt_tokens, 100);
}
