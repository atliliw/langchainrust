//! Token-tracking LLM wrapper and cost estimation

use std::pin::Pin;
use std::sync::Arc;

use crate::language_models::{BaseLanguageModel, LLMResult, StreamChunk, TokenUsage};
use crate::observability::{MetricsSink, ObsEvent};
use crate::runnables::Runnable;
use crate::tools::ToolDefinition;
use crate::{BaseChatModel, RunnableConfig};
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lc_schema::Message;
use tokio::sync::Mutex;

use super::counter::{TokenCounter, TrackerTokenUsage};
use super::tiktoken::TiktokenCounter;
use super::TokenCounterError;

/// LLM wrapper with token statistics
///
/// Wraps any `BaseChatModel`, accumulating prompt / completion token usage automatically,
/// preferring the real usage returned by the LLM, falling back to tiktoken estimates.
///
/// v0.20.1: implements `BaseChatModel` itself, so a tracked LLM can be plugged
/// directly into framework agents (e.g. `FunctionCallingAgent::new`); every
/// `chat`/`stream`/`bind_tools` call inside the agent loop is counted, and
/// `get_usage`/`estimate_cost` reflect the cumulative agent-run usage.
pub struct TokenTrackingLLM<L: BaseChatModel> {
    llm: L,
    counter: Arc<dyn TokenCounter>,
    usage: Arc<Mutex<TrackerTokenUsage>>,
    /// Optional observability sink (v0.20.2): exports a `TokenUsage` event after
    /// each call that reports usage. `None` by default — behavior unchanged.
    metrics_sink: Option<Arc<dyn MetricsSink>>,
}

impl<L: BaseChatModel> TokenTrackingLLM<L> {
    /// Wraps an LLM with a custom counter.
    pub fn new(llm: L, counter: Arc<dyn TokenCounter>) -> Self {
        Self {
            llm,
            counter,
            usage: Arc::new(Mutex::new(TrackerTokenUsage::new())),
            metrics_sink: None,
        }
    }

    /// Wraps with a Tiktoken (cl100k_base) counter
    pub fn for_openai(llm: L) -> Result<Self, TokenCounterError> {
        let counter = TiktokenCounter::new()?;
        Ok(Self::new(llm, Arc::new(counter)))
    }

    /// Attaches an observability sink: after each call that reports usage the
    /// wrapper exports a `TokenUsage` event (real when the provider reports it,
    /// otherwise the tiktoken estimate). The sink is shared across wrappers
    /// rebuilt by `bind_tools`/`with_temperature`/`with_max_tokens`.
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics_sink = Some(sink);
        self
    }

    /// Calls the LLM and counts tokens.
    ///
    /// Inherent method (kept for backward compatibility); delegates to
    /// `Self::chat_tracked`. When the model is reached through
    /// `dyn BaseChatModel` (e.g. inside an agent), the trait `chat` takes over —
    /// both count through the same helper, so the two paths never diverge.
    pub async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, L::Error> {
        self.chat_tracked(messages, config).await
    }

    /// Shared counting entry point for the inherent `chat` and the trait `chat`.
    async fn chat_tracked(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, L::Error> {
        let estimated_prompt = self.counter.count_messages(&messages);
        let result = self.llm.chat(messages, config).await?;

        // prefer the real usage returned by the LLM, otherwise use the estimate.
        // `TrackerTokenUsage` and `language_models::TokenUsage` are both usize,
        // so no precision-loss conversion is needed (Q6).
        let (prompt, completion) = result
            .token_usage
            .as_ref()
            .map(|u| (u.prompt_tokens, u.completion_tokens))
            .unwrap_or((
                estimated_prompt as usize,
                self.counter.count_tokens(&result.content) as usize,
            ));

        self.usage.lock().await.add(prompt, completion);

        // v0.20.2: export a TokenUsage event once counted (real or estimate).
        // Failure is only warned — never propagated to the caller.
        if let Some(sink) = &self.metrics_sink {
            let evt = ObsEvent::TokenUsage(TokenUsage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: result
                    .token_usage
                    .as_ref()
                    .map(|u| u.total_tokens)
                    .unwrap_or(prompt + completion),
            });
            if let Err(e) = sink.export(&evt).await {
                log::warn!(target: "lc_core::token_counter", "token usage export failed: {e}");
            }
        }

        Ok(result)
    }

    /// Returns the cumulative usage
    pub async fn get_usage(&self) -> TrackerTokenUsage {
        self.usage.lock().await.clone()
    }

    /// Resets the statistics
    pub async fn reset(&self) {
        self.usage.lock().await.reset();
    }

    /// Estimates the cost (USD)
    pub async fn estimate_cost(&self, pricing: &ModelPricing) -> f64 {
        let usage = self.get_usage().await;
        pricing.calculate(usage.prompt_tokens, usage.completion_tokens)
    }
}

#[async_trait]
impl<L> Runnable<Vec<Message>, LLMResult> for TokenTrackingLLM<L>
where
    L: BaseChatModel + Send + Sync,
{
    type Error = L::Error;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat_tracked(input, config).await
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error>
    {
        // `batch` keeps the default `Runnable` implementation (concurrent via
        // `invoke`, which counts each input).
        // Convert the counted `StreamChunk` stream into an `LLMResult` stream,
        // mirroring `OpenAIChat::stream`.
        let model = self.llm.model_name().to_string();
        let stream = self.stream_chat(input, config).await?;
        let stream = stream.map(move |item| match item {
            Ok(chunk) => Ok(LLMResult {
                content: chunk.text,
                model: model.clone(),
                token_usage: chunk.token_usage,
                tool_calls: chunk.tool_calls,
                thinking_content: None,
            }),
            Err(e) => Err(e),
        });
        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl<L> BaseLanguageModel<Vec<Message>, LLMResult> for TokenTrackingLLM<L>
where
    L: BaseChatModel + Send + Sync,
{
    fn model_name(&self) -> &str {
        self.llm.model_name()
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        self.llm.get_num_tokens(text)
    }

    fn temperature(&self) -> Option<f32> {
        self.llm.temperature()
    }

    fn max_tokens(&self) -> Option<usize> {
        self.llm.max_tokens()
    }

    fn with_temperature(self, temp: f32) -> Self
    where
        Self: Sized,
    {
        // A rebuilt wrapper shares the same counter + usage Arcs, so the count
        // survives parameter overrides.
        Self {
            llm: self.llm.with_temperature(temp),
            counter: self.counter.clone(),
            usage: self.usage.clone(),
            metrics_sink: self.metrics_sink.clone(),
        }
    }

    fn with_max_tokens(self, max: usize) -> Self
    where
        Self: Sized,
    {
        Self {
            llm: self.llm.with_max_tokens(max),
            counter: self.counter.clone(),
            usage: self.usage.clone(),
            metrics_sink: self.metrics_sink.clone(),
        }
    }
}

#[async_trait]
impl<L> BaseChatModel for TokenTrackingLLM<L>
where
    L: BaseChatModel + Send + Sync,
{
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat_tracked(messages, config).await
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        // Forward the stream, counting only the real usage the provider reports
        // (typically on the terminal chunk). Streaming cannot fall back to
        // tiktoken estimates — there is no complete text to count — so the
        // counted usage may be 0 for providers that never report it
        // (v0.20.1 known boundary; estimation fallback moved to 0.21.0).
        let stream = self.llm.stream_chat(messages, config).await?;
        let usage = self.usage.clone();
        let sink = self.metrics_sink.clone();
        let stream = stream.then(move |item| {
            let usage = usage.clone();
            let sink = sink.clone();
            async move {
                if let Ok(chunk) = &item {
                    if let Some(u) = &chunk.token_usage {
                        usage.lock().await.add(u.prompt_tokens, u.completion_tokens);
                        if let Some(sink) = &sink {
                            let evt = ObsEvent::TokenUsage(u.clone());
                            if let Err(e) = sink.export(&evt).await {
                                log::warn!(target: "lc_core::token_counter", "token usage export failed: {e}");
                            }
                        }
                    }
                }
                item
            }
        });
        Ok(Box::pin(stream))
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        // Bind tools on the inner model, then re-wrap the bound model in a
        // `TokenTrackingLLM` sharing the same counter + usage Arcs, so counting
        // survives tool binding. Requires the `Box<dyn BaseChatModel>` glue
        // (S3) so the box itself satisfies `L: BaseChatModel`.
        let bound = self.llm.bind_tools(tools)?;
        Some(Box::new(TokenTrackingLLM {
            llm: bound,
            counter: self.counter.clone(),
            usage: self.usage.clone(),
            metrics_sink: self.metrics_sink.clone(),
        }))
    }
}

/// Model pricing (per 1K tokens, USD)
pub struct ModelPricing {
    /// Per-1K prompt token price (USD)
    pub prompt_price_per_1k: f64,
    /// Per-1K completion token price (USD)
    pub completion_price_per_1k: f64,
}

impl ModelPricing {
    /// Creates custom model pricing.
    pub fn new(prompt: f64, completion: f64) -> Self {
        Self {
            prompt_price_per_1k: prompt,
            completion_price_per_1k: completion,
        }
    }

    /// gpt-4o-mini pricing (USD / 1K tokens)
    pub fn gpt4o_mini() -> Self {
        Self::new(0.15, 0.60)
    }

    /// gpt-4o pricing (USD / 1K tokens)
    pub fn gpt4o() -> Self {
        Self::new(2.50, 10.00)
    }

    /// Calculates the cost
    pub fn calculate(&self, prompt: usize, completion: usize) -> f64 {
        (prompt as f64 / 1000.0) * self.prompt_price_per_1k
            + (completion as f64 / 1000.0) * self.completion_price_per_1k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_pricing_gpt4o_mini() {
        let p = ModelPricing::gpt4o_mini();
        // 1000 prompt * 0.15/1k + 1000 completion * 0.60/1k = 0.75
        let cost = p.calculate(1000, 1000);
        assert!((cost - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_model_pricing_zero() {
        let p = ModelPricing::gpt4o_mini();
        assert_eq!(p.calculate(0, 0), 0.0);
    }

    #[test]
    fn test_model_pricing_custom() {
        let p = ModelPricing::new(1.0, 2.0);
        // 500 * 1.0/1k + 250 * 2.0/1k = 0.5 + 0.5 = 1.0
        let cost = p.calculate(500, 250);
        assert!((cost - 1.0).abs() < 0.001);
    }

    // NOTE: Tests that require OpenAIChat live in the lc-providers crate
    // because lc-core cannot depend on lc-providers (circular dependency).
    // The TokenTrackingLLM integration is tested there instead.

    use crate::language_models::TokenUsage;
    use crate::observability::ObsError;
    use crate::token_counter::CharRatioCounter;

    /// Mock model driving the `TokenTrackingLLM` trait tests (v0.20.1).
    #[derive(Debug, Clone)]
    struct MockChatModel {
        /// Usage reported by `chat`; `None` = provider reports no usage.
        chat_usage: Option<TokenUsage>,
        /// Usage reported on the final streaming chunk; `None` = none in stream.
        stream_usage: Option<TokenUsage>,
        /// Whether `bind_tools` succeeds (tool-capable mock).
        tool_capable: bool,
    }

    #[derive(Debug)]
    struct MockError;

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "mock error")
        }
    }

    impl std::error::Error for MockError {}

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockChatModel {
        type Error = MockError;

        async fn invoke(
            &self,
            input: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.chat(input, config).await
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockChatModel {
        fn model_name(&self) -> &str {
            "mock-model"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len() / 4
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
    impl BaseChatModel for MockChatModel {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: "mock reply".to_string(),
                model: "mock-model".to_string(),
                token_usage: self.chat_usage.clone(),
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
            let chunks = vec![
                Ok(StreamChunk::new("hello")),
                Ok(StreamChunk {
                    text: " world".to_string(),
                    token_usage: self.stream_usage.clone(),
                    tool_calls: None,
                }),
            ];
            Ok(Box::pin(futures_util::stream::iter(chunks)))
        }

        fn bind_tools(
            &self,
            _tools: Vec<ToolDefinition>,
        ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
            self.tool_capable.then(|| {
                Box::new(self.clone()) as Box<dyn BaseChatModel<Error = MockError> + Send + Sync>
            })
        }
    }

    fn tracked_mock(
        chat_usage: Option<TokenUsage>,
        tool_capable: bool,
    ) -> TokenTrackingLLM<MockChatModel> {
        TokenTrackingLLM::new(
            MockChatModel {
                chat_usage,
                stream_usage: None,
                tool_capable,
            },
            Arc::new(CharRatioCounter::new(4)),
        )
    }

    #[tokio::test]
    async fn chat_accumulates_real_usage_across_calls() {
        let tracked = tracked_mock(
            Some(TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 20,
                total_tokens: 120,
            }),
            false,
        );
        let msgs = vec![Message::human("hi")];
        tracked.chat(msgs.clone(), None).await.unwrap();
        tracked.chat(msgs, None).await.unwrap();

        let usage = tracked.get_usage().await;
        assert_eq!(usage.prompt_tokens, 200);
        assert_eq!(usage.completion_tokens, 40);
        assert_eq!(usage.total_tokens, 240);
    }

    #[tokio::test]
    async fn chat_via_dyn_base_chat_model_counts() {
        // The agent path reaches `TokenTrackingLLM` through `dyn BaseChatModel`,
        // so the trait impl (not the inherent method) must count.
        let tracked = tracked_mock(
            Some(TokenUsage {
                prompt_tokens: 7,
                completion_tokens: 3,
                total_tokens: 10,
            }),
            false,
        );
        let model: &dyn BaseChatModel<Error = MockError> = &tracked;
        model.chat(vec![Message::human("hi")], None).await.unwrap();

        let usage = tracked.get_usage().await;
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(usage.completion_tokens, 3);
    }

    #[tokio::test]
    async fn chat_estimates_when_provider_reports_no_usage() {
        let tracked = tracked_mock(None, false);
        tracked
            .chat(vec![Message::human("hello world")], None)
            .await
            .unwrap();

        let usage = tracked.get_usage().await;
        assert!(usage.prompt_tokens > 0, "prompt should be estimated");
        assert!(
            usage.completion_tokens > 0,
            "completion should be estimated"
        );
    }

    #[tokio::test]
    async fn stream_chat_accumulates_real_usage() {
        let llm = MockChatModel {
            chat_usage: None,
            stream_usage: Some(TokenUsage {
                prompt_tokens: 50,
                completion_tokens: 15,
                total_tokens: 65,
            }),
            tool_capable: false,
        };
        let tracked = TokenTrackingLLM::new(llm, Arc::new(CharRatioCounter::new(4)));

        let stream = tracked
            .stream_chat(vec![Message::human("hi")], None)
            .await
            .unwrap();
        let items: Vec<_> = stream.collect().await;
        assert_eq!(items.len(), 2);
        assert!(items[0].is_ok());

        let usage = tracked.get_usage().await;
        assert_eq!(usage.prompt_tokens, 50);
        assert_eq!(usage.completion_tokens, 15);
    }

    #[tokio::test]
    async fn runnable_stream_counts_through_stream_chat() {
        let llm = MockChatModel {
            chat_usage: None,
            stream_usage: Some(TokenUsage {
                prompt_tokens: 5,
                completion_tokens: 5,
                total_tokens: 10,
            }),
            tool_capable: false,
        };
        let tracked = TokenTrackingLLM::new(llm, Arc::new(CharRatioCounter::new(4)));

        let stream = tracked
            .stream(vec![Message::human("hi")], None)
            .await
            .unwrap();
        let items: Vec<_> = stream.collect().await;
        assert_eq!(items.len(), 2);
        assert!(items[0].is_ok());

        let usage = tracked.get_usage().await;
        assert_eq!(usage.total_tokens, 10);
    }

    #[tokio::test]
    async fn bind_tools_keeps_shared_usage() {
        // bind_tools re-wraps the boxed model in a `TokenTrackingLLM` sharing the
        // same usage `Arc`; counting must continue after tools are attached.
        let tracked = tracked_mock(
            Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 4,
                total_tokens: 14,
            }),
            true,
        );

        let bound = tracked
            .bind_tools(vec![ToolDefinition::new(
                "get_weather",
                "Get current weather",
            )])
            .expect("tool-capable mock must bind");
        bound.chat(vec![Message::human("hi")], None).await.unwrap();

        let usage = tracked.get_usage().await;
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 4);
    }

    #[test]
    fn bind_tools_returns_none_when_model_incapable() {
        let tracked = tracked_mock(None, false);
        assert!(tracked
            .bind_tools(vec![ToolDefinition::new("t", "t")])
            .is_none());
    }

    #[test]
    fn base_model_metadata_passthrough() {
        let tracked = tracked_mock(None, false);
        assert_eq!(tracked.model_name(), "mock-model");
        // "hello world" = 11 bytes, / 4 = 2
        assert_eq!(tracked.get_num_tokens("hello world"), 2);
    }

    #[tokio::test]
    async fn with_temperature_preserves_usage() {
        let tracked = tracked_mock(
            Some(TokenUsage {
                prompt_tokens: 3,
                completion_tokens: 1,
                total_tokens: 4,
            }),
            false,
        );
        let tracked = tracked.with_temperature(0.5).with_max_tokens(128);
        tracked
            .chat(vec![Message::human("hi")], None)
            .await
            .unwrap();

        let usage = tracked.get_usage().await;
        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(usage.completion_tokens, 1);
    }

    // --- v0.20.2: observability sink ---------------------------------------

    /// Mock sink recording events in a shared buffer.
    struct MockSink {
        events: Arc<Mutex<Vec<ObsEvent>>>,
        fail: bool,
    }

    #[async_trait]
    impl MetricsSink for MockSink {
        async fn export(&self, event: &ObsEvent) -> Result<(), ObsError> {
            if self.fail {
                return Err(ObsError::Transport("mock failure".to_string()));
            }
            self.events.lock().await.push(event.clone());
            Ok(())
        }
    }

    fn tracked_mock_with_sink(
        tool_capable: bool,
        events: Arc<Mutex<Vec<ObsEvent>>>,
    ) -> (TokenTrackingLLM<MockChatModel>, Arc<Mutex<Vec<ObsEvent>>>) {
        let tracked = tracked_mock(
            Some(TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 20,
                total_tokens: 120,
            }),
            tool_capable,
        )
        .with_metrics_sink(Arc::new(MockSink {
            events: events.clone(),
            fail: false,
        }));
        (tracked, events)
    }

    #[tokio::test]
    async fn with_metrics_sink_exports_token_usage_after_chat() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let (tracked, events) = tracked_mock_with_sink(false, events);

        tracked
            .chat(vec![Message::human("hi")], None)
            .await
            .unwrap();

        let captured = events.lock().await;
        assert_eq!(captured.len(), 1);
        match &captured[0] {
            ObsEvent::TokenUsage(u) => {
                assert_eq!(u.prompt_tokens, 100);
                assert_eq!(u.completion_tokens, 20);
                assert_eq!(u.total_tokens, 120);
            }
            ObsEvent::AgentMetrics(_) => panic!("unexpected event kind"),
        }
    }

    #[tokio::test]
    async fn stream_chat_exports_usage_when_sink_attached() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let llm = MockChatModel {
            chat_usage: None,
            stream_usage: Some(TokenUsage {
                prompt_tokens: 50,
                completion_tokens: 15,
                total_tokens: 65,
            }),
            tool_capable: false,
        };
        let tracked = TokenTrackingLLM::new(llm, Arc::new(CharRatioCounter::new(4)))
            .with_metrics_sink(Arc::new(MockSink {
                events: events.clone(),
                fail: false,
            }));

        let stream = tracked
            .stream_chat(vec![Message::human("hi")], None)
            .await
            .unwrap();
        let items: Vec<_> = stream.collect().await;
        assert_eq!(items.len(), 2);
        assert!(items[0].is_ok());

        let captured = events.lock().await;
        assert_eq!(captured.len(), 1);
        match &captured[0] {
            ObsEvent::TokenUsage(u) => assert_eq!(u.total_tokens, 65),
            ObsEvent::AgentMetrics(_) => panic!("unexpected event kind"),
        }
    }

    #[tokio::test]
    async fn sink_export_failure_is_warned_and_flow_continues() {
        let tracked = tracked_mock(
            Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 4,
                total_tokens: 14,
            }),
            false,
        )
        .with_metrics_sink(Arc::new(MockSink {
            events: Arc::new(Mutex::new(Vec::new())),
            fail: true,
        }));

        let result = tracked.chat(vec![Message::human("hi")], None).await;
        assert!(result.is_ok(), "export failure must not propagate");

        let usage = tracked.get_usage().await;
        assert_eq!(usage.total_tokens, 14);
    }

    #[tokio::test]
    async fn bind_tools_keeps_metrics_sink_attached() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let (tracked, events) = tracked_mock_with_sink(true, events);

        let bound = tracked
            .bind_tools(vec![ToolDefinition::new(
                "get_weather",
                "Get current weather",
            )])
            .expect("tool-capable mock must bind");
        bound.chat(vec![Message::human("hi")], None).await.unwrap();

        let captured = events.lock().await;
        assert_eq!(captured.len(), 1);
        match &captured[0] {
            ObsEvent::TokenUsage(u) => assert_eq!(u.prompt_tokens, 100),
            ObsEvent::AgentMetrics(_) => panic!("unexpected event kind"),
        }
    }
}
