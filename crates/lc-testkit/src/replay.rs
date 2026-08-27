//! `ReplayProvider`: replays from the recording file in FIFO order, zero network.
//!
//! Replay does **not match messages** (the prompt an LLMChain renders varies per call); it only
//! pops recordings in order. An exhausted queue returns [`TestkitError::ReplayExhausted`].

use std::collections::VecDeque;
use std::io::BufRead;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_core::tools::ToolDefinition;
use lc_schema::Message;

use crate::error::TestkitError;
use crate::recording::RecordedExchange;

/// Replay strategy: decides which recording a request takes in concurrent/out-of-order scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReplayStrategy {
    /// Concurrent FIFO: pops in recording order, atomic pop has no UB, but "which request gets
    /// which response" is not deterministic.
    ///
    /// Suits serial/order-stable chains; parallel scenarios use **order-independent assertions**
    /// (only assert that replay is fully drained and each response is structurally non-empty),
    /// never "the Nth response is tool A's".
    #[default]
    Fifo,
    /// Routes by tool name: tools carry a name when bound (`bind_tools`), replay picks the first
    /// recording in the queue whose tool name matches (request-side `exchange.tools` or
    /// response-side `tool_calls`).
    ///
    /// Covers parallel scenarios where "different tools return different responses and need exact
    /// correspondence" — each parallel branch binds its own tool and gets its own correct response.
    /// Weaker than full message-signature matching, but reproducible.
    ByToolName,
    /// Matches the request messages by **full signature**: the request `messages` must be
    /// **element-wise exactly equal** to `exchange.messages` (`Message`'s `PartialEq` over all
    /// fields, including `content` / `message_type` / `name` / `additional_kwargs` / `tool_calls`).
    ///
    /// Under parallel out-of-order traffic each request exactly gets its own response; a recording
    /// with no match returns an explicit [`TestkitError::ReplayNoMatch`], **never a silent FIFO
    /// fallback** (which would disguise "wrong response" as success). Suits deterministic replays
    /// where "the same message sequence always reproduces the same response" — any field drift
    /// between the recording and the request (e.g. a runtime-assigned `id` that differs each time)
    /// counts as no-match, so fixtures must keep those fields stable.
    Exact,
}

/// Zero-network `BaseChatModel` that replays from a recording file.
///
/// The queue is shared via `Arc<Mutex<VecDeque>>`; `bind_tools` returns a new instance
/// carrying a tool set and sharing the same queue with the original (FIFO order stays
/// consistent across branches).
#[derive(Clone)]
pub struct ReplayProvider {
    queue: Arc<Mutex<VecDeque<RecordedExchange>>>,
    model_name: String,
    strategy: ReplayStrategy,
    /// Tool definitions bound on this instance (the matching key for `ByToolName` routing).
    tools: Option<Vec<ToolDefinition>>,
}

impl ReplayProvider {
    /// Reads a JSONL recording file (missing file / bad line → `Err`).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TestkitError> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut queue = VecDeque::new();
        for line in reader.lines() {
            let line = line?.trim().to_string();
            if line.is_empty() {
                continue;
            }
            let exchange: RecordedExchange = serde_json::from_str(&line).map_err(|e| {
                TestkitError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid recording line: {e}"),
                ))
            })?;
            queue.push_back(exchange);
        }
        Ok(Self {
            queue: Arc::new(Mutex::new(queue)),
            model_name: "replay".to_string(),
            strategy: ReplayStrategy::Fifo,
            tools: None,
        })
    }

    /// In-memory construction (hand-written recordings = the equivalent of a MockProvider).
    pub fn from_exchanges(exchanges: Vec<RecordedExchange>) -> Self {
        Self {
            queue: Arc::new(Mutex::new(exchanges.into())),
            model_name: "replay".to_string(),
            strategy: ReplayStrategy::Fifo,
            tools: None,
        }
    }

    /// A single fixed response: every request returns the same `response` (simplest mock).
    pub fn single(response: LLMResult) -> Self {
        Self::from_exchanges(vec![RecordedExchange {
            messages: Vec::new(),
            response,
            tools: None,
        }])
    }

    /// Sets the replay strategy (default FIFO; `ByToolName` routes by tool name).
    pub fn with_strategy(mut self, strategy: ReplayStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Binds tools: returns a new instance carrying that tool set (sharing the same replay queue).
    ///
    /// Binding tools on the replay side only satisfies the agent's "tools-bound loop" precondition —
    /// the recording already contains the request-side `tools` and response-side `tool_calls`, so the
    /// replay logic needs no change. Under `ByToolName`, `tools` also serves as the routing key.
    pub fn bind_tools(&self, tools: Vec<ToolDefinition>) -> Self {
        Self {
            queue: self.queue.clone(),
            model_name: self.model_name.clone(),
            strategy: self.strategy,
            tools: Some(tools),
        }
    }

    /// Number of remaining recordings.
    pub fn len(&self) -> usize {
        self.queue.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether no recordings remain.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// `ByToolName` matching: true when either the request-side bound tool name or the
/// response-side requested tool-call name hits.
fn exchange_matches(exchange: &RecordedExchange, tool_name: &str) -> bool {
    if let Some(tools) = &exchange.tools {
        if tools.iter().any(|t| t.function.name == tool_name) {
            return true;
        }
    }
    if let Some(calls) = &exchange.response.tool_calls {
        if calls.iter().any(|c| c.name() == tool_name) {
            return true;
        }
    }
    false
}

/// `Exact` matching: the request message sequence and the recorded message sequence are
/// **element-wise exactly equal**.
///
/// `Vec<Message>`'s `PartialEq` compares length + each element, equivalent to "the message
/// count and every field of every message (including `id` / `name` / `additional_kwargs` /
/// `tool_calls`) match".
fn messages_match(request: &[Message], recorded: &[Message]) -> bool {
    request == recorded
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for ReplayProvider {
    type Error = TestkitError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, config).await
    }
}

impl BaseLanguageModel<Vec<Message>, LLMResult> for ReplayProvider {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        // Estimate: about 4 characters per token.
        text.chars().count() / 4 + 1
    }

    fn temperature(&self) -> Option<f32> {
        None
    }

    fn max_tokens(&self) -> Option<usize> {
        None
    }

    fn with_temperature(self, _temp: f32) -> Self {
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self {
        self
    }
}

#[async_trait]
impl BaseChatModel for ReplayProvider {
    async fn chat(
        &self,
        messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let exchange = match self.strategy {
            ReplayStrategy::Fifo => queue.pop_front(),
            ReplayStrategy::ByToolName => {
                let want = self
                    .tools
                    .as_ref()
                    .and_then(|tools| tools.first().map(|t| t.function.name.clone()));
                match want {
                    // Pick the matching recording from the queue by tool name; no bound tool → fall back to FIFO.
                    Some(name) => queue
                        .iter()
                        .position(|ex| exchange_matches(ex, &name))
                        .map(|i| queue.remove(i).expect("position 必有元素")),
                    None => queue.pop_front(),
                }
            }
            ReplayStrategy::Exact => {
                // Match exactly by full message signature, allowing parallel out-of-order; no match →
                // explicit error, never a silent FIFO fallback (otherwise "wrong response" would look like success).
                match queue
                    .iter()
                    .position(|ex| messages_match(&messages, &ex.messages))
                {
                    Some(i) => Some(queue.remove(i).expect("position 必有元素")),
                    None => {
                        return Err(TestkitError::ReplayNoMatch { left: queue.len() });
                    }
                }
            }
        };
        let Some(exchange) = exchange else {
            return Err(TestkitError::ReplayExhausted {
                requested: messages.len(),
            });
        };
        Ok(exchange.response)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        let response = self.chat(messages, config).await?;
        let stream = futures_util::stream::iter(vec![Ok(StreamChunk {
            text: response.content,
            token_usage: response.token_usage,
        })]);
        Ok(Box::pin(stream))
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        // Delegate to the inherent `bind_tools`: share the queue, record the tool set, return a new instance.
        Some(Box::new(self.bind_tools(tools)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::language_models::TokenUsage;
    use lc_core::tools::{ToolCall, ToolDefinition};

    fn exchange(content: &str) -> RecordedExchange {
        RecordedExchange {
            messages: vec![Message::system("ping")],
            response: LLMResult {
                content: content.to_string(),
                model: "replay".to_string(),
                token_usage: Some(TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    total_tokens: 3,
                }),
                ..Default::default()
            },
            tools: None,
        }
    }

    /// A recording with a response-side tool-call name (simulates the model requesting a tool).
    fn exchange_with_tool_call(tool_name: &str, content: &str) -> RecordedExchange {
        let mut response = exchange(content).response;
        response.tool_calls = Some(vec![ToolCall::builder("call_1")
            .name(tool_name)
            .arguments("{}".to_string())
            .build()]);
        RecordedExchange {
            response,
            ..exchange(content)
        }
    }

    /// A recording with a request-side bound tool name (simulates a request issued after binding a tool).
    fn exchange_with_bound_tool(tool_name: &str, content: &str) -> RecordedExchange {
        RecordedExchange {
            tools: Some(vec![ToolDefinition::new(tool_name, "a tool")]),
            ..exchange(content)
        }
    }

    #[tokio::test]
    async fn single_returns_fixed_response_for_any_request() {
        let provider = ReplayProvider::single(exchange("hello").response);
        let result = provider
            .chat(vec![Message::system("any")], None)
            .await
            .unwrap();
        assert_eq!(result.content, "hello");
    }

    #[tokio::test]
    async fn replay_is_fifo_ordered() {
        let provider = ReplayProvider::from_exchanges(vec![exchange("first"), exchange("second")]);
        let first = provider
            .chat(vec![Message::system("a")], None)
            .await
            .unwrap();
        let second = provider
            .chat(vec![Message::system("b")], None)
            .await
            .unwrap();
        assert_eq!(first.content, "first");
        assert_eq!(second.content, "second");
    }

    #[tokio::test]
    async fn replay_exhausted_returns_error() {
        let provider = ReplayProvider::from_exchanges(vec![exchange("only")]);
        provider
            .chat(vec![Message::system("a")], None)
            .await
            .unwrap();
        let err = provider
            .chat(vec![Message::system("b")], None)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            TestkitError::ReplayExhausted { requested: 1 }
        ));
    }

    #[test]
    fn bind_tools_returns_some_and_carries_tools() {
        let provider = ReplayProvider::from_exchanges(vec![exchange("x")]);
        // The inherent bind_tools returns a new instance carrying the tool set (shared queue).
        let bound = provider.bind_tools(vec![ToolDefinition::new("calculator", "calc")]);
        assert!(bound.tools.is_some());
        assert_eq!(bound.tools.as_ref().unwrap()[0].function.name, "calculator");
        assert_eq!(provider.len(), 1);
        assert_eq!(bound.len(), 1);
        // The trait bind_tools (called by agents through `Box<dyn BaseChatModel>`) is always Some.
        let trait_bound = BaseChatModel::bind_tools(&provider, vec![ToolDefinition::new("x", "y")]);
        assert!(trait_bound.is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn parallel_replay_fifo_is_order_independent() {
        // Concurrent FIFO: two requests arrive simultaneously; which gets which response is
        // nondeterministic, but both succeed and the queue drains. Assert with "set equality"
        // rather than "the Nth is xx".
        let provider = ReplayProvider::from_exchanges(vec![
            exchange("first"),
            exchange("second"),
            exchange("third"),
        ]);
        let provider = std::sync::Arc::new(provider);

        let mut handles = Vec::new();
        for _ in 0..3 {
            let p = provider.clone();
            handles.push(tokio::spawn(async move {
                p.chat(vec![Message::system("parallel")], None)
                    .await
                    .expect("并发回放不应失败")
            }));
        }
        let mut contents: Vec<String> = Vec::new();
        for handle in handles {
            contents.push(handle.await.unwrap().content);
        }
        contents.sort();
        assert_eq!(
            contents,
            vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string()
            ]
        );
        assert!(provider.is_empty(), "并发回放应恰好耗尽全部录播");
    }

    #[tokio::test]
    async fn by_tool_name_routes_to_matching_exchange() {
        // Each parallel branch binds its own tool and should get its own correct response — order-independent.
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_tool_call("search", "search result"),
            exchange_with_tool_call("calc", "calc result"),
        ])
        .with_strategy(ReplayStrategy::ByToolName);

        let search =
            BaseChatModel::bind_tools(&provider, vec![ToolDefinition::new("search", "s")]).unwrap();
        let calc =
            BaseChatModel::bind_tools(&provider, vec![ToolDefinition::new("calc", "c")]).unwrap();

        let calc_res = calc.chat(vec![Message::system("q")], None).await.unwrap();
        let search_res = search.chat(vec![Message::system("q")], None).await.unwrap();

        assert_eq!(search_res.content, "search result");
        assert_eq!(calc_res.content, "calc result");
        assert!(provider.is_empty());
    }

    #[tokio::test]
    async fn by_tool_name_matches_request_side_tools() {
        // The request-side `exchange.tools` also participates in matching (recorded from requests issued after binding tools).
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_bound_tool("weather", "sunny"),
            exchange_with_bound_tool("news", "headlines"),
        ])
        .with_strategy(ReplayStrategy::ByToolName);

        let weather =
            BaseChatModel::bind_tools(&provider, vec![ToolDefinition::new("weather", "w")])
                .unwrap();
        let res = weather
            .chat(vec![Message::system("q")], None)
            .await
            .unwrap();
        assert_eq!(res.content, "sunny");
    }

    /// A recording with a specific request message sequence (for `Exact` strategy full-signature matching).
    fn exchange_with_messages(messages: Vec<Message>, content: &str) -> RecordedExchange {
        RecordedExchange {
            messages,
            ..exchange(content)
        }
    }

    #[tokio::test]
    async fn exact_strategy_matches_by_full_signature_out_of_order() {
        // The two request message sequences differ; arriving out of order, each exactly gets its own response.
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_messages(vec![Message::system("ping")], "pong"),
            exchange_with_messages(vec![Message::human("hello")], "hi"),
        ])
        .with_strategy(ReplayStrategy::Exact);

        let hello_res = provider
            .chat(vec![Message::human("hello")], None)
            .await
            .unwrap();
        let ping_res = provider
            .chat(vec![Message::system("ping")], None)
            .await
            .unwrap();

        assert_eq!(hello_res.content, "hi");
        assert_eq!(ping_res.content, "pong");
        assert!(provider.is_empty(), "两条请求应精确消耗两条录播");
    }

    #[tokio::test]
    async fn exact_strategy_matches_full_message_sequence() {
        // Multi-turn history (system + user + AI) acts as one whole signature, matched by the full sequence.
        let msgs = vec![
            Message::system("You are a calculator."),
            Message::human("2 + 2"),
            Message::ai("I'll compute that."),
        ];
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_messages(vec![Message::human("other")], "wrong"),
            exchange_with_messages(msgs.clone(), "42"),
        ])
        .with_strategy(ReplayStrategy::Exact);

        let res = provider.chat(msgs, None).await.unwrap();
        assert_eq!(res.content, "42");
    }

    #[tokio::test]
    async fn exact_strategy_no_match_returns_explicit_error() {
        // No recording matches the request signature → explicit error (not a silent FIFO mispick).
        let provider = ReplayProvider::from_exchanges(vec![exchange_with_messages(
            vec![Message::system("ping")],
            "pong",
        )])
        .with_strategy(ReplayStrategy::Exact);

        let err = provider
            .chat(vec![Message::system("different")], None)
            .await
            .unwrap_err();
        assert!(
            matches!(err, TestkitError::ReplayNoMatch { left: 1 }),
            "无匹配应返回 ReplayNoMatch,剩余录播保留"
        );
    }

    #[tokio::test]
    async fn exact_strategy_distinguishes_message_type() {
        // Same text, different message type = different signature.
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_messages(vec![Message::human("q")], "human response"),
            exchange_with_messages(vec![Message::system("q")], "system response"),
        ])
        .with_strategy(ReplayStrategy::Exact);

        let human = provider
            .chat(vec![Message::human("q")], None)
            .await
            .unwrap();
        assert_eq!(human.content, "human response");

        let system = provider
            .chat(vec![Message::system("q")], None)
            .await
            .unwrap();
        assert_eq!(system.content, "system response");
    }

    #[tokio::test]
    async fn exact_strategy_distinguishes_tool_calls() {
        // A message carrying tool calls and a plain-text message = different signatures.
        let call = ToolCall::builder("call_1")
            .name("weather")
            .arguments("{}".to_string())
            .build();
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_messages(vec![Message::ai("q")], "plain"),
            exchange_with_messages(
                vec![Message::ai_with_tool_calls("q", vec![call.clone()])],
                "with tool",
            ),
        ])
        .with_strategy(ReplayStrategy::Exact);

        let plain = provider.chat(vec![Message::ai("q")], None).await.unwrap();
        assert_eq!(plain.content, "plain");

        let with_tool = provider
            .chat(vec![Message::ai_with_tool_calls("q", vec![call])], None)
            .await
            .unwrap();
        assert_eq!(with_tool.content, "with tool");
    }
}
