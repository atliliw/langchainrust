//! `ReplayProvider`: replays from the recording file in FIFO order, zero network.
//!
//! Replay does **not match messages** (the prompt an LLMChain renders varies per call); it only
//! pops recordings in order. An exhausted queue returns [`TestkitError::ReplayExhausted`].

use std::collections::VecDeque;
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
    /// Routes by tool name: tools carry a name when bound (`bind_tools`), replay walks **every**
    /// bound tool of the branch and takes the first queued recording whose tool name matches any
    /// of them (request-side `exchange.tools` or response-side `tool_calls`).
    ///
    /// Covers parallel scenarios where "different tools return different responses and need exact
    /// correspondence" — each parallel branch binds its own tool and gets its own correct response
    /// (a branch may also bind several tools; it then drains the recordings matching any of them
    /// in queue order). Weaker than full message-signature matching, but reproducible.
    ByToolName,
    /// Matches the request messages by **full signature**: the request `messages` must be
    /// **element-wise exactly equal** to `exchange.messages`, comparing each `Message` field.
    /// Which fields participate is configurable via [`ExactMatchFilter`] (aliased here as
    /// [`ReplayProvider::with_exact_filter`] / [`ReplayProvider::ignore_ids`]) — by default
    /// *every* field, including `id` / `name` / `additional_kwargs` / `tool_calls`.
    ///
    /// Under parallel out-of-order traffic each request exactly gets its own response; a recording
    /// with no match returns an explicit [`TestkitError::ReplayNoMatch`] (with a closest-match
    /// field diff; see [`MatchDiff`]), **never a silent FIFO fallback** (which would disguise
    /// "wrong response" as success). Suits deterministic replays where "the same message sequence
    /// always reproduces the same response" — a runtime-assigned `id` that differs each request
    /// counts as no-match unless you relax the signature with [`ReplayProvider::ignore_ids`].
    Exact,
}

/// Which `Message` fields `ReplayStrategy::Exact` participates when matching request vs recording,
/// expressed as a blacklist of volatile fields to ignore.
///
/// Runtime-assigned fields (notably `Message.id`, and `additional_kwargs` carrying per-request
/// bookkeeping) can drift between a recording and a replayed request even when the *content* is
/// identical; the default ([`ExactMatchFilter::new`]) treats any drift as a no-match, so fixtures
/// must keep those fields stable. [`ExactMatchFilter::IGNORE_ID`] drops the single most volatile
/// field so the signature keys on stable content alone.
///
/// ```
/// use lc_testkit::ExactMatchFilter;
/// // Ignore a runtime `id` so `Exact` keys on stable content alone.
/// let filter = ExactMatchFilter::new().with(ExactMatchFilter::IGNORE_ID);
/// assert!(filter.contains(ExactMatchFilter::IGNORE_ID));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExactMatchFilter(u8);

impl ExactMatchFilter {
    /// Compare every field — the historical strict `Exact` behavior.
    pub const EMPTY: ExactMatchFilter = ExactMatchFilter(0);
    /// Ignore `Message.id` when comparing signatures.
    pub const IGNORE_ID: ExactMatchFilter = ExactMatchFilter(1 << 0);
    /// Ignore `Message.additional_kwargs` (per-request bookkeeping often drifts).
    pub const IGNORE_ADDITIONAL_KWARGS: ExactMatchFilter = ExactMatchFilter(1 << 1);

    /// A filter that compares every field.
    pub const fn new() -> Self {
        Self::EMPTY
    }

    /// Whether `self` turns on `flag`.
    pub const fn contains(self, flag: ExactMatchFilter) -> bool {
        (self.0 & flag.0) == flag.0
    }

    /// Union of `self` and `flag` (a filter that ignores `flag` in addition to whatever `self`
    /// already ignores).
    pub const fn with(self, flag: ExactMatchFilter) -> Self {
        Self(self.0 | flag.0)
    }
}

/// Field-level difference report attached to [`TestkitError::ReplayNoMatch`]: names the closest
/// remaining recorded exchange and *what* differs, so a drift is debuggable instead of a bare
/// "no match" (the usual culprit is a runtime `Message.id` — see [`ExactMatchFilter::IGNORE_ID`]).
#[derive(Debug, Clone)]
pub struct MatchDiff {
    /// 0-based index of the closest exchange within the remaining queue.
    pub position: usize,
    /// Number of request messages.
    pub request_messages: usize,
    /// Number of messages in the closest recorded exchange.
    pub recorded_messages: usize,
    /// The closest recorded exchange's message sequence — the failing signature itself.
    pub recorded: Vec<Message>,
    /// First per-message divergence between the request and the closest recorded exchange.
    pub divergence: MessageDiff,
}

/// A single message-position divergence between a request and a recorded exchange.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MessageDiff {
    /// Message at index `at` and the listed fields differ.
    FieldsDiffer { at: usize, fields: Vec<String> },
    /// The request is shorter: the recording has extra messages from `at` onward.
    RequestShorter { at: usize },
    /// The request is longer: the recording stops at `at`.
    RequestLonger { at: usize },
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
    /// Field filter for `Exact` matching (which `Message` fields participate).
    exact_filter: ExactMatchFilter,
}

impl ReplayProvider {
    /// Reads a JSONL recording file (missing file / bad line / replay-ineligible exchange → `Err`).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TestkitError> {
        // Single parse site for the recording format lives in `recording::read_exchanges`
        // (which also runs the minimal replay-qualification check).
        let queue: VecDeque<RecordedExchange> = crate::recording::read_exchanges(path)?.into();
        Ok(Self {
            queue: Arc::new(Mutex::new(queue)),
            model_name: "replay".to_string(),
            strategy: ReplayStrategy::Fifo,
            tools: None,
            exact_filter: ExactMatchFilter::new(),
        })
    }

    /// In-memory construction (hand-written recordings = the equivalent of a MockProvider).
    pub fn from_exchanges(exchanges: Vec<RecordedExchange>) -> Self {
        Self {
            queue: Arc::new(Mutex::new(exchanges.into())),
            model_name: "replay".to_string(),
            strategy: ReplayStrategy::Fifo,
            tools: None,
            exact_filter: ExactMatchFilter::new(),
        }
    }

    /// A single fixed response: every request returns the same `response` (simplest mock).
    pub fn single(response: LLMResult) -> Self {
        Self::from_exchanges(vec![RecordedExchange {
            messages: Vec::new(),
            response,
            tools: Vec::new(),
        }])
    }

    /// Sets the replay strategy (default FIFO; `ByToolName` routes by tool name).
    pub fn with_strategy(mut self, strategy: ReplayStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets how `Exact` matching treats `Message` fields (default: every field).
    pub fn with_exact_filter(mut self, filter: ExactMatchFilter) -> Self {
        self.exact_filter = filter;
        self
    }

    /// Ignore `Message.id` during `Exact` matching — the common fix for a runtime-assigned `id`
    /// that differs from the recording while the content is identical.
    pub fn ignore_ids(mut self) -> Self {
        self.exact_filter = self.exact_filter.with(ExactMatchFilter::IGNORE_ID);
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
            exact_filter: self.exact_filter,
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
    if exchange.tools.iter().any(|t| t.function.name == tool_name) {
        return true;
    }
    if let Some(calls) = &exchange.response.tool_calls {
        if calls.iter().any(|c| c.name() == tool_name) {
            return true;
        }
    }
    false
}

/// Whether two messages are equivalent under `filter` (all non-ignored fields match). Returns the
/// differing field names; an empty vec means they match.
fn message_diff_fields(a: &Message, b: &Message, filter: ExactMatchFilter) -> Vec<String> {
    let mut diff = Vec::new();
    if a.content != b.content {
        diff.push("content".into());
    }
    if a.images != b.images {
        diff.push("images".into());
    }
    if a.audio != b.audio {
        diff.push("audio".into());
    }
    if a.videos != b.videos {
        diff.push("videos".into());
    }
    if a.files != b.files {
        diff.push("files".into());
    }
    if a.message_type != b.message_type {
        diff.push("message_type".into());
    }
    if a.name != b.name {
        diff.push("name".into());
    }
    if !filter.contains(ExactMatchFilter::IGNORE_ADDITIONAL_KWARGS)
        && a.additional_kwargs != b.additional_kwargs
    {
        diff.push("additional_kwargs".into());
    }
    if !filter.contains(ExactMatchFilter::IGNORE_ID) && a.id != b.id {
        diff.push("id".into());
    }
    if a.tool_calls != b.tool_calls {
        diff.push("tool_calls".into());
    }
    diff
}

fn message_fields_equal(a: &Message, b: &Message, filter: ExactMatchFilter) -> bool {
    message_diff_fields(a, b, filter).is_empty()
}

/// `Exact` matching under `filter`: the request and recorded message sequences are element-wise
/// exactly equal (in every field *not* ignored by `filter`), same length.
fn messages_match(request: &[Message], recorded: &[Message], filter: ExactMatchFilter) -> bool {
    request.len() == recorded.len()
        && request
            .iter()
            .zip(recorded)
            .all(|(a, b)| message_fields_equal(a, b, filter))
}

/// Number of index-aligned messages that match under `filter` (used to rank "closest" for the
/// no-match diagnostic).
fn matching_count(request: &[Message], recorded: &[Message], filter: ExactMatchFilter) -> usize {
    let n = request.len().min(recorded.len());
    (0..n)
        .filter(|&i| message_fields_equal(&request[i], &recorded[i], filter))
        .count()
}

/// The first message-position divergence between a request and a recorded exchange under
/// `filter`; `None` means fully equal (never returned for a `ReplayNoMatch`, which by definition
/// has no equal exchange).
fn first_divergence(
    request: &[Message],
    recorded: &[Message],
    filter: ExactMatchFilter,
) -> Option<MessageDiff> {
    let n = request.len().min(recorded.len());
    for i in 0..n {
        let fields = message_diff_fields(&request[i], &recorded[i], filter);
        if !fields.is_empty() {
            return Some(MessageDiff::FieldsDiffer { at: i, fields });
        }
    }
    if request.len() < recorded.len() {
        Some(MessageDiff::RequestShorter { at: request.len() })
    } else if request.len() > recorded.len() {
        Some(MessageDiff::RequestLonger { at: recorded.len() })
    } else {
        None
    }
}

/// Builds a [`MatchDiff`] for the closest remaining recorded exchange (the one sharing the most
/// index-aligned matching messages with the request; ties resolve to the earliest). `None` only
/// when the queue is empty.
fn closest_match_diff(
    request: &[Message],
    queue: &VecDeque<RecordedExchange>,
    filter: ExactMatchFilter,
) -> Option<MatchDiff> {
    let (best_idx, _best_score) =
        queue
            .iter()
            .enumerate()
            .fold((0usize, usize::MIN), |(bi, bs), (i, ex)| {
                let s = matching_count(request, &ex.messages, filter);
                if s > bs {
                    (i, s)
                } else {
                    (bi, bs)
                }
            });
    let recorded = queue.get(best_idx)?;
    let divergence = first_divergence(request, &recorded.messages, filter).unwrap_or_else(|| {
        // Unreachable for an actual no-match (a fully-equal exchange would have matched), but
        // keep an empty-field placeholder for the "ignored-field-only" edge.
        MessageDiff::FieldsDiffer {
            at: 0,
            fields: Vec::new(),
        }
    });
    Some(MatchDiff {
        position: best_idx,
        request_messages: request.len(),
        recorded_messages: recorded.messages.len(),
        recorded: recorded.messages.clone(),
        divergence,
    })
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
                // Routing key = the distinct set of bound tool names, order-independent
                // (sort + dedup so routing never depends on registration order). A branch
                // may bind MANY tools; take the first queued recording matching ANY of them.
                let names: Vec<String> = self
                    .tools
                    .as_ref()
                    .map(|tools| {
                        let mut n: Vec<String> =
                            tools.iter().map(|t| t.function.name.clone()).collect();
                        n.sort_unstable();
                        n.dedup();
                        n
                    })
                    .unwrap_or_default();
                if names.is_empty() {
                    // No bound tools: nothing to route by, degrade to FIFO.
                    queue.pop_front()
                } else {
                    queue
                        .iter()
                        .position(|ex| names.iter().any(|name| exchange_matches(ex, name)))
                        .map(|i| queue.remove(i).expect("position 必有元素"))
                }
            }
            ReplayStrategy::Exact => {
                // Match exactly by full message signature (under `exact_filter`), allowing parallel
                // out-of-order; no match → explicit error (never a silent FIFO fallback, otherwise
                // "wrong response" would look like success), annotated with the closest match diff.
                match queue
                    .iter()
                    .position(|ex| messages_match(&messages, &ex.messages, self.exact_filter))
                {
                    Some(i) => Some(queue.remove(i).expect("position 必有元素")),
                    None => {
                        let closest = closest_match_diff(&messages, &queue, self.exact_filter);
                        return Err(TestkitError::ReplayNoMatch {
                            left: queue.len(),
                            closest,
                        });
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
        // I2: forward the recorded `tool_calls` instead of hardcoding `None`, so a
        // streamed agent sees the replayed tool request the same way the real
        // provider produced it. Extract before the partial field moves below.
        let tool_calls = response.tool_calls;
        let stream = futures_util::stream::iter(vec![Ok(StreamChunk {
            thinking_content: None,
            text: response.content,
            token_usage: response.token_usage,
            tool_calls,
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
            tools: Vec::new(),
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
            tools: vec![ToolDefinition::new(tool_name, "a tool")],
            ..exchange(content)
        }
    }

    /// A recording with a specific request message sequence (for `Exact` strategy full-signature matching).
    fn exchange_with_messages(messages: Vec<Message>, content: &str) -> RecordedExchange {
        RecordedExchange {
            messages,
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

    #[tokio::test]
    async fn by_tool_name_multi_tool_branch_routes_any_bound_tool() {
        // A single branch binds several tools; the next request drains the first recording that
        // matches ANY of them, deterministically in queue order (not tied to registration order).
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_tool_call("search", "search result"),
            exchange_with_tool_call("calc", "calc result"),
        ])
        .with_strategy(ReplayStrategy::ByToolName);

        let multi = BaseChatModel::bind_tools(
            &provider,
            vec![
                ToolDefinition::new("search", "s"),
                ToolDefinition::new("calc", "c"),
            ],
        )
        .unwrap();

        let first = multi.chat(vec![Message::system("a")], None).await.unwrap();
        let second = multi.chat(vec![Message::system("b")], None).await.unwrap();
        let mut contents = vec![first.content, second.content];
        contents.sort();
        assert_eq!(contents, vec!["calc result", "search result"]);
        assert!(provider.is_empty());
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
            matches!(err, TestkitError::ReplayNoMatch { left: 1, .. }),
            "无匹配应返回 ReplayNoMatch,剩余录播保留"
        );
    }

    #[tokio::test]
    async fn exact_strategy_no_match_reports_closest_and_divergence() {
        // The request differs from the only recording by its `name` field; the error should
        // name that closest exchange and point at the divergent field so the drift is debuggable.
        let mut recorded_msg = Message::system("q");
        recorded_msg.name = Some("agent-a".to_string());
        let provider = ReplayProvider::from_exchanges(vec![exchange_with_messages(
            vec![recorded_msg],
            "pong",
        )])
        .with_strategy(ReplayStrategy::Exact);

        let err = provider
            .chat(vec![Message::system("q")], None)
            .await
            .unwrap_err();
        let TestkitError::ReplayNoMatch { left, closest } = err else {
            panic!("expected ReplayNoMatch");
        };
        assert_eq!(left, 1);
        let diff = closest.expect("queue 非空,应给出最近录播");
        assert_eq!(diff.position, 0);
        assert_eq!(diff.recorded_messages, 1);
        assert_eq!(diff.request_messages, 1);
        let MessageDiff::FieldsDiffer { at, fields } = diff.divergence else {
            panic!("应报字段差异,而非长度差异");
        };
        assert_eq!(at, 0);
        assert_eq!(fields, vec!["name".to_string()]);
    }

    #[tokio::test]
    async fn exact_strategy_ignore_ids_matches_otherwise_identical_signature() {
        // Runtime id drift is the common no-match cause: with `ignore_ids` the otherwise-identical
        // request matches, where the strict default would report ReplayNoMatch.
        fn with_id(id: &str) -> Message {
            let mut m = Message::human("q");
            m.id = Some(id.to_string());
            m
        }
        let provider = ReplayProvider::from_exchanges(vec![exchange_with_messages(
            vec![with_id("recorded-id")],
            "pong",
        )])
        .with_strategy(ReplayStrategy::Exact);

        // strict: different id → no match
        let strict_err = provider
            .chat(vec![with_id("fresh-id")], None)
            .await
            .unwrap_err();
        assert!(matches!(strict_err, TestkitError::ReplayNoMatch { .. }));

        // relaxed: id ignored → matches
        let provider = provider.ignore_ids();
        let res = provider
            .chat(vec![with_id("fresh-id")], None)
            .await
            .unwrap();
        assert_eq!(res.content, "pong");
    }

    #[test]
    fn exact_match_filter_blacklists_fields() {
        let filter = ExactMatchFilter::new().with(ExactMatchFilter::IGNORE_ID);
        assert!(filter.contains(ExactMatchFilter::IGNORE_ID));
        assert!(!filter.contains(ExactMatchFilter::IGNORE_ADDITIONAL_KWARGS));
        assert!(!ExactMatchFilter::new().contains(ExactMatchFilter::IGNORE_ID));
        assert_eq!(ExactMatchFilter::EMPTY, ExactMatchFilter::new());
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

    /// A recording line with an empty-named tool call must be rejected at load time (it could
    /// never be routed/matched faithfully) — the minimal replay-qualification check.
    #[tokio::test]
    async fn read_rejects_recording_with_empty_tool_name() {
        let mut response = exchange("unused").response;
        response.tool_calls = Some(vec![ToolCall::builder("call_1")
            .name("")
            .arguments("{}".to_string())
            .build()]);
        let recording = RecordedExchange {
            response,
            ..exchange("unused")
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        std::fs::write(
            &path,
            format!("{}\n", serde_json::to_string(&recording).unwrap()),
        )
        .unwrap();

        let err = match ReplayProvider::from_file(&path) {
            Err(e) => e,
            Ok(_) => panic!("空 tool name 录播应被 from_file 拒绝"),
        };
        assert!(
            matches!(err, TestkitError::InvalidRecording { index: 0, .. }),
            "应拒绝空 tool name 录播: {err:?}"
        );
    }
}
