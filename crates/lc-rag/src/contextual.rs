// lc-rag/src/contextual.rs
//! Contextual Retrieval: index-time context injection for chunks (0.21.0 S4.1).
//!
//! Chunks lose their context when split out of a document — an ambiguous chunk
//! ("revenue grew 3%") may embed without the subject it refers to and never be
//! recalled for the right query. Contextual Retrieval (Anthropic, ~49% fewer
//! retrieval failures) asks a small LLM to write 1-2 situating sentences per
//! chunk at index time and embeds `context + original content`. The context is
//! also stored in metadata, so it stays auditable and searchable.
//!
//! Design:
//! - pure transform: takes already-split documents, returns enhanced copies —
//!   the caller inserts it before `index_documents`; retrieval flow unchanged;
//! - bounded concurrency ([`tokio::sync::Semaphore`]) to cap index cost;
//! - **fail-open**: a failed context generation logs a warning and keeps the
//!   original content — indexing must never block on the enhancement;
//! - idempotent: chunks already enhanced (metadata marker present) are skipped,
//!   so re-running over an indexed corpus is a no-op.

use lc_core::language_models::BaseChatModel;
use lc_prompts::PromptTemplate;
use lc_providers::ProviderError;
use lc_schema::Message;
use lc_vector_stores::Document;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Metadata key under which the generated context is stored (auditable).
pub const CONTEXTUAL_METADATA_KEY: &str = "contextual_context";

/// Contextual Retrieval error type.
#[derive(Debug)]
#[non_exhaustive]
pub enum ContextualError {
    /// LLM call error.
    LLMError(String),
}

impl std::fmt::Display for ContextualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextualError::LLMError(msg) => write!(f, "LLM error: {}", msg),
        }
    }
}

impl std::error::Error for ContextualError {}

/// Contextual Retrieval configuration.
pub struct ContextualConfig {
    /// Prompt template; `{chunk}` is replaced with the chunk text.
    pub prompt_template: String,
    /// Max LLM calls in flight (index-cost cap).
    pub max_concurrency: usize,
}

impl Default for ContextualConfig {
    fn default() -> Self {
        Self {
            prompt_template: DEFAULT_CONTEXTUAL_PROMPT.to_string(),
            max_concurrency: 4,
        }
    }
}

impl ContextualConfig {
    /// Creates a config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the prompt template (`{chunk}` placeholder required).
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt_template = prompt.into();
        self
    }

    /// Sets the max in-flight context-generation calls.
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency.max(1);
        self
    }
}

const DEFAULT_CONTEXTUAL_PROMPT: &str = r#"You are preparing document chunks for a retrieval index. Write 1-2 short sentences that situate the chunk within its broader document: what the document is about and what part this chunk plays. Answer with the context sentences only — no preamble, no quotes.

Chunk:
{chunk}

Context:"#;

/// Contextual Retrieval enhancer (index-time transform).
///
/// `L` is any [`BaseChatModel`]; a cheap small model is recommended (one call
/// per chunk). Enhancement failures are logged and degrade to the original
/// content — see the module docs.
pub struct ContextualEnhancer {
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    config: ContextualConfig,
    /// Bounds in-flight LLM calls across all `enhance_documents` calls on
    /// clones of this enhancer (the Arc is shared through clones).
    semaphore: Arc<Semaphore>,
}

impl std::fmt::Debug for ContextualEnhancer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextualEnhancer")
            .field("max_concurrency", &self.config.max_concurrency)
            .finish()
    }
}

impl ContextualEnhancer {
    /// Creates an enhancer from any [`BaseChatModel`].
    pub fn new<L>(llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self::new_arc(lc_providers::wrap_chat_model(llm))
    }

    /// Builds from an already-wrapped `Arc<dyn BaseChatModel<Error = ProviderError>>`.
    pub fn new_arc(llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>) -> Self {
        let semaphore = Arc::new(Semaphore::new(4));
        Self {
            llm,
            config: ContextualConfig::default(),
            semaphore,
        }
    }

    /// Sets the configuration.
    pub fn with_config(mut self, config: ContextualConfig) -> Self {
        self.semaphore = Arc::new(Semaphore::new(config.max_concurrency));
        self.config = config;
        self
    }

    /// Sets the max in-flight LLM calls.
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.config.max_concurrency = max_concurrency.max(1);
        self.semaphore = Arc::new(Semaphore::new(max_concurrency.max(1)));
        self
    }

    /// Generates the situating context for one chunk (raw LLM access).
    pub async fn generate_context(&self, chunk: &str) -> Result<String, ContextualError> {
        let template = PromptTemplate::new(&self.config.prompt_template);
        let mut vars = HashMap::new();
        vars.insert("chunk", chunk);
        let prompt = template
            .format(&vars)
            .unwrap_or_else(|_| self.config.prompt_template.clone());

        let response = self
            .llm
            .invoke(vec![Message::human(prompt)], None)
            .await
            .map_err(|e| ContextualError::LLMError(e.to_string()))?;
        Ok(response.content.trim().to_string())
    }

    /// Enhances one document: returns a copy whose content is
    /// `context + "\n" + original` and whose metadata carries the context under
    /// [`CONTEXTUAL_METADATA_KEY`].
    ///
    /// Errors surface to the caller (use [`Self::enhance_documents`] for the
    /// fail-open batch path).
    pub async fn enhance_document(&self, doc: &Document) -> Result<Document, ContextualError> {
        let context = self.generate_context(&doc.content).await?;
        let mut enhanced = Document::new(format!("{}\n{}", context, doc.content))
            .with_metadata(CONTEXTUAL_METADATA_KEY, context.clone());
        // Preserve id and all existing metadata.
        if let Some(id) = &doc.id {
            enhanced = enhanced.with_id(id.clone());
        }
        for (k, v) in &doc.metadata {
            enhanced.metadata.insert(k.clone(), v.clone());
        }
        Ok(enhanced)
    }

    /// Batch enhancement with bounded concurrency and fail-open semantics.
    ///
    /// Each chunk is enhanced concurrently (up to `max_concurrency` in flight);
    /// a chunk whose context generation fails keeps its original content (a
    /// warning is logged) — indexing never blocks on the enhancement.
    /// Idempotent: chunks already carrying [`CONTEXTUAL_METADATA_KEY`] are
    /// passed through untouched, so re-running over an indexed corpus is a
    /// no-op.
    pub async fn enhance_documents(&self, docs: &[Document]) -> Vec<Document> {
        let mut handles = Vec::with_capacity(docs.len());
        for doc in docs {
            let doc = doc.clone();
            let llm = self.llm.clone();
            let prompt_template = self.config.prompt_template.clone();
            let permit = self.semaphore.clone();
            handles.push(tokio::spawn(async move {
                if doc.metadata.contains_key(CONTEXTUAL_METADATA_KEY) {
                    return doc; // idempotent skip
                }
                let _permit = permit.acquire().await;
                let template = PromptTemplate::new(&prompt_template);
                let mut vars = HashMap::new();
                vars.insert("chunk", doc.content.as_str());
                let prompt = template
                    .format(&vars)
                    .unwrap_or_else(|_| prompt_template.clone());
                match llm.invoke(vec![Message::human(prompt)], None).await {
                    Ok(response) => {
                        let context = response.content.trim().to_string();
                        if context.is_empty() {
                            log::warn!("contextual retrieval: empty context generated, keeping original content");
                            doc
                        } else {
                            let mut enhanced =
                                Document::new(format!("{}\n{}", context, doc.content))
                                    .with_metadata(CONTEXTUAL_METADATA_KEY, context);
                            if let Some(id) = &doc.id {
                                enhanced = enhanced.with_id(id.clone());
                            }
                            for (k, v) in &doc.metadata {
                                enhanced.metadata.insert(k.clone(), v.clone());
                            }
                            enhanced
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "contextual retrieval: context generation failed ({}), keeping original content",
                            e
                        );
                        doc
                    }
                }
            }));
        }
        let mut out = Vec::with_capacity(handles.len());
        for handle in handles {
            out.push(handle.await.unwrap_or_else(|e| {
                log::warn!(
                    "contextual retrieval: enhancement task panicked ({}), keeping placeholder",
                    e
                );
                Document::new("")
            }));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::Stream;
    use lc_core::language_models::{BaseLanguageModel, LLMResult, StreamChunk};
    use lc_core::runnables::{Runnable, RunnableConfig};
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock chat model returning a fixed context sentence per call.
    #[derive(Clone)]
    struct MockChat {
        reply: String,
        calls: Arc<AtomicUsize>,
    }

    impl MockChat {
        fn new(reply: &str) -> Self {
            Self {
                reply: reply.to_string(),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[derive(Debug)]
    struct MockChatError(String);

    impl std::fmt::Display for MockChatError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "mock chat error: {}", self.0)
        }
    }
    impl std::error::Error for MockChatError {}

    impl From<MockChatError> for ProviderError {
        fn from(e: MockChatError) -> Self {
            ProviderError::Config(e.to_string())
        }
    }

    #[async_trait::async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockChat {
        type Error = MockChatError;
        async fn invoke(
            &self,
            input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let prompt = input.first().map(|m| m.content.clone()).unwrap_or_default();
            Ok(LLMResult {
                content: format!("{} [from: {}]", self.reply, prompt),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait::async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockChat {
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

    #[async_trait::async_trait]
    impl BaseChatModel for MockChat {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            <Self as Runnable<Vec<Message>, LLMResult>>::invoke(self, messages, config).await
        }
        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            Err(MockChatError("no stream".to_string()))
        }
    }

    /// Failing mock for the fail-open path.
    struct FailingChat;

    #[async_trait::async_trait]
    impl Runnable<Vec<Message>, LLMResult> for FailingChat {
        type Error = MockChatError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(MockChatError("llm down".to_string()))
        }
    }

    #[async_trait::async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for FailingChat {
        fn model_name(&self) -> &str {
            "failing"
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

    #[async_trait::async_trait]
    impl BaseChatModel for FailingChat {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(MockChatError("llm down".to_string()))
        }
        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            Err(MockChatError("no stream".to_string()))
        }
    }

    #[test]
    fn config_defaults_and_builders() {
        let config = ContextualConfig::default();
        assert_eq!(config.max_concurrency, 4);
        assert!(config.prompt_template.contains("{chunk}"));

        let config = ContextualConfig::new()
            .with_prompt("Contextualize: {chunk}")
            .with_max_concurrency(0); // clamped to 1
        assert_eq!(config.max_concurrency, 1);
        assert!(config.prompt_template.contains("{chunk}"));
    }

    /// Single-document enhancement: content is prefixed with the context, the
    /// context is stored in metadata, and the prompt carried the chunk text.
    #[tokio::test]
    async fn enhance_document_prefixes_context_and_stores_metadata() {
        let mock = MockChat::new("This chunk is part of the revenue report.");
        let enhancer = ContextualEnhancer::new(mock);
        let doc = Document::new("Revenue grew 3% year over year.");

        let enhanced = enhancer.enhance_document(&doc).await.unwrap();

        assert!(
            enhanced
                .content
                .starts_with("This chunk is part of the revenue report."),
            "context should prefix the original content"
        );
        assert!(enhanced
            .content
            .ends_with("Revenue grew 3% year over year."));
        let content = &enhanced.content;
        let expected_context_len = content.len() - doc.content.len() - "\n".len();
        assert_eq!(
            enhanced
                .metadata
                .get(CONTEXTUAL_METADATA_KEY)
                .and_then(|v| v.as_str()),
            Some(&content[..expected_context_len]),
            "metadata stores exactly the generated context (the content prefix)"
        );
    }

    /// The prompt sent to the LLM contains the chunk text ({chunk} replaced):
    /// the mock echoes the prompt into its reply, so the stored context must
    /// contain the original chunk text.
    #[tokio::test]
    async fn prompt_sent_to_llm_contains_chunk() {
        let mock = MockChat::new("ctx");
        let enhancer = ContextualEnhancer::new(mock.clone());
        let doc = Document::new("Revenue grew 3% year over year.");
        let _ = enhancer.generate_context(&doc.content).await.unwrap();

        let document = enhancer.enhance_document(&doc).await.unwrap();
        let context = document
            .metadata
            .get(CONTEXTUAL_METADATA_KEY)
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(
            context.contains("Revenue grew 3% year over year."),
            "prompt should carry the chunk text, got: {}",
            context
        );
    }

    /// Fail-open: LLM failure keeps the original content and adds no metadata.
    #[tokio::test]
    async fn batch_fails_open_on_llm_error() {
        let enhancer = ContextualEnhancer::new(FailingChat);
        let docs = vec![Document::new("chunk a"), Document::new("chunk b")];

        let out = enhancer.enhance_documents(&docs).await;
        assert_eq!(out.len(), 2);
        for (original, enhanced) in docs.iter().zip(out.iter()) {
            assert_eq!(enhanced.content, original.content, "original kept");
            assert!(
                !enhanced.metadata.contains_key(CONTEXTUAL_METADATA_KEY),
                "no context metadata on failure"
            );
        }
    }

    /// Batch enhancement prefixes each chunk and reports the call count.
    #[tokio::test]
    async fn batch_enhances_each_chunk() {
        let mock = MockChat::new("ctx");
        let enhancer = ContextualEnhancer::new(mock.clone());
        let docs = vec![Document::new("a"), Document::new("b"), Document::new("c")];

        let out = enhancer.enhance_documents(&docs).await;
        assert_eq!(out.len(), 3);
        for (original, enhanced) in docs.iter().zip(out.iter()) {
            assert!(enhanced.content.starts_with("ctx"));
            assert!(enhanced.content.ends_with(original.content.as_str()));
            assert!(enhanced.metadata.contains_key(CONTEXTUAL_METADATA_KEY));
        }
        assert_eq!(
            mock.calls.load(Ordering::SeqCst),
            3,
            "one LLM call per chunk"
        );
    }

    /// Idempotency: already-enhanced chunks are skipped (no extra LLM calls).
    #[tokio::test]
    async fn batch_is_idempotent() {
        let mock = MockChat::new("ctx");
        let enhancer = ContextualEnhancer::new(mock.clone());
        let docs = vec![Document::new("a"), Document::new("b")];
        let enhanced = enhancer.enhance_documents(&docs).await;
        assert_eq!(mock.calls.load(Ordering::SeqCst), 2);

        // Re-run over the enhanced corpus: no additional calls, content unchanged.
        let rerun = enhancer.enhance_documents(&enhanced).await;
        assert_eq!(
            mock.calls.load(Ordering::SeqCst),
            2,
            "no new calls on rerun"
        );
        for (first, second) in enhanced.iter().zip(rerun.iter()) {
            assert_eq!(first.content, second.content);
        }
    }

    /// Bounded concurrency: max_concurrency=1 serializes LLM calls.
    #[tokio::test]
    async fn concurrency_is_bounded() {
        #[derive(Default)]
        struct CountingState {
            in_flight: std::sync::Mutex<usize>,
            max_seen: AtomicUsize,
        }

        #[derive(Clone)]
        struct CountingChat {
            state: Arc<CountingState>,
        }

        #[async_trait::async_trait]
        impl Runnable<Vec<Message>, LLMResult> for CountingChat {
            type Error = MockChatError;
            async fn invoke(
                &self,
                _input: Vec<Message>,
                _config: Option<RunnableConfig>,
            ) -> Result<LLMResult, Self::Error> {
                let now = {
                    let mut g = self.state.in_flight.lock().unwrap();
                    *g += 1;
                    let n = *g;
                    if n > self.state.max_seen.load(Ordering::SeqCst) {
                        self.state.max_seen.store(n, Ordering::SeqCst);
                    }
                    n
                };
                // Hold the slot briefly so overlaps are observable.
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                {
                    let mut g = self.state.in_flight.lock().unwrap();
                    *g -= 1;
                }
                Ok(LLMResult {
                    content: format!("ctx-{}", now),
                    model: "counting".to_string(),
                    token_usage: None,
                    tool_calls: None,
                    thinking_content: None,
                })
            }
        }

        #[async_trait::async_trait]
        impl BaseLanguageModel<Vec<Message>, LLMResult> for CountingChat {
            fn model_name(&self) -> &str {
                "counting"
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

        #[async_trait::async_trait]
        impl BaseChatModel for CountingChat {
            async fn chat(
                &self,
                messages: Vec<Message>,
                config: Option<RunnableConfig>,
            ) -> Result<LLMResult, Self::Error> {
                <Self as Runnable<Vec<Message>, LLMResult>>::invoke(self, messages, config).await
            }
            async fn stream_chat(
                &self,
                _messages: Vec<Message>,
                _config: Option<RunnableConfig>,
            ) -> Result<
                Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>,
                Self::Error,
            > {
                Err(MockChatError("no stream".to_string()))
            }
        }

        let state = Arc::new(CountingState::default());
        let enhancer = ContextualEnhancer::new(CountingChat {
            state: state.clone(),
        })
        .with_max_concurrency(1);
        let docs: Vec<Document> = (0..4)
            .map(|i| Document::new(format!("chunk {}", i)))
            .collect();
        let out = enhancer.enhance_documents(&docs).await;
        assert_eq!(out.len(), 4);
        assert!(
            state.max_seen.load(Ordering::SeqCst) <= 1,
            "max_concurrency=1 must serialize calls"
        );
    }
}
