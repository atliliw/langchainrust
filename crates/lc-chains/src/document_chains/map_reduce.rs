// lc-chains/src/document_chains/map_reduce.rs
//! MapReduceDocumentsChain - processes documents in parallel then merges results.

use async_trait::async_trait;
use futures_util::future::try_join_all;
use futures_util::StreamExt;
use futures_util::TryStreamExt;
use lc_core::runnables::RunnableConfig;
use lc_core::BaseChatModel;
use lc_providers::{wrap_chat_model, ProviderError};
use lc_schema::Message;
use lc_shared::document::Document;
use serde_json::Value;
use std::collections::HashMap;

use crate::base::{
    run_chain_with_callbacks, stream_chain_with_callbacks, substitute_template, BaseChain,
    ChainError, ChainResult, ChainStream, StreamToken,
};
use crate::BoxedChatModel;

/// Default Map processing prompt template.
pub(crate) const DEFAULT_MAP_PROMPT: &str = "Answer the user's question based on the following document content. Provide a concise answer based on the document content.

Document content:
{context}

Question: {input}

Answer based on this document:";

/// Default Reduce merge prompt template.
pub(crate) const DEFAULT_REDUCE_PROMPT: &str = "Below are answers from multiple documents. Please merge them into a single complete and coherent final answer.

Answers from each document:
{summaries}

Original question: {input}

Final consolidated answer:";

/// Default prompt used for the *intermediate* combine rounds of a hierarchical
/// reduce (see [`MapReduceDocumentsChain::with_reduce_batch_size`]).
///
/// It is deliberately narrower than [`DEFAULT_REDUCE_PROMPT`]: an intermediate
/// round merges one batch of partial answers into a partial summary, it does not
/// have to produce the final answer. Callers that want a single prompt for every
/// round can pass [`MapReduceDocumentsChain::with_combine_prompt`] the same
/// template as the reduce prompt.
pub(crate) const DEFAULT_COMBINE_PROMPT: &str = "Below are partial answers derived from subsets of the documents. Merge each group into one coherent partial answer, keeping every distinct fact.

Partial answers:
{summaries}

Original question: {input}

Merged partial answer:";

/// MapReduceDocumentsChain
///
/// Processes documents in two steps:
/// 1. Map: Calls LLM independently for each document to generate an answer
/// 2. Reduce: Merges the independent answers into a final answer
///
/// # Hierarchical (multi-level) reduce
///
/// By default the reduce step is a single call that receives every map answer
/// (see [`Self::with_reduce_batch_size`] to change that). With N documents that
/// call's prompt grows linearly with N and eventually overflows the model's
/// context window. Setting a batch size turns the reduce step into a
/// hierarchical reduction: the map answers are merged in batches of at most
/// `reduce_batch_size`, the batch results are merged again, and so on until a
/// single answer remains.
///
/// The reduction stops early — without any extra LLM call — as soon as the
/// working set is down to one item, and [`Self::with_max_reduce_levels`] bounds
/// how many hierarchical rounds may run before a final single combine.
pub struct MapReduceDocumentsChain {
    llm: BoxedChatModel,
    map_prompt_template: String,
    reduce_prompt_template: String,
    /// Prompt for the intermediate combine rounds. Defaults to
    /// [`DEFAULT_COMBINE_PROMPT`]; `None` (only reachable if a future setter
    /// clears it) falls back to the reduce prompt template.
    combine_prompt_template: Option<String>,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
    /// Max number of in-flight map LLM calls; `None` = unbounded (P2-6).
    map_concurrency: Option<usize>,
    /// Max items fed to one reduce call; `None` = one single-level reduce over
    /// every map answer (the pre-hierarchy behavior).
    reduce_batch_size: Option<usize>,
    /// Max number of hierarchical combine rounds; `None` = unbounded.
    max_reduce_levels: Option<usize>,
}

impl MapReduceDocumentsChain {
    /// Create a new [`MapReduceDocumentsChain`] with the given LLM.
    pub fn new<L>(llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: wrap_chat_model(llm),
            map_prompt_template: DEFAULT_MAP_PROMPT.to_string(),
            reduce_prompt_template: DEFAULT_REDUCE_PROMPT.to_string(),
            combine_prompt_template: Some(DEFAULT_COMBINE_PROMPT.to_string()),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "map_reduce_documents".to_string(),
            verbose: false,
            map_concurrency: None,
            reduce_batch_size: None,
            max_reduce_levels: None,
        }
    }

    /// Set the map-phase prompt template.
    pub fn with_map_prompt(mut self, template: impl Into<String>) -> Self {
        self.map_prompt_template = template.into();
        self
    }

    /// Set the reduce-phase prompt template.
    pub fn with_reduce_prompt(mut self, template: impl Into<String>) -> Self {
        self.reduce_prompt_template = template.into();
        self
    }

    /// Set the prompt template for the *intermediate* combine rounds of a
    /// hierarchical reduce.
    ///
    /// Only used when [`Self::with_reduce_batch_size`] is set and more than one
    /// combine round is needed; the round that produces the final answer always
    /// uses the reduce prompt ([`Self::with_reduce_prompt`]). Defaults to
    /// [`DEFAULT_COMBINE_PROMPT`].
    ///
    /// The template receives the same variables as the reduce prompt:
    /// `{summaries}` (the batch's partial answers) and `{input}`.
    pub fn with_combine_prompt(mut self, template: impl Into<String>) -> Self {
        self.combine_prompt_template = Some(template.into());
        self
    }

    /// The template an intermediate combine round uses, falling back to the
    /// reduce template when no combine prompt is configured.
    fn combine_template(&self) -> &str {
        self.combine_prompt_template
            .as_deref()
            .unwrap_or(&self.reduce_prompt_template)
    }

    /// Cap how many map answers one reduce call receives, enabling a
    /// hierarchical (multi-level) reduce.
    ///
    /// - `None` (default): a single reduce call receives every map answer — the
    ///   original single-level behavior, kept for backward compatibility.
    /// - `Some(n)`: the reduce step merges at most `n` items per call and then
    ///   reduces the results again, recursively, until one answer remains.
    ///
    /// A batch size below 2 is clamped to 2 (with a warning): a batch of one
    /// item can never shrink the working set, so it could not terminate. A batch
    /// size at least as wide as the remaining set collapses that round into the
    /// final combine.
    ///
    /// # Example
    ///
    /// With 7 map answers and a batch size of 3 the reduce phase performs
    /// `ceil(7 / 3) = 3` combine calls, then one final reduce over those 3
    /// results — 4 reduce calls in total instead of 1, but no single prompt ever
    /// carries more than 3 answers.
    pub fn with_reduce_batch_size(mut self, batch_size: usize) -> Self {
        self.reduce_batch_size = Some(batch_size);
        self
    }

    /// Bound the number of hierarchical combine rounds.
    ///
    /// After `max_levels` rounds, the remaining items are merged by a single
    /// final reduce call regardless of the batch size, so a pathological batch
    /// size cannot produce an unbounded number of rounds. `None` (default) means
    /// unbounded — the hierarchy always runs until one item remains.
    ///
    /// Has no effect unless [`Self::with_reduce_batch_size`] is also set, since
    /// the single-level reduce already collapses everything in one round.
    pub fn with_max_reduce_levels(mut self, max_levels: usize) -> Self {
        self.max_reduce_levels = Some(max_levels);
        self
    }

    /// Set the document variable name used in the map prompt.
    pub fn with_document_variable(mut self, name: impl Into<String>) -> Self {
        self.document_variable_name = name.into();
        self
    }

    /// Set the input key.
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Set the output key.
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// Set the chain name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Cap the number of concurrent map-phase LLM calls (P2-6).
    ///
    /// Default is unbounded (all documents mapped in parallel). Setting a
    /// limit bounds in-flight requests — useful to respect provider rate
    /// limits or bound memory on large document sets.
    pub fn with_map_concurrency(mut self, limit: usize) -> Self {
        self.map_concurrency = Some(limit);
        self
    }

    /// Build the map-phase prompt by substituting the document content and input.
    pub fn build_map_prompt(&self, context: &str, input: &str) -> String {
        // 0.22.0 audit fix (H-C2): single-pass substitution — document
        // contents containing literal `{input}` are never re-replaced.
        let vars = HashMap::from([
            (self.document_variable_name.clone(), context.to_string()),
            ("input".to_string(), input.to_string()),
        ]);
        substitute_template(&self.map_prompt_template, &vars).0
    }

    /// Build the reduce-phase prompt from the per-document summaries and input.
    pub fn build_reduce_prompt(&self, summaries: &[String], input: &str) -> String {
        self.build_merge_prompt(&self.reduce_prompt_template, summaries, input)
    }

    /// Build an intermediate combine-round prompt from a batch of partial
    /// answers and the original input.
    ///
    /// Uses the template set by [`Self::with_combine_prompt`]
    /// ([`DEFAULT_COMBINE_PROMPT`] unless overridden).
    pub fn build_combine_prompt(&self, summaries: &[String], input: &str) -> String {
        self.build_merge_prompt(self.combine_template(), summaries, input)
    }

    /// Render a merge (reduce / combine) prompt from a template and the items it
    /// must merge.
    ///
    /// 0.22.0 audit fix (H-C2): single-pass substitution — map answers
    /// containing literal `{summaries}` / `{input}` are never re-replaced.
    fn build_merge_prompt(&self, template: &str, summaries: &[String], input: &str) -> String {
        let summaries_text = summaries
            .iter()
            .enumerate()
            .map(|(i, s)| format!("Answer from document {}:\n{}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n\n");

        let vars = HashMap::from([
            ("summaries".to_string(), summaries_text),
            ("input".to_string(), input.to_string()),
        ]);
        substitute_template(template, &vars).0
    }

    /// The effective batch size for one hierarchical reduce round.
    ///
    /// Clamped into `[2, len]` (`len >= 2` at every call site): a batch of one
    /// item maps N items to N items and could never shrink the working set, so
    /// it is raised to 2 with a warning; a batch at least as wide as the
    /// remaining set collapses the round into the final combine.
    fn reduce_batch_size_for(&self, len: usize) -> usize {
        let requested = self.reduce_batch_size.unwrap_or(len);
        if requested < 2 {
            log::warn!(
                "MapReduceDocumentsChain: reduce_batch_size {} cannot shrink the working set; \
                 clamping to 2",
                requested
            );
            return 2.min(len);
        }
        requested.min(len)
    }

    /// Run the hierarchical reduce rounds and return the items the *final*
    /// combine must merge.
    ///
    /// Returns early — with no LLM call — when:
    /// - the working set is down to one item (short-circuit), or
    /// - the next round would be the final combine (the remaining set fits in a
    ///   single batch, or `max_reduce_levels` is exhausted).
    ///
    /// The caller performs that final combine so it can use the reduce prompt
    /// and, on the streaming path, stream it. Every executed round shrinks the
    /// set (`ceil(len / batch) < len` because `batch >= 2`), so this terminates.
    async fn reduce_rounds(
        &self,
        summaries: Vec<String>,
        input: &str,
        config: Option<&RunnableConfig>,
    ) -> Result<Vec<String>, ChainError> {
        let mut current = summaries;
        let mut level = 0usize;

        loop {
            // Short-circuit: a single item needs no further LLM call.
            if current.len() <= 1 {
                return Ok(current);
            }

            let cap_reached = self.max_reduce_levels.is_some_and(|max| level >= max);
            let batch = if cap_reached {
                current.len()
            } else {
                self.reduce_batch_size_for(current.len())
            };
            let chunk_count = current.len().div_ceil(batch);

            // The whole remaining set fits in one call (or the level cap forces
            // a full combine) — that is the final round, not an intermediate one.
            if cap_reached || chunk_count <= 1 {
                return Ok(current);
            }

            let combine_template = self.combine_template();

            if self.verbose {
                println!(
                    "--- Reduce level {}: {} item(s) in {} batch(es) of {} ---",
                    level + 1,
                    current.len(),
                    chunk_count,
                    batch
                );
            }

            let mut next = Vec::with_capacity(chunk_count);
            for chunk in current.chunks(batch) {
                let prompt = self.build_merge_prompt(combine_template, chunk, input);
                let messages = vec![Message::human(&prompt)];
                let response = self.llm.invoke(messages, config.cloned()).await.map_err(|e| {
                    ChainError::ExecutionError(format!(
                        "Reduce level {} call failed: {}",
                        level + 1,
                        e
                    ))
                })?;
                next.push(response.content);
            }

            current = next;
            level += 1;
        }
    }

    /// Reduce phase: hierarchical combine rounds followed by one final reduce.
    ///
    /// With no batch size configured this is exactly one reduce call over every
    /// map answer; with a batch size it is `rounds + 1` calls, and with a single
    /// map answer it is zero calls (the answer is returned as-is).
    async fn reduce_phase(
        &self,
        summaries: Vec<String>,
        input: &str,
        config: Option<&RunnableConfig>,
    ) -> Result<String, ChainError> {
        let mut remaining = self.reduce_rounds(summaries, input, config).await?;

        if remaining.is_empty() {
            return Err(ChainError::ExecutionError(
                "Reduce phase received no summaries".to_string(),
            ));
        }
        if remaining.len() == 1 {
            // Short-circuit: nothing left to merge, so no extra LLM call.
            return Ok(remaining.remove(0));
        }

        if self.verbose {
            println!("--- Final reduce over {} item(s) ---", remaining.len());
        }

        let reduce_prompt = self.build_reduce_prompt(&remaining, input);
        let messages = vec![Message::human(&reduce_prompt)];
        let response = self
            .llm
            .invoke(messages, config.cloned())
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Reduce call failed: {}", e)))?;
        Ok(response.content)
    }

    /// Map phase: call LLM for a single document.
    async fn map_document(
        &self,
        doc: &Document,
        input: &str,
        index: usize,
        config: Option<&RunnableConfig>,
    ) -> Result<String, ChainError> {
        let prompt = self.build_map_prompt(&doc.content, input);

        if self.verbose {
            println!("\n--- Map document {} ---", index + 1);
        }

        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, config.cloned()).await.map_err(|e| {
            ChainError::ExecutionError(format!("Map call failed (document {}): {}", index + 1, e))
        })?;

        if self.verbose {
            println!("Document {} answer: {}", index + 1, response.content);
        }

        Ok(response.content)
    }

    /// Map phase: run the per-document LLM calls concurrently (P2-6).
    ///
    /// Default is unbounded parallelism (`try_join_all`, all documents in
    /// flight at once). With [`Self::with_map_concurrency`] the number of
    /// in-flight calls is capped via `buffer_unordered`, which bounds memory
    /// and lets users respect provider rate limits.
    async fn map_phase(
        &self,
        documents: &[Document],
        input: &str,
        config: Option<&RunnableConfig>,
    ) -> Result<Vec<String>, ChainError> {
        // Materialize the per-document futures first: a `.map()` closure
        // returning an async-fn future cannot express the higher-ranked
        // `for<'a> FnMut((usize, &'a Document))` that `buffer_unordered`
        // requires, so collect into a `Vec` of the concrete future type.
        let mut map_futures = Vec::with_capacity(documents.len());
        for (i, doc) in documents.iter().enumerate() {
            map_futures.push(self.map_document(doc, input, i, config));
        }

        match self.map_concurrency {
            Some(limit) => {
                // 0.22.0 audit fix: `buffer_unordered(0)` panics — clamp a
                // zero/negative-style limit to at least 1 with a warning.
                let limit = if limit == 0 {
                    log::warn!(
                        "MapReduceDocumentsChain: map_concurrency 0 is invalid; clamping to 1"
                    );
                    1
                } else {
                    limit
                };
                futures_util::stream::iter(map_futures)
                    .buffer_unordered(limit)
                    .try_collect()
                    .await
            }
            None => try_join_all(map_futures).await,
        }
    }

    /// Invoke with documents and input directly.
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<String, ChainError> {
        self.invoke_with_documents_cfg(documents, input, None).await
    }

    /// M-20: config-aware variant of [`Self::invoke_with_documents`]; the public
    /// default runs with `None`, while `invoke_inner` threads the caller's
    /// `RunnableConfig` into every map and reduce LLM call.
    async fn invoke_with_documents_cfg(
        &self,
        documents: Vec<Document>,
        input: &str,
        config: Option<&RunnableConfig>,
    ) -> Result<String, ChainError> {
        if documents.is_empty() {
            return Err(ChainError::ExecutionError(
                "Document list is empty".to_string(),
            ));
        }

        if self.verbose {
            println!("\n=== MapReduceDocumentsChain ===");
            println!("Document count: {}", documents.len());
            println!("Input: {}", input);
        }

        if self.verbose {
            println!("\n--- Map phase ---");
        }

        let summaries = self.map_phase(&documents, input, config).await?;

        if self.verbose {
            println!("\n--- Reduce phase ---");
            println!("Merging answers from {} documents", summaries.len());
        }

        let final_answer = self.reduce_phase(summaries, input, config).await?;

        if self.verbose {
            println!("Final answer: {}", final_answer);
            println!("=== MapReduceDocumentsChain complete ===\n");
        }

        Ok(final_answer)
    }

    /// M-20: the config-aware invoke body. Validates inputs, extracts the
    /// documents, then delegates to the config-threading document pipeline so a
    /// caller's `RunnableConfig` reaches every map and reduce LLM call.
    async fn invoke_inner(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<&RunnableConfig>,
    ) -> Result<ChainResult, ChainError> {
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents = crate::base::documents_from_input(inputs.get("documents"))?;

        let output = self
            .invoke_with_documents_cfg(documents, input, config)
            .await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));
        Ok(result)
    }

    /// M-20: the config-aware streaming body. The map phase runs via the
    /// config-aware document pipeline (parallel, non-streaming), and the reduce
    /// phase streams via `stream_chat` with `config` threading.
    async fn stream_inner(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<&RunnableConfig>,
    ) -> Result<ChainStream, ChainError> {
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents = crate::base::documents_from_input(inputs.get("documents"))?;

        if documents.is_empty() {
            return Err(ChainError::ExecutionError(
                "Document list is empty".to_string(),
            ));
        }

        // Map phase: run all map calls in parallel (non-streaming), bounded by
        // `map_concurrency` when configured (P2-6).
        let summaries = self.map_phase(&documents, input, config).await?;

        // Reduce phase: hierarchical rounds first (a round is a fan-out of
        // combine calls, so it cannot stream), then stream the final combine.
        let mut remaining = self.reduce_rounds(summaries, input, config).await?;

        if remaining.is_empty() {
            return Err(ChainError::ExecutionError(
                "Reduce phase received no summaries".to_string(),
            ));
        }
        if remaining.len() == 1 {
            // Short-circuit: the hierarchy already collapsed to one answer, so
            // emit it without a further LLM call.
            let text = remaining.remove(0);
            let stream = futures_util::stream::once(async move {
                Ok(StreamToken {
                    token: text,
                    is_final: true,
                })
            });
            return Ok(Box::pin(stream));
        }

        let reduce_prompt = self.build_reduce_prompt(&remaining, input);
        let messages = vec![Message::human(&reduce_prompt)];

        let llm_stream = self
            .llm
            .stream_chat(messages, config.cloned())
            .await
            .map_err(|e| ChainError::StreamError(format!("LLM stream failed: {}", e)))?;

        let stream = llm_stream.map(|result| match result {
            Ok(chunk) => Ok(StreamToken {
                token: chunk.text,
                is_final: false,
            }),
            Err(e) => Err(ChainError::StreamError(format!(
                "Stream token error: {}",
                e
            ))),
        });

        let final_stream = stream.chain(futures_util::stream::once(async move {
            Ok(StreamToken {
                token: String::new(),
                is_final: true,
            })
        }));

        Ok(Box::pin(final_stream))
    }
}

#[async_trait]
impl BaseChain for MapReduceDocumentsChain {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key, "documents"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    /// Config-less invoke: chain-callback dispatch is skipped, so the LLM call
    /// runs with no config (matching the permissive direct-`invoke` contract).
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.invoke_inner(inputs, None).await
    }

    /// M-20: invoke with the caller's `RunnableConfig`, dispatching chain-level
    /// callbacks (like the default) AND threading `config` into the LLM calls so
    /// sampling / cancellation / LLM callbacks reach the provider.
    async fn invoke_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainResult, ChainError> {
        let inner = config.clone();
        run_chain_with_callbacks(self.name(), inputs, config, |inputs| async move {
            self.invoke_inner(inputs, inner.as_ref()).await
        })
        .await
    }

    /// Stream without a config (no callback dispatch, no LLM sampling overrides).
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        self.stream_inner(inputs, None).await
    }

    /// M-20: stream with the caller's `RunnableConfig`, dispatching chain-level
    /// callbacks (like the default) AND threading `config` into the LLM stream.
    async fn stream_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainStream, ChainError> {
        let inner = config.clone();
        let output_key = Some(self.output_key.clone());
        stream_chain_with_callbacks(self.name(), inputs, config, output_key, |inputs| async move {
            self.stream_inner(inputs, inner.as_ref()).await
        })
        .await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::language_models::{LLMResult, StreamChunk};
    use lc_core::runnables::RunnableConfig;
    use lc_core::{BaseLanguageModel, Runnable};
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Tracking chat model that counts `invoke` calls and records the maximum
    /// number of in-flight invokes, so the map phase's parallelism and the
    /// `with_map_concurrency` cap are provable rather than assumed. Each invoke
    /// yields repeatedly so concurrently-polled futures genuinely overlap.
    struct TrackingLLM {
        invokes: Arc<AtomicUsize>,
        in_flight: Arc<AtomicUsize>,
        max_in_flight: Arc<AtomicUsize>,
    }

    impl TrackingLLM {
        fn counters() -> (Arc<AtomicUsize>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
            (
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
            )
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for TrackingLLM {
        type Error = ProviderError;
        async fn invoke(
            &self,
            input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let cur = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(cur, Ordering::SeqCst);

            // Yield repeatedly so every concurrently-polled future is actually
            // scheduled before any of them returns.
            for _ in 0..32 {
                tokio::task::yield_now().await;
            }

            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            self.invokes.fetch_add(1, Ordering::SeqCst);

            let is_reduce = input
                .iter()
                .any(|m| m.content.contains("Below are answers"));
            let content = if is_reduce {
                "final merged answer".to_string()
            } else {
                "map answer".to_string()
            };
            Ok(LLMResult {
                content,
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for TrackingLLM {
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
    impl BaseChatModel for TrackingLLM {
        async fn chat(
            &self,
            messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            // Same counting behavior as invoke so the stream path's map phase
            // is observably identical.
            self.invoke(messages, None).await
        }
        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            let tokens = [Ok(StreamChunk::new("streamed token"))];
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
    }

    fn docs(n: usize) -> Vec<Document> {
        (0..n).map(|i| Document::new(format!("doc {i}"))).collect()
    }

    /// P2-6: the map phase is genuinely parallel — with 6 documents the map
    /// LLM calls overlap in flight (max in-flight > 1), then a single reduce
    /// call produces the final answer.
    #[tokio::test]
    async fn test_map_reduce_map_phase_is_parallel() {
        let (invokes, in_flight, max_in_flight) = TrackingLLM::counters();
        let chain = MapReduceDocumentsChain::new(TrackingLLM {
            invokes: invokes.clone(),
            in_flight,
            max_in_flight: max_in_flight.clone(),
        });

        let out = chain
            .invoke_with_documents(docs(6), "question")
            .await
            .unwrap();
        assert_eq!(out, "final merged answer");
        assert_eq!(invokes.load(Ordering::SeqCst), 7, "6 map + 1 reduce");
        assert!(
            max_in_flight.load(Ordering::SeqCst) > 1,
            "map phase must overlap calls, got max in-flight {}",
            max_in_flight.load(Ordering::SeqCst)
        );
    }

    /// P2-6: `with_map_concurrency(2)` caps in-flight map calls at 2 while
    /// still running concurrently (max in-flight exactly 2).
    #[tokio::test]
    async fn test_map_reduce_concurrency_limit_caps_in_flight() {
        let (invokes, in_flight, max_in_flight) = TrackingLLM::counters();
        let chain = MapReduceDocumentsChain::new(TrackingLLM {
            invokes: invokes.clone(),
            in_flight,
            max_in_flight: max_in_flight.clone(),
        })
        .with_map_concurrency(2);

        let out = chain
            .invoke_with_documents(docs(6), "question")
            .await
            .unwrap();
        assert_eq!(out, "final merged answer");
        assert_eq!(invokes.load(Ordering::SeqCst), 7);
        assert_eq!(
            max_in_flight.load(Ordering::SeqCst),
            2,
            "concurrency cap 2 must bound in-flight map calls"
        );
    }

    /// P2-6: empty documents error loudly before any LLM call.
    #[tokio::test]
    async fn test_map_reduce_empty_documents() {
        let (invokes, in_flight, max_in_flight) = TrackingLLM::counters();
        let chain = MapReduceDocumentsChain::new(TrackingLLM {
            invokes: invokes.clone(),
            in_flight,
            max_in_flight,
        });
        let err = match chain.invoke_with_documents(vec![], "q").await {
            Ok(_) => panic!("expected an execution error"),
            Err(e) => e,
        };
        assert!(matches!(err, ChainError::ExecutionError(_)));
        assert_eq!(invokes.load(Ordering::SeqCst), 0);
    }

    // --- Hierarchical (multi-level) reduce ---------------------------------

    /// Fake chat model that makes the reduce hierarchy *observable* through its
    /// own output: a map prompt yields `M(<document content>)`, and a
    /// combine/reduce prompt yields `C(<item>+<item>+...)` over exactly the
    /// summaries that prompt carried. The returned string therefore encodes the
    /// whole reduction tree, and `invokes` counts every LLM call — so both the
    /// number of rounds and the batch grouping are asserted, not assumed.
    struct HierarchyLLM {
        invokes: Arc<AtomicUsize>,
        prompts: Arc<Mutex<Vec<String>>>,
    }

    impl HierarchyLLM {
        fn new() -> (Self, Arc<AtomicUsize>, Arc<Mutex<Vec<String>>>) {
            let invokes = Arc::new(AtomicUsize::new(0));
            let prompts = Arc::new(Mutex::new(Vec::new()));
            let llm = Self {
                invokes: Arc::clone(&invokes),
                prompts: Arc::clone(&prompts),
            };
            (llm, invokes, prompts)
        }
    }

    /// The document content a *map* prompt carries, or `None` for a merge prompt.
    fn map_prompt_document(prompt: &str) -> Option<&str> {
        const MARKER: &str = "Document content:\n";
        let start = prompt.find(MARKER)? + MARKER.len();
        let rest = &prompt[start..];
        let end = rest.find("\n\nQuestion:")?;
        Some(&rest[..end])
    }

    /// The summaries a merge (reduce/combine) prompt carried, in prompt order.
    fn merge_prompt_items(prompt: &str) -> Vec<&str> {
        prompt
            .lines()
            .filter(|line| line.starts_with("M(") || line.starts_with("C("))
            .collect()
    }

    /// Deterministic reply for a prompt — shared by `invoke` and `stream_chat`
    /// so the two paths produce identical text.
    fn hierarchy_reply(prompt: &str) -> String {
        match map_prompt_document(prompt) {
            Some(document) => format!("M({document})"),
            None => format!("C({})", merge_prompt_items(prompt).join("+")),
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for HierarchyLLM {
        type Error = ProviderError;
        async fn invoke(
            &self,
            input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invokes.fetch_add(1, Ordering::SeqCst);
            let prompt = input.last().map(|m| m.content.clone()).unwrap_or_default();
            self.prompts
                .lock()
                .expect("prompt log mutex must not be poisoned")
                .push(prompt.clone());
            Ok(LLMResult {
                content: hierarchy_reply(&prompt),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for HierarchyLLM {
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
    impl BaseChatModel for HierarchyLLM {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invoke(messages, config).await
        }
        async fn stream_chat(
            &self,
            messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            // Not counted as an `invoke`: the streaming path drives the final
            // combine through `stream_chat`, which the tests assert separately.
            let prompt = messages.last().map(|m| m.content.clone()).unwrap_or_default();
            let tokens = [Ok(StreamChunk::new(hierarchy_reply(&prompt)))];
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
    }

    /// 4 documents with a batch size of 2: level 1 merges (0,1) and (2,3) in two
    /// calls, then the final reduce merges the two partial answers — 4 map + 2
    /// combine + 1 reduce = 7 LLM calls, and no prompt ever carries more than
    /// two summaries.
    #[tokio::test]
    async fn test_map_reduce_hierarchy_batches_and_levels() {
        let (llm, invokes, _) = HierarchyLLM::new();
        let chain = MapReduceDocumentsChain::new(llm).with_reduce_batch_size(2);

        let out = chain
            .invoke_with_documents(docs(4), "question")
            .await
            .unwrap();

        assert_eq!(out, "C(C(M(doc 0)+M(doc 1))+C(M(doc 2)+M(doc 3)))");
        assert_eq!(
            invokes.load(Ordering::SeqCst),
            7,
            "4 map + 2 level-1 combines + 1 final reduce"
        );
    }

    /// 7 documents with a batch size of 3: the level-1 batches are [0,1,2],
    /// [3,4,5] and the remainder [6] — the remainder is still combined (a
    /// deliberate call, not a silent drop), then one final reduce merges the 3
    /// partial answers. 7 map + 3 combine + 1 reduce = 11.
    #[tokio::test]
    async fn test_map_reduce_hierarchy_batch_grouping_with_remainder() {
        let (llm, invokes, _) = HierarchyLLM::new();
        let chain = MapReduceDocumentsChain::new(llm).with_reduce_batch_size(3);

        let out = chain
            .invoke_with_documents(docs(7), "question")
            .await
            .unwrap();

        assert_eq!(
            out,
            "C(C(M(doc 0)+M(doc 1)+M(doc 2))+C(M(doc 3)+M(doc 4)+M(doc 5))+C(M(doc 6)))"
        );
        assert_eq!(invokes.load(Ordering::SeqCst), 11);
    }

    /// A single map answer short-circuits the whole reduce phase: the answer is
    /// returned verbatim with no further LLM call (not even one "combine" call).
    #[tokio::test]
    async fn test_map_reduce_single_summary_short_circuits_reduce() {
        let (llm, invokes, prompts) = HierarchyLLM::new();
        let chain = MapReduceDocumentsChain::new(llm).with_reduce_batch_size(2);

        let out = chain
            .invoke_with_documents(docs(1), "question")
            .await
            .unwrap();

        assert_eq!(out, "M(doc 0)", "the single map answer is the final answer");
        assert_eq!(
            invokes.load(Ordering::SeqCst),
            1,
            "one map call only — the reduce step must not call the model"
        );
        assert_eq!(prompts.lock().unwrap().len(), 1);
    }

    /// `with_max_reduce_levels(1)` allows exactly one hierarchical round, then
    /// forces a single full combine of whatever is left: 8 map + 4 combine + 1
    /// final = 13 calls, and the final prompt carries all 4 partial answers.
    #[tokio::test]
    async fn test_map_reduce_max_reduce_levels_caps_hierarchy() {
        let (llm, invokes, _) = HierarchyLLM::new();
        let chain = MapReduceDocumentsChain::new(llm)
            .with_reduce_batch_size(2)
            .with_max_reduce_levels(1);

        let out = chain
            .invoke_with_documents(docs(8), "question")
            .await
            .unwrap();

        assert_eq!(
            out,
            "C(C(M(doc 0)+M(doc 1))+C(M(doc 2)+M(doc 3))+C(M(doc 4)+M(doc 5))+C(M(doc 6)+M(doc 7)))"
        );
        assert_eq!(invokes.load(Ordering::SeqCst), 13);
    }

    /// Backward compatibility: with no batch size configured the reduce step is
    /// still exactly one call that receives every map answer.
    #[tokio::test]
    async fn test_map_reduce_default_stays_single_level() {
        let (llm, invokes, prompts) = HierarchyLLM::new();
        let chain = MapReduceDocumentsChain::new(llm);

        let out = chain
            .invoke_with_documents(docs(4), "question")
            .await
            .unwrap();

        assert_eq!(out, "C(M(doc 0)+M(doc 1)+M(doc 2)+M(doc 3))");
        assert_eq!(invokes.load(Ordering::SeqCst), 5, "4 map + 1 reduce");
        let prompts = prompts.lock().unwrap();
        assert_eq!(prompts.len(), 5);
        assert!(
            prompts[4].contains("Below are answers from multiple documents"),
            "the single reduce call must use the reduce prompt"
        );
    }

    /// A batch size below 2 cannot shrink the working set, so it is clamped to 2
    /// (with a warning) instead of looping forever: the result is identical to
    /// an explicit batch size of 2.
    #[tokio::test]
    async fn test_map_reduce_reduce_batch_size_below_two_is_clamped() {
        let (llm, invokes, _) = HierarchyLLM::new();
        let chain = MapReduceDocumentsChain::new(llm).with_reduce_batch_size(1);

        let out = chain
            .invoke_with_documents(docs(4), "question")
            .await
            .unwrap();

        assert_eq!(out, "C(C(M(doc 0)+M(doc 1))+C(M(doc 2)+M(doc 3)))");
        assert_eq!(invokes.load(Ordering::SeqCst), 7);
    }

    /// A custom combine prompt drives the intermediate rounds only; the final
    /// round still uses the reduce prompt.
    #[tokio::test]
    async fn test_map_reduce_custom_combine_prompt_used_for_intermediate_rounds() {
        let (llm, _, prompts) = HierarchyLLM::new();
        let chain = MapReduceDocumentsChain::new(llm)
            .with_reduce_batch_size(3)
            .with_combine_prompt("MERGE-GROUP:\n{summaries}\nQ: {input}");

        let out = chain
            .invoke_with_documents(docs(7), "question")
            .await
            .unwrap();
        assert_eq!(
            out,
            "C(C(M(doc 0)+M(doc 1)+M(doc 2))+C(M(doc 3)+M(doc 4)+M(doc 5))+C(M(doc 6)))"
        );

        let prompts = prompts.lock().unwrap();
        let combine_calls = prompts
            .iter()
            .filter(|p| p.contains("MERGE-GROUP:"))
            .count();
        let reduce_calls = prompts
            .iter()
            .filter(|p| p.contains("Below are answers from multiple documents"))
            .count();
        assert_eq!(combine_calls, 3, "3 intermediate combine calls");
        assert_eq!(reduce_calls, 1, "exactly one final reduce call");
    }

    /// `build_combine_prompt` falls back to the reduce template when no custom
    /// combine prompt is configured, and renders `{summaries}` / `{input}`.
    #[test]
    fn test_build_combine_prompt_renders_batch() {
        let (llm, _, _) = HierarchyLLM::new();
        let chain = MapReduceDocumentsChain::new(llm);

        let rendered = chain.build_combine_prompt(&["a".to_string(), "b".to_string()], "q");
        assert!(rendered.contains("Answer from document 1:\na"));
        assert!(rendered.contains("Answer from document 2:\nb"));
        assert!(rendered.contains("Original question: q"));
        assert!(!rendered.contains("{summaries}"));
        assert!(!rendered.contains("{input}"));
    }

    /// Streaming path with a single document: the map answer is emitted as the
    /// only token and the model is called exactly once (the reduce step
    /// short-circuits instead of streaming an empty combine).
    #[tokio::test]
    async fn test_map_reduce_stream_single_summary_short_circuits() {
        let (llm, invokes, _) = HierarchyLLM::new();
        let chain = MapReduceDocumentsChain::new(llm).with_reduce_batch_size(2);

        let docs_value = serde_json::to_value(docs(1)).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Value::String("q".to_string()));
        inputs.insert("documents".to_string(), docs_value);

        let mut stream = chain.stream(inputs).await.unwrap();
        let mut tokens = Vec::new();
        while let Some(item) = stream.next().await {
            tokens.push(item.unwrap());
        }
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].is_final);
        assert_eq!(tokens[0].token, "M(doc 0)");
        assert_eq!(invokes.load(Ordering::SeqCst), 1);
    }

    /// Streaming path with a hierarchy: the intermediate rounds run as plain
    /// calls (6 invokes for 4 documents at batch size 2) and only the *final*
    /// combine is streamed, so the emitted token is the fully reduced answer.
    #[tokio::test]
    async fn test_map_reduce_stream_hierarchy_streams_final_combine() {
        let (llm, invokes, _) = HierarchyLLM::new();
        let chain = MapReduceDocumentsChain::new(llm).with_reduce_batch_size(2);

        let docs_value = serde_json::to_value(docs(4)).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Value::String("q".to_string()));
        inputs.insert("documents".to_string(), docs_value);

        let mut stream = chain.stream(inputs).await.unwrap();
        let mut tokens = Vec::new();
        while let Some(item) = stream.next().await {
            tokens.push(item.unwrap());
        }
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].is_final);
        assert_eq!(
            tokens[0].token,
            "C(C(M(doc 0)+M(doc 1))+C(M(doc 2)+M(doc 3)))"
        );
        assert_eq!(
            invokes.load(Ordering::SeqCst),
            6,
            "4 map + 2 combine invokes; the final combine is streamed, not invoked"
        );
    }
}
