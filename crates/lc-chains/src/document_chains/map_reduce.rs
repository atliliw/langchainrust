// lc-chains/src/document_chains/map_reduce.rs
//! MapReduceDocumentsChain - processes documents in parallel then merges results.

use async_trait::async_trait;
use futures_util::future::try_join_all;
use futures_util::StreamExt;
use futures_util::TryStreamExt;
use lc_core::language_models::LLMResult;
use lc_core::{BaseChatModel, Runnable};
use lc_schema::Message;
use lc_shared::document::Document;
use serde_json::Value;
use std::collections::HashMap;

use crate::base::{BaseChain, ChainError, ChainResult, ChainStream, StreamToken};

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

/// MapReduceDocumentsChain
///
/// Processes documents in two steps:
/// 1. Map: Calls LLM independently for each document to generate an answer
/// 2. Reduce: Merges all independent answers into a final answer
pub struct MapReduceDocumentsChain<M: BaseChatModel> {
    llm: M,
    map_prompt_template: String,
    reduce_prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
    /// Max number of in-flight map LLM calls; `None` = unbounded (P2-6).
    map_concurrency: Option<usize>,
}

impl<M: BaseChatModel> MapReduceDocumentsChain<M> {
    pub fn new(llm: M) -> Self {
        Self {
            llm,
            map_prompt_template: DEFAULT_MAP_PROMPT.to_string(),
            reduce_prompt_template: DEFAULT_REDUCE_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "map_reduce_documents".to_string(),
            verbose: false,
            map_concurrency: None,
        }
    }

    pub fn with_map_prompt(mut self, template: impl Into<String>) -> Self {
        self.map_prompt_template = template.into();
        self
    }

    pub fn with_reduce_prompt(mut self, template: impl Into<String>) -> Self {
        self.reduce_prompt_template = template.into();
        self
    }

    pub fn with_document_variable(mut self, name: impl Into<String>) -> Self {
        self.document_variable_name = name.into();
        self
    }

    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

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

    pub fn build_map_prompt(&self, context: &str, input: &str) -> String {
        self.map_prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
    }

    pub fn build_reduce_prompt(&self, summaries: &[String], input: &str) -> String {
        let summaries_text = summaries
            .iter()
            .enumerate()
            .map(|(i, s)| format!("Answer from document {}:\n{}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n\n");

        self.reduce_prompt_template
            .replace("{summaries}", &summaries_text)
            .replace("{input}", input)
    }

    /// Map phase: call LLM for a single document.
    async fn map_document(
        &self,
        doc: &Document,
        input: &str,
        index: usize,
    ) -> Result<String, ChainError>
    where
        <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
    {
        let prompt = self.build_map_prompt(&doc.content, input);

        if self.verbose {
            println!("\n--- Map document {} ---", index + 1);
        }

        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await.map_err(|e| {
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
    ) -> Result<Vec<String>, ChainError>
    where
        <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
    {
        // Materialize the per-document futures first: a `.map()` closure
        // returning an async-fn future cannot express the higher-ranked
        // `for<'a> FnMut((usize, &'a Document))` that `buffer_unordered`
        // requires, so collect into a `Vec` of the concrete future type.
        let mut map_futures = Vec::with_capacity(documents.len());
        for (i, doc) in documents.iter().enumerate() {
            map_futures.push(self.map_document(doc, input, i));
        }

        match self.map_concurrency {
            Some(limit) => {
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
    ) -> Result<String, ChainError>
    where
        <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
    {
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

        let summaries = self.map_phase(&documents, input).await?;

        if self.verbose {
            println!("\n--- Reduce phase ---");
        }

        let reduce_prompt = self.build_reduce_prompt(&summaries, input);

        if self.verbose {
            println!("Merging answers from {} documents", summaries.len());
        }

        let messages = vec![Message::human(&reduce_prompt)];
        let response = self
            .llm
            .invoke(messages, None)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Reduce call failed: {}", e)))?;

        let final_answer = response.content;

        if self.verbose {
            println!("Final answer: {}", final_answer);
            println!("=== MapReduceDocumentsChain complete ===\n");
        }

        Ok(final_answer)
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseChain for MapReduceDocumentsChain<M>
where
    <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
{
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key, "documents"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        // P2-8: validate inputs on the invoke path too, matching stream.
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents = crate::base::documents_from_input(inputs.get("documents"))?;

        let output = self.invoke_with_documents(documents, input).await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));
        Ok(result)
    }

    /// Stream execution for MapReduceDocumentsChain.
    ///
    /// The map phase runs via invoke (parallel, non-streaming, since reduce
    /// needs all summaries) — bounded by `map_concurrency` when configured.
    /// The reduce phase is streamed token by token via `stream_chat`.
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
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
        let summaries = self.map_phase(&documents, input).await?;

        // Reduce phase: stream the final merged answer
        let reduce_prompt = self.build_reduce_prompt(&summaries, input);
        let messages = vec![Message::human(&reduce_prompt)];

        let llm_stream = self
            .llm
            .stream_chat(messages, None)
            .await
            .map_err(|e| ChainError::StreamError(format!("LLM stream failed: {}", e)))?;

        let stream = llm_stream.map(|result| match result {
            Ok(token) => Ok(StreamToken {
                token,
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

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::runnables::RunnableConfig;
    use lc_core::{BaseLanguageModel, Runnable};
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Debug)]
    struct MockError(String);
    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::error::Error for MockError {}

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
        type Error = MockError;
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
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
        {
            let tokens = [Ok("streamed token".to_string())];
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
}
