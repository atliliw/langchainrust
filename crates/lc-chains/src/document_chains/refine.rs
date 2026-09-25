// lc-chains/src/document_chains/refine.rs
//! RefineDocumentsChain - iteratively refines the answer document by document.

use async_trait::async_trait;
use futures_util::StreamExt;
use lc_core::runnables::RunnableConfig;
use lc_core::BaseChatModel;
use lc_providers::{wrap_chat_model, ProviderError};
use lc_schema::Message;
use lc_shared::document::Document;
use serde_json::Value;
use std::collections::HashMap;

use crate::base::{
    run_chain_with_callbacks, stream_chain_with_callbacks, BaseChain, ChainError, ChainResult,
    ChainStream, StreamToken,
};
use crate::BoxedChatModel;

/// Default initial processing prompt template.
pub(crate) const DEFAULT_REFINE_INITIAL_PROMPT: &str =
    "Answer the question based on the following reference information.

Reference information:
{context}

Question: {input}

Answer:";

/// Default iterative refinement prompt template.
pub(crate) const DEFAULT_REFINE_PROMPT: &str = "You have provided an answer based on partial information. Here is additional reference information.

Existing answer:
{existing_answer}

New reference information:
{context}

Please refine or modify your answer based on the new information. If the new information does not conflict with the existing answer, merge them. If the new information conflicts with the existing answer, prioritize the new information.

Question: {input}

Refined answer:";

/// RefineDocumentsChain
///
/// Iteratively refines the answer document by document.
/// Generates an initial answer from the first document, then refines with each subsequent document.
pub struct RefineDocumentsChain {
    llm: BoxedChatModel,
    initial_prompt_template: String,
    refine_prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
}

impl RefineDocumentsChain {
    /// Create a new [`RefineDocumentsChain`] with the given LLM.
    pub fn new<L>(llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: wrap_chat_model(llm),
            initial_prompt_template: DEFAULT_REFINE_INITIAL_PROMPT.to_string(),
            refine_prompt_template: DEFAULT_REFINE_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "refine_documents".to_string(),
            verbose: false,
        }
    }

    /// Set the initial prompt template used for the first document.
    pub fn with_initial_prompt(mut self, template: impl Into<String>) -> Self {
        self.initial_prompt_template = template.into();
        self
    }

    /// Set the refine prompt template used for subsequent documents.
    pub fn with_refine_prompt(mut self, template: impl Into<String>) -> Self {
        self.refine_prompt_template = template.into();
        self
    }

    /// Set the document variable name used in the prompts.
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

    /// Build the initial prompt from the first document's content and input.
    pub fn build_initial_prompt(&self, context: &str, input: &str) -> String {
        let vars = HashMap::from([
            (self.document_variable_name.clone(), context.to_string()),
            ("input".to_string(), input.to_string()),
        ]);
        crate::base::substitute_template(&self.initial_prompt_template, &vars).0
    }

    /// Build the refine prompt from the new context, input, and existing answer.
    pub fn build_refine_prompt(&self, context: &str, input: &str, existing_answer: &str) -> String {
        let vars = HashMap::from([
            (self.document_variable_name.clone(), context.to_string()),
            ("input".to_string(), input.to_string()),
            ("existing_answer".to_string(), existing_answer.to_string()),
        ]);
        crate::base::substitute_template(&self.refine_prompt_template, &vars).0
    }

    /// Invoke with documents and input directly (iterative refinement).
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<String, ChainError> {
        self.invoke_with_documents_cfg(documents, input, None)
            .await
    }

    /// M-20: config-aware variant of [`Self::invoke_with_documents`]; the public
    /// default runs with `None`, while `invoke_inner` threads the caller's
    /// `RunnableConfig` into every initial and refinement LLM call.
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
            println!("\n=== RefineDocumentsChain ===");
            println!("Document count: {}", documents.len());
            println!("Input: {}", input);
        }

        // Step 1: Generate initial answer from the first document
        let first_context = &documents[0].content;
        let initial_prompt = self.build_initial_prompt(first_context, input);

        if self.verbose {
            println!("\n--- Initial processing (document 1) ---");
        }

        let messages = vec![Message::human(&initial_prompt)];
        let response =
            self.llm.invoke(messages, config.cloned()).await.map_err(|e| {
                ChainError::ExecutionError(format!("LLM initial call failed: {}", e))
            })?;
        let mut answer = response.content;

        if self.verbose {
            println!("Initial answer: {}", answer);
        }

        // Subsequent steps: iteratively refine with remaining documents
        for (i, doc) in documents[1..].iter().enumerate() {
            if self.verbose {
                println!("\n--- Refinement step {} (document {}) ---", i + 1, i + 2);
            }

            let refine_prompt = self.build_refine_prompt(&doc.content, input, &answer);

            let messages = vec![Message::human(&refine_prompt)];
            let response = self.llm.invoke(messages, config.cloned()).await.map_err(|e| {
                ChainError::ExecutionError(format!("LLM refinement call failed: {}", e))
            })?;
            answer = response.content;

            if self.verbose {
                println!("Refined answer: {}", answer);
            }
        }

        if self.verbose {
            println!("=== RefineDocumentsChain complete ===\n");
        }

        Ok(answer)
    }

    /// M-20: the config-aware invoke body. Validates inputs, extracts the
    /// documents, then delegates to the config-threading document pipeline so a
    /// caller's `RunnableConfig` reaches every initial and refinement LLM call.
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

    /// M-20: the config-aware streaming body. Runs the initial + all intermediate
    /// refine steps via invoke (since their output feeds the next step), then
    /// streams the final refine step via `stream_chat`, threading `config` into
    /// every LLM boundary.
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

        // Step 1: Generate initial answer from the first document
        let first_context = &documents[0].content;
        let initial_prompt = self.build_initial_prompt(first_context, input);
        let messages = vec![Message::human(&initial_prompt)];
        let response =
            self.llm.invoke(messages, config.cloned()).await.map_err(|e| {
                ChainError::ExecutionError(format!("LLM initial call failed: {}", e))
            })?;
        let mut answer = response.content;

        // Step 2: Run intermediate refine steps (all but the last) via invoke
        //
        // P2-4: with a single document `last_idx == 0` and `documents[1..0]`
        // would panic on the slice index — iterate with skip/take so zero
        // intermediate documents (1 or 2 documents) is a no-op and the initial
        // answer flows straight to the final step.
        let last_idx = documents.len() - 1;
        for (i, doc) in documents
            .iter()
            .skip(1)
            .take(last_idx.saturating_sub(1))
            .enumerate()
        {
            let refine_prompt = self.build_refine_prompt(&doc.content, input, &answer);
            let messages = vec![Message::human(&refine_prompt)];
            let response = self.llm.invoke(messages, config.cloned()).await.map_err(|e| {
                ChainError::ExecutionError(format!("LLM refinement call failed: {}", e))
            })?;
            answer = response.content;

            if self.verbose {
                println!("Refine step {} completed", i + 1);
            }
        }

        // Step 3: Stream the final refine step
        //
        // P2-4: with a single document the initial invoke above already produced
        // the complete answer — calling `stream_chat` on the identical initial
        // prompt would make a second, redundant LLM call for the same output
        // (the invoke result was previously discarded and the prompt re-sent).
        // Stream the computed answer directly instead.
        if last_idx == 0 {
            let stream = futures_util::stream::once(async move {
                Ok(StreamToken {
                    token: answer,
                    is_final: true,
                })
            });
            return Ok(Box::pin(stream));
        }

        let final_prompt = self.build_refine_prompt(&documents[last_idx].content, input, &answer);

        let messages = vec![Message::human(&final_prompt)];
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
impl BaseChain for RefineDocumentsChain {
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
    use std::sync::Arc;

    /// Mock chat model that counts `invoke`/`stream_chat` calls so the P2-4
    /// single-document fix (no second LLM call) is provable.
    struct CountingLLM {
        invokes: Arc<AtomicUsize>,
        streams: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for CountingLLM {
        type Error = ProviderError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invokes.fetch_add(1, Ordering::SeqCst);
            Ok(LLMResult {
                content: "initial answer".to_string(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for CountingLLM {
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
    impl BaseChatModel for CountingLLM {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invokes.fetch_add(1, Ordering::SeqCst);
            Ok(LLMResult {
                content: "initial answer".to_string(),
                model: "mock".to_string(),
                token_usage: None,
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
            self.streams.fetch_add(1, Ordering::SeqCst);
            let tokens = [Ok(StreamChunk::new("refined answer"))];
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
    }

    fn inputs_for(documents: Vec<Document>) -> HashMap<String, Value> {
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Value::String("question".to_string()));
        inputs.insert(
            "documents".to_string(),
            serde_json::to_value(documents).unwrap(),
        );
        inputs
    }

    /// P2-4: a single document reuses the invoke-computed initial answer —
    /// `stream_chat` is never called on the identical prompt (one LLM call
    /// total, not two).
    #[tokio::test]
    async fn test_refine_stream_single_document_skips_second_llm_call() {
        let invokes = Arc::new(AtomicUsize::new(0));
        let streams = Arc::new(AtomicUsize::new(0));
        let chain = RefineDocumentsChain::new(CountingLLM {
            invokes: invokes.clone(),
            streams: streams.clone(),
        });
        let inputs = inputs_for(vec![Document::new("doc one")]);

        let mut stream = chain.stream(inputs).await.unwrap();
        let mut tokens = Vec::new();
        while let Some(item) = stream.next().await {
            tokens.push(item.unwrap());
        }
        let text: String = tokens.iter().map(|t| t.token.as_str()).collect();
        assert_eq!(text, "initial answer");
        assert!(tokens.last().unwrap().is_final);
        assert_eq!(invokes.load(Ordering::SeqCst), 1, "one initial invoke");
        assert_eq!(
            streams.load(Ordering::SeqCst),
            0,
            "single-document stream must not re-call the LLM"
        );
    }

    /// Multi-document: initial + intermediate refines run via invoke, the final
    /// refine is genuinely streamed (one `stream_chat` call).
    #[tokio::test]
    async fn test_refine_stream_multi_document_streams_final_refine() {
        let invokes = Arc::new(AtomicUsize::new(0));
        let streams = Arc::new(AtomicUsize::new(0));
        let chain = RefineDocumentsChain::new(CountingLLM {
            invokes: invokes.clone(),
            streams: streams.clone(),
        });
        let inputs = inputs_for(vec![
            Document::new("doc one"),
            Document::new("doc two"),
            Document::new("doc three"),
        ]);

        let mut stream = chain.stream(inputs).await.unwrap();
        let mut tokens = Vec::new();
        while let Some(item) = stream.next().await {
            tokens.push(item.unwrap());
        }
        let text: String = tokens.iter().map(|t| t.token.as_str()).collect();
        assert!(text.contains("refined answer"));
        assert!(tokens.last().unwrap().is_final);
        // 3 documents → 1 initial + 1 intermediate invoke, then 1 streamed final.
        assert_eq!(invokes.load(Ordering::SeqCst), 2);
        assert_eq!(streams.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_refine_stream_empty_documents() {
        let chain = RefineDocumentsChain::new(CountingLLM {
            invokes: Arc::new(AtomicUsize::new(0)),
            streams: Arc::new(AtomicUsize::new(0)),
        });
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Value::String("q".to_string()));
        inputs.insert("documents".to_string(), serde_json::json!([]));
        let err = match chain.stream(inputs).await {
            Ok(_) => panic!("expected an execution error"),
            Err(e) => e,
        };
        assert!(matches!(err, ChainError::ExecutionError(_)));
    }
}
