// lc-chains/src/document_chains/stuff.rs
//! StuffDocumentsChain - stuffs all documents into a single prompt.

use async_trait::async_trait;
use futures_util::StreamExt;
use lc_core::BaseChatModel;
use lc_providers::{wrap_chat_model, ProviderError};
use lc_schema::Message;
use lc_shared::document::Document;
use serde_json::Value;
use std::collections::HashMap;

use crate::base::{BaseChain, ChainError, ChainResult, ChainStream, StreamToken};
use crate::BoxedChatModel;

/// Default Stuff prompt template.
pub(crate) const DEFAULT_STUFF_PROMPT: &str =
    "Answer the user's question based on the following reference information.

Reference information:
{context}

Question: {input}

Answer:";

/// StuffDocumentsChain
///
/// Stuffs all documents into a single prompt for LLM processing.
/// Suitable when the total document content fits within the LLM context window.
pub struct StuffDocumentsChain {
    llm: BoxedChatModel,
    prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
    /// Maximum character count per document (truncated if exceeded).
    max_doc_length: Option<usize>,
}

impl StuffDocumentsChain {
    /// Create a new [`StuffDocumentsChain`] with the given LLM.
    pub fn new<L>(llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: wrap_chat_model(llm),
            prompt_template: DEFAULT_STUFF_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "stuff_documents".to_string(),
            verbose: false,
            max_doc_length: None,
        }
    }

    /// Set the prompt template.
    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
        self
    }

    /// Set the document variable name used in the prompt.
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

    /// Set the maximum character count per document (documents are truncated if exceeded).
    pub fn with_max_doc_length(mut self, max: usize) -> Self {
        self.max_doc_length = Some(max);
        self
    }

    /// Format document list into context text.
    pub fn format_documents(&self, documents: &[Document]) -> String {
        let mut parts = Vec::new();
        for (i, doc) in documents.iter().enumerate() {
            let mut content = doc.content.clone();
            if let Some(max_len) = self.max_doc_length {
                let char_count: usize = content.chars().count();
                if char_count > max_len {
                    content = content.chars().take(max_len).collect::<String>();
                    content.push_str("...\n[document truncated]");
                }
            }
            parts.push(format!("Document {}:\n{}", i + 1, content));
        }
        parts.join("\n\n---\n\n")
    }

    /// Build prompt.
    pub fn build_prompt(&self, context: &str, input: &str) -> String {
        self.prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
    }

    /// Invoke with documents and input directly.
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<String, ChainError> {
        // P2-7: same loud empty-documents guard as map_reduce/refine/map_rerank —
        // without it the LLM would be called with an empty context and fabricate
        // an answer that has no reference information at all.
        if documents.is_empty() {
            return Err(ChainError::ExecutionError(
                "Document list is empty".to_string(),
            ));
        }

        let context = self.format_documents(&documents);

        if self.verbose {
            println!("\n=== StuffDocumentsChain ===");
            println!("Document count: {}", documents.len());
            println!("Context length: {} characters", context.len());
        }

        let prompt = self.build_prompt(&context, input);

        if self.verbose {
            println!("Prompt length: {} characters", prompt.len());
        }

        let messages = vec![Message::human(&prompt)];
        let response = self
            .llm
            .invoke(messages, None)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;

        let output = response.content;

        if self.verbose {
            println!("Output: {}", output);
            println!("=== StuffDocumentsChain complete ===\n");
        }

        Ok(output)
    }
}

#[async_trait]
impl BaseChain for StuffDocumentsChain {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key, "documents"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        // P2-8: validate inputs on the invoke path too, matching stream (and the
        // crate-wide invoke+stream convention) so missing keys fail identically.
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

    /// Stream execution for StuffDocumentsChain — token by token output.
    ///
    /// Stuffs all documents into a single prompt, then streams the LLM
    /// response token by token.
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents = crate::base::documents_from_input(inputs.get("documents"))?;

        // P2-7: guard empty documents on the stream path too, mirroring the
        // other document chains — streaming with zero context would still call
        // the LLM and emit a fabricated, reference-free answer.
        if documents.is_empty() {
            return Err(ChainError::ExecutionError(
                "Document list is empty".to_string(),
            ));
        }

        let context = self.format_documents(&documents);
        let prompt = self.build_prompt(&context, input);
        let messages = vec![Message::human(&prompt)];

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
    use lc_core::language_models::LLMResult;
    use lc_core::runnables::RunnableConfig;
    use lc_core::{BaseLanguageModel, Runnable};
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Counting chat model that records `invoke` calls so the P2-7 typed entry
    /// is provable: non-empty documents reach the LLM exactly once, and empty
    /// documents never reach it at all.
    struct CountingLLM {
        invokes: Arc<AtomicUsize>,
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
                content: "stuffed answer".to_string(),
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
                content: "stuffed answer".to_string(),
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
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
        {
            let tokens = [Ok("streamed answer".to_string())];
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
    }

    /// P2-7: the typed `invoke_with_documents` entry skips the HashMap roundtrip
    /// and runs the LLM over the documents directly.
    #[tokio::test]
    async fn test_stuff_typed_invoke_runs_llm() {
        let invokes = Arc::new(AtomicUsize::new(0));
        let chain = StuffDocumentsChain::new(CountingLLM {
            invokes: invokes.clone(),
        });

        let out = chain
            .invoke_with_documents(vec![Document::new("doc one")], "question")
            .await
            .unwrap();
        assert_eq!(out, "stuffed answer");
        assert_eq!(invokes.load(Ordering::SeqCst), 1);
    }

    /// P2-7: empty documents error loudly on the typed invoke path and the LLM
    /// is never called with an empty context (consistent with
    /// map_reduce/refine/map_rerank).
    #[tokio::test]
    async fn test_stuff_typed_invoke_empty_documents_errors() {
        let invokes = Arc::new(AtomicUsize::new(0));
        let chain = StuffDocumentsChain::new(CountingLLM {
            invokes: invokes.clone(),
        });

        let err = match chain.invoke_with_documents(vec![], "q").await {
            Ok(_) => panic!("expected an execution error"),
            Err(e) => e,
        };
        assert!(matches!(err, ChainError::ExecutionError(_)));
        assert_eq!(invokes.load(Ordering::SeqCst), 0);
    }

    /// P2-8: the HashMap `invoke` path validates inputs before any LLM call,
    /// matching the stream path — a missing `input` key yields `MissingInput`
    /// with zero invokes, instead of reaching the LLM first.
    #[tokio::test]
    async fn test_stuff_invoke_missing_input_errors_before_llm() {
        let invokes = Arc::new(AtomicUsize::new(0));
        let chain = StuffDocumentsChain::new(CountingLLM {
            invokes: invokes.clone(),
        });
        let mut inputs = HashMap::new();
        inputs.insert("documents".to_string(), serde_json::json!([]));

        let err = match chain.invoke(inputs).await {
            Ok(_) => panic!("expected a missing-input error"),
            Err(e) => e,
        };
        assert!(matches!(err, ChainError::MissingInput(_)));
        assert_eq!(invokes.load(Ordering::SeqCst), 0);
    }

    /// P2-7: the stream path guards empty documents the same way, erroring
    /// before any `stream_chat` call.
    #[tokio::test]
    async fn test_stuff_stream_empty_documents_errors() {
        let chain = StuffDocumentsChain::new(CountingLLM {
            invokes: Arc::new(AtomicUsize::new(0)),
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
