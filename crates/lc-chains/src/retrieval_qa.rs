// lc-chains/src/retrieval_qa.rs
//! RetrievalQA Chain
//!
//! One-stop retrieval QA chain that encapsulates the complete RAG workflow.

use async_trait::async_trait;
use futures_util::StreamExt;
use lc_core::BaseChatModel;
use lc_providers::{wrap_chat_model, ProviderError};
use lc_rag::retriever::RetrieverTrait;
use lc_schema::Message;
use lc_shared::document::Document;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use crate::base::{BaseChain, ChainError, ChainResult, ChainStream, StreamToken};
use crate::BoxedChatModel;

/// Default QA prompt template.
const DEFAULT_QA_PROMPT: &str = "Answer the question based on the following context. If the context does not contain relevant information, say 'I don't know'.

Context:
{context}

Question: {question}

Answer:";

/// RetrievalQA Chain
///
/// One-stop retrieval QA chain that automatically:
/// 1. Retrieves relevant documents
/// 2. Assembles prompt (context + question)
/// 3. LLM generates answer
pub struct RetrievalQA {
    llm: BoxedChatModel,
    retriever: Arc<dyn RetrieverTrait>,

    prompt_template: String,
    input_key: String,
    output_key: String,
    name: String,

    k: usize,
    verbose: bool,

    return_source_documents: bool,
    source_document_key: String,
}

impl RetrievalQA {
    /// Create a new [`RetrievalQA`] chain with the given LLM and retriever.
    pub fn new<L>(llm: L, retriever: Arc<dyn RetrieverTrait>) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: wrap_chat_model(llm),
            retriever,
            prompt_template: DEFAULT_QA_PROMPT.to_string(),
            input_key: "query".to_string(),
            output_key: "result".to_string(),
            name: "retrieval_qa".to_string(),
            k: 4,
            verbose: false,
            return_source_documents: false,
            source_document_key: "source_documents".to_string(),
        }
    }

    /// Set the prompt template.
    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
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

    /// Set the number of documents to retrieve.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set whether to return the source documents with the answer.
    pub fn with_return_source_documents(mut self, return_source: bool) -> Self {
        self.return_source_documents = return_source;
        self
    }

    /// Set the key under which source documents are placed in the output.
    pub fn with_source_document_key(mut self, key: impl Into<String>) -> Self {
        self.source_document_key = key.into();
        self
    }

    /// Get the retriever reference.
    pub fn retriever(&self) -> &Arc<dyn RetrieverTrait> {
        &self.retriever
    }

    /// Get the number of documents retrieved (`k`).
    pub fn k(&self) -> usize {
        self.k
    }

    fn format_context(&self, documents: &[Document]) -> String {
        documents
            .iter()
            .map(|doc| doc.content.clone())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn build_prompt(&self, context: &str, question: &str) -> String {
        self.prompt_template
            .replace("{context}", context)
            .replace("{question}", question)
    }

    /// Simplified query interface returning the answer string.
    pub async fn query(&self, question: impl Into<String>) -> Result<String, ChainError> {
        let inputs = HashMap::from([(self.input_key.clone(), Value::String(question.into()))]);

        let result = self.invoke(inputs).await?;

        result
            .get(&self.output_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ChainError::OutputError("Missing output result".to_string()))
    }

    /// Query the chain, returning both the answer and the retrieved source documents.
    pub async fn query_with_sources(
        &self,
        question: impl Into<String>,
    ) -> Result<(String, Vec<Document>), ChainError> {
        // Reuses the single `run` pipeline instead of duplicating retrieval +
        // prompt assembly here (P1-5: previously re-implemented the whole chain
        // when `return_source_documents` was false).
        let question = question.into();
        self.run(&question).await
    }

    /// Shared execution pipeline: retrieve → assemble prompt → LLM.
    ///
    /// Used by both [`BaseChain::invoke`] and [`Self::query_with_sources`] so the
    /// RAG path is defined exactly once.
    async fn run(&self, question: &str) -> Result<(String, Vec<Document>), ChainError> {
        if self.verbose {
            println!("\n=== RetrievalQA Execution ===");
            println!("Question: {}", question);
            println!("Retrieval count (k): {}", self.k);
            println!("\n--- Step 1: Retrieve relevant documents ---");
        }

        let documents = self
            .retriever
            .retrieve(question, self.k)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Retrieval failed: {}", e)))?;

        if self.verbose {
            println!("Retrieved {} documents", documents.len());
            for (i, doc) in documents.iter().enumerate() {
                let preview: String = doc.content.chars().take(100).collect();
                println!("Document {}: {}", i + 1, preview);
            }
            if documents.is_empty() {
                println!("Warning: No relevant documents retrieved");
            }
            println!("\n--- Step 2: Assemble Prompt ---");
        }

        let context = self.format_context(&documents);
        let prompt = self.build_prompt(&context, question);

        if self.verbose {
            println!("Context length: {} characters", context.len());
            println!("Prompt length: {} characters", prompt.len());
            println!("\n--- Step 3: LLM generates answer ---");
        }

        let messages = vec![Message::human(&prompt)];
        let response = self
            .llm
            .invoke(messages, None)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {}", e)))?;

        if self.verbose {
            println!("Answer: {}", response.content);
            println!("=== RetrievalQA Complete ===\n");
        }

        Ok((response.content, documents))
    }
}

#[async_trait]
impl BaseChain for RetrievalQA {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key]
    }

    fn output_keys(&self) -> Vec<&str> {
        if self.return_source_documents {
            vec![&self.output_key, &self.source_document_key]
        } else {
            vec![&self.output_key]
        }
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.validate_inputs(&inputs)?;

        let question = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let (answer, documents) = self.run(question).await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(answer));

        if self.return_source_documents {
            // Explicit error instead of silently inserting Value::Null (P1-2).
            let sources = crate::base::documents_to_values(&documents)?;
            result.insert(self.source_document_key.clone(), Value::Array(sources));
        }

        Ok(result)
    }

    /// Stream execution for RetrievalQA -- token by token output.
    ///
    /// P2-2: real streaming — retrieves and assembles the prompt first, then
    /// pushes LLM tokens via `stream_chat` instead of wrapping `invoke` in a
    /// single chunk (the base default).
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        self.validate_inputs(&inputs)?;

        let question = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        if self.verbose {
            println!("\n=== RetrievalQA Stream ===");
            println!("Question: {}", question);
            println!("Retrieval count (k): {}", self.k);
        }

        let documents = self
            .retriever
            .retrieve(question, self.k)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("Retrieval failed: {}", e)))?;

        if self.verbose {
            println!("Retrieved {} documents", documents.len());
        }

        let context = self.format_context(&documents);
        let prompt = self.build_prompt(&context, question);

        let messages = vec![Message::human(&prompt)];
        let llm_stream = self
            .llm
            .stream_chat(messages, None)
            .await
            .map_err(|e| ChainError::StreamError(format!("LLM stream failed: {}", e)))?;

        let stream = llm_stream.map(move |result| match result {
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
    use lc_rag::retriever::RetrieverError;
    use lc_shared::document::SearchResult;
    use std::pin::Pin;

    /// Mock retriever that returns the preloaded documents (up to `k`).
    struct MockRetriever(Vec<Document>);

    #[async_trait]
    impl RetrieverTrait for MockRetriever {
        async fn retrieve(&self, _query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
            Ok(self.0.iter().take(k).cloned().collect())
        }
        async fn retrieve_with_scores(
            &self,
            _query: &str,
            _k: usize,
        ) -> Result<Vec<SearchResult>, RetrieverError> {
            Ok(Vec::new())
        }
        async fn add_documents(&self, _documents: Vec<Document>) -> Result<(), RetrieverError> {
            Ok(())
        }
    }

    /// Mock chat model with a deterministic token stream.
    struct MockLLM;

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockLLM {
        type Error = ProviderError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: "hello world".to_string(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockLLM {
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
    impl BaseChatModel for MockLLM {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: "hello world".to_string(),
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
            let tokens = [
                Ok(StreamChunk::new("hello")),
                Ok(StreamChunk::new(" ")),
                Ok(StreamChunk::new("world")),
            ];
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
    }

    fn doc(content: &str) -> Document {
        Document::new(content.to_string())
    }

    /// P2-2: RetrievalQA streams real tokens from the LLM instead of wrapping
    /// `invoke` in a single chunk.
    #[tokio::test]
    async fn test_retrieval_qa_stream_emits_tokens() {
        let retriever: Arc<dyn RetrieverTrait> = Arc::new(MockRetriever(vec![doc("ctx")]));
        let chain = RetrievalQA::new(MockLLM, retriever);
        let inputs = HashMap::from([("query".to_string(), Value::String("q".to_string()))]);

        let mut stream = chain.stream(inputs).await.unwrap();
        let mut tokens = Vec::new();
        while let Some(item) = stream.next().await {
            tokens.push(item.unwrap());
        }
        let text: String = tokens.iter().map(|t| t.token.as_str()).collect();
        assert_eq!(text, "hello world");
        assert!(tokens.last().unwrap().is_final);
        // Multiple non-final tokens prove the stream is token-by-token, not one
        // wrapped chunk.
        assert!(tokens.iter().filter(|t| !t.is_final).count() >= 2);
    }

    #[tokio::test]
    async fn test_retrieval_qa_stream_missing_input() {
        let retriever: Arc<dyn RetrieverTrait> = Arc::new(MockRetriever(vec![]));
        let chain = RetrievalQA::new(MockLLM, retriever);
        let err = match chain.stream(HashMap::new()).await {
            Ok(_) => panic!("expected a missing-input error"),
            Err(e) => e,
        };
        assert!(matches!(err, ChainError::MissingInput(_)));
    }
}
