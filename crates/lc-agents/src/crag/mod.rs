// src/agents/crag/mod.rs
//! Corrective RAG (CRAG) Agent.
//!
//! Implements the Corrective Retrieval-Augmented Generation pattern:
//!
//! ```text
//! retrieve -> grade_documents -> [transform_query + web_search | keep] -> generate
//! ```
//!
//! When retrieved documents score below a configurable threshold, the agent
//! automatically rewrites the query and optionally falls back to web search
//! before re-retrieving and regenerating the answer.
//!
//! # Example
//!
//! ```rust,ignore
//! use langchainrust::{CorrectiveRAGAgent, OpenAIChat, OpenAIConfig, SimilarityRetriever};
//!
//! let llm = OpenAIChat::new(OpenAIConfig::default());
//! let retriever = SimilarityRetriever::new(store, embeddings);
//!
//! let agent = CorrectiveRAGAgent::new(llm, retriever)
//!     .with_grade_threshold(0.6)
//!     .with_web_fallback(Box::new(DuckDuckGoSearchTool::new()));
//!
//! let result = agent.invoke("What is CRAG?").await?;
//! println!("Answer: {}", result.answer);
//! println!("Grounded: {}", result.grounded);
//! ```

pub mod grader;
pub mod graph;
pub mod rewriter;

use lc_core::language_models::BaseChatModel;
use lc_core::tools::BaseTool;
use lc_rag::RetrieverTrait;
use lc_vector_stores::Document;

use graph::CRAGGraph;

/// CRAG error types.
#[derive(Debug, thiserror::Error)]
pub enum CRAGError {
    /// No documents were retrieved from the retriever.
    #[error("No documents retrieved for the query")]
    NoDocumentsRetrieved,

    /// Document retrieval failed.
    #[error("Retrieval error: {0}")]
    RetrievalError(lc_rag::RetrieverError),

    /// Document grading failed.
    #[error("Grading error: {0}")]
    GradingError(grader::GraderError),

    /// Query rewriting failed.
    #[error("Query rewriting error: {0}")]
    RewritingError(rewriter::RewriterError),

    /// Web search fallback failed.
    #[error("Web search error: {0}")]
    WebSearchError(lc_core::tools::ToolError),

    /// Answer generation failed.
    #[error("Answer generation error: {0}")]
    GenerationError(String),

    /// Hallucination check failed.
    #[error("Hallucination check error: {0}")]
    HallucinationCheckError(String),
}

/// Result of a CRAG invocation.
#[derive(Debug, Clone)]
pub struct CRAGResult {
    /// The generated answer.
    pub answer: String,
    /// Whether the answer is grounded in the source documents.
    pub grounded: bool,
    /// Source documents used to generate the answer.
    pub sources: Vec<Document>,
    /// Relevance grade scores for each source document.
    pub grade_scores: Vec<f64>,
    /// Relevance reasoning from the grader for each source document.
    pub grade_reasoning: Vec<Option<String>>,
}

/// Corrective RAG Agent.
///
/// Implements the CRAG pattern: retrieve documents, grade them for relevance,
/// and if the average score is below a threshold, rewrite the query and
/// optionally use a web search fallback before re-retrieving and generating
/// a new answer.
pub struct CorrectiveRAGAgent<M: BaseChatModel, R: RetrieverTrait> {
    llm: M,
    retriever: R,
    web_fallback: Option<Box<dyn BaseTool>>,
    grade_threshold: f64,
    retrieve_k: usize,
    enable_hallucination_check: bool,
    grader_llm: Option<M>,
    /// Maximum number of tokens for the context in prompts.
    max_context_tokens: Option<usize>,
}

impl<M: BaseChatModel, R: RetrieverTrait> CorrectiveRAGAgent<M, R> {
    /// Creates a new CRAG agent with the given LLM and retriever.
    ///
    /// Default grade threshold is 0.6 and default retrieve count is 4.
    pub fn new(llm: M, retriever: R) -> Self {
        Self {
            llm,
            retriever,
            web_fallback: None,
            grade_threshold: 0.6,
            retrieve_k: 4,
            enable_hallucination_check: true,
            grader_llm: None,
            max_context_tokens: None,
        }
    }

    /// Sets the web search fallback tool.
    ///
    /// When documents score below the threshold, the agent will call this
    /// tool with the rewritten query to supplement the retrieval results.
    pub fn with_web_fallback(mut self, tool: Box<dyn BaseTool>) -> Self {
        self.web_fallback = Some(tool);
        self
    }

    /// Sets the grade threshold for document relevance.
    ///
    /// Documents scoring below this threshold on average will trigger
    /// the corrective path (query rewrite + optional web search).
    /// Must be in [0.0, 1.0]. Values outside this range are clamped.
    pub fn with_grade_threshold(mut self, threshold: f64) -> Self {
        self.grade_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Sets the number of documents to retrieve.
    pub fn with_retrieve_k(mut self, k: usize) -> Self {
        self.retrieve_k = k.max(1);
        self
    }

    /// Enables or disables the hallucination check step.
    ///
    /// When enabled (default), the agent verifies that the generated
    /// answer is grounded in the source documents.
    pub fn with_hallucination_check(mut self, enable: bool) -> Self {
        self.enable_hallucination_check = enable;
        self
    }

    /// Sets a separate LLM for hallucination checking.
    ///
    /// When set, hallucination checks use this LLM instead of the main LLM,
    /// avoiding the self-verification bias where a model tends to endorse
    /// its own output. The grader LLM must be the same concrete type as the
    /// main LLM. When not set, falls back to the main LLM.
    pub fn with_grader_llm(mut self, llm: M) -> Self {
        self.grader_llm = Some(llm);
        self
    }

    /// Sets the maximum number of tokens for the context in prompts.
    ///
    /// When set, source documents are truncated from the lowest-scoring
    /// documents to fit within this budget before being passed to the LLM.
    pub fn with_max_context_tokens(mut self, tokens: usize) -> Self {
        self.max_context_tokens = Some(tokens);
        self
    }

    /// Invokes the CRAG agent on the given query.
    ///
    /// Executes the full CRAG pipeline:
    /// 1. Retrieve documents from the retriever
    /// 2. Grade each document for relevance
    /// 3. If average score < threshold: rewrite query, optionally web search, re-retrieve
    /// 4. Generate answer from filtered documents
    /// 5. Optional: hallucination check
    pub async fn invoke(&self, query: &str) -> Result<CRAGResult, CRAGError> {
        let web_ref: Option<&dyn BaseTool> = self.web_fallback.as_ref().map(|b| b.as_ref());

        let mut graph = CRAGGraph::new(&self.llm, &self.retriever, web_ref, self.grade_threshold)
            .with_retrieve_k(self.retrieve_k)
            .with_hallucination_check(self.enable_hallucination_check);

        if let Some(ref grader) = self.grader_llm {
            graph = graph.with_grader_llm(grader);
        }

        if let Some(tokens) = self.max_context_tokens {
            graph = graph.with_max_context_tokens(tokens);
        }

        graph.run(query).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
    use lc_core::runnables::{Runnable, RunnableConfig};
    use lc_core::tools::ToolError;
    use lc_rag::RetrieverError;
    use lc_schema::Message;
    use lc_vector_stores::SearchResult;
    use std::pin::Pin;

    /// Error type for mock chat model.
    #[derive(Debug, thiserror::Error)]
    #[error("mock error: {0}")]
    struct MockError(String);

    // === Mock LLM ===

    /// A mock chat model that returns configurable responses in sequence.
    #[derive(Debug, Clone)]
    struct MockChatModel {
        responses: Vec<String>,
        call_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl MockChatModel {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: responses.iter().map(|s| s.to_string()).collect(),
                call_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockChatModel {
        type Error = MockError;

        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let idx = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let response = self
                .responses
                .get(idx)
                .unwrap_or(&"relevant".to_string())
                .clone();
            Ok(LLMResult {
                content: response,
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockChatModel {
        fn model_name(&self) -> &str {
            "mock"
        }
        fn get_num_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
        fn with_temperature(self, _temp: f32) -> Self {
            self
        }
        fn with_max_tokens(self, _max: usize) -> Self {
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
            let idx = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let response = self
                .responses
                .get(idx)
                .unwrap_or(&"relevant".to_string())
                .clone();
            Ok(LLMResult {
                content: response,
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
            Err(MockError("streaming not supported".to_string()))
        }
    }

    // === Mock Retriever ===

    #[derive(Debug, Clone)]
    struct MockRetriever {
        documents: Vec<Document>,
    }

    impl MockRetriever {
        fn new(documents: Vec<Document>) -> Self {
            Self { documents }
        }
    }

    #[async_trait]
    impl RetrieverTrait for MockRetriever {
        async fn retrieve(&self, _query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
            Ok(self.documents.iter().take(k).cloned().collect())
        }

        async fn retrieve_with_scores(
            &self,
            query: &str,
            k: usize,
        ) -> Result<Vec<SearchResult>, RetrieverError> {
            let docs = self.retrieve(query, k).await?;
            Ok(docs
                .into_iter()
                .enumerate()
                .map(|(i, doc)| SearchResult {
                    document: doc,
                    score: 1.0 - (i as f32 * 0.1),
                })
                .collect())
        }

        async fn add_documents(&self, _documents: Vec<Document>) -> Result<(), RetrieverError> {
            Ok(())
        }
    }

    // === Mock Web Tool ===

    struct MockWebTool;

    #[async_trait]
    impl BaseTool for MockWebTool {
        fn name(&self) -> &str {
            "web_search"
        }
        fn description(&self) -> &str {
            "Search the web"
        }
        async fn run(&self, _input: String) -> Result<String, ToolError> {
            Ok("Web search result: CRAG is Corrective RAG.".to_string())
        }
    }

    // === Tests ===

    #[tokio::test]
    async fn test_crag_agent_high_score_documents() {
        let llm = MockChatModel::new(vec![
            "Relevance: relevant\nScore: 0.9\nReasoning: Directly addresses the query.",
            "Relevance: relevant\nScore: 0.8\nReasoning: Closely related.",
            "Rust is a systems programming language focused on safety and performance.",
            "grounded",
        ]);

        let retriever = MockRetriever::new(vec![
            Document::new("Rust is a systems programming language."),
            Document::new("Rust emphasizes memory safety."),
        ]);

        let agent = CorrectiveRAGAgent::new(llm, retriever)
            .with_grade_threshold(0.5)
            .with_hallucination_check(true);

        let result = agent.invoke("What is Rust?").await.unwrap();
        assert!(!result.answer.is_empty());
        assert!(result.grounded);
        assert_eq!(result.sources.len(), 2);
        assert_eq!(result.grade_scores.len(), 2);
        assert!(result.grade_scores[0] >= 0.5);
    }

    #[tokio::test]
    async fn test_crag_agent_low_score_triggers_correction() {
        // generate_alternatives returns 3 alternative queries
        let llm = MockChatModel::new(vec![
            "Relevance: irrelevant\nScore: 0.1\nReasoning: Not related.",
            "Relevance: irrelevant\nScore: 0.2\nReasoning: Barely related.",
            "1. What are the key features of Rust?\n2. Rust programming language overview\n3. Rust memory safety and performance",
            "Relevance: relevant\nScore: 0.9\nReasoning: Directly addresses.",
            "Relevance: relevant\nScore: 0.8\nReasoning: Closely related.",
            "Rust provides memory safety without garbage collection.",
            "grounded",
        ]);

        let retriever = MockRetriever::new(vec![
            Document::new("Rust provides memory safety guarantees."),
            Document::new("Rust has zero-cost abstractions."),
        ]);

        let agent = CorrectiveRAGAgent::new(llm, retriever)
            .with_grade_threshold(0.5)
            .with_hallucination_check(true);

        let result = agent.invoke("Tell me about Rust").await.unwrap();
        assert!(!result.answer.is_empty());
        assert!(result.grounded);
    }

    #[tokio::test]
    async fn test_crag_agent_with_web_fallback() {
        let llm = MockChatModel::new(vec![
            "Relevance: irrelevant\nScore: 0.1\nReasoning: Not related.",
            "1. What is CRAG in AI?\n2. Corrective RAG technique\n3. CRAG methodology overview",
            "Relevance: relevant\nScore: 0.9\nReasoning: Direct match.",
            "CRAG stands for Corrective RAG.",
            "grounded",
        ]);

        let retriever = MockRetriever::new(vec![Document::new(
            "CRAG is a retrieval-augmented generation technique.",
        )]);

        let agent = CorrectiveRAGAgent::new(llm, retriever)
            .with_grade_threshold(0.5)
            .with_web_fallback(Box::new(MockWebTool))
            .with_hallucination_check(true);

        let result = agent.invoke("What is CRAG?").await.unwrap();
        assert!(!result.answer.is_empty());
    }

    #[tokio::test]
    async fn test_crag_agent_no_documents_retrieved() {
        let llm = MockChatModel::new(vec![]);
        let retriever = MockRetriever::new(vec![]);

        let agent = CorrectiveRAGAgent::new(llm, retriever);

        let result = agent.invoke("What is Rust?").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            CRAGError::NoDocumentsRetrieved => {}
            other => panic!("Expected NoDocumentsRetrieved, got: {}", other),
        }
    }

    #[tokio::test]
    async fn test_crag_agent_hallucination_detected() {
        let llm = MockChatModel::new(vec![
            "Relevance: relevant\nScore: 0.9\nReasoning: Direct match.",
            "Rust was invented by aliens in 3020.",
            "not grounded",
        ]);

        let retriever = MockRetriever::new(vec![Document::new(
            "Rust was created by Graydon Hoare in 2010.",
        )]);

        let agent = CorrectiveRAGAgent::new(llm, retriever).with_hallucination_check(true);

        let result = agent.invoke("Who created Rust?").await.unwrap();
        assert!(!result.grounded);
    }

    #[tokio::test]
    async fn test_crag_agent_hallucination_check_disabled() {
        let llm = MockChatModel::new(vec![
            "Relevance: relevant\nScore: 0.9\nReasoning: Direct match.",
            "Rust is great.",
        ]);

        let retriever = MockRetriever::new(vec![Document::new("Rust is a programming language.")]);

        let agent = CorrectiveRAGAgent::new(llm, retriever).with_hallucination_check(false);

        let result = agent.invoke("What is Rust?").await.unwrap();
        // grounded defaults to true when check is disabled
        assert!(result.grounded);
    }

    #[test]
    fn test_crag_result_fields() {
        let result = CRAGResult {
            answer: "Test answer".to_string(),
            grounded: true,
            sources: vec![Document::new("Source 1")],
            grade_scores: vec![0.9],
            grade_reasoning: vec![Some("Directly relevant".to_string())],
        };
        assert_eq!(result.answer, "Test answer");
        assert!(result.grounded);
        assert_eq!(result.sources.len(), 1);
        assert_eq!(result.grade_scores.len(), 1);
        assert_eq!(result.grade_reasoning.len(), 1);
    }

    #[test]
    fn test_crag_error_display() {
        let err = CRAGError::NoDocumentsRetrieved;
        assert!(err.to_string().contains("No documents retrieved"));

        let err = CRAGError::GenerationError("timeout".to_string());
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_grade_threshold_clamping() {
        let llm = MockChatModel::new(vec![]);
        let retriever = MockRetriever::new(vec![Document::new("test")]);

        let agent = CorrectiveRAGAgent::new(llm, retriever).with_grade_threshold(1.5);
        assert!((agent.grade_threshold - 1.0).abs() < f64::EPSILON);

        let agent = agent.with_grade_threshold(-0.5);
        assert!((agent.grade_threshold - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_grade_threshold_is_0_6() {
        let llm = MockChatModel::new(vec![]);
        let retriever = MockRetriever::new(vec![Document::new("test")]);

        let agent = CorrectiveRAGAgent::new(llm, retriever);
        assert!((agent.grade_threshold - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_grader_llm_is_stored_when_set() {
        let llm = MockChatModel::new(vec![]);
        let retriever = MockRetriever::new(vec![Document::new("test")]);

        let grader = MockChatModel::new(vec!["grounded"]);
        let agent = CorrectiveRAGAgent::new(llm, retriever).with_grader_llm(grader);
        assert!(agent.grader_llm.is_some());
    }

    #[tokio::test]
    async fn test_crag_agent_with_grader_llm() {
        // Grader LLM returns "not grounded" -> answer marked as ungrounded
        let llm = MockChatModel::new(vec![
            "Relevance: relevant\nScore: 0.9\nReasoning: Direct match.",
            "Rust was invented by aliens.",
            "not grounded",
        ]);
        let grader = MockChatModel::new(vec!["not grounded"]);

        let retriever =
            MockRetriever::new(vec![Document::new("Rust was created by Graydon Hoare.")]);

        let agent = CorrectiveRAGAgent::new(llm, retriever)
            .with_grader_llm(grader)
            .with_hallucination_check(true);

        let result = agent.invoke("Who created Rust?").await.unwrap();
        assert!(!result.grounded);
    }
}
