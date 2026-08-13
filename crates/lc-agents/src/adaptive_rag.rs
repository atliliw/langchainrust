// src/agents/adaptive_rag.rs
//! Adaptive RAG implementation.
//!
//! Uses an LLM to decide whether retrieval is needed and what strategy to use.
//! Three decision branches:
//! - **NoRetrieval**: The query can be answered from general knowledge.
//! - **SingleSearch**: A single search is sufficient.
//! - **MultiQuery**: The query is complex and needs multiple search angles.

use lc_core::language_models::BaseChatModel;
use lc_core::tools::ToolDefinition;
use lc_rag::{RetrieverError, RetrieverTrait};
use lc_schema::Message;
use lc_vector_stores::Document;
use serde_json::json;

// ---------------------------------------------------------------------------
// Decision enum
// ---------------------------------------------------------------------------

/// Decision made by the adaptive router.
#[derive(Debug, Clone, PartialEq)]
pub enum RagDecision {
    /// No retrieval needed - LLM can answer directly.
    NoRetrieval,
    /// Single search query sufficient.
    SingleSearch,
    /// Complex query - use multi-query expansion.
    MultiQuery,
}

impl std::fmt::Display for RagDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RagDecision::NoRetrieval => write!(f, "no_retrieval"),
            RagDecision::SingleSearch => write!(f, "single_search"),
            RagDecision::MultiQuery => write!(f, "multi_query"),
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result returned by [`AdaptiveRAG::invoke`].
#[derive(Debug, Clone)]
pub struct AdaptiveRAGResult {
    /// The generated answer.
    pub answer: String,
    /// The routing decision that was made.
    pub decision: RagDecision,
    /// Source documents used (empty when `decision` is `NoRetrieval`).
    pub sources: Vec<Document>,
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by [`AdaptiveRAG`].
#[derive(Debug, thiserror::Error)]
pub enum AdaptiveRAGError {
    /// LLM invocation failed.
    #[error("LLM error: {0}")]
    Llm(String),

    /// Retrieval failed.
    #[error("retrieval error: {0}")]
    Retrieval(#[from] RetrieverError),

    /// Failed to parse the routing decision from the LLM response.
    #[error("decision parse error: {0}")]
    DecisionParse(String),
}

// ---------------------------------------------------------------------------
// AdaptiveRAG
// ---------------------------------------------------------------------------

/// Adaptive RAG that routes queries to the most appropriate strategy.
///
/// # Overview
///
/// 1. The LLM classifies the query into one of three buckets:
///    `no_retrieval`, `single_search`, or `multi_query`.
/// 2. Based on the decision:
///    - **NoRetrieval**: call the LLM directly.
///    - **SingleSearch**: retrieve documents, then generate.
///    - **MultiQuery**: generate multiple query variants, retrieve for each,
///      merge results, then generate.
///
/// # Example
///
/// ```ignore
/// use langchainrust::agents::adaptive_rag::{AdaptiveRAG, RagDecision};
/// use langchainrust::OpenAIChat;
/// use langchainrust::retrieval::SimilarityRetriever;
///
/// let rag = AdaptiveRAG::new(llm, retriever);
/// let result = rag.invoke("What is the capital of France?").await?;
/// assert_eq!(result.decision, RagDecision::NoRetrieval);
/// ```
pub struct AdaptiveRAG<M: BaseChatModel, R: RetrieverTrait> {
    llm: M,
    retriever: R,
    /// Number of documents to retrieve per query.
    retrieve_k: usize,
    /// Number of alternative queries to generate for multi-query mode.
    multi_query_count: usize,
}

// ---------------------------------------------------------------------------
// Routing prompt
// ---------------------------------------------------------------------------

const ROUTING_PROMPT: &str = r#"Given the following query, decide whether retrieval is needed:
- "no_retrieval": The query can be answered from general knowledge
- "single_search": A single search is sufficient
- "multi_query": The query is complex and needs multiple search angles

Query: {query}

Respond with exactly one of: no_retrieval, single_search, multi_query"#;

// ---------------------------------------------------------------------------
// Generation prompts
// ---------------------------------------------------------------------------

const GENERATE_SYSTEM_PROMPT: &str = r#"You are a helpful assistant. Answer the user's question based on the provided context when available. If no context is provided, use your general knowledge. Be concise and accurate."#;

const MULTI_QUERY_PROMPT: &str = r#"You are an AI language model assistant. Your task is to generate {count} different versions of the given user question to retrieve relevant documents from a vector database.

By generating multiple perspectives on the user question, your goal is to help overcome some of the limitations of distance-based similarity search.

Provide these alternative questions separated by newlines.

Original question: {question}

Alternative questions:"#;

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

impl<M: BaseChatModel, R: RetrieverTrait> AdaptiveRAG<M, R> {
    /// Creates a new `AdaptiveRAG` with the given LLM and retriever.
    pub fn new(llm: M, retriever: R) -> Self {
        Self {
            llm,
            retriever,
            retrieve_k: 4,
            multi_query_count: 3,
        }
    }

    /// Sets the number of documents to retrieve per query.
    pub fn with_retrieve_k(mut self, k: usize) -> Self {
        self.retrieve_k = k;
        self
    }

    /// Sets the number of alternative queries for multi-query mode.
    pub fn with_multi_query_count(mut self, count: usize) -> Self {
        self.multi_query_count = count;
        self
    }

    /// Invokes the adaptive RAG pipeline for the given query.
    pub async fn invoke(&self, query: &str) -> Result<AdaptiveRAGResult, AdaptiveRAGError> {
        // Step 1: Route the query.
        let decision = self.route(query).await?;

        match decision {
            RagDecision::NoRetrieval => self.generate_no_retrieval(query).await,
            RagDecision::SingleSearch => self.generate_single_search(query).await,
            RagDecision::MultiQuery => self.generate_multi_query(query).await,
        }
    }

    /// Streams the AdaptiveRAG execution, emitting pipeline step events.
    ///
    /// Emits `AgentStreamEvent::PipelineStep` events for routing and generation,
    /// and `AgentStreamEvent::FinalAnswer` when the answer is ready.
    pub async fn stream(
        &self,
        query: &str,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures_util::Stream<Item = crate::streaming::AgentStreamEvent> + Send>,
        >,
        AdaptiveRAGError,
    > {
        use crate::streaming::AgentStreamEvent;

        let mut events: Vec<AgentStreamEvent> = Vec::new();

        // Step 1: Route
        events.push(AgentStreamEvent::PipelineStep {
            step: "routing".to_string(),
            detail: Some("Classifying query...".to_string()),
        });

        let decision = self.route(query).await?;

        events.push(AgentStreamEvent::PipelineStep {
            step: "routed".to_string(),
            detail: Some(format!("Decision: {}", decision)),
        });

        // Step 2: Execute based on decision
        let result = match decision {
            RagDecision::NoRetrieval => {
                events.push(AgentStreamEvent::PipelineStep {
                    step: "generating".to_string(),
                    detail: Some("No retrieval needed, generating directly...".to_string()),
                });
                self.generate_no_retrieval(query).await?
            }
            RagDecision::SingleSearch => {
                events.push(AgentStreamEvent::PipelineStep {
                    step: "retrieving".to_string(),
                    detail: Some("Single search retrieval...".to_string()),
                });
                let result = self.generate_single_search(query).await?;
                events.push(AgentStreamEvent::PipelineStep {
                    step: "generating".to_string(),
                    detail: Some(format!("Sources: {} documents", result.sources.len())),
                });
                result
            }
            RagDecision::MultiQuery => {
                events.push(AgentStreamEvent::PipelineStep {
                    step: "multi_query".to_string(),
                    detail: Some("Generating multiple queries...".to_string()),
                });
                let result = self.generate_multi_query(query).await?;
                events.push(AgentStreamEvent::PipelineStep {
                    step: "generating".to_string(),
                    detail: Some(format!("Sources: {} documents", result.sources.len())),
                });
                result
            }
        };

        // Final answer
        events.push(AgentStreamEvent::FinalAnswer {
            content: result.answer,
        });

        Ok(Box::pin(futures_util::stream::iter(events)))
    }

    // -- Routing -----------------------------------------------------------

    /// Asks the LLM to classify the query.
    async fn route(&self, query: &str) -> Result<RagDecision, AdaptiveRAGError> {
        let prompt = ROUTING_PROMPT.replace("{query}", query);
        let messages = vec![Message::human(&prompt)];

        // P1-3:优先 tool_calls 结构化路由,不支持绑定时回落文本解析。
        let structured = crate::structured::chat_structured(
            &self.llm,
            Some(route_tool()),
            messages,
            None,
            &crate::retry::RetryConfig::default(),
        )
        .await
        .map_err(|e| AdaptiveRAGError::Llm(e.to_string()))?;

        if let Some(args) = &structured.tool_args {
            if let Some(decision) = args.get("decision").and_then(|v| v.as_str()) {
                return parse_decision(decision);
            }
        }
        parse_decision(&structured.content)
    }

    // -- No retrieval ------------------------------------------------------

    /// Generates an answer without any retrieval.
    async fn generate_no_retrieval(
        &self,
        query: &str,
    ) -> Result<AdaptiveRAGResult, AdaptiveRAGError> {
        let messages = vec![Message::human(query)];
        let result = self
            .llm
            .chat_with_system(GENERATE_SYSTEM_PROMPT.to_string(), messages)
            .await
            .map_err(|e| AdaptiveRAGError::Llm(e.to_string()))?;

        Ok(AdaptiveRAGResult {
            answer: result.content,
            decision: RagDecision::NoRetrieval,
            sources: Vec::new(),
        })
    }

    // -- Single search -----------------------------------------------------

    /// Retrieves documents with a single query and generates an answer.
    async fn generate_single_search(
        &self,
        query: &str,
    ) -> Result<AdaptiveRAGResult, AdaptiveRAGError> {
        let docs = self.retriever.retrieve(query, self.retrieve_k).await?;

        let context = build_context(&docs);
        let user_msg = format!("Context:\n{}\n\nQuestion: {}\n\nAnswer:", context, query);
        let messages = vec![Message::human(&user_msg)];

        let result = self
            .llm
            .chat_with_system(GENERATE_SYSTEM_PROMPT.to_string(), messages)
            .await
            .map_err(|e| AdaptiveRAGError::Llm(e.to_string()))?;

        Ok(AdaptiveRAGResult {
            answer: result.content,
            decision: RagDecision::SingleSearch,
            sources: docs,
        })
    }

    // -- Multi query -------------------------------------------------------

    /// Generates multiple query variants, retrieves for each, merges, and
    /// generates an answer.
    async fn generate_multi_query(
        &self,
        query: &str,
    ) -> Result<AdaptiveRAGResult, AdaptiveRAGError> {
        let alternative_queries = self.generate_queries(query).await?;

        // Combine original + alternatives.
        let all_queries: Vec<String> = std::iter::once(query.to_string())
            .chain(alternative_queries)
            .collect();

        // Retrieve and merge.
        let docs = self.retrieve_and_merge(&all_queries).await?;

        let context = build_context(&docs);
        let user_msg = format!("Context:\n{}\n\nQuestion: {}\n\nAnswer:", context, query);
        let messages = vec![Message::human(&user_msg)];

        let result = self
            .llm
            .chat_with_system(GENERATE_SYSTEM_PROMPT.to_string(), messages)
            .await
            .map_err(|e| AdaptiveRAGError::Llm(e.to_string()))?;

        Ok(AdaptiveRAGResult {
            answer: result.content,
            decision: RagDecision::MultiQuery,
            sources: docs,
        })
    }

    /// Asks the LLM to generate alternative query variants.
    async fn generate_queries(&self, query: &str) -> Result<Vec<String>, AdaptiveRAGError> {
        let prompt = MULTI_QUERY_PROMPT
            .replace("{count}", &self.multi_query_count.to_string())
            .replace("{question}", query);

        let messages = vec![Message::human(&prompt)];
        let result = crate::retry::retry_chat(
            &self.llm,
            messages,
            None,
            &crate::retry::RetryConfig::default(),
        )
        .await
        .map_err(|e| AdaptiveRAGError::Llm(e.to_string()))?;

        let queries: Vec<String> = result
            .content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .collect();

        Ok(queries)
    }

    /// Retrieves documents for each query and merges/deduplicates by content.
    async fn retrieve_and_merge(
        &self,
        queries: &[String],
    ) -> Result<Vec<Document>, AdaptiveRAGError> {
        // M11: Parallel retrieval instead of sequential.
        let futures: Vec<_> = queries
            .iter()
            .map(|q| self.retriever.retrieve(q, self.retrieve_k))
            .collect();
        let all_results = futures_util::future::join_all(futures).await;

        let mut seen_content: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut merged: Vec<Document> = Vec::new();

        for result in all_results {
            let docs = result?;
            for doc in docs {
                // M6 fix: deduplicate by full content hash instead of first 80 chars
                // to avoid false collisions on documents with common prefixes.
                let key = {
                    use std::hash::Hasher;
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    hasher.write(doc.content.as_bytes());
                    format!("{:016x}", hasher.finish())
                };
                if seen_content.insert(key) {
                    merged.push(doc);
                }
            }
        }

        Ok(merged)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// 路由工具定义:强制 LLM 输出三态决策(P1-3)。
fn route_tool() -> ToolDefinition {
    ToolDefinition::new(
        "route_decision",
        "判断查询是否需要检索及检索策略:no_retrieval / single_search / multi_query",
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["no_retrieval", "single_search", "multi_query"]
            }
        },
        "required": ["decision"]
    }))
}

/// Parses the routing decision from the LLM response.
fn parse_decision(response: &str) -> Result<RagDecision, AdaptiveRAGError> {
    let lower = response.to_lowercase();

    // Check for exact or contained keywords, with precedence.
    if lower.contains("no_retrieval") {
        return Ok(RagDecision::NoRetrieval);
    }
    if lower.contains("multi_query") {
        return Ok(RagDecision::MultiQuery);
    }
    if lower.contains("single_search") {
        return Ok(RagDecision::SingleSearch);
    }

    Err(AdaptiveRAGError::DecisionParse(response.to_string()))
}

/// Builds a context string from source documents.
fn build_context(docs: &[Document]) -> String {
    docs.iter()
        .enumerate()
        .map(|(i, doc)| format!("[Document {}]: {}", i + 1, doc.content))
        .collect::<Vec<_>>()
        .join("\n\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
    use lc_core::runnables::{Runnable, RunnableConfig};
    use std::sync::{Arc, Mutex};

    // -- Mock LLM ---------------------------------------------------------

    /// A mock LLM that returns pre-configured responses in sequence.
    #[derive(Clone)]
    struct MockLLM {
        responses: Arc<Mutex<Vec<String>>>,
    }

    impl MockLLM {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses)),
            }
        }
    }

    #[derive(Debug, thiserror::Error)]
    #[error("mock llm error")]
    struct MockLlmError;

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockLLM {
        type Error = MockLlmError;

        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let content = {
                let mut guard = self.responses.lock().unwrap();
                if guard.is_empty() {
                    "mock response".to_string()
                } else {
                    guard.remove(0)
                }
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
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockLLM {
        fn model_name(&self) -> &str {
            "mock"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
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
    impl BaseChatModel for MockLLM {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let content = {
                let mut guard = self.responses.lock().unwrap();
                if guard.is_empty() {
                    "mock response".to_string()
                } else {
                    guard.remove(0)
                }
            };
            Ok(LLMResult {
                content,
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
        ) -> Result<
            std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<String, Self::Error>> + Send>>,
            Self::Error,
        > {
            let content = {
                let mut guard = self.responses.lock().unwrap();
                if guard.is_empty() {
                    "mock response".to_string()
                } else {
                    guard.remove(0)
                }
            };
            let stream = futures_util::stream::once(async move { Ok(content) });
            Ok(Box::pin(stream))
        }
    }

    /// P1-3:chat() 直接返回结构化 tool_call 决策的 mock(不走 bind_tools)。
    #[derive(Clone)]
    struct MockToolCallLLM {
        decision: String,
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockToolCallLLM {
        type Error = MockLlmError;

        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: String::new(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: Some(vec![lc_core::tools::ToolCall::new(
                    "call_route",
                    "route_decision",
                    format!(r#"{{"decision": "{}"}}"#, self.decision),
                )]),
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockToolCallLLM {
        fn model_name(&self) -> &str {
            "mock"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
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
    impl BaseChatModel for MockToolCallLLM {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invoke(_messages, _config).await
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<
            std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<String, Self::Error>> + Send>>,
            Self::Error,
        > {
            let decision = self.decision.clone();
            let stream = futures_util::stream::once(async move { Ok(decision) });
            Ok(Box::pin(stream))
        }
    }

    // -- Mock Retriever ---------------------------------------------------

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
            _query: &str,
            k: usize,
        ) -> Result<Vec<lc_vector_stores::SearchResult>, RetrieverError> {
            Ok(self
                .documents
                .iter()
                .take(k)
                .enumerate()
                .map(|(i, doc)| lc_vector_stores::SearchResult {
                    document: doc.clone(),
                    score: 1.0 - i as f32 * 0.1,
                })
                .collect())
        }

        async fn add_documents(&self, _documents: Vec<Document>) -> Result<(), RetrieverError> {
            Ok(())
        }
    }

    // -- parse_decision tests ----------------------------------------------

    #[test]
    fn test_parse_decision_no_retrieval() {
        let result = parse_decision("no_retrieval").unwrap();
        assert_eq!(result, RagDecision::NoRetrieval);
    }

    #[test]
    fn test_parse_decision_single_search() {
        let result = parse_decision("single_search").unwrap();
        assert_eq!(result, RagDecision::SingleSearch);
    }

    #[test]
    fn test_parse_decision_multi_query() {
        let result = parse_decision("multi_query").unwrap();
        assert_eq!(result, RagDecision::MultiQuery);
    }

    #[test]
    fn test_parse_decision_case_insensitive() {
        assert_eq!(
            parse_decision("NO_RETRIEVAL").unwrap(),
            RagDecision::NoRetrieval
        );
        assert_eq!(
            parse_decision("Single_Search").unwrap(),
            RagDecision::SingleSearch
        );
        assert_eq!(
            parse_decision("MULTI_QUERY").unwrap(),
            RagDecision::MultiQuery
        );
    }

    #[test]
    fn test_parse_decision_embedded_in_text() {
        assert_eq!(
            parse_decision("The answer is: no_retrieval").unwrap(),
            RagDecision::NoRetrieval
        );
        assert_eq!(
            parse_decision("I think multi_query is best").unwrap(),
            RagDecision::MultiQuery
        );
    }

    #[test]
    fn test_parse_decision_invalid() {
        let result = parse_decision("something else entirely");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AdaptiveRAGError::DecisionParse(_)
        ));
    }

    // -- RagDecision Display test ------------------------------------------

    #[tokio::test]
    async fn test_route_uses_tool_call_decision() {
        // P1-3:LLM 以 tool_calls 返回结构化决策,route 应直接采用,不靠文本关键字。
        let llm = MockToolCallLLM {
            decision: "multi_query".to_string(),
        };
        let retriever = MockRetriever::new(vec![]);
        let rag = AdaptiveRAG::new(llm, retriever);
        let decision = rag.route("some query").await.unwrap();
        assert_eq!(decision, RagDecision::MultiQuery);
    }

    #[test]
    fn test_route_tool_schema() {
        let tool = route_tool();
        assert_eq!(tool.function.name, "route_decision");
        assert!(tool.function.parameters.is_some());
    }

    #[test]
    fn test_rag_decision_display() {
        assert_eq!(format!("{}", RagDecision::NoRetrieval), "no_retrieval");
        assert_eq!(format!("{}", RagDecision::SingleSearch), "single_search");
        assert_eq!(format!("{}", RagDecision::MultiQuery), "multi_query");
    }

    // -- build_context test ------------------------------------------------

    #[test]
    fn test_build_context() {
        let docs = vec![
            Document::new("First document content"),
            Document::new("Second document content"),
        ];
        let context = build_context(&docs);
        assert!(context.contains("[Document 1]: First document content"));
        assert!(context.contains("[Document 2]: Second document content"));
    }

    #[test]
    fn test_build_context_empty() {
        let context = build_context(&[]);
        assert!(context.is_empty());
    }

    // -- Integration tests with mock LLM & retriever ----------------------

    #[tokio::test]
    async fn test_adaptive_rag_no_retrieval() {
        // Router returns "no_retrieval", then the generate call returns an answer.
        let llm = MockLLM::new(vec![
            "no_retrieval".to_string(),
            "Paris is the capital of France.".to_string(),
        ]);
        let retriever = MockRetriever::new(vec![Document::new("irrelevant doc")]);

        let rag = AdaptiveRAG::new(llm, retriever);
        let result = rag.invoke("What is the capital of France?").await.unwrap();

        assert_eq!(result.decision, RagDecision::NoRetrieval);
        assert!(result.answer.contains("Paris"));
        assert!(result.sources.is_empty());
    }

    #[tokio::test]
    async fn test_adaptive_rag_single_search() {
        // Router returns "single_search", then generate returns an answer.
        let llm = MockLLM::new(vec![
            "single_search".to_string(),
            "Rust is a systems programming language.".to_string(),
        ]);
        let retriever = MockRetriever::new(vec![Document::new(
            "Rust emphasizes safety and performance.",
        )]);

        let rag = AdaptiveRAG::new(llm, retriever);
        let result = rag.invoke("Tell me about Rust").await.unwrap();

        assert_eq!(result.decision, RagDecision::SingleSearch);
        assert!(result.answer.contains("Rust"));
        assert_eq!(result.sources.len(), 1);
    }

    #[tokio::test]
    async fn test_adaptive_rag_multi_query() {
        // Router returns "multi_query",
        // then generate_queries returns alternatives,
        // then generate returns the final answer.
        let llm = MockLLM::new(vec![
            "multi_query".to_string(),
            "How does Rust memory management work?\nWhat is ownership in Rust?".to_string(),
            "Rust uses ownership and borrowing for memory management.".to_string(),
        ]);
        let retriever = MockRetriever::new(vec![
            Document::new("Rust ownership model"),
            Document::new("Borrowing and lifetimes"),
        ]);

        let rag = AdaptiveRAG::new(llm, retriever);
        let result = rag.invoke("Explain Rust memory management").await.unwrap();

        assert_eq!(result.decision, RagDecision::MultiQuery);
        assert!(result.answer.contains("ownership"));
        assert!(!result.sources.is_empty());
    }

    #[tokio::test]
    async fn test_adaptive_rag_llm_error() {
        // Use a mock that returns an error by exhausting responses
        // and having the code call chat which would still succeed
        // with "mock response" (fallback). We test parse failure instead.
        let llm = MockLLM::new(vec!["something_unrelated".to_string()]);
        let retriever = MockRetriever::new(vec![]);

        let rag = AdaptiveRAG::new(llm, retriever);
        let result = rag.invoke("test query").await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AdaptiveRAGError::DecisionParse(_)
        ));
    }

    #[tokio::test]
    async fn test_adaptive_rag_with_retrieve_k() {
        let llm = MockLLM::new(vec![
            "single_search".to_string(),
            "Answer based on context.".to_string(),
        ]);
        let retriever = MockRetriever::new(vec![
            Document::new("Doc 1"),
            Document::new("Doc 2"),
            Document::new("Doc 3"),
        ]);

        let rag = AdaptiveRAG::new(llm, retriever).with_retrieve_k(2);
        let result = rag.invoke("test query").await.unwrap();

        assert_eq!(result.decision, RagDecision::SingleSearch);
        assert_eq!(result.sources.len(), 2); // limited by retrieve_k
    }

    // -- Error Display tests -----------------------------------------------

    #[test]
    fn test_adaptive_rag_error_display() {
        let err = AdaptiveRAGError::Llm("timeout".to_string());
        assert!(err.to_string().contains("LLM error"));
        assert!(err.to_string().contains("timeout"));

        let err = AdaptiveRAGError::DecisionParse("bad output".to_string());
        assert!(err.to_string().contains("decision parse error"));
        assert!(err.to_string().contains("bad output"));
    }

    #[tokio::test]
    async fn test_adaptive_rag_stream_no_retrieval() {
        use futures_util::StreamExt;

        let llm = MockLLM::new(vec![
            "no_retrieval".to_string(),
            "Direct answer here.".to_string(),
        ]);
        let retriever = MockRetriever::new(vec![]);

        let agent = AdaptiveRAG::new(llm, retriever);

        let stream = agent.stream("What is 2+2?").await.unwrap();
        let events: Vec<_> = stream.collect().await;

        let step_names: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                crate::streaming::AgentStreamEvent::PipelineStep { step, .. } => {
                    Some(step.as_str())
                }
                _ => None,
            })
            .collect();

        assert!(
            step_names.contains(&"routing"),
            "Expected 'routing' step, got: {:?}",
            step_names
        );
        assert!(
            step_names.contains(&"routed"),
            "Expected 'routed' step, got: {:?}",
            step_names
        );
        assert!(matches!(
            events.last().unwrap(),
            crate::streaming::AgentStreamEvent::FinalAnswer { .. }
        ));
    }

    #[tokio::test]
    async fn test_adaptive_rag_stream_single_search() {
        use futures_util::StreamExt;

        let llm = MockLLM::new(vec![
            "single_search".to_string(),
            "Relevance: relevant\nScore: 0.9\nReasoning: Direct match.".to_string(),
            "Rust is a systems programming language.".to_string(),
            "grounded".to_string(),
        ]);
        let retriever = MockRetriever::new(vec![Document::new("Rust is safe.")]);

        let agent = AdaptiveRAG::new(llm, retriever);

        let stream = agent.stream("What is Rust?").await.unwrap();
        let events: Vec<_> = stream.collect().await;

        let step_names: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                crate::streaming::AgentStreamEvent::PipelineStep { step, .. } => {
                    Some(step.as_str())
                }
                _ => None,
            })
            .collect();

        assert!(
            step_names.contains(&"retrieving"),
            "Expected 'retrieving' step, got: {:?}",
            step_names
        );
        assert!(matches!(
            events.last().unwrap(),
            crate::streaming::AgentStreamEvent::FinalAnswer { .. }
        ));
    }
}
