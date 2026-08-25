// lc-agents/src/adaptive_rag/tests.rs
//! Unit tests for `AdaptiveRAG`.

use super::*;
use async_trait::async_trait;
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_rag::{RetrieverError, RetrieverTrait};
use lc_schema::Message;
use lc_vector_stores::Document;
use std::sync::{Arc, Mutex};

use super::{build_context, parse_decision, route_tool};

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
            let mut guard = self.responses.lock().unwrap_or_else(|e| e.into_inner());
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
            let mut guard = self.responses.lock().unwrap_or_else(|e| e.into_inner());
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
            let mut guard = self.responses.lock().unwrap_or_else(|e| e.into_inner());
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
            tool_calls: Some(vec![lc_core::tools::ToolCall::builder("call_route")
                .name("route_decision")
                .arguments(format!(r#"{{"decision": "{}"}}"#, self.decision))
                .build()]),
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
            crate::streaming::AgentStreamEvent::PipelineStep { step, .. } => Some(step.as_str()),
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
            crate::streaming::AgentStreamEvent::PipelineStep { step, .. } => Some(step.as_str()),
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
