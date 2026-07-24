//! Unit tests for GraphRAG module.
//!
//! Uses a MockChatModel to avoid real LLM calls.

use async_trait::async_trait;
use futures_util::Stream;
use langchainrust::core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
use langchainrust::core::runnables::Runnable;
use langchainrust::retrieval::graph_rag::{
    community, extractor, graph_store, query, GraphRAG, GraphRAGConfig, GraphRAGError, QueryMode,
};
use langchainrust::vector_stores::Document;
use langchainrust::RunnableConfig;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Mock LLM
// ---------------------------------------------------------------------------

/// A simple error type for the mock.
#[derive(Debug, thiserror::Error)]
pub enum MockError {
    #[error("mock error: {0}")]
    Fail(String),
}

/// A mock chat model that returns a pre-configured response.
#[derive(Clone)]
struct MockChatModel {
    response: Arc<Mutex<String>>,
}

impl MockChatModel {
    fn new(response: &str) -> Self {
        Self {
            response: Arc::new(Mutex::new(response.to_string())),
        }
    }
}

#[async_trait]
impl Runnable<Vec<langchainrust::schema::Message>, LLMResult> for MockChatModel {
    type Error = MockError;

    async fn invoke(
        &self,
        _input: Vec<langchainrust::schema::Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let content = self.response.lock().unwrap().clone();
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
impl BaseLanguageModel<Vec<langchainrust::schema::Message>, LLMResult> for MockChatModel {
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
impl BaseChatModel for MockChatModel {
    async fn chat(
        &self,
        _messages: Vec<langchainrust::schema::Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let content = self.response.lock().unwrap().clone();
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
        messages: Vec<langchainrust::schema::Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        let result = self.chat(messages, config).await?;
        let content = result.content;
        Ok(Box::pin(futures_util::stream::once(
            async move { Ok(content) },
        )))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_graph_rag_add_documents() {
    let extraction_json = r#"{"entities":[{"name":"Rust","type":"Technology","description":"A systems programming language"},{"name":"Mozilla","type":"Organization","description":"Organization that created Rust"}],"relations":[{"source":"Rust","target":"Mozilla","type":"created_by","description":"Rust was created by Mozilla"}]}"#;

    let llm = MockChatModel::new(extraction_json);
    let graph_rag = GraphRAG::new(llm).with_config(
        GraphRAGConfig::new()
            .with_max_entities_per_doc(5)
            .with_max_relations_per_doc(5),
    );

    let docs = vec![Document::new(
        "Rust is a systems programming language created by Mozilla.",
    )];
    let result: Result<(), GraphRAGError> = graph_rag.add_documents(&docs).await;
    assert!(result.is_ok(), "add_documents failed: {:?}", result);

    assert_eq!(graph_rag.entity_count().await, 2);
    assert_eq!(graph_rag.relation_count().await, 1);
}

#[tokio::test]
async fn test_graph_rag_add_multiple_documents_dedup() {
    let extraction_json = r#"{"entities":[{"name":"Python","type":"Technology","description":"A scripting language"},{"name":"Guido","type":"Person","description":"Creator of Python"}],"relations":[{"source":"Python","target":"Guido","type":"created_by","description":"Python was created by Guido"}]}"#;

    let llm = MockChatModel::new(extraction_json);
    let graph_rag =
        GraphRAG::new(llm).with_config(GraphRAGConfig::new().with_max_entities_per_doc(5));

    let docs = vec![Document::new(
        "Python is a scripting language created by Guido.",
    )];
    let _: Result<(), GraphRAGError> = graph_rag.add_documents(&docs).await;

    // Second call with same mock response: entities deduplicated.
    let docs2 = vec![Document::new("Python is used in AI.")];
    let _: Result<(), GraphRAGError> = graph_rag.add_documents(&docs2).await;

    assert!(graph_rag.entity_count().await >= 2);
}

#[tokio::test]
async fn test_graph_rag_build_communities() {
    let extraction_json = r#"{"entities":[{"name":"Alice","type":"Person","description":"A developer"},{"name":"Bob","type":"Person","description":"A manager"},{"name":"AcmeCorp","type":"Organization","description":"A tech company"}],"relations":[{"source":"Alice","target":"AcmeCorp","type":"works_at","description":"Alice works at AcmeCorp"},{"source":"Bob","target":"AcmeCorp","type":"works_at","description":"Bob works at AcmeCorp"}]}"#;

    let llm = MockChatModel::new(extraction_json);
    let graph_rag = GraphRAG::new(llm);

    let docs = vec![Document::new("Alice and Bob work at AcmeCorp.")];
    let _: Result<(), GraphRAGError> = graph_rag.add_documents(&docs).await;

    let result: Result<(), GraphRAGError> = graph_rag.build_communities().await;
    assert!(result.is_ok(), "build_communities failed: {:?}", result);

    assert!(graph_rag.community_count().await > 0);
}

#[tokio::test]
async fn test_graph_rag_local_query() {
    let extraction_json = r#"{"entities":[{"name":"Rust","type":"Technology","description":"A systems programming language"},{"name":"Mozilla","type":"Organization","description":"Organization that created Rust"}],"relations":[{"source":"Rust","target":"Mozilla","type":"created_by","description":"Rust was created by Mozilla"}]}"#;

    let llm = MockChatModel::new(extraction_json);
    let graph_rag = GraphRAG::new(llm);

    let docs = vec![Document::new(
        "Rust is a systems programming language created by Mozilla.",
    )];
    let add_result: Result<(), GraphRAGError> = graph_rag.add_documents(&docs).await;
    assert!(add_result.is_ok(), "add_documents failed: {:?}", add_result);
    assert_eq!(
        graph_rag.entity_count().await,
        2,
        "expected 2 entities, got {}",
        graph_rag.entity_count().await
    );

    let result: Result<query::GraphRAGResult, GraphRAGError> =
        graph_rag.query("What is Rust?", QueryMode::Local).await;
    assert!(result.is_ok(), "local query failed: {:?}", result);

    let rag_result = result.unwrap();
    assert!(!rag_result.answer.is_empty());
    assert_eq!(rag_result.mode, QueryMode::Local);
}

#[tokio::test]
async fn test_graph_rag_global_query_no_communities() {
    let extraction_json = r#"{"entities":[{"name":"Rust","type":"Technology","description":"A systems programming language"}],"relations":[]}"#;

    let llm = MockChatModel::new(extraction_json);
    let graph_rag = GraphRAG::new(llm);

    let docs = vec![Document::new("Rust is a systems programming language.")];
    let _: Result<(), GraphRAGError> = graph_rag.add_documents(&docs).await;

    // Global query without building communities should fail.
    let result: Result<query::GraphRAGResult, GraphRAGError> =
        graph_rag.query("What is Rust?", QueryMode::Global).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        GraphRAGError::QueryError(msg) => {
            assert!(msg.contains("No community summaries"));
        }
        other => panic!("Expected QueryError, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_graph_rag_global_query_with_communities() {
    let extraction_json = r#"{"entities":[{"name":"Alice","type":"Person","description":"A developer"},{"name":"AcmeCorp","type":"Organization","description":"A tech company"}],"relations":[{"source":"Alice","target":"AcmeCorp","type":"works_at","description":"Alice works at AcmeCorp"}]}"#;

    let llm = MockChatModel::new(extraction_json);
    let graph_rag = GraphRAG::new(llm);

    let docs = vec![Document::new("Alice works at AcmeCorp.")];
    let _: Result<(), GraphRAGError> = graph_rag.add_documents(&docs).await;
    let _: Result<(), GraphRAGError> = graph_rag.build_communities().await;

    let result: Result<query::GraphRAGResult, GraphRAGError> = graph_rag
        .query("Who works at AcmeCorp?", QueryMode::Global)
        .await;
    assert!(result.is_ok(), "global query failed: {:?}", result);

    let rag_result = result.unwrap();
    assert!(!rag_result.answer.is_empty());
    assert_eq!(rag_result.mode, QueryMode::Global);
}

#[tokio::test]
async fn test_graph_rag_hybrid_query() {
    let extraction_json = r#"{"entities":[{"name":"Alice","type":"Person","description":"A developer"},{"name":"AcmeCorp","type":"Organization","description":"A tech company"}],"relations":[{"source":"Alice","target":"AcmeCorp","type":"works_at","description":"Alice works at AcmeCorp"}]}"#;

    let llm = MockChatModel::new(extraction_json);
    let graph_rag = GraphRAG::new(llm);

    let docs = vec![Document::new("Alice works at AcmeCorp.")];
    let _: Result<(), GraphRAGError> = graph_rag.add_documents(&docs).await;
    let _: Result<(), GraphRAGError> = graph_rag.build_communities().await;

    let result: Result<query::GraphRAGResult, GraphRAGError> = graph_rag
        .query("Who works at AcmeCorp?", QueryMode::Hybrid)
        .await;
    assert!(result.is_ok(), "hybrid query failed: {:?}", result);

    let rag_result = result.unwrap();
    assert!(!rag_result.answer.is_empty());
    assert_eq!(rag_result.mode, QueryMode::Hybrid);
}

#[tokio::test]
async fn test_graph_rag_local_query_no_relevant_entities() {
    let extraction_json = r#"{"entities":[{"name":"Rust","type":"Technology","description":"A systems programming language"}],"relations":[]}"#;

    let llm = MockChatModel::new(extraction_json);
    let graph_rag = GraphRAG::new(llm);

    let docs = vec![Document::new("Rust is a systems programming language.")];
    let _: Result<(), GraphRAGError> = graph_rag.add_documents(&docs).await;

    // Query about something completely unrelated.
    let result: Result<query::GraphRAGResult, GraphRAGError> = graph_rag
        .query("cooking recipe for pasta", QueryMode::Local)
        .await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// GraphStore unit tests
// ---------------------------------------------------------------------------

#[test]
fn test_graph_store_add_entity() {
    let mut store = graph_store::GraphStore::new();
    store.add_entity(graph_store::Entity {
        id: "e1".into(),
        name: "Rust".into(),
        entity_type: "Technology".into(),
        description: "A systems programming language".into(),
    });

    assert_eq!(store.entity_count(), 1);
    let entity = store.get_entity("e1").unwrap();
    assert_eq!(entity.name, "Rust");
}

#[test]
fn test_graph_store_add_relation() {
    let mut store = graph_store::GraphStore::new();
    store.add_entity(graph_store::Entity {
        id: "e1".into(),
        name: "Rust".into(),
        entity_type: "Technology".into(),
        description: String::new(),
    });
    store.add_entity(graph_store::Entity {
        id: "e2".into(),
        name: "Mozilla".into(),
        entity_type: "Organization".into(),
        description: String::new(),
    });
    store.add_relation(graph_store::Relation {
        source: "e1".into(),
        target: "e2".into(),
        relation_type: "created_by".into(),
        description: String::new(),
        doc_id: None,
    });

    assert_eq!(store.relation_count(), 1);
    let neighbors = store.neighbors("e1");
    assert!(neighbors.contains(&"e2".to_string()));
}

#[test]
fn test_graph_store_subgraph() {
    let mut store = graph_store::GraphStore::new();
    for (id, name) in [("e1", "A"), ("e2", "B"), ("e3", "C")] {
        store.add_entity(graph_store::Entity {
            id: id.into(),
            name: name.into(),
            entity_type: "Node".into(),
            description: String::new(),
        });
    }
    store.add_relation(graph_store::Relation {
        source: "e1".into(),
        target: "e2".into(),
        relation_type: "knows".into(),
        description: String::new(),
        doc_id: None,
    });
    store.add_relation(graph_store::Relation {
        source: "e2".into(),
        target: "e3".into(),
        relation_type: "knows".into(),
        description: String::new(),
        doc_id: None,
    });

    // Depth 1 from e1: e1 + e2
    let (ents, rels) = store.subgraph("e1", 1);
    assert_eq!(ents.len(), 2);
    assert_eq!(rels.len(), 1);

    // Depth 2 from e1: e1 + e2 + e3
    let (ents, rels) = store.subgraph("e1", 2);
    assert_eq!(ents.len(), 3);
    assert_eq!(rels.len(), 2);
}

// ---------------------------------------------------------------------------
// Extractor unit tests
// ---------------------------------------------------------------------------

#[test]
fn test_parse_extraction_valid_json() {
    let raw = r#"{"entities":[{"name":"Rust","type":"Technology","description":"A language"}],"relations":[]}"#;
    let result = extractor::parse_extraction(raw).unwrap();
    assert_eq!(result.entities.len(), 1);
    assert_eq!(result.entities[0].name, "Rust");
}

#[test]
fn test_parse_extraction_markdown_wrapped() {
    let raw = "```json\n{\"entities\":[],\"relations\":[]}\n```";
    let result = extractor::parse_extraction(raw).unwrap();
    assert!(result.entities.is_empty());
}

// ---------------------------------------------------------------------------
// Community detection unit tests
// ---------------------------------------------------------------------------

#[test]
fn test_community_detection_triangle() {
    let mut store = graph_store::GraphStore::new();
    for name in ["A", "B", "C"] {
        store.add_entity(graph_store::Entity {
            id: name.to_string(),
            name: name.to_string(),
            entity_type: "Node".into(),
            description: String::new(),
        });
    }
    for (s, t) in [("A", "B"), ("B", "C"), ("C", "A")] {
        store.add_relation(graph_store::Relation {
            source: s.into(),
            target: t.into(),
            relation_type: "knows".into(),
            description: String::new(),
            doc_id: None,
        });
    }

    let communities = community::detect_communities(&store, 3);
    assert_eq!(communities.len(), 1);
    assert_eq!(communities[0].entities.len(), 3);
}

// ---------------------------------------------------------------------------
// QueryMode unit tests
// ---------------------------------------------------------------------------

#[test]
fn test_query_mode_equality() {
    assert_eq!(QueryMode::Global, QueryMode::Global);
    assert_ne!(QueryMode::Local, QueryMode::Hybrid);
}
