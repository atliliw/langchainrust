// src/retrieval/graph_rag/mod.rs
//! GraphRAG (Knowledge Graph RAG) module.
//!
//! Builds a knowledge graph from documents via LLM-based entity and relation
//! extraction, detects communities, and supports Global / Local / Hybrid
//! query modes.
//!
//! # Example
//! ```ignore
//! use langchainrust::retrieval::graph_rag::{GraphRAG, GraphRAGConfig, QueryMode};
//! use langchainrust::OpenAIChat;
//!
//! let llm = OpenAIChat::new(config);
//! let graph_rag = GraphRAG::new(llm).with_config(GraphRAGConfig::default());
//!
//! graph_rag.add_documents(&docs).await?;
//! graph_rag.build_communities().await?;
//! let result = graph_rag.query("What is Rust?", QueryMode::Local).await?;
//! println!("{}", result.answer);
//! ```

pub mod community;
pub mod extractor;
pub mod graph_store;
pub mod query;

pub use graph_store::{Community, Entity, GraphStore, Relation};
pub use query::{GraphRAGResult, QueryMode};

use crate::core::language_models::BaseChatModel;
use crate::vector_stores::Document;
use tokio::sync::RwLock;

/// GraphRAG error type.
#[derive(Debug, thiserror::Error)]
pub enum GraphRAGError {
    #[error("LLM error: {0}")]
    LLMError(String),

    #[error("Extraction error: {0}")]
    ExtractionError(String),

    #[error("Query error: {0}")]
    QueryError(String),

    #[error("Community error: {0}")]
    CommunityError(String),
}

/// Configuration for GraphRAG.
pub struct GraphRAGConfig {
    pub max_entities_per_doc: usize,
    pub max_relations_per_doc: usize,
    pub community_max_levels: usize,
}

impl Default for GraphRAGConfig {
    fn default() -> Self {
        Self {
            max_entities_per_doc: 10,
            max_relations_per_doc: 10,
            community_max_levels: 3,
        }
    }
}

impl GraphRAGConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_entities_per_doc(mut self, n: usize) -> Self {
        self.max_entities_per_doc = n;
        self
    }

    pub fn with_max_relations_per_doc(mut self, n: usize) -> Self {
        self.max_relations_per_doc = n;
        self
    }

    pub fn with_community_max_levels(mut self, n: usize) -> Self {
        self.community_max_levels = n;
        self
    }
}

/// GraphRAG: Knowledge Graph-based Retrieval Augmented Generation.
///
/// Wraps an LLM for entity/relation extraction and community summarization,
/// and an in-memory [`GraphStore`] for graph operations.
pub struct GraphRAG<M: BaseChatModel> {
    llm: M,
    store: RwLock<GraphStore>,
    config: GraphRAGConfig,
}

impl<M: BaseChatModel> GraphRAG<M> {
    /// Creates a new GraphRAG instance with the given LLM.
    pub fn new(llm: M) -> Self {
        Self {
            llm,
            store: RwLock::new(GraphStore::new()),
            config: GraphRAGConfig::default(),
        }
    }

    /// Sets a custom configuration.
    pub fn with_config(mut self, config: GraphRAGConfig) -> Self {
        self.config = config;
        self
    }

    /// Adds documents to the knowledge graph by extracting entities and
    /// relations from each document via the LLM.
    pub async fn add_documents(&self, docs: &[Document]) -> Result<(), GraphRAGError> {
        for doc in docs {
            let extraction = extractor::extract(
                &self.llm,
                &doc.content,
                self.config.max_entities_per_doc,
                self.config.max_relations_per_doc,
            )
            .await?;

            let doc_id = doc.id.clone();
            let mut store = self.store.write().await;

            // Build a name-to-id map for deduplication.
            let mut name_to_id: std::collections::HashMap<String, String> = store
                .all_entities()
                .values()
                .map(|e| (e.name.to_lowercase(), e.id.clone()))
                .collect();

            // Insert extracted entities (deduplicate by name).
            for ext_ent in &extraction.entities {
                let key = ext_ent.name.to_lowercase();
                if let Some(_existing_id) = name_to_id.get(&key) {
                    // Entity already exists; skip (M57: log instead of silent discard).
                    log::info!("GraphRAG: skipping duplicate entity '{}'", ext_ent.name);
                    continue;
                }

                let id = format!("e_{}", uuid::Uuid::new_v4().as_simple());
                name_to_id.insert(key, id.clone());

                store.add_entity(Entity {
                    id,
                    name: ext_ent.name.clone(),
                    entity_type: ext_ent.entity_type.clone(),
                    description: ext_ent.description.clone(),
                });
            }

            // Insert extracted relations (resolve names to ids).
            for ext_rel in &extraction.relations {
                let source_key = ext_rel.source.to_lowercase();
                let target_key = ext_rel.target.to_lowercase();

                let source_id = match name_to_id.get(&source_key) {
                    Some(id) => id.clone(),
                    None => {
                        log::info!(
                            "GraphRAG: skipping relation with unknown source entity '{}'",
                            ext_rel.source
                        );
                        continue;
                    }
                };
                let target_id = match name_to_id.get(&target_key) {
                    Some(id) => id.clone(),
                    None => {
                        log::info!(
                            "GraphRAG: skipping relation with unknown target entity '{}'",
                            ext_rel.target
                        );
                        continue;
                    }
                };

                store.add_relation(Relation {
                    source: source_id,
                    target: target_id,
                    relation_type: ext_rel.relation_type.clone(),
                    description: ext_rel.description.clone(),
                    doc_id: doc_id.clone(),
                });
            }
        }

        Ok(())
    }

    /// Runs community detection and generates community summaries via the LLM.
    pub async fn build_communities(&self) -> Result<(), GraphRAGError> {
        let communities = {
            let store = self.store.read().await;
            community::detect_communities(&store, self.config.community_max_levels)
        };

        // Generate summaries for each community.
        let mut summaries = Vec::with_capacity(communities.len());
        for comm in &communities {
            let store_clone = {
                let store = self.store.read().await;
                store.clone()
            };
            let summary = community::summarize_community(&self.llm, &store_clone, comm).await?;
            summaries.push(summary);
        }

        // Write communities and summaries back.
        let mut store = self.store.write().await;
        store.set_communities(communities);
        store.set_community_summaries(summaries);

        Ok(())
    }

    /// Queries the knowledge graph using the specified mode.
    pub async fn query(&self, q: &str, mode: QueryMode) -> Result<GraphRAGResult, GraphRAGError> {
        let store = {
            let guard = self.store.read().await;
            guard.clone()
        };

        match mode {
            QueryMode::Global => query::global_query(&self.llm, &store, q).await,
            QueryMode::Local => query::local_query(&self.llm, &store, q).await,
            QueryMode::Hybrid => query::hybrid_query(&self.llm, &store, q).await,
        }
    }

    /// Returns the number of entities in the graph.
    pub async fn entity_count(&self) -> usize {
        let store = self.store.read().await;
        store.entity_count()
    }

    /// Returns the number of relations in the graph.
    pub async fn relation_count(&self) -> usize {
        let store = self.store.read().await;
        store.relation_count()
    }

    /// Returns the number of communities.
    pub async fn community_count(&self) -> usize {
        let store = self.store.read().await;
        store.communities().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_rag_config_default() {
        let config = GraphRAGConfig::default();
        assert_eq!(config.max_entities_per_doc, 10);
        assert_eq!(config.max_relations_per_doc, 10);
        assert_eq!(config.community_max_levels, 3);
    }

    #[test]
    fn test_graph_rag_config_builder() {
        let config = GraphRAGConfig::new()
            .with_max_entities_per_doc(5)
            .with_max_relations_per_doc(8)
            .with_community_max_levels(2);

        assert_eq!(config.max_entities_per_doc, 5);
        assert_eq!(config.max_relations_per_doc, 8);
        assert_eq!(config.community_max_levels, 2);
    }

    #[test]
    fn test_graph_error_display() {
        let err = GraphRAGError::LLMError("timeout".into());
        assert!(err.to_string().contains("timeout"));

        let err = GraphRAGError::ExtractionError("bad json".into());
        assert!(err.to_string().contains("bad json"));

        let err = GraphRAGError::QueryError("no entities".into());
        assert!(err.to_string().contains("no entities"));
    }
}
