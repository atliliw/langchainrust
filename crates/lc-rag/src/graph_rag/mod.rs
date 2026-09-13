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
pub mod leiden;
pub mod matcher;
pub mod query;

pub use graph_store::{Community, Entity, GraphStore, Relation};
pub use matcher::{EmbeddingMatcher, EntityMatcher, KeywordMatcher};
pub use query::{GlobalLevel, GraphRAGResult, QueryMode};

use lc_core::language_models::BaseChatModel;
use lc_vector_stores::Document;
use tokio::sync::RwLock;

/// GraphRAG error type.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GraphRAGError {
    /// An error from the underlying LLM call.
    #[error("LLM error: {0}")]
    LLMError(String),

    /// An error during entity/relation extraction.
    #[error("Extraction error: {0}")]
    ExtractionError(String),

    /// An error during query execution.
    #[error("Query error: {0}")]
    QueryError(String),

    /// An error during community detection or summarization.
    #[error("Community error: {0}")]
    CommunityError(String),
}

/// Configuration for GraphRAG.
pub struct GraphRAGConfig {
    /// Maximum number of entities to extract per document.
    pub max_entities_per_doc: usize,
    /// Maximum number of relations to extract per document.
    pub max_relations_per_doc: usize,
    /// Leiden modularity resolution γ for community detection
    /// (see [`community::DEFAULT_RESOLUTION`]). Higher values produce more,
    /// smaller communities; lower values merge more aggressively.
    pub leiden_resolution: f64,
    /// Deterministic RNG seed for the Leiden algorithm. Fixed by default so
    /// rebuilding communities over the same graph yields the same partition.
    pub leiden_seed: u64,
    /// Maximum number of hierarchy levels, including level 0. Each higher
    /// level aggregates communities of the level below (see [`Community`]).
    pub max_community_levels: usize,
    /// Maximum number of tokens for context in query prompts.
    /// When set, community summaries or subgraph context is truncated to fit.
    pub max_context_tokens: Option<usize>,
    /// Custom entity matcher for local/hybrid queries.
    /// When None, uses the default KeywordMatcher.
    pub entity_matcher: Option<Box<dyn EntityMatcher>>,
}

impl Default for GraphRAGConfig {
    fn default() -> Self {
        Self {
            max_entities_per_doc: 10,
            max_relations_per_doc: 10,
            leiden_resolution: community::DEFAULT_RESOLUTION,
            leiden_seed: community::DEFAULT_SEED,
            max_community_levels: community::DEFAULT_MAX_LEVELS,
            max_context_tokens: None,
            entity_matcher: None,
        }
    }
}

impl GraphRAGConfig {
    /// Creates a `GraphRAGConfig` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of entities to extract per document.
    pub fn with_max_entities_per_doc(mut self, n: usize) -> Self {
        self.max_entities_per_doc = n;
        self
    }

    /// Sets the maximum number of relations to extract per document.
    pub fn with_max_relations_per_doc(mut self, n: usize) -> Self {
        self.max_relations_per_doc = n;
        self
    }

    /// Sets the Leiden modularity resolution γ.
    pub fn with_leiden_resolution(mut self, resolution: f64) -> Self {
        self.leiden_resolution = resolution;
        self
    }

    /// Sets the deterministic RNG seed used by Leiden community detection.
    pub fn with_leiden_seed(mut self, seed: u64) -> Self {
        self.leiden_seed = seed;
        self
    }

    /// Sets the maximum hierarchy depth (level 0 is the base partition).
    pub fn with_max_community_levels(mut self, levels: usize) -> Self {
        self.max_community_levels = levels;
        self
    }

    /// Sets the maximum number of tokens for context in query prompts.
    ///
    /// When set, community summaries (Global/Hybrid) or subgraph context
    /// (Local/Hybrid) are truncated from lowest-priority items to fit
    /// within this budget.
    pub fn with_max_context_tokens(mut self, tokens: usize) -> Self {
        self.max_context_tokens = Some(tokens);
        self
    }

    /// Sets a custom entity matcher for local/hybrid queries.
    ///
    /// When set, the matcher is used instead of the default keyword-based
    /// matching to find relevant entities.
    pub fn with_entity_matcher(mut self, matcher: Box<dyn EntityMatcher>) -> Self {
        self.entity_matcher = Some(matcher);
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

    /// Runs hierarchical Leiden community detection and generates one
    /// summary per community via the LLM.
    ///
    /// Level-0 communities are summarized from their entities and internal
    /// relations; each higher level is rolled up from its child-community
    /// summaries plus the relations crossing the children. Community ids
    /// index the summary vector one-to-one.
    pub async fn build_communities(&self) -> Result<(), GraphRAGError> {
        let communities = {
            let store = self.store.read().await;
            community::detect_hierarchy(
                &store,
                self.config.leiden_resolution,
                self.config.max_community_levels,
                self.config.leiden_seed,
            )?
        };

        // Generate summaries in id order; children always precede their
        // parents because ids are assigned level by level.
        let mut summaries: Vec<String> = Vec::with_capacity(communities.len());
        for comm in &communities {
            let store_clone = {
                let store = self.store.read().await;
                store.clone()
            };
            let summary = if comm.level == 0 {
                community::summarize_community(&self.llm, &store_clone, comm).await?
            } else {
                let child_summaries: Vec<String> = communities
                    .iter()
                    .filter(|child| child.parent == Some(comm.id))
                    .map(|child| summaries[child.id].clone())
                    .collect();
                community::summarize_rollup(
                    &self.llm,
                    &store_clone,
                    comm,
                    &communities,
                    &child_summaries,
                )
                .await?
            };
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

        let max_tokens = self.config.max_context_tokens;

        match mode {
            QueryMode::Global => {
                query::global_query(&self.llm, &store, q, max_tokens, GlobalLevel::Coarsest).await
            }
            QueryMode::GlobalAt(level) => {
                query::global_query(&self.llm, &store, q, max_tokens, level).await
            }
            QueryMode::Local => {
                let matcher = self.config.entity_matcher.as_deref();
                query::local_query(&self.llm, &store, q, max_tokens, matcher).await
            }
            QueryMode::Hybrid => {
                let matcher = self.config.entity_matcher.as_deref();
                query::hybrid_query(
                    &self.llm,
                    &store,
                    q,
                    max_tokens,
                    matcher,
                    GlobalLevel::Coarsest,
                )
                .await
            }
            QueryMode::HybridAt(level) => {
                let matcher = self.config.entity_matcher.as_deref();
                query::hybrid_query(&self.llm, &store, q, max_tokens, matcher, level).await
            }
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

    /// Returns the number of communities across all hierarchy levels.
    pub async fn community_count(&self) -> usize {
        let store = self.store.read().await;
        store.communities().len()
    }

    /// Returns a clone of the detected community hierarchy (empty until
    /// [`GraphRAG::build_communities`] has run).
    pub async fn communities(&self) -> Vec<Community> {
        let store = self.store.read().await;
        store.communities().to_vec()
    }

    /// Returns one summary per community, indexed by [`Community::id`].
    pub async fn community_summaries(&self) -> Vec<String> {
        let store = self.store.read().await;
        store.community_summaries().to_vec()
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
        assert_eq!(config.leiden_resolution, community::DEFAULT_RESOLUTION);
        assert_eq!(config.leiden_seed, community::DEFAULT_SEED);
        assert_eq!(config.max_community_levels, community::DEFAULT_MAX_LEVELS);
        assert!(config.max_context_tokens.is_none());
    }

    #[test]
    fn test_graph_rag_config_builder() {
        let config = GraphRAGConfig::new()
            .with_max_entities_per_doc(5)
            .with_max_relations_per_doc(8)
            .with_leiden_resolution(0.5)
            .with_leiden_seed(99)
            .with_max_community_levels(2);

        assert_eq!(config.max_entities_per_doc, 5);
        assert_eq!(config.max_relations_per_doc, 8);
        assert_eq!(config.leiden_resolution, 0.5);
        assert_eq!(config.leiden_seed, 99);
        assert_eq!(config.max_community_levels, 2);
    }

    #[test]
    fn test_graph_error_display() {
        let err = GraphRAGError::LLMError("timeout".into());
        assert!(err.to_string().contains("timeout"));

        let err = GraphRAGError::ExtractionError("bad json".into());
        assert!(err.to_string().contains("bad json"));

        let err = GraphRAGError::QueryError("no entities".into());
        assert!(err.to_string().contains("no entities"));

        let err = GraphRAGError::CommunityError("bad level".into());
        assert!(err.to_string().contains("bad level"));
    }

    // -- End-to-end hierarchy + level-aware query tests -------------------

    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::language_models::{LLMResult, StreamChunk};
    use lc_core::runnables::RunnableConfig;
    use lc_core::{BaseLanguageModel, Runnable};
    use lc_schema::Message;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    type PromptLog = Arc<Mutex<Vec<String>>>;

    /// Fake chat model that distinguishes base summaries, level-1 rollups and
    /// query answers from the prompt text, and records every prompt sent.
    struct ScriptedChatModel {
        prompts: PromptLog,
    }

    impl ScriptedChatModel {
        fn new(prompts: PromptLog) -> Self {
            Self { prompts }
        }
    }

    fn last_prompt(prompts: &PromptLog) -> String {
        prompts.lock().unwrap().last().unwrap().clone()
    }

    fn clear_prompts(prompts: &PromptLog) {
        prompts.lock().unwrap().clear();
    }

    #[derive(Debug)]
    struct ScriptedChatError;
    impl std::fmt::Display for ScriptedChatError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "scripted mock chat error")
        }
    }
    impl std::error::Error for ScriptedChatError {}

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for ScriptedChatModel {
        type Error = ScriptedChatError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(ScriptedChatError)
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for ScriptedChatModel {
        fn model_name(&self) -> &str {
            "graphrag-e2e-mock"
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
    impl BaseChatModel for ScriptedChatModel {
        async fn chat(
            &self,
            messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let prompt = messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            let reply = if prompt.contains("building a level-1 overview") {
                "ROLLUP_SUMMARY"
            } else if prompt.contains("You are a helpful assistant answering questions") {
                "QUESTION_ANSWER"
            } else {
                "BASE_SUMMARY"
            };
            self.prompts.lock().unwrap().push(prompt);
            Ok(LLMResult {
                content: reply.to_string(),
                model: "graphrag-e2e-mock".to_string(),
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
            Err(ScriptedChatError)
        }
    }

    /// Two triangles linked by one bridge, two 8-cliques linked the same way:
    /// level 0 yields four communities; only the two triangles roll into a
    /// level-1 group (the dense cliques stay unmerged).
    fn populate_hierarchical_fixture(store: &mut GraphStore) {
        let groups: [Vec<usize>; 4] = [
            (0..3).collect(),
            (3..6).collect(),
            (6..14).collect(),
            (14..22).collect(),
        ];
        for group in &groups {
            for &i in group {
                let name = format!("n{i}");
                store.add_entity(Entity {
                    id: name.clone(),
                    name: name.to_uppercase(),
                    entity_type: "concept".to_string(),
                    description: format!("Entity {}", name.to_uppercase()),
                });
            }
            for (ai, &a) in group.iter().enumerate() {
                for &b in &group[ai + 1..] {
                    store.add_relation(Relation {
                        source: format!("n{a}"),
                        target: format!("n{b}"),
                        relation_type: "rel".to_string(),
                        description: String::new(),
                        doc_id: None,
                    });
                }
            }
        }
        store.add_relation(Relation {
            source: "n2".into(),
            target: "n3".into(),
            relation_type: "bridge".into(),
            description: String::new(),
            doc_id: None,
        });
        store.add_relation(Relation {
            source: "n13".into(),
            target: "n14".into(),
            relation_type: "bridge".into(),
            description: String::new(),
            doc_id: None,
        });
    }

    fn count_occurrences(haystack: &str, needle: &str) -> usize {
        haystack.matches(needle).count()
    }

    #[tokio::test]
    async fn e2e_hierarchy_build_and_level_aware_queries() {
        let prompts: PromptLog = Arc::new(Mutex::new(Vec::new()));
        let rag = GraphRAG::new(ScriptedChatModel::new(prompts.clone())).with_config(
            GraphRAGConfig::new()
                .with_leiden_resolution(community::DEFAULT_RESOLUTION)
                .with_max_community_levels(3),
        );
        {
            let mut store = rag.store.write().await;
            populate_hierarchical_fixture(&mut store);
        }

        rag.build_communities().await.unwrap();
        clear_prompts(&prompts);

        // Hierarchy: four level-0 communities + one level-1 rollup.
        let communities = rag.communities().await;
        assert_eq!(communities.len(), 5);
        assert_eq!(communities.iter().filter(|c| c.level == 0).count(), 4);
        let level1: Vec<&Community> = communities.iter().filter(|c| c.level == 1).collect();
        assert_eq!(level1.len(), 1);
        assert_eq!(level1[0].entities.len(), 6);
        assert!(level1[0].parent.is_none());
        let parents: Vec<usize> = communities.iter().filter_map(|c| c.parent).collect();
        assert_eq!(parents, vec![level1[0].id, level1[0].id]);

        // Summaries are parallel to ids: 4 base + 1 rollup.
        let summaries = rag.community_summaries().await;
        assert_eq!(summaries.len(), 5);
        assert_eq!(
            summaries
                .iter()
                .filter(|s| s.as_str() == "BASE_SUMMARY")
                .count(),
            4
        );
        assert_eq!(summaries[level1[0].id], "ROLLUP_SUMMARY");

        // Global @ coarsest: two unparented clique communities + the rollup;
        // every one of the 22 entities is covered exactly once.
        let result = rag
            .query("overview please", QueryMode::Global)
            .await
            .unwrap();
        assert_eq!(result.answer, "QUESTION_ANSWER");
        assert_eq!(result.mode, QueryMode::Global);
        assert_eq!(result.sources.len(), 22);
        let prompt = last_prompt(&prompts);
        assert_eq!(count_occurrences(&prompt, "ROLLUP_SUMMARY"), 1);
        assert_eq!(count_occurrences(&prompt, "BASE_SUMMARY"), 2);
        clear_prompts(&prompts);

        // Global @ level 0: four base summaries, no rollup.
        let result = rag
            .query("fine detail", QueryMode::GlobalAt(GlobalLevel::Level(0)))
            .await
            .unwrap();
        assert_eq!(result.mode, QueryMode::GlobalAt(GlobalLevel::Level(0)));
        let prompt = last_prompt(&prompts);
        assert_eq!(count_occurrences(&prompt, "BASE_SUMMARY"), 4);
        assert_eq!(count_occurrences(&prompt, "ROLLUP_SUMMARY"), 0);
        assert_eq!(result.sources.len(), 22);
        clear_prompts(&prompts);

        // Global @ all: four base + one rollup.
        let result = rag
            .query("everything", QueryMode::GlobalAt(GlobalLevel::All))
            .await
            .unwrap();
        assert_eq!(result.answer, "QUESTION_ANSWER");
        assert_eq!(result.mode, QueryMode::GlobalAt(GlobalLevel::All));
        let prompt = last_prompt(&prompts);
        assert_eq!(count_occurrences(&prompt, "BASE_SUMMARY"), 4);
        assert_eq!(count_occurrences(&prompt, "ROLLUP_SUMMARY"), 1);
        clear_prompts(&prompts);

        // Global @ missing level fails fast.
        let err = rag
            .query("ghost", QueryMode::GlobalAt(GlobalLevel::Level(9)))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("level 9"), "{err}");

        // Local query still answers from the subgraph.
        let result = rag.query("n0", QueryMode::Local).await.unwrap();
        assert_eq!(result.mode, QueryMode::Local);
        assert_eq!(result.answer, "QUESTION_ANSWER");
        assert!(result.sources.contains(&"n0".to_string()));
        clear_prompts(&prompts);

        // Hybrid @ coarsest: rollup summary plus local subgraph lines.
        let result = rag.query("n0", QueryMode::Hybrid).await.unwrap();
        assert_eq!(result.mode, QueryMode::Hybrid);
        let prompt = last_prompt(&prompts);
        assert!(prompt.contains("ROLLUP_SUMMARY"));
        assert!(prompt.contains("N0 (concept)"));
    }

    #[tokio::test]
    async fn global_query_without_communities_is_an_error() {
        let prompts: PromptLog = Arc::new(Mutex::new(Vec::new()));
        let rag = GraphRAG::new(ScriptedChatModel::new(prompts));
        {
            let mut store = rag.store.write().await;
            store.add_entity(Entity {
                id: "x".into(),
                name: "X".into(),
                entity_type: "concept".into(),
                description: "Entity X".into(),
            });
        }
        let err = rag.query("q", QueryMode::Global).await.unwrap_err();
        assert!(err.to_string().contains("build_communities"));
    }
}
