// src/retrieval/graph_rag/query.rs
//! Query modes for GraphRAG: Global, Local, and Hybrid.
//!
//! - **Global**: aggregates community summaries, asks the LLM to answer from them.
//! - **Local**: finds relevant entities, retrieves their neighborhood subgraph,
//!   and asks the LLM.
//! - **Hybrid**: combines both global and local context.

use super::graph_store::GraphStore;
use crate::core::language_models::{BaseChatModel, LLMResult};
use crate::schema::Message;
use std::collections::HashSet;

/// Helper: format a relation using entity names instead of IDs.
fn format_relation(r: &super::graph_store::Relation, store: &GraphStore) -> String {
    let source_name = store
        .get_entity(&r.source)
        .map(|e| e.name.as_str())
        .unwrap_or(&r.source);
    let target_name = store
        .get_entity(&r.target)
        .map(|e| e.name.as_str())
        .unwrap_or(&r.target);
    format!(
        "- {} --[{}]--> {}{}",
        source_name,
        r.relation_type,
        target_name,
        if r.description.is_empty() {
            String::new()
        } else {
            format!(": {}", r.description)
        }
    )
}

/// Query mode selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    Global,
    Local,
    Hybrid,
}

/// Result of a GraphRAG query.
#[derive(Debug, Clone)]
pub struct GraphRAGResult {
    pub answer: String,
    pub sources: Vec<String>,
    pub mode: QueryMode,
}

const GLOBAL_QUERY_PROMPT: &str = r#"You are a helpful assistant answering questions based on community summaries from a knowledge graph.

Community Summaries:
{summaries}

Question: {question}

Provide a comprehensive answer based on the community summaries above. If the summaries do not contain enough information, say so.

Answer:"#;

const LOCAL_QUERY_PROMPT: &str = r#"You are a helpful assistant answering questions based on a local subgraph from a knowledge graph.

Entities:
{entities}

Relations:
{relations}

Question: {question}

Provide a detailed answer based on the local subgraph information above. If the subgraph does not contain enough information, say so.

Answer:"#;

const HYBRID_QUERY_PROMPT: &str = r#"You are a helpful assistant answering questions based on both community summaries and a local subgraph from a knowledge graph.

Community Summaries:
{summaries}

Local Subgraph Entities:
{entities}

Local Subgraph Relations:
{relations}

Question: {question}

Provide a comprehensive answer synthesizing both the community-level and local-level information. If there is not enough information, say so.

Answer:"#;

/// Executes a **global** query: aggregates community summaries and asks the LLM.
pub async fn global_query<M: BaseChatModel>(
    llm: &M,
    store: &GraphStore,
    question: &str,
) -> Result<GraphRAGResult, super::GraphRAGError> {
    let summaries = store.community_summaries();
    if summaries.is_empty() {
        return Err(super::GraphRAGError::QueryError(
            "No community summaries available. Call build_communities() first.".into(),
        ));
    }

    let summaries_text = summaries.join("\n\n");
    let prompt = GLOBAL_QUERY_PROMPT
        .replace("{summaries}", &summaries_text)
        .replace("{question}", question);

    let messages = vec![Message::human(prompt)];
    let response: LLMResult = llm
        .chat(messages, None)
        .await
        .map_err(|e| super::GraphRAGError::LLMError(e.to_string()))?;

    let sources: Vec<String> = store
        .communities()
        .iter()
        .flat_map(|c| c.entities.iter().cloned())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    Ok(GraphRAGResult {
        answer: response.content.trim().to_string(),
        sources,
        mode: QueryMode::Global,
    })
}

/// Executes a **local** query: finds relevant entities by keyword match,
/// retrieves their neighborhood subgraph, and asks the LLM.
pub async fn local_query<M: BaseChatModel>(
    llm: &M,
    store: &GraphStore,
    question: &str,
) -> Result<GraphRAGResult, super::GraphRAGError> {
    let seed_entities = find_relevant_entities(store, question);
    if seed_entities.is_empty() {
        return Err(super::GraphRAGError::QueryError(
            "No relevant entities found for the query.".into(),
        ));
    }

    // Collect subgraph from all seed entities (depth 1).
    let mut all_entity_ids: HashSet<String> = HashSet::new();
    let mut all_entities = Vec::new();
    let mut all_relations = Vec::new();

    for seed in &seed_entities {
        let (ents, rels) = store.subgraph(seed, 1);
        for e in ents {
            if all_entity_ids.insert(e.id.clone()) {
                all_entities.push(e);
            }
        }
        for r in rels {
            // M56: O(1) HashSet dedup instead of O(n^2) Vec::iter::any
            let mut seen_relations: HashSet<(String, String, String)> = HashSet::new();
            let key = (r.source.clone(), r.target.clone(), r.relation_type.clone());
            if seen_relations.insert(key) {
                all_relations.push(r);
            }
        }
    }

    let entity_lines: Vec<String> = all_entities
        .iter()
        .map(|e| format!("- {} ({}): {}", e.name, e.entity_type, e.description))
        .collect();

    let relation_lines: Vec<String> = all_relations
        .iter()
        .map(|r| format_relation(r, store))
        .collect();

    let prompt = LOCAL_QUERY_PROMPT
        .replace("{entities}", &entity_lines.join("\n"))
        .replace("{relations}", &relation_lines.join("\n"))
        .replace("{question}", question);

    let messages = vec![Message::human(prompt)];
    let response: LLMResult = llm
        .chat(messages, None)
        .await
        .map_err(|e| super::GraphRAGError::LLMError(e.to_string()))?;

    Ok(GraphRAGResult {
        answer: response.content.trim().to_string(),
        sources: seed_entities,
        mode: QueryMode::Local,
    })
}

/// Executes a **hybrid** query: combines global community summaries with
/// local subgraph context.
pub async fn hybrid_query<M: BaseChatModel>(
    llm: &M,
    store: &GraphStore,
    question: &str,
) -> Result<GraphRAGResult, super::GraphRAGError> {
    let summaries = store.community_summaries();
    let summaries_text = if summaries.is_empty() {
        "No community summaries available.".to_string()
    } else {
        summaries.join("\n\n")
    };

    let seed_entities = find_relevant_entities(store, question);

    let (entity_lines, relation_lines) = if seed_entities.is_empty() {
        (String::from("No relevant entities found."), String::new())
    } else {
        let mut all_entity_ids: HashSet<String> = HashSet::new();
        let mut all_entities = Vec::new();
        let mut all_relations = Vec::new();

        for seed in &seed_entities {
            let (ents, rels) = store.subgraph(seed, 1);
            for e in ents {
                if all_entity_ids.insert(e.id.clone()) {
                    all_entities.push(e);
                }
            }
            for r in rels {
                if !all_relations
                    .iter()
                    .any(|existing: &super::graph_store::Relation| {
                        existing.source == r.source
                            && existing.target == r.target
                            && existing.relation_type == r.relation_type
                    })
                {
                    all_relations.push(r);
                }
            }
        }

        let el: Vec<String> = all_entities
            .iter()
            .map(|e| format!("- {} ({}): {}", e.name, e.entity_type, e.description))
            .collect();
        let rl: Vec<String> = all_relations
            .iter()
            .map(|r| format_relation(r, store))
            .collect();
        (el.join("\n"), rl.join("\n"))
    };

    let prompt = HYBRID_QUERY_PROMPT
        .replace("{summaries}", &summaries_text)
        .replace("{entities}", &entity_lines)
        .replace("{relations}", &relation_lines)
        .replace("{question}", question);

    let messages = vec![Message::human(prompt)];
    let response: LLMResult = llm
        .chat(messages, None)
        .await
        .map_err(|e| super::GraphRAGError::LLMError(e.to_string()))?;

    let mut sources: Vec<String> = seed_entities;
    if !summaries.is_empty() {
        sources.push(format!("{} community summaries", summaries.len()));
    }

    Ok(GraphRAGResult {
        answer: response.content.trim().to_string(),
        sources,
        mode: QueryMode::Hybrid,
    })
}

/// Finds entity ids whose name or description contains query keywords.
fn find_relevant_entities(store: &GraphStore, query: &str) -> Vec<String> {
    let query_lower = query.to_lowercase();
    // Split on whitespace and strip common punctuation so "Rust?" matches "Rust".
    let keywords: Vec<&str> = query_lower
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();

    let mut scored: Vec<(String, usize)> = Vec::new();

    for (id, entity) in store.all_entities() {
        let name_lower = entity.name.to_lowercase();
        let desc_lower = entity.description.to_lowercase();
        let type_lower = entity.entity_type.to_lowercase();

        let mut score = 0usize;
        for kw in &keywords {
            if name_lower.contains(kw) {
                score += 3; // name match is strongest
            }
            if type_lower.contains(kw) {
                score += 2;
            }
            if desc_lower.contains(kw) {
                score += 1;
            }
        }

        if score > 0 {
            scored.push((id.clone(), score));
        }
    }

    scored.sort_by(|a, b| b.1.cmp(&a.1));
    scored.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::graph_rag::graph_store::{Entity, Relation};

    #[test]
    fn test_find_relevant_entities() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "Rust".into(),
            entity_type: "Technology".into(),
            description: "A systems programming language".into(),
        });
        store.add_entity(Entity {
            id: "e2".into(),
            name: "Python".into(),
            entity_type: "Technology".into(),
            description: "A scripting language".into(),
        });
        store.add_entity(Entity {
            id: "e3".into(),
            name: "Alice".into(),
            entity_type: "Person".into(),
            description: "A developer who uses Rust".into(),
        });

        let results = find_relevant_entities(&store, "Rust programming");
        assert!(!results.is_empty());
        // "Rust" entity should rank first (name match + description match)
        assert_eq!(results[0], "e1");
    }

    #[test]
    fn test_find_relevant_entities_no_match() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "Rust".into(),
            entity_type: "Technology".into(),
            description: "A systems programming language".into(),
        });

        let results = find_relevant_entities(&store, "cooking recipe");
        assert!(results.is_empty());
    }
}
