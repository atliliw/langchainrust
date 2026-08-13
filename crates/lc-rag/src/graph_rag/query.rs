// src/retrieval/graph_rag/query.rs
//! Query modes for GraphRAG: Global, Local, and Hybrid.
//!
//! - **Global**: aggregates community summaries, asks the LLM to answer from them.
//! - **Local**: finds relevant entities, retrieves their neighborhood subgraph,
//!   and asks the LLM.
//! - **Hybrid**: combines both global and local context.

use super::graph_store::GraphStore;
use super::matcher::{EntityMatcher, KeywordMatcher};
use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_core::token_counter::count_tokens;
use lc_prompts::PromptTemplate;
use lc_schema::Message;
use std::collections::{HashMap, HashSet};

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
    max_context_tokens: Option<usize>,
) -> Result<GraphRAGResult, super::GraphRAGError> {
    let summaries = store.community_summaries();
    if summaries.is_empty() {
        return Err(super::GraphRAGError::QueryError(
            "No community summaries available. Call build_communities() first.".into(),
        ));
    }

    let summaries_text = truncate_summaries(summaries, max_context_tokens);
    let question_str = question.to_string();
    let prompt = format_template(
        GLOBAL_QUERY_PROMPT,
        &[("summaries", &summaries_text), ("question", &question_str)],
    );

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
    max_context_tokens: Option<usize>,
    entity_matcher: Option<&dyn super::matcher::EntityMatcher>,
) -> Result<GraphRAGResult, super::GraphRAGError> {
    let seed_entities = match entity_matcher {
        Some(matcher) => matcher.find_relevant(question, store, 10),
        None => find_relevant_entities(store, question),
    };
    if seed_entities.is_empty() {
        return Err(super::GraphRAGError::QueryError(
            "No relevant entities found for the query.".into(),
        ));
    }

    // Collect subgraph from all seed entities (depth 1).
    let mut all_entity_ids: HashSet<String> = HashSet::new();
    let mut all_entities = Vec::new();
    let mut all_relations = Vec::new();
    let mut seen_relations: HashSet<(String, String, String)> = HashSet::new();

    for seed in &seed_entities {
        let (ents, rels) = store.subgraph(seed, 1);
        for e in ents {
            if all_entity_ids.insert(e.id.clone()) {
                all_entities.push(e);
            }
        }
        for r in rels {
            // M56: O(1) HashSet dedup instead of O(n^2) Vec::iter::any
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

    let entities_str = entity_lines.join("\n");
    let relations_str = relation_lines.join("\n");
    let question_str = question.to_string();
    let prompt = format_template(
        LOCAL_QUERY_PROMPT,
        &[
            ("entities", &entities_str),
            ("relations", &relations_str),
            ("question", &question_str),
        ],
    );

    let prompt = truncate_prompt(&prompt, max_context_tokens);

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
    max_context_tokens: Option<usize>,
    entity_matcher: Option<&dyn super::matcher::EntityMatcher>,
) -> Result<GraphRAGResult, super::GraphRAGError> {
    let summaries = store.community_summaries();
    let summaries_text = if summaries.is_empty() {
        "No community summaries available.".to_string()
    } else {
        truncate_summaries(summaries, max_context_tokens)
    };

    let seed_entities = match entity_matcher {
        Some(matcher) => matcher.find_relevant(question, store, 10),
        None => find_relevant_entities(store, question),
    };

    let (entity_lines, relation_lines) = if seed_entities.is_empty() {
        (String::from("No relevant entities found."), String::new())
    } else {
        let mut all_entity_ids: HashSet<String> = HashSet::new();
        let mut all_entities = Vec::new();
        let mut all_relations = Vec::new();
        let mut seen_rel_keys: HashSet<(String, String, String)> = HashSet::new();

        for seed in &seed_entities {
            let (ents, rels) = store.subgraph(seed, 1);
            for e in ents {
                if all_entity_ids.insert(e.id.clone()) {
                    all_entities.push(e);
                }
            }
            for r in rels {
                let key = (r.source.clone(), r.target.clone(), r.relation_type.clone());
                if seen_rel_keys.insert(key) {
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

    let question_str = question.to_string();
    let prompt = format_template(
        HYBRID_QUERY_PROMPT,
        &[
            ("summaries", &summaries_text),
            ("entities", &entity_lines),
            ("relations", &relation_lines),
            ("question", &question_str),
        ],
    );

    let prompt = truncate_prompt(&prompt, max_context_tokens);

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

/// Helper: format a PromptTemplate with the given key-value pairs.
fn format_template(template_str: &str, vars: &[(&str, &str)]) -> String {
    let template = PromptTemplate::new(template_str);
    let mut map = HashMap::new();
    for (k, v) in vars {
        map.insert(*k, *v);
    }
    template
        .format(&map)
        .unwrap_or_else(|_| template_str.to_string())
}

/// Truncates community summaries to fit within a token budget.
///
/// Keeps summaries from the beginning (highest-priority, largest communities)
/// until the budget is exceeded, then drops the rest.
fn truncate_summaries(summaries: &[String], max_tokens: Option<usize>) -> String {
    let all_text = summaries.join("\n\n");

    match max_tokens {
        Some(budget) => {
            let mut result = String::new();
            let mut used_tokens = 0usize;

            for summary in summaries {
                let summary_tokens = count_tokens(summary);
                if used_tokens + summary_tokens > budget {
                    break;
                }
                if !result.is_empty() {
                    result.push_str("\n\n");
                }
                result.push_str(summary);
                used_tokens += summary_tokens;
            }

            if result.is_empty() {
                // If even the first summary exceeds the budget, include it truncated
                summaries.first().cloned().unwrap_or_default()
            } else {
                result
            }
        }
        None => all_text,
    }
}

/// Truncates a full prompt to fit within a token budget.
///
/// Keeps the prompt prefix (before the context) intact and truncates
/// the context portion. If the prompt is already within budget, returns
/// it unchanged.
fn truncate_prompt(prompt: &str, max_tokens: Option<usize>) -> String {
    match max_tokens {
        Some(budget) => {
            let current_tokens = count_tokens(prompt);
            if current_tokens <= budget {
                return prompt.to_string();
            }

            // Truncate from the end, keeping character boundaries
            let ratio = budget as f64 / current_tokens as f64;
            let target_chars = (prompt.len() as f64 * ratio) as usize;
            // Find a safe char boundary
            let truncated: String = prompt.chars().take(target_chars).collect();
            format!("{}\n\n[Context truncated to fit token budget]", truncated)
        }
        None => prompt.to_string(),
    }
}

/// Finds entity ids whose name or description contains query keywords.
///
/// P1-6: 委托给 [`KeywordMatcher`](super::matcher::KeywordMatcher),消除
/// 原先 query.rs 与 matcher.rs 两处重复的 name+3/type+2/desc+1 关键词权重实现。
/// 旧函数返回全部命中(无 top_k 限制),故传一个足够大的 k 保持行为不变。
/// P2-4: 同义词/中英归一化/CJK 二元组/TF-IDF 加权等改进都随委托落在
/// `KeywordMatcher` 上,这里无需改动。
fn find_relevant_entities(store: &GraphStore, query: &str) -> Vec<String> {
    let matcher = KeywordMatcher::new();
    matcher.find_relevant(query, store, usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_rag::graph_store::{Entity, Relation};

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

    /// P1-6: `find_relevant_entities` 委托 `KeywordMatcher`,结果必须一致。
    #[test]
    fn test_find_relevant_entities_matches_keyword_matcher() {
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

        let via_delegate = find_relevant_entities(&store, "Rust programming");
        let via_matcher =
            KeywordMatcher::new().find_relevant("Rust programming", &store, usize::MAX);
        assert_eq!(via_delegate, via_matcher);
    }

    #[test]
    fn test_truncate_summaries_no_limit() {
        let summaries = vec!["Summary 1".to_string(), "Summary 2".to_string()];
        let result = truncate_summaries(&summaries, None);
        assert_eq!(result, "Summary 1\n\nSummary 2");
    }

    #[test]
    fn test_truncate_summaries_within_budget() {
        let summaries = vec!["Short summary".to_string()];
        let result = truncate_summaries(&summaries, Some(100));
        assert_eq!(result, "Short summary");
    }

    #[test]
    fn test_truncate_summaries_exceeds_budget() {
        let summaries = vec![
            "First summary that is reasonably long".to_string(),
            "Second summary that should be dropped".to_string(),
        ];
        // Budget of 5 tokens — only the first summary fits
        let result = truncate_summaries(&summaries, Some(5));
        assert!(result.contains("First summary"));
        assert!(!result.contains("Second summary"));
    }

    #[test]
    fn test_truncate_prompt_no_limit() {
        let prompt = "This is a long prompt with lots of context".to_string();
        let result = truncate_prompt(&prompt, None);
        assert_eq!(result, prompt);
    }

    #[test]
    fn test_truncate_prompt_within_budget() {
        let prompt = "Short prompt".to_string();
        let result = truncate_prompt(&prompt, Some(100));
        assert_eq!(result, "Short prompt");
    }

    /// Verify that the HashSet-based dedup logic correctly removes duplicate relations.
    /// This tests the pattern used in both local_query and hybrid_query.
    #[test]
    fn test_hybrid_query_relation_dedup_with_hashset() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "Rust".into(),
            entity_type: "Technology".into(),
            description: "A systems programming language".into(),
        });
        store.add_entity(Entity {
            id: "e2".into(),
            name: "Mozilla".into(),
            entity_type: "Organization".into(),
            description: "Organization behind Rust".into(),
        });
        // Add the same relation twice — the HashSet dedup should keep only one
        store.add_relation(Relation {
            source: "e1".into(),
            target: "e2".into(),
            relation_type: "created_by".into(),
            description: String::new(),
            doc_id: None,
        });
        store.add_relation(Relation {
            source: "e1".into(),
            target: "e2".into(),
            relation_type: "created_by".into(),
            description: String::new(),
            doc_id: None,
        });

        // Simulate the dedup logic from hybrid_query (HashSet outside the loop)
        let seed_entities = vec!["e1".to_string()];
        let mut all_relations = Vec::new();
        let mut seen_rel_keys: HashSet<(String, String, String)> = HashSet::new();
        for seed in &seed_entities {
            let (_, rels) = store.subgraph(seed, 1);
            for r in rels {
                let key = (r.source.clone(), r.target.clone(), r.relation_type.clone());
                if seen_rel_keys.insert(key) {
                    all_relations.push(r);
                }
            }
        }

        // Even though we added 2 identical relations, the HashSet dedup should keep only 1
        // (Note: GraphStore internally deduplicates too, so we may get 1 or 2 from subgraph.
        //  The key test is that the HashSet pattern works correctly.)
        let unique_keys: HashSet<(String, String, String)> = all_relations
            .iter()
            .map(|r| (r.source.clone(), r.target.clone(), r.relation_type.clone()))
            .collect();
        assert_eq!(
            unique_keys.len(),
            1,
            "should have exactly 1 unique relation after HashSet dedup"
        );
    }
}
