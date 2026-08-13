// src/retrieval/graph_rag/community.rs
//! Community detection using label propagation.
//!
//! Implements a simple iterative label-propagation algorithm that groups
//! entities into communities based on graph connectivity. Also provides
//! LLM-based community summary generation.

use super::graph_store::{Community, GraphStore};
use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_schema::Message;
use std::collections::{HashMap, HashSet};

/// Runs label-propagation community detection on the graph store.
///
/// Returns a list of communities (non-singleton groups only) sorted
/// largest-first.
pub fn detect_communities(store: &GraphStore, num_tiers: usize) -> Vec<Community> {
    let entity_ids: Vec<String> = store.entity_ids();
    if entity_ids.is_empty() {
        return Vec::new();
    }

    // Initialize: each entity is its own label.
    let mut labels: HashMap<String, usize> = HashMap::with_capacity(entity_ids.len());
    for (i, id) in entity_ids.iter().enumerate() {
        labels.insert(id.clone(), i);
    }

    // Iterate label propagation.
    for _ in 0..num_tiers * 10 {
        let mut changed = false;
        for eid in &entity_ids {
            let neighbors = store.neighbors(eid);
            if neighbors.is_empty() {
                continue;
            }

            // Count label frequencies among neighbors.
            let mut freq: HashMap<usize, usize> = HashMap::new();
            for nb in &neighbors {
                if let Some(&lbl) = labels.get(nb) {
                    *freq.entry(lbl).or_insert(0) += 1;
                }
            }

            // Pick the most frequent label (ties broken by lowest label id for determinism).
            if let Some((&best_label, _)) = freq
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            {
                if labels.get(eid) != Some(&best_label) {
                    labels.insert(eid.clone(), best_label);
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Group entities by label.
    let mut groups: HashMap<usize, Vec<String>> = HashMap::new();
    for (eid, &lbl) in &labels {
        groups.entry(lbl).or_default().push(eid.clone());
    }

    // Build community list (skip singletons).
    let mut communities: Vec<Community> = groups
        .into_iter()
        .filter(|(_, members)| members.len() > 1)
        .enumerate()
        .map(|(cid, (_, members))| Community {
            id: cid,
            entities: members,
            level: 0,
        })
        .collect();

    // Sort largest first.
    communities.sort_by(|a, b| b.entities.len().cmp(&a.entities.len()));

    // Assign size-tier buckets (NOT hierarchical parent-child levels).
    assign_size_tiers(&mut communities, num_tiers);

    communities
}

/// Assigns size-tier buckets to communities.
///
/// 注意:这是**大小分桶**——按社区大小排名均分为 `num_tiers` 档,
/// `Community::level` 的含义是"第几档大小",**不是**经典的层级社区
/// (逐层合并出父-子包含关系)。命名已诚实化(P1-3)。
fn assign_size_tiers(communities: &mut [Community], num_tiers: usize) {
    if communities.is_empty() || num_tiers <= 1 {
        return;
    }

    // Sort by size descending, then assign tiers based on size buckets.
    let n = communities.len();
    let tier_size = (n as f64 / num_tiers as f64).ceil() as usize;
    if tier_size == 0 {
        return;
    }

    for (i, community) in communities.iter_mut().enumerate() {
        community.level = std::cmp::min(i / tier_size, num_tiers - 1);
    }
}

const COMMUNITY_SUMMARY_PROMPT: &str = r#"You are a knowledge graph summarization assistant.

Given the following entities and their relations within a community, write a concise summary (2-3 sentences) that captures the key information and relationships.

Entities:
{entities}

Relations:
{relations}

Summary:"#;

/// Generates a summary for a single community using the LLM.
pub async fn summarize_community<M: BaseChatModel>(
    llm: &M,
    store: &GraphStore,
    community: &Community,
) -> Result<String, super::GraphRAGError> {
    let entity_lines: Vec<String> = community
        .entities
        .iter()
        .filter_map(|eid| store.get_entity(eid))
        .map(|e| format!("- {} ({}): {}", e.name, e.entity_type, e.description))
        .collect();

    // M55(P1-5): HashSet 只在链外构建一次,而非 filter 闭包内每条 relation 重建。
    let entity_set: HashSet<&String> = community.entities.iter().collect();

    let relation_lines: Vec<String> = community
        .entities
        .iter()
        .flat_map(|eid| store.relations_for(eid))
        .filter(|r| entity_set.contains(&r.source) && entity_set.contains(&r.target))
        .map(|r| {
            // Use entity names instead of IDs for LLM readability.
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
        })
        .collect();

    let prompt = {
        use lc_prompts::PromptTemplate;
        let template = PromptTemplate::new(COMMUNITY_SUMMARY_PROMPT);
        let entities_str = entity_lines.join("\n");
        let relations_str = relation_lines.join("\n");
        let mut vars: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        vars.insert("entities", &entities_str);
        vars.insert("relations", &relations_str);
        template
            .format(&vars)
            .unwrap_or_else(|_| COMMUNITY_SUMMARY_PROMPT.to_string())
    };

    let messages = vec![Message::human(prompt)];

    let response: LLMResult = llm
        .chat(messages, None)
        .await
        .map_err(|e| super::GraphRAGError::LLMError(e.to_string()))?;

    Ok(response.content.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_rag::graph_store::{Entity, Relation};

    #[test]
    fn test_detect_communities_empty() {
        let store = GraphStore::new();
        let communities = detect_communities(&store, 3);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_detect_communities_single() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "A".into(),
            entity_type: "Person".into(),
            description: String::new(),
        });
        let communities = detect_communities(&store, 3);
        assert!(communities.is_empty()); // singletons are excluded
    }

    #[test]
    fn test_detect_communities_connected_pair() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "A".into(),
            entity_type: "Person".into(),
            description: String::new(),
        });
        store.add_entity(Entity {
            id: "e2".into(),
            name: "B".into(),
            entity_type: "Person".into(),
            description: String::new(),
        });
        store.add_relation(Relation {
            source: "e1".into(),
            target: "e2".into(),
            relation_type: "knows".into(),
            description: String::new(),
            doc_id: None,
        });

        let communities = detect_communities(&store, 3);
        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0].entities.len(), 2);
    }

    #[test]
    fn test_detect_communities_triangle() {
        let mut store = GraphStore::new();
        for name in ["A", "B", "C"] {
            store.add_entity(Entity {
                id: name.to_string(),
                name: name.to_string(),
                entity_type: "Person".into(),
                description: String::new(),
            });
        }
        // A-B, B-C, C-A  =>  one community of size 3
        for (s, t) in [("A", "B"), ("B", "C"), ("C", "A")] {
            store.add_relation(Relation {
                source: s.into(),
                target: t.into(),
                relation_type: "knows".into(),
                description: String::new(),
                doc_id: None,
            });
        }

        let communities = detect_communities(&store, 3);
        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0].entities.len(), 3);
    }

    #[test]
    fn test_detect_communities_two_disconnected_groups() {
        let mut store = GraphStore::new();

        // Group 1: A-B-C
        for name in ["A", "B", "C"] {
            store.add_entity(Entity {
                id: name.to_string(),
                name: name.to_string(),
                entity_type: "Person".into(),
                description: String::new(),
            });
        }
        for (s, t) in [("A", "B"), ("B", "C")] {
            store.add_relation(Relation {
                source: s.into(),
                target: t.into(),
                relation_type: "knows".into(),
                description: String::new(),
                doc_id: None,
            });
        }

        // Group 2: X-Y-Z
        for name in ["X", "Y", "Z"] {
            store.add_entity(Entity {
                id: name.to_string(),
                name: name.to_string(),
                entity_type: "Person".into(),
                description: String::new(),
            });
        }
        for (s, t) in [("X", "Y"), ("Y", "Z")] {
            store.add_relation(Relation {
                source: s.into(),
                target: t.into(),
                relation_type: "knows".into(),
                description: String::new(),
                doc_id: None,
            });
        }

        let communities = detect_communities(&store, 3);
        assert_eq!(communities.len(), 2);
    }

    #[test]
    fn test_assign_size_tiers() {
        let mut communities: Vec<Community> = (0..6)
            .map(|i| Community {
                id: i,
                entities: vec![format!("e{}", i)],
                level: 0,
            })
            .collect();
        assign_size_tiers(&mut communities, 3);
        // With 6 communities and 3 levels, tier_size = 2
        assert_eq!(communities[0].level, 0);
        assert_eq!(communities[1].level, 0);
        assert_eq!(communities[2].level, 1);
        assert_eq!(communities[3].level, 1);
        assert_eq!(communities[4].level, 2);
        assert_eq!(communities[5].level, 2);
    }

    /// Verify that community summary formatting uses entity names, not IDs.
    /// This is the regression test for the bug where relations showed "e_xxx"
    /// instead of human-readable entity names.
    #[test]
    fn test_relation_formatting_uses_names_not_ids() {
        let mut store = GraphStore::new();
        // Entity with id != name — this is the critical case
        store.add_entity(Entity {
            id: "e_001".into(),
            name: "Alice".into(),
            entity_type: "Person".into(),
            description: "A software engineer".into(),
        });
        store.add_entity(Entity {
            id: "e_002".into(),
            name: "Google".into(),
            entity_type: "Company".into(),
            description: "A tech company".into(),
        });
        store.add_relation(Relation {
            source: "e_001".into(),
            target: "e_002".into(),
            relation_type: "works_at".into(),
            description: String::new(),
            doc_id: None,
        });

        // Build the community and check relation formatting
        let communities = detect_communities(&store, 3);
        assert!(!communities.is_empty(), "should detect a community");

        let community = &communities[0];

        // Simulate the formatting logic from summarize_community
        let entity_set: HashSet<&String> = community.entities.iter().collect();
        let relation_lines: Vec<String> = community
            .entities
            .iter()
            .flat_map(|eid| store.relations_for(eid))
            .filter(|r| entity_set.contains(&r.source) && entity_set.contains(&r.target))
            .map(|r| {
                let source_name = store
                    .get_entity(&r.source)
                    .map(|e| e.name.as_str())
                    .unwrap_or(&r.source);
                let target_name = store
                    .get_entity(&r.target)
                    .map(|e| e.name.as_str())
                    .unwrap_or(&r.target);
                format!("{} --[{}]--> {}", source_name, r.relation_type, target_name)
            })
            .collect();

        // The formatted relation should contain names, not IDs
        let relation_text = relation_lines.join("\n");
        assert!(
            relation_text.contains("Alice"),
            "relation should contain entity name 'Alice', got: {}",
            relation_text
        );
        assert!(
            relation_text.contains("Google"),
            "relation should contain entity name 'Google', got: {}",
            relation_text
        );
        assert!(
            !relation_text.contains("e_001"),
            "relation should NOT contain raw entity ID 'e_001', got: {}",
            relation_text
        );
        assert!(
            !relation_text.contains("e_002"),
            "relation should NOT contain raw entity ID 'e_002', got: {}",
            relation_text
        );
    }
}
