// src/retrieval/graph_rag/community.rs
//! Community detection using label propagation.
//!
//! Implements a simple iterative label-propagation algorithm that groups
//! entities into communities based on graph connectivity. Also provides
//! LLM-based community summary generation.

use super::graph_store::{Community, GraphStore};
use crate::core::language_models::{BaseChatModel, LLMResult};
use crate::schema::Message;
use std::collections::{HashMap, HashSet};

/// Runs label-propagation community detection on the graph store.
///
/// Returns a list of communities (non-singleton groups only) sorted
/// largest-first.
pub fn detect_communities(store: &GraphStore, max_levels: usize) -> Vec<Community> {
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
    for _ in 0..max_levels * 10 {
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

    // Assign hierarchical levels by merging smaller communities.
    assign_levels(&mut communities, max_levels);

    communities
}

/// Assigns hierarchical levels to communities (simple size-based tiering).
fn assign_levels(communities: &mut [Community], max_levels: usize) {
    if communities.is_empty() || max_levels <= 1 {
        return;
    }

    // Sort by size descending, then assign levels based on size tiers.
    let n = communities.len();
    let tier_size = (n as f64 / max_levels as f64).ceil() as usize;
    if tier_size == 0 {
        return;
    }

    for (i, community) in communities.iter_mut().enumerate() {
        community.level = std::cmp::min(i / tier_size, max_levels - 1);
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

    let relation_lines: Vec<String> = community
        .entities
        .iter()
        .flat_map(|eid| store.relations_for(eid))
        .filter(|r| {
            // M55: O(1) HashSet lookup instead of O(n) Vec::contains
            let entity_set: HashSet<&String> = community.entities.iter().collect();
            entity_set.contains(&r.source) && entity_set.contains(&r.target)
        })
        .map(|r| {
            format!(
                "- {} --[{}]--> {}{}",
                r.source,
                r.relation_type,
                r.target,
                if r.description.is_empty() {
                    String::new()
                } else {
                    format!(": {}", r.description)
                }
            )
        })
        .collect();

    let prompt = COMMUNITY_SUMMARY_PROMPT
        .replace("{entities}", &entity_lines.join("\n"))
        .replace("{relations}", &relation_lines.join("\n"));

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
    use crate::retrieval::graph_rag::graph_store::{Entity, Relation};

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
    fn test_assign_levels() {
        let mut communities: Vec<Community> = (0..6)
            .map(|i| Community {
                id: i,
                entities: vec![format!("e{}", i)],
                level: 0,
            })
            .collect();
        assign_levels(&mut communities, 3);
        // With 6 communities and 3 levels, tier_size = 2
        assert_eq!(communities[0].level, 0);
        assert_eq!(communities[1].level, 0);
        assert_eq!(communities[2].level, 1);
        assert_eq!(communities[3].level, 1);
        assert_eq!(communities[4].level, 2);
        assert_eq!(communities[5].level, 2);
    }
}
