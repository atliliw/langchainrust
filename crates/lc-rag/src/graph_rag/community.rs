// src/retrieval/graph_rag/community.rs
//! Hierarchical community detection and community summarization.
//!
//! Detection runs the deterministic weighted-modularity
//! [Leiden algorithm](super::leiden) on the entity relation graph, then
//! repeatedly aggregates communities into coarser nodes and reruns Leiden —
//! the graspologic / Microsoft GraphRAG hierarchical construction. Level 0
//! is the base partition; each higher level strictly contains the level
//! below (tracked by [`Community::parent`]).
//!
//! Summaries follow the hierarchy: level-0 communities are summarized from
//! their entities and internal relations; higher levels are synthesized
//! from child-community summaries plus the relations crossing children.

use super::graph_store::{Community, Entity, GraphStore, Relation};
use super::leiden::{leiden_modularity, WeightedGraph};
use super::GraphRAGError;
use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_schema::Message;
use std::collections::{HashMap, HashSet};

/// Default Leiden resolution (modularity γ).
pub const DEFAULT_RESOLUTION: f64 = 1.0;
/// Default deterministic RNG seed for community detection.
pub const DEFAULT_SEED: u64 = 42;
/// Default cap on hierarchy depth (level 0..=max-1).
pub const DEFAULT_MAX_LEVELS: usize = 3;
/// Maximum number of cross-community relation lines fed into a roll-up summary.
const MAX_ROLLUP_RELATIONS: usize = 30;

/// A community draft at one hierarchy level before global ids are assigned.
#[derive(Debug, Clone)]
struct Draft {
    /// Entity ids of the whole subtree (union of children), sorted.
    entities: Vec<String>,
    /// Local indices of the immediate child communities in the previous level.
    children: Vec<usize>,
}

/// Sorts drafts largest-first with a deterministic tie-break on members.
fn sort_drafts(drafts: &mut [Draft]) {
    drafts.sort_by(|a, b| {
        b.entities
            .len()
            .cmp(&a.entities.len())
            .then_with(|| a.entities.cmp(&b.entities))
    });
}

/// Builds the base weighted graph: one node per entity (sorted ids for stable
/// indices), every relation contributes one undirected unit of weight
/// (parallel relations aggregate and strengthen the edge).
fn entity_graph(store: &GraphStore) -> (WeightedGraph, Vec<String>) {
    let mut ids: Vec<String> = store.entity_ids();
    ids.sort();
    let index: HashMap<&str, usize> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();

    let edges = store.all_relations().iter().filter_map(|r| {
        match (index.get(r.source.as_str()), index.get(r.target.as_str())) {
            (Some(&u), Some(&v)) => Some((u, v, 1.0f64)),
            _ => None,
        }
    });
    (WeightedGraph::from_edges(ids.len(), edges), ids)
}

/// Groups a flat Leiden membership vector into drafts, dropping singleton
/// groups (isolated entities are not communities).
fn drafts_from_membership(members: &[usize], node_entities: &[Vec<String>]) -> Vec<Draft> {
    let group_count = members.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); group_count];
    for (node, &label) in members.iter().enumerate() {
        groups[label].push(node);
    }

    let mut drafts = Vec::new();
    for nodes in groups {
        if nodes.len() < 2 {
            continue;
        }
        let mut entities: Vec<String> = nodes
            .iter()
            .flat_map(|n| node_entities[*n].clone())
            .collect();
        entities.sort();
        entities.dedup();
        let mut children = nodes;
        children.sort();
        drafts.push(Draft { entities, children });
    }
    sort_drafts(&mut drafts);
    drafts
}

/// Runs hierarchical Leiden detection over the graph store.
///
/// Returns a flat, level-ordered vector of [`Community`] values: level 0
/// first, then each coarser level; within a level communities are sorted
/// largest-first. Each community's globally unique `id` equals its position
/// in the returned vector, which is also the index of its summary. Every
/// merge is recorded in [`Community::parent`].
///
/// - `resolution`: modularity γ (use [`DEFAULT_RESOLUTION`]).
/// - `max_levels`: hierarchy depth cap including level 0.
/// - `seed`: deterministic visit order.
pub fn detect_hierarchy(
    store: &GraphStore,
    resolution: f64,
    max_levels: usize,
    seed: u64,
) -> Result<Vec<Community>, GraphRAGError> {
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err(GraphRAGError::CommunityError(format!(
            "Leiden resolution must be a finite positive number, got {resolution}"
        )));
    }
    if max_levels == 0 {
        return Err(GraphRAGError::CommunityError(
            "max_levels must be at least 1".to_string(),
        ));
    }

    // -- Level 0: Leiden over entities -------------------------------------
    let (graph, ids) = entity_graph(store);
    if graph.node_count() == 0 {
        return Ok(Vec::new());
    }
    let membership = leiden_modularity(&graph, resolution, seed);
    let base_nodes: Vec<Vec<String>> = ids.iter().map(|id| vec![id.clone()]).collect();
    let mut levels: Vec<Vec<Draft>> = vec![drafts_from_membership(&membership, &base_nodes)];
    if levels[0].is_empty() {
        return Ok(Vec::new());
    }

    // -- Levels 1+: aggregate communities and rerun Leiden ------------------
    // Current level nodes = communities of the previous level, in sorted
    // draft order. Unmerged (singleton-on-coarse) children stay as nodes, so
    // they remain able to merge into an even coarser grouping later.
    while levels.len() < max_levels {
        let prev = levels.last().expect("at least level 0 exists");
        let n = prev.len();
        if n < 2 {
            break;
        }

        // entity id -> current aggregate node index
        let mut entity_to_node: HashMap<&str, usize> = HashMap::new();
        for (node, draft) in prev.iter().enumerate() {
            for id in &draft.entities {
                entity_to_node.insert(id.as_str(), node);
            }
        }

        // Internal relations (u == v) become self-loop weight: per the
        // Leiden aggregation step a community's internal density must count
        // as a loop, otherwise dense communities look degree-1 and merge as
        // eagerly as sparse ones. `from_edges` routes u == v to loops.
        let edges = store.all_relations().iter().filter_map(|r| {
            match (
                entity_to_node.get(r.source.as_str()),
                entity_to_node.get(r.target.as_str()),
            ) {
                (Some(&u), Some(&v)) => Some((u, v, 1.0f64)),
                _ => None,
            }
        });
        let aggregate_graph = WeightedGraph::from_edges(n, edges);
        let membership = leiden_modularity(&aggregate_graph, resolution, seed);

        let node_entities: Vec<Vec<String>> =
            prev.iter().map(|draft| draft.entities.clone()).collect();
        let mut merged = drafts_from_membership(&membership, &node_entities);
        // Keep only groups that actually combined at least two child
        // communities; a draft whose children are one prior community is an
        // unchanged singleton and produces no level entry.
        merged.retain(|draft| draft.children.len() >= 2);

        if merged.is_empty() {
            break;
        }
        // `drafts_from_membership` sorted by total entities; resort after the
        // filter is a no-op for order but keeps the invariant explicit.
        sort_drafts(&mut merged);
        levels.push(merged);
    }

    // -- Assign global ids and parent links --------------------------------
    let mut flat: Vec<Community> = Vec::new();
    for (level, drafts) in levels.iter().enumerate() {
        let offset = flat.len();
        // Local index (position inside `drafts`) -> global id at this level.
        for (local, draft) in drafts.iter().enumerate() {
            flat.push(Community {
                id: offset + local,
                entities: draft.entities.clone(),
                level,
                parent: None, // filled when the next level is assigned
            });
        }
    }

    // Fill parent links: children indices in each draft refer to positions
    // within the previous level's draft slice, whose global ids start at the
    // previous level offset.
    let mut level_offsets: Vec<usize> = Vec::with_capacity(levels.len());
    let mut offset = 0usize;
    for drafts in &levels {
        level_offsets.push(offset);
        offset += drafts.len();
    }
    for level in 1..levels.len() {
        let child_offset = level_offsets[level - 1];
        let parent_offset = level_offsets[level];
        for (local, draft) in levels[level].iter().enumerate() {
            let parent_id = parent_offset + local;
            for child_local in &draft.children {
                flat[child_offset + *child_local].parent = Some(parent_id);
            }
        }
    }

    Ok(flat)
}

const COMMUNITY_SUMMARY_PROMPT: &str = r#"You are a knowledge graph summarization assistant.

Given the following entities and their relations within a community, write a concise summary (2-3 sentences) that captures the key information and relationships.

Entities:
{entities}

Relations:
{relations}

Summary:"#;

const COMMUNITY_ROLLUP_PROMPT: &str = r#"You are a knowledge graph summarization assistant building a level-{level} overview.

The community below groups several level-{child_level} sub-communities of a knowledge graph. Synthesize their summaries into a concise overview (3-5 sentences): describe the shared themes, how the sub-communities relate to each other, and what connects them. Avoid repeating details verbatim.

Sub-community summaries:
{children}

Relationships crossing sub-communities:
{relations}

Overview:"#;

/// Formats a relation using entity names instead of ids.
fn format_relation_named(r: &Relation, store: &GraphStore) -> String {
    let source_name = store
        .get_entity(&r.source)
        .map(|e| e.name.as_str())
        .unwrap_or(&r.source);
    let target_name = store
        .get_entity(&r.target)
        .map(|e| e.name.as_str())
        .unwrap_or(&r.target);
    if r.description.is_empty() {
        format!(
            "{source_name} --[{rel}]--> {target_name}",
            rel = r.relation_type
        )
    } else {
        format!(
            "{source_name} --[{rel}]--> {target_name}: {desc}",
            rel = r.relation_type,
            desc = r.description
        )
    }
}

/// Generates a level-0 community summary from its entities and internal
/// relations using the LLM.
pub async fn summarize_community<M: BaseChatModel>(
    llm: &M,
    store: &GraphStore,
    community: &Community,
) -> Result<String, GraphRAGError> {
    let entity_lines: Vec<String> = community
        .entities
        .iter()
        .filter_map(|eid| store.get_entity(eid))
        .map(|e: &Entity| format!("- {} ({}): {}", e.name, e.entity_type, e.description))
        .collect();

    // Built once outside the filter closure (internal relations only).
    let entity_set: HashSet<&String> = community.entities.iter().collect();
    let relation_lines: Vec<String> = community
        .entities
        .iter()
        .flat_map(|eid| store.relations_for(eid))
        .filter(|r| entity_set.contains(&r.source) && entity_set.contains(&r.target))
        .map(|r| format_relation_named(r, store))
        .collect();

    let prompt = {
        use lc_prompts::PromptTemplate;
        let template = PromptTemplate::new(COMMUNITY_SUMMARY_PROMPT);
        let entities_str = entity_lines.join("\n");
        let relations_str = relation_lines.join("\n");
        let mut vars: HashMap<&str, &str> = HashMap::new();
        vars.insert("entities", &entities_str);
        vars.insert("relations", &relations_str);
        template
            .format(&vars)
            .unwrap_or_else(|_| COMMUNITY_SUMMARY_PROMPT.to_string())
    };

    let response: LLMResult = llm
        .chat(vec![Message::human(prompt)], None)
        .await
        .map_err(|e| GraphRAGError::LLMError(e.to_string()))?;
    Ok(response.content.trim().to_string())
}

/// Computes the named, de-duplicated relations that cross the immediate child
/// communities of `parent_id`. The result is capped at a fixed number of
/// lines (30), in stable iteration order.
///
/// `flat` must be the level-ordered output of [`detect_hierarchy`] with
/// community ids equal to vector positions.
pub fn rollup_relations(store: &GraphStore, flat: &[Community], parent_id: usize) -> Vec<String> {
    let parent = match flat.get(parent_id) {
        Some(c) if c.level >= 1 => c,
        _ => return Vec::new(),
    };

    // Entity id -> immediate child community id.
    let mut child_of: HashMap<&str, usize> = HashMap::new();
    for community in flat.iter().filter(|c| c.parent == Some(parent_id)) {
        for eid in &community.entities {
            child_of.insert(eid.as_str(), community.id);
        }
    }

    let in_parent: HashSet<&String> = parent.entities.iter().collect();
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut lines = Vec::new();
    for relation in store.all_relations() {
        if !in_parent.contains(&relation.source) || !in_parent.contains(&relation.target) {
            continue;
        }
        let child_a = child_of.get(relation.source.as_str());
        let child_b = child_of.get(relation.target.as_str());
        match (child_a, child_b) {
            (Some(a), Some(b)) if a != b => {
                let key = (
                    relation.source.clone(),
                    relation.relation_type.clone(),
                    relation.target.clone(),
                );
                if seen.insert(key) {
                    lines.push(format_relation_named(relation, store));
                }
            }
            _ => {}
        }
        if lines.len() >= MAX_ROLLUP_RELATIONS {
            break;
        }
    }
    lines
}

/// Generates a higher-level community overview by rolling up the immediate
/// child-community summaries and the relations crossing them.
pub async fn summarize_rollup<M: BaseChatModel>(
    llm: &M,
    store: &GraphStore,
    community: &Community,
    flat: &[Community],
    child_summaries: &[String],
) -> Result<String, GraphRAGError> {
    let relations = rollup_relations(store, flat, community.id);
    let prompt = {
        use lc_prompts::PromptTemplate;
        let template = PromptTemplate::new(COMMUNITY_ROLLUP_PROMPT);
        let children_str = child_summaries
            .iter()
            .map(|s| format!("- {s}"))
            .collect::<Vec<_>>()
            .join("\n");
        let relations_str = relations.join("\n");
        let level_str = community.level.to_string();
        let child_level_str = (community.level - 1).to_string();
        let mut vars: HashMap<&str, &str> = HashMap::new();
        vars.insert("children", &children_str);
        vars.insert("relations", &relations_str);
        vars.insert("level", &level_str);
        vars.insert("child_level", &child_level_str);
        template
            .format(&vars)
            .unwrap_or_else(|_| COMMUNITY_ROLLUP_PROMPT.to_string())
    };

    let response: LLMResult = llm
        .chat(vec![Message::human(prompt)], None)
        .await
        .map_err(|e| GraphRAGError::LLMError(e.to_string()))?;
    Ok(response.content.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(id: &str, name: &str) -> Entity {
        Entity {
            id: id.to_string(),
            name: name.to_string(),
            entity_type: "concept".to_string(),
            description: format!("Entity {name}"),
        }
    }

    fn relation(source: &str, target: &str) -> Relation {
        Relation {
            source: source.to_string(),
            target: target.to_string(),
            relation_type: "rel".to_string(),
            description: String::new(),
            doc_id: None,
        }
    }

    fn clique(store: &mut GraphStore, ids: &[&str]) {
        for id in ids {
            store.add_entity(entity(id, &id.to_uppercase()));
        }
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                store.add_relation(relation(a, b));
            }
        }
    }

    #[test]
    fn empty_store_produces_no_hierarchy() {
        let store = GraphStore::new();
        assert!(
            detect_hierarchy(&store, DEFAULT_RESOLUTION, DEFAULT_MAX_LEVELS, DEFAULT_SEED)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn singleton_entities_are_not_communities() {
        let mut store = GraphStore::new();
        store.add_entity(entity("a", "A"));
        let flat =
            detect_hierarchy(&store, DEFAULT_RESOLUTION, DEFAULT_MAX_LEVELS, DEFAULT_SEED).unwrap();
        assert!(flat.is_empty());
    }

    #[test]
    fn connected_pair_is_one_level_zero_community() {
        let mut store = GraphStore::new();
        clique(&mut store, &["a", "b"]);
        let flat =
            detect_hierarchy(&store, DEFAULT_RESOLUTION, DEFAULT_MAX_LEVELS, DEFAULT_SEED).unwrap();
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].level, 0);
        assert_eq!(flat[0].id, 0);
        assert_eq!(flat[0].entities, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(flat[0].parent, None);
    }

    #[test]
    fn two_disconnected_triangles_stay_at_level_zero() {
        let mut store = GraphStore::new();
        clique(&mut store, &["a", "b", "c"]);
        clique(&mut store, &["x", "y", "z"]);
        let flat =
            detect_hierarchy(&store, DEFAULT_RESOLUTION, DEFAULT_MAX_LEVELS, DEFAULT_SEED).unwrap();
        assert_eq!(flat.len(), 2);
        assert!(flat.iter().all(|c| c.level == 0 && c.parent.is_none()));
    }

    #[test]
    fn isolated_entity_is_excluded_but_does_not_break_detection() {
        let mut store = GraphStore::new();
        clique(&mut store, &["a", "b", "c"]);
        store.add_entity(entity("lonely", "Lonely"));
        let flat =
            detect_hierarchy(&store, DEFAULT_RESOLUTION, DEFAULT_MAX_LEVELS, DEFAULT_SEED).unwrap();
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].entities.len(), 3);
        assert!(!flat[0].entities.contains(&"lonely".to_string()));
    }

    #[test]
    fn invalid_arguments_are_rejected() {
        let store = GraphStore::new();
        assert!(detect_hierarchy(&store, 0.0, 3, 1).is_err());
        assert!(detect_hierarchy(&store, f64::NAN, 3, 1).is_err());
        assert!(detect_hierarchy(&store, 1.0, 0, 1).is_err());
    }

    /// Fixture: two triangles linked by one bridge edge and two 8-cliques
    /// linked the same way. Flat Leiden yields four level-0 communities; on
    /// the aggregated graph the two low-degree triangle communities merge,
    /// while the heavy clique communities do not.
    fn four_group_fixture() -> GraphStore {
        let mut store = GraphStore::new();
        let groups: [Vec<usize>; 4] = [
            (0..3).collect(),
            (3..6).collect(),
            (6..14).collect(),
            (14..22).collect(),
        ];
        for group in &groups {
            let ids: Vec<String> = group.iter().map(|i| format!("n{i}")).collect();
            let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
            clique(&mut store, &refs);
        }
        store.add_relation(relation("n2", "n3"));
        store.add_relation(relation("n13", "n14"));
        store
    }

    #[test]
    fn hierarchy_merges_small_communities_at_level_one() {
        let store = four_group_fixture();
        let flat =
            detect_hierarchy(&store, DEFAULT_RESOLUTION, DEFAULT_MAX_LEVELS, DEFAULT_SEED).unwrap();

        let level0: Vec<&Community> = flat.iter().filter(|c| c.level == 0).collect();
        let level1: Vec<&Community> = flat.iter().filter(|c| c.level == 1).collect();
        assert_eq!(level0.len(), 4, "flat: {flat:?}");
        assert_eq!(level1.len(), 1, "flat: {flat:?}");

        // Global ids are positional and contiguous across levels.
        for (i, c) in flat.iter().enumerate() {
            assert_eq!(c.id, i);
        }

        // The two triangle communities are children of the level-1 group.
        let top = level1[0];
        assert_eq!(top.entities.len(), 6);
        assert!(top.parent.is_none());
        let children: Vec<&Community> = flat.iter().filter(|c| c.parent == Some(top.id)).collect();
        assert_eq!(children.len(), 2);
        for child in &children {
            assert_eq!(child.level, 0);
            assert_eq!(child.entities.len(), 3);
        }
        // The 8-clique communities have no coarser grouping.
        let unparented: Vec<&Community> = level0
            .iter()
            .copied()
            .filter(|c| c.parent.is_none())
            .collect();
        assert_eq!(unparented.len(), 2);
        assert!(unparented.iter().all(|c| c.entities.len() == 8));
    }

    #[test]
    fn max_levels_one_disables_hierarchical_rollup() {
        let store = four_group_fixture();
        let flat = detect_hierarchy(&store, DEFAULT_RESOLUTION, 1, DEFAULT_SEED).unwrap();
        assert_eq!(flat.len(), 4);
        assert!(flat.iter().all(|c| c.level == 0 && c.parent.is_none()));
    }

    #[test]
    fn hierarchy_is_deterministic() {
        let store = four_group_fixture();
        let a = detect_hierarchy(&store, DEFAULT_RESOLUTION, DEFAULT_MAX_LEVELS, 7).unwrap();
        let b = detect_hierarchy(&store, DEFAULT_RESOLUTION, DEFAULT_MAX_LEVELS, 7).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.entities, y.entities);
            assert_eq!(x.level, y.level);
            assert_eq!(x.parent, y.parent);
        }
    }

    #[test]
    fn rollup_relations_lists_only_cross_child_edges() {
        let store = four_group_fixture();
        let flat =
            detect_hierarchy(&store, DEFAULT_RESOLUTION, DEFAULT_MAX_LEVELS, DEFAULT_SEED).unwrap();
        let top_id = flat.iter().find(|c| c.level == 1).unwrap().id;
        let lines = rollup_relations(&store, &flat, top_id);
        // Exactly the n2 -[rel]-> n3 bridge crosses the two triangle children.
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("N2"));
        assert!(lines[0].contains("N3"));

        // Level-0 communities are not rollups.
        assert!(rollup_relations(&store, &flat, 0).is_empty());
    }

    #[test]
    fn relations_are_named_in_summary_context() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "Alice".into(),
            entity_type: "Person".into(),
            description: "engineer".into(),
        });
        store.add_entity(Entity {
            id: "e2".into(),
            name: "Google".into(),
            entity_type: "Company".into(),
            description: "vendor".into(),
        });
        let mut r = relation("e1", "e2");
        r.relation_type = "works_at".into();
        store.add_relation(r);
        let line = format_relation_named(&store.all_relations()[0], &store);
        assert!(line.contains("Alice"));
        assert!(line.contains("Google"));
        assert!(line.contains("works_at"));
        assert!(!line.contains("e1"));
    }
}
