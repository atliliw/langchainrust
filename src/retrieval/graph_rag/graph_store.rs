// src/retrieval/graph_rag/graph_store.rs
//! In-memory graph store using adjacency lists.
//!
//! Stores entities, relations, and communities without any external graph crate.

use std::collections::HashMap;

/// A named entity extracted from a document.
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub description: String,
}

/// A directed relation between two entities.
#[derive(Debug, Clone)]
pub struct Relation {
    pub source: String,
    pub target: String,
    pub relation_type: String,
    pub description: String,
    pub doc_id: Option<String>,
}

/// A community of entities detected by community detection.
#[derive(Debug, Clone)]
pub struct Community {
    pub id: usize,
    pub entities: Vec<String>,
    pub level: usize,
}

/// In-memory graph store backed by adjacency lists.
#[derive(Clone)]
pub struct GraphStore {
    entities: HashMap<String, Entity>,
    relations: Vec<Relation>,
    adjacency: HashMap<String, Vec<usize>>,
    communities: Vec<Community>,
    community_summaries: Vec<String>,
}

impl GraphStore {
    /// Creates an empty graph store.
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            relations: Vec::new(),
            adjacency: HashMap::new(),
            communities: Vec::new(),
            community_summaries: Vec::new(),
        }
    }

    /// Adds an entity, replacing any existing entity with the same id.
    pub fn add_entity(&mut self, entity: Entity) {
        self.entities.insert(entity.id.clone(), entity);
    }

    /// Adds a directed relation and updates the adjacency list.
    pub fn add_relation(&mut self, relation: Relation) {
        let idx = self.relations.len();
        self.adjacency
            .entry(relation.source.clone())
            .or_default()
            .push(idx);
        self.adjacency
            .entry(relation.target.clone())
            .or_default()
            .push(idx);
        self.relations.push(relation);
    }

    /// Returns a reference to an entity by id.
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// Returns all entity ids.
    pub fn entity_ids(&self) -> Vec<String> {
        self.entities.keys().cloned().collect()
    }

    /// Returns the number of entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns the number of relations.
    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    /// Returns the neighbor entity ids of a given entity (both directions).
    pub fn neighbors(&self, entity_id: &str) -> Vec<String> {
        let mut result = Vec::new();
        if let Some(indices) = self.adjacency.get(entity_id) {
            for &idx in indices {
                let rel = &self.relations[idx];
                if rel.source == entity_id {
                    result.push(rel.target.clone());
                } else {
                    result.push(rel.source.clone());
                }
            }
        }
        result.sort();
        result.dedup();
        result
    }

    /// Returns the subgraph around a seed entity: the seed, its direct
    /// neighbors, and all relations among them.
    pub fn subgraph(&self, seed: &str, depth: usize) -> (Vec<Entity>, Vec<Relation>) {
        let mut visited = std::collections::HashSet::new();
        let mut frontier = vec![seed.to_string()];
        visited.insert(seed.to_string());

        for _ in 0..depth {
            let mut next_frontier = Vec::new();
            for eid in &frontier {
                for nb in self.neighbors(eid) {
                    if visited.insert(nb.clone()) {
                        next_frontier.push(nb);
                    }
                }
            }
            frontier = next_frontier;
        }

        let entities: Vec<Entity> = visited
            .iter()
            .filter_map(|id| self.entities.get(id))
            .cloned()
            .collect();

        let relations: Vec<Relation> = self
            .relations
            .iter()
            .filter(|r| visited.contains(&r.source) && visited.contains(&r.target))
            .cloned()
            .collect();

        (entities, relations)
    }

    /// Returns all relations involving a given entity.
    pub fn relations_for(&self, entity_id: &str) -> Vec<&Relation> {
        self.adjacency
            .get(entity_id)
            .map(|indices| indices.iter().map(|&i| &self.relations[i]).collect())
            .unwrap_or_default()
    }

    /// Returns a reference to all entities.
    pub fn all_entities(&self) -> &HashMap<String, Entity> {
        &self.entities
    }

    /// Returns a reference to all relations.
    pub fn all_relations(&self) -> &[Relation] {
        &self.relations
    }

    // -- Community management --------------------------------------------------

    /// Replaces the stored communities.
    pub fn set_communities(&mut self, communities: Vec<Community>) {
        self.communities = communities;
    }

    /// Returns a reference to the stored communities.
    pub fn communities(&self) -> &[Community] {
        &self.communities
    }

    /// Replaces the stored community summaries.
    pub fn set_community_summaries(&mut self, summaries: Vec<String>) {
        self.community_summaries = summaries;
    }

    /// Returns a reference to the stored community summaries.
    pub fn community_summaries(&self) -> &[String] {
        &self.community_summaries
    }

    /// Returns a mutable reference to the stored community summaries.
    pub fn community_summaries_mut(&mut self) -> &mut Vec<String> {
        &mut self.community_summaries
    }
}

impl Default for GraphStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: &str, name: &str) -> Entity {
        Entity {
            id: id.to_string(),
            name: name.to_string(),
            entity_type: "concept".to_string(),
            description: format!("Entity {}", name),
        }
    }

    fn make_relation(source: &str, target: &str, rel_type: &str) -> Relation {
        Relation {
            source: source.to_string(),
            target: target.to_string(),
            relation_type: rel_type.to_string(),
            description: format!("{} {}", rel_type, target),
            doc_id: None,
        }
    }

    #[test]
    fn test_add_and_get_entity() {
        let mut store = GraphStore::new();
        store.add_entity(make_entity("e1", "Rust"));
        assert_eq!(store.entity_count(), 1);
        assert!(store.get_entity("e1").is_some());
        assert!(store.get_entity("e2").is_none());
    }

    #[test]
    fn test_add_entity_replaces_existing() {
        let mut store = GraphStore::new();
        store.add_entity(make_entity("e1", "Rust"));
        store.add_entity(Entity {
            id: "e1".to_string(),
            name: "Rust Lang".to_string(),
            entity_type: "language".to_string(),
            description: "Updated".to_string(),
        });
        assert_eq!(store.entity_count(), 1);
        assert_eq!(store.get_entity("e1").unwrap().name, "Rust Lang");
    }

    #[test]
    fn test_add_relation_and_count() {
        let mut store = GraphStore::new();
        store.add_entity(make_entity("e1", "Alice"));
        store.add_entity(make_entity("e2", "Bob"));
        store.add_relation(make_relation("e1", "e2", "mentors"));
        assert_eq!(store.relation_count(), 1);
    }

    #[test]
    fn test_neighbors_bidirectional() {
        let mut store = GraphStore::new();
        store.add_entity(make_entity("e1", "Alice"));
        store.add_entity(make_entity("e2", "Bob"));
        store.add_relation(make_relation("e1", "e2", "mentors"));

        let alice_neighbors = store.neighbors("e1");
        assert_eq!(alice_neighbors, vec!["e2"]);

        let bob_neighbors = store.neighbors("e2");
        assert_eq!(bob_neighbors, vec!["e1"]);
    }

    #[test]
    fn test_neighbors_dedup() {
        let mut store = GraphStore::new();
        store.add_entity(make_entity("e1", "Alice"));
        store.add_entity(make_entity("e2", "Bob"));
        store.add_relation(make_relation("e1", "e2", "mentor"));
        store.add_relation(make_relation("e1", "e2", "collaborator"));

        let neighbors = store.neighbors("e1");
        assert_eq!(neighbors, vec!["e2"]); // deduped
    }

    #[test]
    fn test_subgraph_depth_1() {
        let mut store = GraphStore::new();
        store.add_entity(make_entity("e1", "Alice"));
        store.add_entity(make_entity("e2", "Bob"));
        store.add_entity(make_entity("e3", "Charlie"));
        store.add_relation(make_relation("e1", "e2", "mentor"));
        store.add_relation(make_relation("e2", "e3", "colleague"));

        let (entities, relations) = store.subgraph("e1", 1);
        // Depth 1 from Alice: Alice + Bob
        assert_eq!(entities.len(), 2);
        assert_eq!(relations.len(), 1); // only Alice->Bob
    }

    #[test]
    fn test_community_management() {
        let mut store = GraphStore::new();
        store.set_communities(vec![Community {
            id: 0,
            entities: vec!["e1".to_string(), "e2".to_string()],
            level: 0,
        }]);
        assert_eq!(store.communities().len(), 1);

        store.set_community_summaries(vec!["Community of Alice and Bob".to_string()]);
        assert_eq!(store.community_summaries().len(), 1);
        assert_eq!(store.community_summaries()[0], "Community of Alice and Bob");
    }

    #[test]
    fn test_entity_ids() {
        let mut store = GraphStore::new();
        store.add_entity(make_entity("e1", "A"));
        store.add_entity(make_entity("e2", "B"));
        let mut ids = store.entity_ids();
        ids.sort();
        assert_eq!(ids, vec!["e1", "e2"]);
    }

    #[test]
    fn test_relations_for_entity() {
        let mut store = GraphStore::new();
        store.add_entity(make_entity("e1", "Alice"));
        store.add_entity(make_entity("e2", "Bob"));
        store.add_entity(make_entity("e3", "Charlie"));
        store.add_relation(make_relation("e1", "e2", "mentor"));
        store.add_relation(make_relation("e1", "e3", "advisor"));

        let rels = store.relations_for("e1");
        assert_eq!(rels.len(), 2);
    }

    #[test]
    fn test_default_is_empty() {
        let store = GraphStore::default();
        assert_eq!(store.entity_count(), 0);
        assert_eq!(store.relation_count(), 0);
        assert!(store.communities().is_empty());
        assert!(store.community_summaries().is_empty());
    }
}
