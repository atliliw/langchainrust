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
