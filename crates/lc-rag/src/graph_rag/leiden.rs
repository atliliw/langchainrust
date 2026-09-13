// src/retrieval/graph_rag/leiden.rs
//! Deterministic weighted Leiden community detection.
//!
//! Implements the Leiden algorithm (Traag, Waltman & van Eck, 2019) with the
//! **weighted modularity** quality function — the same quality function used
//! by graspologic / Microsoft GraphRAG:
//!
//! ```text
//! Q = Σ_c ( E_c/m − γ (Σ_c / 2m)² )
//! ```
//!
//! where `E_c` is the internal edge weight of community `c` (each undirected
//! edge counted once), `Σ_c` the sum of member degrees, `2m` total degree,
//! and `γ` the resolution (higher → smaller communities). Each outer
//! iteration runs the three canonical phases — fast local moving,
//! guaranteed-refinement, and aggregation — on an ever coarser graph until
//! no community can be merged any further.
//!
//! The implementation is dependency-free and **deterministic** for a given
//! seed (a splitmix64 PRNG drives node visit order; ties break by community
//! id). Graph construction aggregates multi-edges, so an entity pair
//! extracted from many documents is connected more strongly.

use std::collections::{HashMap, VecDeque};

/// Moves must improve modularity by more than this (scaled by `2m`) to be
/// accepted — guards against floating-point noise around zero.
const EPS: f64 = 1e-9;

/// Undirected weighted graph with self-loop weights.
///
/// `adj` holds each edge in **both** directions and never contains
/// self entries; self-loop weight (internal edges after aggregation) lives
/// separately in `loops`.
#[derive(Debug, Clone)]
pub struct WeightedGraph {
    n: usize,
    adj: Vec<Vec<(usize, f64)>>,
    loops: Vec<f64>,
}

impl WeightedGraph {
    /// Creates an empty graph with `n` isolated nodes.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
            loops: vec![0.0; n],
        }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.n
    }

    /// Builds a graph from `(u, v, weight)` undirected edges, aggregating
    /// parallel edges. Edges with non-finite or non-positive weight and edges
    /// referencing unknown nodes are ignored. A self edge contributes to the
    /// node's self-loop weight.
    pub fn from_edges(n: usize, edges: impl IntoIterator<Item = (usize, usize, f64)>) -> Self {
        let mut aggregated: HashMap<(usize, usize), f64> = HashMap::new();
        for (u, v, w) in edges {
            if u >= n || v >= n || !w.is_finite() || w <= 0.0 {
                continue;
            }
            let key = if u <= v { (u, v) } else { (v, u) };
            *aggregated.entry(key).or_insert(0.0) += w;
        }

        let mut graph = Self::new(n);
        for ((u, v), w) in aggregated {
            if u == v {
                graph.loops[u] += w;
            } else {
                graph.adj[u].push((v, w));
                graph.adj[v].push((u, w));
            }
        }
        // Sorted neighbors make traversal order (and hence results) stable.
        for neighbors in &mut graph.adj {
            neighbors.sort_by(|a, b| a.0.cmp(&b.0));
        }
        graph
    }

    /// Degree of `node`: incident edge weight plus twice its self-loop weight.
    fn degree(&self, node: usize) -> f64 {
        let incident: f64 = self.adj[node].iter().map(|(_, w)| *w).sum();
        incident + 2.0 * self.loops[node]
    }

    /// Total weight of edges from `node` to nodes in community `label`,
    /// excluding the node itself (self loops are omitted — they are constant
    /// across moves and cancel from every modularity delta).
    fn weight_to(&self, node: usize, labels: &[i64], label: i64) -> f64 {
        self.adj[node]
            .iter()
            .filter(|(nb, _)| labels[*nb] == label)
            .map(|(_, w)| *w)
            .sum()
    }
}

/// A compacted partition: community labels in `0..k`.
#[derive(Debug, Clone)]
struct Partition {
    /// Community label per node.
    labels: Vec<i64>,
    /// Sum of member degrees per community (indexed by label).
    degree_sums: Vec<f64>,
}

impl Partition {
    fn num_communities(&self) -> usize {
        self.degree_sums.len()
    }
}

/// Compacts arbitrary community ids (seed node ids, inherited parent labels)
/// into the contiguous range `0..k`, ordered by first appearance in node
/// order so labels are stable across runs.
fn compact_labels(raw: &[i64], degrees: &[f64]) -> Partition {
    let mut remap: HashMap<i64, i64> = HashMap::new();
    let mut labels = vec![-1i64; raw.len()];
    let mut degree_sums = Vec::new();
    for (node, &raw_label) in raw.iter().enumerate() {
        let next = remap.len() as i64;
        let label = *remap.entry(raw_label).or_insert(next);
        labels[node] = label;
        if label as usize >= degree_sums.len() {
            degree_sums.push(0.0);
        }
        degree_sums[label as usize] += degrees[node];
    }
    Partition {
        labels,
        degree_sums,
    }
}

/// splitmix64 — a tiny deterministic PRNG (Sebastiano Vigna, public domain /
/// CC0). We only need reproducible visit ordering, not cryptographic quality.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Fisher–Yates shuffle.
    fn shuffle<T>(&mut self, items: &mut [T]) {
        if items.len() < 2 {
            return;
        }
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

/// Scaled modularity delta of moving `node` from its current community into
/// community `to_label`; the common factor `2m` is omitted so only the sign
/// matters.
///
/// ```text
/// 2m·ΔQ = 2(w_to − w_from) + γ·deg(node)·(Σ_from − deg(node) − Σ_to) / m
/// ```
fn move_delta(
    weight_to: f64,
    weight_from: f64,
    degree: f64,
    sum_from_minus_node: f64,
    sum_to: f64,
    resolution: f64,
    m: f64,
) -> f64 {
    2.0 * (weight_to - weight_from) + resolution * degree * (sum_from_minus_node - sum_to) / m
}

/// Fast local-moving phase.
///
/// Starts from `initial` and moves a node to the neighboring community with
/// the largest strictly positive modularity delta until the queue drains.
fn local_moving(
    graph: &WeightedGraph,
    initial: &[i64],
    resolution: f64,
    rng: &mut Rng,
) -> Partition {
    let degrees: Vec<f64> = (0..graph.n).map(|node| graph.degree(node)).collect();
    let m2: f64 = degrees.iter().sum();

    // Edgeless graph: nothing can ever improve modularity.
    if m2 <= 0.0 {
        return compact_labels(initial, &degrees);
    }
    let m = m2 / 2.0;

    let partition = compact_labels(initial, &degrees);
    let mut labels = partition.labels;
    let mut degree_sums = partition.degree_sums;

    let mut order: Vec<usize> = (0..graph.n).collect();
    rng.shuffle(&mut order);

    let mut queue: VecDeque<usize> = order.iter().copied().collect();
    let mut in_queue = vec![true; graph.n];

    while let Some(node) = queue.pop_front() {
        in_queue[node] = false;

        let current = labels[node];
        let node_degree = degrees[node];
        let sum_from_without_node = degree_sums[current as usize] - node_degree;
        let weight_current = graph.weight_to(node, &labels, current);

        // Aggregate edge weight from `node` into each neighboring community.
        let mut candidate_weight: HashMap<i64, f64> = HashMap::new();
        for &(nb, w) in &graph.adj[node] {
            let label = labels[nb];
            *candidate_weight.entry(label).or_insert(0.0) += w;
        }

        // Best strictly-positive delta; ties resolve to the smallest label.
        let mut best_label = current;
        let mut best_delta = 0.0_f64;
        for (&label, &weight_to_label) in &candidate_weight {
            if label == current {
                continue;
            }
            let delta = move_delta(
                weight_to_label,
                weight_current,
                node_degree,
                sum_from_without_node,
                degree_sums[label as usize],
                resolution,
                m,
            );
            if delta > best_delta + EPS
                || (delta > EPS && (delta - best_delta).abs() <= EPS && label < best_label)
            {
                best_delta = delta;
                best_label = label;
            }
        }

        if best_label != current {
            degree_sums[current as usize] -= node_degree;
            degree_sums[best_label as usize] += node_degree;
            labels[node] = best_label;

            // Only neighbors outside the joined community may now want to move.
            for &(nb, _) in &graph.adj[node] {
                if labels[nb] != best_label && !in_queue[nb] {
                    in_queue[nb] = true;
                    queue.push_back(nb);
                }
            }
        }
    }

    Partition {
        labels,
        degree_sums,
    }
}

/// Refinement phase.
///
/// Splits every local-moving community into subsets, seeding each subset with
/// one node and greedily absorbing neighbors on **non-negative** modularity
/// deltas (zero-delta absorption is what lets refinement split a community
/// while keeping every subset well connected). Members of different input
/// communities never share a subset.
fn refine(
    graph: &WeightedGraph,
    communities: &Partition,
    resolution: f64,
    rng: &mut Rng,
) -> Partition {
    let degrees: Vec<f64> = (0..graph.n).map(|node| graph.degree(node)).collect();
    let m2: f64 = degrees.iter().sum();
    let m = m2 / 2.0;

    let mut subset = vec![-1i64; graph.n];
    let mut subset_degree_sums: Vec<f64> = Vec::new();
    let mut next_subset = 0i64;

    let mut community_nodes: HashMap<i64, Vec<usize>> = HashMap::new();
    for (node, &label) in communities.labels.iter().enumerate() {
        community_nodes.entry(label).or_default().push(node);
    }
    let mut community_order: Vec<i64> = community_nodes.keys().copied().collect();
    community_order.sort_unstable();

    for community in community_order {
        let mut nodes = community_nodes.remove(&community).unwrap_or_default();
        rng.shuffle(&mut nodes);

        for node in nodes {
            // Weight from `node` into each existing subset of its own community.
            let mut candidate_weight: HashMap<i64, f64> = HashMap::new();
            for &(nb, w) in &graph.adj[node] {
                let s = subset[nb];
                if s >= 0 && communities.labels[nb] == community {
                    *candidate_weight.entry(s).or_insert(0.0) += w;
                }
            }

            // The node is currently an unassigned singleton: only the
            // "join subset S" side contributes (2·w − γ·deg·Σ_S/m, scaled).
            let mut best_subset = -1i64;
            let mut best_delta = 0.0_f64;
            for (&s, &weight) in &candidate_weight {
                let delta =
                    2.0 * weight - resolution * degrees[node] * subset_degree_sums[s as usize] / m;
                if delta < -EPS {
                    continue;
                }
                if best_subset < 0
                    || delta > best_delta + EPS
                    || ((delta - best_delta).abs() <= EPS && s < best_subset)
                {
                    best_delta = delta;
                    best_subset = s;
                }
            }

            if best_subset < 0 {
                best_subset = next_subset;
                next_subset += 1;
                subset_degree_sums.push(0.0);
            }
            subset[node] = best_subset;
            subset_degree_sums[best_subset as usize] += degrees[node];
        }
    }

    compact_labels(&subset, &degrees)
}

/// An aggregated graph plus the original node indices represented by each
/// aggregate node.
struct Aggregated {
    graph: WeightedGraph,
    members: Vec<Vec<usize>>,
}

/// Aggregates `graph` by the refined partition: one node per subset, edge
/// weights summed across subsets, internal edges becoming self loops.
fn aggregate(graph: &WeightedGraph, refined: &Partition, members: &[Vec<usize>]) -> Aggregated {
    let k = refined.num_communities();

    let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
    for node in 0..graph.n {
        let subset = refined.labels[node] as usize;
        for &(nb, w) in &graph.adj[node] {
            if nb > node {
                let other = refined.labels[nb] as usize;
                let key = if subset <= other {
                    (subset, other)
                } else {
                    (other, subset)
                };
                *edges.entry(key).or_insert(0.0) += w;
            }
        }
        if graph.loops[node] > 0.0 {
            *edges.entry((subset, subset)).or_insert(0.0) += graph.loops[node];
        }
    }

    let mut aggregated = WeightedGraph::new(k);
    for ((u, v), w) in edges {
        if u == v {
            aggregated.loops[u] += w;
        } else {
            aggregated.adj[u].push((v, w));
            aggregated.adj[v].push((u, w));
        }
    }
    for neighbors in &mut aggregated.adj {
        neighbors.sort_by(|a, b| a.0.cmp(&b.0));
    }

    let mut aggregated_members = vec![Vec::new(); k];
    for (node, original) in members.iter().enumerate() {
        aggregated_members[refined.labels[node] as usize].extend_from_slice(original);
    }

    Aggregated {
        graph: aggregated,
        members: aggregated_members,
    }
}

/// Runs Leiden community detection with the weighted-modularity quality
/// function.
///
/// Returns a community label per node in `0..k`. Singleton and disconnected
/// nodes receive their own label; callers decide whether to keep them.
///
/// - `resolution` is γ (canonical default `1.0`; values above 1 favor smaller
///   communities, below 1 favor larger ones).
/// - `seed` makes the node visit order reproducible.
pub fn leiden_modularity(graph: &WeightedGraph, resolution: f64, seed: u64) -> Vec<usize> {
    let n = graph.n;
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let mut rng = Rng::new(seed);
    let members: Vec<Vec<usize>> = (0..n).map(|node| vec![node]).collect();

    // Level 0: local moving from the singleton partition.
    let singleton: Vec<i64> = (0..n as i64).collect();
    let mut partition = local_moving(graph, &singleton, resolution, &mut rng);
    let mut current_graph = graph.clone();
    let mut current_members = members;

    loop {
        // Every node its own community: nothing to refine or merge.
        if partition.num_communities() == current_graph.n {
            break;
        }

        let refined = refine(&current_graph, &partition, resolution, &mut rng);

        // Refinement must be strict; if it did not subdivide anything, another
        // round on the same graph could never change the result either.
        if refined.num_communities() == partition.num_communities() {
            break;
        }

        let aggregated = aggregate(&current_graph, &refined, &current_members);

        // Initial partition on the aggregate graph: each refined subset
        // inherits the community its members had after local moving.
        let mut initial = vec![-1i64; aggregated.graph.n];
        for node in 0..current_graph.n {
            initial[refined.labels[node] as usize] = partition.labels[node];
        }

        partition = local_moving(&aggregated.graph, &initial, resolution, &mut rng);
        current_graph = aggregated.graph;
        current_members = aggregated.members;
    }

    // Expand labels on the coarsest aggregate graph back to original nodes.
    let mut result = vec![0usize; n];
    for (aggregate_node, originals) in current_members.iter().enumerate() {
        for &original in originals {
            result[original] = partition.labels[aggregate_node] as usize;
        }
    }

    // Final compaction in first-appearance order.
    let degrees: Vec<f64> = (0..n).map(|node| graph.degree(node)).collect();
    let compact = compact_labels(
        &result.iter().map(|&x| x as i64).collect::<Vec<_>>(),
        &degrees,
    );
    compact.labels.iter().map(|&x| x as usize).collect()
}

/// Returns true if the subgraph induced by `nodes` is connected using only
/// edges inside `nodes` (Leiden's well-connected-community guarantee).
#[cfg(test)]
pub(crate) fn is_connected_induced(graph: &WeightedGraph, nodes: &[usize]) -> bool {
    if nodes.is_empty() {
        return true;
    }
    let in_set: std::collections::HashSet<usize> = nodes.iter().copied().collect();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![nodes[0]];
    seen.insert(nodes[0]);
    while let Some(node) = stack.pop() {
        for &(nb, _) in &graph.adj[node] {
            if in_set.contains(&nb) && seen.insert(nb) {
                stack.push(nb);
            }
        }
    }
    seen.len() == nodes.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a graph from (u, v) unit edges.
    fn graph_from(n: usize, pairs: &[(usize, usize)]) -> WeightedGraph {
        let edges = pairs.iter().map(|&(u, v)| (u, v, 1.0)).collect::<Vec<_>>();
        WeightedGraph::from_edges(n, edges)
    }

    /// Adds a k-node clique to `pairs`.
    fn push_clique(pairs: &mut Vec<(usize, usize)>, offset: usize, k: usize) {
        for i in 0..k {
            for j in (i + 1)..k {
                pairs.push((offset + i, offset + j));
            }
        }
    }

    fn groups(labels: &[usize]) -> Vec<Vec<usize>> {
        let k = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        let mut groups = vec![Vec::new(); k];
        for (node, &label) in labels.iter().enumerate() {
            groups[label].push(node);
        }
        groups.retain(|g| !g.is_empty());
        groups
    }

    #[test]
    fn empty_and_single_node_graphs() {
        let g = WeightedGraph::new(0);
        assert!(leiden_modularity(&g, 1.0, 1).is_empty());
        let g = WeightedGraph::new(1);
        assert_eq!(leiden_modularity(&g, 1.0, 1), vec![0]);
    }

    #[test]
    fn connected_pair_is_one_community() {
        let g = graph_from(2, &[(0, 1)]);
        let labels = leiden_modularity(&g, 1.0, 1);
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn two_disconnected_triangles_stay_separate() {
        let mut pairs = Vec::new();
        push_clique(&mut pairs, 0, 3);
        push_clique(&mut pairs, 3, 3);
        let g = graph_from(6, &pairs);
        let labels = leiden_modularity(&g, 1.0, 1);
        assert_ne!(labels[0], labels[3]);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
    }

    #[test]
    fn weak_bridge_does_not_merge_dense_cliques() {
        // Two 4-cliques joined by a single bridge edge.
        let mut pairs = Vec::new();
        push_clique(&mut pairs, 0, 4);
        push_clique(&mut pairs, 4, 4);
        pairs.push((3, 4));
        let g = graph_from(8, &pairs);
        let labels = leiden_modularity(&g, 1.0, 7);
        let groups = groups(&labels);
        assert_eq!(groups.len(), 2, "expected two communities, got {groups:?}");
        for group in &groups {
            assert_eq!(group.len(), 4);
            assert!(is_connected_induced(&g, group));
        }
    }

    #[test]
    fn hierarchy_fixture_flat_partition_is_four_groups() {
        // Four dense groups: two small triangles linked by one bridge edge and
        // two large 8-cliques linked the same way, with no edges between the
        // pairs. Flat Leiden keeps all four separate (refinement produces no
        // subdivision, so internal aggregation never runs). The coarser merge
        // of the low-degree pair is exercised one level up via
        // detect_hierarchy in community.rs.
        let mut pairs = Vec::new();
        push_clique(&mut pairs, 0, 3); // A: 0..3
        push_clique(&mut pairs, 3, 3); // B: 3..6
        push_clique(&mut pairs, 6, 8); // C: 6..14
        push_clique(&mut pairs, 14, 8); // D: 14..22
        pairs.push((2, 3));
        pairs.push((13, 14));
        let g = graph_from(22, &pairs);

        let labels = leiden_modularity(&g, 1.0, 7);
        let level0 = groups(&labels);
        assert_eq!(level0.len(), 4, "level 0: {level0:?}");
        // The two triangles stay in distinct level-0 communities...
        assert_ne!(labels[0], labels[3]);
        for group in &level0 {
            assert!(is_connected_induced(&g, group));
        }
    }

    #[test]
    fn heavy_bridge_edge_merges_two_pairs() {
        // Two disconnected pairs stay in two communities...
        let weak = WeightedGraph::from_edges(4, vec![(0, 1, 1.0), (2, 3, 1.0)]);
        assert_eq!(groups(&leiden_modularity(&weak, 1.0, 3)).len(), 2);

        // ...while a weight-10 bridge between the inner nodes dominates the
        // modularity penalty and pulls all four nodes together.
        let strong = WeightedGraph::from_edges(4, vec![(0, 1, 1.0), (2, 3, 1.0), (1, 2, 10.0)]);
        assert_eq!(groups(&leiden_modularity(&strong, 1.0, 3)).len(), 1);
    }

    #[test]
    fn result_is_deterministic_for_a_seed() {
        // A 32-node graph: four 8-node cliques with a few weak bridges.
        let mut pairs = Vec::new();
        for c in 0..4 {
            push_clique(&mut pairs, c * 8, 8);
        }
        pairs.push((6, 10));
        pairs.push((20, 26));
        pairs.push((1, 17));
        let g = graph_from(32, &pairs);

        let a = leiden_modularity(&g, 1.0, 123_456);
        let b = leiden_modularity(&g, 1.0, 123_456);
        assert_eq!(a, b);

        // Every detected community must induce a connected subgraph — the
        // well-connectedness property that distinguishes Leiden from Louvain.
        for group in groups(&a) {
            assert!(
                is_connected_induced(&g, &group),
                "disconnected community {group:?}"
            );
        }
    }

    #[test]
    fn isolated_nodes_remain_singletons() {
        let g = graph_from(5, &[(0, 1), (1, 2)]);
        let labels = leiden_modularity(&g, 1.0, 1);
        let groups = groups(&labels);
        // One connected community + two singletons.
        assert_eq!(groups.len(), 3);
        assert!(groups.iter().any(|g| g.len() == 3));
        assert_eq!(groups.iter().filter(|g| g.len() == 1).count(), 2);
    }

    #[test]
    fn malformed_edges_are_ignored() {
        let g = WeightedGraph::from_edges(
            3,
            vec![
                (0, 1, 1.0),
                (0, 9, 1.0),  // unknown node
                (1, 2, -1.0), // non-positive
                (2, 2, 2.0),  // self loop only
                (0, 2, f64::NAN),
            ],
        );
        assert_eq!(g.node_count(), 3);
        let labels = leiden_modularity(&g, 1.0, 1);
        // 0-1 connected; 2 only has a self loop so it stays singleton.
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn rng_shuffle_is_deterministic() {
        let mut a = Rng::new(99);
        let mut va: Vec<usize> = (0..20).collect();
        a.shuffle(&mut va);
        let mut b = Rng::new(99);
        let mut vb: Vec<usize> = (0..20).collect();
        b.shuffle(&mut vb);
        assert_eq!(va, vb);
        assert_ne!(va, (0..20).collect::<Vec<_>>());
    }
}
