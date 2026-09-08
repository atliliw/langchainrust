// lc-vector-stores/src/hybrid_native.rs
//! Engine-native hybrid search (0.21.0 S4.3).
//!
//! 2026 direction: hybrid fusion (RRF / DBSF) executed **inside the vector
//! engine** — one round trip, no client-side merge. Qdrant's Query API
//! (≥ 1.10) fuses multiple `prefetch` branches server-side. Engines without
//! the capability keep using the universal client-side fallback
//! (`lc_rag::UnifiedHybridIndex` / `reciprocal_rank_fusion`).
//!
//! Capability is declared per store type via [`NativeHybridSearch`]. The
//! qdrant-client crate (1.18) speaks the Query API, so the request-shape
//! contract is enforced by the typed builder; result mapping is unit-tested
//! against constructed `ScoredPoint`s, and the end-to-end path has an
//! `#[ignore]`d integration test for environments with a live server.

use crate::{MetadataFilter, SearchResult, VectorStoreError};
use async_trait::async_trait;

/// Server-side fusion method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion (default parameters, server-side).
    Rrf,
    /// Distribution-Based Score Fusion.
    Dbsf,
}

impl From<FusionMethod> for qdrant_client::qdrant::Fusion {
    fn from(method: FusionMethod) -> Self {
        match method {
            FusionMethod::Rrf => qdrant_client::qdrant::Fusion::Rrf,
            FusionMethod::Dbsf => qdrant_client::qdrant::Fusion::Dbsf,
        }
    }
}

/// Native hybrid query: multiple dense query branches fused by the engine.
#[derive(Debug, Clone)]
pub struct NativeHybridQuery {
    /// One query vector per branch (e.g. dense + a second retriever view).
    /// At least two branches are required — a single branch has nothing to fuse.
    pub query_vectors: Vec<Vec<f32>>,
    /// Final result count.
    pub limit: usize,
    /// Per-branch candidate limit (defaults to `limit * 2`, min 10).
    pub prefetch_limit: Option<usize>,
    /// Fusion method (default RRF).
    pub fusion: FusionMethod,
    /// Optional payload filter applied to every branch.
    pub filter: Option<MetadataFilter>,
}

impl NativeHybridQuery {
    /// Creates a query from ≥ 2 branches with defaults (RRF).
    pub fn new(query_vectors: Vec<Vec<f32>>, limit: usize) -> Result<Self, VectorStoreError> {
        if query_vectors.len() < 2 {
            return Err(VectorStoreError::ConfigError(format!(
                "native hybrid fusion requires at least 2 query branches, got {}",
                query_vectors.len()
            )));
        }
        let dim = query_vectors[0].len();
        if query_vectors.iter().any(|v| v.len() != dim) {
            return Err(VectorStoreError::ConfigError(
                "native hybrid branches must share one vector dimension".to_string(),
            ));
        }
        Ok(Self {
            query_vectors,
            limit,
            prefetch_limit: None,
            fusion: FusionMethod::Rrf,
            filter: None,
        })
    }

    /// Sets the per-branch prefetch limit.
    pub fn with_prefetch_limit(mut self, prefetch_limit: usize) -> Self {
        self.prefetch_limit = Some(prefetch_limit);
        self
    }

    /// Sets the fusion method.
    pub fn with_fusion(mut self, fusion: FusionMethod) -> Self {
        self.fusion = fusion;
        self
    }

    /// Effective per-branch prefetch limit.
    pub fn effective_prefetch_limit(&self) -> u64 {
        self.prefetch_limit.unwrap_or((self.limit * 2).max(10)) as u64
    }
}

/// Capability trait: engine-native hybrid fusion.
///
/// Default implementation reports "not supported" so existing store types are
/// opt-in by construction. Callers check
/// [`NativeHybridSearch::supports_native_hybrid`] before issuing a query; a
/// mismatch returns a config error instead of a silent client fallback.
/// The Qdrant implementation lives in [`crate::qdrant`] (fields/methods are
/// crate-visible there).
#[async_trait]
pub trait NativeHybridSearch: Send + Sync {
    /// Whether this store can fuse hybrid branches server-side.
    fn supports_native_hybrid(&self) -> bool {
        false
    }

    /// Runs the engine-native fusion query.
    async fn native_hybrid_search(
        &self,
        query: &NativeHybridQuery,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let _ = query;
        Err(VectorStoreError::ConfigError(
            "this vector store does not support engine-native hybrid fusion; \
             use the client-side RRF fallback (lc_rag::UnifiedHybridIndex)"
                .to_string(),
        ))
    }
}

/// Test fixture shared across the crate's test modules: a `ScoredPoint` with
/// the payload conventions the mapping relies on.
#[cfg(test)]
pub(crate) mod fixtures {
    use qdrant_client::qdrant::{PointId, ScoredPoint};
    use std::collections::HashMap;

    pub(crate) fn scored_point(id: u64, score: f32, content: &str, source: &str) -> ScoredPoint {
        let mut point = ScoredPoint::default();
        point.id = Some(PointId::from(id.to_string()));
        point.score = score;
        let mut payload = HashMap::new();
        payload.insert(
            "content".to_string(),
            qdrant_client::qdrant::Value::from(content),
        );
        payload.insert(
            "source".to_string(),
            qdrant_client::qdrant::Value::from(source),
        );
        point.payload = payload;
        point
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_native::fixtures::scored_point;

    /// Mapping follows the same payload conventions as plain search:
    /// `content` → document content, `doc_id` → id, other string fields → metadata.
    #[test]
    fn scored_point_maps_to_search_result() {
        let result =
            crate::qdrant::scored_point_to_result(scored_point(1, 0.98, "rust doc", "docs"));
        assert_eq!(result.document.content, "rust doc");
        assert_eq!(result.document.id, None, "no doc_id payload → no id");
        assert_eq!(result.score, 0.98);
        assert_eq!(
            result
                .document
                .metadata
                .get("source")
                .and_then(|v| v.as_str()),
            Some("docs")
        );
        assert!(
            !result.document.metadata.contains_key("content"),
            "content is not duplicated into metadata"
        );
    }

    /// `doc_id` payload becomes the document id.
    #[test]
    fn doc_id_payload_becomes_document_id() {
        let mut point = scored_point(2, 0.9, "hello", "docs");
        point.payload.insert(
            "doc_id".to_string(),
            qdrant_client::qdrant::Value::from("doc-42"),
        );
        let result = crate::qdrant::scored_point_to_result(point);
        assert_eq!(result.document.id.as_deref(), Some("doc-42"));
    }

    #[test]
    fn fusion_maps_to_proto() {
        assert_eq!(
            qdrant_client::qdrant::Fusion::from(FusionMethod::Rrf),
            qdrant_client::qdrant::Fusion::Rrf
        );
        assert_eq!(
            qdrant_client::qdrant::Fusion::from(FusionMethod::Dbsf),
            qdrant_client::qdrant::Fusion::Dbsf
        );
    }

    /// Query validation: ≥ 2 branches, single shared dimension.
    #[test]
    fn query_validates_branches() {
        assert!(
            NativeHybridQuery::new(vec![vec![1.0]], 5).is_err(),
            "1 branch"
        );
        assert!(
            NativeHybridQuery::new(vec![vec![1.0, 0.0], vec![1.0, 0.0, 0.0]], 5).is_err(),
            "mixed dimensions"
        );
        let ok = NativeHybridQuery::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]], 5).unwrap();
        assert_eq!(ok.limit, 5);
        assert_eq!(ok.fusion, FusionMethod::Rrf, "default fusion");
    }

    /// Prefetch limit defaults to `max(limit * 2, 10)`.
    #[test]
    fn prefetch_limit_defaults() {
        let small = NativeHybridQuery::new(vec![vec![1.0], vec![0.0]], 3).unwrap();
        assert_eq!(small.effective_prefetch_limit(), 10, "floor applies");
        let large = NativeHybridQuery::new(vec![vec![1.0], vec![0.0]], 20).unwrap();
        assert_eq!(large.effective_prefetch_limit(), 40, "limit * 2");
        let explicit = NativeHybridQuery::new(vec![vec![1.0], vec![0.0]], 20)
            .unwrap()
            .with_prefetch_limit(7);
        assert_eq!(explicit.effective_prefetch_limit(), 7);
    }

    /// Default capability: stores that do not implement the capability report
    /// false and return an explicit config error (never a silent fallback).
    #[tokio::test]
    async fn default_capability_is_unsupported() {
        struct Noop;
        #[async_trait]
        impl NativeHybridSearch for Noop {}
        assert!(!Noop.supports_native_hybrid());
        let query = NativeHybridQuery::new(vec![vec![1.0], vec![0.0]], 5).unwrap();
        let err = Noop.native_hybrid_search(&query).await.unwrap_err();
        assert!(
            err.to_string().contains("client-side RRF fallback"),
            "error must point to the fallback: {err}"
        );
    }
}
