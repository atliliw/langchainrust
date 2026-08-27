// lc-vector-stores/src/qdrant.rs
//! Qdrant vector store implementation

use crate::{Document, FilterOp, MetadataFilter, SearchResult, VectorStore, VectorStoreError};
use async_trait::async_trait;
use qdrant_client::{
    qdrant::{
        Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointId,
        PointStruct, QueryPointsBuilder, Range, UpsertPointsBuilder, VectorParamsBuilder,
    },
    Payload, Qdrant,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Qdrant configuration
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    /// Qdrant server URL
    pub url: String,
    /// Collection name
    pub collection_name: String,
    /// Vector dimension
    pub vector_size: usize,
    /// Distance metric
    pub distance: QdrantDistance,
}

/// Qdrant distance metric type
#[derive(Debug, Clone, Copy)]
pub enum QdrantDistance {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance
    Euclid,
    /// Dot product
    Dot,
}

impl From<QdrantDistance> for Distance {
    fn from(dist: QdrantDistance) -> Self {
        match dist {
            QdrantDistance::Cosine => Distance::Cosine,
            QdrantDistance::Euclid => Distance::Euclid,
            QdrantDistance::Dot => Distance::Dot,
        }
    }
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            collection_name: "langchainrust".to_string(),
            vector_size: 1536,
            distance: QdrantDistance::Cosine,
        }
    }
}

impl QdrantConfig {
    /// Creates a config from a server URL and collection name; remaining fields use defaults.
    pub fn new(url: impl Into<String>, collection_name: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            collection_name: collection_name.into(),
            ..Default::default()
        }
    }

    /// Sets the vector dimension.
    pub fn with_vector_size(mut self, size: usize) -> Self {
        self.vector_size = size;
        self
    }

    /// Sets the distance metric.
    pub fn with_distance(mut self, distance: QdrantDistance) -> Self {
        self.distance = distance;
        self
    }
}

/// Qdrant vector store
pub struct QdrantVectorStore {
    client: Arc<Qdrant>,
    config: QdrantConfig,
}

impl QdrantVectorStore {
    /// Connects to Qdrant per the config, auto-creating the collection if it does not exist.
    pub async fn new(config: QdrantConfig) -> Result<Self, VectorStoreError> {
        let client = Qdrant::from_url(&config.url).build().map_err(|e| {
            VectorStoreError::ConnectionError(format!("failed to connect to Qdrant: {}", e))
        })?;

        let client = Arc::new(client);

        let exists = client
            .collection_exists(&config.collection_name)
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to check collection: {}", e))
            })?;

        if !exists {
            client
                .create_collection(
                    CreateCollectionBuilder::new(&config.collection_name).vectors_config(
                        VectorParamsBuilder::new(
                            config.vector_size as u64,
                            Distance::from(config.distance),
                        ),
                    ),
                )
                .await
                .map_err(|e| {
                    VectorStoreError::StorageError(format!("failed to create collection: {}", e))
                })?;
        }

        Ok(Self { client, config })
    }

    /// Creates a store from the `QDRANT_URL` and `QDRANT_COLLECTION` environment variables.
    pub async fn from_env() -> Result<Self, VectorStoreError> {
        let url =
            std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());
        let collection_name =
            std::env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "langchainrust".to_string());

        Self::new(QdrantConfig::new(url, collection_name)).await
    }

    /// Deletes points matching a metadata key-value pair, returning the actual deleted count.
    pub async fn delete_by_metadata(
        &self,
        key: &str,
        value: &str,
    ) -> Result<usize, VectorStoreError> {
        let filter = Filter::must([Condition::matches(key, value.to_string())]);

        // Q4: count matching points by metadata first, then delete, returning the true deleted count.
        // the old implementation returned Ok(0) after deleting — callers wrongly assumed "nothing was deleted".
        let total = self.count().await as u64;
        let matched = self
            .client
            .query(
                QueryPointsBuilder::new(&self.config.collection_name)
                    .query(vec![0.0; self.config.vector_size])
                    .filter(filter.clone())
                    .limit(total.max(1))
                    .with_payload(false),
            )
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!(
                    "failed to count matching points by metadata: {}",
                    e
                ))
            })?;

        let deleted = matched.result.len();

        if deleted > 0 {
            self.client
                .delete_points(
                    DeletePointsBuilder::new(&self.config.collection_name).points(filter),
                )
                .await
                .map_err(|e| {
                    VectorStoreError::StorageError(format!(
                        "failed to delete points by metadata: {}",
                        e
                    ))
                })?;
        }

        Ok(deleted)
    }

    /// Builds the similarity-query builder, optionally attaching a metadata filter (S3).
    ///
    /// Plain and filtered retrieval share the same result parsing; this only translates
    /// [`MetadataFilter`] into a Qdrant payload `Filter` attached to the builder.
    fn build_query_builder(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<QueryPointsBuilder, VectorStoreError> {
        if query_embedding.len() != self.config.vector_size {
            return Err(VectorStoreError::StorageError(format!(
                "query vector dimension mismatch: expected {}, got {}",
                self.config.vector_size,
                query_embedding.len()
            )));
        }

        let mut builder = QueryPointsBuilder::new(&self.config.collection_name)
            .query(query_embedding.to_vec())
            .limit(k as u64)
            .with_payload(true);

        if let Some(f) = filter {
            builder = builder.filter(filter_to_qdrant(f)?);
        }

        Ok(builder)
    }

    /// Runs the query and parses the payload → [`SearchResult`] (shared by plain and filtered retrieval).
    async fn search_impl(
        &self,
        builder: QueryPointsBuilder,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let search_result = self
            .client
            .query(builder)
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("search failed: {}", e)))?;

        let results: Vec<SearchResult> = search_result
            .result
            .into_iter()
            .map(|scored_point| {
                let payload = scored_point.payload;

                let content = payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.as_str())
                    .unwrap_or("")
                    .to_string();

                let id = payload
                    .get("doc_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let mut metadata = HashMap::new();
                for (key, value) in &payload {
                    if key != "content" && key != "doc_id" {
                        if let Some(s) = value.as_str() {
                            metadata.insert(key.clone(), s.clone().into());
                        }
                    }
                }

                SearchResult {
                    document: Document {
                        content,
                        metadata,
                        id,
                    },
                    score: scored_point.score,
                }
            })
            .collect();

        Ok(results)
    }
}

// ============================================================================
// S3: MetadataFilter → Qdrant payload Filter translation
// ============================================================================

/// Match condition for a single scalar value: strings/integers/booleans go through `Match`
/// directly, anything else returns [`UnsupportedFilter`](VectorStoreError::UnsupportedFilter).
///
/// Qdrant's `Match` only supports integer exact matching (no floats); floats with an integral
/// value (e.g. `2020.0`) are normalized to i64, and true decimals cannot be expressed exactly,
/// so they error out honestly.
fn match_condition(key: &str, value: &Value) -> Result<Condition, VectorStoreError> {
    match value {
        Value::String(s) => Ok(Condition::matches(key, s.clone())),
        Value::Bool(b) => Ok(Condition::matches(key, *b)),
        Value::Number(n) => {
            let int = n
                .as_i64()
                .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64));
            match int {
                Some(i) => Ok(Condition::matches(key, i)),
                None => Err(VectorStoreError::UnsupportedFilter(format!(
                    "Qdrant match condition requires an integer, string, or boolean value, got {n}"
                ))),
            }
        }
        other => Err(VectorStoreError::UnsupportedFilter(format!(
            "Qdrant match condition requires a scalar value, got {other}"
        ))),
    }
}

/// Single-field condition → Qdrant `Condition`.
///
/// - `Eq` → a `must` match; `Ne` → a `must_not` match.
/// - `Gt/Gte/Lt/Lte` → numeric range [`Condition::range`].
/// - `In` → a `should` set of matches (any hit); `Nin` → a `must_not` set (all excluded).
fn field_to_condition(
    key: &str,
    op: FilterOp,
    value: &Value,
) -> Result<Condition, VectorStoreError> {
    match op {
        FilterOp::Eq => match_condition(key, value),
        FilterOp::Ne => Ok(Condition::from(Filter::must_not([match_condition(
            key, value,
        )?]))),
        FilterOp::Gt | FilterOp::Gte | FilterOp::Lt | FilterOp::Lte => {
            let num = value.as_f64().ok_or_else(|| {
                VectorStoreError::UnsupportedFilter(format!(
                    "Qdrant range condition requires a numeric value, got {value}"
                ))
            })?;
            let mut range = Range::default();
            match op {
                FilterOp::Gt => range.gt = Some(num),
                FilterOp::Gte => range.gte = Some(num),
                FilterOp::Lt => range.lt = Some(num),
                FilterOp::Lte => range.lte = Some(num),
                _ => unreachable!(),
            }
            Ok(Condition::range(key, range))
        }
        FilterOp::In | FilterOp::Nin => {
            let set = value.as_array().ok_or_else(|| {
                VectorStoreError::UnsupportedFilter(format!(
                    "Qdrant {:?} condition requires an array value, got {value}",
                    op
                ))
            })?;
            // Qdrant treats an empty should as always-true, so an empty In ("always false") cannot be expressed; reject explicitly.
            if set.is_empty() && op == FilterOp::In {
                return Err(VectorStoreError::UnsupportedFilter(
                    "Qdrant In condition with an empty array cannot be expressed".to_string(),
                ));
            }
            let conds: Result<Vec<Condition>, _> =
                set.iter().map(|v| match_condition(key, v)).collect();
            match op {
                FilterOp::In => Ok(Condition::from(Filter::should(conds?))),
                FilterOp::Nin => Ok(Condition::from(Filter::must_not(conds?))),
                _ => unreachable!(),
            }
        }
    }
}

/// [`MetadataFilter`] subtree → a single `Condition` (And/Or expressed as nested Filters).
///
/// Qdrant's `Condition` natively supports a `Filter` variant (`From<Filter> for Condition`),
/// so arbitrary boolean nesting lands correctly in the underlying payload filter, rather than
/// simply concatenating a should vector at the top level (which would lose semantics in
/// AND(OR, OR) scenarios).
fn to_condition(filter: &MetadataFilter) -> Result<Condition, VectorStoreError> {
    match filter {
        MetadataFilter::Field { key, op, value } => field_to_condition(key, *op, value),
        MetadataFilter::And(items) => {
            let conds: Result<Vec<Condition>, _> = items.iter().map(to_condition).collect();
            Ok(Condition::from(Filter::must(conds?)))
        }
        MetadataFilter::Or(items) => {
            let conds: Result<Vec<Condition>, _> = items.iter().map(to_condition).collect();
            Ok(Condition::from(Filter::should(conds?)))
        }
    }
}

/// [`MetadataFilter`] → Qdrant payload `Filter`.
///
/// The top level is always wrapped in `must` (an empty `And` is always true, a single condition
/// matches directly, `Or` goes through a nested should).
pub fn filter_to_qdrant(filter: &MetadataFilter) -> Result<Filter, VectorStoreError> {
    Ok(Filter::must([to_condition(filter)?]))
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError> {
        if documents.len() != embeddings.len() {
            return Err(VectorStoreError::StorageError(
                "document count and embedding count mismatch".to_string(),
            ));
        }

        if documents.is_empty() {
            return Ok(Vec::new());
        }

        for embedding in &embeddings {
            if embedding.len() != self.config.vector_size {
                return Err(VectorStoreError::StorageError(format!(
                    "vector dimension mismatch: expected {}, got {}",
                    self.config.vector_size,
                    embedding.len()
                )));
            }
        }

        let mut ids = Vec::new();
        let mut points = Vec::new();

        for (doc, embedding) in documents.into_iter().zip(embeddings) {
            let user_id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());

            // Qdrant PointId only accepts a UUID or a number, so generate an internal UUID
            let internal_uuid = Uuid::new_v4();
            let point_id = PointId::from(internal_uuid.to_string());

            let mut payload = Payload::new();
            payload.insert("content", doc.content.clone());
            payload.insert("doc_id", user_id.clone()); // the user ID is stored in the payload

            for (key, value) in &doc.metadata {
                payload.insert(key.clone(), value.clone());
            }

            let point = PointStruct::new(point_id, embedding, payload);
            points.push(point);
            ids.push(user_id);
        }

        self.client
            .upsert_points(UpsertPointsBuilder::new(
                &self.config.collection_name,
                points,
            ))
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to insert documents: {}", e))
            })?;

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let builder = self.build_query_builder(query_embedding, k, None)?;
        self.search_impl(builder).await
    }

    /// S3: similarity search with metadata filtering — filtering is delegated to the server (payload filter).
    async fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let builder = self.build_query_builder(query_embedding, k, filter)?;
        self.search_impl(builder).await
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let filter = Filter::must([Condition::matches("doc_id", id.to_string())]);

        let results = self
            .client
            .query(
                QueryPointsBuilder::new(&self.config.collection_name)
                    .query(vec![0.0; self.config.vector_size])
                    .filter(filter)
                    .limit(1)
                    .with_payload(true),
            )
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to get document: {}", e))
            })?;

        if let Some(point) = results.result.first() {
            let payload_map = point.payload.clone();

            let content = payload_map
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.as_str())
                .unwrap_or("")
                .to_string();

            let doc_id = payload_map
                .get("doc_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let mut metadata = HashMap::new();
            for (key, value) in &payload_map {
                if key != "content" && key != "doc_id" {
                    if let Some(s) = value.as_str() {
                        metadata.insert(key.clone(), s.clone().into());
                    }
                }
            }

            Ok(Some(Document {
                content,
                metadata,
                id: doc_id,
            }))
        } else {
            Ok(None)
        }
    }

    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let filter = Filter::must([Condition::matches("doc_id", id.to_string())]);

        let results = self
            .client
            .query(
                QueryPointsBuilder::new(&self.config.collection_name)
                    .query(vec![0.0; self.config.vector_size])
                    .filter(filter)
                    .limit(1)
                    .with_payload(true),
            )
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("failed to get vector: {}", e)))?;

        if let Some(point) = results.result.first() {
            if let Some(vectors) = &point.vectors {
                if let Some(qdrant_client::qdrant::vector_output::Vector::Dense(dense)) =
                    vectors.get_vector()
                {
                    return Ok(Some(dense.data.clone()));
                }
            }
        }
        Ok(None)
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let filter = Filter::must([Condition::matches("doc_id", id.to_string())]);

        self.client
            .delete_points(DeletePointsBuilder::new(&self.config.collection_name).points(filter))
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to delete document: {}", e))
            })?;

        Ok(())
    }

    async fn count(&self) -> usize {
        let info = self
            .client
            .collection_info(&self.config.collection_name)
            .await;

        info.map(|i| i.result.and_then(|r| r.points_count).unwrap_or(0) as usize)
            .unwrap_or(0)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let collection_name = self.config.collection_name.clone();

        self.client
            .delete_collection(&collection_name)
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to delete collection: {}", e))
            })?;

        self.client
            .create_collection(
                CreateCollectionBuilder::new(&collection_name).vectors_config(
                    VectorParamsBuilder::new(
                        self.config.vector_size as u64,
                        Distance::from(self.config.distance),
                    ),
                ),
            )
            .await
            .map_err(|e| {
                VectorStoreError::StorageError(format!("failed to recreate collection: {}", e))
            })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = QdrantConfig::default();
        assert_eq!(config.url, "http://localhost:6334");
        assert_eq!(config.collection_name, "langchainrust");
        assert_eq!(config.vector_size, 1536);
    }

    #[test]
    fn test_config_builder() {
        let config = QdrantConfig::new("http://custom:6334", "test_collection")
            .with_vector_size(3072)
            .with_distance(QdrantDistance::Euclid);

        assert_eq!(config.url, "http://custom:6334");
        assert_eq!(config.collection_name, "test_collection");
        assert_eq!(config.vector_size, 3072);
        assert!(matches!(config.distance, QdrantDistance::Euclid));
    }

    /// S3: single-field Eq → must match condition.
    #[test]
    fn test_filter_to_qdrant_eq() {
        let f = filter_to_qdrant(&MetadataFilter::field("lang", FilterOp::Eq, "rust")).unwrap();
        let expected = Filter::must([Condition::matches("lang", "rust".to_string())]);
        assert_eq!(f, expected);
    }

    /// S3: integral floats normalized to i64; Ne → must_not.
    #[test]
    fn test_filter_to_qdrant_ne_number() {
        let f = filter_to_qdrant(&MetadataFilter::field("year", FilterOp::Ne, 2020.0)).unwrap();
        let expected = Filter::must([Condition::from(Filter::must_not([Condition::matches(
            "year", 2020_i64,
        )]))]);
        assert_eq!(f, expected);
    }

    /// S3: Gt/Gte/Lt/Lte → numeric range.
    #[test]
    fn test_filter_to_qdrant_range() {
        let f = filter_to_qdrant(&MetadataFilter::field("year", FilterOp::Gte, 2020)).unwrap();
        let expected = Filter::must([Condition::range(
            "year",
            Range {
                gte: Some(2020.0),
                ..Default::default()
            },
        )]);
        assert_eq!(f, expected);
    }

    /// S3: In → a should set of matches; Nin → a must_not set.
    #[test]
    fn test_filter_to_qdrant_in_nin() {
        let f =
            filter_to_qdrant(&MetadataFilter::field("tag", FilterOp::In, vec!["a", "b"])).unwrap();
        let expected = Filter::must([Condition::from(Filter::should([
            Condition::matches("tag", "a".to_string()),
            Condition::matches("tag", "b".to_string()),
        ]))]);
        assert_eq!(f, expected);

        let f = filter_to_qdrant(&MetadataFilter::field("tag", FilterOp::Nin, vec!["a"])).unwrap();
        let expected = Filter::must([Condition::from(Filter::must_not([Condition::matches(
            "tag",
            "a".to_string(),
        )]))]);
        assert_eq!(f, expected);
    }

    /// S3: AND/OR combination → nested Filter conditions (not flattened, preserving AND(OR,OR) semantics).
    #[test]
    fn test_filter_to_qdrant_and_or() {
        let f = MetadataFilter::and(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "rust"),
            MetadataFilter::or(vec![
                MetadataFilter::field("year", FilterOp::Gte, 2020),
                MetadataFilter::field("tag", FilterOp::In, vec!["ml"]),
            ]),
        ]);
        let expected = Filter::must([Condition::from(Filter::must([
            Condition::matches("lang", "rust".to_string()),
            Condition::from(Filter::should([
                Condition::range(
                    "year",
                    Range {
                        gte: Some(2020.0),
                        ..Default::default()
                    },
                ),
                Condition::from(Filter::should([Condition::matches(
                    "tag",
                    "ml".to_string(),
                )])),
            ])),
        ]))]);
        assert_eq!(filter_to_qdrant(&f).unwrap(), expected);
    }

    /// S3: inexpressible constructs honestly report UnsupportedFilter.
    #[test]
    fn test_filter_to_qdrant_unsupported() {
        // Qdrant Match does not support float exact matching.
        let float_eq = filter_to_qdrant(&MetadataFilter::field("score", FilterOp::Eq, 0.5));
        assert!(matches!(
            float_eq,
            Err(VectorStoreError::UnsupportedFilter(_))
        ));

        // range conditions require a numeric value.
        let range_on_str = filter_to_qdrant(&MetadataFilter::field("year", FilterOp::Gt, "abc"));
        assert!(matches!(
            range_on_str,
            Err(VectorStoreError::UnsupportedFilter(_))
        ));

        // an empty In cannot be expressed (Qdrant treats an empty should as always-true).
        let empty_in = filter_to_qdrant(&MetadataFilter::field(
            "tag",
            FilterOp::In,
            Vec::<String>::new(),
        ));
        assert!(matches!(
            empty_in,
            Err(VectorStoreError::UnsupportedFilter(_))
        ));
    }
}
