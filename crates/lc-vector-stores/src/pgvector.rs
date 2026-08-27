//! PGVector vector store (PostgreSQL + pgvector extension)
//!
//! [`PGVectorStore`] is a typed `VectorStore` implementation backed by `sqlx` + `pgvector`
//! (enabled under the `pgvector-storage` feature). The feature is off by default:
//!
//! - sqlx is a heavy dependency; enabling it noticeably lengthens compile time;
//! - historical comments claimed a conflict with `rusqlite` (libsqlite3-sys linkage) — the
//!   sqlx 0.8 selected here **shares** `libsqlite3-sys` 0.28 with rusqlite 0.31, so the two
//!   can coexist (upgrading sqlx to 0.8.6+ would pull `libsqlite3-sys` 0.31 instead and
//!   trigger a links conflict, hence the 0.8 pin).
//!
//! Prerequisite (run by an administrator): `CREATE EXTENSION vector;`. Then:
//! ```text
//! let store = lc_vector_stores::pgvector::PGVectorStore::connect(
//!     "postgres://postgres:postgres@localhost:5432/vectors", "docs", 768).await?;
//! store.initialize().await?;
//! ```
//!
//! Similarity uses cosine distance (`<=>`), score = `1 - distance`, consistent with the
//! in-memory store's cosine similarity convention (range [-1, 1]). The SQL injection defense
//! follows `validate_table_name` (table name whitelist); filter metadata keys go through the
//! same regex whitelist; all comparison values are bound as sqlx parameters.
//!
//! Honest boundary: runtime correctness depends on an external PG instance, and CI has none —
//! integration tests are marked `#[ignore]` (requires `PGVECTOR_TEST_URL`); local unit tests
//! cover only the pure SQL construction function (`build_filter_sql`).

use std::collections::HashMap;
use std::sync::LazyLock;

use async_trait::async_trait;
use pgvector::Vector;
use regex::Regex;
use serde_json::Value;
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
use sqlx::types::Json;
use sqlx::{query, query_scalar, Row};
use uuid::Uuid;

use crate::{Document, FilterOp, MetadataFilter, SearchResult, VectorStore, VectorStoreError};

static TABLE_NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*$").unwrap());

static META_KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*$").unwrap());

/// Validate that a table name is safe for SQL interpolation.
///
/// Only allows: `^[a-zA-Z_][a-zA-Z0-9_]*$`
/// This prevents SQL injection via table names.
pub fn validate_table_name(table: &str) -> Result<(), VectorStoreError> {
    if TABLE_NAME_RE.is_match(table) {
        Ok(())
    } else {
        Err(VectorStoreError::ConfigError(format!(
            "Invalid table name '{}': must match ^[a-zA-Z_][a-zA-Z0-9_]*$",
            table
        )))
    }
}

/// Validate that a metadata key is safe to interpolate into a SQL expression.
///
/// Keys appear inside `metadata->>'<key>'`, so a key containing a quote could
/// break out of the string literal. Same whitelist as [`validate_table_name`];
/// filters referencing unsafe keys are rejected with `UnsupportedFilter` (honest
/// error, never a silent mismatch).
fn validate_metadata_key(key: &str) -> Result<(), VectorStoreError> {
    if META_KEY_RE.is_match(key) {
        Ok(())
    } else {
        Err(VectorStoreError::UnsupportedFilter(format!(
            "metadata key '{key}' is not safe to interpolate into SQL; \
             must match ^[a-zA-Z_][a-zA-Z0-9_]*$"
        )))
    }
}

/// Build CREATE TABLE SQL (pure function, convenient for testing)
pub fn build_table_sql(table: &str, dim: usize) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {} (id TEXT PRIMARY KEY, content TEXT, metadata JSONB, embedding vector({}))",
        table, dim
    )
}

/// PGVector store configuration.
#[derive(Debug, Clone)]
pub struct PGVectorConfig {
    /// PostgreSQL connection string, e.g. `postgres://postgres:postgres@localhost:5432/vectors`.
    pub database_url: String,
    /// Table name holding vector data (must satisfy the [`validate_table_name`] whitelist).
    pub table: String,
    /// Vector dimension (must match the `vector(dim)` used at table creation).
    pub dimension: usize,
}

impl PGVectorConfig {
    /// Constructs a config.
    pub fn new(
        database_url: impl Into<String>,
        table: impl Into<String>,
        dimension: usize,
    ) -> Self {
        Self {
            database_url: database_url.into(),
            table: table.into(),
            dimension,
        }
    }
}

/// A filter value bound to a SQL parameter.
///
/// Text values bind as `text` (compared against `metadata->>'key'`); numbers bind as `float8`
/// (numeric comparison via `(metadata->>'key')::float8` guarded by `jsonb_typeof`).
#[derive(Debug, Clone, PartialEq)]
pub enum FilterBinding {
    /// Text value (string / boolean).
    Text(String),
    /// Numeric value.
    Number(f64),
}

/// Result of [`build_filter_sql`]: a `WHERE` clause + bound values ordered by `$N` appearance.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterSql {
    /// SQL condition clause without the `WHERE` prefix (`TRUE` / `FALSE` for empty combinations).
    pub clause: String,
    /// Bound values, one per `$N` placeholder in order.
    pub bindings: Vec<FilterBinding>,
}

/// Translates a [`MetadataFilter`] into a PGVector-ready SQL `WHERE` clause.
///
/// Pure function, touches no database — unit tests need no PG instance. `start_idx` is the
/// first `$N` placeholder number (in retrieval queries `$1` is taken by the query vector, so
/// filtering starts at `$2`).
///
/// Semantics align with [`MetadataFilter::matches`](crate::MetadataFilter::matches):
/// - missing fields do **not** match `Eq` / `In` / ordering ops (`metadata->'key'` is NULL);
/// - missing fields **do** match `Ne` / `Nin` (wrapped as `metadata->'key' IS NULL OR …`);
/// - numeric comparisons add a `jsonb_typeof(metadata->'key') = 'number'` guard to avoid a
///   `::float8` cast error on non-numeric fields;
/// - an empty `In` is always false (`FALSE`), an empty `Nin` is always true (`TRUE`).
pub fn build_filter_sql(
    filter: &MetadataFilter,
    start_idx: usize,
) -> Result<FilterSql, VectorStoreError> {
    let mut ctx = FilterContext {
        bindings: Vec::new(),
        next: start_idx,
    };
    let clause = walk_filter(filter, &mut ctx)?;
    Ok(FilterSql {
        clause,
        bindings: ctx.bindings,
    })
}

/// Binding context while building filter SQL: accumulated bound values + the next `$N` number.
struct FilterContext {
    bindings: Vec<FilterBinding>,
    next: usize,
}

impl FilterContext {
    fn text(&mut self, s: String) -> usize {
        self.bindings.push(FilterBinding::Text(s));
        let idx = self.next;
        self.next += 1;
        idx
    }

    fn number(&mut self, f: f64) -> usize {
        self.bindings.push(FilterBinding::Number(f));
        let idx = self.next;
        self.next += 1;
        idx
    }
}

fn walk_filter(
    filter: &MetadataFilter,
    ctx: &mut FilterContext,
) -> Result<String, VectorStoreError> {
    match filter {
        MetadataFilter::Field { key, op, value } => {
            validate_metadata_key(key)?;
            field_clause(key, *op, value, ctx)
        }
        MetadataFilter::And(items) if items.is_empty() => Ok("TRUE".to_string()),
        MetadataFilter::Or(items) if items.is_empty() => Ok("FALSE".to_string()),
        MetadataFilter::And(items) => {
            let parts: Result<Vec<_>, _> = items.iter().map(|f| walk_filter(f, ctx)).collect();
            Ok(format!("({})", parts?.join(" AND ")))
        }
        MetadataFilter::Or(items) => {
            let parts: Result<Vec<_>, _> = items.iter().map(|f| walk_filter(f, ctx)).collect();
            Ok(format!("({})", parts?.join(" OR ")))
        }
    }
}

fn field_clause(
    key: &str,
    op: FilterOp,
    value: &Value,
    ctx: &mut FilterContext,
) -> Result<String, VectorStoreError> {
    match op {
        FilterOp::Eq | FilterOp::Ne => eq_ne_clause(key, op, value, ctx),
        FilterOp::Gt | FilterOp::Gte | FilterOp::Lt | FilterOp::Lte => {
            ordering_clause(key, op, value, ctx)
        }
        FilterOp::In | FilterOp::Nin => in_clause(key, op, value, ctx),
    }
}

fn eq_ne_clause(
    key: &str,
    op: FilterOp,
    value: &Value,
    ctx: &mut FilterContext,
) -> Result<String, VectorStoreError> {
    let cmp = if op == FilterOp::Eq { "=" } else { "<>" };
    let base = match value {
        Value::String(s) => {
            let idx = ctx.text(s.clone());
            format!("metadata->>'{}' {} ${}", key, cmp, idx)
        }
        Value::Bool(b) => {
            let idx = ctx.text(b.to_string());
            format!("metadata->>'{}' {} ${}", key, cmp, idx)
        }
        Value::Number(n) => {
            let f = n.as_f64().ok_or_else(|| {
                VectorStoreError::UnsupportedFilter(format!(
                    "PGVector filter value must be a finite number, got {n}"
                ))
            })?;
            let idx = ctx.number(f);
            format!(
                "jsonb_typeof(metadata->'{}') = 'number' AND (metadata->>'{}')::float8 {} ${}",
                key, key, cmp, idx
            )
        }
        other => {
            return Err(VectorStoreError::UnsupportedFilter(format!(
                "PGVector Eq/Ne requires a string, number, or boolean filter value, got {other}"
            )))
        }
    };
    if op == FilterOp::Ne {
        // SQL NULL semantics: a missing field matches Ne (consistent with filter.rs).
        Ok(format!("(metadata->'{}' IS NULL OR {})", key, base))
    } else {
        Ok(base)
    }
}

fn ordering_clause(
    key: &str,
    op: FilterOp,
    value: &Value,
    ctx: &mut FilterContext,
) -> Result<String, VectorStoreError> {
    let f = value.as_f64().ok_or_else(|| {
        VectorStoreError::UnsupportedFilter(format!(
            "PGVector {:?} filter requires a numeric value, got {}",
            op, value
        ))
    })?;
    let sql_op = match op {
        FilterOp::Gt => ">",
        FilterOp::Gte => ">=",
        FilterOp::Lt => "<",
        FilterOp::Lte => "<=",
        FilterOp::Eq | FilterOp::Ne | FilterOp::In | FilterOp::Nin => {
            unreachable!("ordering_clause only receives ordering ops")
        }
    };
    let idx = ctx.number(f);
    Ok(format!(
        "jsonb_typeof(metadata->'{}') = 'number' AND (metadata->>'{}')::float8 {} ${}",
        key, key, sql_op, idx
    ))
}

fn in_clause(
    key: &str,
    op: FilterOp,
    value: &Value,
    ctx: &mut FilterContext,
) -> Result<String, VectorStoreError> {
    let arr = value.as_array().ok_or_else(|| {
        VectorStoreError::UnsupportedFilter(format!(
            "PGVector In/Nin requires an array value, got {}",
            value
        ))
    })?;
    if arr.is_empty() {
        // an empty In is always false, an empty Nin is always true (consistent with filter.rs semantics).
        return Ok(if op == FilterOp::In {
            "FALSE".to_string()
        } else {
            "TRUE".to_string()
        });
    }
    let neg = if op == FilterOp::Nin { "NOT " } else { "" };
    let all_text = arr.iter().all(|v| v.is_string());
    let all_number = arr.iter().all(|v| v.is_number());

    let base = if all_text {
        let list: Vec<String> = arr
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| format!("${}", ctx.text(s.to_string())))
            .collect();
        format!("metadata->>'{}' {}IN ({})", key, neg, list.join(", "))
    } else if all_number {
        let list: Vec<String> = arr
            .iter()
            .filter_map(|v| v.as_f64())
            .map(|f| format!("${}", ctx.number(f)))
            .collect();
        format!(
            "jsonb_typeof(metadata->'{}') = 'number' AND (metadata->>'{}')::float8 {}IN ({})",
            key,
            key,
            neg,
            list.join(", ")
        )
    } else {
        return Err(VectorStoreError::UnsupportedFilter(format!(
            "PGVector In/Nin requires a homogeneous array of strings or numbers, got {}",
            value
        )));
    };

    if op == FilterOp::Nin {
        Ok(format!("(metadata->'{}' IS NULL OR {})", key, base))
    } else {
        Ok(base)
    }
}

/// Maps sqlx errors to `VectorStoreError` (connection-class errors are distinguished separately).
fn map_sqlx(e: sqlx::Error) -> VectorStoreError {
    match e {
        sqlx::Error::PoolTimedOut | sqlx::Error::PoolClosed => {
            VectorStoreError::ConnectionError(format!("PostgreSQL connection error: {e}"))
        }
        other => VectorStoreError::StorageError(format!("PostgreSQL error: {other}")),
    }
}

/// Decode error (failed to read a row field).
fn map_decode(e: sqlx::Error) -> VectorStoreError {
    VectorStoreError::StorageError(format!("failed to decode PostgreSQL row: {e}"))
}

/// Converts query result rows (score already computed by SQL) into `SearchResult`s.
fn rows_to_results(rows: Vec<PgRow>) -> Result<Vec<SearchResult>, VectorStoreError> {
    rows.into_iter()
        .map(|row| {
            let id: String = row.try_get("id").map_err(map_decode)?;
            let content: String = row.try_get("content").map_err(map_decode)?;
            let metadata_json: Json<Value> = row.try_get("metadata").map_err(map_decode)?;
            let metadata: HashMap<String, Value> = serde_json::from_value(metadata_json.0)
                .map_err(|e| {
                    VectorStoreError::StorageError(format!("invalid metadata JSONB: {e}"))
                })?;
            let score: f64 = row.try_get("score").map_err(map_decode)?;
            Ok(SearchResult {
                document: Document {
                    id: Some(id),
                    content,
                    metadata,
                },
                score: score as f32,
            })
        })
        .collect()
}

/// Typed PGVector vector store.
///
/// Uses a `sqlx` connection pool to access PostgreSQL + the pgvector extension. Table names go
/// through the [`validate_table_name`] whitelist; the vector dimension is validated client-side
/// (the DB-side `vector(dim)` column also backstops it).
pub struct PGVectorStore {
    pool: PgPool,
    table: String,
    dim: usize,
}

impl PGVectorStore {
    /// Creates a store from an existing `sqlx` connection pool.
    pub fn new(pool: PgPool, table: &str, dim: usize) -> Result<Self, VectorStoreError> {
        validate_table_name(table)?;
        if dim == 0 {
            return Err(VectorStoreError::ConfigError(
                "PGVector dimension must be > 0".to_string(),
            ));
        }
        Ok(Self {
            pool,
            table: table.to_string(),
            dim,
        })
    }

    /// Connects to PostgreSQL and creates a store.
    ///
    /// Only builds the connection pool, not the table; call [`initialize`](Self::initialize) to create the table.
    pub async fn connect(
        database_url: &str,
        table: &str,
        dim: usize,
    ) -> Result<Self, VectorStoreError> {
        validate_table_name(table)?;
        if dim == 0 {
            return Err(VectorStoreError::ConfigError(
                "PGVector dimension must be > 0".to_string(),
            ));
        }
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await
            .map_err(|e| {
                VectorStoreError::ConnectionError(format!("failed to connect to PostgreSQL: {e}"))
            })?;
        Ok(Self {
            pool,
            table: table.to_string(),
            dim,
        })
    }

    /// Connects from a config.
    pub async fn from_config(config: PGVectorConfig) -> Result<Self, VectorStoreError> {
        Self::connect(&config.database_url, &config.table, config.dimension).await
    }

    /// Creates the table (`CREATE TABLE IF NOT EXISTS`, idempotent).
    ///
    /// `CREATE EXTENSION vector;` must be run by an administrator first (this method does not
    /// run it — it needs superuser privileges; without them it errors rather than silently).
    pub async fn initialize(&self) -> Result<(), VectorStoreError> {
        let sql = build_table_sql(&self.table, self.dim);
        query(&sql).execute(&self.pool).await.map_err(map_sqlx)?;
        Ok(())
    }

    /// Unified retrieval implementation: optional metadata filtering + optional minimum score threshold.
    ///
    /// SQL shape: `SELECT … (1 - (embedding <=> $1)) AS score FROM <table>
    /// [WHERE <filter> [AND (1 - (embedding <=> $1)) >= $N]]
    /// ORDER BY score DESC LIMIT $M`. When `$1` (the query vector) is referenced by the two
    /// `<=>` uses it is the same parameter, so it only needs to be bound once.
    async fn search_with(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if query_embedding.len() != self.dim {
            return Err(VectorStoreError::ConfigError(format!(
                "query embedding dimension {} does not match store dimension {}",
                query_embedding.len(),
                self.dim
            )));
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut sql =
            String::from("SELECT id, content, metadata, (1 - (embedding <=> $1)) AS score FROM ");
        sql.push_str(&self.table);
        let mut has_where = false;
        let mut next = 2usize;
        let mut binds: Vec<FilterBinding> = Vec::new();

        if let Some(f) = filter {
            let fs = build_filter_sql(f, next)?;
            sql.push_str(" WHERE ");
            sql.push_str(&fs.clause);
            has_where = true;
            next += fs.bindings.len();
            binds = fs.bindings;
        }
        if let Some(t) = min_score {
            if has_where {
                sql.push_str(" AND ");
            } else {
                sql.push_str(" WHERE ");
            }
            sql.push_str(&format!("(1 - (embedding <=> $1)) >= ${}", next));
            binds.push(FilterBinding::Number(t as f64));
            next += 1;
        }
        sql.push_str(&format!(" ORDER BY score DESC LIMIT ${}", next));

        let mut q = query(&sql).bind(Vector::from(query_embedding.to_vec()));
        for b in &binds {
            q = match b {
                FilterBinding::Text(s) => q.bind(s.clone()),
                FilterBinding::Number(f) => q.bind(*f),
            };
        }
        q = q.bind(k as i64);
        let rows = q.fetch_all(&self.pool).await.map_err(map_sqlx)?;
        rows_to_results(rows)
    }
}

#[async_trait]
impl VectorStore for PGVectorStore {
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
        for emb in &embeddings {
            if emb.len() != self.dim {
                return Err(VectorStoreError::ConfigError(format!(
                    "embedding dimension {} does not match store dimension {}",
                    emb.len(),
                    self.dim
                )));
            }
        }

        let mut tx = self.pool.begin().await.map_err(map_sqlx)?;
        let mut ids = Vec::with_capacity(documents.len());
        let sql = format!(
            "INSERT INTO {} (id, content, metadata, embedding) VALUES ($1, $2, $3::jsonb, $4) \
             ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, \
             metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
            self.table
        );
        for (doc, emb) in documents.into_iter().zip(embeddings.into_iter()) {
            let id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
            query(&sql)
                .bind(&id)
                .bind(&doc.content)
                .bind(Json(doc.metadata))
                .bind(Vector::from(emb))
                .execute(&mut *tx)
                .await
                .map_err(map_sqlx)?;
            ids.push(id);
        }
        tx.commit().await.map_err(map_sqlx)?;
        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        self.search_with(query_embedding, k, None, None).await
    }

    async fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        self.search_with(query_embedding, k, filter, None).await
    }

    async fn similarity_search_with_min_score(
        &self,
        query_embedding: &[f32],
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Q2: override to get the exact "filter by threshold first, then take top-k" semantics,
        // not the default implementation's "top-k then re-filter" approximation.
        self.search_with(query_embedding, k, None, min_score).await
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let sql = format!("SELECT content, metadata FROM {} WHERE id = $1", self.table);
        let row = query(&sql)
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(map_sqlx)?;
        let Some(row) = row else {
            return Ok(None);
        };
        let content: String = row.try_get("content").map_err(map_decode)?;
        let metadata_json: Json<Value> = row.try_get("metadata").map_err(map_decode)?;
        let metadata: HashMap<String, Value> = serde_json::from_value(metadata_json.0)
            .map_err(|e| VectorStoreError::StorageError(format!("invalid metadata JSONB: {e}")))?;
        Ok(Some(Document {
            id: Some(id.to_string()),
            content,
            metadata,
        }))
    }

    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let sql = format!("SELECT embedding FROM {} WHERE id = $1", self.table);
        let row = query(&sql)
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(map_sqlx)?;
        let Some(row) = row else {
            return Ok(None);
        };
        let v: Vector = row.try_get("embedding").map_err(map_decode)?;
        Ok(Some(v.to_vec()))
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let sql = format!("DELETE FROM {} WHERE id = $1", self.table);
        query(&sql)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(map_sqlx)?;
        Ok(())
    }

    async fn count(&self) -> usize {
        let sql = format!("SELECT count(*) FROM {}", self.table);
        query_scalar::<_, i64>(&sql)
            .fetch_one(&self.pool)
            .await
            .map(|n| n as usize)
            .unwrap_or(0)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let sql = format!("DELETE FROM {}", self.table);
        query(&sql).execute(&self.pool).await.map_err(map_sqlx)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FilterOp;
    use serde_json::json;

    #[test]
    fn test_build_table_sql() {
        let sql = build_table_sql("docs", 1536);
        assert!(sql.contains("CREATE TABLE"));
        assert!(sql.contains("vector(1536)"));
        assert!(sql.contains("docs"));
    }

    #[test]
    fn test_build_table_sql_different_dim() {
        let sql = build_table_sql("embeddings", 768);
        assert!(sql.contains("vector(768)"));
        assert!(sql.contains("embeddings"));
    }

    #[test]
    fn test_build_table_sql_contains_metadata() {
        let sql = build_table_sql("docs", 1536);
        assert!(sql.contains("metadata JSONB"));
        assert!(sql.contains("id TEXT PRIMARY KEY"));
    }

    #[test]
    fn test_validate_table_name_valid() {
        assert!(validate_table_name("users").is_ok());
        assert!(validate_table_name("my_table").is_ok());
        assert!(validate_table_name("_private").is_ok());
        assert!(validate_table_name("Table123").is_ok());
    }

    #[test]
    fn test_validate_table_name_invalid() {
        // SQL injection attempts
        assert!(validate_table_name("users; DROP TABLE users--").is_err());
        assert!(validate_table_name("users; DROP TABLE users").is_err());
        assert!(validate_table_name("123table").is_err()); // starts with digit
        assert!(validate_table_name("user-table").is_err()); // contains hyphen
        assert!(validate_table_name("user.table").is_err()); // contains dot
        assert!(validate_table_name("").is_err()); // empty
    }

    // ---- build_filter_sql pure-function unit tests (no PG instance needed) ----

    /// Eq string → text comparison; bound values follow `$N` order.
    #[test]
    fn test_filter_eq_string() {
        let f = MetadataFilter::field("lang", FilterOp::Eq, "rust");
        let fs = build_filter_sql(&f, 2).unwrap();
        assert_eq!(fs.clause, "metadata->>'lang' = $2");
        assert_eq!(fs.bindings, vec![FilterBinding::Text("rust".to_string())]);
    }

    /// Eq number → `jsonb_typeof` guard + `::float8` comparison.
    #[test]
    fn test_filter_eq_number() {
        let f = MetadataFilter::field("year", FilterOp::Eq, 2024);
        let fs = build_filter_sql(&f, 2).unwrap();
        assert_eq!(
            fs.clause,
            "jsonb_typeof(metadata->'year') = 'number' AND (metadata->>'year')::float8 = $2"
        );
        assert_eq!(fs.bindings, vec![FilterBinding::Number(2024.0)]);
    }

    /// Ne → missing-field match semantics (IS NULL OR).
    #[test]
    fn test_filter_ne_string() {
        let f = MetadataFilter::field("lang", FilterOp::Ne, "rust");
        let fs = build_filter_sql(&f, 2).unwrap();
        assert_eq!(
            fs.clause,
            "(metadata->'lang' IS NULL OR metadata->>'lang' <> $2)"
        );
    }

    /// Ordering ops → `::float8` comparison; a non-numeric value raises UnsupportedFilter.
    #[test]
    fn test_filter_ordering() {
        let f = MetadataFilter::field("year", FilterOp::Gte, 2021);
        let fs = build_filter_sql(&f, 2).unwrap();
        assert_eq!(
            fs.clause,
            "jsonb_typeof(metadata->'year') = 'number' AND (metadata->>'year')::float8 >= $2"
        );
        assert_eq!(fs.bindings, vec![FilterBinding::Number(2021.0)]);

        let bad = MetadataFilter::field("year", FilterOp::Gt, "abc");
        let err = build_filter_sql(&bad, 2).unwrap_err();
        assert!(matches!(err, VectorStoreError::UnsupportedFilter(_)));
    }

    /// In/Nin string arrays.
    #[test]
    fn test_filter_in_strings() {
        let f = MetadataFilter::field("source", FilterOp::In, vec!["docs", "web"]);
        let fs = build_filter_sql(&f, 2).unwrap();
        assert_eq!(fs.clause, "metadata->>'source' IN ($2, $3)");
        assert_eq!(
            fs.bindings,
            vec![
                FilterBinding::Text("docs".to_string()),
                FilterBinding::Text("web".to_string())
            ]
        );

        let nin = MetadataFilter::field("source", FilterOp::Nin, vec!["docs", "web"]);
        let fs = build_filter_sql(&nin, 2).unwrap();
        assert_eq!(
            fs.clause,
            "(metadata->'source' IS NULL OR metadata->>'source' NOT IN ($2, $3))"
        );
    }

    /// In numeric arrays → `::float8 IN`.
    #[test]
    fn test_filter_in_numbers() {
        let f = MetadataFilter::field("year", FilterOp::In, vec![2020, 2024]);
        let fs = build_filter_sql(&f, 2).unwrap();
        assert_eq!(
            fs.clause,
            "jsonb_typeof(metadata->'year') = 'number' AND (metadata->>'year')::float8 IN ($2, $3)"
        );
        assert_eq!(
            fs.bindings,
            vec![FilterBinding::Number(2020.0), FilterBinding::Number(2024.0)]
        );
    }

    /// Empty In → FALSE, empty Nin → TRUE (consistent with filter.rs semantics).
    #[test]
    fn test_filter_empty_in_nin() {
        let empty_in: Vec<String> = vec![];
        let f = MetadataFilter::field("tag", FilterOp::In, empty_in.clone());
        assert_eq!(build_filter_sql(&f, 2).unwrap().clause, "FALSE");

        let f = MetadataFilter::field("tag", FilterOp::Nin, empty_in);
        assert_eq!(build_filter_sql(&f, 2).unwrap().clause, "TRUE");
    }

    /// And/Or nesting → parenthesized groups; empty And → TRUE, empty Or → FALSE.
    #[test]
    fn test_filter_and_or_nesting() {
        let f = MetadataFilter::and(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "rust"),
            MetadataFilter::field("year", FilterOp::Gt, 2020),
        ]);
        let fs = build_filter_sql(&f, 2).unwrap();
        assert_eq!(
            fs.clause,
            "(metadata->>'lang' = $2 AND jsonb_typeof(metadata->'year') = 'number' AND (metadata->>'year')::float8 > $3)"
        );
        assert_eq!(fs.bindings.len(), 2);

        let empty = MetadataFilter::and(vec![]);
        assert_eq!(build_filter_sql(&empty, 2).unwrap().clause, "TRUE");

        let empty_or = MetadataFilter::or(vec![]);
        assert_eq!(build_filter_sql(&empty_or, 2).unwrap().clause, "FALSE");
    }

    /// `start_idx` decides the first `$N` number ($1 is taken by the query vector in retrieval queries).
    #[test]
    fn test_filter_start_idx() {
        let f = MetadataFilter::field("lang", FilterOp::Eq, "rust");
        let fs = build_filter_sql(&f, 5).unwrap();
        assert_eq!(fs.clause, "metadata->>'lang' = $5");
    }

    /// Invalid metadata key → UnsupportedFilter (SQL injection defense).
    #[test]
    fn test_filter_invalid_key_errors() {
        let f = MetadataFilter::field("user id", FilterOp::Eq, "x");
        let err = build_filter_sql(&f, 2).unwrap_err();
        assert!(matches!(err, VectorStoreError::UnsupportedFilter(_)));

        let inject = MetadataFilter::field("x' OR 1=1--", FilterOp::Eq, "x");
        assert!(build_filter_sql(&inject, 2).is_err());
    }

    /// Mixed-type In array → UnsupportedFilter (rejects ambiguous comparisons).
    #[test]
    fn test_filter_mixed_in_errors() {
        let f = MetadataFilter::field("tag", FilterOp::In, json!([1, "x"]));
        assert!(build_filter_sql(&f, 2).is_err());
    }

    // ---- integration tests (need an external PG + the pgvector extension, #[ignore] by default) ----

    fn test_url() -> String {
        std::env::var("PGVECTOR_TEST_URL")
            .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5432/vectors".to_string())
    }

    /// Full roundtrip: create table + add/delete/query + cosine scores.
    #[ignore = "requires a running PostgreSQL with the pgvector extension (CREATE EXTENSION vector); set PGVECTOR_TEST_URL"]
    #[tokio::test]
    async fn test_pgvector_roundtrip() {
        let table = format!("pgv_roundtrip_{}", Uuid::new_v4().simple());
        let store = PGVectorStore::connect(&test_url(), &table, 3)
            .await
            .unwrap();
        store.initialize().await.unwrap();

        let docs = vec![Document::new("rust doc"), Document::new("python doc")];
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let ids = store.add_documents(docs, embeddings).await.unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(store.count().await, 2);

        // cosine scores: same direction ≈ 1, orthogonal ≈ 0.
        let results = store.similarity_search(&[0.9, 0.1, 0.0], 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].document.content.contains("rust"));
        assert!(results[0].score > results[1].score);
        assert!(results[0].score > 0.9);

        let fetched = store.get_document(&ids[0]).await.unwrap().unwrap();
        assert_eq!(fetched.content, "rust doc");
        let emb = store.get_embedding(&ids[0]).await.unwrap().unwrap();
        assert_eq!(emb, vec![1.0, 0.0, 0.0]);

        store.delete_document(&ids[0]).await.unwrap();
        assert_eq!(store.count().await, 1);
        store.clear().await.unwrap();
        assert_eq!(store.count().await, 0);

        let _ = query(&format!("DROP TABLE IF EXISTS {table}"))
            .execute(&store.pool)
            .await;
    }

    /// S3 filtering is pushed down to the PG side.
    #[ignore = "requires a running PostgreSQL with the pgvector extension (CREATE EXTENSION vector); set PGVECTOR_TEST_URL"]
    #[tokio::test]
    async fn test_pgvector_metadata_filter() {
        let table = format!("pgv_filter_{}", Uuid::new_v4().simple());
        let store = PGVectorStore::connect(&test_url(), &table, 3)
            .await
            .unwrap();
        store.initialize().await.unwrap();

        store
            .add_documents(
                vec![
                    Document::new("rust doc")
                        .with_metadata("lang", "rust")
                        .with_metadata("year", 2024),
                    Document::new("python doc")
                        .with_metadata("lang", "python")
                        .with_metadata("year", 2023),
                    Document::new("rust legacy")
                        .with_metadata("lang", "rust")
                        .with_metadata("year", 2020),
                ],
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.9, 0.1, 0.0],
                ],
            )
            .await
            .unwrap();

        // AND combination: lang=rust AND year>=2021 → only rust doc.
        let and = MetadataFilter::and(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "rust"),
            MetadataFilter::field("year", FilterOp::Gte, 2021),
        ]);
        let results = store
            .similarity_search_with_filter(&[1.0, 0.0, 0.0], 5, Some(&and))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].document.content.contains("rust doc"));

        // Nin + missing-field semantics: lang NOT IN (python) → the two rust docs plus any with the field missing.
        let nin = MetadataFilter::field("lang", FilterOp::Nin, vec!["python"]);
        let results = store
            .similarity_search_with_filter(&[1.0, 0.0, 0.0], 5, Some(&nin))
            .await
            .unwrap();
        assert_eq!(results.len(), 2);

        // invalid key → UnsupportedFilter (not silently ignored).
        let bad = MetadataFilter::field("nonexistent key!", FilterOp::Eq, "x");
        let err = store
            .similarity_search_with_filter(&[1.0, 0.0, 0.0], 5, Some(&bad))
            .await
            .unwrap_err();
        assert!(matches!(err, VectorStoreError::UnsupportedFilter(_)));

        let _ = query(&format!("DROP TABLE IF EXISTS {table}"))
            .execute(&store.pool)
            .await;
    }

    /// Exact min_score semantics: threshold filtering happens before taking top-k.
    #[ignore = "requires a running PostgreSQL with the pgvector extension (CREATE EXTENSION vector); set PGVECTOR_TEST_URL"]
    #[tokio::test]
    async fn test_pgvector_min_score() {
        let table = format!("pgv_min_score_{}", Uuid::new_v4().simple());
        let store = PGVectorStore::connect(&test_url(), &table, 3)
            .await
            .unwrap();
        store.initialize().await.unwrap();

        store
            .add_documents(
                vec![Document::new("same"), Document::new("orthogonal")],
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            )
            .await
            .unwrap();

        let results = store
            .similarity_search_with_min_score(&[1.0, 0.0, 0.0], 5, Some(0.5))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].document.content.contains("same"));

        let _ = query(&format!("DROP TABLE IF EXISTS {table}"))
            .execute(&store.pool)
            .await;
    }
}
