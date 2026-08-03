//! PGVector vector store (PostgreSQL + pgvector extension)
//!
//! This module provides helper utilities for PGVector. The full `PGVectorStore`
//! implementation requires `sqlx` and `pgvector` crates, which must be added
//! by the user to their own `Cargo.toml` due to potential conflicts with
//! `rusqlite` (libsqlite3-sys linkage).
//!
//! To use `PGVectorStore`, add these to your `Cargo.toml`:
//! ```toml
//! sqlx = { version = "0.7", features = ["runtime-tokio", "postgres"] }
//! pgvector = { version = "0.3", features = ["sqlx"] }
//! ```
//!
//! Then enable the `pgvector-storage` feature and include the implementation
//! from the project's `src/vector_stores/pgvector.rs`.

use std::sync::LazyLock;

use regex::Regex;

static TABLE_NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*$").unwrap());

/// Validate that a table name is safe for SQL interpolation.
///
/// Only allows: `^[a-zA-Z_][a-zA-Z0-9_]*$`
/// This prevents SQL injection via table names.
pub fn validate_table_name(table: &str) -> Result<(), String> {
    if TABLE_NAME_RE.is_match(table) {
        Ok(())
    } else {
        Err(format!(
            "Invalid table name '{}': must match ^[a-zA-Z_][a-zA-Z0-9_]*$",
            table
        ))
    }
}

/// Build CREATE TABLE SQL (pure function, convenient for testing)
pub fn build_table_sql(table: &str, dim: usize) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {} (id TEXT PRIMARY KEY, content TEXT, metadata JSONB, embedding vector({}))",
        table, dim
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
