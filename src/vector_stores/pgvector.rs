//! PGVector 向量库(PostgreSQL + pgvector 扩展)

use std::collections::HashMap;

use pgvector::Vector;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

use crate::embeddings::Embeddings;
use crate::vector_stores::Document;

/// PGVector 向量库
pub struct PGVectorStore {
    pool: PgPool,
    table: String,
}

/// 构造建表 SQL(纯函数,便于测试)
pub fn build_table_sql(table: &str, dim: usize) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {} (id TEXT PRIMARY KEY, content TEXT, metadata JSONB, embedding vector({}))",
        table, dim
    )
}

impl PGVectorStore {
    pub async fn new(url: &str, table: &str, dim: usize) -> Result<Self, String> {
        let pool = PgPoolOptions::new()
            .connect(url)
            .await
            .map_err(|e| e.to_string())?;
        let store = Self {
            pool,
            table: table.to_string(),
        };
        store.ensure_table(dim).await?;
        Ok(store)
    }

    pub async fn ensure_table(&self, dim: usize) -> Result<(), String> {
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await
            .map_err(|e| e.to_string())?;
        sqlx::query(&build_table_sql(&self.table, dim))
            .execute(&self.pool)
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn add_documents(
        &self,
        docs: &[Document],
        embeddings: &dyn Embeddings,
    ) -> Result<(), String> {
        let texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
        let vectors = embeddings
            .embed_documents(&texts)
            .await
            .map_err(|e| e.to_string())?;
        for (doc, vec) in docs.iter().zip(vectors.into_iter()) {
            let id = doc
                .id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            let v = Vector::from(vec);
            let metadata_json =
                serde_json::to_string(&doc.metadata).unwrap_or_else(|_| "{}".to_string());
            let sql = format!(
                "INSERT INTO {} (id, content, metadata, embedding) VALUES ($1, $2, $3, $4) \
                 ON CONFLICT (id) DO UPDATE SET content = $2, metadata = $3, embedding = $4",
                self.table
            );
            sqlx::query(&sql)
                .bind(id)
                .bind(&doc.content)
                .bind(&metadata_json)
                .bind(v)
                .execute(&self.pool)
                .await
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub async fn similarity_search(
        &self,
        query: &str,
        k: usize,
        embeddings: &dyn Embeddings,
    ) -> Result<Vec<Document>, String> {
        let qvec = embeddings
            .embed_query(query)
            .await
            .map_err(|e| e.to_string())?;
        let v = Vector::from(qvec);
        let sql = format!(
            "SELECT id, content, metadata FROM {} ORDER BY embedding <-> $1 LIMIT $2",
            self.table
        );
        let rows = sqlx::query_as::<_, (Option<String>, String, Option<String>)>(&sql)
            .bind(v)
            .bind(k as i64)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| e.to_string())?;
        let result = rows
            .into_iter()
            .map(|(id, content, metadata_str)| {
                let metadata: HashMap<String, String> = metadata_str
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                Document {
                    content,
                    metadata,
                    id,
                }
            })
            .collect();
        Ok(result)
    }

    pub async fn delete(&self, id: &str) -> Result<(), String> {
        let sql = format!("DELETE FROM {} WHERE id = $1", self.table);
        sqlx::query(&sql)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }
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
}
