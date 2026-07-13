//! Pinecone 向量库(HTTP API)

use std::collections::HashMap;

use serde::Deserialize;

use crate::embeddings::Embeddings;
use crate::vector_stores::Document;

/// Pinecone 向量库客户端
pub struct PineconeStore {
    api_key: String,
    host: String,
    client: reqwest::Client,
}

impl PineconeStore {
    /// 创建 Pinecone 客户端
    ///
    /// `host` 格式: `https://{index-name}.svc.{environment}.pinecone.io`
    pub fn new(api_key: impl Into<String>, host: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            host: host.into(),
            client: reqwest::Client::new(),
        }
    }

    /// 构造 upsert 请求体(纯函数,便于测试)
    pub fn build_upsert_body(docs: &[Document], vectors: &[Vec<f32>]) -> serde_json::Value {
        let vectors_json: Vec<serde_json::Value> = docs
            .iter()
            .zip(vectors.iter())
            .map(|(doc, vec)| {
                serde_json::json!({
                    "id": doc.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    "values": vec,
                    "metadata": doc.metadata,
                })
            })
            .collect();
        serde_json::json!({ "vectors": vectors_json })
    }

    /// 构造 query 请求体(纯函数,便于测试)
    pub fn build_query_body(query_vec: &[f32], top_k: usize) -> serde_json::Value {
        serde_json::json!({
            "vector": query_vec,
            "topK": top_k,
            "includeMetadata": true,
        })
    }

    /// upsert 文档(自动 embed)
    pub async fn upsert(
        &self,
        docs: &[Document],
        embeddings: &dyn Embeddings,
    ) -> Result<(), String> {
        let texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
        let vectors = embeddings
            .embed_documents(&texts)
            .await
            .map_err(|e| e.to_string())?;
        let body = Self::build_upsert_body(docs, &vectors);
        let url = format!("{}/vectors/upsert", self.host);
        let resp = self
            .client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("Pinecone upsert 错误: {}", resp.status()));
        }
        Ok(())
    }

    /// 查询相似文档
    pub async fn query(&self, query_vec: Vec<f32>, top_k: usize) -> Result<Vec<Document>, String> {
        let body = Self::build_query_body(&query_vec, top_k);
        let url = format!("{}/query", self.host);
        let resp = self
            .client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("Pinecone query 错误: {}", resp.status()));
        }
        let query_resp: QueryResponse = resp.json().await.map_err(|e| e.to_string())?;
        let result = query_resp
            .matches
            .into_iter()
            .map(|m| {
                let content = m
                    .metadata
                    .as_ref()
                    .and_then(|md| md.get("content").cloned())
                    .unwrap_or_default();
                Document {
                    content,
                    metadata: m.metadata.unwrap_or_default(),
                    id: Some(m.id),
                }
            })
            .collect();
        Ok(result)
    }

    /// 按 ID 删除
    pub async fn delete(&self, ids: &[String]) -> Result<(), String> {
        let url = format!("{}/vectors/delete", self.host);
        let body = serde_json::json!({ "ids": ids });
        let resp = self
            .client
            .post(&url)
            .header("Api-Key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("Pinecone delete 错误: {}", resp.status()));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct QueryResponse {
    matches: Vec<QueryMatch>,
}

#[derive(Deserialize)]
struct QueryMatch {
    id: String,
    #[allow(dead_code)]
    score: f64,
    metadata: Option<HashMap<String, String>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_stores::Document;

    fn doc(id: &str, content: &str) -> Document {
        Document {
            content: content.to_string(),
            metadata: HashMap::new(),
            id: Some(id.to_string()),
        }
    }

    #[test]
    fn test_build_upsert_body() {
        let docs = vec![doc("1", "hello"), doc("2", "world")];
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let body = PineconeStore::build_upsert_body(&docs, &vectors);
        let vectors_arr = body.get("vectors").unwrap().as_array().unwrap();
        assert_eq!(vectors_arr.len(), 2);
        assert_eq!(vectors_arr[0]["id"], "1");
        assert_eq!(vectors_arr[0]["values"][0], 1.0);
    }

    #[test]
    fn test_build_upsert_body_generates_id_if_missing() {
        let mut d = doc("", "x");
        d.id = None;
        let body = PineconeStore::build_upsert_body(&[d], &[vec![0.1]]);
        let id = body["vectors"][0]["id"].as_str().unwrap();
        assert!(!id.is_empty());
    }

    #[test]
    fn test_build_query_body() {
        let body = PineconeStore::build_query_body(&[1.0, 2.0, 3.0], 5);
        assert_eq!(body["topK"], 5);
        assert_eq!(body["includeMetadata"], true);
        assert_eq!(body["vector"][2], 3.0);
    }

    #[test]
    fn test_new() {
        let store = PineconeStore::new("key", "https://index.svc.env.pinecone.io");
        assert_eq!(store.host, "https://index.svc.env.pinecone.io");
    }
}
