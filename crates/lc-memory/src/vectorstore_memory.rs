// lc-memory/src/vectorstore_memory.rs
//! Vector retrieval memory
//!
//! Embeds each conversation turn and stores in a vector store, loading history by semantic relevance to the current input.
//! Suitable for long conversations and cross-session knowledge memory.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use super::base::{BaseMemory, MemoryError};
use lc_embeddings::Embeddings;
use lc_vector_stores::{Document, VectorStore};

/// Vector retrieval memory
///
/// Combines `VectorStore` + `Embeddings`: `save_context` embeds each conversation turn and stores in the vector store,
/// `load_memory_variables` retrieves top-k relevant history using the current input.
///
/// # Example
/// ```ignore
/// use lc_memory::VectorStoreRetrieverMemory;
/// use lc_vector_stores::InMemoryVectorStore;
/// use lc_embeddings::MockEmbeddings;
///
/// let mut memory = VectorStoreRetrieverMemory::new(
///     InMemoryVectorStore::new(),
///     MockEmbeddings::new(1536),
///     4,
/// );
/// ```
pub struct VectorStoreRetrieverMemory<V, E> {
    store: V,
    embeddings: E,
    /// Number of history entries to retrieve
    k: usize,
    input_key: String,
    output_key: String,
    memory_key: String,
    /// 存文档向量时用 `embed_documents`(true)还是 `embed_query`(false)。
    /// 对严格区分 query/doc 的 provider(OpenAI `input_type`、Cohere `search_document`),
    /// `true` 口径正确;provider 不区分时两者等效。
    document_api: bool,
    /// Document IDs written by this memory, used for targeted clear (not clearing the entire store).
    ///
    /// P1-5 注意:这是**进程内语义**——重启后 `owned_ids` 为空,`clear()` 退化为
    /// 空操作,孤儿文档留在向量库。需跨重启清理时请直接操作向量库。
    owned_ids: Vec<String>,
}

impl<V, E> VectorStoreRetrieverMemory<V, E> {
    /// Create vector retrieval memory
    ///
    /// # Arguments
    /// * `store` - Vector store
    /// * `embeddings` - Embedding model
    /// * `k` - Number of history entries to retrieve
    pub fn new(store: V, embeddings: E, k: usize) -> Self {
        Self {
            store,
            embeddings,
            k,
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            document_api: true,
            owned_ids: Vec::new(),
        }
    }

    /// Set input key name
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Set output key name
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// Set memory variable name
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// P1-5: 设置存文档向量时使用的嵌入接口。
    ///
    /// 默认 `true`(走 `embed_documents`,query/doc 区分 provider 下口径正确);
    /// 若 provider 不区分,两者等效,可设 `false` 走 `embed_query` 快路径。
    pub fn with_document_api(mut self, use_document_api: bool) -> Self {
        self.document_api = use_document_api;
        self
    }
}

#[async_trait]
impl<V, E> BaseMemory for VectorStoreRetrieverMemory<V, E>
where
    V: VectorStore,
    E: Embeddings,
{
    fn memory_variables(&self) -> Vec<&str> {
        vec![&self.memory_key]
    }

    async fn load_memory_variables(
        &self,
        inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, Value>, MemoryError> {
        let mut result = HashMap::new();
        // Use current input as retrieval query
        let query = inputs.get(&self.input_key).cloned().unwrap_or_default();
        if query.trim().is_empty() {
            result.insert(self.memory_key.clone(), Value::String(String::new()));
            return Ok(result);
        }

        let q_emb = self
            .embeddings
            .embed_query(&query)
            .await
            .map_err(|e| MemoryError::LoadError(e.to_string()))?;

        let results = self
            .store
            .similarity_search(&q_emb, self.k)
            .await
            .map_err(|e| MemoryError::LoadError(e.to_string()))?;

        // P1-5: `type` 元数据真正消费——只召回本记忆写入的文档(type=memory),
        // 共享向量库时避免把其他业务文档混进会话历史。
        // `VectorStore` trait 暂不支持 metadata 过滤,采用召回后过滤(结果可能少于 k)。
        let history = results
            .iter()
            .filter(|r| {
                r.document
                    .metadata
                    .get("type")
                    .map(|t| t == "memory")
                    .unwrap_or(false)
            })
            .map(|r| r.document.content.clone())
            .collect::<Vec<_>>()
            .join("\n\n");

        result.insert(self.memory_key.clone(), Value::String(history));
        Ok(result)
    }

    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        let input = inputs.get(&self.input_key);
        let output = outputs.get(&self.output_key);

        let text = match (input, output) {
            (Some(i), Some(o)) => format!("Human: {}\nAI: {}", i, o),
            (Some(i), None) => format!("Human: {}", i),
            (None, Some(o)) => format!("AI: {}", o),
            (None, None) => return Ok(()),
        };

        if text.trim().is_empty() {
            return Ok(());
        }

        // P1-5: 存的是"Human+AI 整段历史"(文档),用 embed_documents 走 provider 的
        // 文档口径;若 provider 不区分,等效 embed_query。
        let emb = if self.document_api {
            self.embeddings
                .embed_documents(&[text.as_str()])
                .await
                .map_err(|e| MemoryError::SaveError(e.to_string()))?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    MemoryError::SaveError("embed_documents returned no vectors".to_string())
                })?
        } else {
            self.embeddings
                .embed_query(&text)
                .await
                .map_err(|e| MemoryError::SaveError(e.to_string()))?
        };

        let doc = Document::new(text).with_metadata("type", "memory");

        let ids = self
            .store
            .add_documents(vec![doc], vec![emb])
            .await
            .map_err(|e| MemoryError::SaveError(e.to_string()))?;

        self.owned_ids.extend(ids);
        Ok(())
    }

    /// 清空本记忆写入的文档。
    ///
    /// P1-5 进程内语义:删除只覆盖本进程累积的 `owned_ids`;重启后 `owned_ids`
    /// 为空,`clear()` 为空操作,需直接操作向量库清理孤儿文档。
    async fn clear(&mut self) -> Result<(), MemoryError> {
        // M68: Propagate delete errors instead of silently ignoring them
        for id in std::mem::take(&mut self.owned_ids) {
            self.store.delete_document(&id).await.map_err(|e| {
                MemoryError::ClearError(format!("Failed to delete document '{}': {}", id, e))
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_embeddings::MockEmbeddings;
    use lc_vector_stores::{Document, InMemoryVectorStore};

    fn make_memory(k: usize) -> VectorStoreRetrieverMemory<InMemoryVectorStore, MockEmbeddings> {
        VectorStoreRetrieverMemory::new(InMemoryVectorStore::new(), MockEmbeddings::new(32), k)
    }

    fn inputs(s: &str) -> HashMap<String, String> {
        HashMap::from([("input".to_string(), s.to_string())])
    }

    fn outputs(s: &str) -> HashMap<String, String> {
        HashMap::from([("output".to_string(), s.to_string())])
    }

    #[tokio::test]
    async fn test_save_and_load_roundtrip() {
        let mut mem = make_memory(3);
        mem.save_context(&inputs("apple"), &outputs("fruit"))
            .await
            .unwrap();

        // Load should succeed without error (mock embeddings may not produce positive similarity)
        let vars = mem.load_memory_variables(&inputs("apple")).await.unwrap();
        let _history = vars.get("history").unwrap().as_str().unwrap();
    }

    #[tokio::test]
    async fn test_top_k_limit() {
        let mut mem = make_memory(2);
        mem.save_context(&inputs("apple"), &outputs("a"))
            .await
            .unwrap();
        mem.save_context(&inputs("banana"), &outputs("b"))
            .await
            .unwrap();
        mem.save_context(&inputs("cherry"), &outputs("c"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables(&inputs("apple")).await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        let segments = history.split("\n\n").filter(|s| !s.is_empty()).count();
        assert!(
            segments <= 2,
            "k=2 should retrieve at most 2 segments, got {}",
            segments
        );
    }

    #[tokio::test]
    async fn test_all_retrievable_when_k_large() {
        let mut mem = make_memory(5);
        mem.save_context(&inputs("apple"), &outputs("a"))
            .await
            .unwrap();
        mem.save_context(&inputs("banana"), &outputs("b"))
            .await
            .unwrap();
        mem.save_context(&inputs("cherry"), &outputs("c"))
            .await
            .unwrap();

        // k=5 >= document count, should at least retrieve query-relevant documents
        let vars = mem.load_memory_variables(&inputs("apple")).await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(!history.is_empty(), "history should not be empty");
    }

    #[tokio::test]
    async fn test_clear() {
        let mut mem = make_memory(3);
        mem.save_context(&inputs("apple"), &outputs("a"))
            .await
            .unwrap();
        mem.clear().await.unwrap();

        let vars = mem.load_memory_variables(&inputs("apple")).await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_empty_query_returns_empty() {
        let mut mem = make_memory(3);
        mem.save_context(&inputs("apple"), &outputs("a"))
            .await
            .unwrap();

        // Empty query should not error, return empty history
        let empty_inputs = HashMap::new();
        let vars = mem.load_memory_variables(&empty_inputs).await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_memory_variables() {
        let mem = make_memory(3);
        assert_eq!(mem.memory_variables(), vec!["history"]);
    }

    #[tokio::test]
    async fn test_custom_memory_key() {
        let mem = make_memory(3).with_memory_key("chat_history");
        assert_eq!(mem.memory_variables(), vec!["chat_history"]);
    }

    #[tokio::test]
    async fn test_recall_filters_non_memory_documents() {
        // P1-5: 共享向量库混入非 memory 文档时,召回只返回 type=memory 的文档。
        let mut mem = make_memory(3);
        mem.save_context(&inputs("apple"), &outputs("fruit"))
            .await
            .unwrap();

        // 直接往共享库写入一条非 memory 文档(模拟其他业务文档)。
        let foreign_emb = mem
            .embeddings
            .embed_query("unrelated business document")
            .await
            .unwrap();
        mem.store
            .add_documents(
                vec![Document::new("unrelated business document").with_metadata("type", "kb")],
                vec![foreign_emb],
            )
            .await
            .unwrap();

        let vars = mem.load_memory_variables(&inputs("apple")).await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(
            !history.contains("unrelated business document"),
            "recall should filter non-memory documents, got: {}",
            history
        );
    }

    #[tokio::test]
    async fn test_document_api_false_uses_embed_query() {
        // 关闭文档口径后仍能正常写入/召回(provider 不区分时等效)。
        let mut mem = make_memory(3).with_document_api(false);
        mem.save_context(&inputs("apple"), &outputs("fruit"))
            .await
            .unwrap();
        mem.clear().await.unwrap();
    }

    #[tokio::test]
    async fn test_skips_when_nothing_to_save() {
        let mut mem = make_memory(3);
        // Neither input nor output, should return Ok without writing
        mem.save_context(&HashMap::new(), &HashMap::new())
            .await
            .unwrap();

        let vars = mem.load_memory_variables(&inputs("apple")).await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.is_empty());
    }
}
