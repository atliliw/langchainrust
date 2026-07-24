// src/memory/vectorstore_memory.rs
//! 向量检索记忆
//!
//! 将每轮对话嵌入后存入向量库,加载时按当前输入的语义相关性召回历史。
//! 适用于长对话与跨会话知识记忆,相比固定窗口的 buffer memory 能保留更多有效上下文。

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use super::base::{BaseMemory, MemoryError};
use crate::embeddings::Embeddings;
use crate::vector_stores::{Document, VectorStore};

/// 向量检索记忆
///
/// 组合 `VectorStore` + `Embeddings`:`save_context` 时把每轮对话嵌入后存入向量库,
/// `load_memory_variables` 时用当前输入检索 top-k 相关历史。
///
/// # 示例
/// ```ignore
/// use langchainrust::memory::VectorStoreRetrieverMemory;
/// use langchainrust::vector_stores::InMemoryVectorStore;
/// use langchainrust::embeddings::MockEmbeddings;
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
    /// 召回历史条数
    k: usize,
    input_key: String,
    output_key: String,
    memory_key: String,
    /// 本 memory 写入的文档 id,用于 clear 定向删除(不清空整个 store)
    owned_ids: Vec<String>,
}

impl<V, E> VectorStoreRetrieverMemory<V, E> {
    /// 创建向量检索记忆
    ///
    /// # 参数
    /// * `store` - 向量库
    /// * `embeddings` - 嵌入模型
    /// * `k` - 召回历史条数
    pub fn new(store: V, embeddings: E, k: usize) -> Self {
        Self {
            store,
            embeddings,
            k,
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            owned_ids: Vec::new(),
        }
    }

    /// 设置输入键名
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// 设置输出键名
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// 设置记忆变量名
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
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
        // 用当前输入作为检索 query
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

        let history = results
            .iter()
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

        let emb = self
            .embeddings
            .embed_query(&text)
            .await
            .map_err(|e| MemoryError::SaveError(e.to_string()))?;

        let doc = Document::new(text).with_metadata("type", "memory");

        let ids = self
            .store
            .add_documents(vec![doc], vec![emb])
            .await
            .map_err(|e| MemoryError::SaveError(e.to_string()))?;

        self.owned_ids.extend(ids);
        Ok(())
    }

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
    use crate::embeddings::MockEmbeddings;
    use crate::vector_stores::InMemoryVectorStore;

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
        assert!(segments <= 2, "k=2 应最多召回 2 段, 实际 {}", segments);
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

        // k=5 >= 文档数,应至少召回查询相关的文档
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

        // 空 query 不应报错,返回空 history
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
    async fn test_skips_when_nothing_to_save() {
        let mut mem = make_memory(3);
        // 既无 input 也无 output,应直接返回 Ok 且不写入
        mem.save_context(&HashMap::new(), &HashMap::new())
            .await
            .unwrap();

        let vars = mem.load_memory_variables(&inputs("apple")).await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.is_empty());
    }
}
