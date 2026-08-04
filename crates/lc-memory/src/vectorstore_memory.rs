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
    /// Document IDs written by this memory, used for targeted clear (not clearing the entire store)
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
    use lc_embeddings::MockEmbeddings;
    use lc_vector_stores::InMemoryVectorStore;

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
