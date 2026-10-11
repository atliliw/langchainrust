// lc-memory/src/vectorstore_memory.rs
//! Vector retrieval memory
//!
//! Embeds each conversation turn and stores in a vector store, loading history by semantic relevance to the current input.
//! Suitable for long conversations and cross-session knowledge memory.
//!
//! Retrieval is vector-only by default. [`HybridRetrieval`] (T2) adds a full-text
//! channel: over-fetch a wider candidate pool by similarity, then re-rank it by
//! fusing the vector score with a term-frequency score, so exact literals the
//! embedding blurred (error codes, ids, symbol names) can still surface.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use super::base::{BaseMemory, MemoryError};
use super::semantic::LexicalScorer;
use lc_embeddings::Embeddings;
use lc_vector_stores::{Document, VectorStore};

/// Hybrid (vector + full-text) retrieval settings (T2).
///
/// The vector channel alone misses exact literals — an error code, an id, a symbol —
/// because embeddings average them away. Hybrid retrieval over-fetches a wider
/// candidate pool from the vector channel and re-ranks it by fusing the vector score
/// with a full-text (term-frequency) score:
///
/// `fused = (1 − lexical_weight) · vector + lexical_weight · full_text`
///
/// The full-text channel can only reorder candidates the vector channel returned, so
/// `over_fetch` is what buys it room to promote a document that ranked just outside
/// the top-k by similarity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridRetrieval {
    /// Weight of the full-text channel in `[0, 1]`; the vector channel gets the rest.
    pub lexical_weight: f64,
    /// Candidate-pool multiplier: fetch `k * over_fetch` documents before re-ranking.
    pub over_fetch: usize,
}

impl HybridRetrieval {
    /// Enables hybrid retrieval with the given full-text weight (clamped to `[0, 1]`)
    /// and the default 3× over-fetch.
    pub fn new(lexical_weight: f64) -> Self {
        Self {
            lexical_weight: lexical_weight.clamp(0.0, 1.0),
            over_fetch: 3,
        }
    }

    /// Sets the candidate-pool multiplier (clamped to at least 1).
    pub fn with_over_fetch(mut self, over_fetch: usize) -> Self {
        self.over_fetch = over_fetch.max(1);
        self
    }
}

/// Whether a document was written by this memory (`type = memory`).
///
/// Shared vector stores mix in other applications' documents; recall must not pull
/// those into the conversation history.
fn is_memory_document(document: &Document) -> bool {
    document
        .metadata
        .get("type")
        .map(|t| t == "memory")
        .unwrap_or(false)
}

/// Vector retrieval memory
///
/// Combines `VectorStore` + `Embeddings`: `save_context` embeds each conversation turn and stores in the vector store,
/// `load_memory_variables` retrieves top-k relevant history using the current input.
///
/// With [`with_hybrid_retrieval`](Self::with_hybrid_retrieval) the retrieval becomes
/// hybrid: over-fetch by vector, then re-rank by fusing vector similarity with a
/// full-text score (T2).
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
    /// Optional hybrid (vector + full-text) re-ranking of the vector candidates.
    hybrid: Option<HybridRetrieval>,
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
            hybrid: None,
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

    /// T2: enables hybrid retrieval — over-fetch vector candidates, then re-rank them
    /// by fusing the vector score with a full-text score.
    pub fn with_hybrid_retrieval(mut self, hybrid: HybridRetrieval) -> Self {
        self.hybrid = Some(hybrid);
        self
    }

    /// T2: convenience for [`with_hybrid_retrieval`](Self::with_hybrid_retrieval) with
    /// the default over-fetch.
    pub fn with_lexical_weight(self, lexical_weight: f64) -> Self {
        self.with_hybrid_retrieval(HybridRetrieval::new(lexical_weight))
    }

    /// T2: the configured hybrid settings, if any.
    pub fn hybrid_retrieval(&self) -> Option<HybridRetrieval> {
        self.hybrid
    }

    /// Fuses the vector and full-text scores of one candidate.
    ///
    /// Vector scores are cosine similarities in `[-1, 1]`; negatives carry no useful
    /// ranking signal for a weighted sum, so they are clamped to `0` before fusing.
    fn fuse(&self, lexical_weight: f64, vector_score: f32, text: &str, query: &str) -> f64 {
        let vector = (vector_score as f64).clamp(0.0, 1.0);
        let lexical = LexicalScorer::score(query, text);
        (lexical_weight * lexical + (1.0 - lexical_weight) * vector).clamp(0.0, 1.0)
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

        // T2: hybrid retrieval needs a wider candidate pool than it returns, so the
        // full-text channel has something to promote into the top-k.
        let fetch = match self.hybrid {
            Some(hybrid) => self.k.saturating_mul(hybrid.over_fetch),
            None => self.k,
        };

        let results = self
            .store
            .similarity_search(&q_emb, fetch)
            .await
            .map_err(|e| MemoryError::LoadError(e.to_string()))?;

        // P1-5: `type` 元数据真正消费——只召回本记忆写入的文档(type=memory),
        // 共享向量库时避免把其他业务文档混进会话历史。
        // `VectorStore` trait 暂不支持 metadata 过滤,采用召回后过滤(结果可能少于 k)。
        let candidates: Vec<(f32, String)> = results
            .iter()
            .filter(|r| is_memory_document(&r.document))
            .map(|r| (r.score, r.document.content.clone()))
            .collect();

        let history = match self.hybrid {
            None => candidates
                .into_iter()
                .map(|(_, text)| text)
                .collect::<Vec<_>>()
                .join("\n\n"),
            Some(hybrid) => {
                // Re-rank: fused score first, then the raw vector score, then the text
                // — a total order, so equal scores never reorder between runs.
                let mut reranked: Vec<(f64, f32, String)> = candidates
                    .into_iter()
                    .map(|(score, text)| {
                        let fused = self.fuse(hybrid.lexical_weight, score, &text, &query);
                        (fused, score, text)
                    })
                    .collect();
                reranked.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
                        .then_with(|| a.2.cmp(&b.2))
                });
                reranked.truncate(self.k);
                reranked
                    .into_iter()
                    .map(|(_, _, text)| text)
                    .collect::<Vec<_>>()
                    .join("\n\n")
            }
        };

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
        // M-28: don't `take(&mut self.owned_ids)` up front and drop the whole vec on the first
        // failure — a mid-clear delete error would then orphan the remaining documents *and*
        // lose their ids from bookkeeping, so a retry (after fixing the backing store) could no
        // longer find them. Retire an id only once its delete succeeds; on the first failure push
        // the failed id and every not-yet-attempted id back so `clear()` stays retriable.
        let ids = std::mem::take(&mut self.owned_ids);
        let mut iter = ids.into_iter();
        while let Some(id) = iter.next() {
            if let Err(e) = self.store.delete_document(&id).await {
                // clone the failed id into the restore set so the original is still alive for the
                // error message below; everything not-yet-attempted rides along after it.
                let mut restore = Vec::with_capacity(iter.len() + 1);
                restore.push(id.clone());
                restore.extend(iter);
                self.owned_ids = restore;
                return Err(MemoryError::ClearError(format!(
                    "Failed to delete document '{}': {}",
                    id, e
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_embeddings::MockEmbeddings;
    use lc_vector_stores::{Document, InMemoryVectorStore, SearchResult, VectorStoreError};
    use std::sync::atomic::{AtomicBool, Ordering};

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

    // ---- T2: hybrid (vector + full-text) retrieval ----

    /// Store returning a fixed candidate list in the given order, so the vector
    /// channel's scores are fully controlled by the test.
    struct ScriptedStore(Vec<SearchResult>);

    impl ScriptedStore {
        /// `(vector_score, content)` pairs, most similar first.
        fn new(entries: &[(f32, &str)]) -> Self {
            Self(
                entries
                    .iter()
                    .map(|(score, content)| SearchResult {
                        document: Document::new(*content).with_metadata("type", "memory"),
                        score: *score,
                    })
                    .collect(),
            )
        }
    }

    #[async_trait]
    impl VectorStore for ScriptedStore {
        async fn add_documents(
            &self,
            _documents: Vec<Document>,
            _embeddings: Vec<Vec<f32>>,
        ) -> Result<Vec<String>, VectorStoreError> {
            Ok(Vec::new())
        }
        async fn similarity_search(
            &self,
            _query_embedding: &[f32],
            k: usize,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            Ok(self.0.iter().take(k).cloned().collect())
        }
        async fn get_document(&self, _id: &str) -> Result<Option<Document>, VectorStoreError> {
            Ok(None)
        }
        async fn get_embedding(&self, _id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
            Ok(None)
        }
        async fn delete_document(&self, _id: &str) -> Result<(), VectorStoreError> {
            Ok(())
        }
        async fn count(&self) -> usize {
            self.0.len()
        }
        async fn clear(&self) -> Result<(), VectorStoreError> {
            Ok(())
        }
    }

    fn scripted_memory(
        entries: &[(f32, &str)],
        k: usize,
    ) -> VectorStoreRetrieverMemory<ScriptedStore, MockEmbeddings> {
        VectorStoreRetrieverMemory::new(ScriptedStore::new(entries), MockEmbeddings::new(32), k)
    }

    fn segments(vars: &HashMap<String, Value>) -> Vec<String> {
        vars.get("history")
            .unwrap()
            .as_str()
            .unwrap()
            .split("\n\n")
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// The point of hybrid retrieval: the full-text channel promotes an exact-literal
    /// match the vector channel ranked below a topically-similar but literal-free one.
    #[tokio::test]
    async fn hybrid_retrieval_promotes_full_text_match() {
        let entries = [
            (0.9, "we discussed the weather and the weekend plans"),
            (0.4, "the error code E4021 means the quota was exceeded"),
        ];

        // Pure vector: the high-similarity, literal-free turn wins.
        let plain = scripted_memory(&entries, 1);
        let plain_vars = plain
            .load_memory_variables(&inputs("error code E4021"))
            .await
            .unwrap();
        assert!(segments(&plain_vars)[0].contains("weather"));

        // Hybrid with a full-text-heavy weight: the literal match wins.
        let hybrid = scripted_memory(&entries, 1)
            .with_hybrid_retrieval(HybridRetrieval::new(0.8).with_over_fetch(3));
        let hybrid_vars = hybrid
            .load_memory_variables(&inputs("error code E4021"))
            .await
            .unwrap();
        assert_eq!(segments(&hybrid_vars), vec![entries[1].1.to_string()]);
    }

    /// `lexical_weight = 0.0` must degrade exactly to the vector ordering.
    #[tokio::test]
    async fn hybrid_retrieval_with_zero_lexical_weight_is_vector_order() {
        let entries = [
            (0.9, "we discussed the weather and the weekend plans"),
            (0.4, "the error code E4021 means the quota was exceeded"),
        ];
        let mem = scripted_memory(&entries, 2).with_lexical_weight(0.0);
        let vars = mem
            .load_memory_variables(&inputs("error code E4021"))
            .await
            .unwrap();
        let got = segments(&vars);
        assert_eq!(
            got,
            vec![entries[0].1.to_string(), entries[1].1.to_string()]
        );
    }

    /// Over-fetch is what lets the full-text channel reach a document the vector
    /// channel placed outside the top-k.
    #[tokio::test]
    async fn hybrid_retrieval_over_fetch_reaches_beyond_top_k() {
        let entries = [
            (0.95, "alpha unrelated chatter one"),
            (0.90, "beta unrelated chatter two"),
            (0.30, "the migration id ZQ-7781 is the one to use"),
        ];
        // k = 1, over-fetch 3 → the third candidate is fetched and can be promoted.
        let mem = scripted_memory(&entries, 1).with_hybrid_retrieval(HybridRetrieval::new(0.9));
        let vars = mem
            .load_memory_variables(&inputs("migration id ZQ-7781"))
            .await
            .unwrap();
        assert_eq!(segments(&vars), vec![entries[2].1.to_string()]);
    }

    /// The over-fetch pool must never grow the *result* past `k`.
    #[tokio::test]
    async fn hybrid_retrieval_still_caps_results_at_k() {
        let entries = [
            (0.9, "one memory about alpha"),
            (0.8, "two memory about beta"),
            (0.7, "three memory about gamma"),
            (0.6, "four memory about delta"),
        ];
        let mem = scripted_memory(&entries, 2).with_lexical_weight(0.5);
        let vars = mem
            .load_memory_variables(&inputs("memory about"))
            .await
            .unwrap();
        assert_eq!(segments(&vars).len(), 2);
    }

    /// `type = memory` filtering happens *before* re-ranking, so a foreign document
    /// with a perfect full-text match can never enter the history.
    #[tokio::test]
    async fn hybrid_retrieval_filters_foreign_documents_before_reranking() {
        let store = ScriptedStore(vec![
            SearchResult {
                document: Document::new("the error code E4021 means the quota was exceeded")
                    .with_metadata("type", "kb"),
                score: 0.95,
            },
            SearchResult {
                document: Document::new("an unrelated memory turn").with_metadata("type", "memory"),
                score: 0.2,
            },
        ]);
        let mem = VectorStoreRetrieverMemory::new(store, MockEmbeddings::new(32), 1)
            .with_lexical_weight(1.0);
        let vars = mem
            .load_memory_variables(&inputs("error code E4021"))
            .await
            .unwrap();
        assert_eq!(
            segments(&vars),
            vec!["an unrelated memory turn".to_string()]
        );
    }

    #[test]
    fn hybrid_settings_are_clamped() {
        assert_eq!(HybridRetrieval::new(2.0).lexical_weight, 1.0);
        assert_eq!(HybridRetrieval::new(-1.0).lexical_weight, 0.0);
        assert_eq!(HybridRetrieval::new(0.5).over_fetch, 3);
        assert_eq!(HybridRetrieval::new(0.5).with_over_fetch(0).over_fetch, 1);

        let mem = make_memory(2).with_lexical_weight(0.25);
        assert_eq!(
            mem.hybrid_retrieval(),
            Some(HybridRetrieval::new(0.25)),
            "with_lexical_weight must enable hybrid retrieval"
        );
        assert_eq!(make_memory(2).hybrid_retrieval(), None);
    }

    /// VectorStore that fails exactly the first `delete_document` call, succeeding afterwards,
    /// to exercise M-28's mid-clear partial-failure retry. All other ops delegate to an inner
    /// `InMemoryVectorStore`.
    struct FailsOnceDelete(InMemoryVectorStore, AtomicBool);

    #[async_trait]
    impl VectorStore for FailsOnceDelete {
        async fn add_documents(
            &self,
            documents: Vec<Document>,
            embeddings: Vec<Vec<f32>>,
        ) -> Result<Vec<String>, VectorStoreError> {
            self.0.add_documents(documents, embeddings).await
        }
        async fn similarity_search(
            &self,
            query_embedding: &[f32],
            k: usize,
        ) -> Result<Vec<SearchResult>, VectorStoreError> {
            self.0.similarity_search(query_embedding, k).await
        }
        async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
            self.0.get_document(id).await
        }
        async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
            self.0.get_embedding(id).await
        }
        async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
            if self.1.swap(false, Ordering::SeqCst) {
                return Err(VectorStoreError::StorageError(
                    "simulated transient delete failure".to_string(),
                ));
            }
            self.0.delete_document(id).await
        }
        async fn count(&self) -> usize {
            self.0.count().await
        }
        async fn clear(&self) -> Result<(), VectorStoreError> {
            self.0.clear().await
        }
    }

    #[tokio::test]
    async fn test_clear_partial_failure_retriable() {
        // M-28: a mid-clear delete failure must not retire the remaining owned ids, so a
        // subsequent clear() can finish deleting them instead of orphaning the docs.
        let store = FailsOnceDelete(InMemoryVectorStore::new(), AtomicBool::new(true));
        let mut mem = VectorStoreRetrieverMemory::new(store, MockEmbeddings::new(32), 3);
        mem.save_context(&inputs("apple"), &outputs("a"))
            .await
            .unwrap();
        mem.save_context(&inputs("banana"), &outputs("b"))
            .await
            .unwrap();

        // first clear hits the simulated failure -> the failed id + the not-yet-tried ids stay tracked
        assert!(mem.clear().await.is_err());
        // second clear succeeds and actually deletes everything (proves nothing got orphaned)
        mem.clear().await.unwrap();

        let vars = mem.load_memory_variables(&inputs("apple")).await.unwrap();
        assert!(vars.get("history").unwrap().as_str().unwrap().is_empty());
    }
}
