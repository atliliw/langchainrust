// lc-rag/src/retriever_runnable.rs
//! RetrieverRunnable — 任意检索器进 LCEL 的 Runnable 适配器
//!
//! 把 `RetrieverTrait`(异步 `retrieve`)包成 `Runnable<String, Vec<Document>>`:
//! 输入=查询文本,输出=检索到的文档。任意实现 `RetrieverTrait` 的检索器
//! (SimilarityRetriever / BM25Retriever / UnifiedHybridIndex /
//! ParentDocumentRetriever …)都能借此直接进 `RunnableSequence`,和
//! prompt、LLM 组合成"检索 → 提示词 → 生成"的 LCEL 链。

use async_trait::async_trait;
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use lc_vector_stores::Document;
use std::sync::Arc;

use crate::retriever::RetrieverTrait;

/// Runnable 适配器:把任意检索器作为 LCEL 链的一步。
///
/// `k`(返回文档条数)在构造时固定;需要不同条数时用 [`RetrieverRunnable::with_k`]
/// 复制调整。
///
/// # Example
///
/// ```rust,ignore
/// let retriever = Arc::new(SimilarityRetriever::new(store, embeddings));
/// let step = RetrieverRunnable::new(retriever, 4);
/// let chain = step.pipe(prompt).pipe(llm);
/// ```
pub struct RetrieverRunnable {
    retriever: Arc<dyn RetrieverTrait>,
    k: usize,
}

impl RetrieverRunnable {
    /// 创建检索 Runnable,固定返回 `k` 条文档。
    pub fn new(retriever: Arc<dyn RetrieverTrait>, k: usize) -> Self {
        Self { retriever, k }
    }

    /// 以新的 `k` 复制本适配器(返回新实例,不改动原对象)。
    pub fn with_k(&self, k: usize) -> Self {
        Self {
            retriever: self.retriever.clone(),
            k,
        }
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for RetrieverRunnable {
    type Error = LcelError;

    async fn invoke(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<Vec<Document>, LcelError> {
        self.retriever
            .retrieve(&input, self.k)
            .await
            .map_err(|e| LcelError::Other(format!("retriever error: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retriever::SimilarityRetriever;
    use lc_embeddings::MockEmbeddings;
    use lc_vector_stores::InMemoryVectorStore;

    fn test_retriever() -> Arc<dyn RetrieverTrait> {
        Arc::new(SimilarityRetriever::new(
            Arc::new(InMemoryVectorStore::new()),
            Arc::new(MockEmbeddings::new(64)),
        ))
    }

    #[tokio::test]
    async fn retriever_runnable_invokes_retrieve() {
        let retriever = test_retriever();
        retriever
            .add_documents(vec![Document::new(
                "Rust is a systems programming language",
            )])
            .await
            .unwrap();

        let step = RetrieverRunnable::new(retriever, 1);
        let docs = step.invoke("systems".to_string(), None).await.unwrap();
        assert!(!docs.is_empty(), "expected at least one document");
        assert!(docs[0].content.contains("Rust"));
    }

    #[tokio::test]
    async fn retriever_runnable_with_k_copies() {
        let retriever = test_retriever();
        let step = RetrieverRunnable::new(retriever, 4);
        let narrowed = step.with_k(2);
        assert_eq!(narrowed.k, 2);
        assert_eq!(step.k, 4, "with_k must not mutate the original");
    }

    /// E1 验证:检索 Runnable 能作为 LCEL 链的一步参与 `pipe` 组合。
    #[tokio::test]
    async fn retriever_runnable_pipes_into_sequence() {
        use lc_core::runnables::RunnableExt;

        let retriever = test_retriever();
        retriever
            .add_documents(vec![Document::new(
                "Rust is a systems programming language",
            )])
            .await
            .unwrap();

        let step = RetrieverRunnable::new(retriever, 1);
        // 下一段是恒等变换:接收 Vec<Document>,返回数量(证明类型链通)。
        let count = step.pipe(lc_core::runnables::RunnableLambda::new_sync(
            |docs: Vec<Document>| docs.len(),
        ));
        let n = count.invoke("systems".to_string(), None).await.unwrap();
        assert_eq!(n, 1);
    }
}
