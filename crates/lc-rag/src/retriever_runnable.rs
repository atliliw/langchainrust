// lc-rag/src/retriever_runnable.rs
//! RetrieverRunnable — a Runnable adapter to bring any retriever into an LCEL chain
//!
//! Wraps `RetrieverTrait` (async `retrieve`) as a `Runnable<String, Vec<Document>>`:
//! input = query text, output = retrieved documents. Any retriever implementing
//! `RetrieverTrait` (SimilarityRetriever / BM25Retriever / UnifiedHybridIndex /
//! ParentDocumentRetriever …) can plug directly into a `RunnableSequence` via this
//! adapter, composing with prompt and LLM into a "retrieval → prompt → generation"
//! LCEL chain.

use async_trait::async_trait;
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use lc_vector_stores::Document;
use std::sync::Arc;

use crate::retriever::RetrieverTrait;

/// Runnable adapter: uses any retriever as a step in an LCEL chain.
///
/// `k` (number of documents returned) is fixed at construction; for a different count use
/// [`RetrieverRunnable::with_k`] to copy and adjust.
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
    /// Creates a retrieval Runnable that returns `k` documents.
    pub fn new(retriever: Arc<dyn RetrieverTrait>, k: usize) -> Self {
        Self { retriever, k }
    }

    /// Copies this adapter with a new `k` (returns a new instance, leaves the original untouched).
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

    /// E1 verification: the retrieval Runnable can participate in `pipe` composition as an LCEL chain step.
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
        // The next step is an identity transform: receives Vec<Document>, returns the count (proving the type chain holds).
        let count = step.pipe(lc_core::runnables::RunnableLambda::new_sync(
            |docs: Vec<Document>| docs.len(),
        ));
        let n = count.invoke("systems".to_string(), None).await.unwrap();
        assert_eq!(n, 1);
    }
}
