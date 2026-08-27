// lc-rag/src/parent_document.rs
//! ParentDocumentRetriever — a parent/child document retriever

use async_trait::async_trait;
use lc_vector_stores::{ChunkedDocumentStore, ChunkedDocumentStoreTrait, Document, SearchResult};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::bm25::{AutoMergingConfig, ChunkedBM25Retriever};
use crate::retriever::{RetrieverError, RetrieverTrait};

/// A parent/child document retriever.
///
/// On ingestion, documents are split into small chunks (leaves) for indexing; any leaf hit
/// returns the **entire parent document**. This differs from [`ChunkedBM25Retriever`]'s
/// AutoMerging (gated by hit ratio; returns leaf chunks when the threshold is not met):
/// ParentDocument is the classic RAG pattern of "small-chunk recall, full document fed to
/// the LLM" — the leaf chunks provide precise hits, while the whole parent document gives
/// the LLM complete context.
///
/// Internally it wraps a [`ChunkedBM25Retriever`] in an `RwLock`: retrieval takes the read
/// lock, ingestion takes the write lock, so it can implement [`RetrieverTrait`] (whose
/// methods all take `&self`). Combined with
/// [`RetrieverRunnable`](crate::RetrieverRunnable) it can plug directly into an LCEL chain:
///
/// ```rust,ignore
/// let retriever = Arc::new(ParentDocumentRetriever::new(store));
/// let chain = RetrieverRunnable::new(retriever, 4).pipe(prompt).pipe(llm);
/// ```
pub struct ParentDocumentRetriever<S: ChunkedDocumentStoreTrait = ChunkedDocumentStore> {
    inner: RwLock<ChunkedBM25Retriever<S>>,
}

impl<S: ChunkedDocumentStoreTrait> ParentDocumentRetriever<S> {
    /// Creates a retriever with the default configuration.
    pub fn new(store: Arc<S>) -> Self {
        Self {
            inner: RwLock::new(ChunkedBM25Retriever::new(store)),
        }
    }

    /// Creates a retriever with the specified AutoMerging configuration.
    ///
    /// `leaf_chunk_size` in the config determines the leaf-chunk granularity; under the
    /// ParentDocument semantics `merge_threshold` does not take part in hit decisions
    /// (any leaf hit returns the parent document), but it still constrains how the
    /// underlying index is chunked.
    pub fn with_config(store: Arc<S>, config: AutoMergingConfig) -> Self {
        Self {
            inner: RwLock::new(ChunkedBM25Retriever::with_config(store, config)),
        }
    }

    /// Returns the inner retriever reference, for direct access to the underlying index.
    ///
    /// For example, read-only retrieval uses `retriever.inner().read().await`; ingestion
    /// uses `.write().await`.
    pub fn inner(&self) -> &RwLock<ChunkedBM25Retriever<S>> {
        &self.inner
    }
}

#[async_trait]
impl<S: ChunkedDocumentStoreTrait> RetrieverTrait for ParentDocumentRetriever<S> {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
        let inner = self.inner.read().await;
        let matched = inner.search_matched_parents(query, k);
        let docs: Vec<Document> = matched
            .into_iter()
            .filter_map(|(parent_id, _)| inner.get_parent_document(&parent_id))
            .collect();
        Ok(docs)
    }

    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        let inner = self.inner.read().await;
        let matched = inner.search_matched_parents(query, k);
        let results: Vec<SearchResult> = matched
            .into_iter()
            .filter_map(|(parent_id, score)| {
                inner
                    .get_parent_document(&parent_id)
                    .map(|document| SearchResult { document, score })
            })
            .collect();
        Ok(results)
    }

    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        let mut inner = self.inner.write().await;
        inner
            .add_documents_async(documents)
            .await
            .map_err(RetrieverError::StoreError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RetrieverRunnable;
    use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk};
    use lc_core::runnables::{Runnable, RunnableConfig, RunnableExt, RunnableLambda};
    use lc_prompts::ChatPromptTemplate;
    use lc_schema::Message;
    use std::collections::HashMap;
    use std::pin::Pin;

    /// A parent-document content spanning multiple leaves (> leaf_chunk_size 400, with the distinctive word `zebra`).
    fn multi_leaf_parent_content() -> String {
        let filler =
            "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor \
             incididunt ut labore et dolore magna aliqua. ";
        format!("{filler}{filler}{filler}{filler} zebra")
    }

    fn test_retriever() -> ParentDocumentRetriever {
        ParentDocumentRetriever::new(Arc::new(ChunkedDocumentStore::new()))
    }

    #[tokio::test]
    async fn parent_document_returns_full_parent_on_chunk_hit() {
        let retriever = test_retriever();
        let content = multi_leaf_parent_content();
        assert!(
            content.len() > 400,
            "content must span multiple leaves, got {} chars",
            content.len()
        );

        retriever
            .add_documents(vec![Document::new(content.clone())])
            .await
            .unwrap();

        // Query a word that appears in only one leaf: a leaf hit returns the entire parent document.
        let docs = retriever.retrieve("zebra", 1).await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].content, content,
            "a leaf hit must return the FULL parent document, not the leaf chunk"
        );
        assert!(docs[0].content.len() > 400);
    }

    #[tokio::test]
    async fn parent_document_retrieve_with_scores_reports_score() {
        let retriever = test_retriever();
        let content = multi_leaf_parent_content();
        retriever
            .add_documents(vec![Document::new(content.clone())])
            .await
            .unwrap();

        let results = retriever.retrieve_with_scores("zebra", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.0, "BM25 score should be positive");
        assert_eq!(results[0].document.content, content);
    }

    /// E3 verification: the full ParentDocument → prompt → LLM chain.
    /// The retriever returns the whole parent document on a leaf hit, which is composed
    /// into the prompt and fed to the mock LLM to generate a reply.
    #[tokio::test]
    async fn parent_document_chains_into_prompt_and_llm() {
        let retriever = test_retriever();
        let content = multi_leaf_parent_content();
        retriever
            .add_documents(vec![Document::new(content.clone())])
            .await
            .unwrap();

        let step = RetrieverRunnable::new(Arc::new(retriever), 1);
        // Documents → template variables: join the retrieved results into `context`.
        let to_context = RunnableLambda::new_sync(|docs: Vec<Document>| {
            HashMap::from([(
                "context".to_string(),
                docs.iter()
                    .map(|d| d.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            )])
        });
        let prompt = ChatPromptTemplate::from_messages([
            Message::system("你是一个检索问答助手。"),
            Message::human("请根据以下资料回答问题:\n\n{context}"),
        ]);
        let chain = step.pipe(to_context).pipe(prompt).pipe(MockChat);

        let result: LLMResult = chain.invoke("zebra".to_string(), None).await.unwrap();
        assert!(
            result.content.contains("context has zebra: true"),
            "the full parent doc (containing `zebra`) must reach the model, got: {}",
            result.content
        );
    }

    /// Minimal mock chat model: echoes the length of the last human message, proving context reached the model.
    #[derive(Debug)]
    struct MockChat;

    #[derive(Debug)]
    struct MockChatError(String);

    impl std::fmt::Display for MockChatError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockChatError: {}", self.0)
        }
    }

    impl std::error::Error for MockChatError {}

    impl From<MockChatError> for lc_core::runnables::LcelError {
        fn from(e: MockChatError) -> Self {
            lc_core::runnables::LcelError::Other(e.0)
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockChat {
        type Error = MockChatError;

        async fn invoke(
            &self,
            input: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.chat(input, config).await
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockChat {
        fn model_name(&self) -> &str {
            "mock-chat"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len() / 4
        }

        fn with_temperature(self, _temp: f32) -> Self
        where
            Self: Sized,
        {
            self
        }

        fn with_max_tokens(self, _max: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for MockChat {
        async fn chat(
            &self,
            messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let last = messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            Ok(LLMResult {
                content: format!(
                    "context has zebra: {}; received {} chars of context",
                    last.contains("zebra"),
                    last.len()
                ),
                model: "mock-chat".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<
            Pin<Box<dyn futures_util::Stream<Item = Result<StreamChunk, Self::Error>> + Send>>,
            Self::Error,
        > {
            unreachable!("stream_chat not exercised in parent_document tests")
        }
    }
}
