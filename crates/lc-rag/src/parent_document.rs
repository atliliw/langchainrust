// lc-rag/src/parent_document.rs
//! ParentDocumentRetriever — 父子文档检索器

use async_trait::async_trait;
use lc_vector_stores::{ChunkedDocumentStore, ChunkedDocumentStoreTrait, Document, SearchResult};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::bm25::{AutoMergingConfig, ChunkedBM25Retriever};
use crate::retriever::{RetrieverError, RetrieverTrait};

/// 父子文档检索器。
///
/// 文档入库时被切成小块(leaf)索引,查询命中任意小块就返回**整篇父文档**。
/// 这与 [`ChunkedBM25Retriever`] 的 AutoMerging(按命中比例门控,阈值不足
/// 返回小块)不同:ParentDocument 是"小块召回、整文喂给 LLM"的经典 RAG
/// 模式 —— 小块负责精确命中,整篇父文档负责给 LLM 提供完整上下文。
///
/// 内部用 `RwLock` 包一层 [`ChunkedBM25Retriever`]:检索走读锁、入库走写锁,
/// 因此能实现 [`RetrieverTrait`] (其方法全为
/// `&self`)。配合 [`RetrieverRunnable`](crate::RetrieverRunnable) 可以直接进
/// LCEL 链:
///
/// ```rust,ignore
/// let retriever = Arc::new(ParentDocumentRetriever::new(store));
/// let chain = RetrieverRunnable::new(retriever, 4).pipe(prompt).pipe(llm);
/// ```
pub struct ParentDocumentRetriever<S: ChunkedDocumentStoreTrait = ChunkedDocumentStore> {
    inner: RwLock<ChunkedBM25Retriever<S>>,
}

impl<S: ChunkedDocumentStoreTrait> ParentDocumentRetriever<S> {
    /// 使用默认配置创建检索器。
    pub fn new(store: Arc<S>) -> Self {
        Self {
            inner: RwLock::new(ChunkedBM25Retriever::new(store)),
        }
    }

    /// 使用指定 AutoMerging 配置创建检索器。
    ///
    /// 配置中的 `leaf_chunk_size` 决定小块切分粒度;`merge_threshold` 在
    /// ParentDocument 语义下不参与命中判定(命中任意小块即返回父文档),但
    /// 仍用于约束底层索引的切分行为。
    pub fn with_config(store: Arc<S>, config: AutoMergingConfig) -> Self {
        Self {
            inner: RwLock::new(ChunkedBM25Retriever::with_config(store, config)),
        }
    }

    /// 返回内部检索器引用,需要直接访问底层索引时使用。
    ///
    /// 例如只读检索用 `retriever.inner().read().await`,入库用 `.write().await`。
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

    /// 一段跨多个 leaf 的父文档内容(> leaf_chunk_size 400,含独特词 `zebra`)。
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

        // 查询只出现在某个 leaf 里的词:命中小块 → 返回整篇父文档。
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

    /// E3 验证:ParentDocument → prompt → LLM 整条链。
    /// 检索器命中小块返回整篇父文档,拼进 prompt,喂给 mock LLM 生成回复。
    #[tokio::test]
    async fn parent_document_chains_into_prompt_and_llm() {
        let retriever = test_retriever();
        let content = multi_leaf_parent_content();
        retriever
            .add_documents(vec![Document::new(content.clone())])
            .await
            .unwrap();

        let step = RetrieverRunnable::new(Arc::new(retriever), 1);
        // 文档 → 模板变量:把检索结果拼成 `context`。
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

    /// 最小 mock 聊天模型:把最后一条人类消息的长度回显,证明上下文流到了模型侧。
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
