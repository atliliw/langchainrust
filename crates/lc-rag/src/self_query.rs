// lc-rag/src/self_query.rs
//! SelfQueryRetriever — a self-querying retriever
//!
//! Lets an LLM split a natural-language query into `{ query, filter }`: the cleaned query
//! goes to vector retrieval, and the parsed [`MetadataFilter`] is handed to
//! `vector_store.similarity_search_with_filter` for metadata filtering (relying on S3's
//! unified filtering capability). The split goes through [`lc_core::judge::structured_call`]
//! (binds a tool to get structured arguments, falling back to text parsing when the model
//! does not support it), the same execution path as Guardrails / Evaluation; the
//! `allowed_attributes` whitelist blocks the LLM from filtering on fields that do not exist.
//!
//! **No silent degradation**: when a filter references a field outside the whitelist, it
//! explicitly returns [`RetrieverError::InvalidFilter`] rather than dropping the filter and
//! falling back to an unfiltered search — that would return data that should have been
//! filtered out (data-plane over-exposure). An empty whitelist = filtering is entirely
//! disabled: filters are always ignored with a warning (this is the established "disable
//! filtering" pattern, not silent degradation).

use std::sync::Arc;

use async_trait::async_trait;
use lc_core::judge::{structured_call, StructuredJudgeError};
use lc_core::language_models::BaseChatModel;
use lc_core::tools::ToolDefinition;
use lc_embeddings::Embeddings;
use lc_schema::Message;
use lc_vector_stores::{Document, MetadataFilter, SearchResult, VectorStore};
use serde::Deserialize;

use crate::retriever::{RetrieverError, RetrieverTrait};

/// Structured parameters parsed by the LLM: the cleaned query + an optional metadata filter.
///
/// `filter` is deserialized directly as [`MetadataFilter`] (filter.rs is lenient about
/// the JSON shape to tolerate LLM output variance); the default is no filter.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SelfQueryArgs {
    /// The pure semantic query with the filtering constraints stripped out.
    pub query: String,
    /// Metadata filter (optional).
    #[serde(default)]
    pub filter: Option<MetadataFilter>,
}

/// Self-querying retriever: LLM splits the query -> whitelist validation -> filtered similarity search.
///
/// Implements [`RetrieverTrait`] and can be wrapped into an LCEL chain by [`crate::RetrieverRunnable`].
pub struct SelfQueryRetriever<M: BaseChatModel> {
    llm: Arc<M>,
    store: Arc<dyn VectorStore>,
    embeddings: Arc<dyn Embeddings>,
    allowed_attributes: Vec<String>,
}

impl<M: BaseChatModel> SelfQueryRetriever<M> {
    /// Creates a self-querying retriever.
    ///
    /// - `llm`: the model responsible for splitting the natural-language query (supports
    ///   structured output, or falls back to text).
    /// - `store` / `embeddings`: the vector store and embedding model used for the filtered
    ///   similarity search.
    /// - `allowed_attributes`: the whitelist of attributes allowed in filter fields;
    ///   **an empty whitelist disables filtering entirely** (LLM-returned filters are always
    ///   ignored with a warning); **with a non-empty whitelist, a filter referencing a field
    ///   outside the whitelist explicitly returns [`RetrieverError::InvalidFilter`]** rather
    ///   than silently dropping it and falling back to an unfiltered search.
    pub fn new(
        llm: impl Into<Arc<M>>,
        store: Arc<dyn VectorStore>,
        embeddings: Arc<dyn Embeddings>,
        allowed_attributes: Vec<String>,
    ) -> Self {
        Self {
            llm: llm.into(),
            store,
            embeddings,
            allowed_attributes,
        }
    }

    /// Self-query tool definition: lets the LLM return `{ query, filter }` structurally.
    fn self_query_tool() -> ToolDefinition {
        ToolDefinition::new(
            "self_query",
            "把用户的自然语言查询拆成纯语义查询词和可选的元数据过滤条件。",
        )
        .with_parameters(serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "清洗掉过滤约束后的纯语义查询词"
                },
                "filter": {
                    "type": ["object", "null"],
                    "description": "元数据过滤条件(MetadataFilter JSON):单条件 {\"Field\": {\"key\", \"op\", \"value\"}},组合 {\"And\": [...]} / {\"Or\": [...]};op 取 Eq Ne Gt Gte Lt Lte In Nin(In/Nin 的 value 为数组);无过滤时为 null"
                }
            },
            "required": ["query"]
        }))
    }

    /// Builds the prompt: tells the LLM the available fields and the output format.
    fn build_prompt(&self, query: &str) -> String {
        let allowed = if self.allowed_attributes.is_empty() {
            "无(本检索器不启用元数据过滤,filter 必须为 null)".to_string()
        } else {
            self.allowed_attributes.join(", ")
        };
        format!(
            "把下面的自然语言查询拆成两部分:纯语义查询词(query)和可选的元数据过滤条件(filter)。\n\
             filter 的 key 只能取以下允许字段之一: {allowed}\n\
             filter 的 JSON 形状:单条件 {{\"Field\": {{\"key\": ..., \"op\": ..., \"value\": ...}}}};\n\
             组合条件 {{\"And\": [...]}} / {{\"Or\": [...]}};op 取 Eq Ne Gt Gte Lt Lte In Nin(In/Nin 的 value 为数组)。\n\
             没有过滤需求时 filter 为 null。\n\
             用户查询: {query}"
        )
    }

    /// Calls the LLM to split the query (structured or text fallback).
    async fn parse_query(&self, query: &str) -> Result<SelfQueryArgs, RetrieverError> {
        let messages = vec![Message::human(self.build_prompt(query))];
        structured_call(
            &*self.llm,
            Self::self_query_tool(),
            messages,
            parse_text_fallback,
        )
        .await
        .map_err(|e| RetrieverError::LlmError(e.to_string()))
    }

    /// Whitelist validation.
    ///
    /// - No filter -> `Ok(None)`.
    /// - Empty whitelist = filtering is entirely disabled: filters are always ignored (this
    ///   is the established "disable filtering" pattern, not silent degradation), logging a
    ///   warning and returning `Ok(None)`.
    /// - With a non-empty whitelist, a filter referencing a field outside the whitelist ->
    ///   `Err(InvalidFilter)`. It never drops the filter to fall back to an unfiltered search,
    ///   otherwise data that should have been filtered out would be returned (data-plane
    ///   over-exposure).
    fn validated_filter(
        &self,
        filter: &Option<MetadataFilter>,
    ) -> Result<Option<MetadataFilter>, RetrieverError> {
        let Some(f) = filter else {
            return Ok(None);
        };
        if self.allowed_attributes.is_empty() {
            log::warn!(
                "SelfQuery: filtering is disabled (empty allowed_attributes); ignoring filter"
            );
            return Ok(None);
        }
        if let Some(field) = Self::disallowed_field(f, &self.allowed_attributes) {
            return Err(RetrieverError::InvalidFilter(format!(
                "filter references `{field}`, which is not in allowed_attributes [{}]; \
                 refusing to degrade to an unfiltered search",
                self.allowed_attributes.join(", ")
            )));
        }
        Ok(filter.clone())
    }

    /// The first key referencing a field outside the whitelist, traversing nested And/Or
    /// combinations; returns `None` when all fields are listed.
    fn disallowed_field<'a>(filter: &'a MetadataFilter, allowed: &[String]) -> Option<&'a String> {
        match filter {
            MetadataFilter::Field { key, .. } => {
                if allowed.iter().any(|a| a == key) {
                    None
                } else {
                    Some(key)
                }
            }
            MetadataFilter::And(items) | MetadataFilter::Or(items) => items
                .iter()
                .find_map(|f| Self::disallowed_field(f, allowed)),
        }
    }
}

/// Text fallback parsing: tries the whole JSON first (when the LLM outputs structured);
/// otherwise treats the whole text as the query.
fn parse_text_fallback(raw: &str) -> Result<SelfQueryArgs, StructuredJudgeError> {
    let trimmed = raw.trim();
    if !trimmed.is_empty() {
        if let Ok(args) = serde_json::from_str::<SelfQueryArgs>(trimmed) {
            return Ok(args);
        }
    }
    let query = raw.trim().to_string();
    if query.is_empty() {
        return Err(StructuredJudgeError::Parse(
            "self-query fallback produced an empty query".to_string(),
        ));
    }
    Ok(SelfQueryArgs {
        query,
        filter: None,
    })
}

#[async_trait]
impl<M: BaseChatModel> RetrieverTrait for SelfQueryRetriever<M> {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
        let results = self.retrieve_with_scores(query, k).await?;
        Ok(results.into_iter().map(|r| r.document).collect())
    }

    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        let args = self.parse_query(query).await?;
        let filter = self.validated_filter(&args.filter)?;

        let query_embedding = self
            .embeddings
            .embed_query(&args.query)
            .await
            .map_err(|e| RetrieverError::EmbeddingError(e.to_string()))?;

        self.store
            .similarity_search_with_filter(&query_embedding, k, filter.as_ref())
            .await
            .map_err(RetrieverError::from)
    }

    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        let texts: Vec<&str> = documents.iter().map(|d| d.content.as_str()).collect();
        let embeddings = self
            .embeddings
            .embed_documents(&texts)
            .await
            .map_err(|e| RetrieverError::EmbeddingError(e.to_string()))?;
        self.store.add_documents(documents, embeddings).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::language_models::{LLMResult, StreamChunk};
    use lc_core::runnables::RunnableConfig;
    use lc_core::{BaseLanguageModel, Runnable};
    use lc_embeddings::MockEmbeddings;
    use lc_vector_stores::InMemoryVectorStore;
    use std::collections::HashSet;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A mock chat model with a fixed reply (exercises the text fallback path; does not implement bind_tools).
    struct MockChatModel {
        reply: String,
        calls: AtomicUsize,
    }

    impl MockChatModel {
        fn new(reply: &str) -> Self {
            Self {
                reply: reply.to_string(),
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockChatModel {
        type Error = MockChatError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(MockChatError)
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockChatModel {
        fn model_name(&self) -> &str {
            "self-query-mock"
        }
        fn get_num_tokens(&self, t: &str) -> usize {
            t.len()
        }
        fn with_temperature(self, _: f32) -> Self {
            self
        }
        fn with_max_tokens(self, _: usize) -> Self {
            self
        }
    }

    #[derive(Debug)]
    struct MockChatError;
    impl std::fmt::Display for MockChatError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "mock chat error")
        }
    }
    impl std::error::Error for MockChatError {}

    #[async_trait]
    impl BaseChatModel for MockChatModel {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(LLMResult {
                content: self.reply.clone(),
                model: "self-query-mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            Err(MockChatError)
        }
    }

    /// Builds an in-memory vector store with metadata + a mock embedding, returning the store for assertions.
    async fn store_with_docs() -> Arc<InMemoryVectorStore> {
        let store = Arc::new(InMemoryVectorStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(64));
        let docs = vec![
            Document::new("Rust systems programming").with_metadata("source", "docs"),
            Document::new("Rust borrow checker").with_metadata("source", "docs"),
            Document::new("Python scripting").with_metadata("source", "blog"),
        ];
        let texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
        let vecs = embeddings.embed_documents(&texts).await.unwrap();
        store.add_documents(docs, vecs).await.unwrap();
        store
    }

    fn build_retriever(
        llm: MockChatModel,
        store: Arc<dyn VectorStore>,
        allowed: &[&str],
    ) -> SelfQueryRetriever<MockChatModel> {
        SelfQueryRetriever::new(
            Arc::new(llm),
            store,
            Arc::new(MockEmbeddings::new(64)),
            allowed.iter().map(|s| s.to_string()).collect(),
        )
    }

    /// S4: the parsed filter correctly reaches the search — only docs matching source=docs are returned.
    #[tokio::test]
    async fn test_self_query_filter_reaches_search() {
        let store = store_with_docs().await;
        let llm = MockChatModel::new(
            r#"{"query": "rust", "filter": {"key": "source", "op": "eq", "value": "docs"}}"#,
        );
        let retriever = build_retriever(llm, store.clone(), &["source"]);

        let results = retriever
            .retrieve("告诉我关于 Rust 的文档", 10)
            .await
            .unwrap();
        let contents: HashSet<&str> = results.iter().map(|d| d.content.as_str()).collect();
        assert_eq!(
            contents,
            HashSet::from(["Rust systems programming", "Rust borrow checker"])
        );
    }

    /// S2: a disallowed field is blocked by the whitelist — explicit error, not a silent filter drop into an unfiltered search.
    #[tokio::test]
    async fn test_self_query_rejects_disallowed_attribute() {
        let store = store_with_docs().await;
        let llm = MockChatModel::new(
            r#"{"query": "rust", "filter": {"key": "nonexistent", "op": "eq", "value": 1}}"#,
        );
        let retriever = build_retriever(llm, store.clone(), &["source"]);

        let err = retriever.retrieve("rust", 10).await.unwrap_err();
        assert!(matches!(err, RetrieverError::InvalidFilter(_)));
        assert!(err.to_string().contains("nonexistent"));
    }

    /// S2: a whitelist-outside field inside a nested And/Or also errors out explicitly.
    #[tokio::test]
    async fn test_self_query_rejects_disallowed_attribute_in_nested_and() {
        let store = store_with_docs().await;
        let llm = MockChatModel::new(
            r#"{"query": "rust", "filter": {"And": [{"key": "source", "op": "eq", "value": "docs"}, {"key": "private", "op": "eq", "value": false}]}}"#,
        );
        let retriever = build_retriever(llm, store.clone(), &["source"]);

        let err = retriever.retrieve("rust", 10).await.unwrap_err();
        assert!(matches!(err, RetrieverError::InvalidFilter(_)));
        assert!(err.to_string().contains("private"));
    }

    /// S2: empty whitelist = filtering entirely disabled — filter ignored and all docs returned, no error.
    #[tokio::test]
    async fn test_self_query_empty_whitelist_ignores_filter() {
        let store = store_with_docs().await;
        let llm = MockChatModel::new(
            r#"{"query": "rust", "filter": {"key": "source", "op": "eq", "value": "docs"}}"#,
        );
        let retriever = build_retriever(llm, store.clone(), &[]);

        let results = retriever.retrieve("rust", 10).await.unwrap();
        assert_eq!(results.len(), 3, "filtering disabled -> all docs returned");
    }

    /// S4: text fallback — when the model outputs plain text, the whole text is used as the query, no filter.
    #[tokio::test]
    async fn test_self_query_text_fallback_query_only() {
        let store = store_with_docs().await;
        let llm = MockChatModel::new("rust programming");
        let retriever = build_retriever(llm, store.clone(), &["source"]);

        let results = retriever.retrieve("rust", 10).await.unwrap();
        assert_eq!(
            results.len(),
            3,
            "plain-text fallback must search without filter"
        );
    }

    /// S4: derived shapes + nested combinations deserialize too (LLM outputs And/Or).
    #[tokio::test]
    async fn test_self_query_nested_filter_parses() {
        let store = store_with_docs().await;
        let llm = MockChatModel::new(
            r#"{"query": "rust", "filter": {"And": [{"key": "source", "op": "eq", "value": "docs"}]}}"#,
        );
        let retriever = build_retriever(llm, store.clone(), &["source"]);

        let results = retriever.retrieve("rust", 10).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    /// S4: `RetrieverRunnable` composes into an LCEL chain that compiles and runs.
    #[tokio::test]
    async fn test_self_query_pipes_into_retriever_runnable() {
        use crate::RetrieverRunnable;
        use lc_core::runnables::RunnableExt;

        let store = store_with_docs().await;
        let llm = MockChatModel::new(
            r#"{"query": "rust", "filter": {"key": "source", "op": "eq", "value": "docs"}}"#,
        );
        let retriever: Arc<dyn RetrieverTrait> =
            Arc::new(build_retriever(llm, store.clone(), &["source"]));

        let step = RetrieverRunnable::new(retriever, 10);
        let docs = step
            .invoke("告诉我 Rust 的文档".to_string(), None)
            .await
            .unwrap();
        assert_eq!(docs.len(), 2);

        // Chain one more step: verify the type chain (Vec<Document> -> usize).
        let count = step
            .pipe(lc_core::runnables::RunnableLambda::new_sync(
                |docs: Vec<Document>| docs.len(),
            ))
            .invoke("rust 文档".to_string(), None)
            .await
            .unwrap();
        assert_eq!(count, 2);
    }
}
