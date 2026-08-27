// lc-rag/src/self_query.rs
//! SelfQueryRetriever — 自查询检索器
//!
//! 让 LLM 把自然语言查询拆成 `{ query, filter }`:清洗后的查询词走向量检索,
//! 解析出的 [`MetadataFilter`] 交给 `vector_store.similarity_search_with_filter`
//! 做元数据过滤(依赖 S3 的统一过滤能力)。拆解走 [`lc_core::judge::structured_call`]
//! (绑定工具拿结构化参数,模型不支持时回落文本解析),与 Guardrails / Evaluation
//! 同一执行路径;`allowed_attributes` 白名单拦截 LLM 用不存在的字段过滤。
//!
//! **不静默降级**:过滤器引用了白名单外字段时显式返回
//! [`RetrieverError::InvalidFilter`],绝不丢弃过滤器回落无过滤检索——那会让
//! 本该被过滤排除的数据被返回(数据面过曝)。空白名单 = 过滤整体禁用,过滤器
//! 一律忽略并记 warning(这是「禁用过滤」的既定模式,不是静默降级)。

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

/// LLM 拆解出的结构化参数:清洗后的查询词 + 可选元数据过滤。
///
/// `filter` 直接以 [`MetadataFilter`] 反序列化(filter.rs 对 JSON 形状做了宽松
/// 处理,兼容 LLM 输出差异);缺省为无过滤。
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SelfQueryArgs {
    /// 清洗掉过滤约束后的纯语义查询词。
    pub query: String,
    /// 元数据过滤条件(可选)。
    #[serde(default)]
    pub filter: Option<MetadataFilter>,
}

/// 自查询检索器:LLM 拆解查询 → 白名单校验 → 过滤相似度检索。
///
/// 实现 [`RetrieverTrait`],可被 [`crate::RetrieverRunnable`] 包进 LCEL 链。
pub struct SelfQueryRetriever<M: BaseChatModel> {
    llm: Arc<M>,
    store: Arc<dyn VectorStore>,
    embeddings: Arc<dyn Embeddings>,
    allowed_attributes: Vec<String>,
}

impl<M: BaseChatModel> SelfQueryRetriever<M> {
    /// 创建自查询检索器。
    ///
    /// - `llm`:负责拆解自然语言查询的模型(支持结构化输出,或可文本回落)。
    /// - `store` / `embeddings`:过滤相似度检索用的向量存储与嵌入模型。
    /// - `allowed_attributes`:允许出现在过滤字段的白名单;**空白名单 = 过滤整体
    ///   禁用**(LLM 返回的 filter 一律忽略并记 warning);**非空白名单下,过滤器
    ///   引用白名单外字段会显式返回 [`RetrieverError::InvalidFilter`]**,不静默
    ///   丢弃后回落无过滤检索。
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

    /// 自查询工具定义:让 LLM 以 `{ query, filter }` 结构化返回。
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

    /// 构建提示词:告诉 LLM 可用字段与输出格式。
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

    /// 调用 LLM 拆解查询(结构化或文本回落)。
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

    /// 白名单校验。
    ///
    /// - 无过滤器 → `Ok(None)`。
    /// - 空白名单 = 过滤整体禁用:过滤器一律忽略(这是「禁用过滤」的既定模式,
    ///   不是静默降级),记 warning 后 `Ok(None)`。
    /// - 非空白名单下,过滤器引用白名单外字段 → `Err(InvalidFilter)`。绝不丢弃
    ///   过滤器回落无过滤检索,否则本该被过滤排除的数据会被返回(数据面过曝)。
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

    /// 第一个引用白名单外字段的 key,遍历嵌套的 And/Or 组合;全部在列则 `None`。
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

/// 文本回落解析:优先尝试整段 JSON(LLM 输出结构时);否则把整段文本当查询词。
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

    /// 固定回复的 mock 聊天模型(走文本回落路径,不实现 bind_tools)。
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

    /// 造一个带 metadata 的内存向量存储 + mock 嵌入,返回 store 供断言。
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

    /// S4: 拆出的 filter 正确落到检索 —— 只返回匹配 source=docs 的文档。
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

    /// S2: 非法字段被白名单拦截 —— 显式报错,不静默丢弃过滤器回落无过滤检索。
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

    /// S2: 嵌套 And/Or 里出现白名单外字段同样显式报错。
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

    /// S2: 空白名单 = 过滤整体禁用 —— filter 忽略并返回全部文档,不报错。
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

    /// S4: 文本回落 —— 模型输出纯文本时整段当查询词,无过滤。
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

    /// S4: 派生形状 + 嵌套组合也能反序列化(LLM 输出 And/Or)。
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

    /// S4: `RetrieverRunnable` 组合进 LCEL 链编译并执行。
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

        // 链上再挂一步:验证类型链通(Vec<Document> → usize)。
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
