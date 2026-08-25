// lc-memory/src/mongo_memory.rs
//! MongoDB Persistent Memory Implementation

use async_trait::async_trait;
use mongodb::{
    bson::{doc, oid::ObjectId},
    options::ClientOptions,
    Client, Collection, Database,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock as StdRwLock;
use tokio::sync::RwLock;

use super::base::{BaseMemory, MemoryError};
use super::persistent::{MemoryData, PersistenceConfig, PersistentMemory};
use super::summary_buffer::ConversationSummaryBufferMemory;
use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_core::runnables::Runnable;
use lc_schema::{Message, MessageType};

/// MongoDB-stored memory document
#[derive(Debug, Serialize, Deserialize)]
struct MongoMemoryDoc {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    id: Option<ObjectId>,
    session_id: String,
    messages: Vec<Message>,
    summary: Option<String>,
    metadata: HashMap<String, String>,
    created_at: String,
    updated_at: String,
    /// P2-3: 乐观锁版本号,随每次保存 +1。旧文档缺省为 0。
    #[serde(default)]
    version: u64,
}

impl From<MemoryData> for MongoMemoryDoc {
    fn from(data: MemoryData) -> Self {
        Self {
            id: None,
            session_id: data.session_id,
            messages: data.messages,
            summary: data.summary,
            metadata: data.metadata,
            created_at: data.created_at,
            updated_at: data.updated_at,
            version: data.version,
        }
    }
}

impl From<MongoMemoryDoc> for MemoryData {
    fn from(doc: MongoMemoryDoc) -> Self {
        Self {
            session_id: doc.session_id,
            messages: doc.messages,
            summary: doc.summary,
            metadata: doc.metadata,
            created_at: doc.created_at,
            updated_at: doc.updated_at,
            version: doc.version,
        }
    }
}

/// P2-3: 乐观锁冲突合并纯函数(不依赖 DB,便于单测)。
///
/// 并发写入的两个进程各自基于旧版本总结时,以"已落库"的 `remote` 为基:
/// - 消息:追加本地快照中远程没有的新消息(按 角色+内容 去重),保留相对顺序,不丢消息;
/// - 摘要:取两者中较长者,避免摘要回退。
fn merge_memory_data(
    remote: &MemoryData,
    local_messages: &[Message],
    local_summary: &str,
) -> (Vec<Message>, String) {
    let mut seen: HashSet<(String, String)> = remote.messages.iter().map(message_key).collect();

    let mut messages = remote.messages.clone();
    for msg in local_messages {
        if seen.insert(message_key(msg)) {
            messages.push(msg.clone());
        }
    }

    let summary = match remote.summary.as_deref() {
        Some(remote_summary)
            if !remote_summary.trim().is_empty()
                && (local_summary.is_empty() || remote_summary.len() >= local_summary.len()) =>
        {
            remote_summary.to_string()
        }
        _ => local_summary.to_string(),
    };

    (messages, summary)
}

/// 消息去重键:角色 + 内容。`MessageType` 未实现 `Hash`/`Eq`,用稳定字符串表示。
fn message_key(msg: &Message) -> (String, String) {
    let role = match msg.message_type {
        MessageType::System => "system",
        MessageType::Human => "human",
        MessageType::AI => "ai",
        MessageType::Tool { .. } => "tool",
    };
    (role.to_string(), msg.content.clone())
}

/// MongoDB Persistent Memory
///
/// P0-3: 泛型化为 `M: BaseChatModel`,不再硬编码 `OpenAIChat`,可换
/// DeepSeek/Qwen/Ollama 等任意实现 `BaseChatModel` 的 LLM。
pub struct MongoPersistentMemory<M: BaseChatModel> {
    inner: RwLock<ConversationSummaryBufferMemory<M>>,
    database: Database,
    collection_name: String,
    session_id: StdRwLock<Option<String>>,
    config: RwLock<PersistenceConfig>,
}

impl<M: BaseChatModel> MongoPersistentMemory<M> {
    /// Creates a new MongoDB-backed persistent memory connected to the given
    /// database and collection, using `llm` and `token_limit` for the summary buffer.
    pub async fn new(
        mongo_uri: &str,
        database_name: &str,
        collection_name: &str,
        llm: M,
        token_limit: usize,
    ) -> Result<Self, MemoryError> {
        let client_options = ClientOptions::parse(mongo_uri)
            .await
            .map_err(|e| MemoryError::LoadError(format!("MongoDB connection failed: {}", e)))?;

        let client = Client::with_options(client_options)
            .map_err(|e| MemoryError::LoadError(format!("MongoDB client error: {}", e)))?;

        let database = client.database(database_name);

        let inner = ConversationSummaryBufferMemory::<M>::new(llm, token_limit);

        // P1-3: token_limit 单一来源——构造参数与 config 同步,`with_config` 后续覆盖。
        let config = PersistenceConfig::default().with_token_limit(token_limit);

        Ok(Self {
            inner: RwLock::new(inner),
            database,
            collection_name: collection_name.to_string(),
            session_id: StdRwLock::new(None),
            config: RwLock::new(config),
        })
    }

    /// Replaces the persistence config, keeping its token limit in sync with the inner memory.
    pub async fn with_config(mut self, config: PersistenceConfig) -> Self {
        // P1-3: config 的 token_limit 落到 inner 的预算,不再出现"config 改了 inner 没变"。
        self.inner.get_mut().set_max_token_limit(config.token_limit);
        *self.config.write().await = config;
        self
    }

    fn collection(&self) -> Collection<MongoMemoryDoc> {
        self.database.collection(&self.collection_name)
    }

    /// Creates a unique index on `session_id` to prevent duplicate documents.
    pub async fn create_indexes(&self) -> Result<(), MemoryError> {
        let collection = self.collection();

        // P2-3: session_id 唯一索引 —— 品牌新会话并发 upsert(都按 version=0 插入)时的
        // 最后防线,防止重复文档。用独立索引名,避免与历史部署的非唯一索引冲突。
        collection
            .create_index(
                mongodb::IndexModel::builder()
                    .keys(doc! { "session_id": 1 })
                    .options(
                        mongodb::options::IndexOptions::builder()
                            .name("session_id_unique".to_string())
                            .unique(true)
                            .build(),
                    )
                    .build(),
                None,
            )
            .await
            .map_err(|e| MemoryError::SaveError(format!("Index creation failed: {}", e)))?;

        Ok(())
    }

    /// Returns the current session ID, if any.
    pub async fn get_session_id(&self) -> Option<String> {
        self.session_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Sets the session ID used for subsequent persistence operations.
    pub async fn set_session_id_async(&self, session_id: impl Into<String>) {
        *self.session_id.write().unwrap_or_else(|e| e.into_inner()) = Some(session_id.into());
    }

    async fn do_load_from_store(&self, session_id: &str) -> Result<(), MemoryError> {
        let collection = self.collection();

        let filter = doc! { "session_id": session_id };
        let result = collection
            .find_one(filter, None)
            .await
            .map_err(|e| MemoryError::LoadError(format!("MongoDB find failed: {}", e)))?;

        if let Some(doc) = result {
            let data: MemoryData = doc.into();

            let mut inner = self.inner.write().await;
            let chat_memory = inner.chat_memory_mut();
            chat_memory.clear();

            for msg in &data.messages {
                if matches!(msg.message_type, lc_schema::MessageType::Human) {
                    chat_memory.add_user_message(&msg.content);
                } else if matches!(msg.message_type, lc_schema::MessageType::AI) {
                    chat_memory.add_ai_message(&msg.content);
                } else if matches!(msg.message_type, lc_schema::MessageType::System) {
                    chat_memory.add_system_message(&msg.content);
                }
            }

            // P1-3: 回灌摘要状态,保证续写会话从上次摘要继续(而非从空摘要起步)。
            if let Some(summary) = &data.summary {
                if !summary.trim().is_empty() {
                    inner.set_summary(summary.clone());
                }
            }
        }

        *self.session_id.write().unwrap_or_else(|e| e.into_inner()) = Some(session_id.to_string());

        Ok(())
    }

    /// P2-3: Mongo 乐观锁保存。
    ///
    /// 每次写入携带 `version` 并按 `{session_id, version: expected}` 替换;
    /// 未命中说明有并发写入者先落库 -> 读取最新文档、与本地快照合并后重试。
    /// 品牌新会话(version 0)使用 upsert,由 `session_id` 唯一索引兜底防重复。
    async fn do_save_to_store(&self, session_id: &str) -> Result<(), MemoryError> {
        const MAX_ATTEMPTS: u32 = 3;
        let collection = self.collection();

        for _attempt in 0..MAX_ATTEMPTS {
            // 快照当前内存状态(每次重试重新快照,合并基于最新本地状态)。
            let (local_messages, local_summary) = {
                let inner = self.inner.read().await;
                (
                    inner.chat_memory().messages().to_vec(),
                    inner.buffer().await,
                )
            };

            // 读最新远端文档:作为版本基线 + 冲突时合并的来源。
            let existing = collection
                .find_one(doc! { "session_id": session_id }, None)
                .await
                .map_err(|e| MemoryError::SaveError(format!("MongoDB find failed: {}", e)))?;

            let expected_version = existing.as_ref().map(|d| d.version).unwrap_or(0);
            let existing_created_at = existing.as_ref().map(|d| d.created_at.clone());

            let (messages, summary) = match existing {
                Some(remote) => {
                    let remote_data: MemoryData = remote.into();
                    merge_memory_data(&remote_data, &local_messages, &local_summary)
                }
                None => (local_messages, local_summary),
            };

            let now = chrono::Utc::now().to_rfc3339();
            let data = MemoryData {
                session_id: session_id.to_string(),
                messages,
                summary: Some(summary),
                metadata: HashMap::new(),
                created_at: existing_created_at.unwrap_or_else(|| now.clone()),
                updated_at: now,
                version: expected_version + 1,
            };
            let new_doc: MongoMemoryDoc = data.into();

            let opts = mongodb::options::ReplaceOptions::builder()
                .upsert(expected_version == 0)
                .build();
            let res = collection
                .replace_one(
                    doc! { "session_id": session_id, "version": expected_version as i64 },
                    new_doc,
                    opts,
                )
                .await
                .map_err(|e| MemoryError::SaveError(format!("MongoDB save failed: {}", e)))?;

            // 命中即落库成功;未命中说明版本已前进(并发写入),进入下一轮合并重试。
            if res.matched_count > 0 || res.upserted_id.is_some() {
                return Ok(());
            }
        }

        Err(MemoryError::SaveError(format!(
            "MongoDB optimistic lock conflict: session {} kept being concurrently written within {} retries",
            session_id, MAX_ATTEMPTS
        )))
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseMemory for MongoPersistentMemory<M>
where
    <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
{
    fn memory_variables(&self) -> Vec<&str> {
        vec!["history"]
    }

    async fn load_memory_variables(
        &self,
        inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, serde_json::Value>, MemoryError> {
        let config = self.config.read().await;
        if config.auto_load {
            let session_id = self
                .session_id
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            if let Some(sid) = session_id {
                let inner = self.inner.read().await;
                if inner.chat_memory().is_empty() {
                    drop(inner);
                    drop(config);
                    self.do_load_from_store(&sid).await?;
                }
            }
        }

        let inner = self.inner.read().await;
        inner.load_memory_variables(inputs).await
    }

    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        {
            let mut inner = self.inner.write().await;
            inner.save_context(inputs, outputs).await?;
        }

        let config = self.config.read().await;
        if config.auto_save {
            let session_id = self
                .session_id
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            if let Some(sid) = session_id {
                drop(config);
                self.do_save_to_store(&sid).await?;
            }
        }

        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        {
            let mut inner = self.inner.write().await;
            inner.clear().await?;
        }

        let session_id = self
            .session_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(sid) = session_id {
            self.do_delete_session(&sid).await?;
        }

        Ok(())
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> PersistentMemory for MongoPersistentMemory<M>
where
    <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
{
    async fn load_from_store(&mut self, session_id: &str) -> Result<(), MemoryError> {
        self.do_load_from_store(session_id).await
    }

    async fn save_to_store(&mut self, session_id: &str) -> Result<(), MemoryError> {
        self.do_save_to_store(session_id).await
    }

    async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        self.do_delete_session(session_id).await
    }

    async fn session_exists(&self, session_id: &str) -> Result<bool, MemoryError> {
        let collection = self.collection();
        let filter = doc! { "session_id": session_id };
        let result = collection
            .find_one(filter, None)
            .await
            .map_err(|e| MemoryError::LoadError(format!("MongoDB find failed: {}", e)))?;

        Ok(result.is_some())
    }

    /// P0-2: 从内部字段读真实会话 ID(与 `set_session_id` 读写对称)。
    fn current_session_id(&self) -> Option<String> {
        self.session_id
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn set_session_id(&mut self, session_id: String) {
        // C17: Using std::sync::RwLock instead of tokio::sync::RwLock::blocking_write
        // to avoid potential deadlock in async runtime.
        *self.session_id.write().unwrap_or_else(|e| e.into_inner()) = Some(session_id);
    }
}

impl<M: BaseChatModel> MongoPersistentMemory<M> {
    async fn do_delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        let collection = self.collection();

        collection
            .delete_one(doc! { "session_id": session_id }, None)
            .await
            .map_err(|e| MemoryError::ClearError(format!("MongoDB delete failed: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mongo_memory_doc_from_memory_data() {
        let data = MemoryData::new("session_123".to_string());
        let doc: MongoMemoryDoc = data.into();

        assert_eq!(doc.session_id, "session_123");
        assert!(doc.messages.is_empty());
        assert_eq!(doc.version, 0);
    }

    #[test]
    fn test_memory_data_from_mongo_doc() {
        let doc = MongoMemoryDoc {
            id: None,
            session_id: "session_456".to_string(),
            messages: vec![Message::human("Hello")],
            summary: Some("Test summary".to_string()),
            metadata: HashMap::new(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            version: 7,
        };

        let data: MemoryData = doc.into();
        assert_eq!(data.session_id, "session_456");
        assert_eq!(data.messages.len(), 1);
        assert_eq!(data.version, 7);
    }

    /// P2-3: MemoryData <-> MongoMemoryDoc 往返保留版本号。
    #[test]
    fn test_version_roundtrip() {
        let mut data = MemoryData::new("s".to_string());
        data.version = 42;
        let doc: MongoMemoryDoc = data.clone().into();
        let back: MemoryData = doc.into();
        assert_eq!(back.version, 42);
    }

    /// P2-3: 合并时追加本地新消息(按 类型+内容 去重),保留顺序。
    #[test]
    fn test_merge_appends_new_messages() {
        let remote = MemoryData::new("s".to_string())
            .with_messages(vec![Message::human("h1"), Message::ai("a1")]);
        let local = vec![
            Message::human("h1"),
            Message::ai("a1"),
            Message::human("h2"),
            Message::ai("a2"),
        ];

        let (messages, summary) = merge_memory_data(&remote, &local, "");
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[2].content, "h2");
        assert_eq!(messages[3].content, "a2");
        assert_eq!(summary, "");
    }

    /// P2-3: 摘要取两者中较长者,避免并发写入导致摘要回退。
    #[test]
    fn test_merge_prefers_longer_summary() {
        let remote = MemoryData::new("s".to_string()).with_summary("shorter".to_string());
        let (_, summary) = merge_memory_data(&remote, &[], "a much longer local summary");
        assert_eq!(summary, "a much longer local summary");

        let remote =
            MemoryData::new("s".to_string()).with_summary("longer remote summary".to_string());
        let (_, summary) = merge_memory_data(&remote, &[], "short");
        assert_eq!(summary, "longer remote summary");
    }

    /// P2-3: 远端无摘要时回退到本地摘要。
    #[test]
    fn test_merge_uses_local_summary_when_remote_none() {
        let remote = MemoryData::new("s".to_string());
        let (_, summary) = merge_memory_data(&remote, &[], "local-only");
        assert_eq!(summary, "local-only");
    }
}
