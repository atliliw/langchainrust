// lc-memory/src/with_history.rs
//! LCEL 组合用的"带记忆的 LLM 封装"。
//!
//! 把"读记忆 → 拼用户输入 → 调 LLM → 写回记忆"这一组合封装成单个
//! `Runnable<String, LLMResult>`,让"LLM + 记忆"可以直接进入 LCEL 管道
//! (pipe 成链、batch、stream 等),不用在业务代码里手写记忆胶水。
//!
//! # 两种记忆来源
//!
//! - `new(llm, memory)` —— 注入一个具体记忆对象(单会话,原有行为不变)。
//! - `with_session_history(llm, factory)` —— 注入 **session 回调**(对齐 Python
//!   `RunnableWithMessageHistory(llm, get_session_history)`):每次调用从
//!   `config.configurable["session_id"]` 取槽,同一 session 共享历史,不同
//!   session 互不串扰;缺失 `session_id` 返回 `LcelError::Chain`。
//!
//! # 语义
//!
//! `invoke(user_input)` 依次执行:
//! 1. 按模式选定记忆(Shared 直接取;Sessions 按 session_id 取/建槽);
//! 2. 从记忆读取历史,转成消息(`memory_variables_to_messages`);
//! 3. 把用户输入作为 Human 消息追加到末尾;
//! 4. 交给 `llm.chat`(可选 `RunnableConfig` 透传);
//! 5. 把「用户输入 / 模型回答」写回记忆;
//! 6. 返回完整 `LLMResult`。
//!
//! LLM 错误通过 `L::Error: Into<LcelError>` 进入管道错误;记忆读写错误
//! 收敛为 `LcelError::Chain`。
//!
//! # 泛型
//!
//! `L` 是任意实现 `BaseChatModel` 的模型(原生 Provider / `LLMClient` 均可),
//! 只要其错误类型能转进 `LcelError`(`LLMClient` 天然满足;原生 Provider 见
//! lc-providers 的 `From<...> for LcelError`)。记忆以 trait 对象持有,任意
//! `BaseMemory`(Buffer / Window / Summary / SummaryBuffer 等)都可用。

use crate::base::{memory_variables_to_messages, BaseMemory};
use crate::buffer::ConversationBufferMemory;
use async_trait::async_trait;
use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use lc_schema::Message;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::Mutex;

/// 记忆句柄:任意 `BaseMemory` 的共享可变引用。
pub type SharedMemory = Arc<Mutex<Box<dyn BaseMemory>>>;

/// Session 缓存默认上限:防异常/恶意 session_id 无限增长占满内存(M2a)。
const DEFAULT_MAX_SESSIONS: usize = 100;

/// 带记忆的 LLM 封装,作为单个 Runnable 参与 LCEL 组合。
pub struct RunnableWithMessageHistory<L> {
    llm: Arc<L>,
    mode: HistoryMode,
}

/// 记忆来源:共享单槽(SHARED)或按 session 分槽(SESSIONS)。
enum HistoryMode {
    /// 构造时注入的单一记忆对象,所有调用共享。
    Shared(SharedMemory),
    /// 按 `configurable.session_id` 分槽:factory 建新槽,缓存复用已建槽。
    Sessions {
        factory: Arc<dyn Fn(&str) -> SharedMemory + Send + Sync>,
        /// 槽缓存:同一 session 复用同一份记忆(其锁同时串行化同槽并发 invoke)。
        cache: Mutex<SessionCache>,
        /// 缓存上限:超过即淘汰最旧 session,防 session_id 无限增长的内存 DoS(M2a)。
        max_sessions: usize,
        /// 仅用于 `memory()` 访问器的占位记忆(Sessions 模式下实际槽不唯一)。
        default: SharedMemory,
    },
}

/// Session 槽缓存:`slots` 按 session_id 取槽,`order` 记录插入顺序供上限淘汰。
struct SessionCache {
    slots: HashMap<String, SharedMemory>,
    order: VecDeque<String>,
}

impl<L> RunnableWithMessageHistory<L> {
    /// 用 LLM + 单个记忆对象构造封装(所有调用共享这一份记忆)。
    pub fn new(llm: L, memory: impl BaseMemory + 'static) -> Self {
        Self {
            llm: Arc::new(llm),
            mode: HistoryMode::Shared(Arc::new(Mutex::new(Box::new(memory)))),
        }
    }

    /// 用 LLM + session 回调构造封装(对齐 Python
    /// `RunnableWithMessageHistory(llm, get_session_history)`)。
    ///
    /// `factory(session_id)` 为一个 session 槽建出记忆对象;每次调用按
    /// `config.configurable["session_id"]` 选槽,同一 session 复用已建槽。
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipe = RunnableWithMessageHistory::with_session_history(llm, |session_id| {
    ///     Arc::new(Mutex::new(Box::new(
    ///         ConversationBufferMemory::new().with_return_messages(true),
    ///     ) as Box<dyn BaseMemory>))
    /// });
    /// let cfg = RunnableConfig::new().with_configurable("session_id", json!("s1"));
    /// pipe.invoke("我叫小明".into(), Some(cfg)).await?;
    /// ```
    pub fn with_session_history<F>(llm: L, factory: F) -> Self
    where
        F: Fn(&str) -> SharedMemory + Send + Sync + 'static,
    {
        Self {
            llm: Arc::new(llm),
            mode: HistoryMode::Sessions {
                factory: Arc::new(factory),
                cache: Mutex::new(SessionCache {
                    slots: HashMap::new(),
                    order: VecDeque::new(),
                }),
                max_sessions: DEFAULT_MAX_SESSIONS,
                default: Arc::new(Mutex::new(Box::new(
                    ConversationBufferMemory::new(),
                ))),
            },
        }
    }

    /// 设置 session 缓存上限(Sessions 模式)。超过上限后,新 session 会淘汰最旧
    /// 的槽,防 session_id 无限增长的内存 DoS(M2a)。默认 [`DEFAULT_MAX_SESSIONS`]。
    pub fn with_max_sessions(mut self, max: usize) -> Self {
        if let HistoryMode::Sessions { max_sessions, .. } = &mut self.mode {
            *max_sessions = max.max(1);
        }
        self
    }

    /// 暴露内部记忆句柄,便于读取已保存的历史(调试、展示、验证写回等)。
    ///
    /// Sessions 模式下槽不唯一,返回的是占位记忆(不会参与管道读写);
    /// 要检查真实历史请从管道内部或业务侧记忆对象读取。
    pub fn memory(&self) -> SharedMemory {
        match &self.mode {
            HistoryMode::Shared(m) => m.clone(),
            HistoryMode::Sessions { default, .. } => default.clone(),
        }
    }

    /// 按模式选定本次调用要用的记忆槽。
    async fn select_memory(
        &self,
        config: &Option<RunnableConfig>,
    ) -> Result<SharedMemory, LcelError> {
        match &self.mode {
            HistoryMode::Shared(m) => Ok(m.clone()),
            HistoryMode::Sessions {
                factory,
                cache,
                max_sessions,
                ..
            } => {
                let session_id = config
                    .as_ref()
                    .and_then(|c| c.configurable_value("session_id"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        LcelError::Chain(
                            "RunnableWithMessageHistory(session mode) 缺少 configurable.session_id"
                                .to_string(),
                        )
                    })?;
                let mut cache = cache.lock().await;
                if let Some(memory) = cache.slots.get(session_id) {
                    return Ok(memory.clone());
                }
                // M2a: 缓存有上限,先淘汰最旧的 session,再建新槽。
                if cache.slots.len() >= *max_sessions {
                    if let Some(oldest) = cache.order.pop_front() {
                        cache.slots.remove(&oldest);
                    }
                }
                let memory = factory(session_id);
                cache
                    .slots
                    .insert(session_id.to_string(), memory.clone());
                cache.order.push_back(session_id.to_string());
                Ok(memory)
            }
        }
    }
}

#[async_trait]
impl<L> Runnable<String, LLMResult> for RunnableWithMessageHistory<L>
where
    L: BaseChatModel + 'static,
    L::Error: Into<LcelError>,
{
    type Error = LcelError;

    async fn invoke(
        &self,
        input: String,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, LcelError> {
        // 0. 选定记忆槽(Sessions 模式读 configurable.session_id)
        let memory = self.select_memory(&config).await?;

        // M2b: 整个「读记忆 → 调 LLM → 写回」持锁执行,串行化同一记忆槽的并发
        // invoke。旧实现读后即放锁,两个并发 invoke 都读到旧历史、写回互相覆盖
        // 丢历史。代价是同槽调用串行,但带记忆的对话本就应串行。
        let mut memory = memory.lock().await;

        // 1. 读记忆 → 转消息
        let mut messages = {
            let vars = memory
                .load_memory_variables(&HashMap::new())
                .await
                .map_err(|e| LcelError::Chain(format!("load memory: {e}")))?;
            memory_variables_to_messages(&vars)
        };

        // 2. 拼上用户输入
        messages.push(Message::human(&input));

        // 3. 调 LLM(持锁等待:同槽串行,避免并发丢历史)
        let result = self
            .llm
            .chat(messages, config)
            .await
            .map_err(Into::into)?;

        // 4. 写回记忆:失败不丢弃模型答案(否则调用方拿 Err 重试会重复调 LLM),
        //    记 warn 暴露记忆层降级
        let inputs = HashMap::from([("input".to_string(), input)]);
        let outputs = HashMap::from([("output".to_string(), result.content.clone())]);
        if let Err(e) = memory.save_context(&inputs, &outputs).await {
            log::warn!("记忆写回失败(模型答案仍照常返回): {e}");
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::Stream;
    use lc_core::language_models::{BaseChatModel, BaseLanguageModel};
    use lc_core::runnables::RunnableConfig;
    use lc_schema::MessageType;
    use serde_json::json;
    use std::pin::Pin;
    use std::sync::Mutex as StdMutex;

    /// 测试 LLM:记录每次收到的消息,并把最后一条用户消息包成回答。
    struct TestChatModel {
        seen: Arc<StdMutex<Vec<Vec<Message>>>>,
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for TestChatModel {
        type Error = LcelError;

        async fn invoke(
            &self,
            input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, LcelError> {
            self.seen.lock().unwrap().push(input.clone());
            let last = input
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            Ok(LLMResult {
                content: format!("reply to: {last}"),
                ..Default::default()
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for TestChatModel {
        fn model_name(&self) -> &str {
            "test-llm"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len()
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
    impl BaseChatModel for TestChatModel {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, LcelError> {
            self.invoke(messages, config).await
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<String, LcelError>> + Send>>,
            LcelError,
        > {
            unimplemented!("stream_chat not needed for tests")
        }
    }

    /// session 回调:每次建一个空的 Buffer 记忆(return_messages = true)。
    fn session_factory(
        _session_id: &str,
    ) -> SharedMemory {
        Arc::new(Mutex::new(Box::new(
            ConversationBufferMemory::new().with_return_messages(true),
        ) as Box<dyn BaseMemory>))
    }

    /// 读记忆 → LLM → 写回:第二轮调用时,LLM 应看到第一轮的完整对话。
    #[tokio::test]
    async fn reads_memory_writes_back_round_trip() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen: seen.clone() };
        // return_messages = true:历史以消息数组返回,方便断言每轮消息构成
        let memory = ConversationBufferMemory::new().with_return_messages(true);
        let pipe = RunnableWithMessageHistory::new(llm, memory);

        // Act 第一轮:无历史
        let r1 = pipe.invoke("我叫什么名字".to_string(), None).await.unwrap();
        assert_eq!(r1.content, "reply to: 我叫什么名字");

        // Act 第二轮:历史应已写回
        let r2 = pipe.invoke("再问一次".to_string(), None).await.unwrap();
        assert_eq!(r2.content, "reply to: 再问一次");

        // Assert
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 2, "应调用模型两次");
        // 第一轮:只有用户消息
        assert_eq!(calls[0].len(), 1);
        assert_eq!(calls[0][0].content, "我叫什么名字");
        // 第二轮:user + ai + 新 user(记忆已写回)
        assert_eq!(calls[1].len(), 3);
        assert_eq!(calls[1][0].content, "我叫什么名字");
        assert!(matches!(calls[1][0].message_type, MessageType::Human));
        assert!(matches!(calls[1][1].message_type, MessageType::AI));
        assert_eq!(calls[1][2].content, "再问一次");
    }

    /// 记忆写回持久化在封装内:构造新封装、复用同一记忆类型,历史仍在。
    #[tokio::test]
    async fn memory_accumulates_across_invocations() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen: seen.clone() };
        let memory = ConversationBufferMemory::new().with_return_messages(true);
        let pipe = RunnableWithMessageHistory::new(llm, memory);

        // Act 三轮连续调用
        for turn in ["你好", "你在吗", "再见"] {
            pipe.invoke(turn.to_string(), None).await.unwrap();
        }

        // Assert 第三轮应看到前两轮完整四段对话
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[2].len(), 5); // 2 轮 * 2 段 + 当前用户消息
        assert_eq!(calls[2][4].content, "再见");
    }

    /// session 模式:同一 session_id 两轮调用共享历史。
    #[tokio::test]
    async fn session_history_same_session_shares_memory() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen: seen.clone() };
        let pipe = RunnableWithMessageHistory::with_session_history(llm, session_factory);
        let cfg = RunnableConfig::new().with_configurable("session_id", json!("s1"));

        // Act 同一 session 两轮
        let r1 = pipe.invoke("我叫什么名字".to_string(), Some(cfg.clone())).await.unwrap();
        assert_eq!(r1.content, "reply to: 我叫什么名字");
        let r2 = pipe.invoke("再问一次".to_string(), Some(cfg)).await.unwrap();
        assert_eq!(r2.content, "reply to: 再问一次");

        // Assert 第二轮看到第一轮完整对话(user + ai + user)
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].len(), 3);
        assert_eq!(calls[1][0].content, "我叫什么名字");
        assert!(matches!(calls[1][1].message_type, MessageType::AI));
    }

    /// session 模式:不同 session_id 互不串扰。
    #[tokio::test]
    async fn session_history_different_sessions_isolated() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen: seen.clone() };
        let pipe = RunnableWithMessageHistory::with_session_history(llm, session_factory);
        let cfg_s1 = RunnableConfig::new().with_configurable("session_id", json!("s1"));
        let cfg_s2 = RunnableConfig::new().with_configurable("session_id", json!("s2"));

        // Act s1 两轮 + s2 一轮
        pipe.invoke("我是 s1".to_string(), Some(cfg_s1.clone())).await.unwrap();
        pipe.invoke("还在 s1".to_string(), Some(cfg_s1)).await.unwrap();
        pipe.invoke("我是 s2".to_string(), Some(cfg_s2)).await.unwrap();

        // Assert s2 第一轮无历史(1 条),s1 第二轮有历史(3 条)
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[1].len(), 3, "s1 第二轮应看到历史");
        assert_eq!(calls[2].len(), 1, "s2 第一轮应无历史");
    }

    /// session 模式:缺失 configurable.session_id → LcelError::Chain。
    #[tokio::test]
    async fn session_history_missing_session_id_errors() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen };
        let pipe = RunnableWithMessageHistory::with_session_history(llm, session_factory);

        // Act 无 config / 无 session_id
        let err = pipe.invoke("你好".to_string(), None).await.unwrap_err();
        assert!(matches!(err, LcelError::Chain(_)));

        let cfg_no_sid = RunnableConfig::new();
        let err = pipe
            .invoke("你好".to_string(), Some(cfg_no_sid))
            .await
            .unwrap_err();
        assert!(matches!(err, LcelError::Chain(_)));
    }

    /// 可阻塞的测试 LLM:第一次 chat 通知 `entered` 并阻塞在 `release`,用于在
    /// 并发测试中制造「已持锁、模型调用中」的确定性窗口(M2b)。
    struct BlockingChatModel {
        seen: Arc<StdMutex<Vec<Vec<Message>>>>,
        entered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        blocked: Arc<StdMutex<bool>>,
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for BlockingChatModel {
        type Error = LcelError;

        async fn invoke(
            &self,
            input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, LcelError> {
            self.seen.lock().unwrap().push(input.clone());
            let should_block = {
                let mut blocked = self.blocked.lock().unwrap();
                if !*blocked {
                    *blocked = true;
                    true
                } else {
                    false
                }
            };
            self.entered.notify_one();
            if should_block {
                self.release.notified().await;
            }
            let last = input
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            Ok(LLMResult {
                content: format!("reply to: {last}"),
                ..Default::default()
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for BlockingChatModel {
        fn model_name(&self) -> &str {
            "blocking-test-llm"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len()
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
    impl BaseChatModel for BlockingChatModel {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, LcelError> {
            self.invoke(messages, config).await
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<String, LcelError>> + Send>>,
            LcelError,
        > {
            unimplemented!("stream_chat not needed for tests")
        }
    }

    /// session 缓存有上限:超过后淘汰最旧 session,防 session_id 无限增长的内存 DoS(M2a)。
    #[tokio::test]
    async fn session_cache_evicts_oldest_when_over_capacity() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen: seen.clone() };
        let pipe = RunnableWithMessageHistory::with_session_history(llm, session_factory)
            .with_max_sessions(2);

        let cfg_s1 = RunnableConfig::new().with_configurable("session_id", json!("s1"));
        let cfg_s2 = RunnableConfig::new().with_configurable("session_id", json!("s2"));
        let cfg_s3 = RunnableConfig::new().with_configurable("session_id", json!("s3"));

        // Act s1、s2 占满缓存;s3 触发淘汰最旧的 s1。
        pipe.invoke("s1-turn1".to_string(), Some(cfg_s1.clone()))
            .await
            .unwrap();
        pipe.invoke("s2-turn1".to_string(), Some(cfg_s2))
            .await
            .unwrap();
        pipe.invoke("s3-turn1".to_string(), Some(cfg_s3))
            .await
            .unwrap();
        // s1 已被淘汰 → 重入 s1 是全新会话。
        pipe.invoke("s1-turn2".to_string(), Some(cfg_s1))
            .await
            .unwrap();

        // Assert
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 4);
        assert_eq!(
            calls[3].len(),
            1,
            "M2a: s1 槽被淘汰后重入应为全新会话(无历史)"
        );
    }

    /// 同一记忆槽的并发 invoke 必须串行化:读→LLM→写整段持锁,避免并发都读到
    /// 旧历史、写回互相覆盖丢上下文(M2b)。
    #[tokio::test]
    async fn concurrent_same_session_invokes_do_not_lose_history() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let blocked = Arc::new(StdMutex::new(false));
        let llm = BlockingChatModel {
            seen: seen.clone(),
            entered: entered.clone(),
            release: release.clone(),
            blocked,
        };
        let pipe = Arc::new(RunnableWithMessageHistory::new(
            llm,
            ConversationBufferMemory::new().with_return_messages(true),
        ));

        // Act 第一轮 invoke:进入模型调用后阻塞,期间必须持有记忆锁。
        let p1 = pipe.clone();
        let h1 = tokio::spawn(async move { p1.invoke("第一轮".to_string(), None).await });
        entered.notified().await;

        // 第二轮 invoke:若读→LLM→写没有整段持锁,此刻会读到空历史(丢上下文)。
        let p2 = pipe.clone();
        let h2 = tokio::spawn(async move { p2.invoke("第二轮".to_string(), None).await });

        // 放行第一轮:写回后才释放锁,第二轮才能读到第一轮完整对话。
        release.notify_one();
        let r1 = h1.await.unwrap().unwrap();
        let r2 = h2.await.unwrap().unwrap();
        assert_eq!(r1.content, "reply to: 第一轮");
        assert_eq!(r2.content, "reply to: 第二轮");

        // Assert
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[1].len(),
            3,
            "M2b: 第二轮应看到第一轮完整对话(user+ai+user),而非空历史"
        );
        assert_eq!(calls[1][0].content, "第一轮");
        assert_eq!(calls[1][2].content, "第二轮");
    }
}
