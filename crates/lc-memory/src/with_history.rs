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
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// 记忆句柄:任意 `BaseMemory` 的共享可变引用。
pub type SharedMemory = Arc<Mutex<Box<dyn BaseMemory>>>;

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
        cache: Mutex<HashMap<String, SharedMemory>>,
        /// 仅用于 `memory()` 访问器的占位记忆(Sessions 模式下实际槽不唯一)。
        default: SharedMemory,
    },
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
                cache: Mutex::new(HashMap::new()),
                default: Arc::new(Mutex::new(Box::new(
                    ConversationBufferMemory::new(),
                ))),
            },
        }
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
            HistoryMode::Sessions { factory, cache, .. } => {
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
                Ok(cache
                    .entry(session_id.to_string())
                    .or_insert_with(|| factory(session_id))
                    .clone())
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

        // 1. 读记忆 → 转消息(锁在 LLM 调用前释放,不占着锁等网络)
        let mut messages = {
            let memory = memory.lock().await;
            let vars = memory
                .load_memory_variables(&HashMap::new())
                .await
                .map_err(|e| LcelError::Chain(format!("load memory: {e}")))?;
            memory_variables_to_messages(&vars)
        };

        // 2. 拼上用户输入
        messages.push(Message::human(&input));

        // 3. 调 LLM
        let result = self
            .llm
            .chat(messages, config)
            .await
            .map_err(Into::into)?;

        // 4. 写回记忆:失败不丢弃模型答案(否则调用方拿 Err 重试会重复调 LLM),
        //    记 warn 暴露记忆层降级
        {
            let mut memory = memory.lock().await;
            let inputs = HashMap::from([("input".to_string(), input)]);
            let outputs = HashMap::from([("output".to_string(), result.content.clone())]);
            if let Err(e) = memory.save_context(&inputs, &outputs).await {
                log::warn!("记忆写回失败(模型答案仍照常返回): {e}");
            }
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
}
