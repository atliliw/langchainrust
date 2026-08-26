// lc-sessions/src/session_runnable.rs
//! SessionManagerRunnable — 持久会话的 LCEL Runnable 适配器

use async_trait::async_trait;
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use lc_core::BaseChatModel;
use std::sync::Arc;

use super::manager::SessionManager;

/// 把 [`SessionManager`] 的"会话内对话"包成 `Runnable<(session_id, 用户消息), 回复文本>`。
///
/// 输入是一个二元组 `(String, String)`:会话 id + 用户消息;输出是 LLM 回复文本。
/// 会话由 `SessionManager` 的 `SessionStore` 持久化管理 —— 调用前需先用
/// [`SessionManager::create_session`] 创建会话,`invoke` 时同一会话 id 的
/// 多轮对话自动累积历史。
///
/// # 与 `RunnableWithMessageHistory` 的分工
///
/// [`RunnableWithMessageHistory`](lc_memory::RunnableWithMessageHistory) 是
/// "LLM + `BaseMemory`" 的会话 Runnable:用 `config` 里的 `session_id` 定位,
/// 每轮自动读/写记忆,返回 `LLMResult`;它是内存层的能力。本适配器则是
/// **持久会话存储**层的入口:显式管理会话生命周期(创建/归档/清除),
/// 返回 String 回复。需要完整会话生命周期时用本类型;只需要"带记忆的
/// 单轮 LLM"时用 `RunnableWithMessageHistory`。
///
/// # Example
///
/// ```rust,ignore
/// use lc_sessions::{SessionManager, MemorySessionStore, SessionManagerRunnable};
/// use lc_core::BaseChatModel;
/// use std::sync::Arc;
///
/// let manager = Arc::new(SessionManager::new(Arc::new(MemorySessionStore::new())));
/// let llm = Arc::new(openai_chat_model); // 任意实现 BaseChatModel 的模型
/// let step = SessionManagerRunnable::new(manager.clone(), llm);
/// let chain = step.pipe(prompt).pipe(parser); // 可进 RunnableSequence
/// ```
pub struct SessionManagerRunnable<L: BaseChatModel> {
    manager: Arc<SessionManager>,
    llm: Arc<L>,
}

impl<L: BaseChatModel> SessionManagerRunnable<L> {
    /// 创建会话 Runnable。
    ///
    /// `manager` 承载持久会话存储与生命周期;`llm` 负责实际对话生成。
    pub fn new(manager: Arc<SessionManager>, llm: Arc<L>) -> Self {
        Self { manager, llm }
    }
}

#[async_trait]
impl<L> Runnable<(String, String), String> for SessionManagerRunnable<L>
where
    L: BaseChatModel,
    L::Error: std::fmt::Display,
{
    type Error = LcelError;

    async fn invoke(
        &self,
        (session_id, user_message): (String, String),
        _config: Option<RunnableConfig>,
    ) -> Result<String, LcelError> {
        self.manager
            .chat(&session_id, self.llm.as_ref(), user_message)
            .await
            .map_err(|e| LcelError::Chain(format!("session chat: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemorySessionStore, SessionManager};
    use lc_core::language_models::{BaseLanguageModel, LLMResult, StreamChunk};
    use lc_core::runnables::{RunnableExt, RunnableLambda};
    use lc_schema::Message;
    use std::pin::Pin;
    use tokio::sync::Mutex;

    #[derive(Clone)]
    struct MockChat {
        response: String,
        received: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    impl MockChat {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
                received: Arc::new(Mutex::new(Vec::new())),
            }
        }

        async fn received(&self) -> Vec<Vec<Message>> {
            self.received.lock().await.clone()
        }
    }

    #[derive(Debug)]
    struct MockError(String);

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for MockError {}

    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockChat {
        fn model_name(&self) -> &str {
            "mock-chat"
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
    impl Runnable<Vec<Message>, LLMResult> for MockChat {
        type Error = MockError;

        async fn invoke(
            &self,
            input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.received.lock().await.push(input);
            Ok(LLMResult {
                content: self.response.clone(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseChatModel for MockChat {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invoke(messages, config).await
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<
            Pin<Box<dyn futures_util::Stream<Item = Result<StreamChunk, Self::Error>> + Send>>,
            Self::Error,
        > {
            unimplemented!("stream_chat not exercised in session_runnable tests")
        }
    }

    fn test_step() -> (
        Arc<SessionManager>,
        SessionManagerRunnable<MockChat>,
        MockChat,
    ) {
        let manager = Arc::new(SessionManager::new(Arc::new(MemorySessionStore::new())));
        let llm = MockChat::new("reply");
        let step = SessionManagerRunnable::new(manager.clone(), Arc::new(llm.clone()));
        (manager, step, llm)
    }

    /// E2 验证:同一会话 id 连续两轮对话,第二轮能看到第一轮的累积历史。
    #[tokio::test]
    async fn session_runnable_accumulates_two_turns() {
        let (manager, step, llm) = test_step();
        let session_id = manager.create_session().await.unwrap();

        let reply1 = step
            .invoke((session_id.clone(), "第一句".to_string()), None)
            .await
            .unwrap();
        let reply2 = step
            .invoke((session_id.clone(), "第二句".to_string()), None)
            .await
            .unwrap();
        assert_eq!(reply1, "reply");
        assert_eq!(reply2, "reply");

        let received = llm.received().await;
        assert_eq!(received.len(), 2);
        assert_eq!(
            received[0].len(),
            1,
            "first turn sees only the user message"
        );
        assert_eq!(received[0][0].content, "第一句");
        assert_eq!(
            received[1].len(),
            3,
            "second turn must see the accumulated history"
        );
        assert_eq!(received[1][0].content, "第一句");
        assert_eq!(received[1][1].content, "reply");
        assert_eq!(received[1][2].content, "第二句");
    }

    /// 不同的会话 id 互不影响:历史按会话隔离。
    #[tokio::test]
    async fn session_runnable_isolates_different_sessions() {
        let (manager, step, llm) = test_step();
        let a = manager.create_session().await.unwrap();
        let b = manager.create_session().await.unwrap();

        step.invoke((a.clone(), "A 的消息".to_string()), None)
            .await
            .unwrap();
        step.invoke((b.clone(), "B 的消息".to_string()), None)
            .await
            .unwrap();

        let received = llm.received().await;
        assert_eq!(received.len(), 2);
        assert_eq!(received[0].len(), 1);
        assert_eq!(
            received[1].len(),
            1,
            "different sessions must not share history"
        );
    }

    /// 会话不存在时返回错误(映射为 LcelError,不 panic)。
    #[tokio::test]
    async fn session_runnable_reports_missing_session() {
        let (_, step, _) = test_step();
        let err = step
            .invoke(("no-such-session".to_string(), "hi".to_string()), None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("session"), "got: {err}");
    }

    /// E2 验证:会话 Runnable 能作为 LCEL 链的一步参与 `pipe` 组合。
    #[tokio::test]
    async fn session_runnable_pipes_into_sequence() {
        let (manager, step, _) = test_step();
        let session_id = manager.create_session().await.unwrap();

        let len = step.pipe(RunnableLambda::new_sync(|reply: String| reply.len()));
        let n = len
            .invoke((session_id, "hi".to_string()), None)
            .await
            .unwrap();
        assert_eq!(n, "reply".len());
    }
}
