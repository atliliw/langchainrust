// lc-memory/src/with_history.rs
//! LCEL 组合用的"带记忆的 LLM 封装"。
//!
//! 把"读记忆 → 拼用户输入 → 调 LLM → 写回记忆"这一组合封装成单个
//! `Runnable<String, LLMResult>`,让"LLM + 记忆"可以直接进入 LCEL 管道
//! (pipe 成链、batch、stream 等),不用在业务代码里手写记忆胶水。

use crate::base::{memory_variables_to_messages, BaseMemory};
use async_trait::async_trait;
use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use lc_schema::Message;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// 带记忆的 LLM 封装,作为单个 Runnable 参与 LCEL 组合。
///
/// # 语义
///
/// `invoke(user_input)` 依次执行:
/// 1. 从记忆读取历史,转成消息(`memory_variables_to_messages`);
/// 2. 把用户输入作为 Human 消息追加到末尾;
/// 3. 交给 `llm.chat`(可选 `RunnableConfig` 透传);
/// 4. 把「用户输入 / 模型回答」写回记忆;
/// 5. 返回完整 `LLMResult`。
///
/// LLM 错误通过 `L::Error: Into<LcelError>` 进入管道错误;记忆读写错误
/// 收敛为 `LcelError::Chain`。
///
/// # 泛型
///
/// `L` 是任意实现 `BaseChatModel` 的模型(原生 Provider / `LLMClient` 均可),
/// 只要其错误类型能转进 `LcelError`(`LLMClient` 天然满足;原生 Provider 见
/// lc-providers 的 `From<...> for LcelError`)。记忆以 trait 对象持有,任意
/// `BaseMemory`(Buffer / Window / Summary / SummaryBuffer 等)都可用。
pub struct RunnableWithMessageHistory<L> {
    llm: Arc<L>,
    memory: Arc<Mutex<Box<dyn BaseMemory>>>,
}

impl<L> RunnableWithMessageHistory<L> {
    /// 用 LLM + 记忆构造封装。
    pub fn new(llm: L, memory: impl BaseMemory + 'static) -> Self {
        Self {
            llm: Arc::new(llm),
            memory: Arc::new(Mutex::new(Box::new(memory))),
        }
    }

    /// 暴露内部记忆句柄,便于读取已保存的历史(调试、展示、验证写回等)。
    pub fn memory(&self) -> Arc<Mutex<Box<dyn BaseMemory>>> {
        self.memory.clone()
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
        // 1. 读记忆 → 转消息(锁在 LLM 调用前释放,不占着锁等网络)
        let mut messages = {
            let memory = self.memory.lock().await;
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
            let mut memory = self.memory.lock().await;
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
    use crate::buffer::ConversationBufferMemory;
    use futures_util::Stream;
    use lc_core::language_models::{BaseLanguageModel, BaseChatModel};
    use lc_schema::MessageType;
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
}
