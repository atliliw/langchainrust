//! `ReplayProvider`:从录制文件按 FIFO 顺序回放,零网络。
//!
//! 回放**不做消息匹配**(LLMChain 渲染出的 prompt 逐次可变),只按顺序弹出录播。
//! 队列耗尽返回 [`TestkitError::ReplayExhausted`]。

use std::collections::VecDeque;
use std::io::BufRead;
use std::path::Path;
use std::pin::Pin;
use std::sync::Mutex;

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_schema::Message;

use crate::error::TestkitError;
use crate::recording::RecordedExchange;

/// 从录制文件回放的零网络 `BaseChatModel`。
pub struct ReplayProvider {
    queue: Mutex<VecDeque<RecordedExchange>>,
    model_name: String,
}

impl ReplayProvider {
    /// 读 JSONL 录制文件(缺文件/坏行 → `Err`)。
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TestkitError> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut queue = VecDeque::new();
        for line in reader.lines() {
            let line = line?.trim().to_string();
            if line.is_empty() {
                continue;
            }
            let exchange: RecordedExchange = serde_json::from_str(&line).map_err(|e| {
                TestkitError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid recording line: {e}"),
                ))
            })?;
            queue.push_back(exchange);
        }
        Ok(Self {
            queue: Mutex::new(queue),
            model_name: "replay".to_string(),
        })
    }

    /// 内存构造(手写录播 = MockProvider 的等价物)。
    pub fn from_exchanges(exchanges: Vec<RecordedExchange>) -> Self {
        Self {
            queue: Mutex::new(exchanges.into()),
            model_name: "replay".to_string(),
        }
    }

    /// 单一固定响应:任意请求都返回同一 `response`(最简 mock)。
    pub fn single(response: LLMResult) -> Self {
        Self::from_exchanges(vec![RecordedExchange {
            messages: Vec::new(),
            response,
        }])
    }

    /// 剩余录播条数。
    pub fn len(&self) -> usize {
        self.queue.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// 是否已无剩余录播。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for ReplayProvider {
    type Error = TestkitError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, config).await
    }
}

impl BaseLanguageModel<Vec<Message>, LLMResult> for ReplayProvider {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        // 估算:约 4 字符/ token。
        text.chars().count() / 4 + 1
    }

    fn temperature(&self) -> Option<f32> {
        None
    }

    fn max_tokens(&self) -> Option<usize> {
        None
    }

    fn with_temperature(self, _temp: f32) -> Self {
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self {
        self
    }
}

#[async_trait]
impl BaseChatModel for ReplayProvider {
    async fn chat(
        &self,
        messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let Some(exchange) = queue.pop_front() else {
            return Err(TestkitError::ReplayExhausted {
                requested: messages.len(),
            });
        };
        Ok(exchange.response)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        let response = self.chat(messages, config).await?;
        let stream = futures_util::stream::iter(vec![Ok(response.content)]);
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::language_models::TokenUsage;

    fn exchange(content: &str) -> RecordedExchange {
        RecordedExchange {
            messages: vec![Message::system("ping")],
            response: LLMResult {
                content: content.to_string(),
                model: "replay".to_string(),
                token_usage: Some(TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    total_tokens: 3,
                }),
                ..Default::default()
            },
        }
    }

    #[tokio::test]
    async fn single_returns_fixed_response_for_any_request() {
        let provider = ReplayProvider::single(exchange("hello").response);
        let result = provider
            .chat(vec![Message::system("any")], None)
            .await
            .unwrap();
        assert_eq!(result.content, "hello");
    }

    #[tokio::test]
    async fn replay_is_fifo_ordered() {
        let provider = ReplayProvider::from_exchanges(vec![exchange("first"), exchange("second")]);
        let first = provider
            .chat(vec![Message::system("a")], None)
            .await
            .unwrap();
        let second = provider
            .chat(vec![Message::system("b")], None)
            .await
            .unwrap();
        assert_eq!(first.content, "first");
        assert_eq!(second.content, "second");
    }

    #[tokio::test]
    async fn replay_exhausted_returns_error() {
        let provider = ReplayProvider::from_exchanges(vec![exchange("only")]);
        provider
            .chat(vec![Message::system("a")], None)
            .await
            .unwrap();
        let err = provider
            .chat(vec![Message::system("b")], None)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            TestkitError::ReplayExhausted { requested: 1 }
        ));
    }
}
