//! StreamingFunctionCallingAgent - 流式输出 Agent

use std::pin::Pin;
use std::sync::Arc;

use futures_util::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::core::language_models::BaseChatModel;
use crate::error::Error;
use crate::schema::Message;

use super::state::AgentStreamEvent;

/// 流式 Function Calling Agent
///
/// 流式输出 LLM 文本(token),结束后发 FinalAnswer。
/// 工具调用状态通过 `AgentStreamEvent::ToolCall` 暴露。
/// 支持任何实现了 `BaseChatModel` 的 LLM Provider。
pub struct StreamingFunctionCallingAgent {
    llm: Arc<dyn BaseChatModel<Error = Error> + Send + Sync>,
}

impl StreamingFunctionCallingAgent {
    /// 创建新的流式 Function Calling Agent
    ///
    /// # 向后兼容
    /// 旧代码 `StreamingFunctionCallingAgent::new(openai_chat)` 仍然可用。
    pub fn new<L>(llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<Error>,
    {
        Self {
            llm: crate::core::language_models::wrap_chat_model(llm),
        }
    }

    /// 从已包装的 `Arc<dyn BaseChatModel>` 创建 Agent
    pub fn from_arc(llm: Arc<dyn BaseChatModel<Error = Error> + Send + Sync>) -> Self {
        Self { llm }
    }

    /// 流式执行:返回事件流
    pub async fn invoke_stream(
        &self,
        input: String,
    ) -> Pin<Box<dyn Stream<Item = AgentStreamEvent> + Send>> {
        let (tx, rx) = mpsc::channel(32);
        let llm = self.llm.clone();
        let messages = vec![Message::human(input)];

        tokio::spawn(async move {
            let mut stream = match llm.stream_chat(messages, None).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx
                        .send(AgentStreamEvent::Error {
                            message: format!("Stream initialization failed: {}", e),
                        })
                        .await;
                    return;
                }
            };

            let mut full = String::new();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(token) => {
                        full.push_str(&token);
                        if tx
                            .send(AgentStreamEvent::Text { content: token })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(AgentStreamEvent::Error {
                                message: format!("Stream error: {}", e),
                            })
                            .await;
                        break;
                    }
                }
            }

            let _ = tx
                .send(AgentStreamEvent::FinalAnswer { content: full })
                .await;
        });

        Box::pin(ReceiverStream::new(rx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_models::{OpenAIChat, OpenAIConfig};

    #[test]
    fn test_new() {
        let llm = OpenAIChat::new(OpenAIConfig::default());
        let _agent = StreamingFunctionCallingAgent::new(llm);
    }
}
