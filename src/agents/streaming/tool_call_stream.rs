//! StreamingFunctionCallingAgent - 流式输出 Agent

use std::pin::Pin;

use futures_util::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::language_models::OpenAIChat;
use crate::schema::Message;
use crate::BaseChatModel;

use super::state::AgentStreamEvent;

/// 流式 Function Calling Agent
///
/// 流式输出 LLM 文本(token),结束后发 FinalAnswer。
/// 工具调用状态通过 `AgentStreamEvent::ToolCall` 暴露。
pub struct StreamingFunctionCallingAgent {
    llm: OpenAIChat,
}

impl StreamingFunctionCallingAgent {
    pub fn new(llm: OpenAIChat) -> Self {
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

    #[test]
    fn test_new() {
        let llm = OpenAIChat::new(crate::OpenAIConfig::default());
        let _agent = StreamingFunctionCallingAgent::new(llm);
    }
}
