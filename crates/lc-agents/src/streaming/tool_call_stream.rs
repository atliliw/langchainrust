//! StreamingFunctionCallingAgent - streaming output agent

use std::pin::Pin;
use std::sync::Arc;

use futures_util::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use lc_core::language_models::BaseChatModel;
use lc_core::runnables::RunnableConfig;
use lc_providers::ProviderError;
use lc_schema::Message;

use super::state::AgentStreamEvent;

/// Streaming Function Calling Agent
///
/// Streams LLM text (token by token), then emits FinalAnswer at the end.
/// Tool-call state is exposed via `AgentStreamEvent::ToolCall`.
/// Works with any LLM provider implementing `BaseChatModel`.
pub struct StreamingFunctionCallingAgent {
    llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
}

impl StreamingFunctionCallingAgent {
    /// Creates a new Streaming Function Calling Agent
    ///
    /// # Backward compatibility
    /// Old code `StreamingFunctionCallingAgent::new(openai_chat)` still works.
    pub fn new<L>(llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: lc_providers::wrap_chat_model(llm),
        }
    }

    /// Creates an agent from a wrapped `Arc<dyn BaseChatModel>`
    pub fn from_arc(llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>) -> Self {
        Self { llm }
    }

    /// Streams execution: returns an event stream
    pub async fn invoke_stream(
        &self,
        input: String,
    ) -> Pin<Box<dyn Stream<Item = AgentStreamEvent> + Send>> {
        self.invoke_stream_with_config(input, None).await
    }

    /// Streams execution with a [`RunnableConfig`] so the streamed LLM call
    /// emits `on_llm_start/end` to the configured callbacks/OTel backend
    /// (T6, v0.23.0).
    pub async fn invoke_stream_with_config(
        &self,
        input: String,
        config: Option<&RunnableConfig>,
    ) -> Pin<Box<dyn Stream<Item = AgentStreamEvent> + Send>> {
        let (tx, rx) = mpsc::channel(32);
        let llm = self.llm.clone();
        let messages = vec![Message::human(input)];
        // The spawned task is `'static`, so the config must be owned here.
        let config = config.cloned();

        tokio::spawn(async move {
            let mut stream = match llm.stream_chat(messages, config).await {
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
                    Ok(chunk) => {
                        full.push_str(&chunk.text);
                        if tx
                            .send(AgentStreamEvent::Text {
                                content: chunk.text,
                            })
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
    use lc_providers::{OpenAIChat, OpenAIConfig};

    #[test]
    fn test_new() {
        let llm = OpenAIChat::new(OpenAIConfig::default());
        let _agent = StreamingFunctionCallingAgent::new(llm);
    }
}
