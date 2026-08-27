// src/language_models/providers/anthropic/impls.rs
//! Trait implementations for AnthropicChat.

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use serde_json::json;
use std::pin::Pin;

use lc_callbacks::{RunTree, RunType};
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::Runnable;
use lc_core::tools::ToolDefinition;
use lc_core::RunnableConfig;
use lc_schema::Message;

use super::chat::AnthropicChat;
use super::error::AnthropicError;
use super::types::AnthropicStreamToken;

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for AnthropicChat {
    type Error = AnthropicError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, config).await
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error>
    {
        use futures_util::StreamExt;

        let model = self.config.model.clone();
        let (temp, max) = crate::sampling::sampling_overrides(&config);
        let mut effective = self.clone();
        if let Some(t) = temp {
            effective.config.temperature = Some(t);
        }
        if let Some(m) = max {
            effective.config.max_tokens = m;
        }
        let token_stream = effective.stream_chat_internal(input).await?;

        // H4: True streaming — emit one LLMResult per token
        let stream = token_stream.map(move |token_result| match token_result {
            Ok(AnthropicStreamToken::Thinking(t)) => Ok(LLMResult {
                content: String::new(),
                model: model.clone(),
                token_usage: None,
                tool_calls: None,
                thinking_content: Some(t),
            }),
            Ok(AnthropicStreamToken::Text(t)) => Ok(LLMResult {
                content: t,
                model: model.clone(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            }),
            Ok(AnthropicStreamToken::Usage(u)) => Ok(LLMResult {
                content: String::new(),
                model: model.clone(),
                token_usage: Some(TokenUsage {
                    prompt_tokens: u.input_tokens,
                    completion_tokens: u.output_tokens,
                    total_tokens: u.input_tokens + u.output_tokens,
                }),
                tool_calls: None,
                thinking_content: None,
            }),
            Err(e) => Err(e),
        });

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for AnthropicChat {
    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        lc_core::token_counter::count_tokens(text).unwrap_or_else(|e| {
            // If the encoder fails to load, overestimate by byte length (better slightly high than silently counting 0, which would mislead routing/truncation)
            log::warn!("Token counting failed, falling back to byte-length estimation: {e}");
            text.len()
        })
    }

    fn temperature(&self) -> Option<f32> {
        self.config.temperature
    }

    fn max_tokens(&self) -> Option<usize> {
        Some(self.config.max_tokens)
    }

    fn with_temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }

    fn with_max_tokens(mut self, max: usize) -> Self {
        self.config.max_tokens = max;
        self
    }
}

#[async_trait]
impl BaseChatModel for AnthropicChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let run_name = config
            .as_ref()
            .and_then(|c| c.run_name.clone())
            .unwrap_or_else(|| format!("{}:chat", self.config.model));

        let mut run = RunTree::new(
            run_name,
            RunType::Llm,
            json!({
                "messages": messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>(),
                "model": self.config.model,
            }),
        );

        if let Some(ref cfg) = config {
            for tag in &cfg.tags {
                run = run.with_tag(tag.clone());
            }
            for (key, value) in &cfg.metadata {
                run = run.with_metadata(key.clone(), value.clone());
            }
        }

        if let Some(ref cfg) = config {
            if let Some(ref callbacks) = cfg.callbacks {
                for handler in callbacks.handlers() {
                    handler.on_llm_start(&run, &messages).await;
                }
            }
        }

        let (temp, max) = crate::sampling::sampling_overrides(&config);
        let mut effective = self.clone();
        if let Some(t) = temp {
            effective.config.temperature = Some(t);
        }
        if let Some(m) = max {
            effective.config.max_tokens = m;
        }
        let result = effective.chat_internal(messages.clone()).await;

        match result {
            Ok(response) => {
                run.end(json!({
                    "content": &response.content,
                    "model": &response.model,
                    "token_usage": &response.token_usage,
                    "thinking_content": &response.thinking_content,
                }));

                if let Some(ref cfg) = config {
                    if let Some(ref callbacks) = cfg.callbacks {
                        for handler in callbacks.handlers() {
                            handler.on_llm_end(&run, &response.content).await;
                        }
                    }
                }

                Ok(response)
            }
            Err(e) => {
                run.end_with_error(e.to_string());

                if let Some(ref cfg) = config {
                    if let Some(ref callbacks) = cfg.callbacks {
                        for handler in callbacks.handlers() {
                            handler.on_llm_error(&run, &e.to_string()).await;
                        }
                    }
                }

                Err(e)
            }
        }
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        let run_name = config
            .as_ref()
            .and_then(|c| c.run_name.clone())
            .unwrap_or_else(|| format!("{}:stream", self.config.model));

        let run = RunTree::new(
            run_name,
            RunType::Llm,
            json!({
                "messages": messages.len(),
                "model": self.config.model,
            }),
        );

        if let Some(ref cfg) = config {
            if let Some(ref callbacks) = cfg.callbacks {
                for handler in callbacks.handlers() {
                    handler.on_llm_start(&run, &messages).await;
                }
            }
        }

        let (temp, max) = crate::sampling::sampling_overrides(&config);
        let mut effective = self.clone();
        if let Some(t) = temp {
            effective.config.temperature = Some(t);
        }
        if let Some(m) = max {
            effective.config.max_tokens = m;
        }
        let stream = effective.stream_chat_internal(messages).await?;

        let callbacks = config.and_then(|c| c.callbacks);
        let stream = stream.then(move |token_result| {
            let cbs = callbacks.clone();
            let run = run.clone();
            async move {
                match &token_result {
                    Ok(AnthropicStreamToken::Text(token)) => {
                        if let Some(ref cbs) = cbs {
                            for handler in cbs.handlers() {
                                handler.on_llm_new_token(&run, token).await;
                            }
                        }
                    }
                    Ok(AnthropicStreamToken::Thinking(thinking)) => {
                        if let Some(ref cbs) = cbs {
                            for handler in cbs.handlers() {
                                handler.on_llm_thinking(&run, thinking).await;
                            }
                        }
                    }
                    Ok(AnthropicStreamToken::Usage(_)) => {}
                    Err(_) => {}
                }
                token_result
            }
        });

        // Flatten: emit Text tokens as Ok(StreamChunk), drop Thinking tokens
        // from the stream, forward Usage as a usage-carrying chunk.
        let stream = stream.flat_map(|token_result| {
            futures_util::stream::iter(match token_result {
                Ok(AnthropicStreamToken::Text(token)) => vec![Ok(StreamChunk::new(token))],
                Ok(AnthropicStreamToken::Thinking(_)) => vec![],
                Ok(AnthropicStreamToken::Usage(usage)) => vec![Ok(StreamChunk {
                    text: String::new(),
                    token_usage: Some(TokenUsage {
                        prompt_tokens: usage.input_tokens,
                        completion_tokens: usage.output_tokens,
                        total_tokens: usage.input_tokens + usage.output_tokens,
                    }),
                })],
                Err(e) => vec![Err(e)],
            })
        });

        Ok(Box::pin(stream))
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        // Expose the inherent tool-binding capability at the trait level so it
        // survives being wrapped by `ChatModelWrapper` / `LLMClient` (Q1).
        Some(Box::new(self.bind_tools(tools)))
    }
}
