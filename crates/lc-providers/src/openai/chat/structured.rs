// lc-providers/src/openai/chat/structured.rs
//! Structured output support for the OpenAI chat provider.

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::marker::PhantomData;

use crate::openai::OpenAIConfig;
use lc_core::http::HttpClient;
use lc_core::tools::StructuredOutput;
use lc_schema::Message;

use super::{OpenAIChat, OpenAIError};

/// Method for structured output calls
pub struct StructuredOutputMethod<T: DeserializeOwned + JsonSchema> {
    pub(crate) config: OpenAIConfig,
    pub(crate) http_api: HttpClient,
    pub(crate) http_sse: HttpClient,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned + JsonSchema> StructuredOutputMethod<T> {
    /// Invokes the chat API and parses the response into the structured type `T`.
    pub async fn invoke(&self, messages: Vec<Message>) -> Result<T, OpenAIError> {
        let chat = OpenAIChat {
            config: self.config.clone(),
            http_api: self.http_api.clone(),
            http_sse: self.http_sse.clone(),
        };

        let result = chat.chat_internal(messages).await?;
        let structured = StructuredOutput::<T>::new(result);
        structured
            .parse()
            .map_err(|e| OpenAIError::Parse(e.to_string()))
    }
}
