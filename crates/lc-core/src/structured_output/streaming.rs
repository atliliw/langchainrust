// src/core/structured_output/streaming.rs
//! Streaming structured output support for incremental LLM response parsing.

use async_trait::async_trait;
use futures_util::Stream;
use futures_util::StreamExt;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::pin::Pin;

use crate::language_models::BaseChatModel;
use lc_schema::Message;

use super::extract::{build_structured_system_prompt, StructuredOutputError};
use super::parser::{PartialJsonError, PartialJsonParser};

/// Trait that extends `BaseChatModel` with streaming structured output capabilities.
///
/// Provides a streaming variant of `with_structured_output` that yields partial
/// `T` values as the model generates tokens. The target type `T` should derive
/// `#[serde(default)]` or use `Option` fields so that partial JSON can be
/// deserialized with missing fields filled by defaults.
#[async_trait]
pub trait StreamingStructuredOutputExt: BaseChatModel {
    /// Stream structured output from a chat model.
    ///
    /// Returns a stream of `T` values. Each item represents the best partial
    /// result that could be parsed from the tokens received so far. The final
    /// item in the stream is the complete result.
    ///
    /// # Arguments
    ///
    /// * `schema` - A JSON Schema describing the expected output shape.
    /// * `prompt` - The user prompt to send to the LLM.
    ///
    /// # Returns
    ///
    /// A stream of `Result<T, StructuredOutputError>` items.
    async fn stream_structured_output<T>(
        &self,
        schema: Value,
        prompt: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<T, StructuredOutputError>> + Send>>,
        StructuredOutputError,
    >
    where
        T: DeserializeOwned + Serialize + Clone + PartialEq + Unpin + Send + Sync + 'static,
    {
        stream_structured_output(self, schema, prompt).await
    }
}

/// Blanket implementation: every `BaseChatModel` automatically gets
/// `StreamingStructuredOutputExt`.
impl<M: BaseChatModel> StreamingStructuredOutputExt for M {}

/// Stream structured output from a chat model.
///
/// Returns a stream of partial `T` values as the model generates tokens.
/// The implementation:
/// 1. Calls `stream_chat` with a prompt that asks for JSON matching the schema
/// 2. Accumulates tokens through the `PartialJsonParser`
/// 3. At each successful parse, yields a `T` value (partial fields filled by
///    serde defaults, rest default)
/// 4. On stream end, yields the final complete `T`
///
/// # Arguments
///
/// * `llm` - Any type implementing `BaseChatModel`.
/// * `schema` - A JSON Schema describing the expected output.
/// * `prompt` - The user prompt to send to the LLM.
///
/// # Returns
///
/// A `Result` containing a stream of `Result<T, StructuredOutputError>` items,
/// or a `StructuredOutputError` if the stream could not be set up.
///
/// # Type requirements
///
/// The target type `T` should use `#[serde(default)]` or `Option` fields so
/// that partial JSON can be deserialized with missing fields filled by defaults.
/// If `T` does not support default deserialization, partial results will fail
/// and only the final complete result will be yielded.
///
/// # Example
///
/// ```ignore
/// use serde::{Deserialize, Serialize};
/// use langchainrust::core::structured_output::stream_structured_output;
/// use futures_util::StreamExt;
///
/// #[derive(Debug, Deserialize, Serialize, Clone)]
/// #[serde(default)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// impl Default for Person {
///     fn default() -> Self {
///         Self { name: String::new(), age: 0 }
///     }
/// }
///
/// let mut stream = stream_structured_output::<Person, _>(
///     &llm, schema, "Tell me about Alice"
/// ).await?;
/// while let Some(result) = stream.next().await {
///     let person = result?;
///     println!("Partial: name={}, age={}", person.name, person.age);
/// }
/// ```
pub async fn stream_structured_output<T, M>(
    llm: &M,
    schema: Value,
    prompt: &str,
) -> Result<
    Pin<Box<dyn Stream<Item = Result<T, StructuredOutputError>> + Send>>,
    StructuredOutputError,
>
where
    T: DeserializeOwned + Serialize + Clone + PartialEq + Unpin + Send + Sync + 'static,
    M: BaseChatModel + ?Sized,
{
    // Validate the schema is an object
    if !schema.is_object() {
        return Err(StructuredOutputError::SchemaError(format!(
            "Schema must be a JSON object, got: {}",
            schema
        )));
    }

    // Build the system prompt with schema and format instructions
    let system_prompt = build_structured_system_prompt(&schema);

    let messages = vec![Message::system(system_prompt), Message::human(prompt)];

    // Start the stream. We need to erase the model's error type by mapping
    // it to StructuredOutputError before passing to the stream processor.
    let token_stream = llm
        .stream_chat(messages, None)
        .await
        .map_err(|e| StructuredOutputError::LLMError(e.to_string()))?;

    // Map the inner stream: each chunk's text delta becomes the token fed to
    // the partial-JSON parser, and the error type is erased to
    // StructuredOutputError.
    let mapped_stream = token_stream.map(|item| {
        item.map(|chunk| chunk.text)
            .map_err(|e| StructuredOutputError::LLMError(e.to_string()))
    });

    // Box the mapped stream so it has a concrete type for StructuredStreamProcessor
    let boxed: Pin<Box<dyn Stream<Item = Result<String, StructuredOutputError>> + Send>> =
        Box::pin(mapped_stream);

    let output_stream = StructuredStreamProcessor::<T>::new(boxed);

    Ok(Box::pin(output_stream))
}

/// Stream processor that accumulates LLM tokens through a `PartialJsonParser`
/// and yields partial `T` values.
///
/// This is implemented as a manual `Stream` rather than using `stream::unfold`
/// because we need to maintain mutable state (the `PartialJsonParser`) across
/// stream polls.
struct StructuredStreamProcessor<T> {
    inner: Pin<Box<dyn Stream<Item = Result<String, StructuredOutputError>> + Send>>,
    parser: PartialJsonParser,
    last_value: Option<T>,
    done: bool,
}

// Safety: StructuredStreamProcessor is Unpin when T is Unpin because all fields
// are Unpin: Pin<Box<dyn ..>> is Unpin, PartialJsonParser is Unpin,
// Option<T> is Unpin when T is Unpin, and bool is Unpin.
impl<T: Unpin> Unpin for StructuredStreamProcessor<T> {}

impl<T> StructuredStreamProcessor<T>
where
    T: DeserializeOwned + Serialize + Clone + PartialEq + Unpin + Send + Sync + 'static,
{
    fn new(
        inner: Pin<Box<dyn Stream<Item = Result<String, StructuredOutputError>> + Send>>,
    ) -> Self {
        Self {
            inner,
            parser: PartialJsonParser::new(),
            last_value: None,
            done: false,
        }
    }
}

impl<T> Stream for StructuredStreamProcessor<T>
where
    T: DeserializeOwned + Serialize + Clone + PartialEq + Send + Sync + Unpin + 'static,
{
    type Item = Result<T, StructuredOutputError>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // Safety: StructuredStreamProcessor is Unpin because all its fields are Unpin.
        // Pin<Box<dyn Stream + Send>> is Unpin, PartialJsonParser is Unpin,
        // Option<T> is Unpin, and bool is Unpin.
        let this = self.get_mut();

        if this.done {
            return std::task::Poll::Ready(None);
        }

        loop {
            match this.inner.as_mut().poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(token))) => {
                    match this.parser.push_and_parse(&token) {
                        Ok(json_value) => match serde_json::from_value::<T>(json_value) {
                            Ok(value) => {
                                this.last_value = Some(value.clone());
                                return std::task::Poll::Ready(Some(Ok(value)));
                            }
                            Err(_) => {
                                // Partial JSON parsed but cannot deserialize into T yet.
                                // Continue accumulating tokens.
                                continue;
                            }
                        },
                        Err(PartialJsonError::Incomplete(_)) => {
                            // Not enough JSON yet, continue accumulating
                            continue;
                        }
                        Err(PartialJsonError::Invalid(msg)) => {
                            // The accumulated buffer is invalid JSON even after repair.
                            // This can happen with garbage tokens; skip and continue.
                            // We don't terminate the stream for a single bad parse.
                            log::warn!("structured output stream hit invalid JSON fragment (skipped, continuing to accumulate): {msg}");
                            continue;
                        }
                    }
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    // Error from the underlying token stream
                    return std::task::Poll::Ready(Some(Err(e)));
                }
                std::task::Poll::Ready(None) => {
                    // Stream ended. Try to finalize the parser.
                    this.done = true;

                    // Take ownership of the parser to finalize it
                    let parser = std::mem::take(&mut this.parser);

                    match parser.finalize() {
                        Ok(json_value) => match serde_json::from_value::<T>(json_value) {
                            Ok(value) => {
                                // Only yield if this is a new value different from last
                                let is_new = this.last_value.as_ref() != Some(&value);
                                if is_new {
                                    return std::task::Poll::Ready(Some(Ok(value)));
                                }
                                return std::task::Poll::Ready(None);
                            }
                            Err(e) => {
                                return std::task::Poll::Ready(Some(Err(
                                    StructuredOutputError::ParseError(format!(
                                        "Failed to deserialize final JSON into target type: {}",
                                        e
                                    )),
                                )));
                            }
                        },
                        Err(PartialJsonError::Invalid(msg)) => {
                            // If we had a last_value, the stream was still "successful"
                            // in yielding partial results, but the final buffer is invalid.
                            // This can happen if the LLM appended non-JSON text.
                            if this.last_value.is_some() {
                                log::warn!("structured output stream ended with invalid buffer (already returned earlier partial results): {msg}");
                                return std::task::Poll::Ready(None);
                            }
                            return std::task::Poll::Ready(Some(Err(
                                StructuredOutputError::StreamIncomplete(msg),
                            )));
                        }
                        Err(PartialJsonError::Incomplete(msg)) => {
                            if this.last_value.is_some() {
                                log::warn!(
                                    "structured output stream ended with incomplete buffer (already returned earlier partial results): {msg}"
                                );
                                return std::task::Poll::Ready(None);
                            }
                            return std::task::Poll::Ready(Some(Err(
                                StructuredOutputError::StreamIncomplete(msg),
                            )));
                        }
                    }
                }
                std::task::Poll::Pending => {
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}

/// Attempt to deserialize a `serde_json::Value` into `T`, returning `None`
/// if deserialization fails (e.g., missing required fields).
///
/// Only used from `#[cfg(test)]` test helpers, so compile it in for tests only.
#[cfg(test)]
pub(crate) fn try_deserialize_partial<T: DeserializeOwned>(value: Value) -> Option<T> {
    serde_json::from_value::<T>(value).ok()
}
