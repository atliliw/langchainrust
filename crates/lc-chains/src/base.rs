// lc-chains/src/base.rs
//! Chain base trait.

use async_trait::async_trait;
use futures_util::{stream, Stream, StreamExt};
use lc_callbacks::{CallbackManager, RunTree, RunType};
use lc_core::runnables::RunnableConfig;
use lc_schema::Message;
use lc_shared::document::Document;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Chain error type.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ChainError {
    /// Missing input.
    #[error("Missing input: {0}")]
    MissingInput(String),

    /// Input error (present but malformed — e.g. a document that fails to deserialize).
    #[error("Input error: {0}")]
    InputError(String),

    /// Output error.
    #[error("Output error: {0}")]
    OutputError(String),

    /// Execution error.
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Stream error.
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Other error.
    #[error("Chain error: {0}")]
    Other(String),

    /// Nested (sub-chain / LLM) execution error, preserving the original error
    /// chain.
    ///
    /// P2-1: composite chains (SequentialChain / RouterChain) wrap sub-chain or
    /// LLM failures in this variant so the underlying error stays inspectable
    /// via `source()` (e.g. downcast back to a concrete `ChainError` variant)
    /// instead of being flattened into a string by `format!`.
    #[error("{context}: {source}")]
    Nested {
        /// Human-readable context describing which step/chain failed.
        context: String,
        /// The underlying error, preserved for chaining.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Chain execution result.
pub type ChainResult = HashMap<String, Value>;

/// Stream output item: token-by-token output.
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// Token text.
    pub token: String,
    /// Whether this is the final token.
    pub is_final: bool,
}

/// Chain stream output type.
pub type ChainStream = Pin<Box<dyn Stream<Item = Result<StreamToken, ChainError>> + Send>>;

/// Convert memory variables (from `BaseMemory::load_memory_variables`) into a
/// message list for LLM consumption.
///
/// Memory implementations produce two shapes under their variable keys:
/// - `Value::Array` of serialized [`Message`] objects (`return_messages = true`)
/// - `Value::String` rendered history (return_messages = false, summary, vectorstore)
///
/// String-form history is wrapped as a `System` message, matching the convention
/// used by `ConversationSummaryMemory` (summary.rs wraps its buffer as System).
pub(crate) fn variables_to_messages(vars: &HashMap<String, Value>) -> Vec<Message> {
    // P2-1: 统一收敛到 lc-memory 的公共转换,避免两处实现漂移。
    lc_memory::memory_variables_to_messages(vars)
}

/// Deserialize a `documents` input array into `Vec<Document>`, failing loudly
/// when any item cannot be parsed instead of silently dropping it.
///
/// P1-2: replaces the old `filter_map(|v| serde_json::from_value(v.clone()).ok())`
/// pattern, which hid malformed entries from the caller. A missing `documents`
/// key, or a present-but-malformed array, yields [`ChainError::MissingInput`] /
/// [`ChainError::InputError`] respectively.
pub(crate) fn documents_from_input(value: Option<&Value>) -> Result<Vec<Document>, ChainError> {
    let arr = value
        .and_then(|v| v.as_array())
        .ok_or_else(|| ChainError::MissingInput("documents".to_string()))?;

    let mut docs = Vec::with_capacity(arr.len());
    let mut failed = 0usize;
    for item in arr {
        match serde_json::from_value::<Document>(item.clone()) {
            Ok(doc) => docs.push(doc),
            Err(_) => failed += 1,
        }
    }
    if failed > 0 {
        return Err(ChainError::InputError(format!(
            "document deserialization failed: {failed} of {} document(s) lost",
            arr.len()
        )));
    }
    Ok(docs)
}

/// Serialize documents for the `source_documents` output key, failing loudly
/// instead of silently inserting `Value::Null` entries.
pub(crate) fn documents_to_values(documents: &[Document]) -> Result<Vec<Value>, ChainError> {
    documents
        .iter()
        .map(|doc| {
            serde_json::to_value(doc)
                .map_err(|e| ChainError::Other(format!("failed to serialize document: {e}")))
        })
        .collect()
}

/// Base Chain trait.
///
/// Chain is LangChain's core abstraction, representing a sequence of operations.
#[async_trait]
pub trait BaseChain: Send + Sync {
    /// Get input keys.
    fn input_keys(&self) -> Vec<&str>;

    /// Get output keys.
    fn output_keys(&self) -> Vec<&str>;

    /// Execute the Chain.
    ///
    /// # Arguments
    /// * `inputs` - Input parameter dictionary
    ///
    /// # Returns
    /// Output result dictionary
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError>;

    /// Execute the Chain with a RunnableConfig.
    ///
    /// This method propagates callbacks through the chain execution.
    /// The default implementation wraps `invoke()` with `on_chain_start` /
    /// `on_chain_end` (or `on_chain_error`) dispatch, so config.callbacks are
    /// never silently dropped — even for chains that don't override this method.
    ///
    /// Chains that want to propagate LLM callbacks (on_llm_start/end) or thread
    /// config into sub-chains (SequentialChain / RouterChain) override this method.
    async fn invoke_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainResult, ChainError> {
        run_chain_with_callbacks(self.name(), inputs, config, |inputs| async move {
            self.invoke(inputs).await
        })
        .await
    }

    /// Stream execute the Chain -- token by token output.
    ///
    /// Default implementation wraps the invoke result as a single-element stream.
    /// Chains that support LLM streaming (LLMChain / ConversationChain) should
    /// override this method, calling `BaseChatModel::stream_chat` internally.
    ///
    /// # Arguments
    /// * `inputs` - Input parameter dictionary
    ///
    /// # Returns
    /// Token stream
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        // Default: wrap invoke result as single-element stream
        let result = self.invoke(inputs).await?;
        // P2-2: a chain that produces no string output now fails loudly instead
        // of silently streaming a single empty token (`unwrap_or("")`).
        let output_text = result
            .values()
            .next()
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ChainError::OutputError("chain produced no string output to stream".to_string())
            })?
            .to_string();
        let stream = futures_util::stream::once(async move {
            Ok(StreamToken {
                token: output_text,
                is_final: true,
            })
        });
        Ok(Box::pin(stream))
    }

    /// Stream execute the Chain with a RunnableConfig.
    ///
    /// The default implementation wraps `stream()` with `on_chain_start` and
    /// dispatches `on_chain_end` (or `on_chain_error`) once the token stream
    /// completes, so config.callbacks are never silently dropped.
    async fn stream_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainStream, ChainError> {
        stream_chain_with_callbacks(self.name(), inputs, config, |inputs| async move {
            self.stream(inputs).await
        })
        .await
    }

    /// Validate inputs.
    fn validate_inputs(&self, inputs: &HashMap<String, Value>) -> Result<(), ChainError> {
        for key in self.input_keys() {
            if !inputs.contains_key(key) {
                return Err(ChainError::MissingInput(key.to_string()));
            }
        }
        Ok(())
    }

    /// Get Chain name.
    fn name(&self) -> &str {
        "chain"
    }
}

/// Run a chain body wrapped in `on_chain_start` → body → `on_chain_end` /
/// `on_chain_error` callback dispatch.
///
/// Shared by the default `invoke_with_config` and by composite chains
/// (SequentialChain / RouterChain) that override it to thread `config` into
/// their sub-chains. Callbacks are only dispatched when `config` carries one;
/// otherwise this is a plain pass-through so the common path pays no tracing.
pub(crate) async fn run_chain_with_callbacks<F, Fut>(
    name: &str,
    inputs: HashMap<String, Value>,
    config: Option<RunnableConfig>,
    body: F,
) -> Result<ChainResult, ChainError>
where
    F: FnOnce(HashMap<String, Value>) -> Fut,
    Fut: Future<Output = Result<ChainResult, ChainError>> + Send,
{
    let callbacks = config.as_ref().and_then(|c| c.callbacks.clone());
    let mut run = RunTree::new(name, RunType::Chain, json!({ "inputs": inputs }));

    if let Some(ref cb) = callbacks {
        cb.dispatch_chain_start(&run, &run.inputs).await;
    }

    let result = body(inputs).await;

    match result {
        Ok(output) => {
            run.end(json!({ "output": output }));
            if let Some(ref cb) = callbacks {
                cb.dispatch_chain_end(&run, &json!({ "output": output }))
                    .await;
            }
            Ok(output)
        }
        Err(e) => {
            let msg = e.to_string();
            run.end_with_error(msg.clone());
            if let Some(ref cb) = callbacks {
                cb.dispatch_chain_error(&run, &msg).await;
            }
            Err(e)
        }
    }
}

/// Stream a chain body wrapped in `on_chain_start` dispatch, ending the run
/// (and dispatching `on_chain_end` / `on_chain_error`) once the token stream
/// completes or fails.
///
/// Shared by the default `stream_with_config` and by composite chains that
/// override it to thread `config` into their sub-chains.
pub(crate) async fn stream_chain_with_callbacks<F, Fut>(
    name: &str,
    inputs: HashMap<String, Value>,
    config: Option<RunnableConfig>,
    body: F,
) -> Result<ChainStream, ChainError>
where
    F: FnOnce(HashMap<String, Value>) -> Fut,
    Fut: Future<Output = Result<ChainStream, ChainError>> + Send,
{
    let callbacks = config.as_ref().and_then(|c| c.callbacks.clone());
    let mut run = RunTree::new(name, RunType::Chain, json!({ "inputs": inputs }));

    if let Some(ref cb) = callbacks {
        cb.dispatch_chain_start(&run, &run.inputs).await;
    }

    let stream = match body(inputs).await {
        Ok(s) => s,
        Err(e) => {
            let msg = e.to_string();
            run.end_with_error(msg.clone());
            if let Some(ref cb) = callbacks {
                cb.dispatch_chain_error(&run, &msg).await;
            }
            return Err(e);
        }
    };

    Ok(Box::pin(end_stream_on_completion(stream, run, callbacks)))
}

/// Wrap a chain token stream so the RunTree is ended and `on_chain_end` /
/// `on_chain_error` dispatched once the stream completes or errors.
fn end_stream_on_completion(
    inner: ChainStream,
    run: RunTree,
    callbacks: Option<Arc<CallbackManager>>,
) -> impl Stream<Item = Result<StreamToken, ChainError>> + Send {
    stream::unfold(Some((inner, run, callbacks)), |state| async move {
        let (mut inner, run, callbacks) = match state {
            Some(s) => s,
            None => return None,
        };
        match inner.next().await {
            Some(Ok(token)) => Some((Ok(token), Some((inner, run, callbacks)))),
            Some(Err(e)) => {
                let msg = e.to_string();
                let mut run = run;
                run.end_with_error(msg.clone());
                if let Some(cb) = callbacks {
                    cb.dispatch_chain_error(&run, &msg).await;
                }
                Some((Err(e), None))
            }
            None => {
                let mut run = run;
                run.end(json!({ "output": null }));
                if let Some(cb) = callbacks {
                    cb.dispatch_chain_end(&run, &json!({ "output": null }))
                        .await;
                }
                None
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_chain_error_display() {
        let error = ChainError::MissingInput("test".to_string());
        assert!(error.to_string().contains("Missing input"));

        let error = ChainError::ExecutionError("test".to_string());
        assert!(error.to_string().contains("Execution error"));
    }

    #[test]
    fn test_chain_error_all_variants() {
        let err = ChainError::MissingInput("key".to_string());
        assert!(err.to_string().contains("key"));

        let err = ChainError::OutputError("bad".to_string());
        assert!(err.to_string().contains("bad"));

        let err = ChainError::ExecutionError("fail".to_string());
        assert!(err.to_string().contains("fail"));

        let err = ChainError::StreamError("broken".to_string());
        assert!(err.to_string().contains("broken"));

        let err = ChainError::Other("misc".to_string());
        assert!(err.to_string().contains("misc"));
    }

    /// P2-1: `Nested` preserves the original error chain — `source()` must
    /// downcast back to the concrete `ChainError` variant instead of a
    /// flattened string.
    #[test]
    fn test_chain_error_nested_preserves_source() {
        let inner = ChainError::MissingInput("text".to_string());
        let nested = ChainError::Nested {
            context: "Step 0 (echo) execution failed".to_string(),
            source: Box::new(inner),
        };
        assert!(nested
            .to_string()
            .contains("Step 0 (echo) execution failed"));
        assert!(nested.to_string().contains("Missing input"));

        let source = nested.source().expect("Nested must carry a source");
        let downcast = source.downcast_ref::<ChainError>();
        assert!(
            matches!(downcast, Some(ChainError::MissingInput(k)) if k == "text"),
            "source should downcast back to the original variant, got {downcast:?}"
        );
    }

    #[test]
    fn test_stream_token_debug() {
        let token = StreamToken {
            token: "hello".to_string(),
            is_final: false,
        };
        assert!(format!("{:?}", token).contains("hello"));
    }

    /// P2-2: the default stream fails loudly when the chain produces a
    /// non-string output instead of silently emitting a single empty token
    /// (the old `unwrap_or("")`).
    #[tokio::test]
    async fn test_default_stream_errors_on_non_string_output() {
        struct NonStringChain;
        #[async_trait]
        impl BaseChain for NonStringChain {
            fn input_keys(&self) -> Vec<&str> {
                vec![]
            }
            fn output_keys(&self) -> Vec<&str> {
                vec!["count"]
            }
            async fn invoke(
                &self,
                _inputs: HashMap<String, Value>,
            ) -> Result<ChainResult, ChainError> {
                let mut result = HashMap::new();
                result.insert("count".to_string(), json!(3));
                Ok(result)
            }
        }

        let chain = NonStringChain;
        let err = match chain.stream(HashMap::new()).await {
            Ok(_) => panic!("expected an OutputError"),
            Err(e) => e,
        };
        assert!(
            matches!(err, ChainError::OutputError(_)),
            "expected OutputError, got {err:?}"
        );
    }

    #[test]
    fn test_validate_inputs_pass() {
        struct PassthroughChain;
        #[async_trait]
        impl BaseChain for PassthroughChain {
            fn input_keys(&self) -> Vec<&str> {
                vec!["input"]
            }
            fn output_keys(&self) -> Vec<&str> {
                vec!["output"]
            }
            async fn invoke(
                &self,
                inputs: HashMap<String, Value>,
            ) -> Result<ChainResult, ChainError> {
                Ok(inputs)
            }
        }

        let chain = PassthroughChain;
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Value::String("test".to_string()));
        assert!(chain.validate_inputs(&inputs).is_ok());
    }

    #[test]
    fn test_validate_inputs_missing_key() {
        struct PassthroughChain;
        #[async_trait]
        impl BaseChain for PassthroughChain {
            fn input_keys(&self) -> Vec<&str> {
                vec!["input"]
            }
            fn output_keys(&self) -> Vec<&str> {
                vec!["output"]
            }
            async fn invoke(
                &self,
                _inputs: HashMap<String, Value>,
            ) -> Result<ChainResult, ChainError> {
                Ok(HashMap::new())
            }
        }

        let chain = PassthroughChain;
        let inputs = HashMap::new();
        assert!(chain.validate_inputs(&inputs).is_err());
    }

    #[test]
    fn test_default_chain_name() {
        struct MyChain;
        #[async_trait]
        impl BaseChain for MyChain {
            fn input_keys(&self) -> Vec<&str> {
                vec![]
            }
            fn output_keys(&self) -> Vec<&str> {
                vec![]
            }
            async fn invoke(
                &self,
                _inputs: HashMap<String, Value>,
            ) -> Result<ChainResult, ChainError> {
                Ok(HashMap::new())
            }
        }
        let chain = MyChain;
        assert_eq!(chain.name(), "chain");
    }
}
