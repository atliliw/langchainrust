//! GuardedAgent — an Agent wrapper with Guardrails

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use lc_agents::AgentExecutor;
use lc_chains::BaseChain;

use super::guardrail::{ChunkAction, GuardrailError, GuardrailsConfig};
use super::runner::{GuardrailRunner, GuardrailViolation, OutputValidation};

/// Error type for Guardable execution units.
type DynError = Box<dyn std::error::Error + Send + Sync>;

/// Streaming output chunk: the text emitted in one step + whether it is the final one.
#[derive(Debug)]
pub struct GuardableChunk {
    /// The text emitted in one step
    pub token: String,
    /// Whether this is the final chunk
    pub is_final: bool,
}

/// The execution unit `GuardedAgent` can wrap (P1-3 decoupling).
///
/// It depends only on this trait, not directly on `AgentExecutor`:
/// - [`AgentExecutor`] implements it directly
/// - any [`BaseChain`] gets an implementation via the [`ChainGuardable`] adapter
#[async_trait]
pub trait Guardable: Send + Sync {
    /// String in, string out.
    async fn invoke_str(&self, input: &str) -> Result<String, DynError>;

    /// Streaming output; implementations that do not support it return an error, and `GuardedAgent` falls back to a one-shot `invoke_str`.
    async fn stream_str(
        &self,
        input: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GuardableChunk, DynError>> + Send>>, DynError>;
}

#[async_trait]
impl Guardable for AgentExecutor {
    async fn invoke_str(&self, input: &str) -> Result<String, DynError> {
        self.invoke(input.to_string())
            .await
            .map_err(|e| Box::new(e) as DynError)
    }

    async fn stream_str(
        &self,
        input: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GuardableChunk, DynError>> + Send>>, DynError>
    {
        use futures_util::StreamExt;
        use lc_agents::streaming::state::AgentStreamEvent;

        // map the Agent event stream into user-facing output chunks: keep only FinalAnswer,
        // ToolStart / ToolEnd are intermediate steps and do not enter the guardrail check surface.
        let stream = self
            .stream(input.to_string())
            .filter_map(|event| async move {
                match event {
                    Ok(AgentStreamEvent::FinalAnswer { content }) => Some(Ok(GuardableChunk {
                        token: content,
                        is_final: true,
                    })),
                    Ok(AgentStreamEvent::Error { message }) => Some(Err(DynError::from(message))),
                    Ok(_) => None, // ToolStart / ToolEnd
                    Err(e) => Some(Err(Box::new(e) as DynError)),
                }
            });
        Ok(Box::pin(stream))
    }
}

/// Guardable adapter for `BaseChain` (P1-3 decoupling).
///
/// Uses `input_keys()[0]` / `output_keys()[0]` for string I/O; a missing or non-string output
/// key returns an explicit error instead of being silently swallowed.
///
/// Why an adapter instead of `impl Guardable for dyn BaseChain` / a blanket impl:
/// - a blanket `impl<T: BaseChain> Guardable for T` cannot coexist with
///   `impl Guardable for AgentExecutor` due to coherence (rustc cannot rule out AgentExecutor
///   implementing BaseChain in the future);
/// - there is no non-supertrait cast from `dyn BaseChain` to `dyn Guardable` (`Unsize` is not
///   satisfied), so `Arc<dyn BaseChain>` cannot be automatically coerced into `Arc<dyn Guardable>`.
///
/// Hence `ChainGuardable` bridges explicitly, and `GuardedAgent::from_chain` provides the entry point.
pub struct ChainGuardable(pub Arc<dyn BaseChain>);

#[async_trait]
impl Guardable for ChainGuardable {
    async fn invoke_str(&self, input: &str) -> Result<String, DynError> {
        let chain: &dyn BaseChain = self.0.as_ref();
        let input_key = chain
            .input_keys()
            .first()
            .ok_or_else(|| DynError::from("chain has no input key"))?
            .to_string();
        let output_key = chain
            .output_keys()
            .first()
            .ok_or_else(|| DynError::from("chain has no output key"))?
            .to_string();

        let mut inputs = HashMap::new();
        inputs.insert(input_key, serde_json::Value::String(input.to_string()));

        let result = chain
            .invoke(inputs)
            .await
            .map_err(|e| Box::new(e) as DynError)?;

        let value = result
            .get(&output_key)
            .ok_or_else(|| DynError::from(format!("chain output has no key {:?}", output_key)))?;
        value.as_str().map(|s| s.to_string()).ok_or_else(|| {
            DynError::from(format!("chain output key {:?} is not a string", output_key))
        })
    }

    async fn stream_str(
        &self,
        input: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GuardableChunk, DynError>> + Send>>, DynError>
    {
        use futures_util::StreamExt;

        let chain: &dyn BaseChain = self.0.as_ref();
        let input_key = chain
            .input_keys()
            .first()
            .ok_or_else(|| DynError::from("chain has no input key"))?
            .to_string();

        let mut inputs = HashMap::new();
        inputs.insert(input_key, serde_json::Value::String(input.to_string()));

        let stream = chain
            .stream(inputs)
            .await
            .map_err(|e| Box::new(e) as DynError)?;

        let mapped = stream.map(|item| {
            item.map(|t| GuardableChunk {
                token: t.token,
                is_final: t.is_final,
            })
            .map_err(|e| Box::new(e) as DynError)
        });
        Ok(Box::pin(mapped))
    }
}

/// Agent wrapper with Guardrails
///
/// On `invoke`: validate input -> run Guardable -> validate output.
/// On `invoke_stream`: two-phase streaming guardrails (P1-4).
pub struct GuardedAgent {
    inner: Arc<dyn Guardable>,
    runner: GuardrailRunner,
}

impl GuardedAgent {
    /// Constructs from any [`Guardable`]. `Arc<AgentExecutor>` auto-coerces to `Arc<dyn Guardable>`.
    pub fn new(inner: Arc<dyn Guardable>, config: GuardrailsConfig) -> Self {
        Self {
            inner,
            runner: GuardrailRunner::new(config),
        }
    }

    /// Constructs from any [`BaseChain`] (P1-3 decoupling).
    ///
    /// `Arc<dyn BaseChain>` cannot be coerced directly to `Arc<dyn Guardable>` (not a supertrait),
    /// so it is bridged through the [`ChainGuardable`] adapter. Concrete chains such as `Arc<EchoChain>`
    /// are first auto-coerced to `Arc<dyn BaseChain>` and then wrapped.
    pub fn from_chain(chain: Arc<dyn BaseChain>, config: GuardrailsConfig) -> Self {
        Self::new(Arc::new(ChainGuardable(chain)), config)
    }

    /// Executes: validate input -> Guardable -> validate output.
    ///
    /// On blocking, returns [`GuardrailError::Blocked`] carrying partial output + a user suggestion (P1-1/P1-6).
    pub async fn invoke(&mut self, input: String) -> Result<String, GuardrailError> {
        if let Err(e) = self.runner.validate_input(&input).await {
            return Err(match e {
                GuardrailError::Blocked { reason, .. } => GuardrailError::Blocked {
                    reason,
                    partial: Some(input),
                    suggestion: Some("please adjust your input and retry".to_string()),
                },
                other => other,
            });
        }

        let output = self
            .inner
            .invoke_str(&input)
            .await
            .map_err(|e| GuardrailError::AgentError(e.to_string()))?;

        match self.runner.validate_output(&output).await {
            OutputValidation::Passed(value) => Ok(value),
            OutputValidation::Blocked { reason, partial } => Err(GuardrailError::from_blocked(
                reason,
                partial,
                "output was blocked by a safety guardrail; please adjust your request and retry, or omit sensitive content"
                    .to_string(),
            )),
        }
    }

    /// Two-phase streaming execution (P1-4).
    ///
    /// Phase one: each output chunk goes through the streaming guardrails (with a sliding window
    /// to prevent keywords split across chunks); phase two: after the stream ends, re-check the
    /// full output via [`GuardrailRunner::validate_output`]. Input guardrails are validated
    /// synchronously before the stream starts.
    pub async fn invoke_stream(
        &mut self,
        input: String,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<GuardableChunk, GuardrailError>> + Send>>,
        GuardrailError,
    > {
        use futures_util::StreamExt;

        self.runner.validate_input(&input).await?;

        let inner = self.inner.clone();
        let raw_stream = inner
            .stream_str(&input)
            .await
            .map_err(|e| GuardrailError::AgentError(e.to_string()))?;

        // each phase holds its own runner clone to avoid borrowing `self` into the returned stream.
        let mut phase2_runner = self.runner.clone();
        // phase-one state (sliding-window tail, violation-accumulating runner, accumulated output full) is shared across chunks.
        // `then`'s closure is FnMut: each invocation clones an Arc and moves it into the async block,
        // while the state itself stays in the Arc, persisting across chunks.
        let tail = Arc::new(tokio::sync::Mutex::new(String::new()));
        let phase1_runner = Arc::new(tokio::sync::Mutex::new(self.runner.clone()));
        let full = Arc::new(tokio::sync::Mutex::new(String::new()));
        let finalize_full = full.clone();
        const TAIL_WINDOW: usize = 24;

        let phase1 = raw_stream.then(move |item| {
            let tail = tail.clone();
            let runner = phase1_runner.clone();
            let full = full.clone();
            async move {
                let chunk = item.map_err(|e| GuardrailError::AgentError(e.to_string()))?;
                let token = chunk.token;
                // sliding-window probe: tail + chunk, so keywords split across chunks are still detected.
                let probe = {
                    let t = tail.lock().await;
                    format!("{}{}", *t, token)
                };
                let action = runner.lock().await.validate_stream_chunk(&probe).await;
                let emitted = match action {
                    ChunkAction::Pass => token,
                    ChunkAction::Replace(new_value) => new_value,
                    ChunkAction::Block => {
                        let partial = tail.lock().await.clone();
                        return Err(GuardrailError::Blocked {
                            reason: "streaming output was blocked by a guardrail".to_string(),
                            partial: Some(partial),
                            suggestion: Some(
                                "output was blocked by a safety guardrail; please adjust your request and retry"
                                    .to_string(),
                            ),
                        });
                    }
                };
                full.lock().await.push_str(&emitted);
                // update the sliding window: keep only the most recent TAIL_WINDOW characters.
                let new_tail: String = emitted
                    .chars()
                    .rev()
                    .take(TAIL_WINDOW)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                *tail.lock().await = new_tail;
                Ok(GuardableChunk {
                    token: emitted,
                    is_final: false,
                })
            }
        });

        let finalize = futures_util::stream::once(async move {
            let full_text = finalize_full.lock().await.clone();
            match phase2_runner.validate_output(&full_text).await {
                // phase one has already emitted all (possibly rewritten) chunks, so concatenating them yields the full output;
                // when phase two passes, emit only an empty-token end marker without re-outputting.
                OutputValidation::Passed(_value) => Ok(GuardableChunk {
                    token: String::new(),
                    is_final: true,
                }),
                OutputValidation::Blocked { reason, partial } => Err(GuardrailError::from_blocked(
                    reason,
                    partial,
                    "final output re-check failed; please adjust your request and retry, or omit sensitive content"
                        .to_string(),
                )),
            }
        });

        Ok(Box::pin(phase1.chain(finalize)))
    }

    /// Returns a snapshot of violation records (including streaming-path records: the two-phase runner shares the log with `self.runner`).
    pub fn violations(&self) -> Vec<GuardrailViolation> {
        self.runner.violations()
    }

    /// Clears violation records (P1-2).
    pub fn clear_violations(&mut self) {
        self.runner.clear_violations();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validators::MaxLengthGuardrail;
    use lc_agents::{BaseAgent, FunctionCallingAgent};
    use lc_chains::base::{ChainError, ChainResult, ChainStream, StreamToken};
    use lc_providers::{OpenAIChat, OpenAIConfig};
    use serde_json::Value;

    fn guarded_with_maxlen(max: usize) -> GuardedAgent {
        let llm = OpenAIChat::new(OpenAIConfig::default());
        let agent = FunctionCallingAgent::new(llm, vec![], None);
        let executor = Arc::new(AgentExecutor::new(
            Arc::new(agent) as Arc<dyn BaseAgent>,
            vec![],
        ));
        let config =
            GuardrailsConfig::new().with_input(Arc::new(MaxLengthGuardrail::new(max)) as Arc<_>);
        GuardedAgent::new(executor, config)
    }

    #[tokio::test]
    async fn test_blocks_long_input_before_agent() {
        // input over 3 characters is blocked by MaxLength without calling the Agent (no network)
        let mut g = guarded_with_maxlen(3);
        let result = g.invoke("this is too long input".to_string()).await;
        assert!(result.is_err());
        assert_eq!(g.violations().len(), 1);
        // confirm it is Blocked (carrying partial + suggestion), not AgentError
        match result.unwrap_err() {
            GuardrailError::Blocked {
                partial,
                suggestion,
                ..
            } => {
                assert!(partial.is_some());
                assert!(suggestion.is_some());
            }
            other => panic!("应为 Blocked, 实际: {:?}", other),
        }
    }

    /// Simple echo Chain: input -> "echo:{input}".
    struct EchoChain;
    #[async_trait]
    impl BaseChain for EchoChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["input"]
        }
        fn output_keys(&self) -> Vec<&str> {
            vec!["output"]
        }
        async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            let mut out = HashMap::new();
            if let Some(Value::String(s)) = inputs.get("input") {
                out.insert("output".to_string(), Value::String(format!("echo:{}", s)));
            }
            Ok(out)
        }
    }

    /// Chunked-output Chain: simulates a token-by-token stream.
    struct TokenChain;
    #[async_trait]
    impl BaseChain for TokenChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["input"]
        }
        fn output_keys(&self) -> Vec<&str> {
            vec!["output"]
        }
        async fn invoke(&self, _inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            Ok(HashMap::new())
        }
        async fn stream(&self, _inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
            let tokens = vec![
                Ok(StreamToken {
                    token: "Hello ".to_string(),
                    is_final: false,
                }),
                Ok(StreamToken {
                    token: "world".to_string(),
                    is_final: false,
                }),
            ];
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
    }

    #[tokio::test]
    async fn test_guardable_chain_invoke() {
        // any Arc<dyn BaseChain> goes through the ChainGuardable adapter (P1-3 decoupling).
        let chain: Arc<dyn BaseChain> = Arc::new(EchoChain);
        let mut g = GuardedAgent::from_chain(chain, GuardrailsConfig::new());
        let result = g.invoke("hi".to_string()).await.unwrap();
        assert_eq!(result, "echo:hi");
    }

    #[tokio::test]
    async fn test_invoke_stream_two_phase_passes() {
        // no guardrails: the two-phase stream completes normally, final output = concatenation of all chunks.
        let chain: Arc<dyn BaseChain> = Arc::new(TokenChain);
        let mut g = GuardedAgent::from_chain(chain, GuardrailsConfig::new());
        let mut stream = g.invoke_stream("q".to_string()).await.unwrap();

        use futures_util::StreamExt;
        let mut collected = String::new();
        let mut finals = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            if chunk.is_final {
                finals += 1;
            }
            collected.push_str(&chunk.token);
        }
        assert_eq!(collected, "Hello world");
        assert_eq!(finals, 1);
    }

    /// Local streaming guardrail that blocks when "world" is matched.
    struct BlockOnWorld;
    #[async_trait]
    impl crate::guardrail::StreamingOutputGuardrail for BlockOnWorld {
        fn name(&self) -> &str {
            "BlockOnWorld"
        }
        async fn validate_chunk(&self, chunk: &str) -> crate::guardrail::ChunkAction {
            if chunk.contains("world") {
                crate::guardrail::ChunkAction::Block
            } else {
                crate::guardrail::ChunkAction::Pass
            }
        }
    }

    #[tokio::test]
    async fn test_invoke_stream_blocks_keyword() {
        // phase-one streaming guardrail hits the keyword -> Blocked returned mid-stream (carrying the already-emitted part).
        let chain: Arc<dyn BaseChain> = Arc::new(TokenChain);
        let config = GuardrailsConfig::new().with_streaming(
            Arc::new(BlockOnWorld) as Arc<dyn crate::guardrail::StreamingOutputGuardrail>
        );
        let mut g = GuardedAgent::from_chain(chain, config);
        let mut stream = g.invoke_stream("q".to_string()).await.unwrap();

        use futures_util::StreamExt;
        let mut saw_error = false;
        while let Some(chunk) = stream.next().await {
            if chunk.is_err() {
                saw_error = true;
            }
        }
        assert!(saw_error);
        assert!(!g.violations().is_empty());
    }

    #[tokio::test]
    async fn test_clear_violations_passthrough() {
        let mut g = guarded_with_maxlen(3);
        let _ = g.invoke("this is too long input".to_string()).await;
        assert!(!g.violations().is_empty());
        g.clear_violations();
        assert!(g.violations().is_empty());
    }
}
