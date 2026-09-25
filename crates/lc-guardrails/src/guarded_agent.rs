//! GuardedAgent — an Agent wrapper with Guardrails

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use lc_agents::AgentExecutor;
use lc_chains::BaseChain;

use super::guardrail::{ChunkAction, ChunkContext, GuardrailError, GuardrailsConfig};
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
        // K2: reset stateful streaming rails once per stream so hold-back buffers from a prior
        // stream cannot leak into (or double-consume) this one. The runner clones below share
        // the same Arc'd rail instances, so resetting here clears them for every phase.
        self.runner.reset_streaming();

        let inner = self.inner.clone();
        let raw_stream = inner
            .stream_str(&input)
            .await
            .map_err(|e| GuardrailError::AgentError(e.to_string()))?;

        // each phase holds its own runner clone to avoid borrowing `self` into the returned stream.
        let mut phase2_runner = self.runner.clone();
        // phase-one state (sliding-window tail, violation-accumulating runner, accumulated output) is shared across chunks.
        // `then`'s closure is FnMut: each invocation clones an Arc and moves it into the async block,
        // while the state itself stays in the Arc, persisting across chunks.
        let tail = Arc::new(tokio::sync::Mutex::new(String::new()));
        let phase1_runner = Arc::new(tokio::sync::Mutex::new(self.runner.clone()));
        // raw model output (every raw token, pre-rewrite): the `full` view for whole-document rails.
        let raw_full = Arc::new(tokio::sync::Mutex::new(String::new()));
        // released output (post-rewrite): concatenating emitted chunks; re-validated by phase two.
        let released = Arc::new(tokio::sync::Mutex::new(String::new()));
        let finalize_released = released.clone();
        const TAIL_WINDOW: usize = 24;

        let phase1 = raw_stream.then(move |item| {
            let tail = tail.clone();
            let runner = phase1_runner.clone();
            let raw_full = raw_full.clone();
            let released = released.clone();
            async move {
                let chunk = item.map_err(|e| GuardrailError::AgentError(e.to_string()))?;
                let token = chunk.token;
                // sliding-window probe (`window`) + full raw candidate (`full`), built before any
                // rewrite so every rail in the chain sees the same raw model output.
                let (window, candidate) = {
                    let t = tail.lock().await;
                    let raw = raw_full.lock().await;
                    (
                        format!("{}{}", *t, token),
                        format!("{}{}", *raw, token),
                    )
                };
                let ctx = ChunkContext {
                    token: &token,
                    window: &window,
                    full: &candidate,
                };
                let action = runner.lock().await.validate_stream_chunk(&ctx).await;
                let emitted = match action {
                    ChunkAction::Pass => token.clone(),
                    ChunkAction::Replace(new_value) => new_value,
                    ChunkAction::Block => {
                        let partial = released.lock().await.clone();
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
                raw_full.lock().await.push_str(&token);
                released.lock().await.push_str(&emitted);
                // update the sliding window over RAW output: keep only the most recent TAIL_WINDOW characters.
                let new_tail: String = window
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

        // finalization: first let stateful rails (hold-back PII redaction) release their buffered
        // tails, then run the terminal full-output re-check. Emits the flush deltas as ordinary
        // chunks followed by one empty-token end marker.
        let finalize = futures_util::stream::once(async move {
            let mut steps: Vec<Result<GuardableChunk, GuardrailError>> = Vec::new();
            for text in phase2_runner.flush_stream().await {
                finalize_released.lock().await.push_str(&text);
                steps.push(Ok(GuardableChunk {
                    token: text,
                    is_final: false,
                }));
            }
            let full_text = finalize_released.lock().await.clone();
            match phase2_runner.validate_output(&full_text).await {
                // M-26: phase one already emitted every (possibly rewritten) chunk, so concatenating
                // them is the full released output. A whole-document `Modify` output guardrail can only
                // rewrite the full text, which an in-flight stream cannot retract; phase two is therefore
                // validation-only for the released stream. When such a Modify actually diverged, record
                // the drop so the lost correction is auditable instead of silently discarded — callers
                // wanting an inline rewrite under streaming must register the rail via `with_streaming`.
                OutputValidation::Passed(rewritten) => {
                    if rewritten != full_text {
                        phase2_runner
                            .record_violation(GuardrailViolation {
                                guardrail_name: "phase_two_modify".to_string(),
                                stage: "output".to_string(),
                                reason: "whole-document Modify cannot retract already-streamed text; correction dropped in streaming mode".to_string(),
                            })
                            .await;
                    }
                    steps.push(Ok(GuardableChunk {
                        token: String::new(),
                        is_final: true,
                    }))
                }
                OutputValidation::Blocked { reason, partial } => steps.push(Err(
                    GuardrailError::from_blocked(
                        reason,
                        partial,
                        "final output re-check failed; please adjust your request and retry, or omit sensitive content"
                            .to_string(),
                    ),
                )),
            }
            steps
        })
        .flat_map(futures_util::stream::iter);

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
    use crate::guardrail::{OutputGuardrail, StreamingOutputGuardrail};
    use crate::pii::{PiiKind, PiiRedactionGuardrail};
    use crate::schema::SchemaOutputGuardrail;
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

    /// Streams a phrase containing a secret, in single tokens.
    struct StreamSecretChain;
    #[async_trait]
    impl BaseChain for StreamSecretChain {
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
            let tokens: Vec<Result<StreamToken, ChainError>> = "store this secret now"
                .split(' ')
                .map(|w| {
                    Ok(StreamToken {
                        token: format!("{w} "),
                        is_final: false,
                    })
                })
                .collect();
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
        async fn validate_chunk(
            &self,
            ctx: &crate::guardrail::ChunkContext<'_>,
        ) -> crate::guardrail::ChunkAction {
            if ctx.window.contains("world") {
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

    /// Whole-document Modify rail (output-only, no streaming counterpart): rewrites an email address.
    struct RewriteSecret;
    #[async_trait]
    impl crate::guardrail::OutputGuardrail for RewriteSecret {
        fn name(&self) -> &str {
            "RewriteSecret"
        }
        async fn validate(
            &self,
            output: &str,
        ) -> crate::guardrail::OutputGuardrailResult {
            if output.contains("secret") {
                crate::guardrail::OutputGuardrailResult::Modify {
                    new_value: output.replace("secret", "[REDACTED]"),
                }
            } else {
                crate::guardrail::OutputGuardrailResult::Pass
            }
        }
    }

    /// Streams a phrase containing a secret, registered only via `with_output`, so the whole-document
    /// Modify fires in phase two — already-streamed tokens cannot be retracted. M-26: the released
    /// stream stays as-is, but the dropped correction is recorded as an auditable violation.
    #[tokio::test]
    async fn test_invoke_stream_drops_output_modify_with_violation() {
        let chain: Arc<dyn BaseChain> = Arc::new(StreamSecretChain);
        let config = GuardrailsConfig::new().with_output(
            Arc::new(RewriteSecret) as Arc<dyn crate::guardrail::OutputGuardrail>
        );
        let mut g = GuardedAgent::from_chain(chain, config);
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
        // the released (already-streamed) text is untouched — a whole-document Modify can't retract it
        assert_eq!(collected, "store this secret now ");
        assert_eq!(finals, 1);
        // ...but the dropped correction is surfaced in the audit log rather than silently discarded
        let violations = g.violations();
        assert!(
            violations
                .iter()
                .any(|v| v.guardrail_name == "phase_two_modify"),
            "expected an audited phase-two Modify drop, got {violations:?}"
        );
    }

    /// Chain that emits a fixed script of tokens as a stream.
    struct ScriptedChain {
        tokens: Vec<&'static str>,
    }
    #[async_trait]
    impl BaseChain for ScriptedChain {
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
            let tokens = self
                .tokens
                .iter()
                .map(|t| {
                    Ok(StreamToken {
                        token: (*t).to_string(),
                        is_final: false,
                    })
                })
                .collect::<Vec<_>>();
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
    }

    #[tokio::test]
    async fn test_invoke_stream_pii_redacts_split_identifier() {
        // B11: a phone number split across four chunks must never leak raw; the hold-back
        // window withholds the tail, flush() releases the redacted remainder, and the
        // terminal output re-check passes over the rewritten text.
        let chain: Arc<dyn BaseChain> = Arc::new(ScriptedChain {
            tokens: vec!["contact ", "138", "1234", "5678", " end"],
        });
        let rail = Arc::new(
            PiiRedactionGuardrail::new()
                .only([PiiKind::Phone])
                .with_hold_back(20),
        );
        let config = GuardrailsConfig::new()
            .with_streaming(rail.clone() as Arc<dyn StreamingOutputGuardrail>)
            .with_output(rail as Arc<dyn OutputGuardrail>);
        let mut g = GuardedAgent::from_chain(chain, config);
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
            // the complete number must never be observable in the released prefix
            assert!(!collected.contains("13812345678"));
        }
        assert_eq!(collected, "contact [REDACTED_PHONE] end");
        assert_eq!(finals, 1);
        // the redaction is audited as interventions (chunk replace + rewritten flush),
        // but nothing was blocked.
        let violations = g.violations();
        assert!(!violations.is_empty());
        assert!(violations.iter().all(|v| !v.reason.contains("block")));
    }

    fn enum_answer_schema() -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string", "enum": ["yes", "no"] }
            },
            "required": ["answer"]
        })
    }

    #[tokio::test]
    async fn test_invoke_stream_schema_blocks_at_terminal() {
        // B11: an enum violation is tolerated while JSON is still arriving (no false
        // mid-stream block) but the terminal full-output check rejects the finished object.
        let chain: Arc<dyn BaseChain> = Arc::new(ScriptedChain {
            tokens: vec!["{\"answer\":", "\"maybe\"", "}"],
        });
        let rail = Arc::new(SchemaOutputGuardrail::new(enum_answer_schema()));
        let config = GuardrailsConfig::new()
            .with_streaming(rail.clone() as Arc<dyn StreamingOutputGuardrail>)
            .with_output(rail as Arc<dyn OutputGuardrail>);
        let mut g = GuardedAgent::from_chain(chain, config);
        let mut stream = g.invoke_stream("q".to_string()).await.unwrap();

        use futures_util::StreamExt;
        let mut ok_text = String::new();
        let mut saw_block = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(c) => {
                    assert!(!c.is_final, "no final marker may follow a terminal block");
                    ok_text.push_str(&c.token);
                }
                Err(GuardrailError::Blocked {
                    reason, partial, ..
                }) => {
                    saw_block = true;
                    assert!(reason.contains("not one of"), "reason was: {reason}");
                    assert_eq!(partial.unwrap(), "{\"answer\":\"maybe\"}");
                }
                Err(other) => panic!("expected Blocked, got {other:?}"),
            }
        }
        assert!(saw_block);
        assert_eq!(ok_text, "{\"answer\":\"maybe\"}");
        assert!(!g.violations().is_empty());
    }

    #[tokio::test]
    async fn test_invoke_stream_schema_valid_passes() {
        // B11: schema-conformant output streamed in pieces passes phase one and phase two.
        let chain: Arc<dyn BaseChain> = Arc::new(ScriptedChain {
            tokens: vec!["{\"answer\":", "\"yes\"", "}"],
        });
        let rail = Arc::new(SchemaOutputGuardrail::new(enum_answer_schema()));
        let config = GuardrailsConfig::new()
            .with_streaming(rail.clone() as Arc<dyn StreamingOutputGuardrail>)
            .with_output(rail as Arc<dyn OutputGuardrail>);
        let mut g = GuardedAgent::from_chain(chain, config);
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
        assert_eq!(collected, "{\"answer\":\"yes\"}");
        assert_eq!(finals, 1);
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
