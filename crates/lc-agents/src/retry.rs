//! Retry helpers for LLM calls.
//!
//! LLM provider services are subject to transient failures (timeouts, 5xx,
//! rate limits). Wrapping chat calls in an exponential-backoff retry turns a
//! single network blip from a hard task failure into a non-event.

use std::time::Duration;

use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_core::runnables::RunnableConfig;
use lc_schema::Message;

/// Configuration for exponential-backoff retries.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retries after the first failure.
    pub max_retries: usize,
    /// Initial delay before the first retry.
    pub base_delay: Duration,
    /// Upper bound on the backoff delay.
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
        }
    }
}

/// Call `llm.chat(...)` with exponential-backoff retries.
///
/// Returns the first successful result, or the last error once
/// `retry.max_retries` retries have been exhausted. Accepts any reference to
/// a `BaseChatModel` (concrete type, `&M`, or `&dyn ...` via `as_ref()`).
pub(crate) async fn retry_chat<M>(
    llm: &M,
    messages: Vec<Message>,
    config: Option<RunnableConfig>,
    retry: &RetryConfig,
) -> Result<LLMResult, M::Error>
where
    M: BaseChatModel + ?Sized,
{
    let mut attempt = 0usize;
    loop {
        match llm.chat(messages.clone(), config.clone()).await {
            Ok(result) => return Ok(result),
            Err(e) if attempt < retry.max_retries => {
                // Exponential backoff: base_delay * 2^attempt, capped at max_delay.
                let shift = 1u32.checked_shl(attempt as u32).unwrap_or(u32::MAX);
                let delay = retry.base_delay.saturating_mul(shift).min(retry.max_delay);
                log::warn!(
                    "LLM call failed (attempt {}), retrying in {:?}: {}",
                    attempt + 1,
                    delay,
                    e
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use lc_core::language_models::{BaseLanguageModel, StreamChunk};
    use lc_core::runnables::Runnable;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A chat model that fails the first N calls, then succeeds.
    struct FlakyChat {
        calls: AtomicUsize,
        failures_before_success: usize,
    }

    impl FlakyChat {
        fn new(failures_before_success: usize) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                failures_before_success,
            }
        }
    }

    #[derive(Debug, thiserror::Error)]
    #[error("flaky chat error")]
    struct FlakyError;

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for FlakyChat {
        type Error = FlakyError;

        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            unreachable!()
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for FlakyChat {
        fn model_name(&self) -> &str {
            "flaky"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }

        fn with_temperature(self, _temp: f32) -> Self
        where
            Self: Sized,
        {
            self
        }

        fn with_max_tokens(self, _max: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for FlakyChat {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call < self.failures_before_success {
                Err(FlakyError)
            } else {
                Ok(LLMResult {
                    content: "ok".to_string(),
                    model: "flaky".to_string(),
                    token_usage: None,
                    tool_calls: None,
                    thinking_content: None,
                })
            }
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<
            std::pin::Pin<
                Box<dyn futures_util::Stream<Item = Result<StreamChunk, Self::Error>> + Send>,
            >,
            Self::Error,
        > {
            unreachable!()
        }
    }

    #[test]
    fn retry_config_defaults() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.base_delay, Duration::from_secs(1));
        assert_eq!(cfg.max_delay, Duration::from_secs(30));
    }

    #[tokio::test]
    async fn retry_succeeds_after_transient_failures() {
        let llm = FlakyChat::new(2); // fail twice, succeed on 3rd attempt
        let cfg = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
        };
        let result = retry_chat(&llm, vec![Message::human("hi")], None, &cfg).await;
        assert!(result.is_ok());
        assert_eq!(llm.calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_exhausts_and_returns_last_error() {
        let llm = FlakyChat::new(10); // always fails
        let cfg = RetryConfig {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
        };
        let result = retry_chat(&llm, vec![Message::human("hi")], None, &cfg).await;
        assert!(result.is_err());
        // 1 initial call + 2 retries
        assert_eq!(llm.calls.load(Ordering::SeqCst), 3);
    }
}
