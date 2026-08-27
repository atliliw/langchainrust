use async_trait::async_trait;

/// Unified error type for output parsers
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum OutputParserError {
    /// Parse failure: the input format does not match expectations
    #[error("Parse error: {0}")]
    ParseError(String),
    /// JSON format error
    #[error("JSON error: {0}")]
    JsonError(String),
    /// Type conversion error
    #[error("Type error: {0}")]
    TypeError(String),
    /// Custom error
    #[error("{0}")]
    Custom(String),
}

impl From<serde_json::Error> for OutputParserError {
    fn from(e: serde_json::Error) -> Self {
        OutputParserError::JsonError(e.to_string())
    }
}

/// Result type for output parsers
pub type OutputParserResult<T> = Result<T, OutputParserError>;

/// Core trait for output parsers
///
/// Every output parser must implement this trait.
/// Unlike `Runnable`, `parse` takes no config argument,
/// so it fits being called inside a Runnable.
#[async_trait]
pub trait BaseOutputParser<Output: Send + Sync + 'static>: Send + Sync {
    /// Parses raw LLM output text into the target type
    async fn parse(&self, text: &str) -> OutputParserResult<Output>;

    /// Parsing with retry (default: genuinely retries `max_retries` times)
    ///
    /// Calls [`parse`](Self::parse) repeatedly on the same text, at most
    /// `max_retries + 1` times. Retrying the same text only makes sense for
    /// non-deterministic parsing (e.g. a parser that depends on the network /
    /// external services); a deterministic parser that fails once will keep
    /// failing, and the last error is returned. Parsers that need to correct
    /// the input based on the failure reason should override this method.
    async fn parse_with_retry(&self, text: &str, max_retries: usize) -> OutputParserResult<Output> {
        let mut last_err = None;
        for _ in 0..=max_retries {
            match self.parse(text).await {
                Ok(output) => return Ok(output),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| {
            OutputParserError::ParseError("parse_with_retry made no parse attempts".to_string())
        }))
    }

    /// Returns format instructions (to prompt the LLM to output in the expected format)
    fn get_format_instructions(&self) -> String {
        String::new()
    }
}
