use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use super::base::{BaseOutputParser, OutputParserError, OutputParserResult};
use crate::language_models::LLMResult;
use crate::runnables::{Runnable, RunnableConfig};

/// String output parser
///
/// The simplest parser: returns the LLM output as a string verbatim.
/// Equivalent to Python LangChain's `StrOutputParser`.
///
/// As a `Runnable` it receives an `LLMResult`, returns its `content` field unchanged,
/// making `llm.pipe(StrOutputParser)` the tail of an LCEL chain.
///
/// # Example
/// ```ignore
/// use langchainrust::output_parsers::StrOutputParser;
///
/// let parser = StrOutputParser::new();
/// let result = parser.parse("Hello, world!").await?;
/// assert_eq!(result, "Hello, world!");
/// ```
pub struct StrOutputParser;

impl StrOutputParser {
    /// Creates a string output parser.
    pub fn new() -> Self {
        Self
    }
}

impl Default for StrOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseOutputParser<String> for StrOutputParser {
    async fn parse(&self, text: &str) -> OutputParserResult<String> {
        Ok(text.to_string())
    }
}

#[async_trait]
impl Runnable<LLMResult, String> for StrOutputParser {
    type Error = OutputParserError;

    async fn invoke(
        &self,
        input: LLMResult,
        _config: Option<RunnableConfig>,
    ) -> Result<String, Self::Error> {
        self.parse(&input.content).await
    }

    async fn stream(
        &self,
        input: LLMResult,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        let result = self.parse(&input.content).await?;
        let stream = futures_util::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }
}
