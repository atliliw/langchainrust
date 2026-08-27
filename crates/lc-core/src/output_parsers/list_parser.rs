use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use super::base::{BaseOutputParser, OutputParserError, OutputParserResult};
use crate::language_models::LLMResult;
use crate::runnables::{Runnable, RunnableConfig};

/// Comma-separated list output parser
///
/// Parses the LLM's comma-separated text into a list of strings.
/// Supports both the fullwidth comma (U+FF0C) and the ASCII comma (,).
/// Automatically strips whitespace and empty items.
///
/// Equivalent to Python LangChain's `CommaSeparatedListOutputParser`.
///
/// # Example
/// ```ignore
/// use langchainrust::output_parsers::CommaSeparatedListOutputParser;
///
/// let parser = CommaSeparatedListOutputParser::new();
/// let result = parser.parse("apple, banana, cherry").await?;
/// assert_eq!(result, vec!["apple", "banana", "cherry"]);
/// ```
pub struct CommaSeparatedListOutputParser;

impl CommaSeparatedListOutputParser {
    /// Creates a comma-separated list output parser.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CommaSeparatedListOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseOutputParser<Vec<String>> for CommaSeparatedListOutputParser {
    async fn parse(&self, text: &str) -> OutputParserResult<Vec<String>> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // support Chinese and English commas
        let items: Vec<String> = text
            .split(',')
            .flat_map(|item| item.split('，'))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        Ok(items)
    }

    fn get_format_instructions(&self) -> String {
        "请用逗号分隔的列表形式输出，例如：项目1, 项目2, 项目3".to_string()
    }
}

#[async_trait]
impl Runnable<LLMResult, Vec<String>> for CommaSeparatedListOutputParser {
    type Error = OutputParserError;

    async fn invoke(
        &self,
        input: LLMResult,
        _config: Option<RunnableConfig>,
    ) -> Result<Vec<String>, Self::Error> {
        self.parse(&input.content).await
    }

    async fn stream(
        &self,
        input: LLMResult,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Vec<String>, Self::Error>> + Send>>, Self::Error>
    {
        let result = self.parse(&input.content).await?;
        let stream = futures_util::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }
}
