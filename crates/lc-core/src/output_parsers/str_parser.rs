use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use super::base::{BaseOutputParser, OutputParserError, OutputParserResult};
use crate::runnables::{Runnable, RunnableConfig};

/// 字符串输出解析器
///
/// 最简单的解析器，直接将 LLM 输出作为字符串返回。
/// 相当于 Python LangChain 的 `StrOutputParser`。
///
/// # 示例
/// ```ignore
/// use langchainrust::output_parsers::StrOutputParser;
///
/// let parser = StrOutputParser::new();
/// let result = parser.parse("Hello, world!").await?;
/// assert_eq!(result, "Hello, world!");
/// ```
pub struct StrOutputParser;

impl StrOutputParser {
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
impl Runnable<String, String> for StrOutputParser {
    type Error = OutputParserError;

    async fn invoke(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<String, Self::Error> {
        self.parse(&input).await
    }

    async fn stream(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        let result = self.parse(&input).await?;
        let stream = futures_util::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }
}
