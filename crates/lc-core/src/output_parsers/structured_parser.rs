use async_trait::async_trait;
use futures_util::Stream;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::pin::Pin;

use super::base::{BaseOutputParser, OutputParserError, OutputParserResult};
use crate::language_models::LLMResult;
use crate::runnables::{Runnable, RunnableConfig};

/// Structured output parser
///
/// Parses the LLM's key-value output (one `key: value` per line) into a HashMap.
/// Suited to scenarios where the LLM emits structured information in a non-JSON format.
///
/// # Format
/// The input format should be one `key: value` per line, e.g.:
/// ```text
/// 姓名: 张三
/// 年龄: 28
/// 城市: 北京
/// ```
///
/// # Example
/// ```ignore
/// use langchainrust::output_parsers::StructuredOutputParser;
///
/// let parser = StructuredOutputParser::new();
/// let result = parser.parse("姓名: 张三\n年龄: 28").await?;
/// assert_eq!(result.get("姓名").unwrap(), "张三");
/// ```
pub struct StructuredOutputParser {
    /// Separator between a key and its value
    separator: char,
}

impl StructuredOutputParser {
    /// Creates a structured output parser using the default separator (`:`).
    pub fn new() -> Self {
        Self { separator: ':' }
    }

    /// Creates a parser using a custom separator
    pub fn with_separator(separator: char) -> Self {
        Self { separator }
    }
}

impl Default for StructuredOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseOutputParser<HashMap<String, String>> for StructuredOutputParser {
    async fn parse(&self, text: &str) -> OutputParserResult<HashMap<String, String>> {
        let mut map = HashMap::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(pos) = line.find(self.separator) {
                // `pos` is the byte index of the separator's first byte; with a multi-byte separator
                // (e.g. fullwidth `：`) pos+1 would land inside a char and panic on slicing; skip by the separator's UTF-8 width
                let sep_len = self.separator.len_utf8();
                let key = line[..pos].trim().to_string();
                let value = line[pos + sep_len..].trim().to_string();

                if !key.is_empty() {
                    map.insert(key, value);
                }
            }
        }

        Ok(map)
    }

    fn get_format_instructions(&self) -> String {
        format!(
            "请按以下格式输出（每行一个键值对，使用 '{}' 分隔）：\n键{}值",
            self.separator, self.separator
        )
    }
}

#[async_trait]
impl Runnable<LLMResult, HashMap<String, String>> for StructuredOutputParser {
    type Error = OutputParserError;

    async fn invoke(
        &self,
        input: LLMResult,
        _config: Option<RunnableConfig>,
    ) -> Result<HashMap<String, String>, Self::Error> {
        self.parse(&input.content).await
    }

    async fn stream(
        &self,
        input: LLMResult,
        _config: Option<RunnableConfig>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<HashMap<String, String>, Self::Error>> + Send>>,
        Self::Error,
    > {
        let result = self.parse(&input.content).await?;
        let stream = futures_util::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }
}

/// Typed output parser
///
/// Parses the LLM's JSON string output into a specified Rust struct.
/// Equivalent to Python LangChain's `PydanticOutputParser` (serde replaces pydantic).
///
/// The target type must implement `serde::Deserialize`.
///
/// # Example
/// ```ignore
/// use serde::Deserialize;
/// use langchainrust::output_parsers::TypedOutputParser;
///
/// #[derive(Deserialize, Debug, PartialEq)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// let parser = TypedOutputParser::<Person>::new();
/// let person = parser.parse(r#"{"name": "Alice", "age": 30}"#).await?;
/// assert_eq!(person.name, "Alice");
/// ```
pub struct TypedOutputParser<T> {
    _phantom: PhantomData<T>,
}

impl<T> TypedOutputParser<T> {
    /// Creates a typed output parser.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: DeserializeOwned> Default for TypedOutputParser<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<T: DeserializeOwned + Send + Sync + 'static> BaseOutputParser<T> for TypedOutputParser<T> {
    async fn parse(&self, text: &str) -> OutputParserResult<T> {
        let text = text.trim();

        // try to extract JSON from a Markdown code block
        let json_str = Self::extract_from_markdown(text).unwrap_or(text);

        // first parse into a Value to validate it
        serde_json::from_str::<serde_json::Value>(json_str)
            .map_err(|e| OutputParserError::JsonError(format!("input is not valid JSON: {}", e)))?;

        // deserialize into the target type
        serde_json::from_str::<T>(json_str).map_err(|e| {
            OutputParserError::TypeError(format!(
                "type deserialization failed (check whether the JSON fields match): {}",
                e
            ))
        })
    }
}

impl<T: DeserializeOwned> TypedOutputParser<T> {
    /// Extracts a JSON string from a Markdown code block
    fn extract_from_markdown(text: &str) -> Option<&str> {
        // try ```json ... ```
        if let Some(start) = text.find("```json") {
            let after = &text[start + 7..];
            if let Some(end) = after.find("```") {
                return Some(after[..end].trim());
            }
        }
        // try ``` ... ```
        if let Some(start) = text.find("```") {
            let after = &text[start + 3..];
            let after = after.trim();
            let skip = after.find('\n').unwrap_or(0);
            let after = &after[skip..].trim();
            if let Some(end) = after.find("```") {
                return Some(after[..end].trim());
            }
        }
        None
    }
}

#[async_trait]
impl<T: DeserializeOwned + Send + Sync + 'static> Runnable<LLMResult, T> for TypedOutputParser<T> {
    type Error = OutputParserError;

    async fn invoke(
        &self,
        input: LLMResult,
        _config: Option<RunnableConfig>,
    ) -> Result<T, Self::Error> {
        self.parse(&input.content).await
    }

    async fn stream(
        &self,
        input: LLMResult,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<T, Self::Error>> + Send>>, Self::Error> {
        let result = self.parse(&input.content).await?;
        let stream = futures_util::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_structured_parser_default_colon() {
        let parser = StructuredOutputParser::new();
        let map = parser.parse("姓名: 张三\n年龄: 28").await.unwrap();
        assert_eq!(map.get("姓名").unwrap(), "张三");
        assert_eq!(map.get("年龄").unwrap(), "28");
    }

    #[tokio::test]
    async fn test_structured_parser_fullwidth_separator() {
        // fullwidth colon is 3 bytes: before the fix, slicing at pos+1 landed mid-char and panicked
        let parser = StructuredOutputParser::with_separator('：');
        let map = parser.parse("姓名：张三\n年龄：28").await.unwrap();
        assert_eq!(map.get("姓名").unwrap(), "张三");
        assert_eq!(map.get("年龄").unwrap(), "28");
    }

    #[tokio::test]
    async fn test_structured_parser_runnable_invoke() {
        let parser = StructuredOutputParser::new();
        let map = parser
            .invoke(
                LLMResult {
                    content: "状态: 成功".to_string(),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(map.get("状态").unwrap(), "成功");
    }
}
