use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use super::base::{BaseOutputParser, OutputParserError, OutputParserResult};
use crate::language_models::LLMResult;
use crate::runnables::{Runnable, RunnableConfig};
use crate::structured_output::parser::PartialJsonParser;

/// JSON output parser
///
/// Parses the LLM's JSON string output into a `serde_json::Value`.
/// Supports:
/// - standard JSON parsing
/// - extracting JSON from a Markdown code block
/// - optional partial-JSON parsing (for streaming scenarios)
///
/// Equivalent to Python LangChain's `JsonOutputParser`.
///
/// # Example
/// ```ignore
/// use langchainrust::output_parsers::JsonOutputParser;
/// use serde_json::json;
///
/// let parser = JsonOutputParser::new();
/// let result = parser.parse(r#"{"name": "Rust", "year": 2015}"#).await?;
/// assert_eq!(result["name"], "Rust");
/// ```
pub struct JsonOutputParser {
    /// Whether partial-JSON parsing is allowed (for streaming scenarios)
    partial: bool,
}

impl JsonOutputParser {
    /// Creates a standard JSON output parser.
    pub fn new() -> Self {
        Self { partial: false }
    }

    /// Creates a parser that supports partial-JSON parsing
    ///
    /// In streaming scenarios the LLM may emit incomplete JSON;
    /// with this option enabled it tries to parse as much data out as possible.
    pub fn new_partial() -> Self {
        Self { partial: true }
    }

    /// Extracts the JSON string from text
    ///
    /// Strips the Markdown code block ```json ... ``` and leading/trailing text, returning the actual JSON value.
    /// More robust than the old `find("```")` matching: an unclosed fence (opening only, no closing) is stripped
    /// correctly, and a complete fenced JSON is not misread as a parse failure.
    fn extract_json_str<'a>(&self, text: &'a str) -> OutputParserResult<&'a str> {
        let json = PartialJsonParser::strip_markdown_fence(text);
        if json.is_empty() {
            // no JSON structural chars: let the caller's serde report the error (avoid treating an empty string as a valid value)
            Ok(text.trim())
        } else {
            Ok(json.trim())
        }
    }
}

impl Default for JsonOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Takes the first `max_chars` chars of a string for an error preview.
///
/// Byte truncation cannot be used: a multi-byte UTF-8 char would be cut mid-char, panicking on slicing
/// (the error path for invalid CJK JSON once crashed truncating at 200 bytes).
fn preview_slice(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        // the start byte of the max_chars-th char is a safe boundary; slicing to it keeps the first max_chars chars
        Some((i, _)) => &s[..i],
        None => s,
    }
}

#[async_trait]
impl BaseOutputParser<serde_json::Value> for JsonOutputParser {
    async fn parse(&self, text: &str) -> OutputParserResult<serde_json::Value> {
        let json_str = self.extract_json_str(text)?;

        if self.partial {
            self.parse_partial_json(json_str)
        } else {
            serde_json::from_str(json_str).map_err(|e| {
                OutputParserError::JsonError(format!(
                    "JSON parse failed (position {}:{}): {}, input: {}",
                    e.line(),
                    e.column(),
                    e,
                    preview_slice(json_str, 200)
                ))
            })
        }
    }

    fn get_format_instructions(&self) -> String {
        "请使用 JSON 格式输出，例如：{\"key\": \"value\"}。确保 JSON 是合法的。".to_string()
    }
}

impl JsonOutputParser {
    /// Tries to parse partial (incomplete) JSON
    ///
    /// In streaming LLM output the progressively accumulated JSON may be incomplete.
    /// This method attempts to extract as much data from it as possible.
    fn parse_partial_json(&self, text: &str) -> OutputParserResult<serde_json::Value> {
        // first try a complete parse
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
            return Ok(value);
        }

        // Repair via the canonical LIFO implementation shared with structured
        // output (`PartialJsonParser::repair_partial_json`) — not a local,
        // counter-based one. 0.26.0 §3.1/§3.4: the two previously drifted; the
        // counter-based closer produced illegal JSON for nested truncation
        // (`{"a": [1, 2` → `{"a": [1, 2}]` instead of `{"a": [1, 2]}`). A single
        // source of truth now.
        let repaired = PartialJsonParser::repair_partial_json(text);
        serde_json::from_str::<serde_json::Value>(&repaired).map_err(|_| {
            OutputParserError::JsonError(format!(
                "partial JSON parse failed: {}",
                preview_slice(text, 200)
            ))
        })
    }
}

#[async_trait]
impl Runnable<LLMResult, serde_json::Value> for JsonOutputParser {
    type Error = OutputParserError;

    async fn invoke(
        &self,
        input: LLMResult,
        _config: Option<RunnableConfig>,
    ) -> Result<serde_json::Value, Self::Error> {
        self.parse(&input.content).await
    }

    async fn stream(
        &self,
        input: LLMResult,
        _config: Option<RunnableConfig>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<serde_json::Value, Self::Error>> + Send>>,
        Self::Error,
    > {
        let result = self.parse(&input.content).await?;
        let stream = futures_util::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_json_parser_standard_obj() {
        let parser = JsonOutputParser::new();
        let result = parser
            .parse(r#"{"name": "Rust", "year": 2015}"#)
            .await
            .unwrap();
        assert_eq!(result["name"], "Rust");
        assert_eq!(result["year"], 2015);
    }

    #[tokio::test]
    async fn test_json_parser_from_markdown_block() {
        let parser = JsonOutputParser::new();
        let input = "以下是结果：\n```json\n{\"status\": \"ok\"}\n```\n";
        let result = parser.parse(input).await.unwrap();
        assert_eq!(result["status"], "ok");
    }

    #[tokio::test]
    async fn test_json_parser_array() {
        let parser = JsonOutputParser::new();
        let result = parser.parse("[1, 2, 3]").await.unwrap();
        assert_eq!(result[0], 1);
        assert_eq!(result[2], 3);
    }

    #[tokio::test]
    async fn test_json_parser_from_markdown_block_unclosed_fence() {
        // H4: only the opening ```json without the closing ``` (truncated model output) must also strip the fence and parse
        let parser = JsonOutputParser::new();
        let input = "以下是结果：\n```json\n{\"status\": \"ok\"}";
        let result = parser.parse(input).await.unwrap();
        assert_eq!(result["status"], "ok");
    }

    #[tokio::test]
    async fn test_json_parser_from_prose_prefix() {
        // H4: a prose prefix ("result:") before the JSON must also be stripped
        let parser = JsonOutputParser::new();
        let input = "结果是：\n{\"a\": 1}\n以上";
        let result = parser.parse(input).await.unwrap();
        assert_eq!(result["a"], 1);
    }

    #[tokio::test]
    async fn test_json_parser_invalid_json() {
        let parser = JsonOutputParser::new();
        let result = parser.parse("{invalid}").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_json_parser_format_instructions() {
        let parser = JsonOutputParser::new();
        let instructions = parser.get_format_instructions();
        assert!(!instructions.is_empty());
    }

    #[tokio::test]
    async fn test_json_parser_invoke_runnable() {
        // Runnable form takes an LLMResult and parses its content field
        let parser = JsonOutputParser::new();
        let result = parser
            .invoke(
                LLMResult {
                    content: r#"{"key": "value"}"#.to_string(),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(result["key"], "value");
    }

    #[tokio::test]
    async fn test_json_parser_partial_success() {
        let parser = JsonOutputParser::new_partial();
        // complete JSON: partial mode must also parse it
        let result = parser.parse(r#"{"a": 1}"#).await.unwrap();
        assert_eq!(result["a"], 1);
    }

    #[tokio::test]
    async fn test_json_parser_partial_nested_closes_in_lifo_order() {
        // 0.26.0 §3.1: nested structures must close LIFO (inner array first),
        // not "all braces then all brackets". `{"a": [1, 2` with the counter
        // repair became `{"a": [1, 2}]` (illegal); the shared LIFO repair gives
        // `{"a": [1, 2]}` which parses to {"a":[1,2]}.
        let parser = JsonOutputParser::new_partial();
        let result = parser.parse(r#"{"a": [1, 2"#).await.unwrap();
        assert_eq!(result.get("a").unwrap()[1], 2);
        assert_eq!(result["a"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_json_parser_invalid_cjk_over_200_bytes() {
        // >200-byte invalid CJK JSON: if the error path truncated at 200 bytes it would panic mid multi-byte char;
        // after the fix it returns Err instead of crashing
        let parser = JsonOutputParser::new();
        let long_cjk = "汉".repeat(200);
        let bad = format!("{{\"名字\": {}", long_cjk);
        let result = parser.parse(&bad).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_json_parser_partial_invalid_cjk_over_200_bytes() {
        // partial mode uses the same error-preview truncation; must also not panic
        let parser = JsonOutputParser::new_partial();
        let long_cjk = "汉".repeat(200);
        let bad = format!("{{\"名字\": {}", long_cjk);
        let result = parser.parse(&bad).await;
        assert!(result.is_err());
    }
}
