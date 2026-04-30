use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use crate::core::runnables::{Runnable, RunnableConfig};
use super::base::{BaseOutputParser, OutputParserError, OutputParserResult};

/// JSON 输出解析器
///
/// 将 LLM 输出的 JSON 字符串解析为 `serde_json::Value`。
/// 支持：
/// - 标准 JSON 解析
/// - 从 Markdown 代码块中提取 JSON
/// - 可选的部分 JSON 解析（用于流式场景）
///
/// 相当于 Python LangChain 的 `JsonOutputParser`。
///
/// # 示例
/// ```ignore
/// use langchainrust::output_parsers::JsonOutputParser;
/// use serde_json::json;
///
/// let parser = JsonOutputParser::new();
/// let result = parser.parse(r#"{"name": "Rust", "year": 2015}"#).await?;
/// assert_eq!(result["name"], "Rust");
/// ```
pub struct JsonOutputParser {
    /// 是否允许部分 JSON 解析（用于流式场景）
    partial: bool,
}

impl JsonOutputParser {
    pub fn new() -> Self {
        Self { partial: false }
    }

    /// 创建支持部分 JSON 解析的解析器
    ///
    /// 在流式场景中，LLM 可能输出不完整的 JSON，
    /// 启用此选项后会尝试从中解析尽可能多的数据。
    pub fn new_partial() -> Self {
        Self { partial: true }
    }

    /// 从文本中提取 JSON 字符串
    ///
    /// 支持从 Markdown 代码块 ```json ... ``` 中提取 JSON。
    fn extract_json_str<'a>(&self, text: &'a str) -> OutputParserResult<&'a str> {
        let text = text.trim();

        // 尝试从 ```json 代码块中提取
        if let Some(start) = text.find("```json") {
            let content = &text[start + 7..];
            if let Some(end) = content.find("```") {
                return Ok(content[..end].trim());
            }
        }

        // 尝试从 ``` 代码块中提取
        if let Some(start) = text.find("```") {
            let content = &text[start + 3..];
            let content = content.trim();
            // 可能后面还跟了语言标识，跳过到换行
            let skip_to_newline = content.find('\n').unwrap_or(0);
            let content = &content[skip_to_newline..];
            if let Some(end) = content.find("```") {
                return Ok(content[..end].trim());
            }
        }

        // 直接作为 JSON 解析
        Ok(text)
    }
}

impl Default for JsonOutputParser {
    fn default() -> Self {
        Self::new()
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
                    "JSON 解析失败（位置 {}:{}）：{}，输入：{}",
                    e.line(),
                    e.column(),
                    e,
                    &json_str[..std::cmp::min(200, json_str.len())]
                ))
            })
        }
    }

    fn get_format_instructions(&self) -> String {
        "请使用 JSON 格式输出，例如：{\"key\": \"value\"}。确保 JSON 是合法的。".to_string()
    }
}

impl JsonOutputParser {
    /// 尝试解析部分（不完整）JSON
    ///
    /// 在 LLM 流式输出场景中，逐步累积的 JSON 可能是不完整的。
    /// 此方法尝试从中提取尽可能多的数据。
    fn parse_partial_json(&self, text: &str) -> OutputParserResult<serde_json::Value> {
        // 先尝试完整解析
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
            return Ok(value);
        }

        // 尝试修复常见的不完整 JSON 模式
        let repaired = self.repair_partial_json(text);
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&repaired) {
            return Ok(value);
        }

        Err(OutputParserError::JsonError(format!(
            "部分 JSON 解析失败：{}",
            &text[..std::cmp::min(200, text.len())]
        )))
    }

    /// 修复不完整的 JSON 字符串
    ///
    /// 处理常见的不完整 JSON 格式，如：
    /// - 末尾多余的逗号
    /// - 不完整的字符串
    /// - 不完整的对象/数组
    fn repair_partial_json(&self, text: &str) -> String {
        let mut repaired = text.trim().to_string();

        // 处理以 `"` 结束的不完整字符串（去掉最后一个不完整的 token）
        if let Some(stripped) = Self::strip_incomplete_token(&repaired) {
            repaired = stripped;
        }

        // 补全括号
        let open_braces = repaired.matches('{').count();
        let close_braces = repaired.matches('}').count();
        for _ in close_braces..open_braces {
            repaired.push('}');
        }

        let open_brackets = repaired.matches('[').count();
        let close_brackets = repaired.matches(']').count();
        for _ in close_brackets..open_brackets {
            repaired.push(']');
        }

        // 确保字符串以引号结束（如果开始了一个字符串）
        // 去掉最后一个不完整的字符串片段
        let mut chars = repaired.chars().rev().peekable();
        let mut in_string = false;
        let mut truncate_at = repaired.len();

        while let Some(c) = chars.next() {
            if c == '"' {
                // 前面是转义字符？
                if chars.peek() == Some(&'\\') {
                    continue;
                }
                in_string = !in_string;
            }
            if in_string && c == '\n' {
                truncate_at = repaired.len() - chars.count() - 1;
                break;
            }
        }

        if in_string && truncate_at < repaired.len() {
            repaired.truncate(truncate_at);
            repaired.push('"');
        }

        repaired
    }

    /// 去掉末尾的不完整 token
    fn strip_incomplete_token(s: &str) -> Option<String> {
        // 如果以字母/数字结尾，尝试找到最后一个完整的 token 边界
        let trimmed = s.trim_end();
        if trimmed.len() < s.len() {
            // 末尾有空白，可能是完整的
            return None;
        }

        // 检查是否以引号开始但不完整
        if trimmed.ends_with('"') {
            return None; // 以引号结束，可能是完整的
        }

        None
    }
}

#[async_trait]
impl Runnable<String, serde_json::Value> for JsonOutputParser {
    type Error = OutputParserError;

    async fn invoke(&self, input: String, _config: Option<RunnableConfig>) -> Result<serde_json::Value, Self::Error> {
        self.parse(&input).await
    }

    async fn stream(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<serde_json::Value, Self::Error>> + Send>>, Self::Error> {
        let result = self.parse(&input).await?;
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
        let result = parser.parse(r#"{"name": "Rust", "year": 2015}"#).await.unwrap();
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
        let parser = JsonOutputParser::new();
        let result = parser.invoke(r#"{"key": "value"}"#.to_string(), None).await.unwrap();
        assert_eq!(result["key"], "value");
    }

    #[tokio::test]
    async fn test_json_parser_partial_success() {
        let parser = JsonOutputParser::new_partial();
        // 完整 JSON，partial 模式也应该能解析
        let result = parser.parse(r#"{"a": 1}"#).await.unwrap();
        assert_eq!(result["a"], 1);
    }
}
