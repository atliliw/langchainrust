use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

use super::base::{BaseOutputParser, OutputParserError, OutputParserResult};
use crate::language_models::LLMResult;
use crate::runnables::{Runnable, RunnableConfig};
use crate::structured_output::parser::PartialJsonParser;

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
    /// 剥掉 Markdown 代码块 ```json ... ```、前导/尾随文本,返回真正的 JSON 值。
    /// 比旧的 `find("```")` 匹配更稳:未闭合的围栏(只有开头没有结尾)也能正确剥离,
    /// 且带围栏的完整 JSON 不会被误判为解析失败。
    fn extract_json_str<'a>(&self, text: &'a str) -> OutputParserResult<&'a str> {
        let json = PartialJsonParser::strip_markdown_fence(text);
        if json.is_empty() {
            // 没有 JSON 结构字符:交给调用方 serde 报错(避免返回空串被当成合法值)
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

/// 取字符串前 `max_chars` 个字符用于错误预览。
///
/// 不能用字节截断:多字节 UTF-8 字符会被切在字符中间导致切片 panic
/// (非法 CJK JSON 的错误路径曾按字节 200 截断而崩溃)。
fn preview_slice(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        // 第 max_chars 个字符的起始字节是安全边界,切到它即保留前 max_chars 个字符
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
                    "JSON 解析失败（位置 {}:{}）：{}，输入：{}",
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
            preview_slice(text, 200)
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

        // Scan tracking string state to correctly count braces/brackets
        let mut in_string = false;
        let mut escape_next = false;
        let mut open_braces = 0usize;
        let mut close_braces = 0usize;
        let mut open_brackets = 0usize;
        let mut close_brackets = 0usize;

        for ch in repaired.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }
            if ch == '\\' && in_string {
                escape_next = true;
                continue;
            }
            if ch == '"' {
                in_string = !in_string;
                continue;
            }
            if !in_string {
                match ch {
                    '{' => open_braces += 1,
                    '}' => close_braces += 1,
                    '[' => open_brackets += 1,
                    ']' => close_brackets += 1,
                    _ => {}
                }
            }
        }

        // 补全括号
        for _ in close_braces..open_braces {
            repaired.push('}');
        }

        for _ in close_brackets..open_brackets {
            repaired.push(']');
        }

        // 确保字符串以引号结束（如果开始了一个字符串）
        // Scan forward (not backward) to find unclosed strings (M31)
        let mut in_string = false;
        let mut escape_next = false;
        let mut last_open_quote_pos: Option<usize> = None;

        for (i, ch) in repaired.char_indices() {
            if escape_next {
                escape_next = false;
                continue;
            }
            if ch == '\\' && in_string {
                escape_next = true;
                continue;
            }
            if ch == '"' {
                if in_string {
                    in_string = false;
                    last_open_quote_pos = None;
                } else {
                    in_string = true;
                    last_open_quote_pos = Some(i);
                }
                continue;
            }
        }

        // If we're still in a string at the end, truncate at the opening quote
        // and close it, or just close it if the string value is partially complete
        if in_string {
            // Check if the unclosed string contains a newline (invalid in JSON strings)
            if let Some(open_pos) = last_open_quote_pos {
                let after_quote = &repaired[open_pos + 1..];
                if after_quote.contains('\n') {
                    // Truncate at the newline and close the string
                    let newline_pos = repaired[open_pos + 1..].find('\n').unwrap() + open_pos + 1;
                    repaired.truncate(newline_pos);
                    repaired.push('"');
                }
            }
        }

        repaired
    }

    /// 去掉末尾的不完整 token
    fn strip_incomplete_token(s: &str) -> Option<String> {
        let trimmed = s.trim_end();

        // If the string ends with an incomplete key or value token,
        // try to find the last complete token boundary.
        // Look for the last structural character (: , { [ }) and truncate after it.
        let chars: Vec<char> = trimmed.chars().collect();
        if chars.is_empty() {
            return None;
        }

        // Scan backwards to find the last structural boundary
        let mut i = chars.len();
        while i > 0 {
            i -= 1;
            match chars[i] {
                ',' | ':' | '{' | '[' | '}' | ']' => {
                    // Found a structural character; truncate after it
                    let truncate_at: usize = trimmed
                        .char_indices()
                        .nth(i + 1)
                        .map(|(pos, _)| pos)
                        .unwrap_or(trimmed.len());
                    if truncate_at < s.len() {
                        let result = trimmed[..truncate_at].to_string();
                        if result != s.trim_end() {
                            return Some(result);
                        }
                    }
                    return None;
                }
                '"' => {
                    // Check if this is a closing quote (even number of quotes before it)
                    // If so, the JSON might be complete at this point
                    return None;
                }
                _ => {}
            }
        }

        None
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
        // H4: 只有开头 ```json 没有结尾 ```(模型输出被截断)也要能剥掉围栏解析
        let parser = JsonOutputParser::new();
        let input = "以下是结果：\n```json\n{\"status\": \"ok\"}";
        let result = parser.parse(input).await.unwrap();
        assert_eq!(result["status"], "ok");
    }

    #[tokio::test]
    async fn test_json_parser_from_prose_prefix() {
        // H4: 模型先输出一句"结果是:"再给 JSON,也要剥掉前导文本
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
        // Runnable 形态接收 LLMResult,取 content 字段解析
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
        // 完整 JSON，partial 模式也应该能解析
        let result = parser.parse(r#"{"a": 1}"#).await.unwrap();
        assert_eq!(result["a"], 1);
    }

    #[tokio::test]
    async fn test_json_parser_invalid_cjk_over_200_bytes() {
        // >200 字节的非法中文 JSON:错误路径若按字节 200 截断会切在多字节字符中间 panic,
        // 修复后应返回 Err 而非崩溃
        let parser = JsonOutputParser::new();
        let long_cjk = "汉".repeat(200);
        let bad = format!("{{\"名字\": {}", long_cjk);
        let result = parser.parse(&bad).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_json_parser_partial_invalid_cjk_over_200_bytes() {
        // partial 模式同样走错误预览截断,需同样不 panic
        let parser = JsonOutputParser::new_partial();
        let long_cjk = "汉".repeat(200);
        let bad = format!("{{\"名字\": {}", long_cjk);
        let result = parser.parse(&bad).await;
        assert!(result.is_err());
    }
}
