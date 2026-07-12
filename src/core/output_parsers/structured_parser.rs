use async_trait::async_trait;
use futures_util::Stream;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::pin::Pin;

use crate::core::runnables::{Runnable, RunnableConfig};
use super::base::{BaseOutputParser, OutputParserError, OutputParserResult};

/// 结构化输出解析器
///
/// 将 LLM 输出的键值对格式（每行一个 `key: value`）解析为 HashMap。
/// 适用于 LLM 以非 JSON 格式输出结构化信息的场景。
///
/// # 格式
/// 输入格式应为每行一个 `key: value`，例如：
/// ```text
/// 姓名: 张三
/// 年龄: 28
/// 城市: 北京
/// ```
///
/// # 示例
/// ```ignore
/// use langchainrust::output_parsers::StructuredOutputParser;
///
/// let parser = StructuredOutputParser::new();
/// let result = parser.parse("姓名: 张三\n年龄: 28").await?;
/// assert_eq!(result.get("姓名").unwrap(), "张三");
/// ```
pub struct StructuredOutputParser {
    /// 键值对之间的分隔符
    separator: char,
}

impl StructuredOutputParser {
    pub fn new() -> Self {
        Self { separator: ':' }
    }

    /// 使用自定义分隔符创建解析器
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
                let key = line[..pos].trim().to_string();
                let value = line[pos + 1..].trim().to_string();

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
impl Runnable<String, HashMap<String, String>> for StructuredOutputParser {
    type Error = OutputParserError;

    async fn invoke(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<HashMap<String, String>, Self::Error> {
        self.parse(&input).await
    }

    async fn stream(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<HashMap<String, String>, Self::Error>> + Send>>, Self::Error> {
        let result = self.parse(&input).await?;
        let stream = futures_util::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }
}

/// 类型化输出解析器
///
/// 将 LLM 输出的 JSON 字符串解析为指定的 Rust 结构体。
/// 相当于 Python LangChain 的 `PydanticOutputParser`（使用 serde 替代 pydantic）。
///
/// 需要目标类型实现 `serde::Deserialize`。
///
/// # 示例
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

        // 尝试从 Markdown 代码块中提取 JSON
        let json_str = Self::extract_from_markdown(text).unwrap_or(text);

        // 先尝试解析为 Value 验证合法性
        serde_json::from_str::<serde_json::Value>(json_str).map_err(|e| {
            OutputParserError::JsonError(format!("输入不是合法 JSON：{}", e))
        })?;

        // 反序列化为目标类型
        serde_json::from_str::<T>(json_str).map_err(|e| {
            OutputParserError::TypeError(format!(
                "类型反序列化失败（请检查 JSON 字段是否匹配）：{}",
                e
            ))
        })
    }
}

impl<T: DeserializeOwned> TypedOutputParser<T> {
    /// 从 Markdown 代码块中提取 JSON 字符串
    fn extract_from_markdown(text: &str) -> Option<&str> {
        // 尝试 ```json ... ```
        if let Some(start) = text.find("```json") {
            let after = &text[start + 7..];
            if let Some(end) = after.find("```") {
                return Some(after[..end].trim());
            }
        }
        // 尝试 ``` ... ```
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

    #[allow(dead_code)]
    fn get_format_instructions(&self) -> String {
        "请输出符合以下 JSON Schema 的合法 JSON：\n```json\n{\n  // 目标类型的字段定义\n}\n```".to_string()
    }
}

#[async_trait]
impl<T: DeserializeOwned + Send + Sync + 'static> Runnable<String, T> for TypedOutputParser<T> {
    type Error = OutputParserError;

    async fn invoke(&self, input: String, _config: Option<RunnableConfig>) -> Result<T, Self::Error> {
        self.parse(&input).await
    }

    async fn stream(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<T, Self::Error>> + Send>>, Self::Error> {
        let result = self.parse(&input).await?;
        let stream = futures_util::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }
}
