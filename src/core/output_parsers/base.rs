use async_trait::async_trait;
use std::fmt;

/// 输出解析器的统一错误类型
#[derive(Debug, Clone)]
pub enum OutputParserError {
    /// 解析失败：输入格式不符合预期
    ParseError(String),
    /// JSON 格式错误
    JsonError(String),
    /// 类型转换错误
    TypeError(String),
    /// 自定义错误
    Custom(String),
}

impl fmt::Display for OutputParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputParserError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            OutputParserError::JsonError(msg) => write!(f, "JSON error: {}", msg),
            OutputParserError::TypeError(msg) => write!(f, "Type error: {}", msg),
            OutputParserError::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for OutputParserError {}

impl From<serde_json::Error> for OutputParserError {
    fn from(e: serde_json::Error) -> Self {
        OutputParserError::JsonError(e.to_string())
    }
}

/// 输出解析器的结果类型
pub type OutputParserResult<T> = Result<T, OutputParserError>;

/// 输出解析器的核心 trait
///
/// 所有输出解析器必须实现此 trait。
/// 与 `Runnable` 不同，`parse` 不接收 config 参数，
/// 适合在 Runnable 内部调用。
#[async_trait]
pub trait BaseOutputParser<Output: Send + Sync + 'static>: Send + Sync {
    /// 将原始 LLM 输出文本解析为目标类型
    async fn parse(&self, text: &str) -> OutputParserResult<Output>;

    /// 带重试的解析（默认实现：不重试）
    async fn parse_with_retry(
        &self,
        text: &str,
        _max_retries: usize,
    ) -> OutputParserResult<Output> {
        self.parse(text).await
    }

    /// 获取格式指令（用于提示 LLM 按指定格式输出）
    fn get_format_instructions(&self) -> String {
        String::new()
    }
}


