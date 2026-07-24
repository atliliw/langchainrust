use async_trait::async_trait;

/// 输出解析器的统一错误类型
#[derive(Debug, Clone, thiserror::Error)]
pub enum OutputParserError {
    /// 解析失败：输入格式不符合预期
    #[error("Parse error: {0}")]
    ParseError(String),
    /// JSON 格式错误
    #[error("JSON error: {0}")]
    JsonError(String),
    /// 类型转换错误
    #[error("Type error: {0}")]
    TypeError(String),
    /// 自定义错误
    #[error("{0}")]
    Custom(String),
}

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
