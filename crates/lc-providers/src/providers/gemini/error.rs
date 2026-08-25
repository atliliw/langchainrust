// lc-providers/src/providers/gemini/error.rs
//! Error types for the Gemini provider.

/// Gemini 错误类型
#[derive(Debug)]
#[non_exhaustive]
pub enum GeminiError {
    /// Gemini API 返回错误
    ApiError(String),
    /// HTTP 请求错误
    HttpError(String),
    /// 响应解析错误
    ParseError(String),
    /// 无响应
    NoResponse,
    /// 安全过滤器拦截
    SafetyBlock(String),
}

impl std::fmt::Display for GeminiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeminiError::ApiError(msg) => write!(f, "Gemini API error: {}", msg),
            GeminiError::HttpError(msg) => write!(f, "Gemini HTTP error: {}", msg),
            GeminiError::ParseError(msg) => write!(f, "Gemini parse error: {}", msg),
            GeminiError::NoResponse => write!(f, "Gemini returned no response"),
            GeminiError::SafetyBlock(msg) => write!(f, "Gemini blocked by safety filter: {}", msg),
        }
    }
}

impl std::error::Error for GeminiError {}

// L2 fix: add From<String> for GeminiError, matching OpenAIError pattern
impl From<String> for GeminiError {
    fn from(s: String) -> Self {
        GeminiError::ApiError(s)
    }
}
