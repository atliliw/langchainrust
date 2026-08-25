//! 图片内容类型(多模态 vision 支持)

use serde::{Deserialize, Serialize};

/// 图片内容(URL 或 base64 data URI)
///
/// OpenAI Vision 使用 `image_url.url`(可为 https URL 或 `data:image/...;base64,...`),
/// Ollama 使用 base64 原始数据。本类型统一存储为 `url` 字段,
/// 由各 provider 在序列化时转换。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageContent {
    /// 图片 URL 或 base64 data URI
    pub url: String,
}

impl ImageContent {
    /// 从 URL 创建
    pub fn from_url(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }

    /// 从 base64 数据创建(自动包装为 data URI)
    pub fn from_base64(data: impl Into<String>) -> Self {
        Self {
            url: format!("data:image/png;base64,{}", data.into()),
        }
    }

    /// 从指定 MIME 类型的 base64 创建
    pub fn from_base64_with_mime(data: impl Into<String>, mime: &str) -> Self {
        Self {
            url: format!("data:{};base64,{}", mime, data.into()),
        }
    }

    /// 是否为 base64 data URI
    pub fn is_base64(&self) -> bool {
        self.url.starts_with("data:")
    }

    /// 提取 base64 原始数据(若是 data URI)
    pub fn base64_data(&self) -> Option<&str> {
        self.url
            .split_once(',')
            .filter(|(prefix, _)| prefix.contains("base64"))
            .map(|(_, data)| data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_url() {
        let img = ImageContent::from_url("https://example.com/image.jpg");
        assert_eq!(img.url, "https://example.com/image.jpg");
        assert!(!img.is_base64());
    }

    #[test]
    fn test_from_base64() {
        let img = ImageContent::from_base64("abc123");
        assert!(img.is_base64());
        assert_eq!(img.base64_data(), Some("abc123"));
    }

    #[test]
    fn test_from_base64_with_mime() {
        let img = ImageContent::from_base64_with_mime("xyz", "image/jpeg");
        assert!(img.url.starts_with("data:image/jpeg;base64,"));
        assert_eq!(img.base64_data(), Some("xyz"));
    }

    #[test]
    fn test_url_not_base64() {
        let img = ImageContent::from_url("https://example.com/img.png");
        assert_eq!(img.base64_data(), None);
    }
}
