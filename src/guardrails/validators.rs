//! 内置 Guardrail 验证器

use async_trait::async_trait;
use regex::Regex;

use super::guardrail::{GuardrailResult, InputGuardrail, OutputGuardrail};

/// 输入长度限制
pub struct MaxLengthGuardrail {
    max: usize,
}

impl MaxLengthGuardrail {
    pub fn new(max: usize) -> Self {
        Self { max }
    }
}

#[async_trait]
impl InputGuardrail for MaxLengthGuardrail {
    fn name(&self) -> &str {
        "MaxLength"
    }

    async fn validate(&self, input: &str) -> GuardrailResult {
        if input.chars().count() > self.max {
            GuardrailResult::Block {
                reason: format!("输入超过 {} 字符", self.max),
            }
        } else {
            GuardrailResult::Pass
        }
    }
}

/// 禁用词检查
pub struct ForbiddenWordsGuardrail {
    words: Vec<String>,
}

impl ForbiddenWordsGuardrail {
    pub fn new(words: Vec<String>) -> Self {
        Self { words }
    }
}

#[async_trait]
impl InputGuardrail for ForbiddenWordsGuardrail {
    fn name(&self) -> &str {
        "ForbiddenWords"
    }

    async fn validate(&self, input: &str) -> GuardrailResult {
        let lower = input.to_lowercase();
        for w in &self.words {
            if lower.contains(&w.to_lowercase()) {
                return GuardrailResult::Block {
                    reason: format!("输入包含禁用词: {}", w),
                };
            }
        }
        GuardrailResult::Pass
    }
}

/// 敏感信息检测(API key / email / 信用卡 / 关键词)
pub struct SensitiveInfoGuardrail {
    patterns: Vec<Regex>,
    keywords: Vec<String>,
}

impl SensitiveInfoGuardrail {
    pub fn new() -> Self {
        Self {
            patterns: vec![
                Regex::new(r"[sS][kK]-[a-zA-Z0-9]{20,}").unwrap(), // OpenAI key
                Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
                    .unwrap(), // email
                Regex::new(r"\b\d{16,19}\b").unwrap(), // 信用卡号
            ],
            keywords: vec![
                "password".to_string(),
                "密码".to_string(),
                "token".to_string(),
                "secret".to_string(),
                "api_key".to_string(),
                "credential".to_string(),
            ],
        }
    }

    pub fn with_keywords(mut self, k: Vec<String>) -> Self {
        self.keywords.extend(k);
        self
    }
}

impl Default for SensitiveInfoGuardrail {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OutputGuardrail for SensitiveInfoGuardrail {
    fn name(&self) -> &str {
        "SensitiveInfo"
    }

    async fn validate(&self, output: &str) -> GuardrailResult {
        let lower = output.to_lowercase();
        for kw in &self.keywords {
            if lower.contains(&kw.to_lowercase()) {
                return GuardrailResult::Block {
                    reason: format!("输出包含敏感关键词: {}", kw),
                };
            }
        }
        for p in &self.patterns {
            if p.is_match(output) {
                return GuardrailResult::Block {
                    reason: format!("输出匹配敏感模式: {}", p.as_str()),
                };
            }
        }
        GuardrailResult::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_max_length_pass() {
        let g = MaxLengthGuardrail::new(10);
        assert!(g.validate("hi").await.is_pass());
    }

    #[tokio::test]
    async fn test_max_length_block() {
        let g = MaxLengthGuardrail::new(3);
        assert!(g.validate("hello world").await.is_block());
    }

    #[tokio::test]
    async fn test_forbidden_words() {
        let g = ForbiddenWordsGuardrail::new(vec!["spam".to_string()]);
        assert!(g.validate("this is spam").await.is_block());
        assert!(g.validate("this is fine").await.is_pass());
    }

    #[tokio::test]
    async fn test_forbidden_words_case_insensitive() {
        let g = ForbiddenWordsGuardrail::new(vec!["BAD".to_string()]);
        assert!(g.validate("this is bad").await.is_block());
    }

    #[tokio::test]
    async fn test_sensitive_info_keyword() {
        let g = SensitiveInfoGuardrail::new();
        assert!(g.validate("your password is 123").await.is_block());
        assert!(g.validate("hello world").await.is_pass());
    }

    #[tokio::test]
    async fn test_sensitive_info_api_key() {
        let g = SensitiveInfoGuardrail::new();
        assert!(g.validate("key: sk-abcdefghijklmnopqrstuvwxyz123456").await.is_block());
    }

    #[tokio::test]
    async fn test_sensitive_info_email() {
        let g = SensitiveInfoGuardrail::new();
        assert!(g.validate("contact: user@example.com").await.is_block());
    }

    #[tokio::test]
    async fn test_sensitive_info_custom_keywords() {
        let g = SensitiveInfoGuardrail::new().with_keywords(vec!["机密".to_string()]);
        assert!(g.validate("这是机密信息").await.is_block());
    }
}
