//! 内置 Guardrail 验证器

use async_trait::async_trait;
use regex::Regex;
use std::sync::{Arc, LazyLock};

use crate::judge::SensitiveJudge;

use super::guardrail::{
    InputGuardrail, InputGuardrailResult, OutputGuardrail, OutputGuardrailResult,
};

static OPENAI_KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[sS][kK]-[a-zA-Z0-9]{20,}").unwrap());
static EMAIL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap());
static CREDIT_CARD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(\d[\s-]*){15,18}\d\b").unwrap());

/// 高误报"提及"词(P2-2):普通提及(如"如何安全保存密码")不算泄露,
/// 默认只 `log::warn` 不 Block;具体模式(API key / email / 信用卡)与
/// `with_keywords` 自定义的低误报词才直接 Block。
const MENTION_KEYWORDS: &[&str] = &["password", "密码", "token", "secret"];

/// 为关键词构造"上下文敏感"匹配(P2-1):只有赋值/声明结构(值相邻)才命中,
/// 普通提及不命中。覆盖:
/// - `password=abc` / `password: abc` / `"password": "abc"` / `password is abc`
/// - `密码是abc` / `密码为abc`
fn mention_context_re(keyword: &str) -> Regex {
    let kw = regex::escape(keyword);
    Regex::new(&format!(
        r#"(?i)(?:{kw}["']?\s*[=:]\s*\S|{kw}\s+is\s+\S|{kw}是|{kw}为)"#
    ))
    .expect("invalid mention-context regex")
}

static MENTION_CONTEXT_RES: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    MENTION_KEYWORDS
        .iter()
        .map(|kw| (*kw, mention_context_re(kw)))
        .collect()
});

/// 探测文本中"以赋值/声明结构出现"的第一个高误报提及词(P2-1)。
///
/// 返回 `None` = 只有普通提及(如"如何安全保存密码"),这是 warn 判定与
/// Block 判定分流的关键:普通提及直接放行(可配 LLM 裁判二次判断,见 P2-3)。
pub(crate) fn detect_mention_context(text: &str) -> Option<&'static str> {
    for (kw, re) in MENTION_CONTEXT_RES.iter() {
        if re.is_match(text) {
            return Some(kw);
        }
    }
    None
}

/// 输入长度限制
pub struct MaxLengthGuardrail {
    max: usize,
}

impl MaxLengthGuardrail {
    /// 创建一个最大长度限制的输入护栏。
    pub fn new(max: usize) -> Self {
        Self { max }
    }
}

#[async_trait]
impl InputGuardrail for MaxLengthGuardrail {
    fn name(&self) -> &str {
        "MaxLength"
    }

    async fn validate(&self, input: &str) -> InputGuardrailResult {
        if input.chars().count() > self.max {
            InputGuardrailResult::Block {
                reason: format!("input exceeds {} characters", self.max),
            }
        } else {
            InputGuardrailResult::Pass
        }
    }
}

/// 禁用词检查
pub struct ForbiddenWordsGuardrail {
    words: Vec<String>,
}

impl ForbiddenWordsGuardrail {
    /// 创建一个检查禁用词的输入护栏。
    pub fn new(words: Vec<String>) -> Self {
        Self { words }
    }
}

#[async_trait]
impl InputGuardrail for ForbiddenWordsGuardrail {
    fn name(&self) -> &str {
        "ForbiddenWords"
    }

    async fn validate(&self, input: &str) -> InputGuardrailResult {
        let lower = input.to_lowercase();
        for w in &self.words {
            if lower.contains(&w.to_lowercase()) {
                return InputGuardrailResult::Block {
                    reason: format!("input contains forbidden word: {}", w),
                };
            }
        }
        InputGuardrailResult::Pass
    }
}

/// 敏感信息检测(API key / email / 信用卡 / 关键词)
///
/// 检测词表按误报风险分级(P2-2):
/// - 低误报具体模式:API key(`sk-…`)、email、Luhn 校验通过的信用卡号 → 直接 Block;
/// - 低误报自定义关键词(`with_keywords`):任何出现即 Block(用户显式选择);
/// - 高误报提及词(password/密码/token/secret):上下文敏感命中(P2-1)后
///   仅 `log::warn` 不 Block;配置了 [`SensitiveJudge`](P2-3) 时由 LLM 裁判
///   二次判断"真实泄露 vs 正常提及",判泄露才 Block。
pub struct SensitiveInfoGuardrail {
    /// 低误报自定义关键词:任何出现即 Block(用户显式选择)。
    keywords: Vec<String>,
    /// 可选的 LLM 裁判(P2-3):高误报提及词上下文敏感命中后二次判断。
    judge: Option<Arc<dyn SensitiveJudge>>,
}

impl SensitiveInfoGuardrail {
    /// 创建敏感信息检测护栏(默认含 api_key / credential 低误报关键词)。
    pub fn new() -> Self {
        Self {
            // password/密码/token/secret 移到高误报提及词(仅 warn),不再默认 Block。
            keywords: vec!["api_key".to_string(), "credential".to_string()],
            judge: None,
        }
    }

    /// 追加自定义低误报关键词(任何出现即拦截)。
    pub fn with_keywords(mut self, k: Vec<String>) -> Self {
        self.keywords.extend(k);
        self
    }

    /// 挂载 LLM 裁判(P2-3):高误报提及词上下文敏感命中后二次判断真实泄露才 Block。
    pub fn with_judge(mut self, judge: Arc<dyn SensitiveJudge>) -> Self {
        self.judge = Some(judge);
        self
    }

    /// Luhn checksum validation for credit card numbers.
    /// Returns true if the digit sequence passes the Luhn check.
    fn luhn_check(digits: &str) -> bool {
        let digits: Vec<u32> = digits.chars().filter_map(|c| c.to_digit(10)).collect();
        if digits.len() < 13 || digits.len() > 19 {
            return false;
        }
        let sum: u32 = digits
            .iter()
            .rev()
            .enumerate()
            .map(|(i, &d)| {
                if i % 2 == 1 {
                    let doubled = d * 2;
                    doubled / 10 + doubled % 10
                } else {
                    d
                }
            })
            .sum();
        sum % 10 == 0
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

    async fn validate(&self, output: &str) -> OutputGuardrailResult {
        // 1) 低误报具体模式 → 直接 Block,无需裁判。
        if OPENAI_KEY_RE.is_match(output) {
            return OutputGuardrailResult::Block {
                reason: format!(
                    "output matches sensitive pattern: {}",
                    OPENAI_KEY_RE.as_str()
                ),
            };
        }
        if EMAIL_RE.is_match(output) {
            return OutputGuardrailResult::Block {
                reason: format!("output matches sensitive pattern: {}", EMAIL_RE.as_str()),
            };
        }
        // Credit card: match pattern then validate with Luhn
        for cap in CREDIT_CARD_RE.captures_iter(output) {
            let digits: String = cap[0].chars().filter(|c| c.is_ascii_digit()).collect();
            if Self::luhn_check(&digits) {
                return OutputGuardrailResult::Block {
                    reason: "output contains a credit card number".to_string(),
                };
            }
        }

        // 2) 低误报自定义关键词 → 直接 Block(用户显式选择)。
        let lower = output.to_lowercase();
        for kw in &self.keywords {
            if lower.contains(&kw.to_lowercase()) {
                return OutputGuardrailResult::Block {
                    reason: format!("output contains sensitive keyword: {}", kw),
                };
            }
        }

        // 3) 高误报提及词:仅上下文敏感命中(P2-1)才处理。
        //    有 LLM 裁判 → 二次判断"真实泄露 vs 正常提及",判泄露才 Block(P2-3);
        //    无裁判 → 只 warn 不 Block(P2-2),普通提及("如何安全保存密码")直接放行。
        if let Some(kw) = detect_mention_context(output) {
            match &self.judge {
                Some(judge) => match judge.judge(output).await {
                    Ok(true) => {
                        return OutputGuardrailResult::Block {
                            reason: format!(
                                "LLM judge determined the output likely leaks sensitive information (keyword: {})",
                                kw
                            ),
                        };
                    }
                    Ok(false) => {
                        log::info!(
                            "SensitiveInfo: keywords {:?} judged as normal mention by LLM, passing",
                            kw
                        );
                    }
                    Err(e) => {
                        // 裁判失败不误杀:回落到无裁判的 warn-only 行为(宁可放行并留日志)。
                        log::warn!(
                            "SensitiveInfo: LLM judge call failed ({}), keywords {:?} handled as warn-only",
                            e,
                            kw
                        );
                    }
                },
                None => {
                    log::warn!(
                        "SensitiveInfo: output mentions sensitive keywords {:?} in an assignment-like form, possible leak, manual review recommended",
                        kw
                    );
                }
            }
        }

        OutputGuardrailResult::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Arc;

    use crate::guardrail::GuardrailError;
    use crate::judge::SensitiveJudge;

    /// 固定返回泄露/不泄露的 mock 裁判(P2-3)。
    struct MockJudge {
        leak: bool,
    }
    #[async_trait]
    impl SensitiveJudge for MockJudge {
        fn name(&self) -> &str {
            "mock"
        }
        async fn judge(&self, _text: &str) -> Result<bool, GuardrailError> {
            Ok(self.leak)
        }
    }

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
    async fn test_sensitive_info_benign_mention_passes() {
        // P2-1/P2-2: password/密码/token/secret 是高误报"提及"词,普通提及不 Block。
        let g = SensitiveInfoGuardrail::new();
        assert!(g.validate("如何安全保存密码").await.is_pass());
        assert!(g.validate("your password is 123").await.is_pass());
        assert!(g.validate("请妥善保管你的token").await.is_pass());
        assert!(g.validate("hello world").await.is_pass());
    }

    #[tokio::test]
    async fn test_sensitive_info_default_block_keywords() {
        // api_key/credential 仍是低误报默认 Block 词(任何出现即 Block)。
        let g = SensitiveInfoGuardrail::new();
        assert!(g.validate("配置里泄露了 credential").await.is_block());
        assert!(g.validate("api_key=abc123").await.is_block());
    }

    #[test]
    fn test_detect_mention_context() {
        // 赋值/声明结构 → 命中
        assert_eq!(
            detect_mention_context("password: hunter2"),
            Some("password")
        );
        assert_eq!(detect_mention_context("password=abc123"), Some("password"));
        assert_eq!(
            detect_mention_context(r#""password": "abc""#),
            Some("password")
        );
        assert_eq!(
            detect_mention_context("your password is 123"),
            Some("password")
        );
        assert_eq!(detect_mention_context("密码是abc123"), Some("密码"));
        assert_eq!(detect_mention_context("密码为abc"), Some("密码"));
        // 普通提及 → 不命中
        assert_eq!(detect_mention_context("如何安全保存密码"), None);
        assert_eq!(detect_mention_context("记住密码的注意事项"), None);
        assert_eq!(detect_mention_context("请妥善保管你的token"), None);
        assert_eq!(detect_mention_context("hello world"), None);
    }

    #[tokio::test]
    async fn test_sensitive_info_judge_blocks_real_leak() {
        // P2-3: 挂载裁判,上下文敏感命中 + 裁判判泄露 → Block。
        let g = SensitiveInfoGuardrail::new().with_judge(Arc::new(MockJudge { leak: true }));
        assert!(g.validate("密码是abc123").await.is_block());
    }

    #[tokio::test]
    async fn test_sensitive_info_judge_passes_normal_mention() {
        // P2-3: 裁判判正常提及 → 放行。
        let g = SensitiveInfoGuardrail::new().with_judge(Arc::new(MockJudge { leak: false }));
        assert!(g.validate("密码是abc123").await.is_pass());
    }

    #[tokio::test]
    async fn test_sensitive_info_judge_not_called_on_plain_mention() {
        // 普通提及(无赋值结构)连裁判都不触发。
        let g = SensitiveInfoGuardrail::new().with_judge(Arc::new(MockJudge { leak: true }));
        assert!(g.validate("如何安全保存密码").await.is_pass());
    }

    #[tokio::test]
    async fn test_sensitive_info_concrete_patterns_still_block_with_judge() {
        // 低误报具体模式不经过裁判,直接 Block。
        let g = SensitiveInfoGuardrail::new().with_judge(Arc::new(MockJudge { leak: false }));
        assert!(g
            .validate("key: sk-abcdefghijklmnopqrstuvwxyz123456")
            .await
            .is_block());
        assert!(g.validate("contact: user@example.com").await.is_block());
    }

    #[tokio::test]
    async fn test_sensitive_info_api_key() {
        let g = SensitiveInfoGuardrail::new();
        assert!(g
            .validate("key: sk-abcdefghijklmnopqrstuvwxyz123456")
            .await
            .is_block());
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
