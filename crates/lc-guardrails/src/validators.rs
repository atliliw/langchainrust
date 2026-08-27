//! Built-in Guardrail validators

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

/// High-false-positive "mention" keywords (P2-2): plain mentions (e.g. "how to store passwords
/// safely") do not count as leaks and by default only `log::warn`, never Block; concrete patterns
/// (API key / email / credit card) and `with_keywords` custom low-false-positive words Block directly.
const MENTION_KEYWORDS: &[&str] = &["password", "密码", "token", "secret"];

/// Builds a "context-sensitive" matcher for a keyword (P2-1): only assignment/declaration
/// structures (value adjacent) hit, while plain mentions do not. Covers:
/// - `password=abc` / `password: abc` / `"password": "abc"` / `password is abc`
/// - the keyword immediately followed by a Chinese copular particle (see the regex below)
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

/// Detects the first high-false-positive mention keyword appearing in an assignment/declaration
/// structure (P2-1).
///
/// Returns `None` = only a plain mention (e.g. "how to store passwords safely"); this is the
/// branch point between the warn and Block paths: plain mentions pass directly (optionally with
/// an LLM judge's second determination, see P2-3).
pub(crate) fn detect_mention_context(text: &str) -> Option<&'static str> {
    for (kw, re) in MENTION_CONTEXT_RES.iter() {
        if re.is_match(text) {
            return Some(kw);
        }
    }
    None
}

/// Input length limit
pub struct MaxLengthGuardrail {
    max: usize,
}

impl MaxLengthGuardrail {
    /// Creates an input guardrail with a maximum length limit.
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

/// Forbidden word check
pub struct ForbiddenWordsGuardrail {
    words: Vec<String>,
}

impl ForbiddenWordsGuardrail {
    /// Creates an input guardrail that checks for forbidden words.
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

/// Sensitive information detection (API key / email / credit card / keywords)
///
/// The detection vocabulary is tiered by false-positive risk (P2-2):
/// - low-false-positive concrete patterns: API key (`sk-…`), email, Luhn-valid credit card numbers -> block directly;
/// - low-false-positive custom keywords (`with_keywords`): any occurrence blocks (the user opted in explicitly);
/// - high-false-positive mention words (password/token/secret): after a context-sensitive hit (P2-1),
///   only `log::warn` without blocking; when a [`SensitiveJudge`] (P2-3) is configured, the LLM judge
///   makes the second "real leak vs normal mention" determination, blocking only on a real leak.
pub struct SensitiveInfoGuardrail {
    /// Low-false-positive custom keywords: any occurrence blocks (the user opted in explicitly).
    keywords: Vec<String>,
    /// Optional LLM judge (P2-3): makes the second determination after a high-false-positive mention keyword hits context-sensitively.
    judge: Option<Arc<dyn SensitiveJudge>>,
}

impl SensitiveInfoGuardrail {
    /// Creates a sensitive-information guardrail (defaults to the api_key / credential low-false-positive keywords).
    pub fn new() -> Self {
        Self {
            // password/token/secret moved to the high-false-positive mention words (warn only), no longer blocking by default.
            keywords: vec!["api_key".to_string(), "credential".to_string()],
            judge: None,
        }
    }

    /// Appends custom low-false-positive keywords (any occurrence blocks).
    pub fn with_keywords(mut self, k: Vec<String>) -> Self {
        self.keywords.extend(k);
        self
    }

    /// Attaches an LLM judge (P2-3): after a high-false-positive mention keyword hits context-sensitively, a second determination blocks only on a real leak.
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
        // 1) low-false-positive concrete patterns -> block directly, no judge needed.
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

        // 2) low-false-positive custom keywords -> block directly (the user opted in explicitly).
        let lower = output.to_lowercase();
        for kw in &self.keywords {
            if lower.contains(&kw.to_lowercase()) {
                return OutputGuardrailResult::Block {
                    reason: format!("output contains sensitive keyword: {}", kw),
                };
            }
        }

        // 3) high-false-positive mention words: only handled on a context-sensitive hit (P2-1).
        //    with an LLM judge -> second "real leak vs normal mention" determination, blocking only on a leak (P2-3);
        //    without a judge -> warn only, no block (P2-2); plain mentions pass directly.
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
                        // a judge failure must not cause a false block: fall back to the judge-less warn-only behavior (prefer passing and leaving a log).
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

    /// Mock judge that always returns leak/no-leak (P2-3).
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
        // P2-1/P2-2: password/token/secret are high-false-positive "mention" words; plain mentions are not blocked.
        let g = SensitiveInfoGuardrail::new();
        assert!(g.validate("如何安全保存密码").await.is_pass());
        assert!(g.validate("your password is 123").await.is_pass());
        assert!(g.validate("请妥善保管你的token").await.is_pass());
        assert!(g.validate("hello world").await.is_pass());
    }

    #[tokio::test]
    async fn test_sensitive_info_default_block_keywords() {
        // api_key/credential remain low-false-positive default block words (any occurrence blocks).
        let g = SensitiveInfoGuardrail::new();
        assert!(g.validate("配置里泄露了 credential").await.is_block());
        assert!(g.validate("api_key=abc123").await.is_block());
    }

    #[test]
    fn test_detect_mention_context() {
        // assignment/declaration structure -> hit
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
        // plain mention -> no hit
        assert_eq!(detect_mention_context("如何安全保存密码"), None);
        assert_eq!(detect_mention_context("记住密码的注意事项"), None);
        assert_eq!(detect_mention_context("请妥善保管你的token"), None);
        assert_eq!(detect_mention_context("hello world"), None);
    }

    #[tokio::test]
    async fn test_sensitive_info_judge_blocks_real_leak() {
        // P2-3: with a judge attached, context-sensitive hit + judge says leak -> Block.
        let g = SensitiveInfoGuardrail::new().with_judge(Arc::new(MockJudge { leak: true }));
        assert!(g.validate("密码是abc123").await.is_block());
    }

    #[tokio::test]
    async fn test_sensitive_info_judge_passes_normal_mention() {
        // P2-3: judge says normal mention -> pass.
        let g = SensitiveInfoGuardrail::new().with_judge(Arc::new(MockJudge { leak: false }));
        assert!(g.validate("密码是abc123").await.is_pass());
    }

    #[tokio::test]
    async fn test_sensitive_info_judge_not_called_on_plain_mention() {
        // a plain mention (no assignment structure) does not even trigger the judge.
        let g = SensitiveInfoGuardrail::new().with_judge(Arc::new(MockJudge { leak: true }));
        assert!(g.validate("如何安全保存密码").await.is_pass());
    }

    #[tokio::test]
    async fn test_sensitive_info_concrete_patterns_still_block_with_judge() {
        // low-false-positive concrete patterns bypass the judge and block directly.
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
