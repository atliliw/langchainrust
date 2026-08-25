//! 规则类评测器:基于确定性规则打分(无 LLM、无向量,0 成本)。
//!
//! 适合答案空间明确、可机器判定的任务(格式校验、关键词、长度约束等)。

use async_trait::async_trait;
use regex::Regex;

use super::{EvalError, Evaluator, Score};

/// 关键词包含评测器:检查预测是否包含指定关键词。
///
/// `all_required = true`(默认)需全部包含才得分;`false` 包含任一即得分。
pub struct ContainsKeyword {
    keywords: Vec<String>,
    case_sensitive: bool,
    all_required: bool,
}

impl ContainsKeyword {
    /// 创建关键词包含评测器。
    pub fn new(keywords: Vec<String>) -> Self {
        Self {
            keywords,
            case_sensitive: false,
            all_required: true,
        }
    }

    /// 大小写敏感(默认不敏感)
    pub fn case_sensitive(mut self, v: bool) -> Self {
        self.case_sensitive = v;
        self
    }

    /// true=全部包含才得分(默认);false=包含任一即得分
    pub fn all_required(mut self, v: bool) -> Self {
        self.all_required = v;
        self
    }
}

#[async_trait]
impl Evaluator for ContainsKeyword {
    async fn eval(
        &self,
        _input: &str,
        prediction: &str,
        _reference: &str,
    ) -> Result<Score, EvalError> {
        let pred = if self.case_sensitive {
            prediction.to_string()
        } else {
            prediction.to_lowercase()
        };
        let matches: Vec<bool> = self
            .keywords
            .iter()
            .map(|k| {
                let k = if self.case_sensitive {
                    k.clone()
                } else {
                    k.to_lowercase()
                };
                pred.contains(&k)
            })
            .collect();
        let ok = if self.all_required {
            matches.iter().all(|&m| m)
        } else {
            matches.iter().any(|&m| m)
        };
        Ok(Score::new(if ok { 1.0 } else { 0.0 }).with_label(if ok {
            "contains"
        } else {
            "missing"
        }))
    }

    fn name(&self) -> &str {
        "contains_keyword"
    }
}

/// 正则匹配评测器:检查预测是否匹配正则。
pub struct RegexMatch {
    pattern: Regex,
}

impl RegexMatch {
    /// 创建正则匹配评测器(正则非法返回 `ParseError`)。
    pub fn new(pattern: &str) -> Result<Self, EvalError> {
        Ok(Self {
            pattern: Regex::new(pattern).map_err(|e| EvalError::ParseError(e.to_string()))?,
        })
    }
}

#[async_trait]
impl Evaluator for RegexMatch {
    async fn eval(
        &self,
        _input: &str,
        prediction: &str,
        _reference: &str,
    ) -> Result<Score, EvalError> {
        let ok = self.pattern.is_match(prediction);
        Ok(
            Score::new(if ok { 1.0 } else { 0.0 }).with_label(if ok {
                "match"
            } else {
                "no_match"
            }),
        )
    }

    fn name(&self) -> &str {
        "regex_match"
    }
}

/// 长度检查评测器:预测长度(按字符)是否落在 [min, max] 范围。
pub struct LengthCheck {
    min: Option<usize>,
    max: Option<usize>,
}

impl Default for LengthCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl LengthCheck {
    /// 创建长度检查评测器(默认不限长度)。
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
        }
    }

    /// 最小长度(含)
    pub fn min(mut self, m: usize) -> Self {
        self.min = Some(m);
        self
    }

    /// 最大长度(含)
    pub fn max(mut self, m: usize) -> Self {
        self.max = Some(m);
        self
    }
}

#[async_trait]
impl Evaluator for LengthCheck {
    async fn eval(
        &self,
        _input: &str,
        prediction: &str,
        _reference: &str,
    ) -> Result<Score, EvalError> {
        let len = prediction.chars().count();
        let ok = self.min.map_or(true, |m| len >= m) && self.max.map_or(true, |m| len <= m);
        Ok(Score::new(if ok { 1.0 } else { 0.0 }).with_label(if ok {
            "in_range"
        } else {
            "out_of_range"
        }))
    }

    fn name(&self) -> &str {
        "length_check"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_contains_all_required() {
        let ev = ContainsKeyword::new(vec!["巴黎".into(), "法国".into()]);
        assert_eq!(ev.eval("", "巴黎是法国首都", "").await.unwrap().value, 1.0);
        assert_eq!(ev.eval("", "巴黎很大", "").await.unwrap().value, 0.0); // 缺"法国"
    }

    #[tokio::test]
    async fn test_contains_any() {
        let ev = ContainsKeyword::new(vec!["巴黎".into(), "伦敦".into()]).all_required(false);
        assert_eq!(ev.eval("", "伦敦很大", "").await.unwrap().value, 1.0);
        assert_eq!(ev.eval("", "柏林很大", "").await.unwrap().value, 0.0);
    }

    #[tokio::test]
    async fn test_contains_case_insensitive() {
        let ev = ContainsKeyword::new(vec!["hello".into()]);
        assert_eq!(ev.eval("", "Say HELLO world", "").await.unwrap().value, 1.0);
    }

    #[tokio::test]
    async fn test_regex_match() {
        let ev = RegexMatch::new(r"\d{4}-\d{2}-\d{2}").unwrap();
        assert_eq!(
            ev.eval("", "日期是 2024-01-15", "").await.unwrap().value,
            1.0
        );
        assert_eq!(
            ev.eval("", "日期是 2024/01/15", "").await.unwrap().value,
            0.0
        );
    }

    #[tokio::test]
    async fn test_length_check() {
        let ev = LengthCheck::new().min(2).max(5);
        assert_eq!(ev.eval("", "你好", "").await.unwrap().value, 1.0); // 2 字符,在 [2,5]
        assert_eq!(ev.eval("", "你好世界测试", "").await.unwrap().value, 0.0); // 6 字符,超 5
    }

    #[tokio::test]
    async fn test_length_default_passes() {
        // 不设 min/max,任何长度都通过
        let ev = LengthCheck::new();
        assert_eq!(ev.eval("", "任意长度", "").await.unwrap().value, 1.0);
        assert_eq!(ev.eval("", "", "").await.unwrap().value, 1.0);
    }
}
