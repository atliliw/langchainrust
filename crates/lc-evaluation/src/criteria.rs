//! 评测核心类型与 trait:EvalError、Score、Example、Dataset,
//! 以及 Evaluator / Predictor trait。

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// 评测错误
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("IO 错误: {0}")]
    IoError(String),
    #[error("解析错误: {0}")]
    ParseError(String),
    #[error("嵌入错误: {0}")]
    EmbeddingError(String),
    #[error("预测错误: {0}")]
    PredictorError(String),
}

/// P2-6: 共享裁判内核(lc-core::judge)的错误映射进评测错误域,
/// 让 `structured_call(...).await?` 在 `Result<_, EvalError>` 上下文里直接可用。
impl From<lc_core::judge::StructuredJudgeError> for EvalError {
    fn from(e: lc_core::judge::StructuredJudgeError) -> Self {
        match e {
            lc_core::judge::StructuredJudgeError::Call(s) => EvalError::PredictorError(s),
            lc_core::judge::StructuredJudgeError::Parse(s) => EvalError::ParseError(s),
        }
    }
}

/// 评测分数(0.0–1.0,1.0 为最佳)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl Score {
    /// 构造 0.0–1.0 之间的分数。
    ///
    /// P2-8: Rust 的 `f64::clamp(0.0, 1.0)` 对 NaN 返回 NaN,会污染
    /// summary 均值/标准差。这里先做 NaN 前置检查,按 0.0 处理(负无穷、
    /// 正无穷交给 `clamp` 收敛到边界)。
    pub fn new(value: f64) -> Self {
        let value = if value.is_nan() {
            log::warn!("Score::new 收到 NaN,按 0.0 处理");
            0.0
        } else {
            value
        };
        Self {
            value: value.clamp(0.0, 1.0),
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// 评测样例
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub input: String,
    pub reference: String,
}

impl Example {
    pub fn new(input: impl Into<String>, reference: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            reference: reference.into(),
        }
    }
}

/// 数据集
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub examples: Vec<Example>,
}

impl Dataset {
    pub fn new(examples: Vec<Example>) -> Self {
        Self { examples }
    }

    /// 从 JSONL 文件加载(每行一个 ``{input, reference}``)。
    ///
    /// P2-2: 异步 I/O(`tokio::fs`),避免同步阻塞落在 async 评测链路里。
    pub async fn from_jsonl(path: &str) -> Result<Self, EvalError> {
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| EvalError::IoError(e.to_string()))?;
        let mut examples = Vec::new();
        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let ex: Example = serde_json::from_str(line)
                .map_err(|e| EvalError::ParseError(format!("第 {} 行: {}", i + 1, e)))?;
            examples.push(ex);
        }
        Ok(Self { examples })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

/// 评测器 trait
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// 对单条预测打分
    async fn eval(
        &self,
        input: &str,
        prediction: &str,
        reference: &str,
    ) -> Result<Score, EvalError>;

    /// 评测器名称(用于报告汇总)
    fn name(&self) -> &str;
}

/// 成对比较评测器 trait(竞技场模式):对同一输入的 A/B 两个回答判优劣。
///
/// P1-1: 与单点 `Evaluator` 并列的一等公民,`EvalRunner` 同时收纳两种,
/// 竞技场评测因此也能进统一报告。得分约定:1.0 = A 优、0.5 = 平局、0.0 = B 优。
#[async_trait]
pub trait PairwiseEvaluator: Send + Sync {
    /// 比较 A、B 两个回答,返回 0-1 得分
    /// (1.0 = A 优,0.5 = 平局,0.0 = B 优)。
    async fn eval_pair(&self, input: &str, a: &str, b: &str) -> Result<Score, EvalError>;

    /// 评测器名称(用于报告汇总)
    fn name(&self) -> &str;
}

/// 预测器 trait(待评测的对象:LLMChain / Agent 等)
#[async_trait]
pub trait Predictor: Send + Sync {
    async fn predict(&self, input: &str) -> Result<String, EvalError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_new_normal() {
        assert!((Score::new(0.5).value - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_score_new_clamps_overflow() {
        assert_eq!(Score::new(2.0).value, 1.0);
        assert_eq!(Score::new(-1.0).value, 0.0);
        assert_eq!(Score::new(f64::INFINITY).value, 1.0);
        assert_eq!(Score::new(f64::NEG_INFINITY).value, 0.0);
    }

    /// P2-8: NaN 不再穿透 `.clamp(0.0, 1.0)` 污染汇总统计。
    #[test]
    fn test_score_new_nan_guarded() {
        assert_eq!(Score::new(f64::NAN).value, 0.0);
        // 保证 NaN 被清掉,而不是残留在统计里
        assert!(Score::new(f64::NAN).value.is_finite());
    }

    /// P2-2: from_jsonl 异步读文件,单行解析失败带行号。
    #[tokio::test]
    async fn test_from_jsonl_async() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl");
        std::fs::write(
            &path,
            "{\"input\":\"q1\",\"reference\":\"a1\"}\n\n{\"input\":\"q2\",\"reference\":\"a2\"}\n",
        )
        .unwrap();
        let dataset = Dataset::from_jsonl(path.to_str().unwrap()).await.unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.examples[1].input, "q2");
        assert_eq!(dataset.examples[1].reference, "a2");
    }

    #[tokio::test]
    async fn test_from_jsonl_missing_file() {
        let err = Dataset::from_jsonl("不存在-的文件.jsonl")
            .await
            .unwrap_err();
        assert!(matches!(err, EvalError::IoError(_)));
    }

    #[tokio::test]
    async fn test_from_jsonl_bad_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        std::fs::write(&path, "{\"input\":\"q\"}\n").unwrap();
        let err = Dataset::from_jsonl(path.to_str().unwrap())
            .await
            .unwrap_err();
        assert!(matches!(err, EvalError::ParseError(_)));
    }
}
