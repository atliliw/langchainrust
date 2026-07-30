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

/// 评测分数(0.0–1.0,1.0 为最佳)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl Score {
    pub fn new(value: f64) -> Self {
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

    /// 从 JSONL 文件加载(每行一个 ``{input, reference}``)
    pub fn from_jsonl(path: &str) -> Result<Self, EvalError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| EvalError::IoError(e.to_string()))?;
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

/// 预测器 trait(待评测的对象:LLMChain / Agent 等)
#[async_trait]
pub trait Predictor: Send + Sync {
    async fn predict(&self, input: &str) -> Result<String, EvalError>;
}
