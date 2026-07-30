//! 批量运行器:`Report` 与 `EvalRunner`。
//!
//! `EvalRunner` 在数据集上逐条调用 `Predictor`,再交给多个 `Evaluator` 打分,
//! 最终汇总为 `Report`(逐条得分 + 各评测器均值)。

use std::collections::HashMap;

use super::criteria::{Dataset, EvalError, Evaluator, Predictor, Score};

/// 评测报告
#[derive(Debug, Clone, serde::Serialize)]
pub struct Report {
    /// 每条样例的各评测器得分
    pub per_example: Vec<HashMap<String, Score>>,
    /// 各评测器的平均分
    pub summary: HashMap<String, f64>,
}

/// 批量运行器
pub struct EvalRunner {
    evaluators: Vec<Box<dyn Evaluator>>,
}

impl EvalRunner {
    pub fn new(evaluators: Vec<Box<dyn Evaluator>>) -> Self {
        Self { evaluators }
    }

    /// 在数据集上运行所有评测器,返回报告
    pub async fn run(
        &self,
        dataset: &Dataset,
        predictor: &dyn Predictor,
    ) -> Result<Report, EvalError> {
        let mut per_example = Vec::with_capacity(dataset.len());
        let mut sums: HashMap<String, (f64, usize)> = HashMap::new();

        for ex in &dataset.examples {
            let prediction = predictor.predict(&ex.input).await?;
            let mut row = HashMap::new();
            for ev in &self.evaluators {
                let score = ev.eval(&ex.input, &prediction, &ex.reference).await?;
                let entry = sums.entry(ev.name().to_string()).or_insert((0.0, 0));
                entry.0 += score.value;
                entry.1 += 1;
                row.insert(ev.name().to_string(), score);
            }
            per_example.push(row);
        }

        let mut summary = HashMap::new();
        for (name, (total, count)) in sums {
            if count > 0 {
                summary.insert(name, total / count as f64);
            }
        }
        Ok(Report {
            per_example,
            summary,
        })
    }
}
