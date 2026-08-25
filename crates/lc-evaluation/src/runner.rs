//! 批量运行器:`Report` 与 `EvalRunner`。
//!
//! `EvalRunner` 在数据集上逐条调用 `Predictor`,再交给多个单点 `Evaluator`
//! 与成对 `PairwiseEvaluator` 打分,最终汇总为 `Report`。
//!
//! P1-3: 逐条容错——单条 predict 或某评测器打分失败记入 `Report::failures`,
//! 已算好的结果不丢弃,整体不中止。P1-4: `Report` 携带原始文本 + 标准差,
//! 并实现 `Serialize`/`Deserialize`,便于落盘后二次分析。

use std::collections::{HashMap, HashSet};

use super::criteria::{Dataset, EvalError, Evaluator, PairwiseEvaluator, Predictor, Score};

/// 单条样例的完整评测记录(含原始文本,便于出低分时追溯)。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExampleReport {
    /// 样例在数据集中的下标(0 起)
    pub index: usize,
    pub input: String,
    pub reference: String,
    pub prediction: String,
    /// 各评测器对该条的得分(失败或未运行的评测器不在其中)
    pub scores: HashMap<String, Score>,
}

/// 单个评测器的汇总统计(均值 + 总体标准差 + 样本数)。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoreSummary {
    pub mean: f64,
    pub std: f64,
    pub count: usize,
}

/// 单条失败记录:某下标样例的 predict 或某评测器打分失败。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FailureRecord {
    /// 样例在数据集中的下标(0 起)
    pub index: usize,
    /// 失败阶段:`"predict"` 或评测器 `name()`
    pub stage: String,
    pub error: String,
}

/// 评测报告(含原文、标准差、失败清单;可反序列化)。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Report {
    /// 逐条完整记录(含 input/reference/prediction 原文)
    pub per_example: Vec<ExampleReport>,
    /// 各评测器汇总(均值 + 标准差 + 样本数)
    pub summary: HashMap<String, ScoreSummary>,
    /// 逐条容错收集的失败记录(为空表示全部成功)
    pub failures: Vec<FailureRecord>,
}

/// 批量运行器:同时收纳单点与成对评测器。
pub struct EvalRunner {
    evaluators: Vec<Box<dyn Evaluator>>,
    pairwise: Vec<Box<dyn PairwiseEvaluator>>,
}

impl EvalRunner {
    /// 创建批量运行器(仅含单点评测器)。
    pub fn new(evaluators: Vec<Box<dyn Evaluator>>) -> Self {
        Self {
            evaluators,
            pairwise: Vec::new(),
        }
    }

    /// 追加成对评测器(P1-1,竞技场评测进统一报告)。
    pub fn with_pairwise(mut self, pairwise: Vec<Box<dyn PairwiseEvaluator>>) -> Self {
        self.pairwise.extend(pairwise);
        self
    }

    /// 在数据集上运行所有评测器,返回报告。
    ///
    /// P1-3: 逐条容错——单条 predict 失败记 `"predict"` 失败记录并跳过该条;
    /// 某评测器打分失败只记该评测器的失败记录,其它评测器照常出分。
    /// P1-1: 成对评测器同样参与,以 `(prediction, reference)` 作为 A/B 两个候选
    /// (竞技场用法:把待比答案放进 reference 槽)。
    pub async fn run(
        &self,
        dataset: &Dataset,
        predictor: &dyn Predictor,
    ) -> Result<Report, EvalError> {
        Self::warn_duplicate_names(&self.evaluators, &self.pairwise);

        let mut per_example = Vec::with_capacity(dataset.len());
        let mut failures = Vec::new();
        // 每个评测器累计所有成功的样本分,用于算均值/标准差
        let mut per_name: HashMap<String, Vec<f64>> = HashMap::new();

        for (i, ex) in dataset.examples.iter().enumerate() {
            let prediction = match predictor.predict(&ex.input).await {
                Ok(p) => p,
                Err(e) => {
                    failures.push(FailureRecord {
                        index: i,
                        stage: "predict".into(),
                        error: e.to_string(),
                    });
                    continue;
                }
            };

            let mut scores = HashMap::new();
            for ev in &self.evaluators {
                match ev.eval(&ex.input, &prediction, &ex.reference).await {
                    Ok(s) => {
                        per_name
                            .entry(ev.name().to_string())
                            .or_default()
                            .push(s.value);
                        scores.insert(ev.name().to_string(), s);
                    }
                    Err(e) => failures.push(FailureRecord {
                        index: i,
                        stage: ev.name().to_string(),
                        error: e.to_string(),
                    }),
                }
            }
            for ev in &self.pairwise {
                match ev.eval_pair(&ex.input, &prediction, &ex.reference).await {
                    Ok(s) => {
                        per_name
                            .entry(ev.name().to_string())
                            .or_default()
                            .push(s.value);
                        scores.insert(ev.name().to_string(), s);
                    }
                    Err(e) => failures.push(FailureRecord {
                        index: i,
                        stage: ev.name().to_string(),
                        error: e.to_string(),
                    }),
                }
            }

            per_example.push(ExampleReport {
                index: i,
                input: ex.input.clone(),
                reference: ex.reference.clone(),
                prediction,
                scores,
            });
        }

        let mut summary = HashMap::new();
        for (name, values) in per_name {
            let count = values.len();
            let mean = values.iter().sum::<f64>() / count as f64;
            // 总体标准差:分布/方差信息比均值更能反映评测器的稳定性
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
            summary.insert(
                name,
                ScoreSummary {
                    mean,
                    std: variance.sqrt(),
                    count,
                },
            );
        }

        Ok(Report {
            per_example,
            summary,
            failures,
        })
    }

    /// P1-4: 重名评测器会在 summary/报告里静默互相覆盖,至少 `log::warn` 提示。
    fn warn_duplicate_names(
        evaluators: &[Box<dyn Evaluator>],
        pairwise: &[Box<dyn PairwiseEvaluator>],
    ) {
        let mut seen = HashSet::new();
        for ev in evaluators {
            if !seen.insert(ev.name()) {
                log::warn!(
                    "EvalRunner: duplicate evaluator name '{}', report data will be overwritten",
                    ev.name()
                );
            }
        }
        for ev in pairwise {
            if !seen.insert(ev.name()) {
                log::warn!(
                    "EvalRunner: duplicate evaluator name '{}', report data will be overwritten",
                    ev.name()
                );
            }
        }
    }
}
