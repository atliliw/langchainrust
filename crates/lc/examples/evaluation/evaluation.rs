//! Evaluation 模块示例
//!
//! 展示评测器 + 批量评测四件套:Dataset → Predictor → EvalRunner → Report。
//!
//! - 数据集可来自 `Dataset::new` 或 JSONL 文件(`Dataset::from_jsonl(...).await`,
//!   每行一个 `{"input": "...", "reference": "..."}`,异步读文件)
//! - 单条 predict / 打分失败不中断整批,记入 `Report::failures`(P1-3)
//! - 报告携带原文 + 均值/标准差,并可反序列化(落盘二次分析)(P1-4)
//!
//! # 运行
//! ```bash
//! cargo run -p langchainrust --example evaluation [data.jsonl]
//! ```

use async_trait::async_trait;
use langchainrust::evaluation::*;

/// 静态应答预测器:示例用。生产环境实现 `Predictor` 时接入 LLMChain / Agent。
struct StaticPredictor(&'static str);

#[async_trait]
impl Predictor for StaticPredictor {
    async fn predict(&self, _input: &str) -> Result<String, EvalError> {
        Ok(self.0.to_string())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 批量评测:Dataset → Predictor → EvalRunner → Report
    let dataset = match std::env::args().nth(1) {
        Some(path) => Dataset::from_jsonl(&path).await?,
        None => Dataset::new(vec![
            Example::new("2+2?", "4"),
            Example::new("法国首都是哪?", "巴黎"),
        ]),
    };
    if dataset.is_empty() {
        println!("数据集为空,没有可评测的样例。");
        return Ok(());
    }

    let runner = EvalRunner::new(vec![Box::new(ExactMatch), Box::new(StringDistance)]);
    let report = runner.run(&dataset, &StaticPredictor("4")).await?;

    println!("=== 逐条结果(含原文) ===");
    for ex in &report.per_example {
        println!(
            "[{}] input={:?} prediction={:?} scores={:?}",
            ex.index, ex.input, ex.prediction, ex.scores
        );
    }
    println!("=== 汇总(均值 ± 标准差) ===");
    for (name, s) in &report.summary {
        println!(
            "{name}: mean={:.3} std={:.3} count={}",
            s.mean, s.std, s.count
        );
    }
    if !report.failures.is_empty() {
        println!("=== 失败记录(逐条容错) ===");
        for f in &report.failures {
            println!("[{}] {}: {}", f.index, f.stage, f.error);
        }
    }

    // 2. 单个评测器直接打分
    let evaluator = ExactMatch;
    let result = evaluator.eval("What language?", "Rust", "Rust").await?;
    println!("ExactMatch (相同): score = {}", result.value);
    let result = evaluator.eval("What language?", "Rust", "Python").await?;
    println!("ExactMatch (不同): score = {}", result.value);

    Ok(())
}
