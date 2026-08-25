//! Evaluation module example
//!
//! Shows the evaluator + batch evaluation kit: Dataset → Predictor → EvalRunner → Report.
//!
//! - The dataset can come from `Dataset::new` or a JSONL file (`Dataset::from_jsonl(...).await`,
//!   one `{"input": "...", "reference": "..."}` per line, read asynchronously)
//! - A single predict / scoring failure does not abort the whole batch; it is recorded in
//!   `Report::failures` (P1-3)
//! - The report carries the original inputs plus mean / standard deviation, and can be
//!   deserialized for offline analysis (P1-4)
//!
//! # Run
//! ```bash
//! cargo run -p langchainrust --example evaluation [data.jsonl]
//! ```

use async_trait::async_trait;
use langchainrust::evaluation::*;

/// A static-answer predictor: used for the demo. In production, implement `Predictor`
/// by hooking up an LLMChain / Agent.
struct StaticPredictor(&'static str);

#[async_trait]
impl Predictor for StaticPredictor {
    async fn predict(&self, _input: &str) -> Result<String, EvalError> {
        Ok(self.0.to_string())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Batch evaluation: Dataset → Predictor → EvalRunner → Report
    let dataset = match std::env::args().nth(1) {
        Some(path) => Dataset::from_jsonl(&path).await?,
        None => Dataset::new(vec![
            Example::new("2+2?", "4"),
            Example::new("What is the capital of France?", "Paris"),
        ]),
    };
    if dataset.is_empty() {
        println!("The dataset is empty; there are no examples to evaluate.");
        return Ok(());
    }

    let runner = EvalRunner::new(vec![Box::new(ExactMatch), Box::new(StringDistance)]);
    let report = runner.run(&dataset, &StaticPredictor("4")).await?;

    println!("=== Per-example results (including original inputs) ===");
    for ex in &report.per_example {
        println!(
            "[{}] input={:?} prediction={:?} scores={:?}",
            ex.index, ex.input, ex.prediction, ex.scores
        );
    }
    println!("=== Summary (mean ± std) ===");
    for (name, s) in &report.summary {
        println!(
            "{name}: mean={:.3} std={:.3} count={}",
            s.mean, s.std, s.count
        );
    }
    if !report.failures.is_empty() {
        println!("=== Failures (per-item tolerance) ===");
        for f in &report.failures {
            println!("[{}] {}: {}", f.index, f.stage, f.error);
        }
    }

    // 2. Score with a single evaluator directly
    let evaluator = ExactMatch;
    let result = evaluator.eval("What language?", "Rust", "Rust").await?;
    println!("ExactMatch (same): score = {}", result.value);
    let result = evaluator.eval("What language?", "Rust", "Python").await?;
    println!("ExactMatch (different): score = {}", result.value);

    Ok(())
}
