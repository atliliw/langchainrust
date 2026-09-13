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
//! - Every report carries a `run_id` (B9): pin it with `EvalRunner::with_run_id` or let the
//!   runner generate a UUID v4; the same id is handed to `Predictor::begin_run` so the system
//!   under test can stamp it into its traces. `Report::to_table` / `to_jsonl` export the batch
//!   for offline analysis. RAGAS-style RAG metrics (context precision/recall, answer
//!   relevancy) live in the `ragas_eval` example.
//!
//! # Run
//! ```bash
//! cargo run -p langchainrust --example evaluation [data.jsonl]
//! # optional: EVAL_RUN_ID=fixed-id EVAL_EXPORT=out.jsonl
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
            Example::with_contexts(
                "What is the capital of France?",
                "Paris",
                vec!["Paris is the capital of France.".into()],
            ),
        ]),
    };
    if dataset.is_empty() {
        println!("The dataset is empty; there are no examples to evaluate.");
        return Ok(());
    }

    // B9: pin a run id so the exported rows join back to this run's traces; without
    // `with_run_id` the runner mints a UUID v4 itself.
    let run_id = std::env::var("EVAL_RUN_ID").unwrap_or_else(|_| "demo-run".to_string());
    let runner =
        EvalRunner::new(vec![Box::new(ExactMatch), Box::new(StringDistance)]).with_run_id(run_id);
    let report = runner.run(&dataset, &StaticPredictor("4")).await?;
    println!("eval run_id: {}", report.run_id);

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

    // B9: offline exports — a fixed-width table for humans and JSONL for tooling
    // (every JSONL row carries the run id, so rows join back to the run's traces).
    println!("=== Table ===");
    println!("{}", report.to_table());
    if let Ok(path) = std::env::var("EVAL_EXPORT") {
        report.write_jsonl(&path).await?;
        println!("wrote JSONL export to {path}");
    } else {
        println!("=== JSONL (set EVAL_EXPORT=<path> to write a file) ===");
        print!("{}", report.to_jsonl());
    }

    // 2. Score with a single evaluator directly
    let evaluator = ExactMatch;
    let result = evaluator.eval("What language?", "Rust", "Rust").await?;
    println!("ExactMatch (same): score = {}", result.value);
    let result = evaluator.eval("What language?", "Rust", "Python").await?;
    println!("ExactMatch (different): score = {}", result.value);

    Ok(())
}
