//! `lc eval` — run an evaluation over a JSONL dataset and write a JSON report.
//!
//! Each line: `{"input", "expected"[,"actual"]}`. When `actual` is absent the
//! evaluator asks the model for a prediction first.

use std::path::Path;

use serde::Deserialize;
use serde_json::json;

use langchainrust::core::Runnable;
use langchainrust::evaluation::{Evaluator, ExactMatch, StringDistance};
use langchainrust::prelude::*;

#[derive(Deserialize)]
struct EvalLine {
    input: String,
    expected: String,
    #[serde(default)]
    actual: Option<String>,
}

pub async fn run(
    dataset: &Path,
    output: &Path,
    scorer: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(dataset)?;
    let llm = OpenAIChat::new(OpenAIConfig::from_env_result()?);

    let mut rows = Vec::new();
    let mut passed = 0usize;

    for (idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let rec: EvalLine = serde_json::from_str(line)
            .map_err(|e| format!("line {}: {e}", idx + 1))?;
        let prediction = match rec.actual {
            Some(a) => a,
            None => llm.invoke(vec![Message::human(rec.input.clone())], None).await?.content,
        };

        let score = match scorer {
            "distance" => StringDistance
                .eval(&rec.input, &prediction, &rec.expected)
                .await?,
            _ => ExactMatch.eval(&rec.input, &prediction, &rec.expected).await?,
        };
        if score.value >= 0.999 {
            passed += 1;
        }
        rows.push(json!({
            "input": rec.input,
            "expected": rec.expected,
            "actual": prediction,
            "score": score.value,
        }));
    }

    let report = json!({
        "total": rows.len(),
        "passed": passed,
        "scorer": scorer,
        "rows": rows,
    });
    std::fs::write(output, serde_json::to_string_pretty(&report)?)?;

    println!(
        "evaluated {} rows, {} passed ({:.1}%) -> {}",
        report["total"],
        passed,
        100.0 * passed as f64 / rows.len().max(1) as f64,
        output.display()
    );
    Ok(())
}