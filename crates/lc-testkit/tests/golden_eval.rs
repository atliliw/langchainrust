//! N2 (v0.24.0): the full record → golden → score → regression-gate loop, fully offline.
//!
//! Hand-written `RecordedExchange`s stand in for real recorded traces. Nothing here
//! touches the network: scoring is driven by `ReplayPredictor` (FIFO replay) and the
//! "candidate model" is a constant wrong-answer predictor.

use std::io::Write;

use async_trait::async_trait;
use lc_core::language_models::LLMResult;
use lc_evaluation::{compare_reports, Dataset, EvalError, EvalRunner, ExactMatch, Predictor};
use lc_schema::Message;
use lc_testkit::{
    golden_dataset, is_scoring_candidate, replay_golden_from_file, write_golden_jsonl, GoldenError,
    RecordedExchange, ReplayStrategy,
};

fn exchange(messages: Vec<Message>, content: &str) -> RecordedExchange {
    RecordedExchange {
        messages,
        response: LLMResult {
            content: content.to_string(),
            model: "fake".to_string(),
            ..Default::default()
        },
        tools: None,
    }
}

/// Three user turns; the second is a RAG-style trace ending in a tool observation.
fn sample_exchanges() -> Vec<RecordedExchange> {
    vec![
        exchange(vec![Message::system("sys"), Message::human("q1")], "a1"),
        exchange(
            vec![
                Message::system("sys"),
                Message::human("q2"),
                Message::ai_with_tool_calls("looking", vec![]),
                Message::tool("call_1", "DOC-1 retrieved text"),
            ],
            "a2",
        ),
        exchange(
            vec![
                Message::human("old question"),
                Message::ai("old answer"),
                Message::human("q3"),
            ],
            "a3",
        ),
    ]
}

fn write_recording(exchanges: &[RecordedExchange]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.jsonl");
    let mut f = std::fs::File::create(&path).unwrap();
    for ex in exchanges {
        writeln!(f, "{}", serde_json::to_string(ex).unwrap()).unwrap();
    }
    dir
}

#[test]
fn golden_dataset_uses_last_human_response_text_and_tool_contexts() {
    let dataset = golden_dataset(&sample_exchanges()).unwrap();
    assert_eq!(dataset.len(), 3);

    assert_eq!(dataset.examples[0].input, "q1");
    assert_eq!(dataset.examples[0].reference, "a1");
    assert!(dataset.examples[0].contexts.is_empty());

    // tool-result observations become RAG contexts in occurrence order
    assert_eq!(dataset.examples[1].input, "q2");
    assert_eq!(dataset.examples[1].reference, "a2");
    assert_eq!(
        dataset.examples[1].contexts,
        vec!["DOC-1 retrieved text".to_string()]
    );

    // the LAST human message in a multi-turn history is the eval input
    assert_eq!(dataset.examples[2].input, "q3");
    assert_eq!(dataset.examples[2].reference, "a3");
}

#[test]
fn golden_dataset_rejects_unscorable_exchanges_with_indexed_errors() {
    let only_system = exchange(vec![Message::system("s")], "a");
    let err = golden_dataset(&[only_system]).unwrap_err();
    assert!(matches!(err, GoldenError::NoUserMessage(0)));

    let empty_reference = exchange(vec![Message::human("q")], "   ");
    let err = golden_dataset(&[empty_reference]).unwrap_err();
    assert!(matches!(err, GoldenError::EmptyReference(0)));

    // error index points at the offending exchange, not always 0
    let good = exchange(vec![Message::human("q")], "a");
    let pure_tool_call = exchange(vec![Message::human("q")], "");
    let err = golden_dataset(&[good, pure_tool_call]).unwrap_err();
    assert!(matches!(err, GoldenError::EmptyReference(1)));
}

#[test]
fn scoring_candidate_predicate_matches_the_strict_builder() {
    assert!(is_scoring_candidate(&exchange(
        vec![Message::human("q")],
        "a"
    )));
    assert!(!is_scoring_candidate(&exchange(
        vec![Message::system("s")],
        "a"
    )));
    // pure tool-call response: has a prompt but no answer text
    assert!(!is_scoring_candidate(&exchange(
        vec![Message::human("q")],
        ""
    )));

    // the predicate is precisely what lets an agent-loop trace convert cleanly
    let mut mixed = sample_exchanges();
    mixed.push(exchange(vec![Message::human("q4")], ""));
    let filtered: Vec<RecordedExchange> = mixed.into_iter().filter(is_scoring_candidate).collect();
    assert_eq!(filtered.len(), 3);
    assert!(golden_dataset(&filtered).is_ok());
}

#[tokio::test]
async fn replay_golden_run_scores_every_recorded_reference_exactly() {
    let dir = write_recording(&sample_exchanges());
    let (dataset, predictor) =
        replay_golden_from_file(dir.path().join("trace.jsonl"), ReplayStrategy::Fifo).unwrap();

    assert_eq!(predictor.remaining(), 3);
    let runner = EvalRunner::new(vec![Box::new(ExactMatch)]);
    let report = runner.run(&dataset, &predictor).await.unwrap();

    assert_eq!(report.per_example.len(), 3);
    assert!(
        report.failures.is_empty(),
        "no predict/score failures: {:?}",
        report.failures
    );
    let summary = report
        .summary
        .get("exact_match")
        .expect("exact_match scored");
    assert_eq!(summary.count, 3);
    assert!(
        (summary.mean - 1.0).abs() < 1e-9,
        "recorded answers replay exactly"
    );
    assert!(!report.run_id.is_empty());
    assert_eq!(
        predictor.remaining(),
        0,
        "FIFO queue drains one pop per example"
    );
}

#[tokio::test]
async fn golden_jsonl_roundtrips_through_dataset_from_jsonl() {
    let dir = write_recording(&sample_exchanges());
    let dataset = lc_testkit::golden_dataset_from_file(dir.path().join("trace.jsonl")).unwrap();

    let golden_path = dir.path().join("golden.jsonl");
    write_golden_jsonl(&dataset, &golden_path).unwrap();

    let loaded = Dataset::from_jsonl(golden_path.to_str().unwrap())
        .await
        .unwrap();
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded.examples[1].input, "q2");
    assert_eq!(loaded.examples[1].contexts, vec!["DOC-1 retrieved text"]);
}

/// A "new model" candidate that answers every question wrong — the stand-in for a
/// prompt/model change that regressed quality.
struct WrongCandidate;

#[async_trait]
impl Predictor for WrongCandidate {
    async fn predict(&self, _input: &str) -> Result<String, EvalError> {
        Ok("definitely-wrong".to_string())
    }
}

#[tokio::test]
async fn regression_gate_goes_red_when_the_candidate_model_scores_below_baseline() {
    let dir = write_recording(&sample_exchanges());
    let (dataset, baseline_predictor) =
        replay_golden_from_file(dir.path().join("trace.jsonl"), ReplayStrategy::Fifo).unwrap();

    let runner = EvalRunner::new(vec![Box::new(ExactMatch)]);
    // baseline: the pinned recording, candidate: the changed (wrong) model
    let baseline = runner.run(&dataset, &baseline_predictor).await.unwrap();
    let candidate = runner.run(&dataset, &WrongCandidate).await.unwrap();

    let cmp = compare_reports(&baseline, &candidate, 0.0);
    assert!(cmp.is_regressed());
    let regression = &cmp.regressions[0];
    assert_eq!(regression.evaluator, "exact_match");
    assert!((regression.baseline_mean - 1.0).abs() < 1e-9);
    assert!((regression.candidate_mean - 0.0).abs() < 1e-9);
    assert!(cmp.to_table().contains("REGRESSION"));
}
