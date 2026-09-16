// lc-agents/src/orchestrator/tests.rs
//! Unit tests for the orchestrator module.

use super::*;
use crate::task::AgentTask;
use crate::AgentError;
use async_trait::async_trait;
use lc_core::runnables::RunnableConfig;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Verifies trait usability with a minimal testable orchestrator, no real LLM needed.
struct DummyOrchestrator;

#[async_trait]
impl Orchestrator for DummyOrchestrator {
    type Input = String;
    type Output = String;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        Ok(format!("{} via {}", input, ctx.trace_id))
    }
}

#[tokio::test]
async fn test_orchestrator_basic() {
    let orch = DummyOrchestrator;
    let ctx = RunContext::new("trace-1");
    let out = orch.run_with_context("hi".to_string(), &ctx).await.unwrap();
    assert_eq!(out, "hi via trace-1");
}

#[test]
fn test_run_context_from_config() {
    let mut cfg = RunnableConfig::new();
    let mut meta = HashMap::new();
    meta.insert("trace_id".to_string(), Value::String("cfg-trace".into()));
    cfg.metadata = meta;
    let ctx = RunContext::from_config(&cfg);
    assert_eq!(ctx.trace_id, "cfg-trace");
}

#[test]
fn test_run_context_from_config_missing_trace() {
    let cfg = RunnableConfig::new();
    let ctx = RunContext::from_config(&cfg);
    assert!(ctx.trace_id.starts_with("trace-"), "{}", ctx.trace_id);
}

#[test]
fn test_generate_trace_id_unique() {
    let a = generate_trace_id();
    let b = generate_trace_id();
    assert_ne!(a, b);
}

/// Deterministic mock sub-orchestrator: returns `tag:objective`.
struct MockOrch {
    tag: &'static str,
}

#[async_trait]
impl Orchestrator for MockOrch {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        task: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        Ok(format!("{}:{}:{}", self.tag, task.objective, ctx.trace_id))
    }
}

/// Mock sub-orchestrator that always fails.
struct MockOrchFail;

#[async_trait]
impl Orchestrator for MockOrchFail {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        _input: Self::Input,
        _ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        Err(AgentError::Other("boom".to_string()))
    }
}

fn mock_orch(tag: &'static str) -> Arc<dyn Orchestrator<Input = AgentTask, Output = String>> {
    Arc::new(MockOrch { tag })
}

/// Records received tasks, to assert whether constraints (objective/expected output/allowed tools) propagate with dispatch.
struct CapturingOrch {
    tag: &'static str,
    seen: Arc<Mutex<Vec<AgentTask>>>,
}

#[async_trait]
impl Orchestrator for CapturingOrch {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        task: Self::Input,
        _ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(task.clone());
        Ok(format!("{}:{}", self.tag, task.objective))
    }
}

/// P2-3: fan-out broadcasts the same task to all workers, newline-joined by default, stable order.
#[tokio::test]
async fn test_fanout_broadcast_and_join() {
    let orch = FanOutFanIn::new(vec![mock_orch("a"), mock_orch("b")]);
    let ctx = RunContext::new("t1");
    let out = orch
        .run_with_context(AgentTask::new("task"), &ctx)
        .await
        .unwrap();
    assert_eq!(out, "a:task:t1\nb:task:t1");
}

/// P2-3: a custom aggregator takes effect.
#[tokio::test]
async fn test_fanout_custom_aggregator() {
    let orch =
        FanOutFanIn::new(vec![mock_orch("a"), mock_orch("b")]).with_aggregator(|vs| vs.join(" + "));
    let ctx = RunContext::new("t2");
    let out = orch
        .run_with_context(AgentTask::new("x"), &ctx)
        .await
        .unwrap();
    assert_eq!(out, "a:x:t2 + b:x:t2");
}

/// P2-3: if any worker fails the whole run fails, with the worker index in the error.
#[tokio::test]
async fn test_fanout_worker_error_fails_all() {
    let orch = FanOutFanIn::new(vec![mock_orch("ok"), Arc::new(MockOrchFail)]);
    let ctx = RunContext::new("t3");
    let err = orch
        .run_with_context(AgentTask::new("y"), &ctx)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("worker 1 failed"));
}

/// P2-3: an empty worker list errors at run time rather than returning an empty string.
#[tokio::test]
async fn test_fanout_empty_workers_errors() {
    let orch = FanOutFanIn::new(vec![]);
    let ctx = RunContext::new("t4");
    let err = orch
        .run_with_context(AgentTask::new("z"), &ctx)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("at least one worker"));
}

/// P2-3: the pipeline chains stages in order, feeding each stage's output into the next.
#[tokio::test]
async fn test_pipeline_order_and_data_flow() {
    let pipe = SequentialPipeline::new(vec![mock_orch("s1"), mock_orch("s2")]);
    let ctx = RunContext::new("t5");
    let out = pipe
        .run_with_context(AgentTask::new("seed"), &ctx)
        .await
        .unwrap();
    // s1 outputs "s1:seed:t5" as s2's objective → "s2:s1:seed:t5:t5"
    assert_eq!(out, "s2:s1:seed:t5:t5");
}

/// P2-3: appended stages take effect.
#[tokio::test]
async fn test_pipeline_push_stage() {
    let pipe = SequentialPipeline::new(vec![mock_orch("s1")]).push_stage(mock_orch("s2"));
    let ctx = RunContext::new("t6");
    let out = pipe
        .run_with_context(AgentTask::new("p"), &ctx)
        .await
        .unwrap();
    assert_eq!(out, "s2:s1:p:t6:t6");
}

/// P2-3: a failing pipeline stage reports the error with its index.
#[tokio::test]
async fn test_pipeline_stage_error_reports_index() {
    let pipe = SequentialPipeline::new(vec![mock_orch("s1"), Arc::new(MockOrchFail)]);
    let ctx = RunContext::new("t7");
    let err = pipe
        .run_with_context(AgentTask::new("q"), &ctx)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("stage 1"));
}

/// P2-3: the two modes nest (fan-out inside a pipeline).
#[tokio::test]
async fn test_fanout_nested_in_pipeline() {
    let fanout = FanOutFanIn::new(vec![mock_orch("a"), mock_orch("b")]);
    let pipe = SequentialPipeline::new(vec![Arc::new(fanout), mock_orch("tail")]);
    let ctx = RunContext::new("t8");
    let out = pipe
        .run_with_context(AgentTask::new("in"), &ctx)
        .await
        .unwrap();
    // fanout → "a:in:t8\nb:in:t8"; tail wraps one more layer
    assert_eq!(out, "tail:a:in:t8\nb:in:t8:t8");
}

/// P2-5: `TaskAdapter` bridges an `Input=String` orchestrator to accept task dispatch.
#[tokio::test]
async fn test_task_adapter_bridges_string_orchestrator() {
    let inner =
        Arc::new(DummyOrchestrator) as Arc<dyn Orchestrator<Input = String, Output = String>>;
    let worker = task_adapter(inner);
    let orch = FanOutFanIn::new(vec![worker]);
    let ctx = RunContext::new("t9");
    let out = orch
        .run_with_context(
            AgentTask::new("适配目标").with_allowed_tools(["calc"]),
            &ctx,
        )
        .await
        .unwrap();
    // The underlying String orchestrator receives the objective text, not the whole task
    assert_eq!(out, "适配目标 via t9");
}

/// P2-5: on fan-out dispatch, the task's expected output / allowed tools arrive at each worker together with the objective.
#[tokio::test]
async fn test_fanout_dispatches_task_with_constraints() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let cap = |tag: &'static str| -> Arc<dyn Orchestrator<Input = AgentTask, Output = String>> {
        Arc::new(CapturingOrch {
            tag,
            seen: seen.clone(),
        })
    };
    let orch = FanOutFanIn::new(vec![cap("a"), cap("b")]);
    let ctx = RunContext::new("t10");
    let task = AgentTask::new("研究X")
        .with_expected_output("给出一页结论")
        .with_allowed_tools(["web_search", "calculator"]);
    let out = orch.run_with_context(task, &ctx).await.unwrap();
    assert_eq!(out, "a:研究X\nb:研究X");
    let seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(seen.len(), 2);
    for t in seen.iter() {
        assert_eq!(t.objective(), "研究X");
        assert_eq!(t.expected_output(), Some("给出一页结论"));
        assert_eq!(
            t.allowed_tools(),
            &["web_search".to_string(), "calculator".to_string()]
        );
    }
}

/// P2-5: pipeline task-level constraints propagate along the chain; each stage's output becomes the next stage's objective.
#[tokio::test]
async fn test_pipeline_carries_constraints_through_stages() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stage = Arc::new(CapturingOrch {
        tag: "s",
        seen: seen.clone(),
    }) as Arc<dyn Orchestrator<Input = AgentTask, Output = String>>;
    let pipe = SequentialPipeline::new(vec![stage.clone(), stage.clone()]);
    let ctx = RunContext::new("t11");
    let task = AgentTask::new("起点")
        .with_expected_output("要点")
        .with_allowed_tools(["calc"]);
    let out = pipe.run_with_context(task, &ctx).await.unwrap();
    assert_eq!(out, "s:s:起点");
    let seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].objective(), "起点");
    assert_eq!(seen[0].expected_output(), Some("要点"));
    assert_eq!(seen[0].allowed_tools(), &["calc".to_string()]);
    // The second stage receives the previous stage's output as its objective, constraints carried over
    assert_eq!(seen[1].objective(), "s:起点");
    assert_eq!(seen[1].expected_output(), Some("要点"));
    assert_eq!(seen[1].allowed_tools(), &["calc".to_string()]);
}

// === P2-8: ReviewOrchestrator ===

/// Mock worker: records received tasks; with `first_try_good=true` the first
/// try already passes, otherwise it produces a passing text only when the
/// objective contains "修订" (i.e. after redoing with feedback).
struct ReviewWorker {
    calls: Arc<Mutex<Vec<AgentTask>>>,
    first_try_good: bool,
}

#[async_trait]
impl Orchestrator for ReviewWorker {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        task: Self::Input,
        _ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        self.calls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(task.clone());
        if self.first_try_good || task.objective.contains("修订") {
            Ok("good answer".to_string())
        } else {
            Ok("bad answer".to_string())
        }
    }
}

/// Return type of the mock worker factory.
type ReviewWorkerPair = (
    Arc<dyn Orchestrator<Input = AgentTask, Output = String>>,
    Arc<Mutex<Vec<AgentTask>>>,
);

/// Mock worker factory: returns the worker trait object + the recorded task list.
fn review_worker(first_try_good: bool) -> ReviewWorkerPair {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let worker = Arc::new(ReviewWorker {
        calls: calls.clone(),
        first_try_good,
    }) as Arc<dyn Orchestrator<Input = AgentTask, Output = String>>;
    (worker, calls)
}

/// Mock reviewer: output containing "good" → PASS, otherwise FAIL + feedback (delimited format).
struct ReviewChecker;

#[async_trait]
impl Orchestrator for ReviewChecker {
    type Input = String;
    type Output = String;

    async fn run_with_context(
        &self,
        input: Self::Input,
        _ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        if input.contains("good answer") {
            Ok("<<<VERDICT>>>PASS<<<END_VERDICT>>>".to_string())
        } else {
            Ok(
                "<<<VERDICT>>>FAIL<<<END_VERDICT>>>\n<<<FEEDBACK>>>请补充细节<<<END_FEEDBACK>>>"
                    .to_string(),
            )
        }
    }
}

/// Mock reviewer: always returns FAIL + feedback (for verifying the exhaustion path).
struct AlwaysFailReview;

#[async_trait]
impl Orchestrator for AlwaysFailReview {
    type Input = String;
    type Output = String;

    async fn run_with_context(
        &self,
        _input: Self::Input,
        _ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        Ok(
            "<<<VERDICT>>>FAIL<<<END_VERDICT>>>\n<<<FEEDBACK>>>还差得远<<<END_FEEDBACK>>>"
                .to_string(),
        )
    }
}

/// P2-8: the first output already passes; return directly without redoing.
#[tokio::test]
async fn test_review_passes_on_first_attempt() {
    let (worker, calls) = review_worker(true);
    let orch = ReviewOrchestrator::new(worker, Arc::new(ReviewChecker), 3);
    let ctx = RunContext::new("r1");
    let out = orch
        .run_with_context(AgentTask::new("写报告"), &ctx)
        .await
        .unwrap();
    assert_eq!(out, "good answer");
    assert_eq!(
        calls.lock().unwrap_or_else(|e| e.into_inner()).len(),
        1,
        "达标后不应重做"
    );
}

/// P2-8: the first round fails; after redoing with feedback it passes, with task-level constraints kept along the chain.
#[tokio::test]
async fn test_review_redo_until_pass() {
    let (worker, calls) = review_worker(false);
    let orch = ReviewOrchestrator::new(worker, Arc::new(ReviewChecker), 3);
    let ctx = RunContext::new("r2");
    let task = AgentTask::new("写报告")
        .with_expected_output("一页结论")
        .with_allowed_tools(["calc"]);
    let out = orch.run_with_context(task, &ctx).await.unwrap();
    assert_eq!(out, "good answer");
    let calls = calls.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(calls.len(), 2, "首轮不达标应重做一轮");
    assert_eq!(calls[0].objective(), "写报告");
    assert!(
        calls[1].objective().contains("请补充细节"),
        "第二轮目标应携带评审反馈, 实际: {}",
        calls[1].objective()
    );
    // Constraints carried along the chain
    assert_eq!(calls[1].expected_output(), Some("一页结论"));
    assert_eq!(calls[1].allowed_tools(), &["calc".to_string()]);
}

/// P2-8: attempts exhausted without passing → Err by default (never treats an unapproved output as the result).
#[tokio::test]
async fn test_review_exhausts_returns_error_by_default() {
    let (worker, _) = review_worker(false);
    let orch = ReviewOrchestrator::new(worker, Arc::new(AlwaysFailReview), 2);
    let ctx = RunContext::new("r3");
    let err = orch
        .run_with_context(AgentTask::new("任务"), &ctx)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("did not pass"), "{}", err);
}

/// P2-8: `keep_last_output()` returns the latest output on exhaustion instead of erroring.
#[tokio::test]
async fn test_review_keep_last_output_on_exhaustion() {
    let (worker, _) = review_worker(true);
    let orch = ReviewOrchestrator::new(worker, Arc::new(AlwaysFailReview), 2).keep_last_output();
    let ctx = RunContext::new("r4");
    let out = orch
        .run_with_context(AgentTask::new("任务"), &ctx)
        .await
        .unwrap();
    assert_eq!(out, "good answer");
}

/// P2-8: ReviewOrchestrator as a composition pattern nests inside a pipeline.
#[tokio::test]
async fn test_review_orchestrator_composes_in_pipeline() {
    let (worker, _) = review_worker(false);
    let review = ReviewOrchestrator::new(worker, Arc::new(ReviewChecker), 3);
    let pipe = SequentialPipeline::new(vec![Arc::new(review), mock_orch("tail")]);
    let ctx = RunContext::new("r5");
    let out = pipe
        .run_with_context(AgentTask::new("研究X"), &ctx)
        .await
        .unwrap();
    // review redo produces "good answer" → tail wraps one more layer
    assert_eq!(out, "tail:good answer:r5");
}

#[test]
fn test_parse_review_verdict_json() {
    assert_eq!(
        parse_review_verdict(r#"{"passed": true}"#),
        Some(ReviewVerdict::pass())
    );
    assert_eq!(
        parse_review_verdict(r#"{"passed": false, "feedback": "缺引用"}"#),
        Some(ReviewVerdict::fail("缺引用"))
    );
}

#[test]
fn test_parse_review_verdict_delimited() {
    let v = parse_review_verdict(
        "<<<VERDICT>>>FAIL<<<END_VERDICT>>>\n<<<FEEDBACK>>>请补充细节<<<END_FEEDBACK>>>",
    )
    .unwrap();
    assert!(!v.passed);
    assert_eq!(v.feedback, "请补充细节");

    let p = parse_review_verdict("<<<VERDICT>>>PASS<<<END_VERDICT>>>").unwrap();
    assert!(p.passed);
    assert!(p.feedback.is_empty());
}

#[test]
fn test_parse_review_verdict_plain_text() {
    let p = parse_review_verdict("PASS").unwrap();
    assert!(p.passed);

    let f = parse_review_verdict("FAIL: 引用不足").unwrap();
    assert!(!f.passed);
    assert_eq!(f.feedback, "引用不足");
}

#[test]
fn test_parse_review_verdict_invalid() {
    assert!(parse_review_verdict("whatever").is_none());
}

// === N1 (v0.24.0): Supervisor dynamic routing (one-level sub-agent recursion) ===

/// Worker that records the task and the trace id it was called with, and can be
/// told to fail — used to prove delegation, feedback, constraint propagation and
/// error paths without any network.
struct SupWorker {
    tag: &'static str,
    seen: Arc<Mutex<Vec<AgentTask>>>,
    traces: Arc<Mutex<Vec<String>>>,
    fail: bool,
}

#[async_trait]
impl Orchestrator for SupWorker {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        task: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(task.clone());
        self.traces
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(ctx.trace_id.clone());
        if self.fail {
            return Err(AgentError::Other("boom".to_string()));
        }
        Ok(format!("{}:{}", self.tag, task.objective))
    }
}

type SupWorkerHandle = (
    Arc<dyn Orchestrator<Input = AgentTask, Output = String>>,
    Arc<Mutex<Vec<AgentTask>>>,
    Arc<Mutex<Vec<String>>>,
);

fn sup_worker(tag: &'static str, fail: bool) -> SupWorkerHandle {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let traces = Arc::new(Mutex::new(Vec::new()));
    let worker = Arc::new(SupWorker {
        tag,
        seen: seen.clone(),
        traces: traces.clone(),
        fail,
    }) as Arc<dyn Orchestrator<Input = AgentTask, Output = String>>;
    (worker, seen, traces)
}

/// A hermetic supervisor "model": parses the envelope JSON and chooses the next
/// worker purely from the accumulated results, recording call count and trace id.
struct ScriptedRouter {
    calls: Arc<AtomicUsize>,
    traces: Arc<Mutex<Vec<String>>>,
    expected_trace: Option<&'static str>,
    route: fn(&[(String, String)], &str) -> SupervisorNext,
}

#[async_trait]
impl Orchestrator for ScriptedRouter {
    type Input = String;
    type Output = String;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.traces
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(ctx.trace_id.clone());
        if let Some(expected) = self.expected_trace {
            assert_eq!(ctx.trace_id, expected);
        }
        let envelope: Value = serde_json::from_str(&input).expect("router envelope must be JSON");
        let objective = envelope
            .get("objective")
            .and_then(Value::as_str)
            .unwrap_or("");
        let results: Vec<(String, String)> = envelope
            .get("results")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|r| {
                        (
                            r.get("worker")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            r.get("output")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let next = (self.route)(&results, objective);
        Ok(match next {
            SupervisorNext::Work { worker, task } => {
                json!({"next": worker, "task": task}).to_string()
            }
            SupervisorNext::Finish { answer } => {
                json!({"next": SUPERVISOR_FINISH, "answer": answer}).to_string()
            }
        })
    }
}

fn scripted_router(
    route: fn(&[(String, String)], &str) -> SupervisorNext,
    expected_trace: Option<&'static str>,
) -> (
    Arc<dyn Orchestrator<Input = String, Output = String>>,
    Arc<AtomicUsize>,
) {
    let calls = Arc::new(AtomicUsize::new(0));
    let router = Arc::new(ScriptedRouter {
        calls: calls.clone(),
        traces: Arc::new(Mutex::new(Vec::new())),
        expected_trace,
        route,
    }) as Arc<dyn Orchestrator<Input = String, Output = String>>;
    (router, calls)
}

/// research → write → finish, each decision driven by what earlier workers returned.
fn route_research_then_write(results: &[(String, String)], _objective: &str) -> SupervisorNext {
    if !results.iter().any(|(w, _)| w == "researcher") {
        SupervisorNext::Work {
            worker: "researcher".to_string(),
            task: "gather facts".to_string(),
        }
    } else if !results.iter().any(|(w, _)| w == "writer") {
        SupervisorNext::Work {
            worker: "writer".to_string(),
            task: "draft from research".to_string(),
        }
    } else {
        let writer_output = results
            .iter()
            .rev()
            .find(|(w, _)| w == "writer")
            .map(|(_, o)| o.clone())
            .unwrap_or_default();
        SupervisorNext::Finish {
            answer: format!("final: {writer_output}"),
        }
    }
}

/// N1: the supervisor dynamically picks one worker per round and feeds each
/// worker's output back into the next routing decision.
#[tokio::test]
async fn test_supervisor_routes_dynamically_and_feeds_results_back() {
    let (researcher, r_seen, r_traces) = sup_worker("researcher", false);
    let (writer, w_seen, w_traces) = sup_worker("writer", false);
    let (router, router_calls) = scripted_router(route_research_then_write, Some("sup-1"));

    let sup = Supervisor::new(
        router,
        vec![
            ("researcher".to_string(), researcher),
            ("writer".to_string(), writer),
        ],
        5,
    );
    let ctx = RunContext::new("sup-1");
    let out = sup
        .run_with_context(AgentTask::new("做个调研并成稿"), &ctx)
        .await
        .unwrap();

    // The FINISH answer embeds the writer's actual output, proving the worker
    // result was fed back through the scratchpad.
    assert_eq!(out, "final: writer:draft from research");
    // Three routing rounds: delegate researcher, delegate writer, finish.
    assert_eq!(router_calls.load(Ordering::SeqCst), 3);

    // Each sub-agent ran exactly once with the task the supervisor issued.
    let r_seen = r_seen.lock().unwrap_or_else(|e| e.into_inner());
    let w_seen = w_seen.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(r_seen.len(), 1);
    assert_eq!(w_seen.len(), 1);
    assert_eq!(r_seen[0].objective(), "gather facts");
    assert_eq!(w_seen[0].objective(), "draft from research");

    // The run context (trace id) propagates to both the router and every worker.
    assert_eq!(
        &**r_traces.lock().unwrap_or_else(|e| e.into_inner()),
        &["sup-1".to_string()]
    );
    assert_eq!(
        &**w_traces.lock().unwrap_or_else(|e| e.into_inner()),
        &["sup-1".to_string()]
    );
}

/// N1: the worker list/order is exposed for prompts and inspection.
#[test]
fn test_supervisor_worker_names_and_round_clamp() {
    let (worker, _, _) = sup_worker("a", false);
    let (router, _) = scripted_router(route_research_then_write, None);
    let sup = Supervisor::new(router, vec![("a".to_string(), worker)], 0);
    assert_eq!(sup.max_rounds(), 1, "max_rounds must clamp to at least 1");
    assert_eq!(sup.worker_names(), vec!["a"]);
    assert_eq!(sup.with_max_rounds(7).max_rounds(), 7);
}

fn route_unknown_worker(_results: &[(String, String)], _o: &str) -> SupervisorNext {
    SupervisorNext::Work {
        worker: "ghost".to_string(),
        task: "x".to_string(),
    }
}

/// N1: a decision naming a worker that was never registered errors explicitly
/// (and never invokes a real worker by accident).
#[tokio::test]
async fn test_supervisor_unknown_worker_errors() {
    let (worker, seen, _) = sup_worker("a", false);
    let (router, _) = scripted_router(route_unknown_worker, None);
    let sup = Supervisor::new(router, vec![("a".to_string(), worker)], 3);
    let err = sup
        .run_with_context(AgentTask::new("t"), &RunContext::new("sup-2"))
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("unknown worker 'ghost'"), "{msg}");
    assert!(
        msg.contains('['),
        "error should list available workers: {msg}"
    );
    assert!(seen.lock().unwrap_or_else(|e| e.into_inner()).is_empty());
}

fn route_delegate_forever(results: &[(String, String)], _o: &str) -> SupervisorNext {
    SupervisorNext::Work {
        worker: "a".to_string(),
        task: format!("round-{}", results.len()),
    }
}

/// N1: delegation is bounded — never FINISHing within `max_rounds` errors instead
/// of looping forever (the one-level recursion guard).
#[tokio::test]
async fn test_supervisor_exhausts_rounds_and_errors() {
    let (worker, seen, _) = sup_worker("a", false);
    let (router, router_calls) = scripted_router(route_delegate_forever, None);
    let sup = Supervisor::new(router, vec![("a".to_string(), worker)], 3);
    let err = sup
        .run_with_context(AgentTask::new("loop"), &RunContext::new("sup-3"))
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("did not reach FINISH within 3"), "{msg}");
    assert_eq!(router_calls.load(Ordering::SeqCst), 3);
    let seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(seen.len(), 3, "worker should run once per round");
    assert_eq!(seen[2].objective(), "round-2");
}

fn route_to_flaky(results: &[(String, String)], _o: &str) -> SupervisorNext {
    if results.is_empty() {
        SupervisorNext::Work {
            worker: "flaky".to_string(),
            task: "do it".to_string(),
        }
    } else {
        SupervisorNext::Finish {
            answer: "unreachable".to_string(),
        }
    }
}

/// N1: a sub-agent failure propagates with the worker name and round for diagnosis.
#[tokio::test]
async fn test_supervisor_worker_failure_reports_worker_name() {
    let (worker, _, _) = sup_worker("flaky", true);
    let (router, _) = scripted_router(route_to_flaky, None);
    let sup = Supervisor::new(router, vec![("flaky".to_string(), worker)], 3);
    let err = sup
        .run_with_context(AgentTask::new("t"), &RunContext::new("sup-4"))
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("worker 'flaky'"), "{msg}");
    assert!(msg.contains("boom"), "{msg}");
}

fn route_delegate_once_then_finish(results: &[(String, String)], _o: &str) -> SupervisorNext {
    if results.is_empty() {
        SupervisorNext::Work {
            worker: "w".to_string(),
            task: "subtask".to_string(),
        }
    } else {
        SupervisorNext::Finish {
            answer: results[0].1.clone(),
        }
    }
}

/// N1: the parent task's contract (expected output / tool allowlist) propagates
/// to the sub-agent's fresh task.
#[tokio::test]
async fn test_supervisor_propagates_task_constraints() {
    let (worker, seen, traces) = sup_worker("w", false);
    let (router, _) = scripted_router(route_delegate_once_then_finish, None);
    let sup = Supervisor::new(router, vec![("w".to_string(), worker)], 3);
    let task = AgentTask::new("父目标")
        .with_expected_output("一页结论")
        .with_allowed_tools(["web_search", "calculator"]);
    let out = sup
        .run_with_context(task, &RunContext::new("sup-5"))
        .await
        .unwrap();
    assert_eq!(out, "w:subtask");
    let seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].objective(), "subtask");
    assert_eq!(seen[0].expected_output(), Some("一页结论"));
    assert_eq!(
        seen[0].allowed_tools(),
        &["web_search".to_string(), "calculator".to_string()]
    );
    assert_eq!(
        &**traces.lock().unwrap_or_else(|e| e.into_inner()),
        &["sup-5".to_string()]
    );
}

/// N1: no workers is a run-time configuration error.
#[tokio::test]
async fn test_supervisor_empty_workers_errors() {
    let (router, _) = scripted_router(route_research_then_write, None);
    let sup = Supervisor::new(router, vec![], 3);
    let err = sup
        .run_with_context(AgentTask::new("t"), &RunContext::new("sup-6"))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("at least one worker"));
}

#[test]
fn test_parse_supervisor_decision_json() {
    assert_eq!(
        parse_supervisor_decision(r#"{"next": "researcher", "task": "查资料"}"#),
        Some(SupervisorNext::Work {
            worker: "researcher".to_string(),
            task: "查资料".to_string(),
        })
    );
    assert_eq!(
        parse_supervisor_decision(r#"{"next": "FINISH", "answer": "结论"}"#),
        Some(SupervisorNext::Finish {
            answer: "结论".to_string(),
        })
    );
    // FINISH is matched case-insensitively.
    assert_eq!(
        parse_supervisor_decision(r#"{"next": "finish", "answer": "done"}"#),
        Some(SupervisorNext::Finish {
            answer: "done".to_string(),
        })
    );
}

#[test]
fn test_parse_supervisor_decision_delimited() {
    let work =
        parse_supervisor_decision("<<<NEXT>>>searcher<<<END_NEXT>>>\n<<<TASK>>>查 X<<<END_TASK>>>")
            .unwrap();
    assert_eq!(
        work,
        SupervisorNext::Work {
            worker: "searcher".to_string(),
            task: "查 X".to_string(),
        }
    );

    let finish = parse_supervisor_decision(
        "<<<NEXT>>>FINISH<<<END_NEXT>>>\n<<<ANSWER>>>最终稿<<<END_ANSWER>>>",
    )
    .unwrap();
    assert_eq!(
        finish,
        SupervisorNext::Finish {
            answer: "最终稿".to_string(),
        }
    );
}

#[test]
fn test_parse_supervisor_decision_invalid() {
    assert!(parse_supervisor_decision("whatever").is_none());
    assert!(parse_supervisor_decision(r#"{"task": "no next field"}"#).is_none());
}

#[test]
fn test_supervisor_envelope_carries_objective_workers_and_history() {
    let envelope = supervisor_envelope(
        "目标",
        &["a".to_string(), "b".to_string()],
        2,
        &[("a".to_string(), "a-output".to_string())],
    );
    let v: Value = serde_json::from_str(&envelope).unwrap();
    assert_eq!(v["objective"], "目标");
    assert_eq!(v["workers"], json!(["a", "b"]));
    assert_eq!(v["round"], 2);
    assert_eq!(v["results"][0]["worker"], "a");
    assert_eq!(v["results"][0]["output"], "a-output");
}
