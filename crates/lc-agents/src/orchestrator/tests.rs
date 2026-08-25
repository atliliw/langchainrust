// lc-agents/src/orchestrator/tests.rs
//! Unit tests for the orchestrator module.

use super::*;
use crate::task::AgentTask;
use crate::AgentError;
use async_trait::async_trait;
use lc_core::runnables::RunnableConfig;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// 用一个最小可测编排器验证 trait 可用性,不依赖真实 LLM。
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

/// 确定性 mock 子编排器:返回 `tag:objective`。
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

/// 必定失败的 mock 子编排器。
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

/// 记录收到的任务,用于断言约束(目标/预期输出/允许工具)是否随派发传递。
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

/// P2-3: 扇出把同一任务广播给所有 worker,默认换行聚合,顺序稳定。
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

/// P2-3: 自定义聚合函数生效。
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

/// P2-3: 任一 worker 失败则整体失败,错误带 worker 序号。
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

/// P2-3: 空 worker 列表在运行时报错而非返回空串。
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

/// P2-3: 流水线按序串联,前一阶段输出喂给后一阶段。
#[tokio::test]
async fn test_pipeline_order_and_data_flow() {
    let pipe = SequentialPipeline::new(vec![mock_orch("s1"), mock_orch("s2")]);
    let ctx = RunContext::new("t5");
    let out = pipe
        .run_with_context(AgentTask::new("seed"), &ctx)
        .await
        .unwrap();
    // s1 输出 "s1:seed:t5" 作为 s2 目标 → "s2:s1:seed:t5:t5"
    assert_eq!(out, "s2:s1:seed:t5:t5");
}

/// P2-3: 追加阶段生效。
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

/// P2-3: 流水线阶段失败时带序号报错。
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

/// P2-3: 两种模式可互相嵌套(流水线里套扇出)。
#[tokio::test]
async fn test_fanout_nested_in_pipeline() {
    let fanout = FanOutFanIn::new(vec![mock_orch("a"), mock_orch("b")]);
    let pipe = SequentialPipeline::new(vec![Arc::new(fanout), mock_orch("tail")]);
    let ctx = RunContext::new("t8");
    let out = pipe
        .run_with_context(AgentTask::new("in"), &ctx)
        .await
        .unwrap();
    // fanout → "a:in:t8\nb:in:t8"; tail 再包一层
    assert_eq!(out, "tail:a:in:t8\nb:in:t8:t8");
}

/// P2-5: `TaskAdapter` 把 `Input=String` 编排器桥接为可接收任务派发。
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
    // 底层 String 编排器收到目标文本,而非整个任务
    assert_eq!(out, "适配目标 via t9");
}

/// P2-5: 扇出派发时,任务的预期输出 / 允许工具随目标一并到达每个 worker。
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

/// P2-5: 流水线的任务级约束沿链传递,阶段输出成为下一阶段目标。
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
    // 第二段收到前一段输出作为目标,约束沿用
    assert_eq!(seen[1].objective(), "s:起点");
    assert_eq!(seen[1].expected_output(), Some("要点"));
    assert_eq!(seen[1].allowed_tools(), &["calc".to_string()]);
}

// === P2-8: ReviewOrchestrator ===

/// mock worker:记录收到的任务;`first_try_good=true` 首次即达标,
/// 否则只有目标含"修订"(带反馈重做后)才产出达标文本。
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

/// mock worker 工厂返回类型。
type ReviewWorkerPair = (
    Arc<dyn Orchestrator<Input = AgentTask, Output = String>>,
    Arc<Mutex<Vec<AgentTask>>>,
);

/// mock worker 工厂:返回 worker trait 对象 + 记录的任务列表。
fn review_worker(first_try_good: bool) -> ReviewWorkerPair {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let worker = Arc::new(ReviewWorker {
        calls: calls.clone(),
        first_try_good,
    }) as Arc<dyn Orchestrator<Input = AgentTask, Output = String>>;
    (worker, calls)
}

/// mock 评审:产出含 "good" → PASS,否则 FAIL + 反馈(定界符格式)。
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

/// mock 评审:恒返回 FAIL + 反馈(用于验证耗尽路径)。
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

/// P2-8: 首次产出即达标,直接返回,不做重做。
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

/// P2-8: 首轮不达标,带着反馈重做后达标,任务级约束沿链保留。
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
    // 约束沿链保留
    assert_eq!(calls[1].expected_output(), Some("一页结论"));
    assert_eq!(calls[1].allowed_tools(), &["calc".to_string()]);
}

/// P2-8: 尝试耗尽仍未达标,默认返回 Err(不把未过审产出当结果)。
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

/// P2-8: `keep_last_output()` 使耗尽后返回最近产出而非报错。
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

/// P2-8: ReviewOrchestrator 作为组合模式可嵌进流水线。
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
    // review 重做后产出 "good answer" → tail 再包一层
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
