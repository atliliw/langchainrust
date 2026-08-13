//! 高层编排器公共 trait(P1-1)
//!
//! `PlanExecuteAgent` / `DeepResearchAgent` / `CorrectiveRAGAgent` / `AdaptiveRAG`
//! 此前各写各的 `run()`,签名互不兼容,无法组合、无法进 LCEL。这里统一收敛:
//!
//! - [`Orchestrator`] 定义 `run_with_context(input, ctx)`,错误统一到 [`AgentError`]。
//! - [`RunContext`] 携带 `trace_id`(P1-4 可观测性)与跨步骤共享工作区。
//! - [`crate::adapter::OrchestratorRunnable`] 让编排器能进 LCEL 管道。
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_agents::orchestration::{Orchestrator, RunContext};
//!
//! let plan_agent = PlanExecuteAgent::new(llm, tools);
//! let ctx = RunContext::new("trace-abc");
//! let output = plan_agent.run_with_context("目标".to_string(), &ctx).await?;
//! ```

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lc_core::language_models::BaseChatModel;
use lc_core::runnables::RunnableConfig;
use lc_rag::RetrieverTrait;
use serde_json::Value;
use tokio::sync::Semaphore;

use crate::task::AgentTask;
use crate::{
    AdaptiveRAG, AdaptiveRAGResult, AgentError, CRAGResult, CorrectiveRAGAgent, DeepResearchAgent,
    PlanExecuteAgent, ResearchReport,
};

/// 高层编排器公共 trait。
///
/// 关联类型表达各编排器不同的输入/输出(PlanExecute→String,AdaptiveRAG→AdaptiveRAGResult 等),
/// `run_with_context` 统一签名 + 统一 [`AgentError`],让编排器可组合、可进 LCEL。
#[async_trait]
pub trait Orchestrator: Send + Sync {
    /// 输入类型(通常为 `String` 目标/问题)。
    type Input;
    /// 输出类型。
    type Output;

    /// 携带运行上下文的执行入口。
    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError>;
}

/// 编排器运行上下文。
///
/// `trace_id` 在多 Agent / 跨步骤调用链间传播(P1-4);`shared_state` 提供
/// 跨步骤共享的 JSON 工作区。
#[derive(Debug, Clone)]
pub struct RunContext {
    /// 追踪 ID:整条调用链共享,用于日志/审计/指标关联。
    pub trace_id: String,
    /// 跨步骤共享工作区(可选)。
    pub shared_state: Option<Arc<Mutex<Value>>>,
}

/// 生成一个轻量 trace_id(时间戳十六进制)。
pub fn generate_trace_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("trace-{:x}", nanos)
}

impl RunContext {
    /// 创建上下文,指定 `trace_id`。
    pub fn new(trace_id: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            shared_state: None,
        }
    }

    /// 创建上下文,自动生成 `trace_id`。
    pub fn new_random() -> Self {
        Self::new(generate_trace_id())
    }

    /// 携带共享工作区。
    pub fn with_shared_state(mut self, shared_state: Arc<Mutex<Value>>) -> Self {
        self.shared_state = Some(shared_state);
        self
    }

    /// 从 LCEL [`RunnableConfig`] 提取 `trace_id`(读 `metadata["trace_id"]`),
    /// 缺失则自动生成。用于把 LCEL 管道的 trace 贯通到编排器。
    pub fn from_config(config: &RunnableConfig) -> Self {
        let trace_id = config
            .metadata
            .get("trace_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(generate_trace_id);
        Self::new(trace_id)
    }
}

// ---------------------------------------------------------------------------
// 四个编排器 impl
// ---------------------------------------------------------------------------

#[async_trait]
impl Orchestrator for PlanExecuteAgent {
    type Input = String;
    type Output = String;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        log::debug!(
            target: "lc_agents::orchestrator",
            "PlanExecuteAgent start, trace_id = {}",
            ctx.trace_id
        );
        self.run(&input)
            .await
            .map_err(|e| AgentError::Other(format!("PlanExecute: {e}")))
    }
}

#[async_trait]
impl<M, R> Orchestrator for AdaptiveRAG<M, R>
where
    M: BaseChatModel + Send + Sync,
    M::Error: Send + Sync,
    R: RetrieverTrait + Send + Sync,
{
    type Input = String;
    type Output = AdaptiveRAGResult;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        log::debug!(
            target: "lc_agents::orchestrator",
            "AdaptiveRAG start, trace_id = {}",
            ctx.trace_id
        );
        self.invoke(&input)
            .await
            .map_err(|e| AgentError::Other(format!("AdaptiveRAG: {e}")))
    }
}

#[async_trait]
impl<M, R> Orchestrator for CorrectiveRAGAgent<M, R>
where
    M: BaseChatModel + Send + Sync,
    M::Error: Send + Sync,
    R: RetrieverTrait + Send + Sync,
{
    type Input = String;
    type Output = CRAGResult;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        log::debug!(
            target: "lc_agents::orchestrator",
            "CorrectiveRAGAgent start, trace_id = {}",
            ctx.trace_id
        );
        self.invoke(&input)
            .await
            .map_err(|e| AgentError::Other(format!("CorrectiveRAG: {e}")))
    }
}

#[async_trait]
impl<M> Orchestrator for DeepResearchAgent<M>
where
    M: BaseChatModel + Send + Sync,
    M::Error: Send + Sync,
{
    type Input = String;
    type Output = ResearchReport;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        log::debug!(
            target: "lc_agents::orchestrator",
            "DeepResearchAgent start, trace_id = {}",
            ctx.trace_id
        );
        self.research(&input)
            .await
            .map_err(|e| AgentError::Other(format!("DeepResearch: {e}")))
    }
}

// ---------------------------------------------------------------------------
// P2-3 / P2-5: 编排器组合模式(FanOutFanIn / SequentialPipeline)
// ---------------------------------------------------------------------------
//
// 两类"模式实现"本身也实现 [`Orchestrator`],因此可以互相嵌套:流水线里套
// 扇出、扇出的 worker 再是流水线,均通过统一 trait 组合进 LCEL。
//
// P2-5 起派发的任务从裸 `String` 提升为显式 [`AgentTask`](目标 + 预期输出 +
// 允许工具),worker / 阶段以 [`AgentTask`] 为输入;真实 Agent(`Input=String`)
// 经 [`TaskAdapter`] 桥接后即可接收任务派发。

/// 扇出-聚合编排器(Supervisor 的一种形态,P2-3)。
///
/// 把同一任务广播给 N 个子编排器**并行**执行(受 `max_concurrency` 限流),
/// 全部完成后把各自的输出交给聚合函数合并。适合"多角色评审 / 委员会"场景:
/// 多个独立视角各算各的,最后统一裁决。
///
/// 默认聚合是换行拼接;可用 [`FanOutFanIn::with_aggregator`] 换成投票 /
/// 择优等自定义策略。任一 worker 失败即整体失败(不吞错)。
pub struct FanOutFanIn {
    workers: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>,
    aggregator: Box<dyn Fn(Vec<String>) -> String + Send + Sync>,
    max_concurrency: usize,
    semaphore: Arc<Semaphore>,
}

impl FanOutFanIn {
    /// 用一组同构 `AgentTask -> String` 子编排器构造,聚合默认换行拼接。
    ///
    /// 真实 Agent(`Input=String`)用 [`task_adapter`] 包装后再放进 worker 列表。
    pub fn new(workers: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>) -> Self {
        let n = workers.len().max(1);
        Self {
            workers,
            aggregator: Box::new(|results| results.join("\n")),
            max_concurrency: n,
            semaphore: Arc::new(Semaphore::new(n)),
        }
    }

    /// 自定义聚合函数(如择优、投票、拼接模板)。
    pub fn with_aggregator(
        mut self,
        aggregator: impl Fn(Vec<String>) -> String + Send + Sync + 'static,
    ) -> Self {
        self.aggregator = Box::new(aggregator);
        self
    }

    /// 限制并行 worker 数(至少 1)。
    pub fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self.semaphore = Arc::new(Semaphore::new(self.max_concurrency));
        self
    }
}

#[async_trait]
impl Orchestrator for FanOutFanIn {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        if self.workers.is_empty() {
            return Err(AgentError::Other(
                "FanOutFanIn requires at least one worker".to_string(),
            ));
        }
        log::debug!(
            target: "lc_agents::orchestrator",
            "FanOutFanIn start workers={} concurrency={} trace_id={}",
            self.workers.len(),
            self.max_concurrency,
            ctx.trace_id
        );

        let futures = self.workers.iter().enumerate().map(|(i, worker)| {
            let worker = worker.clone();
            let input = input.clone();
            let ctx = ctx.clone();
            let sem = self.semaphore.clone();
            async move {
                let _permit = sem
                    .acquire_owned()
                    .await
                    .map_err(|e| AgentError::Other(format!("FanOutFanIn semaphore: {e}")))?;
                worker
                    .run_with_context(input, &ctx)
                    .await
                    .map_err(|e| AgentError::Other(format!("worker {i} failed: {e}")))
            }
        });

        let outputs = futures_util::future::join_all(futures).await;
        let mut results = Vec::with_capacity(outputs.len());
        for output in outputs {
            results.push(output?);
        }
        Ok((self.aggregator)(results))
    }
}

/// 顺序流水线编排器(P2-3 / P2-5)。
///
/// 把 N 个阶段按序执行:前一阶段的输出作为后一阶段的目标,返回末阶段输出。
/// 任务级约束(预期输出 / 允许工具)沿链传递,各阶段保持一致。
/// 各阶段独立失败即整体失败,报错带阶段序号便于定位。
pub struct SequentialPipeline {
    stages: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>,
}

impl SequentialPipeline {
    /// 用一组顺序执行的阶段构造。
    pub fn new(stages: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>) -> Self {
        Self { stages }
    }

    /// 追加一个阶段(链式)。
    pub fn push_stage(
        mut self,
        stage: Arc<dyn Orchestrator<Input = AgentTask, Output = String>>,
    ) -> Self {
        self.stages.push(stage);
        self
    }
}

#[async_trait]
impl Orchestrator for SequentialPipeline {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        let mut current = input;
        for (i, stage) in self.stages.iter().enumerate() {
            log::debug!(
                target: "lc_agents::orchestrator",
                "SequentialPipeline stage {i} trace_id = {}",
                ctx.trace_id
            );
            let output = stage
                .run_with_context(current.clone(), ctx)
                .await
                .map_err(|e| AgentError::Other(format!("SequentialPipeline stage {i}: {e}")))?;
            // 阶段输出成为下一阶段的目标;任务级约束沿链传递(P2-5)。
            let mut next = AgentTask::new(output);
            if let Some(expected) = current.expected_output.clone() {
                next = next.with_expected_output(expected);
            }
            next = next.with_allowed_tools(current.allowed_tools.clone());
            current = next;
        }
        Ok(current.objective)
    }
}

/// 把 `Input=String` 的编排器适配为消费 [`AgentTask`] 的子 Agent(P2-5)。
///
/// 桥接两类编排器:真实 Agent(PlanExecuteAgent / DeepResearchAgent 等,
/// `Input=String`)经此包装后,可放进 `FanOutFanIn` / `SequentialPipeline`
/// 接受 [`AgentTask`] 派发。取目标喂给底层编排器;任务声明了 `allowed_tools`
/// 时,由消费方(AgentExecutor 等)按白名单装配,此处不越界替其过滤。
pub struct TaskAdapter {
    inner: Arc<dyn Orchestrator<Input = String, Output = String>>,
}

impl TaskAdapter {
    /// 包装一个 `Input=String` 的编排器。
    pub fn new(inner: Arc<dyn Orchestrator<Input = String, Output = String>>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Orchestrator for TaskAdapter {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        task: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        log::debug!(
            target: "lc_agents::orchestrator",
            "TaskAdapter dispatch objective='{}' trace_id = {}",
            task.objective,
            ctx.trace_id
        );
        self.inner.run_with_context(task.objective, ctx).await
    }
}

/// 便捷包装:把 `Input=String` 编排器转成可接收 [`AgentTask`] 派发的 trait 对象。
pub fn task_adapter(
    inner: Arc<dyn Orchestrator<Input = String, Output = String>>,
) -> Arc<dyn Orchestrator<Input = AgentTask, Output = String>> {
    Arc::new(TaskAdapter::new(inner))
}

// ---------------------------------------------------------------------------
// P2-8: 自省/验证评审(ReviewOrchestrator)
// ---------------------------------------------------------------------------
//
// 借鉴 DeepResearch 的 gap 检查思路泛化:worker 产出 → 评审 Agent 检验 →
// 不达标则带着评审反馈重做,直到达标或尝试耗尽。评审 Agent 本身也是一个
// [`Orchestrator`]:输入为 JSON 信封(目标 + 预期输出 + 产出,见
// [`review_envelope`]),输出为评审结论(格式见 [`parse_review_verdict`])。

/// 评审结论:是否达标 + 不达标时的修订反馈。
#[derive(Debug, Clone, PartialEq)]
pub struct ReviewVerdict {
    /// 是否达标。
    pub passed: bool,
    /// 不达标时给 worker 的修订反馈(达标时为空)。
    pub feedback: String,
}

impl ReviewVerdict {
    /// 达标结论。
    pub fn pass() -> Self {
        Self {
            passed: true,
            feedback: String::new(),
        }
    }

    /// 不达标结论,携带修订反馈。
    pub fn fail(feedback: impl Into<String>) -> Self {
        Self {
            passed: false,
            feedback: feedback.into(),
        }
    }
}

/// 构造评审 Agent 的输入信封:把任务(目标 / 预期输出)连同产出打包成 JSON,
/// 评审 Agent 据此判断产出是否达标。
pub fn review_envelope(task: &AgentTask, output: &str) -> String {
    serde_json::json!({
        "objective": task.objective,
        "expected_output": task.expected_output,
        "output": output,
    })
    .to_string()
}

/// 解析评审 Agent 的结论文本为 [`ReviewVerdict`]。
///
/// LLM 输出各异,逐级回退支持三种形态:
/// 1. JSON:`{"passed": true}` 或 `{"passed": false, "feedback": "..."}`;
/// 2. 定界符:`<<<VERDICT>>>PASS|FAIL<<<END_VERDICT>>>`,反馈可选包在
///    `<<<FEEDBACK>>>...<<<END_FEEDBACK>>>`;
/// 3. 纯文本:以 `PASS` / `FAIL` 开头(大小写不敏感),`FAIL` 后接反馈文本。
pub fn parse_review_verdict(text: &str) -> Option<ReviewVerdict> {
    let text = text.trim();

    // 1. JSON
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        if let Some(passed) = value.get("passed").and_then(Value::as_bool) {
            let feedback = value
                .get("feedback")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            return Some(ReviewVerdict { passed, feedback });
        }
    }

    // 2. 定界符
    if let Some(inner) = between(text, "<<<VERDICT>>>", "<<<END_VERDICT>>>") {
        if inner.eq_ignore_ascii_case("PASS") || inner.eq_ignore_ascii_case("FAIL") {
            let passed = inner.eq_ignore_ascii_case("PASS");
            let feedback = between(text, "<<<FEEDBACK>>>", "<<<END_FEEDBACK>>>")
                .unwrap_or("")
                .to_string();
            return Some(ReviewVerdict { passed, feedback });
        }
    }

    // 3. 纯文本
    let upper = text.to_uppercase();
    if upper.starts_with("PASS") {
        return Some(ReviewVerdict::pass());
    }
    if upper.starts_with("FAIL") {
        let feedback = text
            .trim_start_matches("FAIL")
            .trim()
            .trim_start_matches(':')
            .trim()
            .to_string();
        return Some(ReviewVerdict::fail(feedback));
    }

    None
}

/// 取两个标记之间的文本(含 trim),标记缺失返回 `None`。
fn between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let s = text.find(start)?;
    let rest = &text[s + start.len()..];
    let e = rest.find(end)?;
    Some(rest[..e].trim())
}

/// 自省/验证评审编排器(P2-8)。
///
/// 组合模式:worker 产出输出 → 评审 Agent(`reviewer`)检验 → 不达标就把评审
/// 反馈拼进任务目标重做,直到达标或尝试耗尽。默认尝试耗尽仍未达标返回
/// [`AgentError`](宁可失败也不把未过审的产出当结果);[`Self::keep_last_output`]
/// 可改为返回最近一次产出(对应 DeepResearch "轮数用完即收"的语义)。
///
/// worker / reviewer 都是 [`Orchestrator`],本组合器自身也实现 [`Orchestrator`],
/// 因此可再嵌进 [`FanOutFanIn`] / [`SequentialPipeline`](评审委员会、流水线里
/// 校验某阶段等)。
pub struct ReviewOrchestrator {
    worker: Arc<dyn Orchestrator<Input = AgentTask, Output = String>>,
    reviewer: Arc<dyn Orchestrator<Input = String, Output = String>>,
    max_attempts: usize,
    fail_on_unresolved: bool,
}

impl ReviewOrchestrator {
    /// 构造评审编排器。
    ///
    /// # Arguments
    /// * `worker` — 产出方(接受 [`AgentTask`] 派发,输出待评审文本)。
    /// * `reviewer` — 评审方(输入 [`review_envelope`] 信封,输出可被
    ///   [`parse_review_verdict`] 解析的结论)。
    /// * `max_attempts` — 最多"产出 + 评审"轮数(至少 1)。
    pub fn new(
        worker: Arc<dyn Orchestrator<Input = AgentTask, Output = String>>,
        reviewer: Arc<dyn Orchestrator<Input = String, Output = String>>,
        max_attempts: usize,
    ) -> Self {
        Self {
            worker,
            reviewer,
            max_attempts: max_attempts.max(1),
            fail_on_unresolved: true,
        }
    }

    /// 调整最大轮数(至少 1)。
    pub fn with_max_attempts(mut self, max_attempts: usize) -> Self {
        self.max_attempts = max_attempts.max(1);
        self
    }

    /// 尝试耗尽仍未达标时返回最近一次产出,而不是报错。
    pub fn keep_last_output(mut self) -> Self {
        self.fail_on_unresolved = false;
        self
    }

    /// 最大轮数。
    pub fn max_attempts(&self) -> usize {
        self.max_attempts
    }
}

#[async_trait]
impl Orchestrator for ReviewOrchestrator {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        let mut task = input;
        let mut last_output = String::new();
        let mut last_feedback = String::new();

        for attempt in 0..self.max_attempts {
            log::debug!(
                target: "lc_agents::orchestrator",
                "ReviewOrchestrator attempt {}/{} trace_id = {}",
                attempt + 1,
                self.max_attempts,
                ctx.trace_id
            );

            last_output = self
                .worker
                .run_with_context(task.clone(), ctx)
                .await
                .map_err(|e| {
                    AgentError::Other(format!(
                        "ReviewOrchestrator worker (attempt {}): {e}",
                        attempt + 1
                    ))
                })?;

            let review_text = self
                .reviewer
                .run_with_context(review_envelope(&task, &last_output), ctx)
                .await
                .map_err(|e| {
                    AgentError::Other(format!(
                        "ReviewOrchestrator reviewer (attempt {}): {e}",
                        attempt + 1
                    ))
                })?;

            let verdict = parse_review_verdict(&review_text).ok_or_else(|| {
                AgentError::Other(format!(
                    "ReviewOrchestrator: 无法解析评审结论: {review_text}"
                ))
            })?;

            if verdict.passed {
                log::debug!(
                    target: "lc_agents::orchestrator",
                    "ReviewOrchestrator passed on attempt {}",
                    attempt + 1
                );
                return Ok(last_output);
            }

            last_feedback = verdict.feedback.clone();
            if attempt + 1 >= self.max_attempts {
                log::warn!(
                    target: "lc_agents::orchestrator",
                    "ReviewOrchestrator unresolved after {} attempts",
                    self.max_attempts
                );
                break;
            }

            // 带着评审反馈重做:目标追加修订指令,任务级约束沿链保留。
            let feedback_suffix = if verdict.feedback.trim().is_empty() {
                "[评审未通过,请修订输出质量]".to_string()
            } else {
                format!("[评审未通过,请根据反馈修订: {}]", verdict.feedback.trim())
            };
            let mut next = AgentTask::new(format!("{}\n{}", task.objective, feedback_suffix));
            if let Some(expected) = task.expected_output.clone() {
                next = next.with_expected_output(expected);
            }
            next = next.with_allowed_tools(task.allowed_tools.clone());
            task = next;
        }

        if self.fail_on_unresolved {
            let detail = if last_feedback.trim().is_empty() {
                "(无)".to_string()
            } else {
                last_feedback.trim().to_string()
            };
            Err(AgentError::Other(format!(
                "ReviewOrchestrator: 重做 {} 次后仍未达标, 最近反馈: {}",
                self.max_attempts, detail
            )))
        } else {
            Ok(last_output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
            self.seen.lock().unwrap().push(task.clone());
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
        let orch = FanOutFanIn::new(vec![mock_orch("a"), mock_orch("b")])
            .with_aggregator(|vs| vs.join(" + "));
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
        let seen = seen.lock().unwrap();
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
        let seen = seen.lock().unwrap();
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
            self.calls.lock().unwrap().push(task.clone());
            if self.first_try_good || task.objective.contains("修订") {
                Ok("good answer".to_string())
            } else {
                Ok("bad answer".to_string())
            }
        }
    }

    /// mock worker 工厂:返回 worker trait 对象 + 记录的任务列表。
    fn review_worker(
        first_try_good: bool,
    ) -> (
        Arc<dyn Orchestrator<Input = AgentTask, Output = String>>,
        Arc<Mutex<Vec<AgentTask>>>,
    ) {
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
        assert_eq!(calls.lock().unwrap().len(), 1, "达标后不应重做");
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
        let calls = calls.lock().unwrap();
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
        assert!(err.to_string().contains("未达标"), "{}", err);
    }

    /// P2-8: `keep_last_output()` 使耗尽后返回最近产出而非报错。
    #[tokio::test]
    async fn test_review_keep_last_output_on_exhaustion() {
        let (worker, _) = review_worker(true);
        let orch =
            ReviewOrchestrator::new(worker, Arc::new(AlwaysFailReview), 2).keep_last_output();
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
}
