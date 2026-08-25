//! 自省/验证评审编排器 `ReviewOrchestrator`(P2-8)。
//!
//! 借鉴 DeepResearch 的 gap 检查思路泛化:worker 产出 → 评审 Agent 检验 →
//! 不达标则带着评审反馈重做,直到达标或尝试耗尽。评审 Agent 本身也是一个
//! [`Orchestrator`]:输入为 JSON 信封(目标 + 预期输出 + 产出,见
//! [`review_envelope`]),输出为评审结论(格式见 [`parse_review_verdict`])。

use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use super::{Orchestrator, RunContext};
use crate::task::AgentTask;
use crate::AgentError;

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
/// 因此可再嵌进 `FanOutFanIn` / `SequentialPipeline`(评审委员会、流水线里
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
                    "ReviewOrchestrator: failed to parse review verdict: {review_text}"
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
                "(none)".to_string()
            } else {
                last_feedback.trim().to_string()
            };
            Err(AgentError::Other(format!(
                "ReviewOrchestrator: did not pass after {} attempts, latest feedback: {}",
                self.max_attempts, detail
            )))
        } else {
            Ok(last_output)
        }
    }
}
