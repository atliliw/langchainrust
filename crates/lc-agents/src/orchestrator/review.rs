//! Introspection/validation review orchestrator `ReviewOrchestrator` (P2-8).
//!
//! Generalizes DeepResearch's gap-check idea: worker produces → reviewer Agent
//! inspects → if not passing, redo with review feedback until passing or out of
//! attempts. The reviewer Agent is itself an [`Orchestrator`]: its input is a
//! JSON envelope (objective + expected output + produced output, see
//! [`review_envelope`]), and its output is a review verdict (format per
//! [`parse_review_verdict`]).

use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use super::{Orchestrator, RunContext};
use crate::task::AgentTask;
use crate::AgentError;

/// Review verdict: whether it passes + revision feedback when it does not.
#[derive(Debug, Clone, PartialEq)]
pub struct ReviewVerdict {
    /// Whether the output passes.
    pub passed: bool,
    /// Revision feedback given to the worker when it does not pass (empty when it passes).
    pub feedback: String,
}

impl ReviewVerdict {
    /// Passing verdict.
    pub fn pass() -> Self {
        Self {
            passed: true,
            feedback: String::new(),
        }
    }

    /// Failing verdict, carrying revision feedback.
    pub fn fail(feedback: impl Into<String>) -> Self {
        Self {
            passed: false,
            feedback: feedback.into(),
        }
    }
}

/// Builds the reviewer Agent's input envelope: packs the task (objective /
/// expected output) together with the produced output into JSON, which the
/// reviewer Agent uses to judge whether the output passes.
pub fn review_envelope(task: &AgentTask, output: &str) -> String {
    serde_json::json!({
        "objective": task.objective,
        "expected_output": task.expected_output,
        "output": output,
    })
    .to_string()
}

/// Parses the reviewer Agent's conclusion text into a [`ReviewVerdict`].
///
/// LLM output varies, so this falls back through three formats:
/// 1. JSON: `{"passed": true}` or `{"passed": false, "feedback": "..."}`;
/// 2. Delimiters: `<<<VERDICT>>>PASS|FAIL<<<END_VERDICT>>>`, with optional
///    feedback wrapped in `<<<FEEDBACK>>>...<<<END_FEEDBACK>>>`;
/// 3. Plain text: starting with `PASS` / `FAIL` (case-insensitive), with
///    feedback text following `FAIL`.
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

    // 2. Delimiters
    if let Some(inner) = between(text, "<<<VERDICT>>>", "<<<END_VERDICT>>>") {
        if inner.eq_ignore_ascii_case("PASS") || inner.eq_ignore_ascii_case("FAIL") {
            let passed = inner.eq_ignore_ascii_case("PASS");
            let feedback = between(text, "<<<FEEDBACK>>>", "<<<END_FEEDBACK>>>")
                .unwrap_or("")
                .to_string();
            return Some(ReviewVerdict { passed, feedback });
        }
    }

    // 3. Plain text
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

/// Returns the text between two markers (trimmed), or `None` if a marker is missing.
fn between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let s = text.find(start)?;
    let rest = &text[s + start.len()..];
    let e = rest.find(end)?;
    Some(rest[..e].trim())
}

/// Introspection/validation review orchestrator (P2-8).
///
/// Composition pattern: worker produces output → reviewer Agent (`reviewer`)
/// inspects → if not passing, the review feedback is folded into the task
/// objective and redone, until passing or out of attempts. By default, when
/// attempts are exhausted without passing it returns an `AgentError` (better to
/// fail than to return an unapproved output as the result);
/// [`Self::keep_last_output`] instead returns the latest output (matching
/// DeepResearch's "collect when rounds run out" semantics).
///
/// Both worker and reviewer are [`Orchestrator`]s, and this compositor itself
/// also implements [`Orchestrator`], so it can be nested into `FanOutFanIn` /
/// `SequentialPipeline` (review panels, validating a stage in a pipeline, etc.).
pub struct ReviewOrchestrator {
    worker: Arc<dyn Orchestrator<Input = AgentTask, Output = String>>,
    reviewer: Arc<dyn Orchestrator<Input = String, Output = String>>,
    max_attempts: usize,
    fail_on_unresolved: bool,
}

impl ReviewOrchestrator {
    /// Builds the review orchestrator.
    ///
    /// # Arguments
    /// * `worker` — the producer (accepts an [`AgentTask`] dispatch, outputs
    ///   text to review).
    /// * `reviewer` — the reviewer (takes the [`review_envelope`] envelope,
    ///   outputs a conclusion parseable by [`parse_review_verdict`]).
    /// * `max_attempts` — maximum number of "produce + review" rounds (at least 1).
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

    /// Adjusts the maximum number of rounds (at least 1).
    pub fn with_max_attempts(mut self, max_attempts: usize) -> Self {
        self.max_attempts = max_attempts.max(1);
        self
    }

    /// Returns the latest output when attempts are exhausted without passing, instead of erroring.
    pub fn keep_last_output(mut self) -> Self {
        self.fail_on_unresolved = false;
        self
    }

    /// The maximum number of rounds.
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

            // Redo with review feedback: append the revision directive to the objective, keeping task-level constraints along the chain.
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
