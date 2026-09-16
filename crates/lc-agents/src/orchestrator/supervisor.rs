//! N1 (v0.24.0): Supervisor dynamic-routing orchestrator — "agent as tool",
//! one-level sub-agent recursion.
//!
//! [`SequentialPipeline`](super::SequentialPipeline) runs a *fixed* stage list and
//! [`FanOutFanIn`](super::FanOutFanIn) *broadcasts* the same task to every worker;
//! the [`Supervisor`] instead decides **at runtime** which named worker (a
//! sub-agent) to hand the current sub-task to — or whether to finish — by asking a
//! supervisor model once per round. Each worker runs a complete, independent
//! lifecycle behind the [`Orchestrator`] trait (its own executor / budget / cost /
//! interrupt / checkpoint): the compositor only builds the [`AgentTask`] and
//! collects the output. That output is fed **back** into the supervisor's
//! scratchpad, so the routing decision on the next round can depend on what prior
//! workers produced.
//!
//! This is exactly one level of sub-agent recursion: workers are leaf
//! orchestrators. The routing decision the supervisor model returns is parsed by
//! [`parse_supervisor_decision`] (JSON first, delimiters as a fallback, mirroring
//! [`super::parse_review_verdict`]).

use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

use super::{Orchestrator, RunContext};
use crate::task::AgentTask;
use crate::AgentError;

/// Reserved worker name meaning "delegation is done; return the answer".
/// A real worker must not be registered under this name (case-insensitive).
pub const SUPERVISOR_FINISH: &str = "FINISH";

/// One routing decision the supervisor reaches each round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupervisorNext {
    /// Delegate a subtask to the named worker (sub-agent), then route again with
    /// the worker's output fed back into the scratchpad.
    Work {
        /// Worker to delegate to (must be one of the registered names).
        worker: String,
        /// The subtask objective handed to that worker.
        task: String,
    },
    /// Stop delegating and return `answer` to the caller.
    Finish {
        /// The final answer.
        answer: String,
    },
}

/// Returns the text between two markers (trimmed), or `None` if a marker is missing.
fn between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let s = text.find(start)?;
    let rest = &text[s + start.len()..];
    let e = rest.find(end)?;
    Some(rest[..e].trim())
}

fn json_str<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

/// Classifies a parsed `next` token plus its task/answer strings.
fn classify(next: &str, task: &str, answer: &str) -> SupervisorNext {
    if next.eq_ignore_ascii_case(SUPERVISOR_FINISH) {
        SupervisorNext::Finish {
            answer: answer.to_string(),
        }
    } else {
        SupervisorNext::Work {
            worker: next.to_string(),
            task: task.to_string(),
        }
    }
}

/// Parses the supervisor model's conclusion into a [`SupervisorNext`].
///
/// Accepted formats (the model output is trimmed first):
/// 1. JSON delegate: `{"next": "<worker>", "task": "<subtask>"}`.
/// 2. JSON finish: `{"next": "FINISH", "answer": "<final answer>"}`.
/// 3. Delimited: `<<<NEXT>>>name<<<END_NEXT>>>` together with optional
///    `<<<TASK>>>...<<<END_TASK>>>` (delegation) or
///    `<<<ANSWER>>>...<<<END_ANSWER>>>` (finish).
///
/// Returns `None` when no recognizable decision is present.
pub fn parse_supervisor_decision(text: &str) -> Option<SupervisorNext> {
    let text = text.trim();

    // 1. JSON
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        if let Some(next) = value.get("next").and_then(Value::as_str) {
            return Some(classify(
                next.trim(),
                json_str(&value, "task"),
                json_str(&value, "answer"),
            ));
        }
    }

    // 2. Delimiters
    let next = between(text, "<<<NEXT>>>", "<<<END_NEXT>>>")?;
    let task = between(text, "<<<TASK>>>", "<<<END_TASK>>>").unwrap_or("");
    let answer = between(text, "<<<ANSWER>>>", "<<<END_ANSWER>>>").unwrap_or(task);
    Some(classify(next, task, answer))
}

/// Builds the supervisor model's input envelope: the original objective, the
/// workers it may delegate to, the current round, and every worker result so far.
/// The history is what makes routing *dynamic* — the model chooses the next worker
/// from what earlier workers returned rather than from a fixed schedule.
pub fn supervisor_envelope(
    objective: &str,
    worker_names: &[String],
    round: usize,
    history: &[(String, String)],
) -> String {
    let results: Vec<Value> = history
        .iter()
        .map(|(worker, output)| json!({ "worker": worker, "output": output }))
        .collect();
    json!({
        "objective": objective,
        "workers": worker_names,
        "round": round,
        "results": results,
        "instruction": format!(
            "Delegate to one worker via {{\"next\":\"<worker>\",\"task\":\"<subtask>\"}}, \
             or finish via {{\"next\":\"{SUPERVISOR_FINISH}\",\"answer\":\"<final answer>\"}}."
        ),
    })
    .to_string()
}

type Worker = Arc<dyn Orchestrator<Input = AgentTask, Output = String>>;

/// Supervisor dynamic-routing orchestrator (N1).
///
/// Each round it asks `supervisor_llm` (a `String -> String` [`Orchestrator`],
/// typically an LLM with a routing prompt) where to send the current sub-task,
/// parses the decision, and either returns the final answer or invokes the chosen
/// worker as a sub-agent. Worker outputs accumulate in the scratchpad and are
/// re-presented to the model on the next round. Delegation is bounded by
/// `max_rounds`, which is the one-level recursion guard: the run errors rather
/// than looping forever if the model never reaches [`SUPERVISOR_FINISH`].
pub struct Supervisor {
    workers: Vec<(String, Worker)>,
    supervisor_llm: Arc<dyn Orchestrator<Input = String, Output = String>>,
    max_rounds: usize,
}

impl Supervisor {
    /// Build a supervisor.
    ///
    /// # Arguments
    /// * `supervisor_llm` — the router (takes [`supervisor_envelope`], returns a
    ///   decision parseable by [`parse_supervisor_decision`]).
    /// * `workers` — `(name, sub-agent)` pairs in delegation/listing order; names
    ///   must be unique and must not be [`SUPERVISOR_FINISH`].
    /// * `max_rounds` — maximum delegation rounds (at least 1).
    pub fn new(
        supervisor_llm: Arc<dyn Orchestrator<Input = String, Output = String>>,
        workers: Vec<(String, Worker)>,
        max_rounds: usize,
    ) -> Self {
        Self {
            workers,
            supervisor_llm,
            max_rounds: max_rounds.max(1),
        }
    }

    /// Adjust the maximum delegation rounds (at least 1).
    pub fn with_max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = max_rounds.max(1);
        self
    }

    /// The maximum delegation rounds.
    pub fn max_rounds(&self) -> usize {
        self.max_rounds
    }

    /// Registered worker names in registration order.
    pub fn worker_names(&self) -> Vec<&str> {
        self.workers.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Registered worker names as owned strings (envelope/error friendly).
    fn names(&self) -> Vec<String> {
        self.workers.iter().map(|(n, _)| n.clone()).collect()
    }
}

#[async_trait]
impl Orchestrator for Supervisor {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        if self.workers.is_empty() {
            return Err(AgentError::Other(
                "Supervisor requires at least one worker".to_string(),
            ));
        }

        let names = self.names();
        let objective = input.objective.clone();
        // (worker, output) pairs accumulated in delegation order — the feedback
        // scratchpad the next routing decision is based on.
        let mut history: Vec<(String, String)> = Vec::new();

        for round in 0..self.max_rounds {
            log::debug!(
                target: "lc_agents::orchestrator",
                "Supervisor round {}/{} workers={} trace_id={}",
                round + 1,
                self.max_rounds,
                names.len(),
                ctx.trace_id
            );

            let envelope = supervisor_envelope(&objective, &names, round, &history);
            let decision_text = self
                .supervisor_llm
                .run_with_context(envelope, ctx)
                .await
                .map_err(|e| {
                    AgentError::Other(format!("Supervisor router (round {round}): {e}"))
                })?;
            let decision = parse_supervisor_decision(&decision_text).ok_or_else(|| {
                AgentError::Other(format!(
                    "Supervisor: unparseable routing decision (round {round}): {decision_text}"
                ))
            })?;

            match decision {
                SupervisorNext::Finish { answer } => {
                    log::debug!(
                        target: "lc_agents::orchestrator",
                        "Supervisor finished on round {}",
                        round + 1
                    );
                    return Ok(answer);
                }
                SupervisorNext::Work { worker, task } => {
                    let (_, worker_orch) = self
                        .workers
                        .iter()
                        .find(|(name, _)| name == &worker)
                        .ok_or_else(|| {
                            AgentError::Other(format!(
                                "Supervisor: unknown worker '{worker}' (round {round}); available: {names:?}"
                            ))
                        })?;

                    // Each delegation is a fresh sub-agent task; the parent task's
                    // constraints (expected output / tool allowlist) propagate so
                    // the sub-agent stays inside the same contract.
                    let mut sub_task = AgentTask::new(task);
                    if let Some(expected) = &input.expected_output {
                        sub_task = sub_task.with_expected_output(expected.clone());
                    }
                    sub_task = sub_task.with_allowed_tools(input.allowed_tools.clone());

                    // The worker runs its own full lifecycle here; only its text
                    // output is folded back into the supervisor scratchpad.
                    let output =
                        worker_orch
                            .run_with_context(sub_task, ctx)
                            .await
                            .map_err(|e| {
                                AgentError::Other(format!(
                                    "Supervisor worker '{worker}' (round {round}): {e}"
                                ))
                            })?;
                    history.push((worker, output));
                }
            }
        }

        Err(AgentError::Other(format!(
            "Supervisor: did not reach {SUPERVISOR_FINISH} within {} delegation round(s)",
            self.max_rounds
        )))
    }
}
