//! Integration tests for the approval gate + budget gates (§4.2): Allow / Deny /
//! Modify / resume (Deny→Allow) / the four budgets (max_tool_calls / max_tokens /
//! max_duration / max_iterations) / default-off behavior unchanged.
//!
//! Drives a real `AgentExecutor` decision loop through the public API; tool
//! calls share counters via `Arc`, asserting "did it really execute / with which
//! arguments", without any network.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lc_agents::types::{AgentAction, AgentFinish, AgentOutput, AgentStep, ToolInput};
use lc_agents::{
    AgentError, AgentExecutor, AgentStreamEvent, AllowAll, ApprovalDecision, ApprovalHandler,
    BaseAgent, BudgetConfig, BudgetExceeded,
};
use lc_core::language_models::TokenUsage;
use lc_core::tools::{BaseTool, ToolError};

/// Records call counts and arguments: lets tests assert "it really executed / with which arguments".
struct RecordingTool {
    calls: Arc<AtomicUsize>,
    inputs: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl BaseTool for RecordingTool {
    fn name(&self) -> &str {
        "recorder"
    }
    fn description(&self) -> &str {
        "records invocations for tests"
    }
    async fn run(&self, input: String) -> Result<String, ToolError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inputs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(input.clone());
        Ok(format!("echo: {input}"))
    }
}

/// Agent that plans one `recorder` call then finishes (standard plan→act→observe shape).
struct ActOnceAgent;

#[async_trait]
impl BaseAgent for ActOnceAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        if intermediate_steps.is_empty() {
            return Ok(AgentOutput::Action(AgentAction {
                tool: "recorder".to_string(),
                tool_input: ToolInput::Object {
                    value: serde_json::json!({"x": 1}),
                },
                log: "call_1".to_string(),
            }));
        }
        Ok(AgentOutput::Finish(AgentFinish::new(
            "done".to_string(),
            String::new(),
        )))
    }
}

/// Agent that plans `recorder` calls forever (for budget tests: never finishes on its own).
struct LoopAgent;

#[async_trait]
impl BaseAgent for LoopAgent {
    async fn plan(
        &self,
        _intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        Ok(AgentOutput::Action(AgentAction {
            tool: "recorder".to_string(),
            tool_input: ToolInput::Object {
                value: serde_json::json!({"x": 1}),
            },
            log: "loop".to_string(),
        }))
    }
}

/// Agent reporting a fixed token usage on every plan (for the max_tokens budget test).
struct TokenLoopAgent;

#[async_trait]
impl BaseAgent for TokenLoopAgent {
    async fn plan(
        &self,
        _intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        Ok(AgentOutput::Action(AgentAction {
            tool: "recorder".to_string(),
            tool_input: ToolInput::Object {
                value: serde_json::json!({"x": 1}),
            },
            log: "loop".to_string(),
        }))
    }
    fn last_token_usage(&self) -> Option<TokenUsage> {
        Some(TokenUsage {
            prompt_tokens: 4,
            completion_tokens: 2,
            total_tokens: 6,
        })
    }
}

/// Agent that keeps trying the same tool after a Deny (for resume-semantics tests: retries when it sees a DENIED observation).
struct ResumeAgent;

#[async_trait]
impl BaseAgent for ResumeAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        if intermediate_steps.is_empty() {
            return Ok(AgentOutput::Action(AgentAction {
                tool: "recorder".to_string(),
                tool_input: ToolInput::Object {
                    value: serde_json::json!({"x": 1}),
                },
                log: "call_1".to_string(),
            }));
        }
        if intermediate_steps
            .last()
            .is_some_and(|s| s.observation.contains("DENIED"))
        {
            return Ok(AgentOutput::Action(AgentAction {
                tool: "recorder".to_string(),
                tool_input: ToolInput::Object {
                    value: serde_json::json!({"x": 1}),
                },
                log: "call_2".to_string(),
            }));
        }
        Ok(AgentOutput::Finish(AgentFinish::new(
            "done".to_string(),
            String::new(),
        )))
    }
}

/// Denies everything.
struct DenyAll {
    reason: String,
}

#[async_trait]
impl ApprovalHandler for DenyAll {
    async fn approve(&self, _ctx: &lc_agents::hooks::ToolCallContext) -> ApprovalDecision {
        ApprovalDecision::Deny {
            reason: self.reason.clone(),
        }
    }
}

/// Rewrites to fixed arguments.
struct ModifyAll {
    arguments: serde_json::Value,
    note: String,
}

#[async_trait]
impl ApprovalHandler for ModifyAll {
    async fn approve(&self, _ctx: &lc_agents::hooks::ToolCallContext) -> ApprovalDecision {
        ApprovalDecision::Modify {
            arguments: self.arguments.clone(),
            note: self.note.clone(),
        }
    }
}

/// Denies the first round, allows subsequent ones (simulating "suspended waiting for an approval signal → the signal arrives and it resumes").
struct DenyOnceThenAllow {
    count: AtomicUsize,
}

#[async_trait]
impl ApprovalHandler for DenyOnceThenAllow {
    async fn approve(&self, _ctx: &lc_agents::hooks::ToolCallContext) -> ApprovalDecision {
        if self.count.fetch_add(1, Ordering::SeqCst) == 0 {
            ApprovalDecision::Deny {
                reason: "hold for review".to_string(),
            }
        } else {
            ApprovalDecision::Allow
        }
    }
}

fn recorder_harness(
    agent: Arc<dyn BaseAgent>,
) -> (AgentExecutor, Arc<AtomicUsize>, Arc<Mutex<Vec<String>>>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let inputs = Arc::new(Mutex::new(Vec::new()));
    let tool = RecordingTool {
        calls: calls.clone(),
        inputs: inputs.clone(),
    };
    let executor = AgentExecutor::new(agent, vec![Arc::new(tool)]);
    (executor, calls, inputs)
}

// ── Approval gate ────────────────────────────────────────────────────────

/// Allow: the tool executes as-is and the agent finishes normally.
#[tokio::test]
async fn approval_allow_runs_tool() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(ActOnceAgent));
    let executor = executor.with_approval(Arc::new(AllowAll));
    let answer = executor.invoke("go".to_string()).await.unwrap();
    assert_eq!(answer, "done");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Deny: the tool does not execute; the reason feeds back into the loop as an observation, and the agent finishes next round.
#[tokio::test]
async fn approval_deny_skips_tool() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(ActOnceAgent));
    let executor = executor.with_approval(Arc::new(DenyAll {
        reason: "manual review required".to_string(),
    }));
    let answer = executor.invoke("go".to_string()).await.unwrap();
    assert_eq!(answer, "done");
    assert_eq!(calls.load(Ordering::SeqCst), 0, "被拒工具不得执行");
}

/// Modify: executes with the post-approval arguments.
#[tokio::test]
async fn approval_modify_rewrites_arguments() {
    let (executor, calls, inputs) = recorder_harness(Arc::new(ActOnceAgent));
    let executor = executor.with_approval(Arc::new(ModifyAll {
        arguments: serde_json::json!({"x": 42}),
        note: "sanitized by reviewer".to_string(),
    }));
    let answer = executor.invoke("go".to_string()).await.unwrap();
    assert_eq!(answer, "done");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let inputs = inputs.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(inputs.len(), 1);
    assert!(
        inputs[0].contains("42"),
        "工具应收到修改后的参数,实际: {}",
        inputs[0]
    );
}

/// resume: first round Deny suspends, then Allow arrives, the tool finally executes, and the loop stays continuous.
#[tokio::test]
async fn approval_resume_after_deny_then_allow() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(ResumeAgent));
    let executor = executor.with_approval(Arc::new(DenyOnceThenAllow {
        count: AtomicUsize::new(0),
    }));
    let answer = executor.invoke("go".to_string()).await.unwrap();
    assert_eq!(answer, "done");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "Deny 后 Allow 应恰好执行一次"
    );
}

// ── Budget gates ──────────────────────────────────────────────────────────

/// max_tool_calls: exactly `limit` executions allowed; the limit+1-th hard-stops with an error.
#[tokio::test]
async fn budget_max_tool_calls_stops() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(LoopAgent));
    let executor = executor.with_budget(BudgetConfig {
        max_tool_calls: Some(2),
        ..Default::default()
    });
    let err = executor.invoke("go".to_string()).await.unwrap_err();
    assert!(
        matches!(
            err,
            AgentError::BudgetExceeded(BudgetExceeded::ToolCalls {
                limit: 2,
                actual: 3
            })
        ),
        "got: {err:?}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2, "应恰好执行 2 次后停");
}

/// max_tokens: 6 tokens accumulate per plan; cap 10 → hard-stops after the 2nd plan.
#[tokio::test]
async fn budget_max_tokens_stops() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(TokenLoopAgent));
    let executor = executor.with_budget(BudgetConfig {
        max_tokens: Some(10),
        ..Default::default()
    });
    let err = executor.invoke("go".to_string()).await.unwrap_err();
    assert!(
        matches!(
            err,
            AgentError::BudgetExceeded(BudgetExceeded::Tokens {
                limit: 10,
                actual: 12
            })
        ),
        "got: {err:?}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// max_duration: over the limit the moment the clock starts (0 duration) → hard-stops at the first iteration.
#[tokio::test]
async fn budget_max_duration_stops() {
    let (executor, _calls, _inputs) = recorder_harness(Arc::new(LoopAgent));
    let executor = executor.with_budget(BudgetConfig {
        max_duration: Some(Duration::ZERO),
        ..Default::default()
    });
    let err = executor.invoke("go".to_string()).await.unwrap_err();
    assert!(
        matches!(
            err,
            AgentError::BudgetExceeded(BudgetExceeded::Duration { .. })
        ),
        "got: {err:?}"
    );
}

/// max_iterations: tightens the default iteration cap; exceeding it returns an error rather than a placeholder string.
#[tokio::test]
async fn budget_max_iterations_stops() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(LoopAgent));
    let executor = executor.with_budget(BudgetConfig {
        max_iterations: Some(1),
        ..Default::default()
    });
    let err = executor.invoke("go".to_string()).await.unwrap_err();
    assert!(
        matches!(
            err,
            AgentError::BudgetExceeded(BudgetExceeded::Iterations { limit: 1 })
        ),
        "got: {err:?}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "第 1 次迭代执行,第 2 次迭代被预算截停"
    );
}

/// Default-off: with no gates attached, behavior matches existing behavior.
#[tokio::test]
async fn defaults_off_behavior_unchanged() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(ActOnceAgent));
    let answer = executor.invoke("go".to_string()).await.unwrap();
    assert_eq!(answer, "done");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

// ── Streaming budget gates ────────────────────────────────────────────────
// Before the 0.18.2 fix, `stream()`'s closure capture list lacked `budget`, so
// the four budget gates were only checked on the invoke path — streaming budget
// protection was completely inert (the 0.18.0 Added section's claim was false).
// The tests below lock the streaming path to the same semantics as invoke:
// over-limit terminates by sending `Err(BudgetExceeded)` through the channel.

/// Drains the whole stream, keeping the `Err(BudgetExceeded)` terminal event (no unwrap panic).
async fn stream_events(
    executor: &AgentExecutor,
    input: &str,
) -> Vec<Result<AgentStreamEvent, AgentError>> {
    use futures_util::StreamExt;
    let mut stream = executor.stream(input.to_string());
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

/// Streaming max_tool_calls: exactly `limit` executions allowed; the limit+1-th terminates with Err via the channel.
#[tokio::test]
async fn stream_budget_tool_calls() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(LoopAgent));
    let executor = executor.with_budget(BudgetConfig {
        max_tool_calls: Some(2),
        ..Default::default()
    });
    let events = stream_events(&executor, "go").await;

    let tool_ends = events
        .iter()
        .filter(|e| matches!(e, Ok(AgentStreamEvent::ToolEnd { .. })))
        .count();
    assert_eq!(tool_ends, 2, "流式应恰好执行 2 次工具后停");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        matches!(
            events.last(),
            Some(Err(AgentError::BudgetExceeded(BudgetExceeded::ToolCalls {
                limit: 2,
                actual: 3
            })))
        ),
        "流式最后一个事件应为 Err(BudgetExceeded::ToolCalls),got: {:?}",
        events.last()
    );
}

/// Streaming max_iterations: tightens the default iteration cap; exceeding it terminates with Err instead of a placeholder FinalAnswer.
#[tokio::test]
async fn stream_budget_iterations() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(LoopAgent));
    let executor = executor.with_max_iterations(10).with_budget(BudgetConfig {
        max_iterations: Some(1),
        ..Default::default()
    });
    let events = stream_events(&executor, "go").await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "第 1 次迭代执行,第 2 次迭代被预算截停"
    );
    assert!(
        matches!(
            events.last(),
            Some(Err(AgentError::BudgetExceeded(
                BudgetExceeded::Iterations { limit: 1 }
            )))
        ),
        "应为 Err(BudgetExceeded::Iterations) 终止而非占位 FinalAnswer,got: {:?}",
        events.last()
    );
}

/// Streaming max_tokens: 6 tokens accumulate per plan; cap 10 → Err after the 2nd plan.
#[tokio::test]
async fn stream_budget_tokens() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(TokenLoopAgent));
    let executor = executor.with_budget(BudgetConfig {
        max_tokens: Some(10),
        ..Default::default()
    });
    let events = stream_events(&executor, "go").await;

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        matches!(
            events.last(),
            Some(Err(AgentError::BudgetExceeded(BudgetExceeded::Tokens {
                limit: 10,
                actual: 12
            })))
        ),
        "got: {:?}",
        events.last()
    );
}

/// Streaming max_duration: over the limit the moment the clock starts (0 duration) → immediate Err at the top of the loop, zero tool executions.
#[tokio::test]
async fn stream_budget_duration() {
    let (executor, _calls, _inputs) = recorder_harness(Arc::new(LoopAgent));
    let executor = executor.with_budget(BudgetConfig {
        max_duration: Some(Duration::ZERO),
        ..Default::default()
    });
    let events = stream_events(&executor, "go").await;

    assert_eq!(events.len(), 1, "首个事件即 Err(Duration),got: {events:?}");
    assert!(
        matches!(
            &events[0],
            Err(AgentError::BudgetExceeded(BudgetExceeded::Duration { .. }))
        ),
        "got: {:?}",
        events[0]
    );
}

/// Streaming metrics publishing: after the stream drains, `last_metrics()` is Some (publish happens before the terminal event, no race).
#[tokio::test]
async fn stream_publishes_metrics() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(ActOnceAgent));
    let events = stream_events(&executor, "go").await;

    assert!(
        matches!(
            events.last(),
            Some(Ok(AgentStreamEvent::FinalAnswer { .. }))
        ),
        "工具调用 + Finish 应正常收尾,got: {:?}",
        events.last()
    );

    let m = executor
        .last_metrics()
        .expect("stream 收完后 metrics 应已发布");
    assert_eq!(m.tool_calls, 1);
    assert_eq!(m.llm_calls, 2);
    assert!(m.duration > Duration::ZERO);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
