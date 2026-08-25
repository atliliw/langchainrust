//! 人审门 + 预算门(§4.2)的集成测试:Allow / Deny / Modify / resume(Deny→Allow) /
//! 三预算(max_tool_calls / max_tokens / max_duration / max_iterations) /
//! 默认关行为不变。
//!
//! 通过公开 API 驱动真实 `AgentExecutor` 决策循环;工具调用用 `Arc` 共享计数,
//! 断言"是否真的执行 / 用哪个参数执行",不依赖任何网络。

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lc_agents::types::{AgentAction, AgentFinish, AgentOutput, AgentStep, ToolInput};
use lc_agents::{
    AgentError, AgentExecutor, AllowAll, ApprovalDecision, ApprovalHandler, BaseAgent,
    BudgetConfig, BudgetExceeded,
};
use lc_core::language_models::TokenUsage;
use lc_core::tools::{BaseTool, ToolError};

/// 记录调用次数与入参的工具:让测试断言"真的执行了 / 用哪个参数执行"。
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

/// 计划一次 `recorder` 调用后收尾的 agent(标准 plan→act→observe 形状)。
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

/// 无限计划 `recorder` 调用的 agent(预算测试用:永不主动收尾)。
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

/// 每次 plan 上报固定 token 用量的 agent(max_tokens 预算测试用)。
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

/// Deny 后继续尝试同一工具的 agent(resume 语义测试用:看见 DENIED 观察就重试)。
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

/// 拒绝一切。
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

/// 改成固定参数。
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

/// 首轮 Deny、后续 Allow(模拟"挂起等待审批信号 → 信号到续跑")。
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

// ── 人审门 ──────────────────────────────────────────────────────────────

/// Allow:工具原样执行,agent 正常收尾。
#[tokio::test]
async fn approval_allow_runs_tool() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(ActOnceAgent));
    let executor = executor.with_approval(Arc::new(AllowAll));
    let answer = executor.invoke("go".to_string()).await.unwrap();
    assert_eq!(answer, "done");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Deny:工具不执行,理由作为 observation 喂回循环,agent 下一轮收尾。
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

/// Modify:用审批后的参数执行。
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

/// resume:首轮 Deny 挂起、信号到后 Allow,工具最终执行,循环连续。
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

// ── 预算门 ──────────────────────────────────────────────────────────────

/// max_tool_calls:允许恰好 limit 次执行,第 limit+1 次硬停并报错。
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

/// max_tokens:每次 plan 累计 6 token,上限 10 → 第 2 次 plan 后硬停。
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

/// max_duration:起表即超限(0 时长) → 首个迭代硬停。
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

/// max_iterations:收紧默认迭代上限,超限返回错误而非占位串。
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

/// 默认关:不带任何闸,行为与存量一致。
#[tokio::test]
async fn defaults_off_behavior_unchanged() {
    let (executor, calls, _inputs) = recorder_harness(Arc::new(ActOnceAgent));
    let answer = executor.invoke("go".to_string()).await.unwrap();
    assert_eq!(answer, "done");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
