//! HandoffManager + HandoffTool

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use serde_json::Value;

use crate::base::AgentExecutor;
use lc_core::tools::BaseTool;
use lc_core::tools::ToolError;

use super::handoff::{Handoff, HandoffContext, HandoffError, HandoffRecord, HandoffResult};

/// Internal state of the Handoff manager (a single Mutex avoids deadlocks from inconsistent multi-lock acquisition order)
struct HandoffState {
    agents: HashMap<String, Arc<AgentExecutor>>,
    primary: Option<String>,
    history: Vec<HandoffRecord>,
    /// Current handoff chain (from primary, including the agent currently executing), for cycle detection (P1-7).
    chain: Vec<String>,
}

/// Handoff manager: registers multiple Agents and supports task handoff
pub struct HandoffManager {
    state: Mutex<HandoffState>,
    /// Handoff depth limit; exceeding it is rejected (P1-7).
    max_handoff_depth: usize,
}

/// Marker for the conversation-summary segment in handoff input (P2-4).
const SUMMARY_MARKER: &str = "【交接摘要】";
/// Marker for the task segment in handoff input (P2-4).
const TASK_MARKER: &str = "【交接任务】";

/// Folds the conversation summary from the handoff context into the target Agent's input (P2-4).
///
/// Falls back to bare task text when the summary is missing or empty, preserving old behavior.
fn build_handoff_input(handoff: &Handoff) -> String {
    let summary = handoff
        .context
        .as_ref()
        .and_then(|c| c.conversation_summary.as_deref())
        .unwrap_or_default();
    if summary.is_empty() {
        return handoff.task.clone();
    }
    format!(
        "{SUMMARY_MARKER}\n{summary}\n\n{TASK_MARKER}\n{}",
        handoff.task
    )
}

impl HandoffManager {
    /// Creates a new HandoffManager.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(HandoffState {
                agents: HashMap::new(),
                primary: None,
                history: Vec::new(),
                chain: Vec::new(),
            }),
            max_handoff_depth: 10,
        }
    }

    /// Sets the handoff depth limit (default 10, P1-7).
    pub fn with_max_handoff_depth(mut self, depth: usize) -> Self {
        self.max_handoff_depth = depth.max(1);
        self
    }

    /// Registers an Agent
    pub fn register_agent(
        &self,
        name: impl Into<String>,
        executor: Arc<AgentExecutor>,
    ) -> Result<(), HandoffError> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .agents
            .insert(name.into(), executor);
        Ok(())
    }

    /// Sets the primary Agent
    pub fn set_primary(&self, name: &str) -> Result<(), HandoffError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.agents.contains_key(name) {
            return Err(HandoffError::AgentNotFound(name.to_string()));
        }
        state.primary = Some(name.to_string());
        Ok(())
    }

    /// Executes a handoff: gives the task to the target Agent
    pub async fn execute_handoff(&self, handoff: Handoff) -> Result<HandoffResult, HandoffError> {
        let executor = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

            // P1-7: handoff depth limit
            if state.chain.len() >= self.max_handoff_depth {
                return Err(HandoffError::MaxHandoffDepthExceeded(
                    self.max_handoff_depth,
                ));
            }
            // P1-7: cycle detection - the target is already in the chain, meaning an A→B→A cycle
            if state.chain.contains(&handoff.target_agent) {
                return Err(HandoffError::HandoffCycleDetected(
                    handoff.target_agent.clone(),
                ));
            }
            let executor = state
                .agents
                .get(&handoff.target_agent)
                .ok_or_else(|| HandoffError::AgentNotFound(handoff.target_agent.clone()))?
                .clone();
            state.chain.push(handoff.target_agent.clone());
            executor
        };

        // P2-4: fold the conversation summary from the handoff context into the target Agent's input, rather than transferring control raw.
        let input = build_handoff_input(&handoff);
        let result = executor
            .invoke(input)
            .await
            .map_err(|e| HandoffError::ExecutionError(e.to_string()));

        // Pop the current chain node whether it succeeds or fails, so a single failure doesn't poison subsequent handoffs
        let result = match result {
            Ok(r) => r,
            Err(e) => {
                self.state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .chain
                    .pop();
                return Err(e);
            }
        };

        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.chain.pop();
            let from = state
                .chain
                .last()
                .cloned()
                .or_else(|| state.primary.clone())
                .unwrap_or_default();
            state.history.push(HandoffRecord {
                from_agent: from,
                to_agent: handoff.target_agent.clone(),
                task: handoff.task.clone(),
                result: result.clone(),
                timestamp: Utc::now().to_rfc3339(),
            });
        }

        Ok(HandoffResult {
            agent_name: handoff.target_agent,
            result,
            next_handoff: None,
        })
    }

    /// Runs the primary Agent
    pub async fn run(&self, input: String) -> Result<String, HandoffError> {
        let executor = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let primary = state
                .primary
                .clone()
                .ok_or_else(|| HandoffError::AgentNotFound("primary not set".to_string()))?;
            let executor = state
                .agents
                .get(&primary)
                .ok_or_else(|| HandoffError::AgentNotFound(primary.clone()))?
                .clone();
            // P1-7: primary is the chain's start, taking part in cycle detection
            state.chain.clear();
            state.chain.push(primary);
            executor
        };
        let result = executor
            .invoke(input)
            .await
            .map_err(|e| HandoffError::ExecutionError(e.to_string()));
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .chain
            .clear();
        result
    }

    /// Gets the handoff history
    pub fn history(&self) -> Vec<HandoffRecord> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .history
            .clone()
    }

    /// Generates a HandoffTool for each registered Agent (for the primary Agent to call)
    pub fn handoff_tools(self: &Arc<Self>) -> Vec<Arc<dyn BaseTool>> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state
            .agents
            .keys()
            .map(|name| Arc::new(HandoffTool::new(self.clone(), name.clone())) as Arc<dyn BaseTool>)
            .collect()
    }
}

impl Default for HandoffManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Handoff Tool - the handoff tool exposed to the LLM
pub struct HandoffTool {
    manager: Arc<HandoffManager>,
    target_agent: String,
    name: String,
    description: String,
}

impl HandoffTool {
    /// Creates a handoff tool targeting the given Agent.
    pub fn new(manager: Arc<HandoffManager>, target_agent: impl Into<String>) -> Self {
        let target_agent = target_agent.into();
        let name = format!("handoff_to_{}", target_agent);
        let description = format!("将任务交接给 {} agent", target_agent);
        Self {
            manager,
            target_agent,
            name,
            description,
        }
    }
}

#[async_trait]
impl BaseTool for HandoffTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        // The input may be JSON {"task": "...", "summary": "..."} or plain text
        let parsed = serde_json::from_str::<Value>(&input).ok();
        let task = parsed
            .as_ref()
            .and_then(|v| v.get("task").and_then(|t| t.as_str()))
            .map(|s| s.to_string())
            .unwrap_or(input.clone());
        // P2-4: the JSON may carry a summary, passed to the target Agent via the context.
        let summary = parsed
            .as_ref()
            .and_then(|v| v.get("summary").and_then(|s| s.as_str()))
            .map(|s| s.to_string());
        let original_request = parsed
            .as_ref()
            .and_then(|v| v.get("original_request").and_then(|s| s.as_str()))
            .unwrap_or(&input);

        let context =
            Some(HandoffContext::new(original_request).with_summary(summary.unwrap_or_default()));
        let handoff = Handoff {
            target_agent: self.target_agent.clone(),
            task,
            context,
        };
        let result = self
            .manager
            .execute_handoff(handoff)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        Ok(result.result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handoffs::HandoffContext;
    use crate::types::{AgentAction, AgentFinish, AgentOutput, AgentStep, ToolInput};
    use crate::{AgentError, BaseAgent, FunctionCallingAgent};
    use lc_providers::{OpenAIChat, OpenAIConfig};

    fn mock_executor() -> Arc<AgentExecutor> {
        let llm = OpenAIChat::new(OpenAIConfig::default());
        let agent = FunctionCallingAgent::new(llm, vec![], None);
        Arc::new(AgentExecutor::new(
            Arc::new(agent) as Arc<dyn BaseAgent>,
            vec![],
        ))
    }

    /// Offline-capable mock agent for P1-7 nested handoff tests (no real LLM).
    ///
    /// Behavior: the first round either issues one tool call (`first_action`),
    /// or directly triggers one handoff (`direct_handoff`, bypassing
    /// HandoffTool); later rounds return Finish.
    struct OfflineAgent {
        first_action: Option<(&'static str, String)>,
        direct_handoff: Option<String>,
        manager: Option<Arc<HandoffManager>>,
    }

    #[async_trait]
    impl BaseAgent for OfflineAgent {
        async fn plan(
            &self,
            intermediate_steps: &[AgentStep],
            _inputs: &HashMap<String, String>,
        ) -> Result<AgentOutput, AgentError> {
            if !intermediate_steps.is_empty() {
                return Ok(AgentOutput::Finish(AgentFinish::new(
                    "done".to_string(),
                    String::new(),
                )));
            }
            if let Some(target) = &self.direct_handoff {
                // First round triggers the handoff directly inside plan; cycle detection makes execute_handoff reject it immediately.
                let mgr = self.manager.as_ref().unwrap();
                let result = mgr
                    .execute_handoff(Handoff {
                        target_agent: target.clone(),
                        task: "inner".to_string(),
                        context: None,
                    })
                    .await;
                return match result {
                    Ok(r) => Ok(AgentOutput::Finish(AgentFinish::new(
                        r.result,
                        String::new(),
                    ))),
                    Err(e) => Err(AgentError::Other(format!("handoff failed: {}", e))),
                };
            }
            if let Some((tool, input)) = &self.first_action {
                return Ok(AgentOutput::Action(AgentAction {
                    tool: tool.to_string(),
                    tool_input: ToolInput::String {
                        value: input.clone(),
                    },
                    log: String::new(),
                }));
            }
            Ok(AgentOutput::Finish(AgentFinish::new(
                "done".to_string(),
                String::new(),
            )))
        }
    }

    #[test]
    fn test_register_and_set_primary() {
        let mgr = HandoffManager::new();
        mgr.register_agent("researcher", mock_executor()).unwrap();
        mgr.set_primary("researcher").unwrap();
        // Non-existent agent
        assert!(mgr.set_primary("nope").is_err());
    }

    #[test]
    fn test_set_primary_without_register_errors() {
        let mgr = HandoffManager::new();
        assert!(mgr.set_primary("ghost").is_err());
    }

    #[tokio::test]
    async fn test_execute_handoff_not_found() {
        let mgr = HandoffManager::new();
        let handoff = Handoff {
            target_agent: "nope".to_string(),
            task: "task".to_string(),
            context: None,
        };
        assert!(mgr.execute_handoff(handoff).await.is_err());
    }

    #[tokio::test]
    async fn test_run_without_primary_errors() {
        let mgr = HandoffManager::new();
        assert!(mgr.run("hi".to_string()).await.is_err());
    }

    #[test]
    fn test_handoff_tool_name() {
        let mgr = Arc::new(HandoffManager::new());
        let tool = HandoffTool::new(mgr, "writer".to_string());
        assert_eq!(tool.name(), "handoff_to_writer");
        assert!(tool.description().contains("writer"));
    }

    #[test]
    fn test_handoff_tools_generated() {
        let mgr = HandoffManager::new();
        mgr.register_agent("a", mock_executor()).unwrap();
        mgr.register_agent("b", mock_executor()).unwrap();
        let mgr = Arc::new(mgr);
        let tools = mgr.handoff_tools();
        assert_eq!(tools.len(), 2);
    }

    /// P1-7: A hands off to B, B hands back to A → cycle detected and error, no infinite loop.
    #[tokio::test]
    async fn test_handoff_cycle_detected() {
        let manager = Arc::new(HandoffManager::new());
        let agent_a = OfflineAgent {
            first_action: Some(("handoff_to_b", "task for b".to_string())),
            direct_handoff: None,
            manager: None,
        };
        let agent_b = OfflineAgent {
            first_action: None,
            direct_handoff: Some("a".to_string()),
            manager: Some(manager.clone()),
        };
        let tool_b = HandoffTool::new(manager.clone(), "b".to_string());
        let executor_a = AgentExecutor::new(
            Arc::new(agent_a) as Arc<dyn BaseAgent>,
            vec![Arc::new(tool_b) as Arc<dyn BaseTool>],
        );
        let executor_b = AgentExecutor::new(Arc::new(agent_b) as Arc<dyn BaseAgent>, vec![]);

        manager.register_agent("a", Arc::new(executor_a)).unwrap();
        manager.register_agent("b", Arc::new(executor_b)).unwrap();
        manager.set_primary("a").unwrap();

        let err = manager.run("start".to_string()).await.unwrap_err();
        assert!(
            err.to_string().contains("handoff cycle"),
            "should return a cycle-detection error, got: {}",
            err
        );
    }

    /// P1-7: handoff depth exceeds the limit → rejected with an error.
    #[tokio::test]
    async fn test_handoff_depth_limit() {
        // Depth limit 1: primary(a) already takes one layer, so handing off to b exceeds it.
        let manager = Arc::new(HandoffManager::new().with_max_handoff_depth(1));
        let agent_a = OfflineAgent {
            first_action: Some(("handoff_to_b", "t".to_string())),
            direct_handoff: None,
            manager: None,
        };
        let agent_b = OfflineAgent {
            first_action: None,
            direct_handoff: None,
            manager: None,
        };
        let tool_b = HandoffTool::new(manager.clone(), "b".to_string());
        let executor_a = AgentExecutor::new(
            Arc::new(agent_a) as Arc<dyn BaseAgent>,
            vec![Arc::new(tool_b) as Arc<dyn BaseTool>],
        );
        let executor_b = AgentExecutor::new(Arc::new(agent_b) as Arc<dyn BaseAgent>, vec![]);

        manager.register_agent("a", Arc::new(executor_a)).unwrap();
        manager.register_agent("b", Arc::new(executor_b)).unwrap();
        manager.set_primary("a").unwrap();

        let err = manager.run("start".to_string()).await.unwrap_err();
        assert!(
            err.to_string().contains("handoff depth"),
            "should return a depth-exceeded error, got: {}",
            err
        );
    }

    /// Captures the input the target Agent receives (for P2-4 tests).
    struct CaptureAgent {
        received: Arc<Mutex<Option<String>>>,
    }

    #[async_trait]
    impl BaseAgent for CaptureAgent {
        async fn plan(
            &self,
            _intermediate_steps: &[AgentStep],
            inputs: &HashMap<String, String>,
        ) -> Result<AgentOutput, AgentError> {
            *self.received.lock().unwrap_or_else(|e| e.into_inner()) = inputs.get("input").cloned();
            Ok(AgentOutput::Finish(AgentFinish::new(
                "done".to_string(),
                String::new(),
            )))
        }
    }

    fn capture_executor(received: &Arc<Mutex<Option<String>>>) -> Arc<AgentExecutor> {
        Arc::new(AgentExecutor::new(
            Arc::new(CaptureAgent {
                received: received.clone(),
            }) as Arc<dyn BaseAgent>,
            vec![],
        ))
    }

    /// P2-4: when a handoff carries a summary, the target Agent's input contains the summary and task markers.
    #[tokio::test]
    async fn test_handoff_carries_conversation_summary() {
        let received = Arc::new(Mutex::new(None));
        let manager = HandoffManager::new();
        manager
            .register_agent("researcher", capture_executor(&received))
            .unwrap();

        let ctx = HandoffContext::new("原始请求").with_summary("此前对话要点A");
        let handoff = Handoff {
            target_agent: "researcher".to_string(),
            task: "继续研究".to_string(),
            context: Some(ctx),
        };
        manager.execute_handoff(handoff).await.unwrap();

        let input = received
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .unwrap();
        assert!(
            input.contains("此前对话要点A"),
            "目标应收到摘要,实际: {input}"
        );
        assert!(input.contains("继续研究"), "目标应收到任务,实际: {input}");
        assert!(input.contains("【交接摘要】"), "应带摘要标记");
    }

    /// P2-4: when a handoff carries no summary, the target Agent receives bare task text (old behavior unchanged).
    #[tokio::test]
    async fn test_handoff_without_summary_bare_task() {
        let received = Arc::new(Mutex::new(None));
        let manager = HandoffManager::new();
        manager
            .register_agent("researcher", capture_executor(&received))
            .unwrap();

        let handoff = Handoff {
            target_agent: "researcher".to_string(),
            task: "只做这个".to_string(),
            context: None,
        };
        manager.execute_handoff(handoff).await.unwrap();

        let input = received
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .unwrap();
        assert_eq!(input, "只做这个");
    }

    /// P2-4: an empty summary degrades to the bare task, not polluting the input.
    #[tokio::test]
    async fn test_handoff_empty_summary_bare_task() {
        let received = Arc::new(Mutex::new(None));
        let manager = HandoffManager::new();
        manager
            .register_agent("researcher", capture_executor(&received))
            .unwrap();

        let ctx = HandoffContext::new("原始请求").with_summary("");
        let handoff = Handoff {
            target_agent: "researcher".to_string(),
            task: "taskY".to_string(),
            context: Some(ctx),
        };
        manager.execute_handoff(handoff).await.unwrap();

        let input = received
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .unwrap();
        assert_eq!(input, "taskY");
    }

    /// P2-4: when HandoffTool's JSON input carries a summary, it reaches the target via the context.
    #[tokio::test]
    async fn test_handoff_tool_summary_json_flows_to_target() {
        let received = Arc::new(Mutex::new(None));
        let manager = Arc::new(HandoffManager::new());
        manager
            .register_agent("writer", capture_executor(&received))
            .unwrap();

        let tool = HandoffTool::new(manager, "writer".to_string());
        let json = r#"{"task": "写总结", "summary": "会议要点S"}"#;
        tool.run(json.to_string()).await.unwrap();

        let input = received
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .unwrap();
        assert!(
            input.contains("会议要点S"),
            "工具 JSON summary 应传到目标,实际: {input}"
        );
        assert!(input.contains("写总结"));
    }

    /// P2-4: HandoffTool plain-text input stays a bare-task transfer.
    #[tokio::test]
    async fn test_handoff_tool_plain_text_bare_task() {
        let received = Arc::new(Mutex::new(None));
        let manager = Arc::new(HandoffManager::new());
        manager
            .register_agent("writer", capture_executor(&received))
            .unwrap();

        let tool = HandoffTool::new(manager, "writer".to_string());
        tool.run("纯文本任务".to_string()).await.unwrap();

        let input = received
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .unwrap();
        assert_eq!(input, "纯文本任务");
    }
}
