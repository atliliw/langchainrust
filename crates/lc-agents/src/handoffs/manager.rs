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

/// Handoff 管理器内部状态(单 Mutex 避免多锁获取顺序不一致导致死锁)
struct HandoffState {
    agents: HashMap<String, Arc<AgentExecutor>>,
    primary: Option<String>,
    history: Vec<HandoffRecord>,
    /// 当前交接链(从 primary 起,含正在执行的 agent),用于环检测(P1-7)。
    chain: Vec<String>,
}

/// Handoff 管理器:注册多个 Agent,支持任务交接
pub struct HandoffManager {
    state: Mutex<HandoffState>,
    /// 交接深度上限,超过即拒绝(P1-7)。
    max_handoff_depth: usize,
}

/// 交接输入里,对话摘要段的标记(P2-4)。
const SUMMARY_MARKER: &str = "【交接摘要】";
/// 交接输入里,任务段的标记(P2-4)。
const TASK_MARKER: &str = "【交接任务】";

/// 把交接上下文里的对话摘要拼进目标 Agent 的输入(P2-4)。
///
/// 摘要缺失或为空时退化为裸任务文本,保持旧行为。
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
    /// 创建新的 HandoffManager。
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

    /// 设置交接深度上限,默认 10(P1-7)。
    pub fn with_max_handoff_depth(mut self, depth: usize) -> Self {
        self.max_handoff_depth = depth.max(1);
        self
    }

    /// 注册 Agent
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

    /// 设置主 Agent
    pub fn set_primary(&self, name: &str) -> Result<(), HandoffError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.agents.contains_key(name) {
            return Err(HandoffError::AgentNotFound(name.to_string()));
        }
        state.primary = Some(name.to_string());
        Ok(())
    }

    /// 执行交接:把任务交给目标 Agent
    pub async fn execute_handoff(&self, handoff: Handoff) -> Result<HandoffResult, HandoffError> {
        let executor = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

            // P1-7:交接深度上限
            if state.chain.len() >= self.max_handoff_depth {
                return Err(HandoffError::MaxHandoffDepthExceeded(
                    self.max_handoff_depth,
                ));
            }
            // P1-7:环检测 - 目标已在交接链中,说明 A→B→A 循环
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

        // P2-4:把交接上下文里的对话摘要拼进目标 Agent 输入,而非裸转移控制权。
        let input = build_handoff_input(&handoff);
        let result = executor
            .invoke(input)
            .await
            .map_err(|e| HandoffError::ExecutionError(e.to_string()));

        // 无论成败都弹出当前链节点,避免一次失败后 chain 污染后续交接
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

    /// 运行主 Agent
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
            // P1-7:primary 作为交接链起点,参与环检测
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

    /// 获取交接历史
    pub fn history(&self) -> Vec<HandoffRecord> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .history
            .clone()
    }

    /// 为每个注册的 Agent 生成 HandoffTool(供主 Agent 调用)
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

/// Handoff Tool - 暴露给 LLM 的交接工具
pub struct HandoffTool {
    manager: Arc<HandoffManager>,
    target_agent: String,
    name: String,
    description: String,
}

impl HandoffTool {
    /// 创建一个指向指定目标 Agent 的交接工具。
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
        // 输入可能是 JSON {"task": "...", "summary": "..."} 或纯文本
        let parsed = serde_json::from_str::<Value>(&input).ok();
        let task = parsed
            .as_ref()
            .and_then(|v| v.get("task").and_then(|t| t.as_str()))
            .map(|s| s.to_string())
            .unwrap_or(input.clone());
        // P2-4:JSON 里可带 summary,经上下文传给目标 Agent。
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

    /// 可离线运行的 mock agent,用于 P1-7 嵌套交接测试(不走真实 LLM)。
    ///
    /// 行为:首轮要么发一次工具调用(`first_action`),要么直接触发一次交接
    /// (`direct_handoff`,绕过 HandoffTool),后续轮返回 Finish。
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
                // 首轮在 plan 内直接触发交接;环检测时 execute_handoff 会立刻拒绝。
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
        // 不存在的 agent
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

    /// P1-7:A 交接给 B,B 又交接给 A → 检测到环并报错,不死循环。
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

    /// P1-7:交接深度超过上限 → 拒绝并报错。
    #[tokio::test]
    async fn test_handoff_depth_limit() {
        // 深度上限 1:primary(a) 已占一层,再交接 b 即超限。
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

    /// 捕获目标 Agent 收到的输入(P2-4 测试用)。
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

    /// P2-4:交接携带摘要时,目标 Agent 收到的输入包含摘要与任务标记。
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

    /// P2-4:交接不携带摘要时,目标 Agent 收到裸任务文本(旧行为不变)。
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

    /// P2-4:空摘要退化为裸任务,不污染输入。
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

    /// P2-4:HandoffTool 的 JSON 输入里带 summary 时,经上下文传给目标。
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

    /// P2-4:HandoffTool 纯文本输入保持裸任务传递。
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
