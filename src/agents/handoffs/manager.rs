//! HandoffManager + HandoffTool

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use serde_json::Value;

use crate::agents::AgentExecutor;
use crate::core::tools::ToolError;
use crate::BaseTool;

use super::handoff::{Handoff, HandoffError, HandoffRecord, HandoffResult};

/// Handoff 管理器:注册多个 Agent,支持任务交接
pub struct HandoffManager {
    agents: Mutex<HashMap<String, Arc<AgentExecutor>>>,
    primary: Mutex<Option<String>>,
    history: Mutex<Vec<HandoffRecord>>,
}

impl HandoffManager {
    pub fn new() -> Self {
        Self {
            agents: Mutex::new(HashMap::new()),
            primary: Mutex::new(None),
            history: Mutex::new(Vec::new()),
        }
    }

    /// 注册 Agent
    pub fn register_agent(
        &self,
        name: impl Into<String>,
        executor: Arc<AgentExecutor>,
    ) -> Result<(), HandoffError> {
        self.agents.lock().unwrap().insert(name.into(), executor);
        Ok(())
    }

    /// 设置主 Agent
    pub fn set_primary(&self, name: &str) -> Result<(), HandoffError> {
        let agents = self.agents.lock().unwrap();
        if !agents.contains_key(name) {
            return Err(HandoffError::AgentNotFound(name.to_string()));
        }
        *self.primary.lock().unwrap() = Some(name.to_string());
        Ok(())
    }

    /// 执行交接:把任务交给目标 Agent
    pub async fn execute_handoff(&self, handoff: Handoff) -> Result<HandoffResult, HandoffError> {
        let executor = {
            let agents = self.agents.lock().unwrap();
            agents
                .get(&handoff.target_agent)
                .ok_or_else(|| HandoffError::AgentNotFound(handoff.target_agent.clone()))?
                .clone()
        };

        let result = executor
            .invoke(handoff.task.clone())
            .await
            .map_err(|e| HandoffError::ExecutionError(e.to_string()))?;

        let from = self
            .primary
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_default();
        self.history.lock().unwrap().push(HandoffRecord {
            from_agent: from,
            to_agent: handoff.target_agent.clone(),
            task: handoff.task.clone(),
            result: result.clone(),
            timestamp: Utc::now().to_rfc3339(),
        });

        Ok(HandoffResult {
            agent_name: handoff.target_agent,
            result,
            next_handoff: None,
        })
    }

    /// 运行主 Agent
    pub async fn run(&self, input: String) -> Result<String, HandoffError> {
        let primary = self
            .primary
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| HandoffError::AgentNotFound("未设置 primary".to_string()))?;
        let executor = {
            let agents = self.agents.lock().unwrap();
            agents
                .get(&primary)
                .ok_or_else(|| HandoffError::AgentNotFound(primary.clone()))?
                .clone()
        };
        executor
            .invoke(input)
            .await
            .map_err(|e| HandoffError::ExecutionError(e.to_string()))
    }

    /// 获取交接历史
    pub fn history(&self) -> Vec<HandoffRecord> {
        self.history.lock().unwrap().clone()
    }

    /// 为每个注册的 Agent 生成 HandoffTool(供主 Agent 调用)
    pub fn handoff_tools(self: &Arc<Self>) -> Vec<Arc<dyn BaseTool>> {
        let agents = self.agents.lock().unwrap();
        agents
            .keys()
            .map(|name| {
                Arc::new(HandoffTool::new(self.clone(), name.clone())) as Arc<dyn BaseTool>
            })
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
    pub fn new(manager: Arc<HandoffManager>, target_agent: String) -> Self {
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
        // 输入可能是 JSON {"task": "..."} 或纯文本
        let task = serde_json::from_str::<Value>(&input)
            .ok()
            .and_then(|v| v.get("task").and_then(|t| t.as_str()).map(|s| s.to_string()))
            .unwrap_or(input);

        let handoff = Handoff {
            target_agent: self.target_agent.clone(),
            task,
            context: None,
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
    use crate::{BaseAgent, FunctionCallingAgent, OpenAIChat, OpenAIConfig};

    fn mock_executor() -> Arc<AgentExecutor> {
        let llm = OpenAIChat::new(OpenAIConfig::default());
        let agent = FunctionCallingAgent::new(llm, vec![], None);
        Arc::new(AgentExecutor::new(
            Arc::new(agent) as Arc<dyn BaseAgent>,
            vec![],
        ))
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
}
