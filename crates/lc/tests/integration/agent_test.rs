// tests/agent_test.rs
//! Agent 系统测试

mod tool_callbacks_integration;

use async_trait::async_trait;
use langchainrust::{
    AgentAction, AgentError, AgentExecutor, AgentFinish, AgentOutput, AgentStep, BaseAgent,
    Calculator,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Mock Agent 用于测试
struct MockAgent {
    /// 预设的动作序列
    actions: Vec<AgentOutput>,
    /// 当前索引
    current: std::sync::atomic::AtomicUsize,
}

impl MockAgent {
    fn new(actions: Vec<AgentOutput>) -> Self {
        Self {
            actions,
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl BaseAgent for MockAgent {
    async fn plan(
        &self,
        _intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        let index = self
            .current
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        if index < self.actions.len() {
            Ok(self.actions[index].clone())
        } else {
            // 默认返回完成
            Ok(AgentOutput::Finish(AgentFinish::new(
                "默认答案".to_string(),
                String::new(),
            )))
        }
    }
}

/// 测试 AgentAction 序列化
#[test]
fn test_agent_action_serialization() {
    let action = AgentAction {
        tool: "calculator".to_string(),
        tool_input: langchainrust::ToolInput::String {
            value: "2 + 2".to_string(),
        },
        log: "Thought: 计算加法\nAction: calculator\nAction Input: 2 + 2".to_string(),
    };

    let json = serde_json::to_string(&action).unwrap();
    assert!(json.contains("calculator"));
    assert!(json.contains("2 + 2"));

    let deserialized: AgentAction = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.tool, "calculator");
}

/// 测试 AgentFinish
#[test]
fn test_agent_finish() {
    let finish = AgentFinish::new(
        "答案是 42".to_string(),
        "Final Answer: 答案是 42".to_string(),
    );

    assert_eq!(finish.output(), Some("答案是 42"));
    assert!(finish.return_values.contains_key("output"));
}

/// 测试 AgentStep
#[test]
fn test_agent_step() {
    let action = AgentAction {
        tool: "test".to_string(),
        tool_input: langchainrust::ToolInput::String {
            value: "input".to_string(),
        },
        log: "log".to_string(),
    };

    let step = AgentStep::new(action.clone(), "observation".to_string());

    assert_eq!(step.action.tool, "test");
    assert_eq!(step.observation, "observation");
}

/// 测试 AgentExecutor 直接返回答案
#[tokio::test]
async fn test_agent_executor_direct_answer() {
    // 创建直接返回答案的 Agent
    let agent = Arc::new(MockAgent::new(vec![AgentOutput::Finish(AgentFinish::new(
        "直接答案".to_string(),
        String::new(),
    ))]));

    let executor = AgentExecutor::new(agent, vec![]);

    let result = executor.invoke("测试问题".to_string()).await.unwrap();
    assert_eq!(result, "直接答案");
}

/// 测试 AgentExecutor 执行工具
#[tokio::test]
async fn test_agent_executor_with_tool() {
    // 创建先调用工具再返回答案的 Agent
    let agent = Arc::new(MockAgent::new(vec![
        AgentOutput::Action(AgentAction {
            tool: "calculator".to_string(),
            tool_input: langchainrust::ToolInput::String {
                value: "{\"expression\": \"2 + 3\"}".to_string(),
            },
            log: String::new(),
        }),
        AgentOutput::Finish(AgentFinish::new("计算完成".to_string(), String::new())),
    ]));

    // 创建工具
    let calculator = Arc::new(Calculator::new());

    let executor = AgentExecutor::new(agent, vec![calculator]);

    let result = executor.invoke("计算 2 + 3".to_string()).await.unwrap();
    assert_eq!(result, "计算完成");
}

/// 测试 AgentExecutor 工具未找到
#[tokio::test]
async fn test_agent_executor_tool_not_found() {
    let agent = Arc::new(MockAgent::new(vec![AgentOutput::Action(AgentAction {
        tool: "nonexistent".to_string(),
        tool_input: langchainrust::ToolInput::String {
            value: "input".to_string(),
        },
        log: String::new(),
    })]));

    let executor = AgentExecutor::new(agent, vec![]);

    let result = executor.invoke("测试".to_string()).await;
    assert!(result.is_err());

    match result.unwrap_err() {
        AgentError::ToolNotFound(name) => assert_eq!(name, "nonexistent"),
        _ => panic!("期望 ToolNotFound 错误"),
    }
}

/// 测试 AgentExecutor 最大迭代次数
#[tokio::test]
async fn test_agent_executor_max_iterations() {
    // 创建总是返回 Action 的 Agent
    let agent = Arc::new(MockAgent::new(vec![
        AgentOutput::Action(AgentAction {
            tool: "calculator".to_string(),
            tool_input: langchainrust::ToolInput::String { value: "{\"expression\": \"1\"}".to_string() },
            log: String::new(),
        });
        20 // 足够多的动作
    ]));

    let calculator = Arc::new(Calculator::new());

    let executor = AgentExecutor::new(agent, vec![calculator]).with_max_iterations(3);

    let result = executor.invoke("测试".to_string()).await.unwrap();
    assert!(result.contains("iteration limit"));
}

/// 测试 AgentExecutor 详细输出
#[tokio::test]
async fn test_agent_executor_verbose() {
    let agent = Arc::new(MockAgent::new(vec![AgentOutput::Finish(AgentFinish::new(
        "答案".to_string(),
        String::new(),
    ))]));

    let executor = AgentExecutor::new(agent, vec![]).with_verbose(true);

    let result = executor.invoke("测试".to_string()).await.unwrap();
    assert_eq!(result, "答案");
}

/// 测试 ToolInput 显示
#[test]
fn test_tool_input_display() {
    let input = langchainrust::ToolInput::String {
        value: "test".to_string(),
    };
    assert_eq!(format!("{}", input), "test");

    let json_input = langchainrust::ToolInput::Object {
        value: serde_json::json!({"key": "value"}),
    };
    assert!(format!("{}", json_input).contains("key"));
}

/// 测试 AgentError 显示
#[test]
fn test_agent_error_display() {
    let error = AgentError::ToolNotFound("test".to_string());
    assert!(error.to_string().contains("Tool not found"));

    let error = AgentError::MaxIterationsReached;
    assert!(error.to_string().contains("Max iterations reached"));

    let error = AgentError::OutputParsingError("parse error".to_string());
    assert!(error.to_string().contains("Output parsing error"));
}
