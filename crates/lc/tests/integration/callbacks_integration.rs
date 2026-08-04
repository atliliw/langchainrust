// tests/integration/callbacks_integration.rs
//! 回调系统集成测试
//!
//! 验证回调系统与 LLM 和 Agent 等真实组件的集成

use async_trait::async_trait;
use langchainrust::{
    AgentError, AgentExecutor, AgentFinish, AgentOutput, AgentStep, BaseAgent, CallbackHandler,
    CallbackManager, RunTree, RunType, RunnableConfig,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ============================================================================
// Mock Agent（用于测试）
// ============================================================================

/// 简单的 Mock Agent，返回固定响应
/// 用于测试 AgentExecutor 的回调集成，无需依赖真实 LLM
struct MockAgent;

#[async_trait]
impl BaseAgent for MockAgent {
    async fn plan(
        &self,
        _intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        let input = inputs.get("input").unwrap();
        Ok(AgentOutput::Finish(AgentFinish::new(
            format!("Processed: {}", input),
            String::new(),
        )))
    }
}

// ============================================================================
// 计数回调处理器（用于测试）
// ============================================================================

/// 记录所有回调调用的处理器
/// 用于验证回调按正确顺序被调用
struct CountingCallbackHandler {
    call_count: Arc<Mutex<Vec<String>>>,
}

impl CountingCallbackHandler {
    fn new() -> Self {
        Self {
            call_count: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl CallbackHandler for CountingCallbackHandler {
    async fn on_run_start(&self, run: &RunTree) {
        self.call_count
            .lock()
            .unwrap()
            .push(format!("start:{}", run.name));
    }

    async fn on_run_end(&self, run: &RunTree) {
        self.call_count
            .lock()
            .unwrap()
            .push(format!("end:{}", run.name));
    }

    async fn on_run_error(&self, run: &RunTree, error: &str) {
        self.call_count
            .lock()
            .unwrap()
            .push(format!("error:{}:{}", run.name, error));
    }

    async fn on_chain_start(&self, run: &RunTree, _inputs: &serde_json::Value) {
        self.call_count
            .lock()
            .unwrap()
            .push(format!("chain_start:{}", run.name));
    }

    async fn on_chain_end(&self, run: &RunTree, _outputs: &serde_json::Value) {
        self.call_count
            .lock()
            .unwrap()
            .push(format!("chain_end:{}", run.name));
    }

    async fn on_tool_start(&self, run: &RunTree, tool_name: &str, _input: &str) {
        self.call_count
            .lock()
            .unwrap()
            .push(format!("tool_start:{}:{}", run.name, tool_name));
    }

    async fn on_tool_end(&self, run: &RunTree, _output: &str) {
        self.call_count
            .lock()
            .unwrap()
            .push(format!("tool_end:{}", run.name));
    }
}

// ============================================================================
// AgentExecutor 集成测试
// ============================================================================

#[tokio::test]
async fn test_agent_executor_with_callbacks() {
    // 验证 AgentExecutor 在处理输入时触发 chain_start 和 chain_end 回调
    let handler = Arc::new(CountingCallbackHandler::new());
    let calls = Arc::clone(&handler.call_count);

    let callbacks = Arc::new(CallbackManager::new().add_handler(handler));

    let agent = Arc::new(MockAgent);
    let executor = AgentExecutor::new(agent, vec![]).with_callbacks(callbacks);

    let result = executor.invoke("test input".to_string()).await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Processed: test input");

    // 验证回调被调用
    let calls = calls.lock().unwrap();
    assert!(
        calls
            .iter()
            .any(|c| c.starts_with("chain_start:AgentExecutor")),
        "应调用 chain_start 回调"
    );
    assert!(
        calls
            .iter()
            .any(|c| c.starts_with("chain_end:AgentExecutor")),
        "应调用 chain_end 回调"
    );
}

// ============================================================================
// RunnableConfig 集成测试
// ============================================================================

#[tokio::test]
async fn test_runnable_config_with_callbacks() {
    // 验证 RunnableConfig 正确存储和传递回调
    let handler = Arc::new(CountingCallbackHandler::new());
    let _calls = Arc::clone(&handler.call_count);

    let callbacks = Arc::new(CallbackManager::new().add_handler(handler));

    let config = RunnableConfig::new()
        .with_callbacks(callbacks)
        .with_tag("test")
        .with_run_name("test_run");

    // 验证配置字段设置正确
    assert!(config.callbacks.is_some(), "callbacks 应被设置");
    assert!(
        config.tags.contains(&"test".to_string()),
        "tags 应包含 'test'"
    );
    assert_eq!(config.run_name, Some("test_run".to_string()));
}

// ============================================================================
// 多处理器测试
// ============================================================================

#[tokio::test]
async fn test_multiple_handlers() {
    // 验证多个处理器都接收到回调
    // 这测试了事件分发到所有注册处理器的扇出模式
    let handler1 = Arc::new(CountingCallbackHandler::new());
    let handler2 = Arc::new(CountingCallbackHandler::new());

    let calls1 = Arc::clone(&handler1.call_count);
    let calls2 = Arc::clone(&handler2.call_count);

    let callbacks = Arc::new(
        CallbackManager::new()
            .add_handler(handler1)
            .add_handler(handler2),
    );

    let run = RunTree::new("Test", RunType::Chain, serde_json::json!({}));

    for h in callbacks.handlers() {
        h.on_chain_start(&run, &serde_json::json!({})).await;
    }

    let calls1 = calls1.lock().unwrap();
    let calls2 = calls2.lock().unwrap();

    assert_eq!(calls1.len(), 1, "handler1 应接收到回调");
    assert_eq!(calls2.len(), 1, "handler2 应接收到回调");
}

// ============================================================================
// RunTree 集成测试
// ============================================================================

#[test]
fn test_run_tree_with_project() {
    // 验证可为 LangSmith 组织设置项目名称
    // 用于追踪到特定的 LangSmith 项目
    let run =
        RunTree::new("Test", RunType::Chain, serde_json::json!({})).with_project("my-project");

    assert_eq!(run.project_name, Some("my-project".to_string()));
}

#[test]
fn test_run_tree_chain() {
    // 验证典型链层次结构创建
    // Chain 通常包含 Tool 调用和 LLM 调用作为子运行
    let parent = RunTree::new(
        "Chain",
        RunType::Chain,
        serde_json::json!({"input": "query"}),
    );
    let tool_run = parent.create_child(
        "Calculator",
        RunType::Tool,
        serde_json::json!({"expr": "1+1"}),
    );
    let llm_run = parent.create_child("LLM", RunType::Llm, serde_json::json!({"prompt": "..."}));

    // 两个子运行应引用同一个父运行
    assert_eq!(tool_run.parent_run_id, Some(parent.id));
    assert_eq!(llm_run.parent_run_id, Some(parent.id));
    // 所有运行应共享相同的 trace_id（根运行的 ID）
    assert_eq!(tool_run.trace_id, Some(parent.id));
    assert_eq!(llm_run.trace_id, Some(parent.id));
}

#[test]
fn test_run_tree_tags_and_metadata() {
    // 验证在生产场景中标签和元数据的组合使用
    let run = RunTree::new("Test", RunType::Chain, serde_json::json!({}))
        .with_tag("production")
        .with_tag("v2")
        .with_metadata("user_id", serde_json::json!("123"))
        .with_metadata("session", serde_json::json!({"id": "abc", "start": 12345}));

    // 验证标签
    assert_eq!(run.tags.len(), 2);
    assert!(run.tags.contains(&"production".to_string()));
    assert!(run.tags.contains(&"v2".to_string()));

    // 验证元数据
    assert_eq!(run.metadata.len(), 2);
    assert_eq!(run.metadata.get("user_id").unwrap(), "123");
}
