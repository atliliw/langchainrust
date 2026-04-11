// tests/integration/tool_callbacks_integration.rs
//! 工具回调系统集成测试
//!
//! 测试真实工具执行时回调系统的完整集成：
//! - AgentExecutor 执行工具时的回调触发
//! - 工具执行成功/失败的回调记录
//! - 多工具执行的回调追踪
//! - 回调数据的完整性验证

use langchainrust::{
    AgentExecutor, BaseAgent, AgentError, AgentOutput, AgentStep, AgentFinish, AgentAction,
    CallbackManager, CallbackHandler, RunTree,
    tools::{Calculator, SimpleMathTool},
    BaseTool,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ============================================================================
// 工具追踪回调处理器
// ============================================================================

/// 详细记录工具调用的回调处理器
/// 用于测试验证工具回调的完整流程
pub struct ToolTrackingHandler {
    /// 记录所有回调调用（格式: "事件类型:工具名:数据"）
    calls: Arc<Mutex<Vec<String>>>,
    /// 工具开始调用次数
    tool_start_count: Arc<Mutex<usize>>,
    /// 工具结束调用次数  
    tool_end_count: Arc<Mutex<usize>>,
    /// 工具错误调用次数
    tool_error_count: Arc<Mutex<usize>>,
    /// 记录工具输入
    tool_inputs: Arc<Mutex<Vec<(String, String)>>>,
    /// 记录工具输出
    tool_outputs: Arc<Mutex<Vec<(String, String)>>>,
    /// 记录工具执行时间（毫秒）
    tool_durations: Arc<Mutex<Vec<(String, i64)>>>,
}

impl ToolTrackingHandler {
    pub fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            tool_start_count: Arc::new(Mutex::new(0)),
            tool_end_count: Arc::new(Mutex::new(0)),
            tool_error_count: Arc::new(Mutex::new(0)),
            tool_inputs: Arc::new(Mutex::new(Vec::new())),
            tool_outputs: Arc::new(Mutex::new(Vec::new())),
            tool_durations: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// 获取所有调用记录
    pub fn get_calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
    
    /// 获取工具开始次数
    pub fn get_tool_start_count(&self) -> usize {
        *self.tool_start_count.lock().unwrap()
    }
    
    /// 获取工具结束次数
    pub fn get_tool_end_count(&self) -> usize {
        *self.tool_end_count.lock().unwrap()
    }
    
    /// 获取工具错误次数
    pub fn get_tool_error_count(&self) -> usize {
        *self.tool_error_count.lock().unwrap()
    }
    
    /// 获取工具输入记录
    pub fn get_tool_inputs(&self) -> Vec<(String, String)> {
        self.tool_inputs.lock().unwrap().clone()
    }
    
    /// 获取工具输出记录
    pub fn get_tool_outputs(&self) -> Vec<(String, String)> {
        self.tool_outputs.lock().unwrap().clone()
    }
    
    /// 验证工具调用顺序正确（start → end 或 start → error）
    pub fn verify_call_order(&self) -> bool {
        let calls = self.get_calls();
        let mut last_was_start = false;
        
        for call in calls {
            if call.starts_with("tool_start:") {
                if last_was_start {
                    // 连续两个 start，顺序错误
                    return false;
                }
                last_was_start = true;
            } else if call.starts_with("tool_end:") || call.starts_with("tool_error:") {
                if !last_was_start {
                    // end/error 前没有 start，顺序错误
                    return false;
                }
                last_was_start = false;
            }
        }
        
        // 最后应该是 end/error，不能是 start
        !last_was_start
    }
}

#[async_trait]
impl CallbackHandler for ToolTrackingHandler {
    async fn on_run_start(&self, run: &RunTree) {
        self.calls.lock().unwrap().push(format!("run_start:{}", run.name));
    }
    
    async fn on_run_end(&self, run: &RunTree) {
        self.calls.lock().unwrap().push(format!("run_end:{}", run.name));
    }
    
    async fn on_run_error(&self, run: &RunTree, error: &str) {
        self.calls.lock().unwrap().push(format!("run_error:{}:{}", run.name, error));
    }
    
    async fn on_chain_start(&self, run: &RunTree, _inputs: &serde_json::Value) {
        self.calls.lock().unwrap().push(format!("chain_start:{}", run.name));
    }
    
    async fn on_chain_end(&self, run: &RunTree, _outputs: &serde_json::Value) {
        self.calls.lock().unwrap().push(format!("chain_end:{}", run.name));
    }
    
    async fn on_chain_error(&self, run: &RunTree, error: &str) {
        self.calls.lock().unwrap().push(format!("chain_error:{}:{}", run.name, error));
    }
    
    async fn on_tool_start(&self, run: &RunTree, tool_name: &str, input: &str) {
        // 记录调用
        self.calls.lock().unwrap().push(format!("tool_start:{}:{}", run.name, tool_name));
        
        // 计数
        let mut count = self.tool_start_count.lock().unwrap();
        *count += 1;
        
        // 记录输入
        self.tool_inputs.lock().unwrap().push((tool_name.to_string(), input.to_string()));
        
        // 记录开始时间（用于计算耗时）
        let start_time = std::time::Instant::now();
        self.tool_durations.lock().unwrap().push((format!("{}_start", tool_name), start_time.elapsed().as_millis() as i64));
    }
    
    async fn on_tool_end(&self, run: &RunTree, output: &str) {
        // 记录调用
        let tool_name = run.name.clone();
        self.calls.lock().unwrap().push(format!("tool_end:{}:{}", run.name, output));
        
        // 计数
        let mut count = self.tool_end_count.lock().unwrap();
        *count += 1;
        
        // 记录输出
        self.tool_outputs.lock().unwrap().push((tool_name.clone(), output.to_string()));
        
        // 记录结束时间
        let end_time = std::time::Instant::now();
        self.tool_durations.lock().unwrap().push((format!("{}_end", tool_name), end_time.elapsed().as_millis() as i64));
    }
    
    async fn on_tool_error(&self, run: &RunTree, error: &str) {
        // 记录调用
        self.calls.lock().unwrap().push(format!("tool_error:{}:{}", run.name, error));
        
        // 计数
        let mut count = self.tool_error_count.lock().unwrap();
        *count += 1;
    }
    
    async fn on_llm_start(&self, run: &RunTree, _messages: &[langchainrust::schema::Message]) {
        self.calls.lock().unwrap().push(format!("llm_start:{}", run.name));
    }
    
    async fn on_llm_end(&self, run: &RunTree, _response: &str) {
        self.calls.lock().unwrap().push(format!("llm_end:{}", run.name));
    }
}

// ============================================================================
// Mock Agent（调用工具）
// ============================================================================

/// 会调用工具的 Mock Agent
/// 用于测试 Agent → Tool → Callback 的完整链路
struct ToolCallingAgent {
    /// 要调用的工具名称（第一次迭代）
    tool_to_call: String,
    /// 工具输入
    tool_input: String,
    /// 最终答案
    final_answer: String,
}

impl ToolCallingAgent {
    fn new(tool_name: &str, tool_input: &str, final_answer: &str) -> Self {
        Self {
            tool_to_call: tool_name.to_string(),
            tool_input: tool_input.to_string(),
            final_answer: final_answer.to_string(),
        }
    }
}

#[async_trait]
impl BaseAgent for ToolCallingAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        // 第一轮：调用工具
        if intermediate_steps.is_empty() {
            return Ok(AgentOutput::Action(AgentAction {
                tool: self.tool_to_call.clone(),
                tool_input: langchainrust::agents::ToolInput::String(self.tool_input.clone()),
                log: format!("我需要使用 {} 工具", self.tool_to_call),
            }));
        }
        
        // 第二轮：返回结果
        let observation = &intermediate_steps[0].observation;
        Ok(AgentOutput::Finish(AgentFinish::new(
            format!("{} (工具结果: {})", self.final_answer, observation),
            format!("使用了 {} 工具", self.tool_to_call),
        )))
    }
}

/// 会调用多个工具的 Mock Agent
/// 用于测试多工具执行的回调追踪
struct MultiToolAgent {
    /// 工具调用列表
    tool_calls: Vec<(String, String)>,
    /// 当前调用索引
    current_index: Arc<Mutex<usize>>,
}

impl MultiToolAgent {
    fn new(tool_calls: Vec<(String, String)>) -> Self {
        Self {
            tool_calls,
            current_index: Arc::new(Mutex::new(0)),
        }
    }
}

#[async_trait]
impl BaseAgent for MultiToolAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        let mut index = self.current_index.lock().unwrap();
        
        if *index < self.tool_calls.len() {
            let (tool_name, tool_input) = &self.tool_calls[*index];
            *index += 1;
            
            return Ok(AgentOutput::Action(AgentAction {
                tool: tool_name.clone(),
                tool_input: langchainrust::agents::ToolInput::String(tool_input.clone()),
                log: format!("调用工具 {}", tool_name),
            }));
        }
        
        // 所有工具调用完成后，返回结果
        let observations = intermediate_steps.iter()
            .map(|s| format!("{}: {}", s.action.tool, s.observation))
            .collect::<Vec<_>>()
            .join("; ");
        
        Ok(AgentOutput::Finish(AgentFinish::new(
            format!("完成所有工具调用。结果: {}", observations),
            String::new(),
        )))
    }
}

/// 会失败的 Mock Agent（工具执行错误）
struct ErrorToolAgent;

#[async_trait]
impl BaseAgent for ErrorToolAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        // 第一轮：尝试调用一个不存在的工具
        if intermediate_steps.is_empty() {
            return Ok(AgentOutput::Action(AgentAction {
                tool: "nonexistent_tool".to_string(),
                tool_input: langchainrust::agents::ToolInput::String("test".to_string()),
                log: "尝试调用不存在的工具".to_string(),
            }));
        }
        
        // 工具失败后，直接返回错误信息
        let error_observation = &intermediate_steps[0].observation;
        Ok(AgentOutput::Finish(AgentFinish::new(
            format!("工具调用失败: {}", error_observation),
            String::new(),
        )))
    }
}

// ============================================================================
// 测试用例
// ============================================================================

/// 测试：单个工具调用触发完整回调链
#[tokio::test]
async fn test_single_tool_callback_chain() {
    // 创建回调处理器
    let handler = Arc::new(ToolTrackingHandler::new());
    
    // 创建回调管理器
    let callbacks = Arc::new(CallbackManager::new().add_handler(handler.clone()));
    
    // 创建工具
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
    ];
    
    // 创建 Agent（会调用 calculator）
    let agent = Arc::new(ToolCallingAgent::new(
        "calculator",
        "{\"expression\": \"10 + 20\"}",
        "计算完成"
    ));
    
    // 创建 Executor
    let executor = AgentExecutor::new(agent, tools)
        .with_callbacks(callbacks);
    
    // 执行
    let result = executor.invoke("计算 10 + 20".to_string()).await;
    
    // 验证执行成功
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("30")); // Calculator 应返回 30
    
    // 验证回调被触发
    assert_eq!(handler.get_tool_start_count(), 1, "on_tool_start 应被调用 1 次");
    assert_eq!(handler.get_tool_end_count(), 1, "on_tool_end 应被调用 1 次");
    assert_eq!(handler.get_tool_error_count(), 0, "不应有错误回调");
    
    // 验证调用顺序
    assert!(handler.verify_call_order(), "回调顺序应正确: start → end");
    
    // 验证输入输出记录
    let inputs = handler.get_tool_inputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].0, "calculator");
    assert!(inputs[0].1.contains("10 + 20"));
    
    let outputs = handler.get_tool_outputs();
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].1.contains("30"));
}

/// 测试：多个工具调用触发多次回调
#[tokio::test]
async fn test_multiple_tools_callback_chain() {
    let handler = Arc::new(ToolTrackingHandler::new());
    let callbacks = Arc::new(CallbackManager::new().add_handler(handler.clone()));
    
    // 创建多个工具
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
        Arc::new(SimpleMathTool::new()),
    ];
    
    // 创建 Agent（会调用 2 个工具）
    let agent = Arc::new(MultiToolAgent::new(vec![
        ("calculator".to_string(), "{\"expression\": \"5 + 5\"}".to_string()),
        ("math".to_string(), "{\"operation\": \"sqrt\", \"value\": 100}".to_string()),
    ]));
    
    let executor = AgentExecutor::new(agent, tools)
        .with_callbacks(callbacks);
    
    let result = executor.invoke("计算多个数学问题".to_string()).await;
    
    // 验证执行成功
    assert!(result.is_ok());
    
    // 验证回调次数
    assert_eq!(handler.get_tool_start_count(), 2, "应调用 2 个工具");
    assert_eq!(handler.get_tool_end_count(), 2);
    assert_eq!(handler.get_tool_error_count(), 0);
    
    // 验证调用顺序
    assert!(handler.verify_call_order(), "多工具回调顺序应正确");
    
    // 验证两个工具都被记录
    let calls = handler.get_calls();
    assert!(calls.iter().any(|c| c.contains("tool_start") && c.contains("calculator")));
    assert!(calls.iter().any(|c| c.contains("tool_start") && c.contains("math")));
}

/// 测试：工具执行失败触发错误回调
#[tokio::test]
async fn test_tool_error_callback() {
    let handler = Arc::new(ToolTrackingHandler::new());
    let callbacks = Arc::new(CallbackManager::new().add_handler(handler.clone()));
    
    // 创建工具（不包含 nonexistent_tool）
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
    ];
    
    // 创建 Agent（尝试调用不存在的工具）
    let agent = Arc::new(ErrorToolAgent);
    
    let executor = AgentExecutor::new(agent, tools)
        .with_max_iterations(2)
        .with_callbacks(callbacks);
    
    let result = executor.invoke("测试错误".to_string()).await;
    
    // 验证执行结果：预期会有错误
    assert!(result.is_err(), "调用不存在的工具应返回错误");
    
    // 验证错误回调被触发（或至少 chain_error）
    let calls = handler.get_calls();
    
    // 应该有 chain 生命周期
    assert!(calls.iter().any(|c| c.starts_with("chain_start")), "应有 chain_start");
    assert!(calls.iter().any(|c| c.starts_with("chain_error")), "应有 chain_error");
    
    // 工具回调可能触发也可能不触发（取决于 AgentExecutor 内部逻辑）
    // 这里验证至少有错误处理流程
    let has_error_handling = calls.iter().any(|c| c.contains("error"));
    assert!(has_error_handling, "应有错误处理回调");
}

/// 测试：工具无效输入触发错误回调
#[tokio::test]
async fn test_tool_invalid_input_callback() {
    let handler = Arc::new(ToolTrackingHandler::new());
    let callbacks = Arc::new(CallbackManager::new().add_handler(handler.clone()));
    
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
    ];
    
    // 创建 Agent（使用无效的 JSON 输入）
    let agent = Arc::new(ToolCallingAgent::new(
        "calculator",
        "invalid json input", // 无效输入
        "计算完成"
    ));
    
    let executor = AgentExecutor::new(agent, tools)
        .with_callbacks(callbacks);
    
    let result = executor.invoke("测试无效输入".to_string()).await;
    
    // 工具可能返回错误信息或处理结果
    // 验证回调被触发
    assert!(handler.get_tool_start_count() >= 1);
    
    // 如果工具成功处理了错误输入，验证 end 被调用
    // 如果工具失败，验证 error 被调用
    let total_callbacks = handler.get_tool_start_count() + handler.get_tool_end_count() + handler.get_tool_error_count();
    assert!(total_callbacks >= 2, "至少应有 start + end/error");
}

/// 测试：回调处理器记录完整的执行流程
#[tokio::test]
async fn test_full_execution_trace() {
    let handler = Arc::new(ToolTrackingHandler::new());
    let callbacks = Arc::new(CallbackManager::new().add_handler(handler.clone()));
    
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
    ];
    
    let agent = Arc::new(ToolCallingAgent::new(
        "calculator",
        "{\"expression\": \"2 * 3\"}",
        "乘法计算完成"
    ));
    
    let executor = AgentExecutor::new(agent, tools)
        .with_callbacks(callbacks);
    
    executor.invoke("计算 2 * 3".to_string()).await.unwrap();
    
    // 验证完整流程
    let calls = handler.get_calls();
    
    // 应包含 chain 生命周期
    assert!(calls.iter().any(|c| c.starts_with("chain_start")), "应有 chain_start");
    assert!(calls.iter().any(|c| c.starts_with("chain_end")), "应有 chain_end");
    
    // 应包含 tool 生命周期
    assert!(calls.iter().any(|c| c.starts_with("tool_start")), "应有 tool_start");
    assert!(calls.iter().any(|c| c.starts_with("tool_end")), "应有 tool_end");
    
    // 验证顺序：chain_start → tool_start → tool_end → chain_end
    let chain_start_idx = calls.iter().position(|c| c.starts_with("chain_start")).unwrap();
    let tool_start_idx = calls.iter().position(|c| c.starts_with("tool_start")).unwrap();
    let tool_end_idx = calls.iter().position(|c| c.starts_with("tool_end")).unwrap();
    let chain_end_idx = calls.iter().position(|c| c.starts_with("chain_end")).unwrap();
    
    assert!(chain_start_idx < tool_start_idx, "chain_start 应在 tool_start 之前");
    assert!(tool_start_idx < tool_end_idx, "tool_start 应在 tool_end 之前");
    assert!(tool_end_idx < chain_end_idx, "tool_end 应在 chain_end 之前");
}

/// 测试：多个回调处理器同时接收工具事件
#[tokio::test]
async fn test_multiple_handlers_receive_tool_events() {
    // 创建两个回调处理器
    let handler1 = Arc::new(ToolTrackingHandler::new());
    let handler2 = Arc::new(ToolTrackingHandler::new());
    
    let callbacks = Arc::new(CallbackManager::new()
        .add_handler(handler1.clone())
        .add_handler(handler2.clone()));
    
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
    ];
    
    let agent = Arc::new(ToolCallingAgent::new(
        "calculator",
        "{\"expression\": \"1 + 1\"}",
        "完成"
    ));
    
    let executor = AgentExecutor::new(agent, tools)
        .with_callbacks(callbacks);
    
    executor.invoke("测试".to_string()).await.unwrap();
    
    // 验证两个处理器都收到了回调
    assert_eq!(handler1.get_tool_start_count(), 1, "handler1 应收到 tool_start");
    assert_eq!(handler2.get_tool_start_count(), 1, "handler2 应收到 tool_start");
    assert_eq!(handler1.get_tool_end_count(), 1, "handler1 应收到 tool_end");
    assert_eq!(handler2.get_tool_end_count(), 1, "handler2 应收到 tool_end");
}

/// 测试：工具输入输出数据完整性
#[tokio::test]
async fn test_tool_input_output_integrity() {
    let handler = Arc::new(ToolTrackingHandler::new());
    let callbacks = Arc::new(CallbackManager::new().add_handler(handler.clone()));
    
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(SimpleMathTool::new()),
    ];
    
    // 使用数学工具进行特定计算
    let agent = Arc::new(ToolCallingAgent::new(
        "math",
        "{\"operation\": \"factorial\", \"value\": 5}",
        "阶乘计算完成"
    ));
    
    let executor = AgentExecutor::new(agent, tools)
        .with_callbacks(callbacks);
    
    executor.invoke("计算 5 的阶乘".to_string()).await.unwrap();
    
    // 验证输入记录
    let inputs = handler.get_tool_inputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].0, "math");
    assert!(inputs[0].1.contains("factorial"));
    assert!(inputs[0].1.contains("5"));
    
    // 验证输出记录
    let outputs = handler.get_tool_outputs();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].0, "math");
    assert!(outputs[0].1.contains("120")); // 5! = 120
}

/// 测试：直接使用 BaseTool.run() 不触发回调（验证回调需要 AgentExecutor）
#[tokio::test]
async fn test_direct_tool_run_no_callback() {
    let handler = Arc::new(ToolTrackingHandler::new());
    let callbacks = Arc::new(CallbackManager::new().add_handler(handler.clone()));
    
    // 直接调用工具（不通过 AgentExecutor）
    let calc = Calculator::new();
    let result = calc.run("{\"expression\": \"3 + 4\"}".to_string()).await.unwrap();
    
    // 验证结果正确
    assert!(result.contains("7"));
    
    // 验证回调未被触发（因为没有通过 AgentExecutor）
    assert_eq!(handler.get_tool_start_count(), 0, "直接调用不应触发回调");
    assert_eq!(handler.get_tool_end_count(), 0);
    
    // 这说明回调系统依赖 AgentExecutor 的 execute_tool 方法
    // 如需独立触发回调，需要添加 run_with_callbacks 方法
}

/// 测试：RunTree 工具子运行正确创建
#[tokio::test]
async fn test_tool_run_tree_hierarchy() {
    let handler = Arc::new(ToolTrackingHandler::new());
    let callbacks = Arc::new(CallbackManager::new().add_handler(handler.clone()));
    
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
    ];
    
    let agent = Arc::new(ToolCallingAgent::new(
        "calculator",
        "{\"expression\": \"100 / 25\"}",
        "除法计算完成"
    ));
    
    let executor = AgentExecutor::new(agent, tools)
        .with_callbacks(callbacks);
    
    executor.invoke("计算 100 / 25".to_string()).await.unwrap();
    
    // 验证回调调用中包含正确的 run 信息
    let calls = handler.get_calls();
    
    // 验证 chain 和 tool 的 run name 都被记录
    let chain_calls = calls.iter().filter(|c| c.contains("AgentExecutor")).count();
    let tool_calls = calls.iter().filter(|c| c.contains("calculator")).count();
    
    assert!(chain_calls >= 2, "应有 chain start/end");
    assert!(tool_calls >= 2, "应有 tool start/end");
}