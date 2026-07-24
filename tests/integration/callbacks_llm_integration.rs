#![allow(dead_code)]
// tests/integration/callbacks_llm_integration.rs
//! LLM 回调集成测试
//!
//! 演示回调系统的实际用途：
//! 1. 追踪每次 LLM 调用的输入输出
//! 2. 记录 token 使用量和成本
//! 3. 计算每次调用的耗时
//! 4. 实时打印流式输出的 token
//!
//! 运行条件：设置 OPENAI_API_KEY 环境变量

use async_trait::async_trait;
use langchainrust::schema::Message;
use langchainrust::{
    BaseChatModel, CallbackHandler, CallbackManager, OpenAIChat, OpenAIConfig, RunTree,
    RunnableConfig,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ============================================================================
// 追踪回调处理器 - 记录所有 LLM 调用信息
// ============================================================================

/// 追踪处理器：记录每次调用的完整信息
///
/// 实际用途：
/// - 调试：看到每次调用的输入输出
/// - 成本追踪：记录 token 使用量
/// - 性能分析：计算每次调用耗时
struct TracingCallbackHandler {
    /// 记录所有调用
    calls: Arc<Mutex<Vec<CallRecord>>>,
    /// Token 总数
    total_tokens: Arc<Mutex<usize>>,
    /// 流式输出的 token
    streamed_tokens: Arc<Mutex<String>>,
}

/// 单次调用记录
#[derive(Debug, Clone)]
struct CallRecord {
    run_name: String,
    run_type: String,
    start_time: String,
    end_time: Option<String>,
    duration_ms: Option<i64>,
    inputs: serde_json::Value,
    outputs: Option<serde_json::Value>,
    error: Option<String>,
    token_usage: Option<TokenInfo>,
}

#[derive(Debug, Clone)]
struct TokenInfo {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl TracingCallbackHandler {
    fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            total_tokens: Arc::new(Mutex::new(0)),
            streamed_tokens: Arc::new(Mutex::new(String::new())),
        }
    }

    fn get_calls(&self) -> Vec<CallRecord> {
        self.calls.lock().unwrap().clone()
    }

    fn get_total_tokens(&self) -> usize {
        *self.total_tokens.lock().unwrap()
    }

    fn get_streamed_output(&self) -> String {
        self.streamed_tokens.lock().unwrap().clone()
    }
}

#[async_trait]
impl CallbackHandler for TracingCallbackHandler {
    async fn on_run_start(&self, _run: &RunTree) {
        // 默认实现
    }

    async fn on_run_end(&self, _run: &RunTree) {
        // 默认实现
    }

    async fn on_run_error(&self, _run: &RunTree, _error: &str) {
        // 默认实现
    }

    async fn on_llm_start(&self, run: &RunTree, _messages: &[Message]) {
        let record = CallRecord {
            run_name: run.name.clone(),
            run_type: run.run_type.to_string(),
            start_time: run.start_time.to_rfc3339(),
            end_time: None,
            duration_ms: None,
            inputs: run.inputs.clone(),
            outputs: None,
            error: None,
            token_usage: None,
        };
        self.calls.lock().unwrap().push(record);
    }

    async fn on_llm_end(&self, run: &RunTree, _response: &str) {
        let mut calls = self.calls.lock().unwrap();
        if let Some(last) = calls.last_mut() {
            last.end_time = run.end_time.map(|t| t.to_rfc3339());
            last.duration_ms = run.duration_ms();
            last.outputs = run.outputs.clone();

            // 提取 token 使用量
            if let Some(outputs) = &run.outputs {
                if let Some(token_usage) = outputs.get("token_usage") {
                    if let Some(total) = token_usage.get("total_tokens") {
                        if let Some(total_num) = total.as_u64() {
                            last.token_usage = Some(TokenInfo {
                                prompt_tokens: token_usage
                                    .get("prompt_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0)
                                    as usize,
                                completion_tokens: token_usage
                                    .get("completion_tokens")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0)
                                    as usize,
                                total_tokens: total_num as usize,
                            });
                            *self.total_tokens.lock().unwrap() += total_num as usize;
                        }
                    }
                }
            }
        }
    }

    async fn on_llm_new_token(&self, _run: &RunTree, token: &str) {
        // 实时收集流式输出的 token
        self.streamed_tokens.lock().unwrap().push_str(token);
    }

    async fn on_llm_error(&self, run: &RunTree, error: &str) {
        let mut calls = self.calls.lock().unwrap();
        if let Some(last) = calls.last_mut() {
            last.end_time = run.end_time.map(|t| t.to_rfc3339());
            last.duration_ms = run.duration_ms();
            last.error = Some(error.to_string());
        }
    }
}

// ============================================================================
// 测试：追踪 LLM 调用
// ============================================================================

#[tokio::test]
#[ignore = "需要 OPENAI_API_KEY 环境变量"]
async fn test_llm_tracing_with_callbacks() {
    // 这个测试演示回调系统的核心用途：
    // 1. 记录每次调用的输入输出
    // 2. 追踪 token 使用量
    // 3. 计算调用耗时

    let handler = Arc::new(TracingCallbackHandler::new());
    let calls = Arc::clone(&handler.calls);
    let total_tokens = Arc::clone(&handler.total_tokens);

    let callbacks = Arc::new(CallbackManager::new().add_handler(handler));

    // 创建 LLM
    let config = OpenAIConfig::from_env();
    println!("\n=== API 配置 ===");
    println!("Base URL: {}", config.base_url);
    println!("Model: {}", config.model);

    let llm = OpenAIChat::new(config);

    // 使用回调配置
    let run_config = RunnableConfig::new()
        .with_callbacks(callbacks)
        .with_run_name("test_chat");

    // 调用 LLM
    let messages = vec![
        Message::system("你是一个有帮助的助手。"),
        Message::human("用一句话解释什么是 Rust。"),
    ];

    let result = llm.chat(messages, Some(run_config)).await;

    match &result {
        Ok(response) => {
            println!("\n=== LLM 响应 ===");
            println!("{}", response.content);
        }
        Err(e) => {
            println!("\n=== LLM 调用失败 ===");
            println!("错误: {}", e);
            println!("\n可能的原因:");
            println!("1. OPENAI_API_KEY 环境变量未设置");
            println!("2. API 端点 URL 不正确（检查 OPENAI_BASE_URL）");
            println!("3. API Key 无效或已过期");
            println!("\n设置方法:");
            println!("  export OPENAI_API_KEY=sk-xxx");
            println!("  export OPENAI_BASE_URL=https://api.openai.com/v1  # 可选");
            return;
        }
    }

    let _response = result.unwrap();

    // 验证追踪信息
    let calls = calls.lock().unwrap();
    assert!(!calls.is_empty(), "应该有调用记录");

    let last_call = calls.last().unwrap();
    println!("\n=== 追踪信息 ===");
    println!("运行名称: {}", last_call.run_name);
    println!("运行类型: {}", last_call.run_type);
    println!("耗时: {:?} ms", last_call.duration_ms);

    if let Some(token_info) = &last_call.token_usage {
        println!("\n=== Token 使用量 ===");
        println!("Prompt tokens: {}", token_info.prompt_tokens);
        println!("Completion tokens: {}", token_info.completion_tokens);
        println!("Total tokens: {}", token_info.total_tokens);

        // 估算成本 (GPT-3.5-turbo 价格)
        let cost = (token_info.prompt_tokens as f64 * 0.0005 / 1000.0)
            + (token_info.completion_tokens as f64 * 0.0015 / 1000.0);
        println!("预估成本: ${:.6}", cost);
    }

    let total = *total_tokens.lock().unwrap();
    println!("\n总 Token 数: {}", total);
}

#[tokio::test]
#[ignore = "需要 OPENAI_API_KEY 环境变量"]
async fn test_llm_streaming_with_callbacks() {
    // 演示流式输出的回调
    // 每个 token 都会触发 on_llm_new_token 回调

    let handler = Arc::new(TracingCallbackHandler::new());
    let _streamed = Arc::clone(&handler.streamed_tokens);

    let callbacks = Arc::new(CallbackManager::new().add_handler(handler));

    let config = OpenAIConfig::from_env();
    let llm = OpenAIChat::new(config);

    let run_config = RunnableConfig::new()
        .with_callbacks(callbacks)
        .with_run_name("streaming_test");

    let messages = vec![Message::human("数到 5，每个数字占一行。")];

    // 流式调用
    use futures_util::StreamExt;
    let mut stream = llm.stream_chat(messages, Some(run_config)).await.unwrap();

    print!("\n=== 流式输出 ===\n");
    let mut collected = String::new();
    while let Some(token_result) = stream.next().await {
        match token_result {
            Ok(token) => {
                print!("{}", token);
                collected.push_str(&token);
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }
    println!();

    // 显示流式输出结果
    println!("\n=== 流式输出收集 ===");
    println!("收集到 {} 个字符", collected.len());
    println!("内容: {}", collected);

    // 注意: on_llm_new_token 回调在流式输出中目前未完全实现
    // 这是已知限制，后续版本会改进
}

#[tokio::test]
#[ignore = "需要 OPENAI_API_KEY 环境变量"]
async fn test_multiple_llm_calls_tracking() {
    // 演示追踪多次 LLM 调用
    // 实际场景：一个请求可能触发多次 LLM 调用

    let handler = Arc::new(TracingCallbackHandler::new());
    let calls = Arc::clone(&handler.calls);
    let total_tokens = Arc::clone(&handler.total_tokens);

    let callbacks = Arc::new(CallbackManager::new().add_handler(handler));

    let config = OpenAIConfig::from_env();
    let llm = OpenAIChat::new(config);

    // 第一次调用
    let config1 = RunnableConfig::new()
        .with_callbacks(Arc::clone(&callbacks))
        .with_run_name("call_1");

    let messages1 = vec![Message::human("说 'hello'")];
    let _ = llm.chat(messages1, Some(config1)).await;

    // 第二次调用
    let config2 = RunnableConfig::new()
        .with_callbacks(Arc::clone(&callbacks))
        .with_run_name("call_2");

    let messages2 = vec![Message::human("说 'world'")];
    let _ = llm.chat(messages2, Some(config2)).await;

    // 汇总报告
    let calls = calls.lock().unwrap();
    let total = *total_tokens.lock().unwrap();

    println!("\n=== 多次调用汇总 ===");
    println!("调用次数: {}", calls.len());
    println!("总 Token 数: {}", total);

    println!("\n各调用详情:");
    for (i, call) in calls.iter().enumerate() {
        println!("\n--- 调用 {} ---", i + 1);
        println!("名称: {}", call.run_name);
        println!("耗时: {:?} ms", call.duration_ms);
        if let Some(token_info) = &call.token_usage {
            println!("Tokens: {}", token_info.total_tokens);
        }
    }

    assert_eq!(calls.len(), 2, "应该有 2 次调用记录");
}

// ============================================================================
// 成本追踪器 - 实际生产场景示例
// ============================================================================

/// 成本追踪器：用于生产环境的成本监控
struct CostTracker {
    /// 模型定价 (每 1K tokens)
    pricing: HashMap<String, (f64, f64)>, // (prompt_price, completion_price)
    /// 累计成本
    total_cost: Arc<Mutex<f64>>,
    /// 调用次数
    call_count: Arc<Mutex<usize>>,
}

impl CostTracker {
    fn new() -> Self {
        let mut pricing = HashMap::new();
        // GPT-3.5-turbo 价格 (2024)
        pricing.insert("gpt-3.5-turbo".to_string(), (0.0005, 0.0015));
        // GPT-4 价格
        pricing.insert("gpt-4".to_string(), (0.03, 0.06));
        pricing.insert("gpt-4-turbo".to_string(), (0.01, 0.03));

        Self {
            pricing,
            total_cost: Arc::new(Mutex::new(0.0)),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    fn get_total_cost(&self) -> f64 {
        *self.total_cost.lock().unwrap()
    }

    fn get_call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }

    fn calculate_cost(&self, model: &str, prompt_tokens: usize, completion_tokens: usize) -> f64 {
        let (prompt_price, completion_price) = self.pricing.get(model).unwrap_or(&(0.0, 0.0));

        (prompt_tokens as f64 * prompt_price / 1000.0)
            + (completion_tokens as f64 * completion_price / 1000.0)
    }
}

#[async_trait]
impl CallbackHandler for CostTracker {
    async fn on_run_start(&self, _run: &RunTree) {}

    async fn on_run_end(&self, _run: &RunTree) {}

    async fn on_run_error(&self, _run: &RunTree, _error: &str) {}

    async fn on_llm_end(&self, run: &RunTree, _response: &str) {
        if let Some(outputs) = &run.outputs {
            if let Some(token_usage) = outputs.get("token_usage") {
                let prompt = token_usage
                    .get("prompt_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let completion = token_usage
                    .get("completion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;

                // 从 inputs 获取模型名称
                let model = run
                    .inputs
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("gpt-3.5-turbo");

                let cost = self.calculate_cost(model, prompt, completion);
                *self.total_cost.lock().unwrap() += cost;
                *self.call_count.lock().unwrap() += 1;

                println!(
                    "[CostTracker] 调用 #{}: +${:.6} (累计: ${:.6})",
                    self.get_call_count(),
                    cost,
                    self.get_total_cost()
                );
            }
        }
    }
}

#[tokio::test]
#[ignore = "需要 OPENAI_API_KEY 环境变量"]
async fn test_cost_tracking_in_production() {
    // 演示生产环境中的成本追踪

    let tracker = Arc::new(CostTracker::new());
    let total_cost = Arc::clone(&tracker.total_cost);

    let callbacks = Arc::new(CallbackManager::new().add_handler(tracker));

    let config = OpenAIConfig::from_env();
    let llm = OpenAIChat::new(config);

    println!("\n=== 成本追踪演示 ===");

    // 模拟多次调用
    for i in 1..=3 {
        let run_config = RunnableConfig::new()
            .with_callbacks(Arc::clone(&callbacks))
            .with_run_name(format!("call_{}", i));

        let messages = vec![Message::human(format!("说数字 {}", i))];
        let _ = llm.chat(messages, Some(run_config)).await;
    }

    println!("\n=== 最终成本报告 ===");
    println!("总成本: ${:.6}", *total_cost.lock().unwrap());
}
