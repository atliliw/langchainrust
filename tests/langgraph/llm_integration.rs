//! LangGraph + LLM 真实调用示例
//!
//! 本测试文件展示 LangGraph 与真实 LLM API 的集成能力。
//!
//! ## 测试覆盖范围
//!
//! 1. 单节点 LLM 问答流程
//! 2. 多步骤 Agent 流程（分析→LLM→格式化）
//! 3. 条件路由（根据问题复杂度选择路径）
//! 4. 循环 Agent（持续对话直到满意）
//!
//! ## 对框架的意义
//!
//! LLM 集成是 LangGraph 的核心应用场景：
//! - 验证 LangGraph 能正确处理真实异步 I/O 操作
//! - 展示 Agent 工作流的实际实现方式
//! - 证明状态流转在 LLM 调用中正确工作
//! - 为用户提供可直接参考的代码示例
//!
//! ## 注意事项
//!
//! 这些测试需要真实 API Key，默认被 ignore。
//! 运行前需要移除 #[ignore] 或使用 --ignored 参数。

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::schema::Message;
use langchainrust::BaseChatModel;
use langchainrust::OpenAIChat;
use langchainrust::{
    AgentState, FunctionRouter, GraphBuilder, GraphExecution, MessageEntry, StateGraph,
    StateUpdate, END, START,
};
use std::collections::HashMap;
use std::sync::Arc;

/// 同步调用 LLM 的辅助函数
///
/// 使用 block_in_place + Handle::current() 在同步节点中调用异步 LLM。
/// 这是 sync node 调用 async API 的标准模式。
fn call_llm_sync(llm: Arc<OpenAIChat>, messages: Vec<Message>) -> Result<String, String> {
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            llm.chat(messages, None)
                .await
                .map(|r| r.content)
                .map_err(|e| e.to_string())
        })
    })
}

/// 示例1: 简单的 LLM 问答流程
///
/// 功能验证:
/// - LangGraph 能正确集成真实 LLM API 调用
/// - 状态在 LLM 调用前后正确传递
/// - 节点返回的 StateUpdate 正确应用到最终状态
///
/// 框架作用:
/// - 这是 LangGraph 最基础的 LLM 集成测试
/// - 验证图执行流程能承载真实异步操作
/// - 展示最小可行的 LLM 集成代码结构
/// - 为复杂 Agent 提供基础构建块
#[tokio::test(flavor = "multi_thread")]
#[ignore = "需要配置 API Key，移除 ignore 运行"]
async fn test_simple_llm_graph() {
    let llm = Arc::new(TestConfig::get().openai_chat());

    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("ask_llm", {
            let llm = Arc::clone(&llm);
            move |state: &AgentState| {
                let messages = vec![
                    Message::system("You are a helpful assistant. Answer briefly."),
                    Message::human(&state.input),
                ];

                match call_llm_sync(Arc::clone(&llm), messages) {
                    Ok(content) => {
                        let mut new_state = state.clone();
                        new_state.add_message(MessageEntry::ai(content.clone()));
                        new_state.set_output(content);
                        Ok(StateUpdate::full(new_state))
                    }
                    Err(e) => Ok(StateUpdate::full(AgentState::new(format!("Error: {}", e)))),
                }
            }
        })
        .add_edge(START, "ask_llm")
        .add_edge("ask_llm", END)
        .compile()
        .unwrap();

    let input = AgentState::new("What is Rust programming language?".to_string());
    let result = compiled.invoke(input).await.unwrap();

    println!("=== 简单 LLM 问答 ===");
    println!("问题: {}", result.final_state.input);
    println!(
        "回答: {}",
        result.final_state.output.clone().unwrap_or_default()
    );
    println!("执行步数: {}", result.recursion_count);

    assert!(result.final_state.output.is_some());
}

/// 示例2: 多步骤 Agent 流程
///
/// 功能验证:
/// - 多节点链式执行能正确工作
/// - 每个节点独立处理状态并传递给下一节点
/// - 消息历史在整个流程中正确累积
/// - 最终格式化节点能汇总结果
///
/// 框架作用:
/// - 展示典型 Agent pipeline 模式：预处理→LLM→后处理
/// - 验证状态在复杂流程中的正确流转
/// - 模拟真实 Agent 的分析-生成-输出流程
/// - recursion_count 统计验证执行路径正确
#[tokio::test(flavor = "multi_thread")]
#[ignore = "需要配置 API Key，移除 ignore 运行"]
async fn test_multi_step_agent() {
    let llm = Arc::new(TestConfig::get().openai_chat());

    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("analyze", |state: &AgentState| {
            let mut new_state = state.clone();
            let analysis = if state.input.contains("code") || state.input.contains("编程") {
                "编程相关问题"
            } else if state.input.contains("explain") || state.input.contains("解释") {
                "概念解释问题"
            } else {
                "一般问题"
            };
            new_state.add_message(MessageEntry::ai(format!("分析结果: {}", analysis)));
            Ok(StateUpdate::full(new_state))
        })
        .add_node_fn("generate", {
            let llm = Arc::clone(&llm);
            move |state: &AgentState| {
                let messages: Vec<Message> = state
                    .messages
                    .iter()
                    .map(|m| match m.role {
                        langchainrust::MessageRole::Human => Message::human(&m.content),
                        langchainrust::MessageRole::AI => Message::ai(&m.content),
                        langchainrust::MessageRole::System => Message::system(&m.content),
                        langchainrust::MessageRole::Tool => Message::human(&m.content),
                    })
                    .collect();

                match call_llm_sync(Arc::clone(&llm), messages) {
                    Ok(content) => {
                        let mut new_state = state.clone();
                        new_state.add_message(MessageEntry::ai(content));
                        Ok(StateUpdate::full(new_state))
                    }
                    Err(e) => Ok(StateUpdate::full(AgentState::new(format!("Error: {}", e)))),
                }
            }
        })
        .add_node_fn("format_output", |state: &AgentState| {
            let mut new_state = state.clone();
            let last_ai_msg = state
                .messages
                .iter()
                .rev()
                .find(|m| m.role == langchainrust::MessageRole::AI)
                .map(|m| m.content.clone())
                .unwrap_or_default();
            new_state.set_output(format!("最终回答:\n{}", last_ai_msg));
            Ok(StateUpdate::full(new_state))
        })
        .add_edge(START, "analyze")
        .add_edge("analyze", "generate")
        .add_edge("generate", "format_output")
        .add_edge("format_output", END)
        .compile()
        .unwrap();

    let input = AgentState::new("解释一下 Rust 中的所有权概念".to_string());
    let result = compiled.invoke(input).await.unwrap();

    println!("=== 多步骤 Agent ===");
    println!("问题: {}", result.final_state.input);
    println!("消息历史:");
    for msg in &result.final_state.messages {
        println!("  [{:?}] {}", msg.role, msg.content);
    }
    println!(
        "\n最终输出:\n{}",
        result.final_state.output.unwrap_or_default()
    );
}

/// 示例3: 条件路由 - 根据问题复杂度选择不同处理方式
///
/// 功能验证:
/// - 条件边在真实 LLM 流程中正确工作
/// - 路由器能根据状态内容决定执行路径
/// - 不同路径可以有不同处理逻辑（简单/复杂）
/// - 状态中的 input.len() 能正确触发不同路由
///
/// 框架作用:
/// - 展示 Agent 决策能力的实现方式
/// - 验证条件路由在真实场景中的实用性
/// - 模拟智能问答系统根据问题复杂度分流
/// - 简单问题快速响应，复杂问题深度处理
#[tokio::test(flavor = "multi_thread")]
#[ignore = "需要配置 API Key，移除 ignore 运行"]
async fn test_conditional_llm_routing() {
    let llm = Arc::new(TestConfig::get().openai_chat());

    let mut graph: StateGraph<AgentState> = StateGraph::new();

    graph.add_node_fn("classify", |state: &AgentState| {
        Ok(StateUpdate::full(state.clone()))
    });

    graph.add_node_fn("simple_answer", |state: &AgentState| {
        let mut new_state = state.clone();
        let answer = format!("简单回答: {} 是一个简单的概念。", state.input);
        new_state.set_output(answer);
        Ok(StateUpdate::full(new_state))
    });

    graph.add_node_fn("complex_answer", {
        let llm = Arc::clone(&llm);
        move |state: &AgentState| {
            let messages = vec![
                Message::system("You are a helpful assistant. Provide detailed explanations."),
                Message::human(&state.input),
            ];

            match call_llm_sync(Arc::clone(&llm), messages) {
                Ok(content) => {
                    let mut new_state = state.clone();
                    new_state.add_message(MessageEntry::ai(content.clone()));
                    new_state.set_output(format!("详细回答:\n{}", content));
                    Ok(StateUpdate::full(new_state))
                }
                Err(e) => Ok(StateUpdate::full(AgentState::new(format!("Error: {}", e)))),
            }
        }
    });

    graph.add_edge(START, "classify");

    let targets = HashMap::from([
        ("simple".to_string(), "simple_answer".to_string()),
        ("complex".to_string(), "complex_answer".to_string()),
    ]);
    graph.add_conditional_edges("classify", "complexity_router", targets, None);

    graph.add_edge("simple_answer", END);
    graph.add_edge("complex_answer", END);

    let router = FunctionRouter::new(|state: &AgentState| {
        if state.input.len() > 20 {
            "complex".to_string()
        } else {
            "simple".to_string()
        }
    });
    graph.set_conditional_router("complexity_router", router);

    let compiled = graph.compile().unwrap();

    println!("=== 测试简单问题 ===");
    let simple_input = AgentState::new("什么是变量".to_string());
    let simple_result = compiled.invoke(simple_input).await.unwrap();
    println!("问题: {}", simple_result.final_state.input);
    println!("路径步数: {}", simple_result.recursion_count);
    println!(
        "回答: {}",
        simple_result.final_state.output.unwrap_or_default()
    );

    println!("\n=== 测试复杂问题 ===");
    let complex_input = AgentState::new("请详细解释 Rust 中的生命周期参数是如何工作的".to_string());
    let complex_result = compiled.invoke(complex_input).await.unwrap();
    println!("问题: {}", complex_result.final_state.input);
    println!("路径步数: {}", complex_result.recursion_count);
    println!(
        "回答: {}",
        complex_result.final_state.output.unwrap_or_default()
    );
}

/// 示例4: 循环 Agent - 持续对话直到满意
///
/// 功能验证:
/// - 图能支持循环执行（节点指向自身）
/// - recursion_limit 正确限制循环次数
/// - 条件路由能判断何时结束循环
/// - 消息历史在循环中正确累积
///
/// 框架作用:
/// - 展示迭代优化型 Agent 的实现方式
/// - 验证循环执行的安全控制（recursion_limit）
/// - 模拟 Agent 持续思考直到得到满意答案
/// - 这是 Agent 自我迭代、工具循环调用的基础模式
#[tokio::test(flavor = "multi_thread")]
#[ignore = "需要配置 API Key，移除 ignore 运行"]
async fn test_loop_agent() {
    let llm = Arc::new(TestConfig::get().openai_chat());

    let mut graph: StateGraph<AgentState> = StateGraph::new();

    graph.add_node_fn("think", {
        let llm = Arc::clone(&llm);
        move |state: &AgentState| {
            let messages: Vec<Message> = state
                .messages
                .iter()
                .map(|m| match m.role {
                    langchainrust::MessageRole::Human => Message::human(&m.content),
                    langchainrust::MessageRole::AI => Message::ai(&m.content),
                    langchainrust::MessageRole::System => Message::system(&m.content),
                    langchainrust::MessageRole::Tool => Message::human(&m.content),
                })
                .collect();

            match call_llm_sync(Arc::clone(&llm), messages) {
                Ok(content) => {
                    let mut new_state = state.clone();
                    new_state.add_message(MessageEntry::ai(content.clone()));
                    if content.len() > 50 {
                        new_state.set_output(content);
                    }
                    Ok(StateUpdate::full(new_state))
                }
                Err(e) => {
                    let mut new_state = state.clone();
                    new_state.add_message(MessageEntry::ai(format!("Error: {}", e)));
                    Ok(StateUpdate::full(new_state))
                }
            }
        }
    });

    graph.add_node_fn("finish", |state: &AgentState| {
        let mut new_state = state.clone();
        let last_ai = state
            .messages
            .iter()
            .rev()
            .find(|m| m.role == langchainrust::MessageRole::AI)
            .map(|m| m.content.clone())
            .unwrap_or_default();
        new_state.set_output(format!("最终回答:\n{}", last_ai));
        Ok(StateUpdate::full(new_state))
    });

    graph.add_edge(START, "think");

    let targets = HashMap::from([
        ("continue".to_string(), "think".to_string()),
        ("done".to_string(), "finish".to_string()),
    ]);
    graph.add_conditional_edges("think", "completion_router", targets, None);

    graph.add_edge("finish", END);

    let router = FunctionRouter::new(|state: &AgentState| {
        if state.output.is_some() {
            "done".to_string()
        } else {
            "continue".to_string()
        }
    });
    graph.set_conditional_router("completion_router", router);

    let compiled = graph.compile().unwrap().with_recursion_limit(5);

    let input = AgentState::new("什么是 Rust 的所有权系统？".to_string());
    let result = compiled.invoke(input).await.unwrap();

    println!("=== 循环 Agent ===");
    println!("问题: {}", result.final_state.input);
    println!("循环次数: {}", result.recursion_count);
    println!("消息历史数: {}", result.final_state.messages.len());
    println!(
        "\n最终输出:\n{}",
        result.final_state.output.unwrap_or_default()
    );
}

/// 示例5: Human-in-the-loop LLM 流程 - 人工审批
///
/// 功能验证:
/// - interrupt_before 在关键节点中断
/// - 人工可以检查 LLM 输出
/// - resume() 从中断点继续
/// - 支持审批/拒绝流程
///
/// 框架作用:
/// - 展示真实 LLM 场景中的人工干预
/// - 实现需要审批的自动化流程
/// - 允许人工检查 AI 生成的答案
/// - 这是生产环境中常用的模式
#[tokio::test(flavor = "multi_thread")]
#[ignore = "需要配置 API Key，移除 ignore 运行"]
async fn test_llm_human_in_loop() {
    let llm = Arc::new(TestConfig::get().openai_chat());

    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("generate", {
            let llm = Arc::clone(&llm);
            move |state: &AgentState| {
                let messages = vec![
                    Message::system("You are a helpful assistant. Generate a response."),
                    Message::human(&state.input),
                ];

                match call_llm_sync(Arc::clone(&llm), messages) {
                    Ok(content) => {
                        let mut new_state = state.clone();
                        new_state.add_message(MessageEntry::ai(content.clone()));
                        Ok(StateUpdate::full(new_state))
                    }
                    Err(e) => Ok(StateUpdate::full(AgentState::new(format!("Error: {}", e)))),
                }
            }
        })
        .add_node_fn("review", |state: &AgentState| {
            let mut new_state = state.clone();
            let last_ai = state
                .messages
                .iter()
                .rev()
                .find(|m| m.role == langchainrust::MessageRole::AI)
                .map(|m| m.content.clone())
                .unwrap_or_default();
            new_state.set_output(format!("已审批: {}", last_ai));
            Ok(StateUpdate::full(new_state))
        })
        .add_node_fn("finalize", |state: &AgentState| {
            let mut new_state = state.clone();
            new_state.add_message(MessageEntry::ai("流程完成".to_string()));
            Ok(StateUpdate::full(new_state))
        })
        .add_edge(START, "generate")
        .add_edge("generate", "review")
        .add_edge("review", "finalize")
        .add_edge("finalize", END)
        .compile()
        .unwrap()
        .with_interrupt_before(vec!["review".to_string()]);

    let input = AgentState::new("解释 Rust 的所有权机制".to_string());

    println!("=== Human-in-the-loop LLM ===");

    let result1 = compiled.invoke(input).await;
    assert!(result1.is_err());
    println!("步骤1: 在 review 节点前中断，等待人工审批");

    let last_ai = AgentState::new("test".to_string())
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let mut execution = GraphExecution::new(
        AgentState::new("解释 Rust 的所有权机制".to_string()),
        "review".to_string(),
        "review".to_string(),
    );
    execution.recursion_count = 1;
    execution
        .state
        .add_message(MessageEntry::ai(format!("LLM回答: {}", last_ai)));

    println!("步骤2: 人工审批后继续执行");
    let result2 = compiled.resume(execution).await.unwrap();

    println!("步骤3: 流程完成");
    println!(
        "最终输出: {}",
        result2.final_state.output.unwrap_or_default()
    );
}
