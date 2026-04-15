//! LangGraph + LLM 真实调用示例

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::{
    StateGraph, GraphBuilder, START, END,
    AgentState, StateUpdate, MessageEntry,
    FunctionRouter,
};
use langchainrust::schema::Message;
use langchainrust::OpenAIChat;
use langchainrust::BaseChatModel;
use std::collections::HashMap;
use std::sync::Arc;

fn call_llm_sync(llm: Arc<OpenAIChat>, messages: Vec<Message>) -> Result<String, String> {
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            llm.chat(messages, None).await
                .map(|r| r.content)
                .map_err(|e| e.to_string())
        })
    })
}

/// 示例1: 简单的 LLM 问答流程
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
                    Err(e) => Ok(StateUpdate::full(AgentState::new(format!("Error: {}", e))))
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
    println!("回答: {}", result.final_state.output.clone().unwrap_or_default());
    println!("执行步数: {}", result.recursion_count);
    
    assert!(result.final_state.output.is_some());
}

/// 示例2: 多步骤 Agent 流程
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
                let messages: Vec<Message> = state.messages.iter().map(|m| {
                    match m.role {
                        langchainrust::MessageRole::Human => Message::human(&m.content),
                        langchainrust::MessageRole::AI => Message::ai(&m.content),
                        langchainrust::MessageRole::System => Message::system(&m.content),
                        langchainrust::MessageRole::Tool => Message::human(&m.content),
                    }
                }).collect();
                
                match call_llm_sync(Arc::clone(&llm), messages) {
                    Ok(content) => {
                        let mut new_state = state.clone();
                        new_state.add_message(MessageEntry::ai(content));
                        Ok(StateUpdate::full(new_state))
                    }
                    Err(e) => Ok(StateUpdate::full(AgentState::new(format!("Error: {}", e))))
                }
            }
        })
        .add_node_fn("format_output", |state: &AgentState| {
            let mut new_state = state.clone();
            let last_ai_msg = state.messages.iter()
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
    println!("\n最终输出:\n{}", result.final_state.output.unwrap_or_default());
}

/// 示例3: 条件路由 - 根据问题复杂度选择不同处理方式
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
                Err(e) => Ok(StateUpdate::full(AgentState::new(format!("Error: {}", e))))
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
        if state.input.len() > 20 { "complex".to_string() } else { "simple".to_string() }
    });
    graph.set_conditional_router("complexity_router", router);
    
    let compiled = graph.compile().unwrap();
    
    println!("=== 测试简单问题 ===");
    let simple_input = AgentState::new("什么是变量".to_string());
    let simple_result = compiled.invoke(simple_input).await.unwrap();
    println!("问题: {}", simple_result.final_state.input);
    println!("路径步数: {}", simple_result.recursion_count);
    println!("回答: {}", simple_result.final_state.output.unwrap_or_default());
    
    println!("\n=== 测试复杂问题 ===");
    let complex_input = AgentState::new("请详细解释 Rust 中的生命周期参数是如何工作的".to_string());
    let complex_result = compiled.invoke(complex_input).await.unwrap();
    println!("问题: {}", complex_result.final_state.input);
    println!("路径步数: {}", complex_result.recursion_count);
    println!("回答: {}", complex_result.final_state.output.unwrap_or_default());
}

/// 示例4: 循环 Agent - 持续对话直到满意
#[tokio::test(flavor = "multi_thread")]
#[ignore = "需要配置 API Key，移除 ignore 运行"]
async fn test_loop_agent() {
    let llm = Arc::new(TestConfig::get().openai_chat());
    
    let mut graph: StateGraph<AgentState> = StateGraph::new();
    
    graph.add_node_fn("think", {
        let llm = Arc::clone(&llm);
        move |state: &AgentState| {
            let messages: Vec<Message> = state.messages.iter().map(|m| {
                match m.role {
                    langchainrust::MessageRole::Human => Message::human(&m.content),
                    langchainrust::MessageRole::AI => Message::ai(&m.content),
                    langchainrust::MessageRole::System => Message::system(&m.content),
                    langchainrust::MessageRole::Tool => Message::human(&m.content),
                }
            }).collect();
            
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
        let last_ai = state.messages.iter()
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
        if state.output.is_some() { "done".to_string() } else { "continue".to_string() }
    });
    graph.set_conditional_router("completion_router", router);
    
    let compiled = graph.compile().unwrap()
        .with_recursion_limit(5);
    
    let input = AgentState::new("什么是 Rust 的所有权系统？".to_string());
    let result = compiled.invoke(input).await.unwrap();
    
    println!("=== 循环 Agent ===");
    println!("问题: {}", result.final_state.input);
    println!("循环次数: {}", result.recursion_count);
    println!("消息历史数: {}", result.final_state.messages.len());
    println!("\n最终输出:\n{}", result.final_state.output.unwrap_or_default());
}