use langchainrust::{
    StateGraph, START, END,
    AgentState, StateUpdate, MessageEntry,
    FunctionRouter,
};
use std::collections::HashMap;

// 示例: 智能问答处理流程
// 展示 LangGraph 的分支路由、状态流转、消息记录功能

#[tokio::test]
async fn example_smart_qa_pipeline() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();
    
    // 节点1: 接收用户问题
    graph.add_node_fn("receive_question", |state: &AgentState| {
        let mut new_state = state.clone();
        new_state.add_message(MessageEntry::ai(format!("收到问题: {}", state.input)));
        Ok(StateUpdate::full(new_state))
    });
    
    // 节点2: 分析问题复杂度
    graph.add_node_fn("analyze_complexity", |state: &AgentState| {
        let is_complex = state.input.len() > 20;
        let mut new_state = state.clone();
        new_state.add_message(MessageEntry::ai(
            format!("分析完成 - 问题类型: {}", if is_complex { "复杂" } else { "简单" })
        ));
        Ok(StateUpdate::full(new_state))
    });
    
    // 节点3a: 简单问题快速回答
    graph.add_node_fn("simple_answer", |state: &AgentState| {
        let mut new_state = state.clone();
        let answer = format!("快速回答: {}", state.input);
        new_state.set_output(answer);
        Ok(StateUpdate::full(new_state))
    });
    
    // 节点3b: 复杂问题详细分析
    graph.add_node_fn("complex_answer", |state: &AgentState| {
        let mut new_state = state.clone();
        new_state.add_message(MessageEntry::ai("正在深度分析...".to_string()));
        let answer = format!("详细分析: {}", state.input);
        new_state.set_output(answer);
        Ok(StateUpdate::full(new_state))
    });
    
    // 节点4: 最终处理
    graph.add_node_fn("finalize", |state: &AgentState| {
        let mut new_state = state.clone();
        new_state.add_message(MessageEntry::ai(format!("处理完成，共 {} 条消息", state.messages.len())));
        Ok(StateUpdate::full(new_state))
    });
    
    // 构建边
    graph.add_edge(START, "receive_question");
    graph.add_edge("receive_question", "analyze_complexity");
    
    // 条件边: 根据复杂度选择路径
    let targets = HashMap::from([
        ("simple".to_string(), "simple_answer".to_string()),
        ("complex".to_string(), "complex_answer".to_string()),
    ]);
    graph.add_conditional_edges("analyze_complexity", "complexity_router", targets, None);
    
    graph.add_edge("simple_answer", "finalize");
    graph.add_edge("complex_answer", "finalize");
    graph.add_edge("finalize", END);
    
    // 路由器
    let router = FunctionRouter::new(|state: &AgentState| {
        if state.input.len() <= 20 { "simple" } else { "complex" }.to_string()
    });
    graph.set_conditional_router("complexity_router", router);
    
    let compiled = graph.compile().unwrap();
    
    // 测试简单问题
    let simple_result = compiled.invoke(AgentState::new("什么是 Rust?".to_string())).await.unwrap();
    assert_eq!(simple_result.recursion_count, 4);
    let output = simple_result.final_state.output.as_ref().unwrap();
    assert!(output.contains("快速回答"));
    println!("=== 简单问题 ===");
    println!("输出: {}", output);
    println!("消息数: {}", simple_result.final_state.messages.len());
    
    // 测试复杂问题
    let complex_result = compiled.invoke(AgentState::new("请详细解释所有权系统和生命周期概念?".to_string())).await.unwrap();
    let output2 = complex_result.final_state.output.as_ref().unwrap();
    assert!(output2.contains("详细分析"));
    println!("=== 复杂问题 ===");
    println!("输出: {}", output2);
}

// 示例: 三分支路由系统

#[tokio::test]
async fn example_multi_branch_routing() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();
    
    graph.add_node_fn("classify_input", |state: &AgentState| {
        Ok(StateUpdate::full(state.clone()))
    });
    graph.add_node_fn("math_handler", |state: &AgentState| {
        let mut s = state.clone();
        s.set_output(format!("数学解答: {}", state.input));
        Ok(StateUpdate::full(s))
    });
    graph.add_node_fn("text_handler", |state: &AgentState| {
        let mut s = state.clone();
        s.set_output(format!("文本解答: {}", state.input));
        Ok(StateUpdate::full(s))
    });
    graph.add_node_fn("code_handler", |state: &AgentState| {
        let mut s = state.clone();
        s.set_output(format!("代码解答: {}", state.input));
        Ok(StateUpdate::full(s))
    });
    graph.add_node_fn("default_handler", |state: &AgentState| {
        let mut s = state.clone();
        s.set_output(format!("通用解答: {}", state.input));
        Ok(StateUpdate::full(s))
    });
    
    graph.add_edge(START, "classify_input");
    
    let targets = HashMap::from([
        ("math".to_string(), "math_handler".to_string()),
        ("text".to_string(), "text_handler".to_string()),
        ("code".to_string(), "code_handler".to_string()),
    ]);
    graph.add_conditional_edges("classify_input", "type_router", targets, Some("default_handler".to_string()));
    
    graph.add_edge("math_handler", END);
    graph.add_edge("text_handler", END);
    graph.add_edge("code_handler", END);
    graph.add_edge("default_handler", END);
    
    let router = FunctionRouter::new(|state: &AgentState| {
        let input = state.input.to_lowercase();
        if input.contains("计算") || input.contains("数学") { "math" }
        else if input.contains("翻译") || input.contains("写作") { "text" }
        else if input.contains("代码") || input.contains("编程") { "code" }
        else { "unknown" }
    }.to_string());
    graph.set_conditional_router("type_router", router);
    
    let compiled = graph.compile().unwrap();
    
    // 测试四种类型
    let math_result = compiled.invoke(AgentState::new("帮我计算 1+1".to_string())).await.unwrap();
    assert!(math_result.final_state.output.as_ref().unwrap().contains("数学解答"));
    
    let text_result = compiled.invoke(AgentState::new("请翻译这段文字".to_string())).await.unwrap();
    assert!(text_result.final_state.output.as_ref().unwrap().contains("文本解答"));
    
    let code_result = compiled.invoke(AgentState::new("写一段排序代码".to_string())).await.unwrap();
    assert!(code_result.final_state.output.as_ref().unwrap().contains("代码解答"));
    
    let default_result = compiled.invoke(AgentState::new("今天天气怎么样".to_string())).await.unwrap();
    assert!(default_result.final_state.output.as_ref().unwrap().contains("通用解答"));
    
    println!("=== 三分支路由 ===");
    println!("数学 -> {}", math_result.final_state.output.as_ref().unwrap());
    println!("文本 -> {}", text_result.final_state.output.as_ref().unwrap());
    println!("代码 -> {}", code_result.final_state.output.as_ref().unwrap());
    println!("未知 -> {}", default_result.final_state.output.as_ref().unwrap());
}

// 示例: 循环处理

#[tokio::test]
async fn example_loop_with_limit() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();
    
    // 计数器
    graph.add_node_fn("increment_counter", |state: &AgentState| {
        let mut new_state = state.clone();
        let current: i32 = state.input.parse().unwrap_or(0);
        new_state.input = (current + 1).to_string();
        Ok(StateUpdate::full(new_state))
    });
    
    // 检查限制
    graph.add_node_fn("check_limit", |state: &AgentState| {
        Ok(StateUpdate::full(state.clone()))
    });
    
    // 完成
    graph.add_node_fn("finish", |state: &AgentState| {
        let mut s = state.clone();
        s.set_output(format!("最终计数: {}", state.input));
        Ok(StateUpdate::full(s))
    });
    
    graph.add_edge(START, "increment_counter");
    graph.add_edge("increment_counter", "check_limit");
    
    let targets = HashMap::from([
        ("continue".to_string(), "increment_counter".to_string()),
        ("done".to_string(), "finish".to_string()),
    ]);
    graph.add_conditional_edges("check_limit", "limit_router", targets, None);
    
    graph.add_edge("finish", END);
    
    let router = FunctionRouter::new(|state: &AgentState| {
        let current: i32 = state.input.parse().unwrap_or(0);
        if current >= 5 { "done" } else { "continue" }.to_string()
    });
    graph.set_conditional_router("limit_router", router);
    
    let compiled = graph.compile().unwrap().with_recursion_limit(20);
    
    let result = compiled.invoke(AgentState::new("0".to_string())).await.unwrap();
    assert!(result.final_state.output.as_ref().unwrap().contains("最终计数: 5"));
    
    println!("=== 循环处理 ===");
    println!("最终输出: {}", result.final_state.output.as_ref().unwrap());
    println!("执行节点数: {}", result.recursion_count);
}
