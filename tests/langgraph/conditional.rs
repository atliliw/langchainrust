use langchainrust::{
    StateGraph, START, END,
    AgentState, StateUpdate,
    FunctionRouter,
};
use std::collections::HashMap;

// 测试条件路由 - 短路径
// 验证: 当输入长度 < 10 时, 路由到 short 节点
#[tokio::test]
async fn test_conditional_routing_short_path() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();
    
    // 添加节点
    graph.add_node_fn("entry", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("short", |state| {
        let mut s = state.clone();
        s.set_output("short".to_string());
        Ok(StateUpdate::full(s))
    });
    graph.add_node_fn("long", |state| {
        let mut s = state.clone();
        s.set_output("long".to_string());
        Ok(StateUpdate::full(s))
    });
    
    graph.add_edge(START, "entry");
    
    // 设置条件边: 根据路由结果选择目标
    let targets = HashMap::from([
        ("short".to_string(), "short".to_string()),
        ("long".to_string(), "long".to_string()),
    ]);
    graph.add_conditional_edges("entry", "router", targets, None);
    
    graph.add_edge("short", END);
    graph.add_edge("long", END);
    
    // 路由函数: 根据输入长度决定路径
    let router = FunctionRouter::new(|state: &AgentState| {
        if state.input.len() < 10 { "short" } else { "long" }.to_string()
    });
    graph.set_conditional_router("router", router);
    
    let compiled = graph.compile().unwrap();
    
    // 短输入 "hi" 应路由到 short 节点
    let short_result = compiled.invoke(AgentState::new("hi".to_string())).await.unwrap();
    assert_eq!(short_result.final_state.output, Some("short".to_string()));
}

// 测试条件路由 - 长路径
// 验证: 当输入长度 >= 10 时, 路由到 long 节点
#[tokio::test]
async fn test_conditional_routing_long_path() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();
    
    graph.add_node_fn("entry", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("short", |state| {
        let mut s = state.clone();
        s.set_output("short".to_string());
        Ok(StateUpdate::full(s))
    });
    graph.add_node_fn("long", |state| {
        let mut s = state.clone();
        s.set_output("long".to_string());
        Ok(StateUpdate::full(s))
    });
    
    graph.add_edge(START, "entry");
    
    let targets = HashMap::from([
        ("short".to_string(), "short".to_string()),
        ("long".to_string(), "long".to_string()),
    ]);
    graph.add_conditional_edges("entry", "router", targets, None);
    
    graph.add_edge("short", END);
    graph.add_edge("long", END);
    
    let router = FunctionRouter::new(|state: &AgentState| {
        if state.input.len() < 10 { "short" } else { "long" }.to_string()
    });
    graph.set_conditional_router("router", router);
    
    let compiled = graph.compile().unwrap();
    
    // 长输入应路由到 long 节点
    let long_result = compiled.invoke(AgentState::new("this is a very long input".to_string())).await.unwrap();
    assert_eq!(long_result.final_state.output, Some("long".to_string()));
}

// 测试条件路由 - 带默认路径
// 验证: 当路由结果不在 targets 中时, 使用默认路径
#[tokio::test]
async fn test_conditional_routing_with_default() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();
    
    graph.add_node_fn("entry", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("path_a", |state| {
        let mut s = state.clone();
        s.set_output("a".to_string());
        Ok(StateUpdate::full(s))
    });
    graph.add_node_fn("path_b", |state| {
        let mut s = state.clone();
        s.set_output("b".to_string());
        Ok(StateUpdate::full(s))
    });
    graph.add_node_fn("default_path", |state| {
        let mut s = state.clone();
        s.set_output("default".to_string());
        Ok(StateUpdate::full(s))
    });
    
    graph.add_edge(START, "entry");
    
    // 只定义 "a" 和 "b" 路径, 其他走默认
    let targets = HashMap::from([
        ("a".to_string(), "path_a".to_string()),
        ("b".to_string(), "path_b".to_string()),
    ]);
    // 设置默认路径
    graph.add_conditional_edges("entry", "router", targets, Some("default_path".to_string()));
    
    graph.add_edge("path_a", END);
    graph.add_edge("path_b", END);
    graph.add_edge("default_path", END);
    
    // 路由: 根据输入首字母决定路径
    let router = FunctionRouter::new(|state: &AgentState| {
        let input = state.input.as_str();
        if input.starts_with("a") { "a" }
        else if input.starts_with("b") { "b" }
        else { "unknown" }  // 不在 targets 中
    }.to_string());
    graph.set_conditional_router("router", router);
    
    let compiled = graph.compile().unwrap();
    
    // "xyz" 不以 a 或 b 开头, 应走默认路径
    let default_result = compiled.invoke(AgentState::new("xyz".to_string())).await.unwrap();
    assert_eq!(default_result.final_state.output, Some("default".to_string()));
}

// 测试多分支条件路由
// 验证: 三条独立路径都能正确路由
#[tokio::test]
async fn test_multiple_conditional_branches() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();
    
    graph.add_node_fn("decide", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("alpha", |state| {
        let mut s = state.clone();
        s.set_output("alpha_result".to_string());
        Ok(StateUpdate::full(s))
    });
    graph.add_node_fn("beta", |state| {
        let mut s = state.clone();
        s.set_output("beta_result".to_string());
        Ok(StateUpdate::full(s))
    });
    graph.add_node_fn("gamma", |state| {
        let mut s = state.clone();
        s.set_output("gamma_result".to_string());
        Ok(StateUpdate::full(s))
    });
    
    graph.add_edge(START, "decide");
    
    // 三条路径
    let targets = HashMap::from([
        ("alpha".to_string(), "alpha".to_string()),
        ("beta".to_string(), "beta".to_string()),
        ("gamma".to_string(), "gamma".to_string()),
    ]);
    graph.add_conditional_edges("decide", "branch_router", targets, None);
    
    graph.add_edge("alpha", END);
    graph.add_edge("beta", END);
    graph.add_edge("gamma", END);
    
    // 路由: 输入内容直接决定路径
    let router = FunctionRouter::new(|state: &AgentState| {
        state.input.clone()
    });
    graph.set_conditional_router("branch_router", router);
    
    let compiled = graph.compile().unwrap();
    
    // 测试三条路径都能正确路由
    let alpha_result = compiled.invoke(AgentState::new("alpha".to_string())).await.unwrap();
    assert_eq!(alpha_result.final_state.output, Some("alpha_result".to_string()));
    
    let beta_result = compiled.invoke(AgentState::new("beta".to_string())).await.unwrap();
    assert_eq!(beta_result.final_state.output, Some("beta_result".to_string()));
    
    let gamma_result = compiled.invoke(AgentState::new("gamma".to_string())).await.unwrap();
    assert_eq!(gamma_result.final_state.output, Some("gamma_result".to_string()));
}

// 测试条件边必须有对应的路由器
// 验证: 定义了条件边但没有设置路由器, 编译应失败
#[test]
fn test_conditional_edge_validation_requires_router() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();
    
    graph.add_node_fn("entry", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("target", |state| Ok(StateUpdate::full(state.clone())));
    
    graph.add_edge(START, "entry");
    
    let targets = HashMap::from([
        ("route".to_string(), "target".to_string()),
    ]);
    // 添加条件边但没有设置路由器
    graph.add_conditional_edges("entry", "missing_router", targets, None);
    
    graph.add_edge("target", END);
    
    // 编译应失败, 因为路由器不存在
    let result = graph.compile();
    assert!(result.is_err());
}
