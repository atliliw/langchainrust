//! Graph 验证增强测试
//!
//! 本测试文件验证 LangGraph 图结构静态验证功能。
//!
//! ## 测试覆盖范围
//!
//! 1. 死循环检测 - 无 END 路径的循环
//! 2. 孤立节点检测 - 不可达节点
//! 3. 重复边检测 - 相同的 source-target
//! 4. 条件边目标验证 - 无效目标节点
//!
//! ## 对框架的意义
//!
//! 图验证增强是防止错误配置的关键：
//! - 在编译阶段发现问题，避免运行时失败
//! - 提供清晰的错误信息，帮助开发者定位问题
//! - 确保图结构正确，减少调试时间
//! - 提升框架的可靠性和易用性

use langchainrust::{AgentState, GraphBuilder, GraphError, StateGraph, StateUpdate, END, START};
use std::collections::HashMap;

/// 测试死循环检测 - 简单循环无 END
///
/// 功能验证:
/// - 两节点互相连接形成循环
/// - 没有 END 路径
/// - 编译时报错 InfiniteCycleError
///
/// 框架作用:
/// - 防止运行时无限循环
/// - 在编译阶段提示开发者
/// - 确保所有路径最终能到达 END
#[test]
fn test_cycle_detection_simple() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();

    graph.add_node_fn("node1", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("node2", |state| Ok(StateUpdate::full(state.clone())));

    graph.add_edge(START, "node1");
    graph.add_edge("node1", "node2");
    graph.add_edge("node2", "node1"); // 循环！

    let result = graph.compile();

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, GraphError::InfiniteCycleError(_)));
}

/// 测试孤立节点检测
///
/// 功能验证:
/// - 孤立节点没有入边
/// - 无法从 START 到达
/// - 编译时报错 OrphanNodeError
///
/// 框架作用:
/// - 发现遗漏的节点连接
/// - 提示开发者补充边
/// - 确保图的所有节点都有意义
#[test]
fn test_unreachable_node_detection() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();

    graph.add_node_fn("connected", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("isolated", |state| Ok(StateUpdate::full(state.clone()))); // 没有边连接

    graph.add_edge(START, "connected");
    graph.add_edge("connected", END);

    let result = graph.compile();

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, GraphError::OrphanNodeError(_)));
    assert!(err.to_string().contains("isolated"));
}

/// 测试重复边检测
///
/// 功能验证:
/// - 相同的 source-target 出现两次
/// - 编译时报错 DuplicateEdgeError
/// - 错误信息包含重复的边信息
///
/// 框架作用:
/// - 防止逻辑错误
/// - 提示开发者检查边的定义
/// - 保持图结构的简洁性
#[test]
fn test_duplicate_edge_detection() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();

    graph.add_node_fn("node1", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("node2", |state| Ok(StateUpdate::full(state.clone())));

    graph.add_edge(START, "node1");
    graph.add_edge("node1", "node2");
    graph.add_edge("node1", "node2"); // 重复边！
    graph.add_edge("node2", END);

    let result = graph.compile();

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, GraphError::DuplicateEdgeError(_)));
}

/// 测试条件边目标不存在
///
/// 功能验证:
/// - 条件边的 target 指向不存在的节点
/// - 编译时报错 ValidationError
/// - 错误信息包含无效目标
///
/// 框架作用:
/// - 阿止条件路由配置错误
/// - 在编译阶段检查 targets 完整性
/// - 防止运行时 RoutingError
#[test]
fn test_conditional_edge_invalid_target() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();

    graph.add_node_fn("decision", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("path_a", |state| Ok(StateUpdate::full(state.clone())));

    graph.add_edge(START, "decision");

    let targets = HashMap::from([
        ("a".to_string(), "path_a".to_string()),
        ("b".to_string(), "nonexistent_node".to_string()), // 不存在！
    ]);
    graph.add_conditional_edges("decision", "router", targets, None);

    graph.add_edge("path_a", END);

    let router = langchainrust::FunctionRouter::new(|_state| "a".to_string());
    graph.set_conditional_router("router", router);

    let result = graph.compile();

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, GraphError::ValidationError(_)));
}

/// 测试边指向 START 的错误
///
/// 功能验证:
/// - 边的目标是 START（应该只有 source）
/// - 编译时报错 ValidationError
///
/// 框架作用:
/// - 防止不合理的图结构
/// - START 应该只能是起点
/// - 防止潜在的无限循环
#[test]
fn test_edge_targeting_start() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();

    graph.add_node_fn("loop_node", |state| Ok(StateUpdate::full(state.clone())));

    graph.add_edge(START, "loop_node");
    graph.add_edge("loop_node", START); // 目标是 START！

    let result = graph.compile();

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("START"));
}

/// 测试正常图编译成功
///
/// 功能验证:
/// - 正确的图结构能编译成功
/// - 所有节点可达
/// - 有 END 路径
/// - 无重复边
///
/// 框架作用:
/// - 验证正常流程不受影响
/// - 确保验证规则不误报
#[test]
fn test_valid_graph_compiles() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("step1", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("step2", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("step3", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .add_edge("step2", "step3")
        .add_edge("step3", END)
        .compile();

    assert!(compiled.is_ok());
}
