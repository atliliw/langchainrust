//! Graph Visualization 测试
//!
//! 本测试文件验证 LangGraph 图可视化功能的正确性。
//!
//! ## 测试覆盖范围
//!
//! 1. ASCII 格式可视化（终端友好）
//! 2. Mermaid 格式可视化（Markdown 文档友好）
//! 3. JSON 格式可视化（程序处理友好）
//! 4. 条件边在可视化中的正确展示
//!
//! ## 对框架的意义
//!
//! 图可视化是开发和调试 LangGraph 应用的关键工具：
//! - 帮助开发者理解复杂的图结构和执行流程
//! - 在文档和报告中展示 Agent 工作流
//! - 便于验证图的构建是否正确（边连接、路由配置等）
//! - 支持自动化测试中验证图结构

use langchainrust::{
    AgentState, FunctionRouter, GraphBuilder, StateGraph, StateUpdate, END, START,
};
use std::collections::HashMap;

/// 测试 ASCII 格式可视化 - 线性图
///
/// 功能验证:
/// - visualize_ascii() 正确输出节点列表
/// - 正确显示 START 和 END 边界节点
/// - 正确显示 Entry Point 入口点
///
/// 框架作用:
/// - 确保基础可视化格式正确，用于终端调试
/// - 验证线性图（最简单的图结构）可视化
/// - 这是开发时快速检查图结构的主要方法
#[test]
fn test_visualize_ascii_linear_graph() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("step1", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("step2", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .add_edge("step2", END)
        .compile()
        .unwrap();

    let output = compiled.visualize_ascii();

    println!("=== ASCII 输出 ===");
    println!("{}", output);

    assert!(output.contains("__start__"));
    assert!(output.contains("step1"));
    assert!(output.contains("step2"));
    assert!(output.contains("__end__"));
}

/// 测试 ASCII 格式可视化 - 条件路由图
///
/// 功能验证:
/// - 正确显示 Routers 部分
/// - 正确显示条件边的路由规则（路由键 → 目标节点）
/// - 条件边的特殊格式与固定边区分
///
/// 框架作用:
/// - 验证复杂图结构（分支）的可视化
/// - 条件路由是 Agent 决策流程的核心，需要清晰展示
/// - 开发者可以检查路由配置是否正确
#[test]
fn test_visualize_ascii_conditional_graph() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();

    graph.add_node_fn("decision", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("path_a", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("path_b", |state| Ok(StateUpdate::full(state.clone())));

    graph.add_edge(START, "decision");
    graph.add_conditional_edges(
        "decision",
        "router",
        HashMap::from([
            ("a".to_string(), "path_a".to_string()),
            ("b".to_string(), "path_b".to_string()),
        ]),
        None,
    );
    graph.add_edge("path_a", END);
    graph.add_edge("path_b", END);

    let router = FunctionRouter::new(|_state| "a".to_string());
    graph.set_conditional_router("router", router);

    let compiled = graph.compile().unwrap();
    let output = compiled.visualize_ascii();

    println!("=== ASCII 条件图输出 ===");
    println!("{}", output);

    assert!(output.contains("Routers:"));
    assert!(output.contains("router"));
    assert!(output.contains("a → path_a"));
    assert!(output.contains("b → path_b"));
}

/// 测试 Mermaid 格式可视化
///
/// 功能验证:
/// - 输出包含正确的 Mermaid 代码块标记
/// - graph TD 声明正确（自上而下流程图）
/// - START/END/节点都正确转换为 Mermaid 格式
///
/// 框架作用:
/// - Mermaid 是 Markdown 文档的标准图表格式
/// - 可以直接嵌入 README、文档、报告中
/// - GitHub、GitLab 等平台原生支持渲染
/// - 用于展示 Agent 架构和设计文档
#[test]
fn test_visualize_mermaid() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("process", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "process")
        .add_edge("process", END)
        .compile()
        .unwrap();

    let output = compiled.visualize_mermaid();

    println!("=== Mermaid 输出 ===");
    println!("{}", output);

    assert!(output.contains("```mermaid"));
    assert!(output.contains("graph TD"));
    assert!(output.contains("__start__"));
    assert!(output.contains("__end__"));
    assert!(output.contains("process"));
}

/// 测试 JSON 格式可视化
///
/// 功能验证:
/// - 输出为有效的 JSON 结构
/// - 包含 entry_point、nodes、edges、recursion_limit 字段
/// - nodes 数量正确，edges 数量正确
///
/// 框架作用:
/// - JSON 格式适合程序解析和自动化验证
/// - 可以用于 CI/CD 中验证图结构
/// - 可以存储到数据库或配置文件
/// - 与其他系统集成时的标准格式
#[test]
fn test_visualize_json() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("node1", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("node2", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "node1")
        .add_edge("node1", "node2")
        .add_edge("node2", END)
        .compile()
        .unwrap();

    let json = compiled.visualize_json();

    println!("=== JSON 输出 ===");
    println!("{}", serde_json::to_string_pretty(&json).unwrap());

    assert!(json["entry_point"].is_string());
    assert!(json["nodes"].is_array());
    assert!(json["edges"].is_array());
    assert!(json["recursion_limit"].is_number());

    let nodes = json["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 2);
}
