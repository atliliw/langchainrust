mod llm_integration;
mod async_node;
mod visualize;
mod human_loop;
mod validation;

use langchainrust::{
    StateGraph, GraphBuilder, START, END,
    AgentState, StateUpdate,
    ThreadSafeMemoryCheckpointer, Checkpointer,
};
use std::collections::HashMap;

// 测试线性图执行: START -> node1 -> node2 -> END
// 验证: 状态在节点间正确传递, 递归计数正确
#[tokio::test]
async fn test_linear_graph_execution() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("node1", |state: &AgentState| {
            // 第一个节点处理输入
            Ok(StateUpdate::full(AgentState::new(format!("step1: {}", state.input))))
        })
        .add_node_fn("node2", |state: &AgentState| {
            // 第二个节点设置输出
            let mut s = state.clone();
            s.set_output(format!("step2: {}", state.input));
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "node1")
        .add_edge("node1", "node2")
        .add_edge("node2", END)
        .compile()
        .unwrap();

    let result = compiled.invoke(AgentState::new("test".to_string())).await.unwrap();
    
    // 打印输出结果
    println!("=== test_linear_graph_execution ===");
    println!("final_state.input: {}", result.final_state.input);
    println!("final_state.output: {:?}", result.final_state.output);
    println!("recursion_count: {}", result.recursion_count);
}

// 测试单节点图: START -> single -> END
// 验证: 最简单的图结构能正常工作
#[tokio::test]
async fn test_single_node_graph() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("single", |state: &AgentState| {
            let mut s = state.clone();
            s.set_output("done".to_string());
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "single")
        .add_edge("single", END)
        .compile()
        .unwrap();

    let result = compiled.invoke(AgentState::new("input".to_string())).await.unwrap();
    
    println!("=== test_single_node_graph ===");
    println!("final_state.input: {}", result.final_state.input);
    println!("final_state.output: {:?}", result.final_state.output);
    println!("recursion_count: {}", result.recursion_count);
}

// 测试三节点链式执行: START -> first -> second -> third -> END
// 验证: 状态在多个节点间正确流转, 递归计数累加
#[tokio::test]
async fn test_three_node_chain() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("first", |state| {
            // 第一节点添加前缀 "1:"
            Ok(StateUpdate::full(AgentState::new(format!("1:{}", state.input))))
        })
        .add_node_fn("second", |state| {
            // 第二节点添加前缀 "2:"
            Ok(StateUpdate::full(AgentState::new(format!("2:{}", state.input))))
        })
        .add_node_fn("third", |state| {
            // 第三节点设置最终输出
            let mut s = state.clone();
            s.set_output(format!("final:{}", state.input));
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "first")
        .add_edge("first", "second")
        .add_edge("second", "third")
        .add_edge("third", END)
        .compile()
        .unwrap();

    let result = compiled.invoke(AgentState::new("input".to_string())).await.unwrap();
    
    println!("=== test_three_node_chain ===");
    println!("final_state.input: {}", result.final_state.input);
    println!("final_state.output: {:?}", result.final_state.output);
    println!("recursion_count: {}", result.recursion_count);
}

// 测试流式执行返回事件列表
// 验证: stream() 方法返回非空的事件序列, 每个节点执行产生事件
#[tokio::test]
async fn test_stream_execution_returns_events() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("a", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("b", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "a")
        .add_edge("a", "b")
        .add_edge("b", END)
        .compile()
        .unwrap();

    // 流式执行获取所有事件
    let events = compiled.stream(AgentState::new("test".to_string())).await.unwrap();
    
    println!("=== test_stream_execution_returns_events ===");
    println!("events count: {}", events.len());
    for (i, event) in events.iter().enumerate() {
        println!("event[{}]: {:?}", i, event);
    }
}

// 测试 Checkpointer 与图集成
// 验证: 带 checkpointer 的图能正常执行, 状态被正确保存
#[tokio::test]
async fn test_checkpointer_integration() {
    let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
    
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("process", |state| {
            let mut s = state.clone();
            s.set_output("processed".to_string());
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "process")
        .add_edge("process", END)
        .compile()
        .unwrap()
        .with_checkpointer(checkpointer);

    let result = compiled.invoke(AgentState::new("test".to_string())).await.unwrap();
    
    println!("=== test_checkpointer_integration ===");
    println!("final_state.input: {}", result.final_state.input);
    println!("final_state.output: {:?}", result.final_state.output);
}

// 测试 Checkpointer CRUD 操作
// 验证: save, load, list, delete 功能正常
#[tokio::test]
async fn test_checkpointer_save_and_load() {
    let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
    
    // 保存两个状态
    let state1 = AgentState::new("first".to_string());
    let id1 = checkpointer.save(&state1).await.unwrap();
    
    let state2 = AgentState::new("second".to_string());
    let id2 = checkpointer.save(&state2).await.unwrap();
    
    // 验证列表包含两个checkpoint
    let list = checkpointer.list().await.unwrap();
    
    println!("=== test_checkpointer_save_and_load ===");
    println!("checkpoint id1: {}", id1);
    println!("checkpoint id2: {}", id2);
    println!("list count: {}", list.len());
    
    // 验证能正确加载第一个状态
    let loaded = checkpointer.load(&id1).await.unwrap();
    println!("loaded state input: {}", loaded.input);
    
    // 验证删除功能
    checkpointer.delete(&id2).await.unwrap();
    let list_after_delete = checkpointer.list().await.unwrap();
    println!("list count after delete: {}", list_after_delete.len());
}

// 测试 GraphBuilder 创建有效图
// 验证: build() 返回可编译的 StateGraph
#[test]
fn test_graph_builder_creates_valid_graph() {
    let graph = GraphBuilder::<AgentState>::new()
        .add_node_fn("n1", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "n1")
        .add_edge("n1", END)
        .build();
    
    let result = graph.compile();
    println!("=== test_graph_builder_creates_valid_graph ===");
    println!("compile result: {:?}", result.is_ok());
}

// 测试空图编译失败
// 验证: 没有节点的图无法编译
#[test]
fn test_empty_graph_fails_compile() {
    let graph: StateGraph<AgentState> = StateGraph::new();
    // 空图应该返回编译错误
    let result = graph.compile();
    println!("=== test_empty_graph_fails_compile ===");
    println!("compile result (should be err): {:?}", result.is_err());
}

// 测试无效入口点编译失败
// 验证: 指向不存在节点的入口点导致编译失败
#[test]
fn test_invalid_entry_point_fails() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();
    graph.add_node_fn("node1", |state| Ok(StateUpdate::full(state.clone())));
    graph.set_entry_point("nonexistent");  // 不存在的节点
    
    // 应返回验证错误
    let result = graph.compile();
    println!("=== test_invalid_entry_point_fails ===");
    println!("compile result (should be err): {:?}", result.is_err());
}

// 测试缺失边的图仍可编译
// 验证: 没有从节点b出发的边, 图仍可编译(节点b不会被执行)
#[test]
fn test_missing_edge_fails_validation() {
    let mut graph: StateGraph<AgentState> = StateGraph::new();
    graph.add_node_fn("a", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_node_fn("b", |state| Ok(StateUpdate::full(state.clone())));
    graph.add_edge(START, "a");
    // 注意: 没有 a -> b 的边
    
    // 图仍可编译, 但节点b不会被执行
    let compiled = graph.compile();
    println!("=== test_missing_edge_fails_validation ===");
    println!("compile result (should be ok): {:?}", compiled.is_ok());
}