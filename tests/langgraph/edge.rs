use langchainrust::{GraphEdge, EdgeTarget};

// 测试固定边创建
// 验证: fixed() 创建从 source 到 target 的固定边
#[test]
fn test_fixed_edge_creation() {
    let edge = GraphEdge::fixed("source", "target");
    
    // 验证源节点
    assert_eq!(edge.source(), "source");
    // 验证目标节点
    assert_eq!(edge.fixed_target(), Some("target"));
}

// 测试从 START 出发的固定边
// 验证: START 可以作为边的源
#[test]
fn test_fixed_edge_from_start() {
    let edge = GraphEdge::fixed("__start__", "first_node");
    
    assert_eq!(edge.source(), "__start__");
    assert_eq!(edge.fixed_target(), Some("first_node"));
}

// 测试到 END 的固定边
// 验证: END 可以作为边的目标
#[test]
fn test_fixed_edge_to_end() {
    let edge = GraphEdge::fixed("last_node", "__end__");
    
    assert_eq!(edge.source(), "last_node");
    assert_eq!(edge.fixed_target(), Some("__end__"));
}

// 测试条件边创建
// 验证: conditional() 创建带路由器的条件边
#[test]
fn test_conditional_edge_creation() {
    use std::collections::HashMap;
    
    // 定义路由目标映射
    let targets = HashMap::from([
        ("route_a".to_string(), "node_a".to_string()),
        ("route_b".to_string(), "node_b".to_string()),
    ]);
    let edge = GraphEdge::conditional("source", "router_name", targets, None);
    
    // 验证源节点
    assert_eq!(edge.source(), "source");
    // 条件边没有固定目标
    assert!(edge.fixed_target().is_none());
}

// 测试带默认路径的条件边
// 验证: 条件边可设置默认目标
#[test]
fn test_conditional_edge_with_default() {
    use std::collections::HashMap;
    
    let targets = HashMap::from([
        ("yes".to_string(), "proceed".to_string()),
    ]);
    // 设置默认路径 "fallback"
    let edge = GraphEdge::conditional("decision", "router", targets, Some("fallback".to_string()));
    
    assert_eq!(edge.source(), "decision");
}

// 测试 EdgeTarget 固定目标
// 验证: EdgeTarget::to() 创建固定目标
#[test]
fn test_edge_target_fixed() {
    let target = EdgeTarget::to("node_name");
    
    match target {
        EdgeTarget::Fixed(name) => assert_eq!(name, "node_name"),
        EdgeTarget::Conditional(_) => panic!("Expected Fixed target"),
    }
}

// 测试 EdgeTarget 条件目标
// 验证: EdgeTarget::conditional() 创建条件目标
#[test]
fn test_edge_target_conditional() {
    let target = EdgeTarget::conditional("router_func");
    
    match target {
        EdgeTarget::Fixed(_) => panic!("Expected Conditional target"),
        EdgeTarget::Conditional(name) => assert_eq!(name, "router_func"),
    }
}
