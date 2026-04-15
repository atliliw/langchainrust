use langchainrust::{
    GraphBuilder, START, END,
    AgentState, StateUpdate,
};

// 测试递归限制生效
// 验证: 自循环图在达到递归限制时抛出错误
#[tokio::test]
async fn test_recursion_limit_respected() {
    // 创建自循环图: loop_node -> loop_node
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("loop_node", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "loop_node")
        .add_edge("loop_node", "loop_node")  // 自循环
        .compile()
        .unwrap()
        .with_recursion_limit(5);  // 限制5次

    // 执行应因递归限制而失败
    let result = compiled.invoke(AgentState::new("test".to_string())).await;
    assert!(result.is_err());
}

// 测试默认递归限制
// 验证: 不设置限制时, 使用默认值(25), 正常图可执行
#[tokio::test]
async fn test_default_recursion_limit() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("a", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("b", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "a")
        .add_edge("a", "b")
        .add_edge("b", END)
        .compile()
        .unwrap();

    let result = compiled.invoke(AgentState::new("test".to_string())).await.unwrap();
    // 应执行2个节点
    assert_eq!(result.recursion_count, 2);
}

// 测试自定义递归限制允许有效深度
// 验证: 设置足够大的限制, 深度图可正常执行
#[tokio::test]
async fn test_custom_recursion_limit_allows_valid_depth() {
    // 创建5节点深度图
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("n1", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("n2", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("n3", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("n4", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("n5", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "n1")
        .add_edge("n1", "n2")
        .add_edge("n2", "n3")
        .add_edge("n3", "n4")
        .add_edge("n4", "n5")
        .add_edge("n5", END)
        .compile()
        .unwrap()
        .with_recursion_limit(10);  // 限制10次, 5节点足够

    let result = compiled.invoke(AgentState::new("test".to_string())).await.unwrap();
    assert_eq!(result.recursion_count, 5);
}

// 测试低递归限制阻断有效图
// 验证: 设置过小的限制, 即使有效图也会失败
#[tokio::test]
async fn test_low_recursion_limit_fails_valid_graph() {
    // 3节点图但限制只有2次
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("n1", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("n2", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("n3", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "n1")
        .add_edge("n1", "n2")
        .add_edge("n2", "n3")
        .add_edge("n3", END)
        .compile()
        .unwrap()
        .with_recursion_limit(2);  // 限制2次, 3节点无法完成

    let result = compiled.invoke(AgentState::new("test".to_string())).await;
    assert!(result.is_err());
}

// 测试图在限制前到达终点
// 验证: 正常图在到达END前不会触发递归限制
#[tokio::test]
async fn test_graph_reaches_end_before_limit() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("single", |state| {
            let mut s = state.clone();
            s.set_output("done".to_string());
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "single")
        .add_edge("single", END)
        .compile()
        .unwrap()
        .with_recursion_limit(100);  // 大限制

    let result = compiled.invoke(AgentState::new("test".to_string())).await.unwrap();
    // 只执行1个节点就到达END
    assert_eq!(result.recursion_count, 1);
    assert_eq!(result.final_state.output, Some("done".to_string()));
}
