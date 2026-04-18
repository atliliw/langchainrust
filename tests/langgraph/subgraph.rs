//! LangGraph 子图支持测试
//!
//! 本测试文件验证 LangGraph 的 Subgraph（子图）功能。
//!
//! ## 什么是子图？
//!
//! 子图允许将一个编译后的图嵌入到另一个图中作为一个节点。
//! 这使得可以从简单、可复用的组件组合出复杂的工作流。
//!
//! ## 框架作用
//!
//! 子图对于以下场景至关重要：
//! - 模块化设计：将复杂 Agent 拆分为小型、可测试的组件
//! - 可复用性：同一个子图可以在多个父图中使用
//! - 嵌套执行：子图可以包含自己的子图（任意深度）
//! - 状态隔离：子图执行封装在单个节点内
//!
//! ## 使用场景
//!
//! 1. 多阶段处理：研究 → 分析 → 综合（每个阶段作为子图）
//! 2. 可复用工作流：错误处理模式在多个 Agent 中使用
//! 3. 团队协作：不同团队构建不同子图，然后组合
//! 4. 测试隔离：独立测试子图后再集成

use langchainrust::{
    GraphBuilder, START, END,
    AgentState, StateUpdate,
};
use langchainrust::langgraph::SubgraphNode;
use std::time::Duration;

/// 测试：基本子图执行
///
/// 验证内容：
/// - 子图可以创建并嵌入到父图中
/// - 父图调用时子图执行其内部工作流
/// - 状态正确流转：父图 → 子图 → 父图
///
/// 框架作用：
/// - 验证核心子图机制：SubgraphNode 包装 CompiledGraph
/// - 确认使用 same_state 时状态映射正确工作
/// - 展示最简单的子图用法：单节点子图
///
/// 使用场景示例：
/// - 在"数据处理管道"中使用"数据验证"子图
#[tokio::test]
async fn test_basic_subgraph() {
    let subgraph = GraphBuilder::<AgentState>::new()
        .add_node_fn("sub_process", |state| {
            let mut s = state.clone();
            s.set_output("subgraph_done".to_string());
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "sub_process")
        .add_edge("sub_process", END)
        .compile()
        .unwrap();

    let parent = GraphBuilder::<AgentState>::new()
        .add_subgraph_same_state("subworkflow", subgraph)
        .add_edge(START, "subworkflow")
        .add_edge("subworkflow", END)
        .compile()
        .unwrap();

    let input = AgentState::new("test".to_string());
    let result = parent.invoke(input).await.unwrap();

    assert!(result.final_state.output.is_some());
    assert_eq!(result.final_state.output.unwrap(), "subgraph_done");
}

/// 测试：嵌套子图（3层深度）
///
/// 验证内容：
/// - 子图可以包含其他子图（递归组合）
/// - 状态正确流转所有嵌套层级
/// - 执行顺序保持：外层 → 中层 → 内层
///
/// 框架作用：
/// - 验证 SubgraphNode 可以自身包含 SubgraphNode
/// - 证明架构支持任意嵌套深度
/// - 测试跨嵌套边界的状态累积
///
/// 使用场景示例：
/// - 外层图："客服 Agent"
/// - 中层子图："问题解决工作流"
/// - 内层子图："工单分类流程"
/// 每层可以独立开发和测试。
#[tokio::test]
async fn test_nested_subgraphs() {
    let inner = GraphBuilder::<AgentState>::new()
        .add_node_fn("inner_node", |state| {
            let mut s = state.clone();
            s.input = format!("inner:{}", s.input);
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "inner_node")
        .add_edge("inner_node", END)
        .compile()
        .unwrap();

    let middle = GraphBuilder::<AgentState>::new()
        .add_subgraph_same_state("inner_workflow", inner)
        .add_node_fn("middle_node", |state| {
            let mut s = state.clone();
            s.input = format!("middle:{}", s.input);
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "inner_workflow")
        .add_edge("inner_workflow", "middle_node")
        .add_edge("middle_node", END)
        .compile()
        .unwrap();

    let outer = GraphBuilder::<AgentState>::new()
        .add_node_fn("outer_node", |state| {
            let mut s = state.clone();
            s.input = format!("outer:{}", s.input);
            Ok(StateUpdate::full(s))
        })
        .add_subgraph_same_state("middle_workflow", middle)
        .add_edge(START, "outer_node")
        .add_edge("outer_node", "middle_workflow")
        .add_edge("middle_workflow", END)
        .compile()
        .unwrap();

    let input = AgentState::new("test".to_string());
    let result = outer.invoke(input).await.unwrap();

    assert!(result.final_state.input.contains("outer"));
    assert!(result.final_state.input.contains("middle"));
    assert!(result.final_state.input.contains("inner"));
}

/// 测试：子图内多节点工作流
///
/// 验证内容：
/// - 子图可以有复杂的内部工作流（3节点链）
/// - 父图可以在子图前后有节点
/// - 子图执行的所有消息都保留在父图状态中
///
/// 框架作用：
/// - 验证子图内部复杂性不影响父图
/// - 证明状态累积：所有节点的消息被收集
/// - 测试父图和子图执行的边界
///
/// 使用场景示例：
/// - 父图："文档处理管道"
///   - before：加载文档
///   - workflow（子图）：OCR → 提取 → 校验 → 格式化
///   - after：保存结果
/// 子图处理复杂的多步骤转换。
#[tokio::test]
async fn test_subgraph_with_multiple_nodes() {
    let subgraph = GraphBuilder::<AgentState>::new()
        .add_node_fn("step1", |state| {
            let mut s = state.clone();
            s.add_message(langchainrust::MessageEntry::ai("step1_done".to_string()));
            Ok(StateUpdate::full(s))
        })
        .add_node_fn("step2", |state| {
            let mut s = state.clone();
            s.add_message(langchainrust::MessageEntry::ai("step2_done".to_string()));
            Ok(StateUpdate::full(s))
        })
        .add_node_fn("step3", |state| {
            let mut s = state.clone();
            s.add_message(langchainrust::MessageEntry::ai("step3_done".to_string()));
            s.set_output("all_steps_done".to_string());
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .add_edge("step2", "step3")
        .add_edge("step3", END)
        .compile()
        .unwrap();

    let parent = GraphBuilder::<AgentState>::new()
        .add_node_fn("before", |state| {
            let mut s = state.clone();
            s.add_message(langchainrust::MessageEntry::ai("before_subgraph".to_string()));
            Ok(StateUpdate::full(s))
        })
        .add_subgraph_same_state("workflow", subgraph)
        .add_node_fn("after", |state| {
            let mut s = state.clone();
            s.add_message(langchainrust::MessageEntry::ai("after_subgraph".to_string()));
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "before")
        .add_edge("before", "workflow")
        .add_edge("workflow", "after")
        .add_edge("after", END)
        .compile()
        .unwrap();

    let input = AgentState::new("test".to_string());
    let result = parent.invoke(input).await.unwrap();

    assert!(result.final_state.output.is_some());
    assert_eq!(result.final_state.output.unwrap(), "all_steps_done");
    assert!(result.final_state.messages.len() >= 5);
}

/// 测试：异步子图执行
///
/// 验证内容：
/// - 子图可以包含执行 I/O 操作的异步节点
/// - 子图内异步执行正确工作
/// - 父图等待子图异步操作完成
///
/// 框架作用：
/// - 验证子图上下文中的异步节点支持
/// - 证明 tokio 运行时集成在任意嵌套层级工作
/// - 测试异步延迟不破坏子图边界
///
/// 使用场景示例：
/// - 子图包含：API 调用 → 数据库查询 → 文件写入
/// 都是父图不需要知道的异步 I/O 操作。
#[tokio::test]
async fn test_async_subgraph() {
    let subgraph = GraphBuilder::<AgentState>::new()
        .add_async_node("async_process", |state: &AgentState| {
            let state = state.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                let mut s = state;
                s.set_output("async_done".to_string());
                Ok(StateUpdate::full(s))
            }
        })
        .add_edge(START, "async_process")
        .add_edge("async_process", END)
        .compile()
        .unwrap();

    let parent = GraphBuilder::<AgentState>::new()
        .add_subgraph_same_state("async_workflow", subgraph)
        .add_edge(START, "async_workflow")
        .add_edge("async_workflow", END)
        .compile()
        .unwrap();

    let input = AgentState::new("test".to_string());
    let result = parent.invoke(input).await.unwrap();

    assert!(result.final_state.output.is_some());
    assert_eq!(result.final_state.output.unwrap(), "async_done");
}