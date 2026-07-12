//! Async Node API 测试
//!
//! 本测试文件验证 LangGraph 异步节点 API 的核心功能。
//!
//! ## 测试覆盖范围
//!
//! 1. 异步节点基本执行能力
//! 2. 多异步节点链式执行
//! 3. 同步/异步节点混合使用
//! 4. 异步操作（如延迟、I/O）的正确处理
//!
//! ## 对框架的意义
//!
//! 异步节点 API 是 LangGraph 与 LLM 集成的核心基础设施：
//! - LLM API 调用天然是异步操作，需要 async/await 支持
//! - 解决了之前需要在 sync node 中使用 block_in_place 的复杂方案
//! - 提供更简洁的 API：直接用 async closure 定义节点
//! - 使 Agent、Tool 调用等异步操作能自然融入图执行流程
//!
//! ## API 使用示例
//!
//! ```rust,ignore
//! graph.add_async_node("llm_call", |state| async {
//!     let response = llm.chat(messages).await?;
//!     Ok(StateUpdate::full(new_state))
//! });
//! ```

use langchainrust::{
    GraphBuilder, START, END,
    AgentState, StateUpdate,
};

/// 测试异步节点基本功能
///
/// 功能验证:
/// - add_async_node() 方法能正确注册异步节点
/// - 异步闭包在图执行时能被正确调用
/// - 返回的 StateUpdate 能正确应用到状态
///
/// 框架作用:
/// - 这是异步节点 API 的基础测试，确保核心执行路径正确
/// - 验证 AsyncNode + AsyncFn trait 实现正确
/// - 为后续 LLM 集成提供基础保障
#[tokio::test(flavor = "multi_thread")]
async fn test_async_node_basic() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_async_node("async_process", |state: &AgentState| {
            let state = state.clone();
            async move {
                let mut new_state = state;
                new_state.add_message(langchainrust::MessageEntry::ai("Async processed".to_string()));
                Ok(StateUpdate::full(new_state))
            }
        })
        .add_edge(START, "async_process")
        .add_edge("async_process", END)
        .compile()
        .unwrap();
    
    let input = AgentState::new("Hello async".to_string());
    let result = compiled.invoke(input).await.unwrap();
    
    assert_eq!(result.recursion_count, 1);
    assert!(result.final_state.messages.len() > 1);
}

/// 测试多个异步节点链式执行
///
/// 功能验证:
/// - 多个异步节点能按边定义的顺序依次执行
/// - 每个节点的状态更新能正确传递到下一个节点
/// - 消息历史在链式执行中正确累积
///
/// 框架作用:
/// - 验证图的链式执行能力，这是 Agent pipeline 的基础模式
/// - 确保状态在异步节点间正确流转
/// - recursion_count 统计正确，用于调试和监控
#[tokio::test(flavor = "multi_thread")]
async fn test_multiple_async_nodes() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_async_node("step1", |state: &AgentState| {
            let state = state.clone();
            async move {
                let mut new_state = state;
                new_state.add_message(langchainrust::MessageEntry::ai("Step 1".to_string()));
                Ok(StateUpdate::full(new_state))
            }
        })
        .add_async_node("step2", |state: &AgentState| {
            let state = state.clone();
            async move {
                let mut new_state = state;
                new_state.add_message(langchainrust::MessageEntry::ai("Step 2".to_string()));
                Ok(StateUpdate::full(new_state))
            }
        })
        .add_async_node("step3", |state: &AgentState| {
            let state = state.clone();
            async move {
                let mut new_state = state;
                new_state.set_output("Done".to_string());
                Ok(StateUpdate::full(new_state))
            }
        })
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .add_edge("step2", "step3")
        .add_edge("step3", END)
        .compile()
        .unwrap();
    
    let input = AgentState::new("Test".to_string());
    let result = compiled.invoke(input).await.unwrap();
    
    assert_eq!(result.recursion_count, 3);
    assert_eq!(result.final_state.output, Some("Done".to_string()));
    // step3 只调用了 set_output，没有 add_message，所以只有 3 条消息
    // 初始 1 (human) + step1 1 (ai) + step2 1 (ai) = 3
    assert_eq!(result.final_state.messages.len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_mixed_sync_async_nodes() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("sync_step", |state: &AgentState| {
            let mut new_state = state.clone();
            new_state.add_message(langchainrust::MessageEntry::ai("Sync".to_string()));
            Ok(StateUpdate::full(new_state))
        })
        .add_async_node("async_step", |state: &AgentState| {
            let state = state.clone();
            async move {
                let mut new_state = state;
                new_state.add_message(langchainrust::MessageEntry::ai("Async".to_string()));
                Ok(StateUpdate::full(new_state))
            }
        })
        .add_edge(START, "sync_step")
        .add_edge("sync_step", "async_step")
        .add_edge("async_step", END)
        .compile()
        .unwrap();
    
    let input = AgentState::new("Mixed".to_string());
    let result = compiled.invoke(input).await.unwrap();
    
    assert_eq!(result.recursion_count, 2);
    assert_eq!(result.final_state.messages.len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_async_node_with_delay() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_async_node("delayed", |state: &AgentState| {
            let state = state.clone();
            async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                let mut new_state = state;
                new_state.set_output("Delayed result".to_string());
                Ok(StateUpdate::full(new_state))
            }
        })
        .add_edge(START, "delayed")
        .add_edge("delayed", END)
        .compile()
        .unwrap();
    
    let input = AgentState::new("Test delay".to_string());
    let result = compiled.invoke(input).await.unwrap();
    
    assert_eq!(result.final_state.output, Some("Delayed result".to_string()));
}