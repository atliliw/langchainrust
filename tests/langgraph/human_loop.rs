//! Human-in-the-loop 测试
//!
//! 本测试文件验证 LangGraph 的人工干预功能。
//!
//! ## 测试覆盖范围
//!
//! 1. interrupt_before - 在节点执行前中断
//! 2. interrupt_after - 在节点执行后中断
//! 3. resume() - 从中断点恢复执行
//! 4. 多节点中断和恢复流程
//!
//! ## 对框架的意义
//!
//! Human-in-the-loop 是 Agent 调试和交互的关键能力：
//! - 允许人工在关键节点检查状态
//! - 支持修改状态后继续执行
//! - 用于调试复杂 Agent 工作流
//! - 实现需要人工审批的自动化流程

use langchainrust::{
    GraphBuilder, START, END,
    AgentState, StateUpdate, GraphExecution,
};

/// 测试 interrupt_before - 在节点执行前中断
///
/// 功能验证:
/// - interrupt_before 配置能正确触发中断
/// - 中断发生在节点执行之前
/// - 返回 ExecutionInterrupted 错误
/// - 错误信息包含中断节点名称
///
/// 框架作用:
/// - 这是人工干预的基础模式
/// - 适合需要审批/检查的场景
/// - 允许在关键决策点暂停
#[tokio::test]
async fn test_interrupt_before() {
    // ========== 构建图结构 ==========
    // 图结构: START → step1 → step2 → step3 → END
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("step1", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("step2", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("step3", |state| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .add_edge("step2", "step3")
        .add_edge("step3", END)
        .compile()
        .unwrap()
        // ========== 配置中断点 ==========
        // 关键：在 step2 执行前中断
        // 执行流程：START → step1 → [准备执行 step2，被中断！]
        .with_interrupt_before(vec!["step2".to_string()]);
    
    let input = AgentState::new("test".to_string());
    
    // ========== 第一次执行：被打断 ==========
    let result = compiled.invoke(input).await;
    // 
    // 执行过程：
    //   START → step1 执行成功
    //   准备执行 step2 → 检测到 interrupt_before 包含 "step2" → 抛出 ExecutionInterrupted
    //   ⚠️ step2 没有被执行！
    //   ⚠️ step3 也没有被执行！
    //   返回错误，包含中断位置 "step2"
    //
    
    // ========== 验证中断 ==========
    assert!(result.is_err());  // ← 这里体现被打断：返回错误而不是成功结果
    let err = result.unwrap_err();
    assert!(matches!(err, langchainrust::GraphError::ExecutionInterrupted(_)));
    
    let interrupted_at = err.to_string();
    assert!(interrupted_at.contains("step2"));  // ← 错误信息告诉我们在哪里被打断
}

/// 测试 interrupt_after - 在节点执行后中断
///
/// 功能验证:
/// - interrupt_after 配置能正确触发中断
/// - 中断发生在节点执行之后
/// - 状态已经更新
///
/// 框架作用:
/// - 用于查看节点执行结果
/// - 适合需要检查中间结果的场景
/// - 可在批准后继续执行
#[tokio::test]
async fn test_interrupt_after() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("process", |state| {
            let mut new_state = state.clone();
            new_state.add_message(langchainrust::MessageEntry::ai("processed".to_string()));
            Ok(StateUpdate::full(new_state))
        })
        .add_node_fn("finalize", |state| {
            let mut new_state = state.clone();
            new_state.set_output("done".to_string());
            Ok(StateUpdate::full(new_state))
        })
        .add_edge(START, "process")
        .add_edge("process", "finalize")
        .add_edge("finalize", END)
        .compile()
        .unwrap()
        // ========== 配置中断点 ==========
        // 关键：在 process 执行后中断
        // 执行流程：START → process 执行成功 → [执行完 process，被中断！]
        .with_interrupt_after(vec!["process".to_string()]);
    
    let input = AgentState::new("test".to_string());
    
    // ========== 第一次执行：被打断 ==========
    let result = compiled.invoke(input).await;
    //
    // 执行过程：
    //   START → process 执行成功 ✅
    //   process 执行完毕，状态已更新（messages 增加了 "processed"）
    //   检测到 interrupt_after 包含 "process" → 抛出 ExecutionInterrupted
    //   ⚠️ finalize 没有被执行！
    //   ⚠️ output 没有被设置！
    //
    
    // ========== 验证中断 ==========
    assert!(result.is_err());  // ← 这里体现被打断
    let err = result.unwrap_err();
    assert!(matches!(err, langchainrust::GraphError::ExecutionInterrupted(_)));
    
    let interrupted_at = err.to_string();
    assert!(interrupted_at.contains("after_process"));  // ← 错误信息显示在 process 之后被打断
}

/// 测试 resume - 从中断点恢复执行
///
/// 功能验证:
/// - resume() 能正确恢复中断的执行
/// - 从 interrupt_after 中断点继续
/// - 执行后续节点
/// - 最终成功完成
///
/// 框架作用:
/// - 这是 Human-in-the-loop 的核心功能
/// - 支持人工检查后继续
/// - 实现需要审批的自动化流程
#[tokio::test]
async fn test_resume_from_interrupt() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("step1", |state| Ok(StateUpdate::full(state.clone())))
        .add_node_fn("step2", |state| {
            let mut new_state = state.clone();
            new_state.add_message(langchainrust::MessageEntry::ai("step2_done".to_string()));
            Ok(StateUpdate::full(new_state))
        })
        .add_node_fn("step3", |state| {
            let mut new_state = state.clone();
            new_state.set_output("completed".to_string());
            Ok(StateUpdate::full(new_state))
        })
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .add_edge("step2", "step3")
        .add_edge("step3", END)
        .compile()
        .unwrap()
        .with_interrupt_after(vec!["step1".to_string()]);
    
    let input = AgentState::new("test".to_string());
    
    // ========== 第一阶段：被打断 ==========
    let result = compiled.invoke(input).await;
    //
    // 执行过程：
    //   START → step1 执行成功 ✅
    //   step1 完成后被打断 ⚠️
    //   step2、step3 未执行
    //
    assert!(result.is_err());  // ← 打断点
    
    // ========== 第二阶段：人工介入 ==========
    // 人工可以：
    // 1. 查看当前状态（step1 已执行，messages 可能有记录）
    // 2. 修改状态（比如修改 input、添加审批标记）
    // 3. 决定继续或终止
    //
    // 这里模拟人工决定继续，创建 GraphExecution 来保存中断状态
    let mut execution = GraphExecution::new(
        AgentState::new("test".to_string()),  // 人工可以修改这个状态
        "step2".to_string(),                  // 下一步要执行的节点（从 step1 的下一个节点）
        "after_step1".to_string(),            // 中断位置记录
    );
    execution.recursion_count = 1;           // 记录已执行的步数（step1 已执行）
    
    // ========== 第三阶段：恢复执行 ==========
    let resumed_result = compiled.resume(execution).await.unwrap();
    //
    // 恢复过程：
    //   从 step2 开始执行 ✅
    //   step2 执行成功，messages 增加 "step2_done"
    //   step3 执行成功，output 设置为 "completed"
    //   END → 流程完成 ✅
    //
    // 这是恢复点 ↑ resume 从中断处继续
    
    // ========== 验证恢复成功 ==========
    assert!(resumed_result.final_state.output.is_some());      // ← 恢复后完成
    assert_eq!(resumed_result.final_state.output.unwrap(), "completed");
    assert_eq!(resumed_result.recursion_count, 2);             // step2 + step3 = 2 步
}

/// 测试多次中断和恢复
///
/// 功能验证:
/// - 可以在多个节点设置中断点
/// - 每次恢复后可以再次中断
/// - 最终完成整个流程
///
/// 框架作用:
/// - 支持多阶段审批流程
/// - 每个阶段都可以人工干预
/// - 适合复杂决策链路
#[tokio::test]
async fn test_multiple_interrupts() {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("check1", |state| {
            let mut new_state = state.clone();
            new_state.add_message(langchainrust::MessageEntry::ai("check1_pass".to_string()));
            Ok(StateUpdate::full(new_state))
        })
        .add_node_fn("check2", |state| {
            let mut new_state = state.clone();
            new_state.add_message(langchainrust::MessageEntry::ai("check2_pass".to_string()));
            Ok(StateUpdate::full(new_state))
        })
        .add_node_fn("final", |state| {
            let mut new_state = state.clone();
            new_state.set_output("all_checks_passed".to_string());
            Ok(StateUpdate::full(new_state))
        })
        .add_edge(START, "check1")
        .add_edge("check1", "check2")
        .add_edge("check2", "final")
        .add_edge("final", END)
        .compile()
        .unwrap()
        // ========== 配置多个中断点 ==========
        // check1 和 check2 执行前都会被打断
        .with_interrupt_before(vec!["check1".to_string(), "check2".to_string()]);
    
    let input = AgentState::new("test".to_string());
    
    // ========== 第一轮：第一次被打断 ==========
    let result1 = compiled.invoke(input).await;
    //
    // 执行过程：
    //   START → 准备执行 check1 → 打断 ⚠️
    //   check1 未执行
    //
    assert!(result1.is_err());  // ← 第一次打断点
    assert!(result1.unwrap_err().to_string().contains("check1"));  // 在 check1 处打断
    
    // ========== 人工第一次审批 ==========
    // 人工查看状态，决定批准 check1
    let execution1 = GraphExecution::new(
        AgentState::new("test".to_string()),
        "check1".to_string(),   // 从 check1 继续
        "check1".to_string(),
    );
    
    // ========== 第一轮恢复：执行后再次被打断 ==========
    let result2 = compiled.resume(execution1).await;
    //
    // 恢复过程：
    //   check1 执行成功 ✅
    //   准备执行 check2 → 再次打断 ⚠️
    //   check2 未执行
    //
    assert!(result2.is_err());  // ← 第二次打断点
    assert!(result2.unwrap_err().to_string().contains("check2"));  // 在 check2 处打断
    
    // ========== 人工第二次审批 ==========
    // 人工查看 check1 的结果，决定批准 check2
    let mut execution2 = GraphExecution::new(
        AgentState::new("test".to_string()),
        "check2".to_string(),   // 从 check2 继续
        "check2".to_string(),
    );
    execution2.recursion_count = 1;  // check1 已执行
    execution2.state.add_message(langchainrust::MessageEntry::ai("check1_pass".to_string()));  // 模拟 check1 的结果
    
    // ========== 第二轮恢复：流程完成 ==========
    let result3 = compiled.resume(execution2).await.unwrap();
    //
    // 恢复过程：
    //   check2 执行成功 ✅
    //   final 执行成功 ✅
    //   END → 流程完成 ✅
    //
    
    // ========== 验证最终完成 ==========
    assert!(result3.final_state.output.is_some());  // ← 最终成功
}