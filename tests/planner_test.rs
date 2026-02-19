#[path = "common.rs"]
mod common;

use langchainrust::agent::{
    AgentExecutor, PlannedExecutor, ReActAgent, SimplePlannedExecutor, TaskPlanner,
};
use langchainrust::llms::LLM;
use langchainrust::memory::SimpleMemory;
use langchainrust::tools::Calculator;
use std::sync::Arc;

#[tokio::test]
async fn test_task_planner_simple_question() {
    println!("\n=== 测试简单问题（不需要分解）===\n");

    let llm = LLM::new(common::llm_config());
    let planner = TaskPlanner::new(llm);

    let plan = planner.plan("什么是 Rust 语言？").await.unwrap();

    println!("原始问题: {}", plan.original_question);
    println!("子任务数量: {}", plan.sub_tasks.len());
    for task in &plan.sub_tasks {
        println!("  [{}] {} (依赖前序: {})", task.id, task.description, task.depends_on_previous);
    }

    assert!(!plan.sub_tasks.is_empty());
}

#[tokio::test]
async fn test_task_planner_complex_question() {
    println!("\n=== 测试复杂问题（需要分解）===\n");

    let llm = LLM::new(common::llm_config());
    let planner = TaskPlanner::new(llm).with_max_sub_tasks(5);

    let plan = planner
        .plan("分析 Python 和 Rust 的区别，然后给出选择建议")
        .await
        .unwrap();

    println!("原始问题: {}", plan.original_question);
    println!("子任务数量: {}", plan.sub_tasks.len());
    for task in &plan.sub_tasks {
        println!("  [{}] {} (依赖前序: {})", task.id, task.description, task.depends_on_previous);
    }

    assert!(!plan.sub_tasks.is_empty());
    // 复杂问题应该分解为多个任务
    println!("任务分解成功，共 {} 个子任务", plan.sub_tasks.len());
}

#[tokio::test]
async fn test_task_planner_summarize() {
    println!("\n=== 测试结果汇总 ===\n");

    use langchainrust::agent::TaskResult;

    let llm = LLM::new(common::llm_config());
    let planner = TaskPlanner::new(llm);

    let results = vec![
        TaskResult {
            id: 1,
            description: "分析 Python 特点".to_string(),
            result: "Python 是解释型语言，语法简洁，适合快速开发。".to_string(),
            success: true,
        },
        TaskResult {
            id: 2,
            description: "分析 Rust 特点".to_string(),
            result: "Rust 是编译型语言，注重内存安全和高性能。".to_string(),
            success: true,
        },
    ];

    let summary = planner
        .summarize("比较 Python 和 Rust", &results)
        .await
        .unwrap();

    println!("汇总结果: {}", summary);
    assert!(!summary.is_empty());
}

#[tokio::test]
async fn test_simple_planned_executor() {
    println!("\n=== 测试 SimplePlannedExecutor ===\n");

    let llm = LLM::new(common::llm_config());
    let executor = SimplePlannedExecutor::new(llm);

    // 只测试规划功能
    let plan = executor.plan("解释什么是闭包").await.unwrap();

    println!("任务规划结果:");
    for task in &plan.sub_tasks {
        println!("  [{}] {}", task.id, task.description);
    }

    assert!(!plan.sub_tasks.is_empty());
}

#[tokio::test]
async fn test_planned_executor_with_agent() {
    println!("\n=== 测试 PlannedExecutor + Agent ===\n");

    let llm = LLM::new(common::llm_config());
    let tools: Vec<Arc<dyn langchainrust::tools::Tool>> = vec![];

    let agent = ReActAgent::new(llm.clone(), tools.clone(), None);
    let agent_executor = AgentExecutor::new(Box::new(agent), tools);

    let planned_executor = PlannedExecutor::new(llm, Box::new(ReActAgent::new(LLM::new(common::llm_config()), vec![], None)), vec![])
        .with_max_sub_tasks(3)
        .with_max_iterations(2);

    let result = planned_executor
        .run("简单介绍一下 Python 语言的特点")
        .await
        .unwrap();

    println!("\n最终结果: {}", result);
    assert!(!result.is_empty());
}

#[tokio::test]
async fn test_planned_executor_with_plan_details() {
    println!("\n=== 测试 PlannedExecutor 返回详细结果 ===\n");

    let llm = LLM::new(common::llm_config());

    let planned_executor = PlannedExecutor::new(
        llm,
        Box::new(ReActAgent::new(LLM::new(common::llm_config()), vec![], None)),
        vec![],
    )
    .with_max_sub_tasks(3);

    let (plan, results) = planned_executor
        .run_with_plan("介绍一下 JavaScript 的用途")
        .await
        .unwrap();

    println!("\n=== 规划详情 ===");
    println!("原始问题: {}", plan.original_question);
    println!("子任务数: {}", plan.sub_tasks.len());

    println!("\n=== 执行结果 ===");
    for result in &results {
        println!(
            "任务 {}: {} - {}",
            result.id,
            result.description,
            if result.success { "成功" } else { "失败" }
        );
        println!("  结果: {}", result.result);
    }

    assert!(!results.is_empty());
    // 至少有一个成功的结果
    assert!(results.iter().any(|r| r.success));
}

#[tokio::test]
async fn test_planned_executor_with_tools() {
    println!("\n=== 测试 PlannedExecutor + Tools ===\n");

    let llm = LLM::new(common::llm_config());
    let tools: Vec<Arc<dyn langchainrust::tools::Tool>> = vec![Arc::new(Calculator)];

    let planned_executor = PlannedExecutor::new(
        llm,
        Box::new(ReActAgent::new(LLM::new(common::llm_config()), tools.clone(), None)),
        tools,
    )
    .with_max_sub_tasks(2);

    let result = planned_executor
        .run("计算 10 + 20，然后解释结果")
        .await
        .unwrap();

    println!("\n最终结果: {}", result);
    assert!(!result.is_empty());
}

#[tokio::test]
async fn test_planned_executor_with_memory() {
    println!("\n=== 测试 PlannedExecutor + Memory ===\n");

    let llm = LLM::new(common::llm_config());

    let planned_executor = PlannedExecutor::new(
        llm,
        Box::new(ReActAgent::new(LLM::new(common::llm_config()), vec![], None)),
        vec![],
    )
    .with_memory(Box::new(SimpleMemory::default()))
    .with_max_sub_tasks(2);

    // 第一次执行
    let result1 = planned_executor
        .run("什么是变量？")
        .await
        .unwrap();
    println!("第一次结果: {}", result1);

    // 第二次执行（可以利用记忆）
    let result2 = planned_executor
        .run("它有什么类型？")
        .await
        .unwrap();
    println!("第二次结果: {}", result2);

    assert!(!result2.is_empty());
}

#[tokio::test]
async fn test_planner_json_parsing() {
    println!("\n=== 测试 JSON 解析 ===\n");

    let llm = LLM::new(common::llm_config());
    let planner = TaskPlanner::new(llm);

    // 直接测试解析功能（通过规划一个简单任务）
    let plan = planner.plan("计算 1+1").await.unwrap();

    println!("解析结果:");
    for task in &plan.sub_tasks {
        println!(
            "  id: {}, desc: {}, depends: {}",
            task.id, task.description, task.depends_on_previous
        );
    }

    // 验证 ID 顺序
    for (i, task) in plan.sub_tasks.iter().enumerate() {
        assert_eq!(task.id, i + 1);
    }
}

#[tokio::test]
async fn test_full_planning_workflow() {
    println!("\n=== 完整规划工作流演示 ===\n");

    // 1. 创建 LLM
    let llm = LLM::new(common::llm_config());

    // 2. 创建规划器
    let planner = TaskPlanner::new(llm.clone()).with_max_sub_tasks(3);

    // 3. 规划任务
    println!("步骤1: 任务规划");
    let plan = planner
        .plan("介绍 Go 语言，并比较它与 Rust 的性能")
        .await
        .unwrap();

    println!("  分解为 {} 个子任务:", plan.sub_tasks.len());
    for task in &plan.sub_tasks {
        println!("    - {}", task.description);
    }

    // 4. 模拟执行（使用 Agent）
    println!("\n步骤2: 执行子任务");
    let mut results = Vec::new();
    
    for task in &plan.sub_tasks {
        println!("  执行: {}", task.description);
        
        let task_llm = LLM::new(common::llm_config());
        let result = task_llm.invoke(&task.description).await.unwrap();
        
        results.push(langchainrust::agent::TaskResult {
            id: task.id,
            description: task.description.clone(),
            result,
            success: true,
        });
    }

    // 5. 汇总结果
    println!("\n步骤3: 汇总结果");
    let summary = planner
        .summarize(&plan.original_question, &results)
        .await
        .unwrap();

    println!("  最终答案: {}", summary);
    assert!(!summary.is_empty());
}
