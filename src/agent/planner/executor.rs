use crate::agent::{Agent, AgentExecutor};
use crate::llms::LLM;
use crate::memory::Memory;
use crate::tools::Tool;
use std::sync::{Arc, Mutex};

use super::planner::TaskPlanner;
use super::types::{Plan, TaskResult};

/// 带任务规划能力的执行器
/// 
/// 自动将复杂任务分解为子任务，依次执行，最后汇总结果
pub struct PlannedExecutor {
    planner: TaskPlanner,
    agent_executor: AgentExecutor,
    memory: Option<Arc<Mutex<Box<dyn Memory>>>>,
}

impl PlannedExecutor {
    /// 创建新的规划执行器
    pub fn new(
        llm: LLM,
        agent: Box<dyn Agent>,
        tools: Vec<Arc<dyn Tool>>,
    ) -> Self {
        let planner = TaskPlanner::new(llm);
        let agent_executor = AgentExecutor::new(agent, tools);
        Self {
            planner,
            agent_executor,
            memory: None,
        }
    }

    /// 设置最大子任务数量
    pub fn with_max_sub_tasks(mut self, max: usize) -> Self {
        self.planner = self.planner.with_max_sub_tasks(max);
        self
    }

    /// 设置记忆模块
    pub fn with_memory(mut self, memory: Box<dyn Memory>) -> Self {
        self.memory = Some(Arc::new(Mutex::new(memory)));
        self
    }

    /// 设置最大迭代次数
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.agent_executor = self.agent_executor.with_max_iterations(max);
        self
    }

    /// 执行复杂任务（自动规划 + 执行 + 汇总）
    pub async fn run(&self, question: &str) -> Result<String, Box<dyn std::error::Error>> {
        let (plan, results) = self.run_with_plan(question).await?;
        
        // 汇总结果
        self.planner.summarize(&plan.original_question, &results).await
    }

    /// 执行任务并返回详细结果（包含规划信息）
    pub async fn run_with_plan(
        &self,
        question: &str,
    ) -> Result<(Plan, Vec<TaskResult>), Box<dyn std::error::Error>> {
        println!("[规划] 正在分析任务...");

        // 1. 规划任务
        let plan = self.planner.plan(question).await?;
        
        println!("[规划] 任务已分解为 {} 个子任务:", plan.sub_tasks.len());
        for task in &plan.sub_tasks {
            println!(
                "  [{}] {} {}",
                task.id,
                task.description,
                if task.depends_on_previous { "(依赖前序)" } else { "" }
            );
        }

        // 2. 执行子任务
        let mut results = Vec::new();
        let mut previous_result = String::new();

        for task in &plan.sub_tasks {
            println!("\n[执行] 任务 {}/{}: {}", task.id, plan.sub_tasks.len(), task.description);

            // 构建任务输入
            let task_input = if task.depends_on_previous && !previous_result.is_empty() {
                format!(
                    "{}\n\n前序任务结果：\n{}",
                    task.description, previous_result
                )
            } else {
                task.description.clone()
            };

            // 执行任务
            let result = self.agent_executor.run(&task_input).await;
            
            match result {
                Ok(output) => {
                    println!("[完成] 任务 {} 执行成功", task.id);
                    previous_result = output.clone();
                    results.push(TaskResult {
                        id: task.id,
                        description: task.description.clone(),
                        result: output,
                        success: true,
                    });
                }
                Err(e) => {
                    println!("[失败] 任务 {} 执行失败: {}", task.id, e);
                    let error_msg = format!("执行失败: {}", e);
                    previous_result = error_msg.clone();
                    results.push(TaskResult {
                        id: task.id,
                        description: task.description.clone(),
                        result: error_msg,
                        success: false,
                    });
                }
            }

            // 记录到记忆模块
            if let Some(mem) = &self.memory {
                let mut m = mem.lock().unwrap();
                m.add(&task.description, &previous_result);
            }
        }

        println!("\n[汇总] 正在汇总所有任务结果...");
        Ok((plan, results))
    }
}

/// 简化版：直接使用 LLM 进行任务规划和执行
pub struct SimplePlannedExecutor {
    planner: TaskPlanner,
}

impl SimplePlannedExecutor {
    pub fn new(llm: LLM) -> Self {
        let planner = TaskPlanner::new(llm);
        Self { planner }
    }

    /// 规划任务（不执行，只返回计划）
    pub async fn plan(&self, question: &str) -> Result<Plan, Box<dyn std::error::Error>> {
        self.planner.plan(question).await
    }

    /// 汇总结果
    pub async fn summarize(
        &self,
        original_question: &str,
        results: &[TaskResult],
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.planner.summarize(original_question, results).await
    }
}
