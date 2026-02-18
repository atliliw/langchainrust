use crate::agent::{Agent, AgentAction};
use crate::tools::{Tool, ToolInput};
use std::collections::HashMap;
use std::sync::Arc;

pub const DEFAULT_MAX_ITERATIONS: usize = 10;

pub struct AgentExecutor {
    agent: Box<dyn Agent>,
    tools: Vec<Arc<dyn Tool>>,
    max_iterations: usize,
}

/// 执行结果，包含最终答案和执行过程信息
#[derive(Debug)]
pub struct ExecutionResult {
    /// 最终答案
    pub answer: String,
    /// 是否使用了工具
    pub used_tools: bool,
    /// 调用的工具名称列表
    pub tool_calls: Vec<String>,
    /// 迭代次数
    pub iterations: usize,
}

impl AgentExecutor {
    pub fn new(agent: Box<dyn Agent>, tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            agent,
            tools,
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    fn find_tool(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name)
    }

    pub async fn run(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        self.run_with_vars(input, HashMap::new()).await
    }

    /// 执行并返回详细信息（包含是否使用了工具）
    pub async fn run_with_details(&self, input: &str) -> Result<ExecutionResult, Box<dyn std::error::Error>> {
        self.run_with_vars_and_details(input, HashMap::new()).await
    }

    /// 带变量执行并返回详细信息
    pub async fn run_with_vars_and_details(
        &self,
        input: &str,
        vars: HashMap<String, String>,
    ) -> Result<ExecutionResult, Box<dyn std::error::Error>> {
        let mut iteration = 0;
        let mut intermediate_steps: Option<String> = None;
        let mut tool_calls = Vec::new();

        loop {
            if iteration >= self.max_iterations {
                return Err("达到最大迭代次数，未能生成答案".into());
            }
            iteration += 1;

            let action = self
                .agent
                .get_next_step_with_vars(input, intermediate_steps.as_deref(), &vars)
                .await
                .map_err(|e| e.0)?;

            match action {
                AgentAction::FinalAnswer(answer) => {
                    self.agent.add_memory(input, &answer);
                    return Ok(ExecutionResult {
                        answer,
                        used_tools: !tool_calls.is_empty(),
                        tool_calls,
                        iterations: iteration,
                    });
                }
                AgentAction::ToolCall(tool_name, params) => {
                    println!("[工具调用] {} {:?}", tool_name, params);
                    tool_calls.push(tool_name.clone());
                    
                    let tool = self
                        .find_tool(&tool_name)
                        .ok_or_else(|| format!("工具未找到: {}", tool_name))?;

                    let tool_input = ToolInput {
                        tool_name: tool_name.clone(),
                        parameters: params,
                    };

                    let output = tool
                        .invoke(tool_input)
                        .await
                        .map_err(|e| format!("工具 '{}' 执行失败: {}", tool_name, e))?;

                    println!("[工具结果] {}", output.result);

                    if output.success {
                        intermediate_steps = Some(output.result);
                    } else {
                        intermediate_steps = Some(format!("工具错误: {}", output.result));
                    }
                    if let Some(ref obs) = intermediate_steps {
                        self.agent.add_memory(input, obs);
                    }
                }
            }
        }
    }

    pub async fn run_with_vars(
        &self,
        input: &str,
        vars: HashMap<String, String>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let result = self.run_with_vars_and_details(input, vars).await?;
        Ok(result.answer)
    }
}
