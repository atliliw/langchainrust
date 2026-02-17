use crate::agent::{Agent, AgentAction, AgentExecutor};
use crate::tools::{Tool, ToolInput};
use std::collections::HashMap;
use std::sync::Arc;

pub const DEFAULT_MAX_ITERATIONS: usize = 10;

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

    pub async fn run_with_vars(
        &self,
        input: &str,
        vars: HashMap<String, String>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut iteration = 0;
        let mut intermediate_steps: Option<String> = None;

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
                    return Ok(answer);
                }
                AgentAction::ToolCall(tool_name, params) => {
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
}
