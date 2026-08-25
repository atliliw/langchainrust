// lc-agents/src/executor/agent_loop.rs
//! `AgentExecutor` 的决策循环:主 agent loop + 工具顺序/并行执行。
//!
//! 与 `executor.rs`(结构体 + 构建器 + invoke/stream 入口)、`plan.rs`(缓存规划)配合。

use super::engine::AgentExecutor;
use super::tools::run_tool_with_timeout;
use super::AgentError;
use crate::hooks::{ToolCallAction, ToolCallContext, ToolResultContext};
use crate::metrics::AgentMetrics;
use crate::types::{AgentAction, AgentOutput, AgentStep, ToolInput};
use lc_callbacks::{RunTree, RunType};
use serde_json::json;
use std::collections::HashMap;

impl AgentExecutor {
    /// Runs the agent loop.
    ///
    /// Accumulates `metrics` (LLM calls, tool calls, token usage) as it goes.
    pub(crate) async fn run_agent_loop(
        &self,
        inputs: HashMap<String, String>,
        mut intermediate_steps: Vec<AgentStep>,
        root_run: &mut RunTree,
        metrics: &mut AgentMetrics,
    ) -> Result<String, AgentError> {
        for iteration in 0..self.max_iterations {
            if self.verbose {
                log::info!("=== Iteration {} ===", iteration + 1);
            }

            let output = self
                .plan_cached(&intermediate_steps, &inputs, metrics)
                .await?;

            match output {
                AgentOutput::Finish(finish) => {
                    if self.verbose {
                        log::info!("Final answer: {:?}", finish.return_values);
                    }
                    return Ok(finish.output().unwrap_or("").to_string());
                }

                AgentOutput::Action(action) => {
                    metrics.tool_calls += 1;
                    if self.verbose {
                        log::info!("Action: {}({})", action.tool, action.tool_input);
                    }

                    let observation = self.execute_tool(&action, root_run).await?;

                    if self.verbose {
                        log::info!("Observation: {}", observation);
                    }

                    intermediate_steps.push(AgentStep::new(action, observation));
                }

                AgentOutput::Actions(actions) => {
                    metrics.tool_calls += actions.len();
                    if self.verbose {
                        log::info!("Parallel actions: {} count", actions.len());
                        for action in &actions {
                            log::info!("  - {}({})", action.tool, action.tool_input);
                        }
                    }

                    let observations = self.execute_tools_parallel(&actions, root_run).await?;

                    if self.verbose {
                        for (i, obs) in observations.iter().enumerate() {
                            log::info!("Observation {}: {}", i + 1, obs);
                        }
                    }

                    for (action, observation) in actions.into_iter().zip(observations.into_iter()) {
                        intermediate_steps.push(AgentStep::new(action, observation));
                    }
                }
            }
        }

        // 达到迭代上限返回占位串,不能只靠 verbose 可见:调用方无法区分真实答案与占位,
        // 这里记 error 级日志显式暴露
        log::warn!(
            "agent reached max iterations {} without returning a final answer; returning a placeholder result (not the real final answer)",
            self.max_iterations
        );

        let finish = self.agent.return_stopped_response(&intermediate_steps);
        Ok(finish.output().unwrap_or("").to_string())
    }

    /// Executes multiple tools in parallel.
    ///
    /// Collects successful results and reports failures as error observations
    /// rather than discarding partial results when one tool fails.
    /// Concurrency is capped by the executor's global `concurrency_sem`.
    async fn execute_tools_parallel(
        &self,
        actions: &[AgentAction],
        root_run: &RunTree,
    ) -> Result<Vec<String>, AgentError> {
        use futures_util::future::join_all;

        let sem = self.concurrency_sem.clone();
        let futures = actions.iter().map(|action| {
            let sem = sem.clone();
            async move {
                let _permit = sem
                    .acquire_owned()
                    .await
                    .map_err(|e| AgentError::Other(format!("concurrency semaphore closed: {e}")))?;
                self.execute_tool(action, root_run).await
            }
        });

        let results = join_all(futures).await;

        let mut observations = Vec::with_capacity(results.len());
        for result in results {
            match result {
                Ok(output) => observations.push(output),
                Err(e) => observations.push(format!("[Tool execution error: {}]", e)),
            }
        }
        Ok(observations)
    }

    /// Executes a single tool.
    async fn execute_tool(
        &self,
        action: &AgentAction,
        root_run: &RunTree,
    ) -> Result<String, AgentError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == action.tool)
            .ok_or_else(|| AgentError::ToolNotFound(action.tool.clone()))?;

        let _input_str = match &action.tool_input {
            ToolInput::String { value: s } => s.clone(),
            ToolInput::Object { value: v } => serde_json::to_string(v)
                .map_err(|e| AgentError::Other(format!("Failed to serialize tool input: {}", e)))?,
        };

        // Run hooks: on_before_tool_call
        let mut tool_ctx = ToolCallContext {
            name: action.tool.clone(),
            arguments: match &action.tool_input {
                ToolInput::String { value: s } => {
                    // If the string is valid JSON, parse it as a Value to avoid
                    // double-encoding when serde_json::to_string() is called later.
                    // Otherwise wrap it as Value::String.
                    serde_json::from_str::<serde_json::Value>(s)
                        .unwrap_or(serde_json::Value::String(s.clone()))
                }
                ToolInput::Object { value: v } => v.clone(),
            },
            tool_id: String::new(),
        };

        for hook in &self.hooks {
            match hook.on_before_tool_call(&mut tool_ctx) {
                ToolCallAction::Continue => {}
                ToolCallAction::Modify { name, arguments } => {
                    tool_ctx.name = name;
                    tool_ctx.arguments = arguments;
                }
                ToolCallAction::Reject { reason } => {
                    return Err(AgentError::Other(format!(
                        "Tool call rejected by hook: {}",
                        reason
                    )));
                }
                ToolCallAction::Skip => {
                    return Ok("[Skipped by hook]".to_string());
                }
            }
        }

        let tool_name = tool_ctx.name.clone();

        // P2-9: 工具权限策略(权限分级 + 沙箱门禁)。未配置则放行。
        if let Some(policy) = &self.tool_policy {
            policy.check(&tool_name)?;
        }

        let input_for_tool = serde_json::to_string(&tool_ctx.arguments)
            .unwrap_or_else(|_| tool_ctx.arguments.to_string());

        let mut tool_run = root_run.create_child(
            &tool_name,
            RunType::Tool,
            json!({"input": input_for_tool.clone()}),
        );

        if let Some(ref callbacks) = self.callbacks {
            for handler in callbacks.handlers() {
                handler
                    .on_tool_start(&tool_run, &tool_name, &input_for_tool)
                    .await;
            }
        }

        let tool_started = std::time::Instant::now();
        let result = run_tool_with_timeout(tool, input_for_tool.clone(), self.tool_timeout).await;
        let tool_duration_ms = tool_started.elapsed().as_millis();
        let trace = root_run
            .trace_id
            .map(|id| id.to_string())
            .unwrap_or_default();

        match result {
            Ok(output) => {
                // P1-6: tool call audit log with trace/input/duration/outcome.
                log::info!(
                    target: "lc_agents::audit",
                    "tool_call trace_id={} name={} input={} duration_ms={} outcome=ok",
                    trace,
                    tool_name,
                    input_for_tool,
                    tool_duration_ms
                );
                tool_run.end(json!({"output": output.clone()}));
                if let Some(ref callbacks) = self.callbacks {
                    for handler in callbacks.handlers() {
                        handler.on_tool_end(&tool_run, &output).await;
                    }
                }

                // Run hooks: on_after_tool_call
                let mut result_ctx = ToolResultContext {
                    name: tool_name,
                    result: output.clone(),
                    tool_id: String::new(),
                };
                for hook in &self.hooks {
                    if let Err(e) = hook.on_after_tool_call(&mut result_ctx) {
                        log::warn!("Hook on_after_tool_call error: {}", e);
                    }
                }

                Ok(result_ctx.result)
            }
            Err(e) => {
                log::info!(
                    target: "lc_agents::audit",
                    "tool_call trace_id={} name={} input={} duration_ms={} outcome=error:{}",
                    trace,
                    tool_name,
                    input_for_tool,
                    tool_duration_ms,
                    e
                );
                tool_run.end_with_error(e.to_string());
                if let Some(ref callbacks) = self.callbacks {
                    for handler in callbacks.handlers() {
                        handler.on_tool_error(&tool_run, &e.to_string()).await;
                    }
                }
                Err(AgentError::ToolExecutionError(e.to_string()))
            }
        }
    }
}
