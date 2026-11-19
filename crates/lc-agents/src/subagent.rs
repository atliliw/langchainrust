//! Subagent support: run a child agent in a **fresh context** and return only
//! its structured summary to the parent.
//!
//! This is deliberately different from the orchestrators
//! ([`crate::orchestrator::Supervisor`], [`crate::orchestrator::FanOutFanIn`],
//! [`crate::orchestrator::SequentialPipeline`]): those share one message
//! history and change *roles*; a subagent gets an empty history and only the
//! task text, so its intermediate tool calls and failures never pollute the
//! parent context.
//!
//! # Example
//!
//! ```ignore
//! use lc_agents::subagent::{SubagentConfig, SubagentTool};
//! use std::sync::Arc;
//!
//! // The factory builds a *fresh* executor for each call. It also receives the
//! // child depth so recursive subagents can enforce a depth cap.
//! let factory = Arc::new(|depth: usize| {
//!     // build an AgentExecutor with the tools/budget/approval you want the
//!     // child to have, then return it
//!     # let _ = depth; todo!()
//! });
//!
//! let tool = SubagentTool::new(
//!     SubagentConfig::new("researcher", "Delegate a research sub-task"),
//!     factory,
//! );
//! ```
//!
//! # Invariants
//!
//! 1. The child starts from an empty context; only `task` is passed in.
//! 2. Only the child's final answer is returned, wrapped in a small JSON
//!    envelope — no intermediate steps.
//! 3. `depth` is capped by [`SubagentConfig::max_depth`]; exceeding it returns
//!    a deterministic `ToolError::ExecutionFailed`, never a panic.
//! 4. Tool subset / budget / approval belong to the factory: it constructs the
//!    child executor, so the parent cannot silently grant more than it passes.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::executor::{AgentError, AgentExecutor};
use lc_core::tools::{BaseTool, ToolError};

/// Configuration for one subagent tool.
#[derive(Debug, Clone)]
pub struct SubagentConfig {
    /// Tool name the parent model sees.
    pub name: String,
    /// Tool description the parent model sees.
    pub description: String,
    /// Maximum nesting depth. Depth 0 is the first child; a subagent created
    /// with `depth >= max_depth` refuses to run.
    pub max_depth: usize,
}

impl SubagentConfig {
    /// Creates a config with the default depth cap (2).
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            max_depth: 2,
        }
    }

    /// Sets the nesting depth cap.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }
}

impl Default for SubagentConfig {
    fn default() -> Self {
        Self::new("subagent", "Delegate a self-contained sub-task to a child agent")
    }
}

/// Builds a fresh child executor for the given depth.
///
/// The factory owns the child's tool set, budget, approval handler and model;
/// `SubagentTool` only enforces depth and the summary-only contract.
pub type SubagentFactory =
    Arc<dyn Fn(usize) -> Result<Arc<AgentExecutor>, AgentError> + Send + Sync>;

/// A [`BaseTool`] that delegates to a child agent in an isolated context.
pub struct SubagentTool {
    config: SubagentConfig,
    factory: SubagentFactory,
    depth: usize,
}

impl SubagentTool {
    /// Creates a root-level subagent tool (depth 0).
    pub fn new(config: SubagentConfig, factory: SubagentFactory) -> Self {
        Self {
            config,
            factory,
            depth: 0,
        }
    }

    /// Sets the current depth (used by a factory when wiring nested subagents).
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Returns the configured depth cap.
    pub fn max_depth(&self) -> usize {
        self.config.max_depth
    }
}

#[derive(Debug, Deserialize)]
struct SubagentInput {
    task: String,
}

#[derive(Debug, Serialize)]
struct SubagentReport {
    status: &'static str,
    summary: String,
}

#[async_trait]
impl BaseTool for SubagentTool {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn description(&self) -> &str {
        &self.config.description
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Self-contained description of the sub-task. The child cannot see the parent conversation."
                }
            },
            "required": ["task"]
        }))
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        if self.depth >= self.config.max_depth {
            return Err(ToolError::ExecutionFailed(format!(
                "subagent '{}' depth limit reached (depth={}, max_depth={})",
                self.config.name, self.depth, self.config.max_depth
            )));
        }

        let args: SubagentInput = serde_json::from_str(&input).map_err(|e| {
            ToolError::InvalidInput(format!(
                "subagent '{}' expects {{\"task\": \"...\"}}: {e}",
                self.config.name
            ))
        })?;

        // A fresh executor is built per call: no parent messages, no shared
        // scratchpad. The child only sees `task`.
        let child = (self.factory)(self.depth + 1).map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "subagent '{}' factory failed: {e}",
                self.config.name
            ))
        })?;

        let output = child.invoke(args.task).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("subagent '{}' failed: {e}", self.config.name))
        })?;

        // Only the final answer crosses the boundary. `AgentExecutor::invoke`
        // already returns the finish output (not intermediate steps), but we
        // wrap it so the parent can branch on `status` without parsing prose.
        serde_json::to_string(&SubagentReport {
            status: "ok",
            summary: output,
        })
        .map_err(|e| ToolError::ExecutionFailed(format!("subagent report encode failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::types::{AgentFinish, AgentOutput};
    use crate::AgentStep;
    use lc_core::runnables::RunnableConfig;
    use std::pin::Pin;

    /// A minimal agent that immediately finishes with the input echoed back.
    struct EchoAgent;

    #[async_trait]
    impl crate::BaseAgent for EchoAgent {
        async fn plan(
            &self,
            _intermediate_steps: &[AgentStep],
            inputs: &HashMap<String, String>,
            _config: Option<&RunnableConfig>,
        ) -> Result<AgentOutput, AgentError> {
            let input = inputs.get("input").cloned().unwrap_or_default();
            Ok(AgentOutput::Finish(AgentFinish::new(
                format!("summary: {input}"),
                String::new(),
            )))
        }

        async fn plan_stream(
            &self,
            intermediate_steps: &[AgentStep],
            inputs: &HashMap<String, String>,
            on_token: &mut (dyn FnMut(
                String,
            ) -> Pin<Box<dyn std::future::Future<Output = ()> + Send>>
                  + Send),
            config: Option<&RunnableConfig>,
        ) -> Result<AgentOutput, AgentError> {
            let output = self.plan(intermediate_steps, inputs, config).await?;
            if let AgentOutput::Finish(finish) = &output {
                on_token(finish.output().unwrap_or("").to_string()).await;
            }
            Ok(output)
        }
    }

    fn echoing_factory() -> SubagentFactory {
        Arc::new(|_depth| Ok(Arc::new(AgentExecutor::new(Arc::new(EchoAgent), vec![]))))
    }

    #[tokio::test]
    async fn subagent_returns_only_summary_envelope() {
        let tool = SubagentTool::new(
            SubagentConfig::new("researcher", "delegate"),
            echoing_factory(),
        );
        let out = tool
            .run(json!({"task": "find X"}).to_string())
            .await
            .expect("subagent run");
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["status"], "ok");
        assert_eq!(parsed["summary"], "summary: find X");
        // The envelope is exactly {status, summary}: no intermediate steps leak.
        assert_eq!(parsed.as_object().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn subagent_depth_limit_rejects_recursion() {
        let tool = SubagentTool::new(
            SubagentConfig::new("researcher", "delegate").with_max_depth(2),
            echoing_factory(),
        )
        .with_depth(2);
        let err = tool
            .run(json!({"task": "too deep"}).to_string())
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::ExecutionFailed(ref m) if m.contains("depth limit")),
            "expected depth-limit error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn subagent_rejects_invalid_input() {
        let tool = SubagentTool::new(SubagentConfig::default(), echoing_factory());
        let err = tool.run("not json".to_string()).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn factory_receives_child_depth() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_factory = seen.clone();
        let factory: SubagentFactory = Arc::new(move |depth| {
            seen_factory.lock().unwrap().push(depth);
            Ok(Arc::new(AgentExecutor::new(Arc::new(EchoAgent), vec![])))
        });
        let tool = SubagentTool::new(SubagentConfig::default(), factory);
        tool.run(json!({"task": "x"}).to_string()).await.unwrap();
        assert_eq!(*seen.lock().unwrap(), vec![1]);
    }
}
