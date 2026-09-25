//! T10 (v0.23): the agent executor stamps 2026 GenAI semconv metadata on its
//! run trees *before* dispatching start callbacks, so OTel (and any other
//! metadata-reading handler) sees:
//! - `gen_ai.operation.name="invoke_agent"` on the agent root chain run,
//! - `gen_ai.tool.description` on every tool child run (Recommended),
//! - `gen_ai.tool.call.id` only when the action carried a provider tool-call id
//!   (locally-planned actions have none).
//!
//! Both execution surfaces (invoke + stream) stamp the same metadata.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lc_agents::types::{AgentAction, AgentFinish, AgentOutput, AgentStep, ToolInput};
use lc_agents::{AgentError, AgentExecutor, BaseAgent};
use lc_callbacks::{CallbackHandler, CallbackManager, RunTree};
use lc_core::tools::{BaseTool, ToolError};
use tokio::sync::Mutex;

/// One search action, then a finish (drives exactly one tool child run).
struct SearchThenFinish;

#[async_trait]
impl BaseAgent for SearchThenFinish {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
        _config: Option<&lc_core::runnables::RunnableConfig>,
    ) -> Result<AgentOutput, AgentError> {
        if intermediate_steps.is_empty() {
            return Ok(AgentOutput::Action(AgentAction {
                tool: "web_search".to_string(),
                tool_input: ToolInput::Object {
                    value: serde_json::json!({"q": "rust"}),
                },
                log: "search".to_string(),
                tool_call_id: None,
            }));
        }
        Ok(AgentOutput::Finish(AgentFinish::new(
            "answer".to_string(),
            String::new(),
        )))
    }
}

struct SearchTool;

#[async_trait]
impl BaseTool for SearchTool {
    fn name(&self) -> &str {
        "web_search"
    }
    fn description(&self) -> &str {
        "searches the web"
    }
    async fn run(&self, _input: String) -> Result<String, ToolError> {
        Ok("result".to_string())
    }
}

/// Records the semconv metadata on the root chain run and the tool child run.
#[derive(Default)]
struct MetadataRecorder {
    root_operation: Mutex<Option<String>>,
    tool_description: Mutex<Option<String>>,
    tool_call_id: Mutex<Option<String>>,
}

#[async_trait]
impl CallbackHandler for MetadataRecorder {
    async fn on_run_start(&self, _run: &RunTree) {}
    async fn on_run_end(&self, _run: &RunTree) {}
    async fn on_run_error(&self, _run: &RunTree, _error: &str) {}

    async fn on_chain_start(&self, run: &RunTree, _inputs: &serde_json::Value) {
        // The executor root runs as "AgentExecutor"; ignore any nested chains.
        if run.name == "AgentExecutor" {
            *self.root_operation.lock().await = run
                .metadata
                .get("gen_ai.operation.name")
                .and_then(|v| v.as_str())
                .map(str::to_string);
        }
    }

    async fn on_tool_start(&self, run: &RunTree, _tool_name: &str, _input: &str) {
        *self.tool_description.lock().await = run
            .metadata
            .get("gen_ai.tool.description")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        *self.tool_call_id.lock().await = run
            .metadata
            .get("gen_ai.tool.call.id")
            .and_then(|v| v.as_str())
            .map(str::to_string);
    }
}

fn instrumented_executor(recorder: Arc<MetadataRecorder>) -> AgentExecutor {
    let manager = Arc::new(CallbackManager::new().add_handler(recorder));
    AgentExecutor::new(Arc::new(SearchThenFinish), vec![Arc::new(SearchTool)])
        .with_callbacks(manager)
}

#[tokio::test]
async fn invoke_root_and_tool_runs_carry_semconv_metadata() {
    let recorder = Arc::new(MetadataRecorder::default());
    let executor = instrumented_executor(recorder.clone());

    let out = executor.invoke("go".to_string()).await.expect("invoke ok");
    assert_eq!(out, "answer");

    assert_eq!(
        recorder.root_operation.lock().await.as_deref(),
        Some("invoke_agent"),
        "root chain run must be classified as invoke_agent before on_chain_start"
    );
    assert_eq!(
        recorder.tool_description.lock().await.as_deref(),
        Some("searches the web"),
        "tool child run must carry the tool description"
    );
    // B5: even locally-planned actions (which carry no provider id) get a
    // framework-generated uuid stamped into the RunTree, so observability /
    // resume never persist an empty tool-call id.
    let call_id = recorder.tool_call_id.lock().await.clone();
    assert!(
        matches!(call_id.as_deref(), Some(id) if !id.is_empty()),
        "the gate must stamp a non-empty tool-call id, got {:?}",
        call_id
    );
}

#[tokio::test]
async fn stream_root_run_carries_invoke_agent_metadata() {
    use futures_util::StreamExt;

    let recorder = Arc::new(MetadataRecorder::default());
    let executor = instrumented_executor(recorder.clone());

    let mut stream = executor.stream("go".to_string());
    let mut terminal = false;
    while let Some(event) = stream.next().await {
        if matches!(event, Ok(lc_agents::AgentStreamEvent::FinalAnswer { .. })) {
            terminal = true;
        }
        event.expect("stream event ok");
    }
    assert!(terminal, "stream must reach FinalAnswer");

    assert_eq!(
        recorder.root_operation.lock().await.as_deref(),
        Some("invoke_agent"),
        "streaming path must stamp the same root operation as invoke"
    );
    // B5: the streaming path now runs tools through the shared gate too (tool-level
    // RunTree + metadata); tool metadata is asserted in detail on invoke above.
}
