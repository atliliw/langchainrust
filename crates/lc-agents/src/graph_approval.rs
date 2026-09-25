// lc-agents/src/graph_approval.rs
//! #2 convergence: approval-as-graph-interrupt.
//!
//! When an agent runs as a graph, an approval-gated tool is a
//! [`GraphNode`](lc_langgraph::GraphNode)
//! that suspends via the runtime interrupt protocol instead of holding the
//! approval signal inside the process:
//!
//! - **First pass** the node raises
//!   [`GraphError::InterruptRequest`](lc_langgraph::GraphError::InterruptRequest) whose
//!   payload is the approval context (tool + intended command). The graph
//!   checkpointer is then the **single persistence** — the separate
//!   [`crate::resume::ResumeStore`] is *not* used on this path.
//! - A human / caller feeds an [`ApprovalDecision`] back through
//!   [`lc_langgraph::compiled::CompiledGraph::resume_with_value`]; the same
//!   node re-enters **with** the decision and applies it — `Allow` / `Modify`
//!   run the tool, `Deny` skips it. Nothing is replayed.
//!
//! This collapses the two "resume" concepts (#1/#2) onto one set: the decision
//! *is* the graph resume value, and the graph checkpoint is the recovery
//! mechanism.

use async_trait::async_trait;
use lc_langgraph::{
    AgentState, GraphError, GraphNode, NodeConfig, NodeResult, StateUpdate, INTERRUPT_RESUME_KEY,
};
use std::sync::Arc;

use crate::approval::ApprovalDecision;

/// An approval-gated tool, as a graph node.
///
/// The command to perform is read from `state.output` (the agent loop wrote the
/// intended action there); the decision arrives via the resume-injected
/// `INTERRUPT_RESUME_KEY`. `actions` runs the tool with a JSON string argument
/// and returns the observation text — representing the side effect that must
/// only happen once, on the approved pass.
pub struct ApprovalGate {
    name: String,
    actions: Arc<dyn Fn(&str) -> String + Send + Sync>,
}

/// Rebuilds a state carrying `out` as its output (the bridge only needs the
/// input/output channels; the richer agent channels stay empty here).
fn carry_output(state: &AgentState, out: String) -> AgentState {
    let mut next = AgentState::new(state.input.clone());
    next.set_output(out);
    next
}

impl ApprovalGate {
    /// Create a graph approval node for tool `name`.
    pub fn new(
        name: impl Into<String>,
        actions: impl Fn(&str) -> String + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            actions: Arc::new(actions),
        }
    }
}

#[async_trait]
impl GraphNode<AgentState> for ApprovalGate {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(
        &self,
        state: &AgentState,
        config: Option<NodeConfig>,
    ) -> NodeResult<AgentState> {
        let resume: Option<ApprovalDecision> = config
            .and_then(|c| c.metadata.get(INTERRUPT_RESUME_KEY).cloned())
            .map(|v| {
                serde_json::from_value(v).map_err(|e| {
                    GraphError::ExecutionError(format!(
                        "approval resume value is not an ApprovalDecision: {e}"
                    ))
                })
            })
            .transpose()?;

        // The intended tool call (the loop staged it before entering this node).
        let command = state.output.clone().unwrap_or_default();

        match resume {
            // First pass: suspend, hand the approval context to the outside world.
            None => Err(GraphError::InterruptRequest {
                payload: serde_json::json!({
                    "kind": "tool_approval",
                    "tool": self.name,
                    "command": command,
                }),
            }),

            Some(ApprovalDecision::Allow) => {
                let obs = (self.actions)(&command);
                Ok(StateUpdate::full(carry_output(state, format!("ok:{obs}"))))
            }

            Some(ApprovalDecision::Deny { reason }) => {
                // Tool skipped entirely — no side effect.
                Ok(StateUpdate::full(carry_output(
                    state,
                    format!("denied:{reason}"),
                )))
            }

            Some(ApprovalDecision::Modify { arguments, note }) => {
                // B5: a Modify decision now delivers its rewritten `arguments` to the
                // `actions` bridge instead of discarding them — the staged command is
                // replaced by the operator/approval-approved arguments, so the rewritten
                // parameters actually reach the side effect.
                let command =
                    serde_json::to_string(&arguments).unwrap_or_else(|_| arguments.to_string());
                let obs = (self.actions)(&command);
                Ok(StateUpdate::full(carry_output(
                    state,
                    format!("modified({note}):{obs}"),
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_langgraph::{FileCheckpointer, GraphBuilder, ThreadSafeMemoryCheckpointer, END, START};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Linear graph: START -> approval -> END, with a side-effect counter on
    /// the tool execution.
    fn charge_graph(
        counter: Arc<AtomicUsize>,
        name: &str,
    ) -> lc_langgraph::compiled::CompiledGraph<AgentState> {
        let c = counter.clone();
        GraphBuilder::<AgentState>::new()
            .add_node(ApprovalGate::new(name, move |args| {
                c.fetch_add(1, Ordering::SeqCst);
                format!("charged {args}")
            }))
            .add_edge(START, name)
            .add_edge(name, END)
            .compile()
            .unwrap()
            .with_recursion_limit(10)
    }

    /// A state purely carrying the staged command to approve.
    fn staged(output: &str) -> AgentState {
        let mut s = AgentState::new("x");
        s.set_output(output);
        s
    }

    /// #2: the approval gate's first pass is a graph interrupt carrying the
    /// approval context; a decision resumes the SAME node through the graph and
    /// the tool runs exactly once (no separate ResumeStore).
    #[tokio::test]
    async fn modify_resumes_node_and_runs_tool_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        let compiled = charge_graph(counter.clone(), "charge")
            .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());

        let err = compiled.invoke(staged("credit_card 99")).await.unwrap_err();
        let (node, payload) = match err {
            GraphError::DynamicInterrupt { node, payload } => (node, payload),
            other => panic!("expected DynamicInterrupt, got {other:?}"),
        };
        assert_eq!(node, "charge");
        assert_eq!(payload["kind"], "tool_approval");
        assert_eq!(payload["command"], "credit_card 99");
        // Tool must NOT have run on the interrupt pass.
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        let d = ApprovalDecision::Modify {
            arguments: serde_json::json!({"amount": 99}),
            note: "approved by ops".to_string(),
        };
        let inv = compiled
            .resume_with_value("charge", serde_json::to_value(d).unwrap())
            .await
            .unwrap();
        let out = inv.final_state.output.as_deref().unwrap();
        // B5: the Modify bridge passes the rewritten arguments to the tool, so the
        // staged "credit_card 99" command is replaced by the operator-approved
        // `{"amount":99}` payload.
        assert!(out.starts_with("modified(approved by ops):charged"));
        assert!(
            out.contains(r#"{"amount":99}"#),
            "rewritten arguments must reach the tool bridge, got: {out}"
        );
        assert!(
            !out.contains("credit_card 99"),
            "staged command must be replaced by the rewritten arguments, got: {out}"
        );
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "tool must run exactly once"
        );
    }

    /// #2: a Deny decision resumes without executing the side-effecting tool.
    #[tokio::test]
    async fn deny_resumes_without_running_tool() {
        let counter = Arc::new(AtomicUsize::new(0));
        let compiled = charge_graph(counter.clone(), "charge")
            .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());

        let err = compiled.invoke(staged("credit_card 99")).await.unwrap_err();
        match err {
            GraphError::DynamicInterrupt { node, .. } => assert_eq!(node, "charge"),
            other => panic!("expected DynamicInterrupt, got {other:?}"),
        }
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        let d = ApprovalDecision::Deny {
            reason: "too expensive".to_string(),
        };
        let inv = compiled
            .resume_with_value("charge", serde_json::to_value(d).unwrap())
            .await
            .unwrap();
        assert!(inv
            .final_state
            .output
            .as_deref()
            .unwrap()
            .starts_with("denied:too expensive"));
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "denied tool must not run"
        );
    }

    /// #2: approval/suspend follows the graph file checkpoint alone (no
    /// ResumeStore) — a brand-new process over the same dir resumes the
    /// interrupted node with the decision.
    #[tokio::test]
    async fn approval_converges_on_graph_file_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();

        let build = |path: std::path::PathBuf, counter: Arc<AtomicUsize>| {
            charge_graph(counter, "charge").with_checkpointer(FileCheckpointer::new(path).unwrap())
        };

        let counter = Arc::new(AtomicUsize::new(0));

        // Process A: suspend.
        {
            let compiled = build(dir_path.clone(), counter.clone());
            let err = compiled.invoke(staged("credit_card 99")).await.unwrap_err();
            match err {
                GraphError::DynamicInterrupt { node, .. } => assert_eq!(node, "charge"),
                other => panic!("expected DynamicInterrupt, got {other:?}"),
            }
        } // graph dropped (process A gone)

        // Process B: brand-new graph over the same checkpoint dir, resume Allow.
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        let compiled2 = build(dir_path, counter.clone());
        let d = ApprovalDecision::Allow;
        let inv = compiled2
            .resume_with_value("charge", serde_json::to_value(d).unwrap())
            .await
            .unwrap();
        let out = inv.final_state.output.as_deref().unwrap();
        assert!(out.starts_with("ok:charged credit_card 99"), "got {out}");
        // Allow → the tool ran exactly once, on the resumed process.
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    /// #2: two distinct approval-gated tools on one graph both route through
    /// the graph interrupt/resume — sequential multi-tool approval converges on
    /// one persistence set.
    #[tokio::test]
    async fn sequential_multi_tool_approval_on_graph() {
        let first = Arc::new(AtomicUsize::new(0));
        let second = Arc::new(AtomicUsize::new(0));
        let f1 = first.clone();
        let f2 = second.clone();

        let compiled = GraphBuilder::<AgentState>::new()
            .add_node(ApprovalGate::new("send_email", move |args| {
                f1.fetch_add(1, Ordering::SeqCst);
                format!("emailed {args}")
            }))
            .add_node(ApprovalGate::new("remote_exec", move |args| {
                f2.fetch_add(1, Ordering::SeqCst);
                format!("ran {args}")
            }))
            .add_edge(START, "send_email")
            .add_edge("send_email", "remote_exec")
            .add_edge("remote_exec", END)
            .compile()
            .unwrap()
            .with_recursion_limit(10)
            .with_checkpointer(ThreadSafeMemoryCheckpointer::<AgentState>::new());

        // First tool interrupts.
        let err = compiled.invoke(staged("notify admin")).await.unwrap_err();
        match err {
            GraphError::DynamicInterrupt { node, .. } => assert_eq!(node, "send_email"),
            other => panic!("expected DynamicInterrupt, got {other:?}"),
        }
        assert_eq!(first.load(Ordering::SeqCst), 0, "nothing ran yet");

        // Resume send_email with Allow -> it runs, then the graph continues and
        // the NEXT approval gate suspends. Resume cascades tool->tool.
        let err = compiled
            .resume_with_value(
                "send_email",
                serde_json::to_value(ApprovalDecision::Allow).unwrap(),
            )
            .await
            .unwrap_err();
        match err {
            GraphError::DynamicInterrupt { node, .. } => assert_eq!(node, "remote_exec"),
            other => panic!("expected DynamicInterrupt, got {other:?}"),
        }
        assert_eq!(first.load(Ordering::SeqCst), 1, "approved tool ran once");

        // Resume remote_exec with Deny -> skipped, graph reaches END.
        let d = ApprovalDecision::Deny {
            reason: "no ssh".to_string(),
        };
        let inv = compiled
            .resume_with_value("remote_exec", serde_json::to_value(d).unwrap())
            .await
            .unwrap();
        assert!(inv
            .final_state
            .output
            .as_deref()
            .unwrap()
            .starts_with("denied:no ssh"));
        assert_eq!(first.load(Ordering::SeqCst), 1, "approved tool ran once");
        assert_eq!(second.load(Ordering::SeqCst), 0, "denied tool never ran");
    }
}
