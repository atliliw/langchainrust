//! Tool orchestration (P2-10): arranges a task's multiple tool calls into a dependency graph, executed in parallel + serial.
//!
//! A complex task is usually a chain of interdependent tool calls — one tool's output is the next one's input.
//! [`ToolOrchestrator`] declares the steps as a DAG (`depends_on` expresses the dependencies) and, at execution:
//!
//! 1. Kahn topological sort + cycle detection — cycles / undefined dependencies error out **before** execution;
//! 2. Round-robin progress: each round takes out the steps whose dependencies are all satisfied and runs them
//!    concurrently, capped by [`tokio::sync::Semaphore`] (default 4);
//! 3. Step arguments support `${id}` / `${id.field}` templates: prior steps' outputs are substituted into the
//!    arguments before execution (`${id}` references the whole output, `${id.field}` extracts an object field).
//!
//! The [`ToolCaller`] trait abstracts "call a tool by name", implemented by [`MCPGateway`] —
//! hanging multi-tool orchestration directly on the Gateway's `server:tool` namespace.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use futures_util::future::join_all;
use serde_json::Value;
use tokio::sync::Semaphore;

use crate::gateway::MCPGateway;
use lc_core::tools::ToolError;

/// Abstraction of "call one tool by its full name" (P2-10).
///
/// [`MCPGateway`] implements it (parses the `server:tool` string result into JSON, degrading to `Value::String`
/// on parse failure); custom callers can implement this trait to hook in any tool registry. Each call returns
/// structured output for `${id.field}` template references.
#[async_trait::async_trait]
pub trait ToolCaller: Send + Sync {
    /// Calls the tool and returns its structured output.
    async fn call(&self, tool: &str, args: Value) -> Result<Value, ToolError>;
}

/// A single step of an orchestration (P2-10).
///
/// `id` uniquely identifies this step within the orchestration (referenced by other steps' `depends_on` and `${id}`);
/// `tool` is the full tool name (`server:tool`); `args` may contain `${...}` templates, substituted with prior
/// steps' outputs at execution.
#[derive(Debug, Clone)]
pub struct ToolStep {
    /// Unique step identifier.
    pub id: String,
    /// Full tool name (e.g. `fs:read_file`).
    pub tool: String,
    /// Call arguments (supports `${id}` / `${id.field}` templates).
    pub args: Value,
    /// Step ids this step depends on; it only runs once they are all done.
    pub depends_on: Vec<String>,
}

impl ToolStep {
    /// Creates a step with an empty `depends_on` by default.
    pub fn new(id: impl Into<String>, tool: impl Into<String>, args: Value) -> Self {
        Self {
            id: id.into(),
            tool: tool.into(),
            args,
            depends_on: Vec::new(),
        }
    }

    /// Declares a prerequisite dependency (chainable).
    pub fn after(mut self, dep: impl Into<String>) -> Self {
        self.depends_on.push(dep.into());
        self
    }
}

/// Error category for tool orchestration (P2-10).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OrchestrateError {
    /// `depends_on` references an undefined step id.
    UnknownStep(String),
    /// Duplicated step id.
    DuplicateId(String),
    /// The dependency graph has a cycle; topological sort is impossible.
    Cycle,
    /// Orchestration did not converge (steps waiting on each other; shouldn't happen after cycle detection passes).
    Deadlock,
    /// The parameter template references an unexecuted step.
    MissingStep(String),
    /// The `id.field` template's field does not exist.
    MissingField(String, String),
    /// The underlying tool call failed.
    Tool(String),
    /// The concurrency gate is closed.
    SemaphoreClosed,
}

impl fmt::Display for OrchestrateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrchestrateError::UnknownStep(id) => {
                write!(f, "orchestration references an undefined step: {id}")
            }
            OrchestrateError::DuplicateId(id) => {
                write!(f, "orchestration step id is duplicated: {id}")
            }
            OrchestrateError::Cycle => write!(f, "orchestration has a circular dependency"),
            OrchestrateError::Deadlock => write!(
                f,
                "orchestration did not converge: steps waiting on each other"
            ),
            OrchestrateError::MissingStep(id) => {
                write!(f, "parameter template references an unexecuted step: {id}")
            }
            OrchestrateError::MissingField(id, field) => {
                write!(
                    f,
                    "parameter template field missing: step {id} has no field {field}"
                )
            }
            OrchestrateError::Tool(msg) => {
                write!(f, "tool call failed inside orchestration: {msg}")
            }
            OrchestrateError::SemaphoreClosed => {
                write!(f, "orchestration concurrency gate is closed")
            }
        }
    }
}

impl std::error::Error for OrchestrateError {}

impl From<ToolError> for OrchestrateError {
    fn from(e: ToolError) -> Self {
        OrchestrateError::Tool(e.to_string())
    }
}

/// Tool orchestrator (P2-10).
///
/// Holds the step DAG and a concurrency cap; `execute` advances in rounds: each round runs all steps whose
/// dependencies are satisfied concurrently (concurrency ≤ `max_concurrency`), storing results in a map for the
/// next round's `${id}` references. Cycle detection runs first; cycles / undefined dependencies are rejected outright.
#[derive(Debug)]
pub struct ToolOrchestrator {
    steps: Vec<ToolStep>,
    max_concurrency: usize,
}

impl Default for ToolOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolOrchestrator {
    /// Empty orchestrator, default concurrency cap 4.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            max_concurrency: 4,
        }
    }

    /// Sets the concurrency cap (maximum number of steps running at once).
    pub fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self
    }

    /// Adds a step (chainable).
    pub fn add_step(mut self, step: ToolStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Executes the whole DAG, returning a `step id → output` map.
    ///
    /// - Pre-check: step ids are unique, all dependencies defined, no cycles;
    /// - Each round takes the steps whose dependencies are satisfied and runs them concurrently (capped by a semaphore);
    /// - `${id}` / `${id.field}` templates in step arguments are substituted with prior results.
    pub async fn execute<C: ToolCaller>(
        &self,
        caller: &C,
    ) -> Result<HashMap<String, Value>, OrchestrateError> {
        self.validate()?;
        let semaphore = Semaphore::new(self.max_concurrency);
        let sem = &semaphore;
        let mut results: HashMap<String, Value> = HashMap::new();
        let mut pending: HashSet<&str> = self.steps.iter().map(|s| s.id.as_str()).collect();

        while !pending.is_empty() {
            let ready: Vec<&ToolStep> = self
                .steps
                .iter()
                .filter(|s| {
                    pending.contains(s.id.as_str())
                        && s.depends_on
                            .iter()
                            .all(|d| results.contains_key(d.as_str()))
                })
                .collect();
            if ready.is_empty() {
                // Shouldn't happen after topological detection passes; error out defensively rather than loop forever.
                return Err(OrchestrateError::Deadlock);
            }

            let mut futures = Vec::with_capacity(ready.len());
            for step in ready {
                let args = resolve_template(&step.args, &results)?;
                let tool = step.tool.clone();
                let id = step.id.clone();
                futures.push(async move {
                    let _guard = sem
                        .acquire()
                        .await
                        .map_err(|_| OrchestrateError::SemaphoreClosed)?;
                    let out = caller.call(&tool, args).await?;
                    Ok::<(String, Value), OrchestrateError>((id, out))
                });
            }
            for result in join_all(futures).await {
                let (id, out) = result?;
                pending.remove(id.as_str());
                results.insert(id, out);
            }
        }
        Ok(results)
    }

    /// Pre-check: ids unique, dependencies defined, dependency graph acyclic.
    fn validate(&self) -> Result<(), OrchestrateError> {
        let mut seen = HashSet::with_capacity(self.steps.len());
        let mut by_id: HashMap<&str, &ToolStep> = HashMap::with_capacity(self.steps.len());
        for s in &self.steps {
            if !seen.insert(s.id.as_str()) {
                return Err(OrchestrateError::DuplicateId(s.id.clone()));
            }
            by_id.insert(s.id.as_str(), s);
        }
        for s in &self.steps {
            for d in &s.depends_on {
                if !by_id.contains_key(d.as_str()) {
                    return Err(OrchestrateError::UnknownStep(d.clone()));
                }
            }
        }
        // Kahn topological sort: each step's in-degree is its deduplicated dependency count; it can only run once the count hits 0.
        let mut in_degree: HashMap<&str, usize> = HashMap::with_capacity(self.steps.len());
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();
        for s in &self.steps {
            let distinct: HashSet<&str> = s.depends_on.iter().map(|d| d.as_str()).collect();
            in_degree.insert(s.id.as_str(), distinct.len());
            for d in distinct {
                dependents.entry(d).or_default().push(&s.id);
            }
        }
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &n)| n == 0)
            .map(|(&k, _)| k)
            .collect();
        let mut done = 0usize;
        while let Some(id) = queue.pop_front() {
            done += 1;
            if let Some(deps) = dependents.get(id) {
                for &dep in deps {
                    let e = in_degree
                        .get_mut(dep)
                        .ok_or_else(|| OrchestrateError::MissingStep(dep.to_string()))?;
                    *e -= 1;
                    if *e == 0 {
                        queue.push_back(dep);
                    }
                }
            }
        }
        if done != self.steps.len() {
            return Err(OrchestrateError::Cycle);
        }
        Ok(())
    }
}

/// Resolves `${id}` / `${id.field}` templates in the arguments.
///
/// Substitutes only when the whole string is exactly a single `${...}` reference; plain text is kept as-is.
/// Walks the JSON tree: arrays element by element, objects field by field; reference parts are substituted with their prior outputs.
fn resolve_template(
    value: &Value,
    results: &HashMap<String, Value>,
) -> Result<Value, OrchestrateError> {
    match value {
        Value::String(s) => {
            if let Some(rest) = s.strip_prefix("${").and_then(|r| r.strip_suffix('}')) {
                let mut parts = rest.split('.');
                let Some(id) = parts.next() else {
                    return Err(OrchestrateError::MissingStep(rest.to_string()));
                };
                let cur = results
                    .get(id)
                    .ok_or_else(|| OrchestrateError::MissingStep(id.to_string()))?;
                let mut out = cur;
                for field in parts {
                    out = out.get(field).ok_or_else(|| {
                        OrchestrateError::MissingField(id.to_string(), field.to_string())
                    })?;
                }
                Ok(out.clone())
            } else {
                Ok(value.clone())
            }
        }
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(resolve_template(item, results)?);
            }
            Ok(Value::Array(out))
        }
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                out.insert(k.clone(), resolve_template(v, results)?);
            }
            Ok(Value::Object(out))
        }
        other => Ok(other.clone()),
    }
}

#[async_trait::async_trait]
impl ToolCaller for MCPGateway {
    async fn call(&self, tool: &str, args: Value) -> Result<Value, ToolError> {
        let raw = MCPGateway::call(self, tool, args).await?;
        // The Gateway returns serialized text; if it parses as JSON, hand back the structured output,
        // otherwise degrade to a plain-text value (for a whole `${id}` reference).
        match serde_json::from_str(&raw) {
            Ok(v) => Ok(v),
            Err(_) => Ok(Value::String(raw)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{start_fake_sse_server, PostMode};
    use crate::{GatewayServerSpec, MCPConfig};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Scripted caller: returns structured output by tool name, making template substitution easy to assert.
    struct ScriptedCaller;

    #[async_trait::async_trait]
    impl ToolCaller for ScriptedCaller {
        async fn call(&self, tool: &str, args: Value) -> Result<Value, ToolError> {
            match tool {
                "sum" => Ok(json!({
                    "sum": args["a"].as_i64().unwrap_or(0) + args["b"].as_i64().unwrap_or(0)
                })),
                "upper" => Ok(json!({
                    "text": args["text"].as_str().unwrap_or("").to_uppercase()
                })),
                "echo" => Ok(args),
                _ => Err(ToolError::ExecutionFailed(format!("未知工具 {tool}"))),
            }
        }
    }

    /// Linear chain: b depends on a; the `${a.sum}` argument substitutes a's output.
    #[tokio::test]
    async fn test_linear_chain_resolves_template() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("a", "sum", json!({ "a": 1, "b": 2 })))
            .add_step(ToolStep::new("b", "sum", json!({ "a": "${a.sum}", "b": 4 })).after("a"));

        let results = orch.execute(&ScriptedCaller).await.unwrap();
        assert_eq!(results["a"]["sum"], 3);
        assert_eq!(
            results["b"]["sum"], 7,
            "${{a.sum}} should be substituted with a's result 3"
        );
    }

    /// Parallel dependencies: x and y are independent and run in parallel; z aggregates both.
    #[tokio::test]
    async fn test_parallel_deps_join() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("x", "sum", json!({ "a": 2, "b": 3 })))
            .add_step(ToolStep::new("y", "sum", json!({ "a": 4, "b": 1 })))
            .add_step(
                ToolStep::new("z", "sum", json!({ "a": "${x.sum}", "b": "${y.sum}" }))
                    .after("x")
                    .after("y"),
            );

        let results = orch.execute(&ScriptedCaller).await.unwrap();
        assert_eq!(results["z"]["sum"], 10, "x.sum=5 and y.sum=5 should sum");
    }

    /// Whole-result reference: the argument is just `${id}`, replaced with the prior step's complete output object.
    #[tokio::test]
    async fn test_whole_result_reference() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("s", "upper", json!({ "text": "hi" })))
            .add_step(ToolStep::new("t", "echo", json!("${s}")).after("s"));

        let results = orch.execute(&ScriptedCaller).await.unwrap();
        assert_eq!(
            results["t"],
            json!({ "text": "HI" }),
            "whole result should be substituted"
        );
    }

    /// Cycle detection: a → b → a reports Cycle before execution.
    #[tokio::test]
    async fn test_cycle_detected() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("a", "sum", json!({})).after("b"))
            .add_step(ToolStep::new("b", "sum", json!({})).after("a"));

        let err = orch.execute(&ScriptedCaller).await.unwrap_err();
        assert_eq!(err, OrchestrateError::Cycle);
    }

    /// Undefined dependency: reports UnknownStep before execution.
    #[tokio::test]
    async fn test_unknown_step_rejected() {
        let orch =
            ToolOrchestrator::new().add_step(ToolStep::new("a", "sum", json!({})).after("nope"));

        let err = orch.execute(&ScriptedCaller).await.unwrap_err();
        assert_eq!(err, OrchestrateError::UnknownStep("nope".to_string()));
    }

    /// Duplicate id: reports DuplicateId before execution.
    #[tokio::test]
    async fn test_duplicate_id_rejected() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("a", "sum", json!({})))
            .add_step(ToolStep::new("a", "sum", json!({})));

        let err = orch.execute(&ScriptedCaller).await.unwrap_err();
        assert_eq!(err, OrchestrateError::DuplicateId("a".to_string()));
    }

    /// Missing template field: a exists but has no `nope` field → MissingField.
    #[tokio::test]
    async fn test_missing_template_field() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("a", "sum", json!({ "a": 1, "b": 2 })))
            .add_step(ToolStep::new("b", "sum", json!({ "a": "${a.nope}", "b": 1 })).after("a"));

        let err = orch.execute(&ScriptedCaller).await.unwrap_err();
        assert_eq!(
            err,
            OrchestrateError::MissingField("a".to_string(), "nope".to_string())
        );
    }

    /// Concurrency cap: 6 independent steps with `max_concurrency=2`; concurrent executions ≤ 2.
    #[tokio::test]
    async fn test_concurrency_capped() {
        struct GatedCaller {
            in_flight: AtomicUsize,
            max_observed: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl ToolCaller for GatedCaller {
            async fn call(&self, _tool: &str, _args: Value) -> Result<Value, ToolError> {
                let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_observed.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(15)).await;
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(Value::String("done".into()))
            }
        }

        let mut orch = ToolOrchestrator::new().with_max_concurrency(2);
        for i in 0..6 {
            orch = orch.add_step(ToolStep::new(format!("s{i}"), "sum", json!({})));
        }
        let caller = GatedCaller {
            in_flight: AtomicUsize::new(0),
            max_observed: AtomicUsize::new(0),
        };
        let results = orch.execute(&caller).await.unwrap();
        assert_eq!(results.len(), 6, "all steps should complete");
        assert!(
            caller.max_observed.load(Ordering::SeqCst) <= 2,
            "concurrent executions must not exceed the concurrency cap, actual {}",
            caller.max_observed.load(Ordering::SeqCst)
        );
    }

    /// Integration with the real Gateway: both steps go through the fake SSE server `fs:echo`,
    /// the inter-step dependency holds, and the `impl ToolCaller for MCPGateway` path works end to end.
    #[tokio::test]
    async fn test_orchestrator_with_gateway() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let gateway = MCPGateway::new();
        gateway
            .register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
            .await
            .unwrap();

        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("one", "fs:echo", json!({ "msg": "hi" })))
            .add_step(ToolStep::new("two", "fs:echo", json!({})).after("one"));

        let results = orch.execute(&gateway).await.unwrap();
        assert_eq!(results["one"], Value::String("echo".to_string()));
        assert_eq!(results["two"], Value::String("echo".to_string()));
        assert_eq!(results.len(), 2);
    }
}
