// lc-agents/src/executor/engine.rs
//! `AgentExecutor` — the execution loop (plan -> act -> observe).

use super::budget::{budget_iteration_gate, budget_token_gate, budget_tool_gate, BudgetConfig};
use super::hooks::{run_after_completion_hooks, run_before_completion_hooks};
use super::tools::{
    execute_tool_for_stream, execute_tools_parallel_for_stream, tool_error_observation,
};
use super::{
    AgentError, BaseAgent, CACHE_NS, DEFAULT_MAX_CONCURRENCY, MAX_MAX_ITERATIONS,
    MIN_MAX_ITERATIONS,
};
use crate::approval::{ApprovalDecision, ApprovalHandler};
use crate::cache::ResponseCache;
use crate::hooks::{AgentHook, HookError};
use crate::metrics::AgentMetrics;
use crate::policy::ToolPolicy;
use crate::resume::{PendingApproval, ResumeStore};
use crate::streaming::state::AgentStreamEvent;
use crate::types::{AgentAction, AgentOutput, AgentStep, ToolInput};
use futures_util::Stream;
use lc_callbacks::{CallbackManager, RunTree, RunType};
use lc_core::runnables::RunnableConfig;
use lc_core::tools::BaseTool;
use lc_memory::BaseMemory;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// Agent executor.
///
/// Responsible for executing the agent's decision loop: Plan -> Act -> Observe.
pub struct AgentExecutor {
    /// Agent instance.
    pub(crate) agent: Arc<dyn BaseAgent>,

    /// Available tools.
    pub(crate) tools: Vec<Arc<dyn BaseTool>>,

    /// Max iterations.
    pub(crate) max_iterations: usize,

    /// Verbose output.
    pub(crate) verbose: bool,

    /// Memory (optional).
    pub(crate) memory: Option<Arc<tokio::sync::Mutex<dyn BaseMemory>>>,

    /// Callback manager (optional).
    pub(crate) callbacks: Option<Arc<CallbackManager>>,

    /// Agent hooks (optional).
    pub(crate) hooks: Vec<Arc<dyn AgentHook>>,

    /// Tool execution timeout (None = no timeout).
    pub(crate) tool_timeout: Option<Duration>,

    /// Maximum number of tools executed concurrently.
    pub(crate) max_concurrency: usize,

    /// Semaphore guarding concurrent tool execution.
    pub(crate) concurrency_sem: Arc<Semaphore>,

    /// Most recent execution metrics (P1-5). Arc-shared so merged executors
    /// created by `invoke_with_config` write back to the original executor.
    pub(crate) metrics_store: Arc<Mutex<Option<AgentMetrics>>>,

    /// LLM result cache (P2-1): `plan()` results hit on `(namespace, inputs, steps)`;
    /// deterministic prompts are reused directly, skipping the LLM round-trip.
    /// `None` = no caching.
    pub(crate) response_cache: Option<Arc<dyn ResponseCache>>,
    /// This instance's cache namespace (isolates executors sharing the same cache).
    pub(crate) cache_namespace: String,

    /// Tool permission policy (permission tiering + sandbox gate, P2-9).
    /// `None` = no checks.
    pub(crate) tool_policy: Option<ToolPolicy>,

    /// Approval gate (§4.2): async approval before each tool execution. `None` = no
    /// interception (default off).
    pub(crate) approval: Option<Arc<dyn ApprovalHandler>>,
    /// Budget gate (§4.2): hard limits. `None` = unlimited (default off).
    pub(crate) budget: Option<BudgetConfig>,

    /// Cross-process resume (§4.2): checkpoint store. When `Some`, `execute_tool`
    /// persists the pending approval before awaiting approval and clears it once the
    /// decision lands; a new process can inspect it via `pending_approval()` and
    /// continue via `resume(decision)`. `None` = off (default).
    pub(crate) resume_store: Option<Arc<dyn ResumeStore>>,
}

impl AgentExecutor {
    /// Creates a new AgentExecutor.
    pub fn new(agent: Arc<dyn BaseAgent>, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        Self {
            agent,
            tools,
            max_iterations: 10,
            verbose: false,
            memory: None,
            callbacks: None,
            hooks: Vec::new(),
            tool_timeout: None,
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
            concurrency_sem: Arc::new(Semaphore::new(DEFAULT_MAX_CONCURRENCY)),
            metrics_store: Arc::new(Mutex::new(None)),
            response_cache: None,
            cache_namespace: format!("exec-{}", CACHE_NS.fetch_add(1, Ordering::SeqCst)),
            tool_policy: None,
            approval: None,
            budget: None,
            resume_store: None,
        }
    }

    /// Sets max iterations, clamped to `[1, 100]`.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations.clamp(MIN_MAX_ITERATIONS, MAX_MAX_ITERATIONS);
        if max_iterations > MAX_MAX_ITERATIONS {
            log::warn!(
                "max_iterations {} clamped to {}",
                max_iterations,
                MAX_MAX_ITERATIONS
            );
        }
        self
    }

    /// Sets the tool execution timeout.
    ///
    /// A tool that exceeds the timeout returns an error instead of hanging the
    /// whole agent loop. `None` (the default) disables the timeout.
    pub fn with_tool_timeout(mut self, timeout: Duration) -> Self {
        self.tool_timeout = Some(timeout);
        self
    }

    /// Sets the maximum number of tools executed concurrently.
    ///
    /// Clamped to at least 1. The default is 8.
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        let max_concurrency = max_concurrency.max(1);
        self.max_concurrency = max_concurrency;
        self.concurrency_sem = Arc::new(Semaphore::new(max_concurrency));
        self
    }

    /// Enables the LLM result cache (P2-1).
    ///
    /// For deterministic prompts, `plan()` results with the same `(inputs,
    /// intermediate_steps)` are reused directly, skipping the LLM round-trip — suited to
    /// cost-sensitive / repeatedly-evaluated deterministic tasks. Tool execution results
    /// enter the cache key; tools themselves are not cached; the cache applies to the
    /// non-streaming `invoke` path.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let cache = Arc::new(MemoryCache::with_capacity(256));
    /// let executor = AgentExecutor::new(agent, tools).with_response_cache(cache);
    /// ```
    pub fn with_response_cache(mut self, cache: Arc<dyn ResponseCache>) -> Self {
        self.response_cache = Some(cache);
        self
    }

    /// Tool permission policy (permission tiering + sandbox gate, P2-9).
    ///
    /// Checked before every tool execution: tools whose risk exceeds `max_permitted`
    /// are rejected; high-risk tools that are not declared sandboxed
    /// ([`ToolPolicy::sandboxed`]) are also rejected. Unconfigured = everything allowed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let policy = ToolPolicy::new()
    ///     .risk("code_interpreter", ToolRisk::Dangerous)
    ///     .sandboxed("code_interpreter"); // moved into a restricted environment, allowed to run
    /// let executor = AgentExecutor::new(agent, tools).with_tool_policy(policy);
    /// ```
    pub fn with_tool_policy(mut self, policy: ToolPolicy) -> Self {
        self.tool_policy = Some(policy);
        self
    }

    /// Approval gate (§4.2): async approval before each tool execution.
    ///
    /// Default `None` = no interception; existing behavior unchanged. Approval
    /// decisions (implemented by the caller via [`ApprovalHandler`]):
    /// - [`ApprovalDecision::Allow`](crate::approval::ApprovalDecision::Allow): run as-is;
    /// - [`ApprovalDecision::Deny`](crate::approval::ApprovalDecision::Deny): skip the tool,
    ///   feed the reason back as an observation, and re-plan next round;
    /// - [`ApprovalDecision::Modify`](crate::approval::ApprovalDecision::Modify): run with the
    ///   new arguments substituted.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let executor = AgentExecutor::new(agent, tools)
    ///     .with_approval(Arc::new(AllowAll));
    /// ```
    pub fn with_approval(mut self, handler: Arc<dyn ApprovalHandler>) -> Self {
        self.approval = Some(handler);
        self
    }

    /// Budget gate (§4.2): hard limits, effective on both the `invoke` and `stream`
    /// paths.
    ///
    /// - `invoke`: any limit hit returns [`AgentError::BudgetExceeded`] and stops
    ///   immediately;
    /// - `stream`: any limit hit sends `Err(AgentError::BudgetExceeded)` on the channel
    ///   and stops.
    ///
    /// The caller can catch this error to distinguish a "budget stop" from "the model did
    /// not converge". Default `None` = unlimited.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let budget = BudgetConfig {
    ///     max_tool_calls: Some(3),
    ///     max_tokens: Some(10_000),
    ///     max_duration: Some(Duration::from_secs(60)),
    ///     max_iterations: Some(5),
    /// };
    /// let executor = AgentExecutor::new(agent, tools).with_budget(budget);
    /// ```
    pub fn with_budget(mut self, budget: BudgetConfig) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Cross-process resume (§4.2): checkpoint store.
    ///
    /// When enabled, before each tool call enters the approval gate to await approval,
    /// the framework writes the pending tool + the context needed to resume the agent
    /// loop ([`PendingApproval`]) into the store; it is cleared once the approval
    /// decision **lands**. If the process crashes, the checkpoint stays on disk; a new
    /// process rebuilding an executor with the same configuration calls
    /// [`pending_approval`](Self::pending_approval) / [`resume`](Self::resume) to
    /// continue instead of replaying the whole conversation from scratch.
    ///
    /// Applies only to the non-streaming `invoke` path (the streaming path has no
    /// approval gate); only meaningful together with
    /// [`with_approval`](Self::with_approval). Parallel tool execution (multiple tools
    /// approved concurrently) does not participate in cross-process persistence — the
    /// in-process approval still works.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let store = Arc::new(FileResumeStore::new("/var/checkpoints/app")?);
    /// let executor = AgentExecutor::new(agent, tools)
    ///     .with_resume_store(store)
    ///     .with_approval(Arc::new(MyHandler));
    /// ```
    pub fn with_resume_store(mut self, store: Arc<dyn ResumeStore>) -> Self {
        self.resume_store = Some(store);
        self
    }

    /// Tool registration validation (P2-2).
    ///
    /// The Agent declares the tool names it may call via `get_allowed_tools()`; when it
    /// declares some, every one must be present in this executor's `tools`, otherwise an
    /// error is returned listing all missing tools. When the Agent declares nothing
    /// (returns `None`, e.g. a base Agent with no tools), validation is skipped.
    ///
    /// Called before each `invoke` / `stream`: startup fail-fast, turning a mid-loop
    /// `ToolNotFound` into a one-shot, all-configuration-errors-at-once report before
    /// first execution.
    pub fn validate_tool_registration(&self) -> Result<(), AgentError> {
        let Some(allowed) = self.agent.get_allowed_tools() else {
            return Ok(());
        };
        let registered: HashSet<&str> = self.tools.iter().map(|t| t.name()).collect();
        let missing: Vec<&str> = allowed
            .into_iter()
            .filter(|name| !registered.contains(name))
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        Err(AgentError::ToolNotFound(format!(
            "tools not registered on executor: {}",
            missing.join(", ")
        )))
    }

    /// Sets verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Sets memory.
    pub fn with_memory(mut self, memory: Arc<tokio::sync::Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Sets callback manager.
    pub fn with_callbacks(mut self, callbacks: Arc<CallbackManager>) -> Self {
        self.callbacks = Some(callbacks);
        self
    }

    /// Adds an agent hook.
    pub fn hook(mut self, hook: impl AgentHook + 'static) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    /// Returns metrics from the most recent invocation, if any.
    pub fn last_metrics(&self) -> Option<AgentMetrics> {
        self.metrics_store.lock().ok().and_then(|g| g.clone())
    }

    /// Reads the currently pending approval checkpoint (cross-process resume).
    ///
    /// Returns `Ok(None)` when no [`ResumeStore`] is configured or the store is empty.
    /// After getting a [`PendingApproval`], the caller shows `tool_name` / `arguments`
    /// to an operator, collects the approval decision, then calls
    /// [`resume`](Self::resume) to continue.
    pub async fn pending_approval(&self) -> Result<Option<PendingApproval>, AgentError> {
        let Some(store) = &self.resume_store else {
            return Ok(None);
        };
        store
            .load_pending()
            .await
            .map_err(|e| AgentError::Resume(e.to_string()))
    }

    /// Resumes from a checkpoint (cross-process resume): processes the pending tool with
    /// the given decision, then continues the agent loop from the suspended iteration and
    /// returns the final answer.
    ///
    /// - No [`ResumeStore`] configured or no checkpoint → `Ok(None)` (no-op).
    /// - A checkpoint exists → first **claims** it (clears it) to prevent duplicate
    ///   approval, executes the pending tool, then continues the loop from
    ///   `iteration + 1`; budgets (tool / token / iteration) keep counting from the
    ///   checkpoint's accumulated amounts, and `max_duration` restarts its timer at the
    ///   resume moment (a cross-process monotonic clock is not portable — an honest
    ///   approximation).
    ///
    /// The resuming executor must be constructed identically to the one before the crash
    /// (same agent / tools / store directory) to resume correctly; the approval decision
    /// is injected by the caller and [`ApprovalHandler`] is not re-run.
    pub async fn resume(&self, decision: ApprovalDecision) -> Result<Option<String>, AgentError> {
        let Some(store) = &self.resume_store else {
            return Ok(None);
        };
        let Some(pending) = store
            .load_pending()
            .await
            .map_err(|e| AgentError::Resume(e.to_string()))?
        else {
            return Ok(None);
        };
        // Claim the checkpoint: clear it first. If resume crashes midway, approval is
        // not repeated (at most once).
        store
            .clear_pending()
            .await
            .map_err(|e| AgentError::Resume(e.to_string()))?;

        let action = AgentAction {
            tool: pending.tool_name.clone(),
            tool_input: ToolInput::Object {
                value: pending.arguments.clone(),
            },
            log: String::new(),
        };

        let mut root_run = RunTree::new(
            "AgentExecutor",
            RunType::Chain,
            json!({"input": pending.inputs.get("input").cloned().unwrap_or_default()}),
        );
        // Reuse the original trace_id so the resumed tool child runs keep trace
        // continuity.
        if let Some(tid) = &pending.trace_id {
            if let Ok(id) = uuid::Uuid::parse_str(tid) {
                root_run.trace_id = Some(id);
                root_run = root_run.with_metadata("trace_id", json!(tid));
            }
        }

        let started = std::time::Instant::now();
        let mut metrics = AgentMetrics {
            trace_id: root_run.trace_id.map(|id| id.to_string()),
            tool_calls: pending.tool_calls_consumed,
            total_tokens: pending.tokens_consumed,
            ..Default::default()
        };

        // Execute the pending tool (inject the given decision; do not re-run the
        // approval handler).
        let observation = self
            .execute_tool_inner(&action, &root_run, None, Some(decision))
            .await?;
        let mut steps = pending.steps;
        steps.push(AgentStep::new(action, observation));

        let result = self
            .run_agent_loop_from(
                pending.inputs,
                steps,
                pending.iteration + 1,
                &mut root_run,
                &mut metrics,
            )
            .await;

        metrics.duration = started.elapsed();
        metrics.log_summary();
        if let Ok(mut guard) = self.metrics_store.lock() {
            *guard = Some(metrics);
        }
        result.map(Some)
    }

    /// Builds the `plan()` cache key: namespace + inputs + intermediate steps (including
    /// tool observations).
    ///
    /// A deterministic Agent always produces the same `AgentOutput` for the same
    /// `(inputs, steps)`, so this hash is the "LLM result" fingerprint; observations are
    /// part of the key, so the cache cannot wrongly hit across different tool results.
    fn cache_key(namespace: &str, inputs: &HashMap<String, String>, steps: &[AgentStep]) -> String {
        use std::hash::{Hash, Hasher};
        let payload = json!({ "ns": namespace, "inputs": inputs, "steps": steps });
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        payload.to_string().hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// One-stop lookup / `plan()` / write-back.
    ///
    /// On a hit, returns the previous `AgentOutput` directly and records
    /// `metrics.cache_hits`, without calling the LLM; on a miss, calls `plan()` and
    /// serializes the result back. Corrupt cache content degrades to re-planning.
    pub(crate) async fn plan_cached(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
        metrics: &mut AgentMetrics,
    ) -> Result<AgentOutput, AgentError> {
        let cache = self.response_cache.as_ref();
        let key = cache.map(|_| Self::cache_key(&self.cache_namespace, inputs, intermediate_steps));

        if let (Some(cache), Some(key)) = (cache, &key) {
            if let Some(cached) = cache.get(key) {
                match serde_json::from_str::<AgentOutput>(&cached) {
                    Ok(output) => {
                        metrics.cache_hits += 1;
                        log::debug!(target: "lc_agents::cache", "plan cache hit: {}", key);
                        return Ok(output);
                    }
                    Err(_) => {
                        log::warn!(target: "lc_agents::cache", "plan cache corrupt, re-plan");
                    }
                }
            }
        }

        metrics.llm_calls += 1;
        // P2-9: rate-limit / quota check before the LLM call (Reject → abort this round).
        run_before_completion_hooks(&self.hooks, inputs)?;
        let output = self.agent.plan(intermediate_steps, inputs).await?;
        let usage = self.agent.last_token_usage();
        if let Some(usage) = &usage {
            metrics.add_token_usage(usage);
        }
        // P2-9: accumulate the real token usage after the LLM call (for the rate-limit
        // hook's accounting).
        run_after_completion_hooks(&self.hooks, &output, usage.as_ref());

        if let (Some(cache), Some(key)) = (cache, &key) {
            if let Ok(serialized) = serde_json::to_string(&output) {
                cache.put(key.clone(), serialized);
            }
        }
        Ok(output)
    }

    /// Executes the agent.
    pub async fn invoke(&self, input: String) -> Result<String, AgentError> {
        self.invoke_inner(input, None).await
    }

    /// Internal execution with optional trace propagation (P1-4).
    ///
    /// `trace_id` comes from `RunnableConfig.metadata["trace_id"]` when present;
    /// `None` for plain `invoke`. The trace_id is stamped onto the root
    /// `RunTree` so every tool child run inherits it via `create_child`.
    async fn invoke_inner(
        &self,
        input: String,
        trace_id: Option<String>,
    ) -> Result<String, AgentError> {
        // P2-2: startup fail-fast — error out on unregistered tools before any LLM call.
        self.validate_tool_registration()?;

        let started = std::time::Instant::now();

        // Hooks: on_agent_start (P1-6)
        for hook in &self.hooks {
            if let Err(e) = hook.on_agent_start(&input) {
                log::warn!("Hook on_agent_start error: {}", e);
            }
        }

        let mut root_run = RunTree::new(
            "AgentExecutor",
            RunType::Chain,
            json!({"input": input.clone()}),
        );

        // P1-4: stamp trace_id onto the root run so tool child runs inherit it.
        if let Some(tid) = trace_id {
            match uuid::Uuid::parse_str(&tid) {
                Ok(id) => {
                    root_run.trace_id = Some(id);
                    root_run = root_run.with_metadata("trace_id", json!(tid));
                }
                Err(_) => log::warn!(target: "lc_agents", "invalid trace_id '{}' ignored", tid),
            }
        }

        if let Some(ref callbacks) = self.callbacks {
            for handler in callbacks.handlers() {
                handler.on_chain_start(&root_run, &root_run.inputs).await;
            }
        }

        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input.clone());

        if let Some(memory) = &self.memory {
            let memory_guard = memory.lock().await;
            // P1-7: inject every key from memory_variables() into the prompt rather than
            // hardcoding "history" — so the Agent can also read it when the memory
            // component uses a different key (e.g. VectorStore's "memory").
            let variable_keys: Vec<String> = memory_guard
                .memory_variables()
                .into_iter()
                .map(|k| k.to_string())
                .collect();
            let memory_vars = memory_guard
                .load_memory_variables(&inputs)
                .await
                .map_err(|e| AgentError::Other(format!("Failed to load memory: {}", e)))?;
            drop(memory_guard);

            for key in variable_keys {
                if let Some(value) = memory_vars.get(&key) {
                    if let Some(s) = value.as_str() {
                        inputs.insert(key, s.to_string());
                    }
                }
            }
        }

        let intermediate_steps: Vec<AgentStep> = Vec::new();

        let mut metrics = AgentMetrics {
            trace_id: root_run.trace_id.map(|id| id.to_string()),
            ..Default::default()
        };

        let result = self
            .run_agent_loop(
                inputs.clone(),
                intermediate_steps,
                &mut root_run,
                &mut metrics,
            )
            .await;

        if let Some(memory) = &self.memory {
            match &result {
                Ok(output) => {
                    let mut outputs = HashMap::new();
                    outputs.insert("output".to_string(), output.clone());

                    memory
                        .lock()
                        .await
                        .save_context(&inputs, &outputs)
                        .await
                        .map_err(|e| AgentError::Other(format!("Failed to save memory: {}", e)))?;
                }
                // F7: the errored round is still part of the conversation — write the
                // user input + error message back to memory, otherwise the next round
                // loses context (the user's previous input is entirely gone). The error
                // text is saved as `output`, honoring save_context's input/output
                // two-key contract. A save failure only warns and does not mask the
                // original error.
                Err(e) => {
                    let mut outputs = HashMap::new();
                    outputs.insert("output".to_string(), format!("[error] {}", e));

                    if let Err(save_err) = memory.lock().await.save_context(&inputs, &outputs).await
                    {
                        log::warn!("failed to save errored round to memory: {}", save_err);
                    }
                }
            }
        }

        match &result {
            Ok(output) => {
                root_run.end(json!({"output": output}));
                if let Some(ref callbacks) = self.callbacks {
                    if let Some(ref outputs) = root_run.outputs {
                        for handler in callbacks.handlers() {
                            handler.on_chain_end(&root_run, outputs).await;
                        }
                    }
                }

                // Hooks: on_agent_end (P1-6)
                for hook in &self.hooks {
                    if let Err(e) = hook.on_agent_end(output) {
                        log::warn!("Hook on_agent_end error: {}", e);
                    }
                }
            }
            Err(e) => {
                root_run.end_with_error(e.to_string());
                if let Some(ref callbacks) = self.callbacks {
                    for handler in callbacks.handlers() {
                        handler.on_chain_error(&root_run, &e.to_string()).await;
                    }
                }

                // Hooks: on_error (P1-6)
                for hook in &self.hooks {
                    hook.on_error(&HookError::Other(e.to_string()));
                }
            }
        }

        // P1-5: finalize and publish metrics.
        metrics.duration = started.elapsed();
        metrics.log_summary();
        if let Ok(mut store) = self.metrics_store.lock() {
            *store = Some(metrics);
        }

        result
    }

    /// Execute the agent with a RunnableConfig, merging config callbacks
    /// with the executor's own callbacks.
    ///
    /// This is the entry point used by `AgentRunnable` (LCEL adapter).
    /// Config callbacks take precedence over the executor's callbacks.
    pub async fn invoke_with_config(
        &self,
        input: String,
        config: Option<RunnableConfig>,
    ) -> Result<String, AgentError> {
        // If config has callbacks, temporarily use them; otherwise use executor's own
        let effective_callbacks = config
            .as_ref()
            .and_then(|c| c.callbacks.clone())
            .or_else(|| self.callbacks.clone());

        // P1-4: thread trace_id from config metadata so child runs inherit it.
        let trace_id = config
            .as_ref()
            .and_then(|c| c.metadata.get("trace_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Create a temporary executor with merged callbacks; metrics_store is
        // Arc-shared so metrics written here propagate back to this executor.
        let merged_executor = AgentExecutor {
            agent: self.agent.clone(),
            tools: self.tools.clone(),
            max_iterations: self.max_iterations,
            verbose: self.verbose,
            memory: self.memory.clone(),
            callbacks: effective_callbacks,
            hooks: self.hooks.clone(),
            tool_timeout: self.tool_timeout,
            max_concurrency: self.max_concurrency,
            concurrency_sem: self.concurrency_sem.clone(),
            metrics_store: self.metrics_store.clone(),
            response_cache: self.response_cache.clone(),
            cache_namespace: self.cache_namespace.clone(),
            tool_policy: self.tool_policy.clone(),
            approval: self.approval.clone(),
            budget: self.budget.clone(),
            resume_store: self.resume_store.clone(),
        };

        merged_executor.invoke_inner(input, trace_id).await
    }

    /// Stream agent execution as a true async stream of events.
    ///
    /// Each step of the agent loop (tool calls, observations, final answer)
    /// is emitted as an `AgentStreamEvent` as soon as it occurs.
    ///
    /// # `Text` event granularity (F3, honest)
    ///
    /// `Text` events carry model text, but their granularity depends on the
    /// agent's [`BaseAgent::plan_stream`] implementation:
    ///
    /// * **ReAct and FunctionCalling agents** stream from the model's chat API,
    ///   so `Text` events arrive **per token** — concat them as they come for a
    ///   live word-stream. A function-calling step that calls a tool streams
    ///   back empty model text (tool calls aren't carried in stream chunks);
    ///   such steps fall back to the non-streaming path internally, so no
    ///   phantom empty `Text` is emitted.
    /// * **Other agents** (plan-and-execute without a streaming inner agent, …)
    ///   use the non-streaming default, so the whole final answer arrives as a
    ///   single `Text` event immediately before `FinalAnswer`.
    ///
    /// `ToolStart`/`ToolEnd` events are always emitted per tool call.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut stream = executor.stream("What is Rust?".to_string());
    /// while let Some(event) = stream.next().await {
    ///     match event {
    ///         Ok(AgentStreamEvent::ToolStart { name, input }) => { /* show tool call */ }
    ///         Ok(AgentStreamEvent::ToolEnd { name, output }) => { /* show result */ }
    ///         Ok(AgentStreamEvent::Text { content }) => { print!("{}", content); } /* model text */
    ///         Ok(AgentStreamEvent::FinalAnswer { content }) => { /* show answer */ }
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn stream(
        &self,
        input: String,
    ) -> Pin<Box<dyn Stream<Item = Result<AgentStreamEvent, AgentError>> + Send>> {
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        // P2-2: the streaming path also fails fast — unregistered tools emit one error
        // event before ending.
        if let Err(e) = self.validate_tool_registration() {
            tokio::spawn(async move {
                let _ = tx.send(Err(e)).await;
            });
            return Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx));
        }

        let agent = self.agent.clone();
        let tools = self.tools.clone();
        let max_iterations = self.max_iterations;
        let verbose = self.verbose;
        let tool_timeout = self.tool_timeout;
        let max_concurrency = self.max_concurrency;
        let hooks = self.hooks.clone();
        let tool_policy = self.tool_policy.clone();
        let budget = self.budget.clone();
        let metrics_store = self.metrics_store.clone();

        tokio::spawn(async move {
            let mut intermediate_steps: Vec<AgentStep> = Vec::new();
            let mut inputs = HashMap::new();
            inputs.insert("input".to_string(), input);

            // Budget gate (§4.2): start the stream timer + accumulate metrics (same
            // semantics as the invoke path).
            let loop_start = Instant::now();
            let mut metrics = AgentMetrics::default();

            for iteration in 0..max_iterations {
                if verbose {
                    log::info!("=== Stream Iteration {} ===", iteration + 1);
                }

                // Budget gate: iteration-level (iteration count + wall-clock). Over the
                // limit → send Err and stop.
                if let Some(err) =
                    budget_iteration_gate(budget.as_ref(), max_iterations, iteration, loop_start)
                {
                    publish_metrics(&metrics, &metrics_store, loop_start);
                    let _ = tx.send(Err(err)).await;
                    return;
                }

                // P2-9: rate-limit / quota check before the LLM call (also applies on
                // the streaming path).
                if let Err(e) = run_before_completion_hooks(&hooks, &inputs) {
                    publish_metrics(&metrics, &metrics_store, loop_start);
                    let _ = tx
                        .send(Ok(AgentStreamEvent::Error {
                            message: e.to_string(),
                        }))
                        .await;
                    return;
                }
                // F3: streaming planning — the agent forwards model text token by token
                // through on_token as Text events. ReAct / FunctionCalling override
                // plan_stream to go through `stream_chat` for a real word-by-word stream;
                // other agents use the default implementation (the whole answer as a
                // single Text event), matching the old path's behavior.
                let output = {
                    // Must not shadow the outer tx: the closure's `move` would carry it
                    // away, and the ToolStart/FinalAnswer below would no longer be able
                    // to use the outer tx.
                    let send_tx = tx.clone();
                    // The callback receives its own String (F3): the async block owns
                    // the token directly instead of borrowing the argument, so the future
                    // is 'static and can be cast to a trait object with `as`.
                    let mut on_token = move |token: String| {
                        let tx = send_tx.clone();
                        Box::pin(async move {
                            let _ = tx.send(Ok(AgentStreamEvent::Text { content: token })).await;
                        }) as Pin<Box<dyn Future<Output = ()> + Send>>
                    };
                    match agent
                        .plan_stream(&intermediate_steps, &inputs, &mut on_token)
                        .await
                    {
                        Ok(o) => o,
                        Err(e) => {
                            publish_metrics(&metrics, &metrics_store, loop_start);
                            let _ = tx
                                .send(Ok(AgentStreamEvent::Error {
                                    message: e.to_string(),
                                }))
                                .await;
                            return;
                        }
                    }
                };
                let usage = agent.last_token_usage();
                // P2-9: accumulate the real token usage after the LLM call (same semantics
                // as plan_cached on the invoke path).
                metrics.llm_calls += 1;
                if let Some(u) = &usage {
                    metrics.add_token_usage(u);
                }
                run_after_completion_hooks(&hooks, &output, usage.as_ref());
                // Budget gate: cumulative tokens after the LLM call. Over the limit →
                // send Err and stop.
                if let Some(err) = budget_token_gate(budget.as_ref(), &metrics) {
                    publish_metrics(&metrics, &metrics_store, loop_start);
                    let _ = tx.send(Err(err)).await;
                    return;
                }

                match output {
                    AgentOutput::Finish(finish) => {
                        let content = finish.output().unwrap_or("").to_string();
                        // P1-8 streaming fusion: the model text was already emitted piece
                        // by piece by plan_stream through on_token (Text events); here
                        // only the FinalAnswer terminal event is sent — the full answer is
                        // not repeated.
                        publish_metrics(&metrics, &metrics_store, loop_start);
                        let _ = tx.send(Ok(AgentStreamEvent::FinalAnswer { content })).await;
                        return;
                    }

                    AgentOutput::Action(action) => {
                        // P2-9: the streaming path also enforces the tool permission
                        // policy.
                        if let Some(policy) = &tool_policy {
                            if let Err(e) = policy.check(&action.tool) {
                                publish_metrics(&metrics, &metrics_store, loop_start);
                                let _ = tx
                                    .send(Ok(AgentStreamEvent::Error {
                                        message: e.to_string(),
                                    }))
                                    .await;
                                return;
                            }
                        }
                        let tool_name = action.tool.clone();
                        let tool_input_str = match &action.tool_input {
                            ToolInput::String { value: s } => s.clone(),
                            ToolInput::Object { value: v } => {
                                serde_json::to_string(v).unwrap_or_default()
                            }
                        };

                        let _ = tx
                            .send(Ok(AgentStreamEvent::ToolStart {
                                name: tool_name.clone(),
                                input: tool_input_str.clone(),
                            }))
                            .await;

                        // Budget gate: check cumulative call count and wall-clock before
                        // the tool runs.
                        metrics.tool_calls += 1;
                        if let Some(err) = budget_tool_gate(budget.as_ref(), &metrics, loop_start) {
                            publish_metrics(&metrics, &metrics_store, loop_start);
                            let _ = tx.send(Err(err)).await;
                            return;
                        }

                        // Execute the tool. A tool failure becomes an observation fed
                        // back to the loop (S3), matching the streaming parallel path —
                        // it no longer aborts the whole stream.
                        let observation =
                            match execute_tool_for_stream(&tools, &action, tool_timeout).await {
                                Ok(obs) => obs,
                                Err(e) => tool_error_observation(&e),
                            };

                        let _ = tx
                            .send(Ok(AgentStreamEvent::ToolEnd {
                                name: tool_name,
                                output: observation.clone(),
                            }))
                            .await;

                        intermediate_steps.push(AgentStep::new(action, observation));
                    }

                    AgentOutput::Actions(actions) => {
                        // P2-9: parallel tools also pass the permission policy first.
                        if let Some(policy) = &tool_policy {
                            for action in &actions {
                                if let Err(e) = policy.check(&action.tool) {
                                    publish_metrics(&metrics, &metrics_store, loop_start);
                                    let _ = tx
                                        .send(Ok(AgentStreamEvent::Error {
                                            message: e.to_string(),
                                        }))
                                        .await;
                                    return;
                                }
                            }
                        }
                        for action in &actions {
                            let tool_name = action.tool.clone();
                            let tool_input_str = match &action.tool_input {
                                ToolInput::String { value: s } => s.clone(),
                                ToolInput::Object { value: v } => {
                                    serde_json::to_string(v).unwrap_or_default()
                                }
                            };

                            let _ = tx
                                .send(Ok(AgentStreamEvent::ToolStart {
                                    name: tool_name.clone(),
                                    input: tool_input_str,
                                }))
                                .await;
                        }

                        // Budget gate: check cumulative call count and wall-clock before
                        // the parallel tools run.
                        metrics.tool_calls += actions.len();
                        if let Some(err) = budget_tool_gate(budget.as_ref(), &metrics, loop_start) {
                            publish_metrics(&metrics, &metrics_store, loop_start);
                            let _ = tx.send(Err(err)).await;
                            return;
                        }

                        let observations = execute_tools_parallel_for_stream(
                            &tools,
                            &actions,
                            tool_timeout,
                            max_concurrency,
                        )
                        .await;

                        for (action, observation) in
                            actions.into_iter().zip(observations.into_iter())
                        {
                            let _ = tx
                                .send(Ok(AgentStreamEvent::ToolEnd {
                                    name: action.tool.clone(),
                                    output: observation.clone(),
                                }))
                                .await;

                            intermediate_steps.push(AgentStep::new(action, observation));
                        }
                    }
                }
            }

            // Max iterations reached: return a placeholder result; log it so a non-answer
            // is not mistaken for the real final answer
            log::warn!("agent reached max iterations; streaming a placeholder result (not the real final answer)");
            publish_metrics(&metrics, &metrics_store, loop_start);
            let finish = agent.return_stopped_response(&intermediate_steps);
            let content = finish.output().unwrap_or("").to_string();
            let _ = tx.send(Ok(AgentStreamEvent::FinalAnswer { content })).await;
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}

impl std::fmt::Debug for AgentExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentExecutor")
            .field("max_iterations", &self.max_iterations)
            .field("verbose", &self.verbose)
            .field("tools_count", &self.tools.len())
            .field("has_memory", &self.memory.is_some())
            .field("tool_timeout", &self.tool_timeout)
            .field("max_concurrency", &self.max_concurrency)
            .field("has_response_cache", &self.response_cache.is_some())
            .field("has_tool_policy", &self.tool_policy.is_some())
            .field("has_resume_store", &self.resume_store.is_some())
            .field(
                "has_metrics",
                &self
                    .metrics_store
                    .lock()
                    .ok()
                    .map(|guard| guard.is_some())
                    .unwrap_or(false),
            )
            .finish()
    }
}

/// Publishes `AgentMetrics` at the end of a stream (aligned with the invoke path):
/// clone → fill duration → audit log → write `metrics_store`.
///
/// **Ordering constraint (race)**: the stream closure runs in `tokio::spawn`, so every
/// termination path must **`publish_metrics` before `tx.send(terminal event)`** —
/// otherwise a consumer that checks `last_metrics()` immediately after draining the
/// stream may read `None` (the event arrived but the write has not happened yet).
fn publish_metrics(
    metrics: &AgentMetrics,
    metrics_store: &Arc<Mutex<Option<AgentMetrics>>>,
    started: Instant,
) {
    let mut m = metrics.clone();
    m.duration = started.elapsed();
    m.log_summary();
    if let Ok(mut guard) = metrics_store.lock() {
        *guard = Some(m);
    }
}
