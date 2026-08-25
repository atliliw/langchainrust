// lc-agents/src/executor/executor.rs
//! `AgentExecutor` — the execution loop (plan -> act -> observe).

use super::hooks::{run_after_completion_hooks, run_before_completion_hooks};
use super::tools::{execute_tool_for_stream, execute_tools_parallel_for_stream};
use super::{
    AgentError, BaseAgent, CACHE_NS, DEFAULT_MAX_CONCURRENCY, MAX_MAX_ITERATIONS,
    MIN_MAX_ITERATIONS,
};
use crate::cache::ResponseCache;
use crate::hooks::{AgentHook, HookError};
use crate::metrics::AgentMetrics;
use crate::policy::ToolPolicy;
use crate::streaming::state::AgentStreamEvent;
use crate::types::{AgentOutput, AgentStep, ToolInput};
use futures_util::Stream;
use lc_callbacks::{CallbackManager, RunTree, RunType};
use lc_core::runnables::RunnableConfig;
use lc_core::tools::BaseTool;
use lc_memory::BaseMemory;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;
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

    /// LLM 结果缓存(P2-1):`plan()` 结果按 `(namespace, inputs, steps)` 命中,
    /// 确定性 prompt 直接复用,跳过 LLM 往返。`None` = 不缓存。
    pub(crate) response_cache: Option<Arc<dyn ResponseCache>>,
    /// 当前实例的缓存命名空间(隔离不同 executor 共享同一缓存)。
    pub(crate) cache_namespace: String,

    /// 工具权限策略(权限分级 + 沙箱门禁,P2-9)。`None` = 不校验。
    pub(crate) tool_policy: Option<ToolPolicy>,
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

    /// 开启 LLM 结果缓存(P2-1)。
    ///
    /// 确定性 prompt 场景下,相同 `(输入, 中间步骤)` 的 `plan()` 结果直接复用,
    /// 跳过 LLM 往返,适合成本敏感 / 反复评测的确定性任务。工具执行结果会进入
    /// 缓存 key,工具本身不被缓存;缓存对非流式 `invoke` 路径生效。
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

    /// 工具权限策略(权限分级 + 沙箱门禁,P2-9)。
    ///
    /// 每次工具执行前校验:风险高于 `max_permitted` 的工具被拒绝;高风险工具
    /// 未声明沙箱化([`ToolPolicy::sandboxed`])时也被拒绝。未配置则完全放行。
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let policy = ToolPolicy::new()
    ///     .risk("code_interpreter", ToolRisk::Dangerous)
    ///     .sandboxed("code_interpreter"); // 已搬进受限环境,允许执行
    /// let executor = AgentExecutor::new(agent, tools).with_tool_policy(policy);
    /// ```
    pub fn with_tool_policy(mut self, policy: ToolPolicy) -> Self {
        self.tool_policy = Some(policy);
        self
    }

    /// 工具注册校验(P2-2)。
    ///
    /// Agent 通过 `get_allowed_tools()` 声明它可能调用的工具名;若声明了,
    /// 这些名字必须全部出现在本 executor 的 `tools` 中,否则返回错误并列全量
    /// 缺失工具。Agent 未声明(返回 `None`,如无工具的基础 Agent)则跳过校验。
    ///
    /// 每次 `invoke` / `stream` 开始前调用:启动期 fail-fast,把"循环中途
    /// `ToolNotFound`"提前为"首次执行前一次性报全量配置错误"。
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

    /// 生成 `plan()` 缓存的 key:命名空间 + 输入 + 中间步骤(含工具观察)。
    ///
    /// 确定性 Agent 对相同 `(inputs, steps)` 必产出相同 `AgentOutput`,
    /// 故该哈希即"LLM 结果"的指纹;观察进入 key,缓存不会跨工具结果误命中。
    fn cache_key(namespace: &str, inputs: &HashMap<String, String>, steps: &[AgentStep]) -> String {
        use std::hash::{Hash, Hasher};
        let payload = json!({ "ns": namespace, "inputs": inputs, "steps": steps });
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        payload.to_string().hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// 查缓存 / 调 `plan()` / 写缓存 三合一。
    ///
    /// 命中时直接返回上次的 `AgentOutput` 并记 `metrics.cache_hits`,不调 LLM;
    /// 未命中调 `plan()` 后序列化写回。缓存内容损坏时降级为重新规划。
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
        // P2-9: LLM 调用前的限流/配额校验(Reject → 中止本轮)。
        run_before_completion_hooks(&self.hooks, inputs)?;
        let output = self.agent.plan(intermediate_steps, inputs).await?;
        let usage = self.agent.last_token_usage();
        if let Some(usage) = &usage {
            metrics.add_token_usage(usage);
        }
        // P2-9: LLM 调用后累计真实 token 用量(供限流 hook 记账)。
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
        // P2-2: 启动期 fail-fast——工具未注册时直接报错,不做任何 LLM 调用。
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
            // P1-7: 遍历 memory_variables() 的所有 key 注入 prompt,
            // 而非硬编码 "history"——记忆组件用别的 key(如 VectorStore 的
            // "memory")时 Agent 也能读到。
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
            if let Ok(ref output) = result {
                let mut outputs = HashMap::new();
                outputs.insert("output".to_string(), output.clone());

                memory
                    .lock()
                    .await
                    .save_context(&inputs, &outputs)
                    .await
                    .map_err(|e| AgentError::Other(format!("Failed to save memory: {}", e)))?;
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
        };

        merged_executor.invoke_inner(input, trace_id).await
    }

    /// Stream agent execution as a true async stream of events.
    ///
    /// Each step of the agent loop (tool calls, observations, final answer)
    /// is emitted as an `AgentStreamEvent` as soon as it occurs.
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

        // P2-2: 流式同样 fail-fast,工具未注册时先抛一个错误事件再结束。
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

        tokio::spawn(async move {
            let mut intermediate_steps: Vec<AgentStep> = Vec::new();
            let mut inputs = HashMap::new();
            inputs.insert("input".to_string(), input);

            for iteration in 0..max_iterations {
                if verbose {
                    log::info!("=== Stream Iteration {} ===", iteration + 1);
                }

                // P2-9: LLM 调用前限流/配额(流式路径同样生效)。
                if let Err(e) = run_before_completion_hooks(&hooks, &inputs) {
                    let _ = tx
                        .send(Ok(AgentStreamEvent::Error {
                            message: e.to_string(),
                        }))
                        .await;
                    return;
                }
                let output = match agent.plan(&intermediate_steps, &inputs).await {
                    Ok(o) => o,
                    Err(e) => {
                        let _ = tx
                            .send(Ok(AgentStreamEvent::Error {
                                message: e.to_string(),
                            }))
                            .await;
                        return;
                    }
                };
                let usage = agent.last_token_usage();
                // P2-9: LLM 调用后累计 token 用量。
                run_after_completion_hooks(&hooks, &output, usage.as_ref());

                match output {
                    AgentOutput::Finish(finish) => {
                        let content = finish.output().unwrap_or("").to_string();
                        // P1-8 流式融合:先发 Text 事件承载模型文本,再发 FinalAnswer 终态,
                        // 使流内既有工具事件(ToolStart/ToolEnd)又有文本 token。
                        // 非流式 `plan()` 一次性返回整段答案,故 Text 为单个事件;
                        // 流式 agent 则逐 token 发出 Text。
                        let _ = tx
                            .send(Ok(AgentStreamEvent::Text {
                                content: content.clone(),
                            }))
                            .await;
                        let _ = tx.send(Ok(AgentStreamEvent::FinalAnswer { content })).await;
                        return;
                    }

                    AgentOutput::Action(action) => {
                        // P2-9: 流式路径同样执行工具权限策略。
                        if let Some(policy) = &tool_policy {
                            if let Err(e) = policy.check(&action.tool) {
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

                        // Execute the tool
                        let observation =
                            match execute_tool_for_stream(&tools, &action, tool_timeout).await {
                                Ok(obs) => obs,
                                Err(e) => {
                                    let _ = tx
                                        .send(Ok(AgentStreamEvent::Error {
                                            message: e.to_string(),
                                        }))
                                        .await;
                                    return;
                                }
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
                        // P2-9: 并行工具同样先过权限策略。
                        if let Some(policy) = &tool_policy {
                            for action in &actions {
                                if let Err(e) = policy.check(&action.tool) {
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

            // Max iterations reached:返回占位结果,必须记日志以免把非答案当最终答案
            log::warn!("agent reached max iterations; streaming a placeholder result (not the real final answer)");
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
