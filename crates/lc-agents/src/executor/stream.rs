// lc-agents/src/executor/stream.rs
//! Streaming execution for [`AgentExecutor`](super::engine::AgentExecutor) — the
//! background event loop behind [`AgentExecutor::stream`].
//!
//! Extracted verbatim from `engine.rs` in v0.23 (T9, zero behavior change): the
//! streaming path runs in a detached `tokio::spawn` task and pushes
//! [`AgentStreamEvent`]s over an mpsc channel; [`AgentEventStream`] cancels the loop
//! cooperatively when the consumer drops the stream. The non-streaming `invoke` path
//! stays in `engine.rs`. **Do not let the two paths diverge** — the gates, hooks and
//! terminal handling here intentionally mirror `run_agent_loop`.

use super::budget::{budget_cost_gate, budget_iteration_gate, budget_token_gate, budget_tool_gate};
use super::engine::{build_plan_config, AgentExecutor, MaxIterationsPolicy};
use super::hooks::{run_after_completion_hooks, run_before_completion_hooks};
use super::semantic_memory::SEMANTIC_MEMORY_INPUT_KEY;
use super::tools::{
    execute_tool_for_stream, execute_tools_parallel_for_stream, tool_error_observation,
};
use super::AgentError;
use crate::hooks::HookError;
use crate::metrics::AgentMetrics;
use crate::streaming::state::AgentStreamEvent;
use crate::types::{AgentOutput, AgentStep, ToolInput};
use futures_util::Stream;
use lc_callbacks::{semconv::GEN_AI_OPERATION_NAME, CallbackManager, RunTree, RunType};
use lc_core::observability::{MetricsSink, ObsEvent};
use serde_json::json;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Instant;

impl AgentExecutor {
    /// Stream agent execution as a true async stream of events.
    ///
    /// Each step of the agent loop (tool calls, observations, final answer)
    /// is emitted as an `AgentStreamEvent` as soon as it occurs.
    ///
    /// # Error semantics (A9, unified)
    /// The stream item is `Result<AgentStreamEvent, AgentError>`. A terminal
    /// failure — a permission-policy rejection, a tool timeout, a guarded-tool
    /// abort, or budget exhaustion (A-S2 / A-H1) — is delivered as an
    /// `Err(AgentError)`, which terminates the stream. There is no successful
    /// `Ok(AgentStreamEvent::Error { .. })`; that variant exists for infallible
    /// streams (e.g. [`crate::StreamingFunctionCallingAgent`]) and in-band errors.
    ///
    /// # `Text` event granularity (F3, honest)
    ///
    /// `Text` events carry model text, but their granularity depends on the
    /// agent's [`crate::executor::BaseAgent::plan_stream`] implementation:
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
    ///         Err(e) => { /* terminal failure — the loop has ended */ }
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn stream(
        &self,
        input: String,
    ) -> Pin<Box<dyn Stream<Item = Result<AgentStreamEvent, AgentError>> + Send>> {
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        // 0.20.0 A-H2: dropping the returned stream must stop the background agent
        // loop. Without this, a consumer that stops reading (a client disconnect, an
        // early UI cancel) left the loop running — consuming tool calls and LLM tokens
        // for a listener that is gone. The watch channel is the cancel signal: the loop
        // checks it at iteration / tool boundaries, and the wrapper (`AgentEventStream`)
        // sends `true` on drop.
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

        // P2-2: the streaming path also fails fast — unregistered tools emit one error
        // event before ending.
        if let Err(e) = self.validate_tool_registration() {
            tokio::spawn(async move {
                let _ = tx.send(Err(e)).await;
            });
            return Box::pin(AgentEventStream {
                inner: tokio_stream::wrappers::ReceiverStream::new(rx),
                cancel: cancel_tx,
            });
        }

        let agent = self.agent.clone();
        // A11: the stream loop looks tools up by name via the prebuilt index.
        let tools_by_name = self.tools_by_name.clone();
        let max_iterations = self.max_iterations;
        let verbose = self.verbose;
        let tool_timeout = self.tool_timeout;
        let max_concurrency = self.max_concurrency;
        let hooks = self.hooks.clone();
        let tool_policy = self.tool_policy.clone();
        let budget = self.budget.clone();
        let compaction = self.compaction.clone();
        let metrics_store = self.metrics_store.clone();
        let metrics_sink = self.metrics_sink.clone();
        let cost_tracker = self.cost_tracker.clone();
        let on_max_iterations = self.on_max_iterations;
        // 0.22.0 audit fix (H-A1): the stream path previously dropped the
        // cross-cutting capabilities invoke has. Clone callbacks + memory into
        // the spawned task so chain-level callbacks are dispatched, memory
        // history is loaded before the loop and the final answer is saved.
        let callbacks = self.callbacks.clone();
        let memory = self.memory.clone();
        // B4: cloned into the 'static stream task like `memory`; recall runs
        // before the loop, extraction spawns detached from the final answer.
        let semantic_memory = self.semantic_memory.clone();
        // v0.22.1 §S8: copy the A1/A2 toggles so the spawned stream loop reads locals,
        // not `&self` (disjoint capture holds here; referencing `self.` would borrow the
        // whole executor into the `'static` task because `Mutex<dyn BaseMemory>` is invariant).
        let rule_of_two = self.rule_of_two;
        let spotlight_tool_output = self.spotlight_tool_output;

        tokio::spawn(async move {
            let mut intermediate_steps: Vec<AgentStep> = Vec::new();
            let mut inputs = HashMap::new();
            inputs.insert("input".to_string(), input.clone());

            // H-A1: chain callbacks / trace parity with invoke — build the root
            // RunTree, dispatch on_chain_start and on_agent_start hooks, and load
            // memory variables into the inputs before the loop.
            //
            // A18: planning-round `on_llm_*` callbacks are now dispatched on this
            // path too — `plan_config` carries the same callbacks + trace linkage
            // (`__lc_parent_run_id` / `__lc_trace_id`) the invoke path stamps, so
            // the provider-built LLM runs are children of this chain root.
            //
            // Remaining known gaps (honest): tool-level `on_tool_*` callbacks and
            // RunTree trace_id stamping from RunnableConfig metadata (stream()
            // takes no config) are still not dispatched on this path — invoke's
            // tool child-run tracing has no equivalent here because tool execution
            // goes through `execute_tool_for_stream` without a RunTree.
            let mut root_run = RunTree::new(
                "AgentExecutor",
                RunType::Chain,
                json!({"input": inputs.get("input").cloned().unwrap_or_default()}),
            );
            // T10: same `invoke_agent` classification as the non-streaming path.
            root_run = root_run.with_metadata(GEN_AI_OPERATION_NAME, json!("invoke_agent"));
            if let Some(ref callbacks) = callbacks {
                for handler in callbacks.handlers() {
                    handler.on_chain_start(&root_run, &root_run.inputs).await;
                }
            }
            let plan_config = build_plan_config(&callbacks, &root_run);
            for hook in &hooks {
                if let Err(e) = hook.on_agent_start(&input) {
                    log::warn!("Hook on_agent_start error: {}", e);
                }
            }

            if let Some(memory) = &memory {
                let memory_guard = memory.lock().await;
                let variable_keys: Vec<String> = memory_guard
                    .memory_variables()
                    .into_iter()
                    .map(|k| k.to_string())
                    .collect();
                let loaded = match memory_guard.load_memory_variables(&inputs).await {
                    Ok(vars) => vars,
                    Err(e) => {
                        let msg = format!("Failed to load memory: {e}");
                        stream_chain_error(&callbacks, &mut root_run, &msg).await;
                        for hook in &hooks {
                            hook.on_error(&HookError::Other(msg.clone()));
                        }
                        let _ = tx.send(Err(AgentError::Other(msg))).await;
                        return;
                    }
                };
                drop(memory_guard);
                for key in variable_keys {
                    if let Some(value) = loaded.get(&key) {
                        if let Some(s) = value.as_str() {
                            inputs.insert(key, s.to_string());
                        }
                    }
                }
            }

            // B4: semantic recall, same best-effort semantics as invoke.
            if let Some(hook) = &semantic_memory {
                if let Some(block) = hook.recall(&input).await {
                    inputs.insert(SEMANTIC_MEMORY_INPUT_KEY.to_string(), block);
                }
            }

            // Budget gate (§4.2): start the stream timer + accumulate metrics (same
            // semantics as the invoke path).
            let loop_start = Instant::now();
            let mut metrics = AgentMetrics::default();

            for iteration in 0..max_iterations {
                if verbose {
                    log::info!("=== Stream Iteration {} ===", iteration + 1);
                }

                // 0.20.0 A-H2: the consumer dropped the stream → stop before the next
                // plan. Any tool already in flight is allowed to finish (cooperative
                // cancellation), but no new plan / tool starts.
                if *cancel_rx.borrow() {
                    return;
                }

                // Budget gate: iteration-level (iteration count + wall-clock). Over the
                // limit → send Err and stop.
                if let Some(err) =
                    budget_iteration_gate(budget.as_ref(), max_iterations, iteration, loop_start)
                {
                    stream_chain_error(&callbacks, &mut root_run, &err.to_string()).await;
                    publish_metrics(&metrics, &metrics_store, &metrics_sink, loop_start).await;
                    let _ = tx.send(Err(err)).await;
                    return;
                }

                // P2-9: rate-limit / quota check before the LLM call (also applies on
                // the streaming path).
                if let Err(e) = run_before_completion_hooks(&hooks, &inputs) {
                    let msg = e.to_string();
                    stream_chain_error(&callbacks, &mut root_run, &msg).await;
                    publish_metrics(&metrics, &metrics_store, &metrics_sink, loop_start).await;
                    let _ = tx.send(Err(AgentError::Other(msg))).await;
                    return;
                }
                // F3: streaming planning — the agent forwards model text token by token
                // through on_token as Text events. ReAct / FunctionCalling override
                // plan_stream to go through `stream_chat` for a real word-by-word stream;
                // other agents use the default implementation (the whole answer as a
                // single Text event), matching the old path's behavior.
                // 0.21.0 S6.1: context compaction before planning — same semantics as
                // the invoke path (`run_agent_loop_from`), so the two paths cannot diverge.
                if let Some(config) = &compaction {
                    let tokens = metrics.total_tokens.unwrap_or(0);
                    let (kept, dropped) = config.compact(&intermediate_steps, tokens);
                    if dropped > 0 {
                        log::info!(
                            target: "lc_agents::compaction",
                            "compacted {} of {} steps ({} remain) [stream]",
                            dropped,
                            dropped + kept.len(),
                            kept.len()
                        );
                        intermediate_steps = kept;
                        metrics.compactions += 1;
                    }
                }
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
                        .plan_stream(
                            &intermediate_steps,
                            &inputs,
                            &mut on_token,
                            plan_config.as_ref(),
                        )
                        .await
                    {
                        Ok(o) => o,
                        Err(e) => {
                            let msg = e.to_string();
                            stream_chain_error(&callbacks, &mut root_run, &msg).await;
                            for hook in &hooks {
                                hook.on_error(&HookError::Other(msg.clone()));
                            }
                            publish_metrics(&metrics, &metrics_store, &metrics_sink, loop_start)
                                .await;
                            let _ = tx.send(Err(AgentError::Other(msg))).await;
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
                    stream_chain_error(&callbacks, &mut root_run, &err.to_string()).await;
                    publish_metrics(&metrics, &metrics_store, &metrics_sink, loop_start).await;
                    let _ = tx.send(Err(err)).await;
                    return;
                }
                // B3 (0.22.4): cumulative USD spend gate, same semantics as invoke.
                if let Some(tracker) = &cost_tracker {
                    let spent = tracker.total_cost_usd().await;
                    if let Some(err) = budget_cost_gate(budget.as_ref(), spent) {
                        stream_chain_error(&callbacks, &mut root_run, &err.to_string()).await;
                        publish_metrics(&metrics, &metrics_store, &metrics_sink, loop_start).await;
                        let _ = tx.send(Err(err)).await;
                        return;
                    }
                }

                match output {
                    AgentOutput::Finish(finish) => {
                        let content = finish.output().unwrap_or("").to_string();
                        // H-A1: memory save on the final answer, same as invoke's
                        // post-answer save_context. A save failure only warns — it
                        // must not mask a successfully finished stream.
                        if let Some(memory) = &memory {
                            let mut outputs = HashMap::new();
                            outputs.insert("output".to_string(), content.clone());
                            if let Err(e) =
                                memory.lock().await.save_context(&inputs, &outputs).await
                            {
                                log::warn!("failed to save final answer to memory [stream]: {e}");
                            }
                        }
                        // B4: detached fact extraction, same as invoke.
                        if let Some(hook) = &semantic_memory {
                            hook.spawn_extraction(input.clone(), content.clone());
                        }

                        root_run.end(json!({"output": content.clone()}));
                        if let Some(ref callbacks) = callbacks {
                            if let Some(ref outputs) = root_run.outputs {
                                for handler in callbacks.handlers() {
                                    handler.on_chain_end(&root_run, outputs).await;
                                }
                            }
                        }
                        for hook in &hooks {
                            if let Err(e) = hook.on_agent_end(&content) {
                                log::warn!("Hook on_agent_end error: {}", e);
                            }
                        }
                        // P1-8 streaming fusion: the model text was already emitted piece
                        // by piece by plan_stream through on_token (Text events); here
                        // only the FinalAnswer terminal event is sent — the full answer is
                        // not repeated.
                        publish_metrics(&metrics, &metrics_store, &metrics_sink, loop_start).await;
                        let _ = tx.send(Ok(AgentStreamEvent::FinalAnswer { content })).await;
                        return;
                    }

                    AgentOutput::Action(action) => {
                        // 0.22.0 audit fix (H-A5): the ReAct parse-repair pseudo-tool
                        // is not a real tool — feed its message back as the
                        // observation so the model can retry (standard ReAct repair
                        // loop), matching the invoke path.
                        if action.tool == crate::react::agent::PARSE_ERROR_TOOL {
                            let observation = match &action.tool_input {
                                ToolInput::String { value } => value.clone(),
                                ToolInput::Object { value } => value.to_string(),
                            };
                            if verbose {
                                log::info!("Parse repair observation: {}", observation);
                            }
                            intermediate_steps.push(AgentStep::new(action, observation));
                            continue;
                        }
                        // P2-9: the streaming path also enforces the tool permission
                        // policy.
                        if let Some(policy) = &tool_policy {
                            if let Err(e) = policy.check(&action.tool) {
                                let msg = e.to_string();
                                stream_chain_error(&callbacks, &mut root_run, &msg).await;
                                publish_metrics(
                                    &metrics,
                                    &metrics_store,
                                    &metrics_sink,
                                    loop_start,
                                )
                                .await;
                                let _ = tx.send(Err(AgentError::Other(msg))).await;
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

                        // A11: the budget gate runs **before** `ToolStart` is emitted.
                        // Previously the gate ran after, so a rejection left an orphan
                        // `ToolStart` with no matching `ToolEnd`/error. Order now mirrors
                        // the invoke path: reject first, emit the start event only when
                        // the call is actually allowed.
                        metrics.tool_calls += 1;
                        if let Some(err) = budget_tool_gate(budget.as_ref(), &metrics, loop_start) {
                            stream_chain_error(&callbacks, &mut root_run, &err.to_string()).await;
                            publish_metrics(&metrics, &metrics_store, &metrics_sink, loop_start)
                                .await;
                            let _ = tx.send(Err(err)).await;
                            return;
                        }

                        let _ = tx
                            .send(Ok(AgentStreamEvent::ToolStart {
                                name: tool_name.clone(),
                                input: tool_input_str.clone(),
                            }))
                            .await;

                        // 0.20.0 A-H2: dropped mid-iteration → do not start a new tool.
                        if *cancel_rx.borrow() {
                            return;
                        }

                        // Execute the tool. A tool **execution** failure becomes an
                        // observation fed back to the loop (S3.1) so the agent can
                        // recover. Framework guardrails (A-H1, 0.20.0) —
                        // `ControlAbort` handoff/depth guard, `ToolNotFound`, input
                        // serialization — reject the call *before* execution; the agent
                        // cannot recover from them by re-planning, so they end the
                        // stream hard, matching the non-streaming invoke path.
                        let observation = match execute_tool_for_stream(
                            &tools_by_name,
                            &action,
                            tool_timeout,
                            spotlight_tool_output,
                            rule_of_two,
                        )
                        .await
                        {
                            Ok(obs) => obs,
                            Err(e @ AgentError::ToolExecutionError(_)) => {
                                tool_error_observation(&e)
                            }
                            Err(e) => {
                                let msg = e.to_string();
                                stream_chain_error(&callbacks, &mut root_run, &msg).await;
                                publish_metrics(
                                    &metrics,
                                    &metrics_store,
                                    &metrics_sink,
                                    loop_start,
                                )
                                .await;
                                let _ = tx.send(Err(AgentError::Other(msg))).await;
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
                        // P2-9: parallel tools also pass the permission policy first.
                        if let Some(policy) = &tool_policy {
                            for action in &actions {
                                if let Err(e) = policy.check(&action.tool) {
                                    let msg = e.to_string();
                                    stream_chain_error(&callbacks, &mut root_run, &msg).await;
                                    publish_metrics(
                                        &metrics,
                                        &metrics_store,
                                        &metrics_sink,
                                        loop_start,
                                    )
                                    .await;
                                    let _ = tx.send(Err(AgentError::Other(msg))).await;
                                    return;
                                }
                            }
                        }
                        // A11: budget gate runs **before** any `ToolStart` is emitted for the
                        // batch, so a rejection leaves no orphan start events.
                        metrics.tool_calls += actions.len();
                        if let Some(err) = budget_tool_gate(budget.as_ref(), &metrics, loop_start) {
                            stream_chain_error(&callbacks, &mut root_run, &err.to_string()).await;
                            publish_metrics(&metrics, &metrics_store, &metrics_sink, loop_start)
                                .await;
                            let _ = tx.send(Err(err)).await;
                            return;
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

                        // 0.20.0 A-H2: dropped mid-iteration → do not start a new batch.
                        if *cancel_rx.borrow() {
                            return;
                        }

                        let observations = match execute_tools_parallel_for_stream(
                            &tools_by_name,
                            &actions,
                            tool_timeout,
                            max_concurrency,
                            spotlight_tool_output,
                            rule_of_two,
                        )
                        .await
                        {
                            Ok(obs) => obs,
                            // A-H1 (0.20.0): a framework guardrail in any one tool
                            // of the batch ends the stream hard, matching the
                            // invoke-parallel path. Execution errors were already
                            // converted to observations inside the helper.
                            Err(e) => {
                                let msg = e.to_string();
                                stream_chain_error(&callbacks, &mut root_run, &msg).await;
                                publish_metrics(
                                    &metrics,
                                    &metrics_store,
                                    &metrics_sink,
                                    loop_start,
                                )
                                .await;
                                let _ = tx.send(Err(AgentError::Other(msg))).await;
                                return;
                            }
                        };

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

            // 0.22.0 C4 fix: the iteration cap is a failure by default —
            // surface `MaxIterationsReached` on the stream instead of streaming
            // a placeholder that looks like a real answer.
            log::warn!(
                "agent reached max iterations; policy: {:?}",
                on_max_iterations
            );
            if on_max_iterations == MaxIterationsPolicy::Error {
                stream_chain_error(&callbacks, &mut root_run, "max iterations reached").await;
                publish_metrics(&metrics, &metrics_store, &metrics_sink, loop_start).await;
                let _ = tx.send(Err(AgentError::MaxIterationsReached)).await;
                return;
            }
            // Legacy placeholder policy: the stopped response is treated as a final
            // answer — run the H-A1 terminal path (memory save + chain end) too.
            let finish = agent.return_stopped_response(&intermediate_steps);
            let content = finish.output().unwrap_or("").to_string();
            if let Some(memory) = &memory {
                let mut outputs = HashMap::new();
                outputs.insert("output".to_string(), content.clone());
                if let Err(e) = memory.lock().await.save_context(&inputs, &outputs).await {
                    log::warn!("failed to save final answer to memory [stream]: {e}");
                }
            }
            // B4: detached fact extraction, same as invoke.
            if let Some(hook) = &semantic_memory {
                hook.spawn_extraction(input.clone(), content.clone());
            }
            root_run.end(json!({"output": content.clone()}));
            if let Some(ref callbacks) = callbacks {
                if let Some(ref outputs) = root_run.outputs {
                    for handler in callbacks.handlers() {
                        handler.on_chain_end(&root_run, outputs).await;
                    }
                }
            }
            for hook in &hooks {
                if let Err(e) = hook.on_agent_end(&content) {
                    log::warn!("Hook on_agent_end error: {}", e);
                }
            }
            publish_metrics(&metrics, &metrics_store, &metrics_sink, loop_start).await;
            let _ = tx.send(Ok(AgentStreamEvent::FinalAnswer { content })).await;
        });

        Box::pin(AgentEventStream {
            inner: tokio_stream::wrappers::ReceiverStream::new(rx),
            cancel: cancel_tx,
        })
    }
}

/// Stream wrapper returned by [`AgentExecutor::stream`]: cancels the background agent
/// loop when the consumer drops the stream (0.20.0 A-H2). A dropped stream means the
/// listener is gone — the loop must stop burning tool calls and LLM tokens instead of
/// running the remaining iterations invisibly.
struct AgentEventStream {
    /// The live event channel.
    inner: tokio_stream::wrappers::ReceiverStream<Result<AgentStreamEvent, AgentError>>,
    /// Set to `true` on drop; the loop observes it via `cancel_rx` at iteration / tool
    /// boundaries and stops cooperatively (letting any in-flight tool finish).
    cancel: tokio::sync::watch::Sender<bool>,
}

impl Stream for AgentEventStream {
    type Item = Result<AgentStreamEvent, AgentError>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl Drop for AgentEventStream {
    fn drop(&mut self) {
        let _ = self.cancel.send(true);
    }
}

/// 0.22.0 audit fix (H-A1): dispatches the chain-error callbacks on the stream
/// path, mirroring invoke's `on_chain_error` handling. Best-effort — never
/// fails, only marks the root run as errored first.
async fn stream_chain_error(
    callbacks: &Option<Arc<CallbackManager>>,
    root_run: &mut RunTree,
    message: &str,
) {
    root_run.end_with_error(message.to_string());
    if let Some(callbacks) = callbacks {
        for handler in callbacks.handlers() {
            handler.on_chain_error(root_run, message).await;
        }
    }
}

/// Publishes `AgentMetrics` at the end of a stream (aligned with the invoke path):
/// clone → fill duration → audit log → write `metrics_store`.
///
/// **Ordering constraint (race)**: the stream closure runs in `tokio::spawn`, so every
/// termination path must **`publish_metrics` before `tx.send(terminal event)`** —
/// otherwise a consumer that checks `last_metrics()` immediately after draining the
/// stream may read `None` (the event arrived but the write has not happened yet).
async fn publish_metrics(
    metrics: &AgentMetrics,
    metrics_store: &Arc<Mutex<Option<AgentMetrics>>>,
    metrics_sink: &Option<Arc<dyn MetricsSink>>,
    started: Instant,
) {
    let mut m = metrics.clone();
    m.duration = started.elapsed();
    m.log_summary();
    if let Ok(mut guard) = metrics_store.lock() {
        *guard = Some(m.clone());
    }
    if let Some(sink) = metrics_sink {
        let evt = ObsEvent::AgentMetrics(m);
        if let Err(e) = sink.export(&evt).await {
            log::warn!(target: "lc_agents::metrics", "agent metrics export failed: {e}");
        }
    }
}
