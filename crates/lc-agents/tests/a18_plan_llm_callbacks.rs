//! A18 regression tests: per-round planning LLM calls (`BaseAgent::plan` /
//! `plan_stream`) must carry the executor's `RunnableConfig` so the provider
//! layer fires `on_llm_start` / `on_llm_end` for every planning round and the
//! LLM runs join the agent chain's trace tree (`parent_run_id` / `trace_id`).
//!
//! Before A18 the agent passed `None` at every planning call site, so these
//! tests saw zero LLM events and the (pre-fix) trait signature could not even
//! accept a config. The mock chat model below mirrors the real providers'
//! contract: it builds its `RunTree` via `run_tree_from_config` and fires the
//! LLM callbacks itself from whatever config it received.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::{stream, Stream, StreamExt};
use lc_agents::{AgentExecutor, AgentStreamEvent, BaseAgent, ReActAgent};
use lc_callbacks::{CallbackHandler, CallbackManager, RunTree, RunType};
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk};
use lc_core::runnables::{run_tree_from_config, Runnable, RunnableConfig};
use lc_core::tools::{BaseTool, ToolError};
use lc_providers::ProviderError;
use lc_schema::Message;
use serde_json::json;
use uuid::Uuid;

/// Scripted two-round chat model (round 0: ReAct tool action; round 1+: final
/// answer). Records the `CallbackManager` pointer seen on every call so tests
/// can assert exactly which config reached the provider layer.
struct ScriptedChat {
    responses: Vec<String>,
    calls: Arc<AtomicUsize>,
    seen_callbacks: Arc<Mutex<Vec<Option<usize>>>>,
}

impl ScriptedChat {
    fn two_round() -> Self {
        Self {
            responses: vec![
                "Thought: I need the tool\nAction: succeed\nAction Input: hi".to_string(),
                "Thought: done\nFinal Answer: all done".to_string(),
            ],
            calls: Arc::new(AtomicUsize::new(0)),
            seen_callbacks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn script(&self) -> String {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        self.responses
            .get(n)
            .cloned()
            .unwrap_or_else(|| self.responses.last().cloned().unwrap_or_default())
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for ScriptedChat {
    type Error = ProviderError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, config).await
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for ScriptedChat {
    fn model_name(&self) -> &str {
        "scripted-mock"
    }

    fn get_num_tokens(&self, _text: &str) -> usize {
        0
    }

    fn with_temperature(self, _temp: f32) -> Self
    where
        Self: Sized,
    {
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self
    where
        Self: Sized,
    {
        self
    }
}

#[async_trait]
impl BaseChatModel for ScriptedChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.seen_callbacks.lock().unwrap().push(
            config
                .as_ref()
                .and_then(|c| c.callbacks.as_ref())
                .map(|cm| Arc::as_ptr(cm) as usize),
        );

        let content = self.script();
        // Provider-faithful behavior: build the run from config and fire the
        // LLM lifecycle callbacks when (and only when) a config with callbacks
        // arrived.
        if let Some(cfg) = &config {
            if let Some(cm) = &cfg.callbacks {
                let run = run_tree_from_config(
                    "scripted-mock:chat",
                    RunType::Llm,
                    json!({"model": "scripted-mock"}),
                    Some(cfg),
                );
                for handler in cm.handlers() {
                    handler.on_llm_start(&run, &messages).await;
                    handler.on_llm_end(&run, &content).await;
                }
            }
        }

        Ok(LLMResult {
            content,
            model: "scripted-mock".to_string(),
            token_usage: None,
            tool_calls: None,
            thinking_content: None,
        })
    }

    async fn stream_chat(
        &self,
        _messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        self.seen_callbacks.lock().unwrap().push(
            config
                .as_ref()
                .and_then(|c| c.callbacks.as_ref())
                .map(|cm| Arc::as_ptr(cm) as usize),
        );

        let content = self.script();
        // Split into two chunks to exercise real streaming.
        let mid = content.len() / 2;
        let pieces = vec![content[..mid].to_string(), content[mid..].to_string()];

        let callbacks = config.as_ref().and_then(|c| c.callbacks.clone());
        let run = run_tree_from_config(
            "scripted-mock:stream",
            RunType::Llm,
            json!({"model": "scripted-mock"}),
            config.as_ref(),
        );
        if let Some(cm) = &callbacks {
            for handler in cm.handlers() {
                handler.on_llm_start(&run, &[]).await;
            }
        }

        // State: (next piece index, pieces, end-fired?, run, manager, response text).
        let state: (
            usize,
            Vec<String>,
            bool,
            RunTree,
            Option<Arc<CallbackManager>>,
            String,
        ) = (0, pieces, false, run, callbacks, content);
        let s = stream::unfold(state, |mut st| async move {
            if st.0 < st.1.len() {
                let chunk = Ok(StreamChunk {
                    text: st.1[st.0].clone(),
                    token_usage: None,
                    tool_calls: None,
                });
                st.0 += 1;
                Some((chunk, st))
            } else if !st.2 {
                st.2 = true;
                if let Some(cm) = &st.4 {
                    for handler in cm.handlers() {
                        handler.on_llm_end(&st.3, &st.5).await;
                    }
                }
                // Terminal poll: end fired, no item. unfold never polls again.
                None
            } else {
                None
            }
        });
        Ok(Box::pin(s))
    }
}

/// Trivial tool the scripted action round targets.
struct SucceedTool;

#[async_trait]
impl BaseTool for SucceedTool {
    fn name(&self) -> &str {
        "succeed"
    }
    fn description(&self) -> &str {
        "always succeeds"
    }
    async fn run(&self, _input: String) -> Result<String, ToolError> {
        Ok("ok-value".to_string())
    }
}

#[derive(Debug, Clone)]
struct Evt {
    kind: &'static str,
    run_type: RunType,
    id: Uuid,
    parent: Option<Uuid>,
    trace: Option<Uuid>,
    name: String,
}

/// Funnels every typed lifecycle callback through the three generic methods
/// (their trait defaults do exactly this), tagging each with the run type.
struct Recorder(Mutex<Vec<Evt>>);

#[async_trait]
impl CallbackHandler for Recorder {
    async fn on_run_start(&self, run: &RunTree) {
        self.0.lock().unwrap().push(Evt {
            kind: "start",
            run_type: run.run_type,
            id: run.id,
            parent: run.parent_run_id,
            trace: run.trace_id,
            name: run.name.clone(),
        });
    }
    async fn on_run_end(&self, run: &RunTree) {
        self.0.lock().unwrap().push(Evt {
            kind: "end",
            run_type: run.run_type,
            id: run.id,
            parent: run.parent_run_id,
            trace: run.trace_id,
            name: run.name.clone(),
        });
    }
    async fn on_run_error(&self, run: &RunTree, _error: &str) {
        self.0.lock().unwrap().push(Evt {
            kind: "error",
            run_type: run.run_type,
            id: run.id,
            parent: run.parent_run_id,
            trace: run.trace_id,
            name: run.name.clone(),
        });
    }
}

fn llm_events<'a>(events: &'a [Evt], kind: &str) -> Vec<&'a Evt> {
    events
        .iter()
        .filter(|e| e.kind == kind && matches!(e.run_type, RunType::Llm))
        .collect()
}

fn pos(events: &[Evt], pred: impl Fn(&Evt) -> bool) -> usize {
    events.iter().position(pred).unwrap()
}

/// A18 (invoke path): a two-round tool-using agent fires one LLM start/end
/// pair **per planning round**, each nested under the agent chain run.
#[tokio::test]
async fn invoke_planning_rounds_fire_llm_callbacks_under_chain_run() {
    let recorder = Arc::new(Recorder(Mutex::new(Vec::new())));
    let manager = Arc::new(CallbackManager::new().add_handler(recorder.clone()));

    let mock = ScriptedChat::two_round();
    let seen = mock.seen_callbacks.clone();
    let agent = ReActAgent::new(mock, vec![Arc::new(SucceedTool) as Arc<dyn BaseTool>], None);
    let executor = AgentExecutor::new(
        Arc::new(agent),
        vec![Arc::new(SucceedTool) as Arc<dyn BaseTool>],
    )
    .with_callbacks(manager);

    let output = executor
        .invoke("use the tool please".to_string())
        .await
        .expect("two-round agent run should succeed");
    assert!(output.contains("all done"), "got: {output}");

    let events = recorder.0.lock().unwrap().clone();

    // Exactly two planning rounds → two LLM start/end pairs.
    assert_eq!(llm_events(&events, "start").len(), 2, "events: {events:?}");
    assert_eq!(llm_events(&events, "end").len(), 2, "events: {events:?}");

    // The chain root is the AgentExecutor run; every LLM run is its child and
    // shares its trace id (plain invoke → trace id is the root run id itself).
    let root = events
        .iter()
        .find(|e| e.kind == "start" && matches!(e.run_type, RunType::Chain))
        .expect("chain start event");
    assert_eq!(root.name, "AgentExecutor");
    for evt in events.iter().filter(|e| matches!(e.run_type, RunType::Llm)) {
        assert_eq!(
            evt.parent,
            Some(root.id),
            "LLM run {} must be a child of the chain root, events: {events:?}",
            evt.name
        );
        assert_eq!(
            evt.trace,
            Some(root.id),
            "LLM run {} must join the root trace id, events: {events:?}",
            evt.name
        );
    }

    // Ordering: chain start → llm pair 1 → llm pair 2 → chain end.
    let chain_start = pos(&events, |e| {
        e.kind == "start" && matches!(e.run_type, RunType::Chain)
    });
    let chain_end = pos(&events, |e| {
        e.kind == "end" && matches!(e.run_type, RunType::Chain)
    });
    let starts: Vec<usize> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| (e.kind == "start" && matches!(e.run_type, RunType::Llm)).then_some(i))
        .collect();
    let end_for = |start_idx: usize| {
        pos(&events, |e| {
            e.kind == "end" && matches!(e.run_type, RunType::Llm) && e.id == events[start_idx].id
        })
    };
    let first_llm_end = end_for(starts[0]);
    let second_llm_end = end_for(starts[1]);
    assert!(
        chain_start < starts[0]
            && starts[0] < first_llm_end
            && first_llm_end < starts[1]
            && starts[1] < second_llm_end
            && second_llm_end < chain_end,
        "callback order broken, events: {events:?}"
    );

    // Both provider calls received the same manager instance.
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert!(seen.iter().all(Option::is_some), "events: {seen:?}");
}

/// A18 (stream path): the same observability contract holds on `stream()` —
/// `plan_stream` rounds must carry the executor callbacks too.
#[tokio::test]
async fn stream_planning_rounds_fire_llm_callbacks_under_chain_run() {
    let recorder = Arc::new(Recorder(Mutex::new(Vec::new())));
    let manager = Arc::new(CallbackManager::new().add_handler(recorder.clone()));

    let agent = ReActAgent::new(
        ScriptedChat::two_round(),
        vec![Arc::new(SucceedTool) as Arc<dyn BaseTool>],
        None,
    );
    let executor = AgentExecutor::new(
        Arc::new(agent),
        vec![Arc::new(SucceedTool) as Arc<dyn BaseTool>],
    )
    .with_callbacks(manager);

    let mut stream = executor.stream("use the tool please".to_string());
    let mut final_answer = None;
    while let Some(item) = stream.next().await {
        if let Ok(AgentStreamEvent::FinalAnswer { content }) = item {
            final_answer = Some(content);
            break;
        }
    }
    drop(stream);
    assert_eq!(final_answer.as_deref(), Some("all done"));

    let events = recorder.0.lock().unwrap().clone();
    assert_eq!(llm_events(&events, "start").len(), 2, "events: {events:?}");
    assert_eq!(llm_events(&events, "end").len(), 2, "events: {events:?}");

    let root = events
        .iter()
        .find(|e| e.kind == "start" && matches!(e.run_type, RunType::Chain))
        .expect("chain start event");
    for evt in events.iter().filter(|e| matches!(e.run_type, RunType::Llm)) {
        assert_eq!(evt.parent, Some(root.id), "events: {events:?}");
        assert_eq!(evt.trace, Some(root.id), "events: {events:?}");
    }
}

/// A18 (agent-level): `plan` forwards the exact `RunnableConfig` it was given
/// to the chat model; `None` stays `None` (backward-compatible behavior).
#[tokio::test]
async fn plan_forwards_config_verbatim_and_none_stays_none() {
    let mock = ScriptedChat::two_round();
    let seen = mock.seen_callbacks.clone();
    let agent = ReActAgent::new(mock, vec![], None);

    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), "hi".to_string());

    let manager = Arc::new(CallbackManager::new());
    let manager_ptr = Arc::as_ptr(&manager) as usize;
    let config = RunnableConfig::new().with_callbacks(manager);

    let output = agent
        .plan(&[], &inputs, Some(&config))
        .await
        .expect("plan with config");
    assert!(matches!(output, lc_agents::types::AgentOutput::Action(_)));

    let output_none = agent.plan(&[], &inputs, None).await.expect("plan none");
    assert!(matches!(
        output_none,
        lc_agents::types::AgentOutput::Finish(_)
    ));

    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen, vec![Some(manager_ptr), None]);
}

/// A18 regression: an executor without callbacks still passes `None`
/// end-to-end (provider fires nothing; behavior identical to pre-A18).
#[tokio::test]
async fn executor_without_callbacks_passes_none_to_provider() {
    let mock = ScriptedChat::two_round();
    let seen = mock.seen_callbacks.clone();
    let agent = ReActAgent::new(mock, vec![Arc::new(SucceedTool) as Arc<dyn BaseTool>], None);
    let executor = AgentExecutor::new(
        Arc::new(agent),
        vec![Arc::new(SucceedTool) as Arc<dyn BaseTool>],
    );

    let output = executor
        .invoke("use the tool".to_string())
        .await
        .expect("plain invoke works");
    assert!(output.contains("all done"));

    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen, vec![None, None]);
}
