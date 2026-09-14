//! T6 (v0.23.0) regression tests: the legacy independent workflows
//! (plan-execute Planner, CRAG rewriter/grader, deep-research planner/
//! synthesizer, AdaptiveRAG router, StreamingFunctionCallingAgent) used to
//! hardcode `None` at every planning LLM call, so their LLM runs were
//! invisible to callbacks/OTel. They now take an `Option<&RunnableConfig>` and
//! must forward it verbatim to the provider layer, where `on_llm_start` /
//! `on_llm_end` fire exactly as on the A18 executor main link.
//!
//! The mock chat model mirrors the real providers' contract: it builds its
//! `RunTree` via `run_tree_from_config` and fires the LLM callbacks itself
//! from whatever config it received.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::{stream, Stream, StreamExt};
use lc_agents::adaptive_rag::AdaptiveRAG;
use lc_agents::crag::grader::DocumentGrader;
use lc_agents::crag::rewriter::QueryRewriter;
use lc_agents::deep_research::DeepResearchAgent;
use lc_agents::plan_execute::Planner;
use lc_agents::streaming::StreamingFunctionCallingAgent;
use lc_callbacks::{CallbackHandler, CallbackManager, RunTree, RunType};
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk};
use lc_core::runnables::{run_tree_from_config, Runnable, RunnableConfig};
use lc_core::tools::{BaseTool, ToolError};
use lc_providers::ProviderError;
use lc_rag::{RetrieverError, RetrieverTrait};
use lc_schema::Message;
use lc_vector_stores::Document;
use serde_json::json;

/// Scripted chat model: returns queued responses in order, records the
/// callback-manager pointer (if any) seen on every call, and fires
/// `on_llm_start/end` exactly like a real provider would.
#[derive(Clone)]
struct ScriptedChat {
    responses: Arc<Mutex<VecDeque<String>>>,
    calls: Arc<AtomicUsize>,
    seen_callbacks: Arc<Mutex<Vec<Option<usize>>>>,
}

impl ScriptedChat {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
            calls: Arc::new(AtomicUsize::new(0)),
            seen_callbacks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn next_script(&self) -> String {
        let mut q = self.responses.lock().unwrap_or_else(|e| e.into_inner());
        q.pop_front()
            .unwrap_or_else(|| q.back().cloned().unwrap_or_default())
    }

    async fn fire_llm_callbacks(
        &self,
        messages: &[Message],
        content: &str,
        config: &Option<RunnableConfig>,
    ) {
        self.seen_callbacks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(
                config
                    .as_ref()
                    .and_then(|c| c.callbacks.as_ref())
                    .map(|cm| Arc::as_ptr(cm) as usize),
            );
        if let Some(cfg) = config {
            if let Some(cm) = &cfg.callbacks {
                let run = run_tree_from_config(
                    "scripted-mock:chat",
                    RunType::Llm,
                    json!({"model": "scripted-mock"}),
                    Some(cfg),
                );
                for handler in cm.handlers() {
                    handler.on_llm_start(&run, messages).await;
                    handler.on_llm_end(&run, content).await;
                }
            }
        }
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        let content = self.next_script();
        self.fire_llm_callbacks(&messages, &content, &config).await;
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        let content = self.next_script();

        self.seen_callbacks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(
                config
                    .as_ref()
                    .and_then(|c| c.callbacks.as_ref())
                    .map(|cm| Arc::as_ptr(cm) as usize),
            );

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

        // (next index, pieces, end fired?, run, manager, full text)
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
                None
            } else {
                None
            }
        });
        Ok(Box::pin(s))
    }
}

/// Counts LLM lifecycle events.
#[derive(Default)]
struct Recorder {
    starts: Mutex<Vec<String>>,
    ends: Mutex<Vec<String>>,
}

#[async_trait]
impl CallbackHandler for Recorder {
    async fn on_run_start(&self, run: &RunTree) {
        if matches!(run.run_type, RunType::Llm) {
            self.starts
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(run.name.clone());
        }
    }

    async fn on_run_end(&self, run: &RunTree) {
        if matches!(run.run_type, RunType::Llm) {
            self.ends
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(run.name.clone());
        }
    }

    async fn on_run_error(&self, _run: &RunTree, _error: &str) {}
}

struct Harness {
    recorder: Arc<Recorder>,
    manager_ptr: usize,
    config: RunnableConfig,
}

fn harness() -> Harness {
    let recorder = Arc::new(Recorder::default());
    let manager = Arc::new(CallbackManager::new().add_handler(recorder.clone()));
    let manager_ptr = Arc::as_ptr(&manager) as usize;
    let config = RunnableConfig::new().with_callbacks(manager);
    Harness {
        recorder,
        manager_ptr,
        config,
    }
}

/// T6 site 1: plan-execute `Planner::plan` / `replan` (planner.rs:68,111)
/// forward the config; one LLM start/end pair fires per planning call.
#[tokio::test]
async fn plan_execute_planner_forwards_config() {
    let h = harness();
    let chat = ScriptedChat::new(vec![
        r#"["step one", "step two"]"#.to_string(),
        r#"["remaining step"]"#.to_string(),
    ]);
    let seen = chat.seen_callbacks.clone();
    let arc: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync> = Arc::new(chat);
    let planner = Planner::new(arc);

    let plan = planner
        .plan("objective", Some(&h.config))
        .await
        .expect("plan ok");
    assert_eq!(plan.steps.len(), 2, "two scripted steps");

    let replanned = planner
        .replan(
            "objective",
            "step one",
            "boom",
            "step one: done",
            Some(&h.config),
        )
        .await
        .expect("replan ok");
    assert_eq!(replanned.steps.len(), 1);

    assert_eq!(h.recorder.starts.lock().unwrap().len(), 2);
    assert_eq!(h.recorder.ends.lock().unwrap().len(), 2);
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen, vec![Some(h.manager_ptr), Some(h.manager_ptr)]);
}

/// T6 sites 2/3: CRAG `QueryRewriter::rewrite` / `generate_alternatives`
/// (rewriter.rs:42,70) and `DocumentGrader::grade` fire LLM callbacks.
#[tokio::test]
async fn crag_rewriter_and_grader_forward_config() {
    let h = harness();
    let chat = ScriptedChat::new(vec![
        "Rewritten query: better retrieval query".to_string(),
        "Relevance: relevant\nScore: 0.9\nReasoning: direct hit".to_string(),
    ]);
    let seen = chat.seen_callbacks.clone();

    let rewritten = QueryRewriter::new(&chat)
        .rewrite("original query", Some(&h.config))
        .await
        .expect("rewrite ok");
    assert_eq!(rewritten, "better retrieval query");

    let grade = DocumentGrader::new(&chat)
        .grade(
            "original query",
            &Document::new("matching document body"),
            Some(&h.config),
        )
        .await
        .expect("grade ok");
    assert!(grade.score >= 0.8, "score: {}", grade.score);

    assert_eq!(h.recorder.starts.lock().unwrap().len(), 2);
    assert_eq!(h.recorder.ends.lock().unwrap().len(), 2);
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen, vec![Some(h.manager_ptr), Some(h.manager_ptr)]);
}

/// T6 sites 4-6: `DeepResearchAgent::research_with_config` carries the config
/// through both planning LLM calls (decompose planner + synthesizer).
struct MockSearch;

#[async_trait]
impl BaseTool for MockSearch {
    fn name(&self) -> &str {
        "mock_search"
    }
    fn description(&self) -> &str {
        "always returns one canned result"
    }
    async fn run(&self, _input: String) -> Result<String, ToolError> {
        Ok(r#"{"results":[{"title":"T","snippet":"S","url":"http://example.com"}]}"#.to_string())
    }
}

#[tokio::test]
async fn deep_research_research_forwards_config_to_planning_calls() {
    let h = harness();
    let chat = ScriptedChat::new(vec![
        r#"[{"name":"A","queries":["q1"]}]"#.to_string(),
        "<<<REPORT>>>\n# Report\nfinding [1]\n<<<END_REPORT>>>\n<<<GAPS>>>\n[]\n<<<END_GAPS>>>"
            .to_string(),
    ]);
    let seen = chat.seen_callbacks.clone();

    let report = DeepResearchAgent::new(chat)
        .with_searcher(Box::new(MockSearch))
        .research_with_config("topic", Some(&h.config))
        .await
        .expect("research ok");
    assert!(report.markdown.contains("finding"));
    assert_eq!(report.subtopics, vec!["A".to_string()]);
    assert_eq!(report.rounds_completed, 1);

    // plan + synthesize = 2 planning LLM calls, both carrying the config.
    assert_eq!(h.recorder.starts.lock().unwrap().len(), 2);
    assert_eq!(h.recorder.ends.lock().unwrap().len(), 2);
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen, vec![Some(h.manager_ptr), Some(h.manager_ptr)]);
}

/// T6 site 7: AdaptiveRAG routing call (adaptive_rag/mod.rs) carries config.
/// The no-retrieval *generation* call deliberately stays untraced
/// (`chat_with_system` has no config parameter; out of T6 scope).
struct EmptyRetriever;

#[async_trait]
impl RetrieverTrait for EmptyRetriever {
    async fn retrieve(&self, _query: &str, _k: usize) -> Result<Vec<Document>, RetrieverError> {
        Ok(Vec::new())
    }

    async fn retrieve_with_scores(
        &self,
        _query: &str,
        _k: usize,
    ) -> Result<Vec<lc_vector_stores::SearchResult>, RetrieverError> {
        Ok(Vec::new())
    }

    async fn add_documents(&self, _documents: Vec<Document>) -> Result<(), RetrieverError> {
        Ok(())
    }
}

#[tokio::test]
async fn adaptive_rag_routing_forwards_config() {
    let h = harness();
    let chat = ScriptedChat::new(vec![
        "no_retrieval".to_string(),
        "direct knowledge answer".to_string(),
    ]);
    let seen = chat.seen_callbacks.clone();

    let result = AdaptiveRAG::new(chat, EmptyRetriever)
        .invoke_with_config("capital of France?", Some(&h.config))
        .await
        .expect("adaptive invoke ok");
    assert_eq!(result.answer, "direct knowledge answer");

    // Exactly the routing call was traced: one LLM pair...
    assert_eq!(h.recorder.starts.lock().unwrap().len(), 1);
    assert_eq!(h.recorder.ends.lock().unwrap().len(), 1);
    // ...the router saw the manager, the untraced generation saw None.
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen, vec![Some(h.manager_ptr), None]);
}

/// T6 site 8: `StreamingFunctionCallingAgent::invoke_stream_with_config`
/// (tool_call_stream.rs:55) forwards the config to `stream_chat`.
#[tokio::test]
async fn streaming_function_agent_forwards_config() {
    let h = harness();
    let chat = ScriptedChat::new(vec!["streamed final answer".to_string()]);
    let seen = chat.seen_callbacks.clone();

    let agent = StreamingFunctionCallingAgent::new(chat);
    let mut stream = agent
        .invoke_stream_with_config("hi".to_string(), Some(&h.config))
        .await;

    let mut final_answer = None;
    while let Some(evt) = stream.next().await {
        if let lc_agents::AgentStreamEvent::FinalAnswer { content } = evt {
            final_answer = Some(content);
            break;
        }
    }
    assert_eq!(final_answer.as_deref(), Some("streamed final answer"));
    drop(stream);

    assert_eq!(h.recorder.starts.lock().unwrap().len(), 1);
    assert_eq!(h.recorder.ends.lock().unwrap().len(), 1);
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen, vec![Some(h.manager_ptr)]);
}
