// lc-callbacks/src/handlers/generic_handler.rs
//! Closure-based callback handler
//!
//! Lets you wire up a [`CallbackHandler`] from plain closures without defining
//! a new struct. Every hook is optional: hooks you don't set fall back to the
//! generic lifecycle layer (`on_run_start` / `on_run_end` / `on_run_error`),
//! exactly like the [`CallbackHandler`] trait defaults — so you can observe
//! only what you care about.
//!
//! # Example
//!
//! ```ignore
//! use lc_callbacks::{CallbackManager, GenericHandler};
//! use std::sync::Arc;
//!
//! let manager = CallbackManager::new().add_handler(Arc::new(
//!     GenericHandler::new()
//!         .with_llm_start(|run, _msgs| println!("llm start: {}", run.name))
//!         .with_llm_new_token(|_run, token| print!("{}", token)),
//! ));
//! ```

use async_trait::async_trait;
use std::sync::Arc;

use crate::{CallbackHandler, RunTree};
use lc_schema::Message;

type RunStartFn = Arc<dyn Fn(&RunTree) + Send + Sync>;
type RunEndFn = Arc<dyn Fn(&RunTree) + Send + Sync>;
type RunErrorFn = Arc<dyn Fn(&RunTree, &str) + Send + Sync>;
type LlmStartFn = Arc<dyn Fn(&RunTree, &[Message]) + Send + Sync>;
type LlmEndFn = Arc<dyn Fn(&RunTree, &str) + Send + Sync>;
type TokenFn = Arc<dyn Fn(&RunTree, &str) + Send + Sync>;
type ToolStartFn = Arc<dyn Fn(&RunTree, &str, &str) + Send + Sync>;
type RetrieverStartFn = Arc<dyn Fn(&RunTree, &str) + Send + Sync>;

/// Closure-based [`CallbackHandler`] for ad-hoc observability hooks.
///
/// Unset hooks delegate to the generic lifecycle layer, matching the
/// [`CallbackHandler`] trait defaults (e.g. an unset `on_llm_start` forwards
/// to `on_run_start`).
#[derive(Default)]
pub struct GenericHandler {
    on_run_start: Option<RunStartFn>,
    on_run_end: Option<RunEndFn>,
    on_run_error: Option<RunErrorFn>,
    on_llm_start: Option<LlmStartFn>,
    on_llm_end: Option<LlmEndFn>,
    on_llm_new_token: Option<TokenFn>,
    on_llm_thinking: Option<TokenFn>,
    on_tool_start: Option<ToolStartFn>,
    on_retriever_start: Option<RetrieverStartFn>,
}

impl GenericHandler {
    /// Create a handler with no hooks set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the generic run-start hook.
    pub fn with_run_start(mut self, f: impl Fn(&RunTree) + Send + Sync + 'static) -> Self {
        self.on_run_start = Some(Arc::new(f));
        self
    }

    /// Set the generic run-end hook.
    pub fn with_run_end(mut self, f: impl Fn(&RunTree) + Send + Sync + 'static) -> Self {
        self.on_run_end = Some(Arc::new(f));
        self
    }

    /// Set the generic run-error hook.
    pub fn with_run_error(mut self, f: impl Fn(&RunTree, &str) + Send + Sync + 'static) -> Self {
        self.on_run_error = Some(Arc::new(f));
        self
    }

    /// Set the LLM start hook (falls back to the run-start hook when unset).
    pub fn with_llm_start(
        mut self,
        f: impl Fn(&RunTree, &[Message]) + Send + Sync + 'static,
    ) -> Self {
        self.on_llm_start = Some(Arc::new(f));
        self
    }

    /// Set the LLM end hook (falls back to the run-end hook when unset).
    pub fn with_llm_end(mut self, f: impl Fn(&RunTree, &str) + Send + Sync + 'static) -> Self {
        self.on_llm_end = Some(Arc::new(f));
        self
    }

    /// Set the streaming-token hook (default: no-op).
    pub fn with_llm_new_token(
        mut self,
        f: impl Fn(&RunTree, &str) + Send + Sync + 'static,
    ) -> Self {
        self.on_llm_new_token = Some(Arc::new(f));
        self
    }

    /// Set the extended-thinking token hook (default: no-op).
    pub fn with_llm_thinking(mut self, f: impl Fn(&RunTree, &str) + Send + Sync + 'static) -> Self {
        self.on_llm_thinking = Some(Arc::new(f));
        self
    }

    /// Set the tool start hook (falls back to the run-start hook when unset).
    pub fn with_tool_start(
        mut self,
        f: impl Fn(&RunTree, &str, &str) + Send + Sync + 'static,
    ) -> Self {
        self.on_tool_start = Some(Arc::new(f));
        self
    }

    /// Set the retriever start hook (falls back to the run-start hook when unset).
    pub fn with_retriever_start(
        mut self,
        f: impl Fn(&RunTree, &str) + Send + Sync + 'static,
    ) -> Self {
        self.on_retriever_start = Some(Arc::new(f));
        self
    }
}

#[async_trait]
impl CallbackHandler for GenericHandler {
    async fn on_run_start(&self, run: &RunTree) {
        if let Some(f) = &self.on_run_start {
            f(run);
        }
    }

    async fn on_run_end(&self, run: &RunTree) {
        if let Some(f) = &self.on_run_end {
            f(run);
        }
    }

    async fn on_run_error(&self, run: &RunTree, error: &str) {
        if let Some(f) = &self.on_run_error {
            f(run, error);
        }
    }

    async fn on_llm_start(&self, run: &RunTree, messages: &[Message]) {
        if let Some(f) = &self.on_llm_start {
            f(run, messages);
        } else {
            self.on_run_start(run).await;
        }
    }

    async fn on_llm_end(&self, run: &RunTree, response: &str) {
        if let Some(f) = &self.on_llm_end {
            f(run, response);
        } else {
            self.on_run_end(run).await;
        }
    }

    async fn on_llm_new_token(&self, run: &RunTree, token: &str) {
        if let Some(f) = &self.on_llm_new_token {
            f(run, token);
        }
    }

    async fn on_llm_thinking(&self, run: &RunTree, thinking: &str) {
        if let Some(f) = &self.on_llm_thinking {
            f(run, thinking);
        }
    }

    async fn on_tool_start(&self, run: &RunTree, tool_name: &str, input: &str) {
        if let Some(f) = &self.on_tool_start {
            f(run, tool_name, input);
        } else {
            self.on_run_start(run).await;
        }
    }

    async fn on_retriever_start(&self, run: &RunTree, query: &str) {
        if let Some(f) = &self.on_retriever_start {
            f(run, query);
        } else {
            self.on_run_start(run).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn run(name: &str) -> RunTree {
        RunTree::new(name, crate::RunType::Chain, serde_json::json!({}))
    }

    #[tokio::test]
    async fn test_run_hooks_fire() {
        let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let ev_start = Arc::clone(&events);
        let ev_end = Arc::clone(&events);

        let handler = GenericHandler::new()
            .with_run_start(move |r| {
                ev_start
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(format!("start:{}", r.name))
            })
            .with_run_end(move |r| {
                ev_end
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(format!("end:{}", r.name))
            });

        let h = run("r1");
        handler.on_run_start(&h).await;
        handler.on_run_end(&h).await;

        let got = events.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert_eq!(got, vec!["start:r1".to_string(), "end:r1".to_string()]);
    }

    #[tokio::test]
    async fn test_run_error_hook() {
        let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let events2 = Arc::clone(&events);

        let handler = GenericHandler::new().with_run_error(move |_r, e| {
            events2
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(format!("error:{e}"))
        });

        handler.on_run_error(&run("r1"), "boom").await;

        let got = events.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert_eq!(got, vec!["error:boom".to_string()]);
    }

    #[tokio::test]
    async fn test_llm_hooks_delegate_to_run_when_unset() {
        let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let events2 = Arc::clone(&events);

        // Only the generic layer is set: typed hooks must forward to it.
        let handler = GenericHandler::new().with_run_start(move |r| {
            events2
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(format!("start:{}", r.name))
        });

        let h = run("llm1");
        handler.on_llm_start(&h, &[]).await;

        let got = events.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert_eq!(got, vec!["start:llm1".to_string()]);
    }

    #[tokio::test]
    async fn test_llm_token_and_thinking_hooks() {
        let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let ev_tokens = Arc::clone(&events);
        let ev_thinking = Arc::clone(&events);

        let handler = GenericHandler::new()
            .with_llm_new_token(move |_r, t| {
                ev_tokens
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(format!("tok:{t}"))
            })
            .with_llm_thinking(move |_r, t| {
                ev_thinking
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(format!("think:{t}"))
            });

        let h = run("llm1");
        handler.on_llm_new_token(&h, "hello").await;
        handler.on_llm_thinking(&h, "hmm").await;

        let got = events.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert_eq!(got, vec!["tok:hello".to_string(), "think:hmm".to_string()]);
    }

    #[tokio::test]
    async fn test_tool_and_retriever_hooks() {
        let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let ev_tool = Arc::clone(&events);
        let ev_retriever = Arc::clone(&events);

        let handler = GenericHandler::new()
            .with_tool_start(move |_r, name, input| {
                ev_tool
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(format!("tool:{name}:{input}"))
            })
            .with_retriever_start(move |_r, q| {
                ev_retriever
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(format!("retriever:{q}"))
            });

        let h = run("t1");
        handler.on_tool_start(&h, "search", "rust").await;
        handler.on_retriever_start(&h, "doc").await;

        let got = events.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert_eq!(
            got,
            vec!["tool:search:rust".to_string(), "retriever:doc".to_string()]
        );
    }
}
