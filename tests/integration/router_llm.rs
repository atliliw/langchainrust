//! RouterLLM 集成测试
//!
//! 端到端验证 RouterLLM 的路由 + fallback 行为。

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use langchainrust::{
    BaseChatModel, BaseLanguageModel, LLMResult, Message, RouterError, RouterLLM, RoutingStrategy,
    Runnable, RunnableConfig,
};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// -- Mock LLM that returns Ok or Err on demand --

#[derive(Debug)]
struct MockError(String);
impl std::fmt::Display for MockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for MockError {}

struct MockLLM {
    name: String,
    should_fail: bool,
    calls: Arc<AtomicUsize>,
}

impl MockLLM {
    fn ok(name: &str) -> Self {
        Self {
            name: name.to_string(),
            should_fail: false,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn failing(name: &str) -> Self {
        Self {
            name: name.to_string(),
            should_fail: true,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn calls(&self) -> Arc<AtomicUsize> {
        self.calls.clone()
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for MockLLM {
    type Error = MockError;
    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, config).await
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for MockLLM {
    fn model_name(&self) -> &str {
        &self.name
    }
    fn get_num_tokens(&self, text: &str) -> usize {
        text.len()
    }
    fn with_temperature(self, _: f32) -> Self {
        self
    }
    fn with_max_tokens(self, _: usize) -> Self {
        self
    }
}

#[async_trait]
impl BaseChatModel for MockLLM {
    async fn chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.should_fail {
            Err(MockError(format!("{} is down", self.name)))
        } else {
            Ok(LLMResult {
                content: format!("response from {}", self.name),
                model: self.name.clone(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        let r = self.chat(messages, config).await?;
        let content = r.content;
        Ok(Box::pin(futures_util::stream::once(
            async move { Ok(content) },
        )))
    }
}

#[tokio::test]
async fn test_fallback_primary_ok_no_backup_called() {
    let primary = MockLLM::ok("primary");
    let backup = MockLLM::ok("backup");
    let backup_calls = backup.calls();

    let router = RouterLLM::with_fallbacks(primary, vec![backup]);
    let result = router
        .chat(vec![Message::human("hello")], None)
        .await
        .unwrap();

    assert_eq!(result.content, "response from primary");
    assert_eq!(backup_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn test_fallback_primary_down_uses_backup() {
    let primary = MockLLM::failing("primary");
    let backup = MockLLM::ok("backup");

    let router = RouterLLM::with_fallbacks(primary, vec![backup]);
    let result = router
        .chat(vec![Message::human("hello")], None)
        .await
        .unwrap();

    assert_eq!(result.content, "response from backup");
}

#[tokio::test]
async fn test_all_models_down_returns_all_failed() {
    let primary = MockLLM::failing("primary");
    let backup = MockLLM::failing("backup");

    let router = RouterLLM::with_fallbacks(primary, vec![backup]);
    let result = router.chat(vec![Message::human("hello")], None).await;

    assert!(matches!(result, Err(RouterError::AllFailed { .. })));
}

#[tokio::test]
async fn test_round_robin_distributes_evenly() {
    let a = MockLLM::ok("a");
    let b = MockLLM::ok("b");

    let router = RouterLLM::new(RoutingStrategy::RoundRobin)
        .with_model(a)
        .with_model(b);

    let r1 = router.chat(vec![Message::human("x")], None).await.unwrap();
    let r2 = router.chat(vec![Message::human("x")], None).await.unwrap();

    // Should alternate
    assert_ne!(r1.model, r2.model);
}

#[tokio::test]
async fn test_input_directed_routing() {
    let general = MockLLM::ok("general");
    let code = MockLLM::ok("code");

    let router = RouterLLM::new(RoutingStrategy::InputDirected(Arc::new(|input: &str| {
        if input.contains("code") {
            1
        } else {
            0
        }
    })))
    .with_model(general)
    .with_model(code);

    let r1 = router
        .chat(vec![Message::human("hello")], None)
        .await
        .unwrap();
    assert_eq!(r1.model, "general");

    let r2 = router
        .chat(vec![Message::human("write code")], None)
        .await
        .unwrap();
    assert_eq!(r2.model, "code");
}
