//! RouterLLM 测试 - v0.5.0 #2 模型路由 + Fallback + 负载均衡
//!
//! 验证:
//! - 空路由返回 `RouterError::Empty`
//! - Fallback 策略:primary 成功不调 backup;primary 失败切 backup
//! - 全失败返回 `RouterError::AllFailed`
//! - RoundRobin 轮询分发
//! - InputDirected 按输入选模型
//! - LowestCost / LeastLatency 按元数据选模型
//! - RouterLLM 可作为 `Box<dyn BaseChatModel<Error = RouterError>>` 接入
//! - stream_chat 在主模型失败时 fallback

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use langchainrust::{
    BaseChatModel, BaseLanguageModel, LLMResult, Message, RouterError, RouterLLM, RoutingStrategy,
    Runnable, RunnableConfig,
};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;

// ============================================================================
// Mock chat model
// ============================================================================

#[derive(Debug)]
struct MockError(String);
impl std::fmt::Display for MockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for MockError {}

enum MockResp {
    Ok(String),
    Err(String),
}

/// 可编程 mock:按入队顺序返回 Ok/Err,记录调用次数,可选延迟。
struct MockChatModel {
    name: String,
    resps: Mutex<VecDeque<MockResp>>,
    calls: Arc<AtomicUsize>,
    delay_ms: u64,
}

impl MockChatModel {
    fn new(name: &str, resps: Vec<MockResp>) -> Self {
        Self {
            name: name.to_string(),
            resps: Mutex::new(resps.into_iter().collect()),
            calls: Arc::new(AtomicUsize::new(0)),
            delay_ms: 0,
        }
    }
    fn with_delay(mut self, ms: u64) -> Self {
        self.delay_ms = ms;
        self
    }
    /// 克隆调用计数句柄,即便模型本身被 move 进 router 也可外部读取。
    fn calls_handle(&self) -> Arc<AtomicUsize> {
        self.calls.clone()
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for MockChatModel {
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
impl BaseLanguageModel<Vec<Message>, LLMResult> for MockChatModel {
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
impl BaseChatModel for MockChatModel {
    async fn chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.delay_ms > 0 {
            sleep(Duration::from_millis(self.delay_ms)).await;
        }
        let mut q = self.resps.lock().expect("resps mutex poisoned");
        match q.pop_front() {
            Some(MockResp::Ok(content)) => Ok(LLMResult {
                content,
                model: self.name.clone(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            }),
            Some(MockResp::Err(msg)) => Err(MockError(msg)),
            None => Err(MockError("no more mock responses".to_string())),
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

fn ok(s: &str) -> MockResp {
    MockResp::Ok(s.to_string())
}
fn err(s: &str) -> MockResp {
    MockResp::Err(s.to_string())
}

// ============================================================================
// Tests
// ============================================================================

#[tokio::test]
async fn test_router_empty_returns_empty_error() {
    let router = RouterLLM::new(RoutingStrategy::Fallback);
    let res = router.chat(vec![Message::human("hi")], None).await;
    assert!(matches!(res, Err(RouterError::Empty)));
}

#[tokio::test]
async fn test_router_fallback_primary_succeeds() {
    let primary = MockChatModel::new("a", vec![ok("from-a")]);
    let backup = MockChatModel::new("b", vec![ok("from-b")]);
    let backup_calls = backup.calls_handle();
    let router = RouterLLM::with_fallbacks(primary, vec![backup]);
    let r = router.chat(vec![Message::human("hi")], None).await.unwrap();
    assert_eq!(r.content, "from-a");
    assert_eq!(r.model, "a");
    assert_eq!(
        backup_calls.load(Ordering::SeqCst),
        0,
        "backup must not be called when primary succeeds"
    );
}

#[tokio::test]
async fn test_router_fallback_primary_fails_then_backup() {
    let primary = MockChatModel::new("a", vec![err("a-down")]);
    let backup = MockChatModel::new("b", vec![ok("from-b")]);
    let primary_calls = primary.calls_handle();
    let backup_calls = backup.calls_handle();
    let router = RouterLLM::with_fallbacks(primary, vec![backup]);
    let r = router.chat(vec![Message::human("hi")], None).await.unwrap();
    assert_eq!(r.content, "from-b");
    assert_eq!(r.model, "b");
    assert_eq!(primary_calls.load(Ordering::SeqCst), 1);
    assert_eq!(backup_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_router_all_fail_returns_all_failed() {
    let primary = MockChatModel::new("a", vec![err("a-down")]);
    let backup = MockChatModel::new("b", vec![err("b-down")]);
    let router = RouterLLM::with_fallbacks(primary, vec![backup]);
    let res = router.chat(vec![Message::human("hi")], None).await;
    match res {
        Err(RouterError::AllFailed { tried, .. }) => assert_eq!(tried, 2),
        other => panic!("expected AllFailed, got {:?}", other),
    }
}

#[tokio::test]
async fn test_router_round_robin_distributes() {
    let a = MockChatModel::new("a", vec![ok("a"), ok("a")]);
    let b = MockChatModel::new("b", vec![ok("b"), ok("b")]);
    let c = MockChatModel::new("c", vec![ok("c"), ok("c")]);
    let a_calls = a.calls_handle();
    let b_calls = b.calls_handle();
    let c_calls = c.calls_handle();
    let router = RouterLLM::new(RoutingStrategy::RoundRobin)
        .with_model(a)
        .with_model(b)
        .with_model(c);
    let r1 = router.chat(vec![Message::human("x")], None).await.unwrap();
    let r2 = router.chat(vec![Message::human("x")], None).await.unwrap();
    let r3 = router.chat(vec![Message::human("x")], None).await.unwrap();
    let models = [r1.model, r2.model, r3.model];
    assert!(models.contains(&"a".to_string()));
    assert!(models.contains(&"b".to_string()));
    assert!(models.contains(&"c".to_string()));
    assert_eq!(a_calls.load(Ordering::SeqCst), 1);
    assert_eq!(b_calls.load(Ordering::SeqCst), 1);
    assert_eq!(c_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_router_input_directed_selects_by_input() {
    let a = MockChatModel::new("a", vec![ok("from-a")]);
    let b = MockChatModel::new("b", vec![ok("from-b")]);
    let router = RouterLLM::new(RoutingStrategy::InputDirected(Arc::new(|input: &str| {
        if input.contains("code") {
            1
        } else {
            0
        }
    })))
    .with_model(a)
    .with_model(b);
    let r1 = router
        .chat(vec![Message::human("hello")], None)
        .await
        .unwrap();
    assert_eq!(r1.model, "a");
    let r2 = router
        .chat(vec![Message::human("write code")], None)
        .await
        .unwrap();
    assert_eq!(r2.model, "b");
}

#[tokio::test]
async fn test_router_lowest_cost_picks_cheapest() {
    let pricey = MockChatModel::new("pricey", vec![ok("pricey")]);
    let cheap = MockChatModel::new("cheap", vec![ok("cheap")]);
    let router = RouterLLM::new(RoutingStrategy::LowestCost)
        .with_cost(pricey, 10.0) // idx 0
        .with_cost(cheap, 1.0); // idx 1
    let r = router.chat(vec![Message::human("hi")], None).await.unwrap();
    assert_eq!(r.model, "cheap");
}

#[tokio::test]
async fn test_router_least_latency_picks_fastest_after_warmup() {
    let slow = MockChatModel::new("slow", vec![ok("slow")]).with_delay(80);
    let fast = MockChatModel::new("fast", vec![ok("fast"), ok("fast")]).with_delay(5);
    let router = RouterLLM::new(RoutingStrategy::LeastLatency)
        .with_model(slow) // idx 0
        .with_model(fast); // idx 1
                           // 首次:两者 latency 均为 0,稳定排序保持注册顺序 -> 选 slow(idx 0)
    let r1 = router.chat(vec![Message::human("x")], None).await.unwrap();
    assert_eq!(r1.model, "slow");
    // 次次:slow 已有 ~80ms 延迟,fast 仍为 0 -> 选 fast
    let r2 = router.chat(vec![Message::human("x")], None).await.unwrap();
    assert_eq!(r2.model, "fast");
}

#[tokio::test]
async fn test_router_usable_as_boxed_base_chat_model() {
    let router = RouterLLM::with_fallbacks(MockChatModel::new("a", vec![ok("ok")]), vec![]);
    let boxed: Box<dyn BaseChatModel<Error = RouterError> + Send + Sync> = Box::new(router);
    let r = boxed.chat(vec![Message::human("hi")], None).await.unwrap();
    assert_eq!(r.content, "ok");
}

#[tokio::test]
async fn test_router_stream_chat_falls_back() {
    let primary = MockChatModel::new("a", vec![err("a-down")]);
    let backup = MockChatModel::new("b", vec![ok("from-b")]);
    let router = RouterLLM::with_fallbacks(primary, vec![backup]);
    let mut s = router
        .stream_chat(vec![Message::human("hi")], None)
        .await
        .unwrap();
    let token = s.next().await.unwrap().unwrap();
    assert_eq!(token, "from-b");
}

#[tokio::test]
async fn test_router_invoke_routes_through_chat() {
    let router = RouterLLM::with_fallbacks(
        MockChatModel::new("a", vec![err("a-down")]),
        vec![MockChatModel::new("b", vec![ok("from-b")])],
    );
    // Runnable::invoke 应与 chat 一致地走 fallback
    let r = router
        .invoke(vec![Message::human("hi")], None)
        .await
        .unwrap();
    assert_eq!(r.content, "from-b");
}
