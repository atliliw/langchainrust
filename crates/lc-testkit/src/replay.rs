//! `ReplayProvider`:从录制文件按 FIFO 顺序回放,零网络。
//!
//! 回放**不做消息匹配**(LLMChain 渲染出的 prompt 逐次可变),只按顺序弹出录播。
//! 队列耗尽返回 [`TestkitError::ReplayExhausted`]。

use std::collections::VecDeque;
use std::io::BufRead;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_core::tools::ToolDefinition;
use lc_schema::Message;

use crate::error::TestkitError;
use crate::recording::RecordedExchange;

/// 回放策略:决定并发/乱序场景下一条请求取哪条录播。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReplayStrategy {
    /// 并发 FIFO:按录制顺序弹出,原子 pop 无 UB,但"哪条请求拿到哪条响应"不确定。
    ///
    /// 适合串行/顺序稳定的链;并行场景配**顺序无关断言**(只断言回放全部耗尽、
    /// 各响应结构非空),不断言"第 N 条是工具 A 的响应"。
    #[default]
    Fifo,
    /// 按工具名路由:绑定工具时携带名字(`bind_tools`),回放时从队列里挑
    /// 工具名匹配的第一条录播(请求侧 `exchange.tools` 或响应侧 `tool_calls`)。
    ///
    /// 覆盖"不同工具响应不同、需要精确对应"的并行场景——每个并行分支各绑
    /// 各的工具,拿到各自正确的响应。比"整段消息签名匹配"弱但可复现。
    ByToolName,
    /// 按请求消息**完整签名**精确匹配:请求 `messages` 与录播 `exchange.messages`
    /// **逐条完全相等**(`Message` 的 `PartialEq` 全字段,含 `content` /
    /// `message_type` / `name` / `additional_kwargs` / `tool_calls` 等)才命中。
    ///
    /// 并行乱序下每个请求精确取到自己的响应;录播中无匹配 → 返回明确
    /// [`TestkitError::ReplayNoMatch`],**不做静默 FIFO 兜底**(避免"拿错响应"
    /// 伪装成成功)。适合"同一消息序列必定复现同一响应"的确定性重放——录播
    /// 与请求在任意字段上不一致(如运行时给 `id` 赋了每次不同的值)都会视为
    /// 不匹配,fixture 需保证这些字段稳定。
    Exact,
}

/// 从录制文件回放的零网络 `BaseChatModel`。
///
/// 队列用 `Arc<Mutex<VecDeque>>` 共享,`bind_tools` 返回携带工具集的新实例
/// 时与原件共享同一队列(FIFO 顺序在分支间一致)。
#[derive(Clone)]
pub struct ReplayProvider {
    queue: Arc<Mutex<VecDeque<RecordedExchange>>>,
    model_name: String,
    strategy: ReplayStrategy,
    /// 本实例绑定的工具定义(`ByToolName` 路由的匹配键)。
    tools: Option<Vec<ToolDefinition>>,
}

impl ReplayProvider {
    /// 读 JSONL 录制文件(缺文件/坏行 → `Err`)。
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TestkitError> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut queue = VecDeque::new();
        for line in reader.lines() {
            let line = line?.trim().to_string();
            if line.is_empty() {
                continue;
            }
            let exchange: RecordedExchange = serde_json::from_str(&line).map_err(|e| {
                TestkitError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid recording line: {e}"),
                ))
            })?;
            queue.push_back(exchange);
        }
        Ok(Self {
            queue: Arc::new(Mutex::new(queue)),
            model_name: "replay".to_string(),
            strategy: ReplayStrategy::Fifo,
            tools: None,
        })
    }

    /// 内存构造(手写录播 = MockProvider 的等价物)。
    pub fn from_exchanges(exchanges: Vec<RecordedExchange>) -> Self {
        Self {
            queue: Arc::new(Mutex::new(exchanges.into())),
            model_name: "replay".to_string(),
            strategy: ReplayStrategy::Fifo,
            tools: None,
        }
    }

    /// 单一固定响应:任意请求都返回同一 `response`(最简 mock)。
    pub fn single(response: LLMResult) -> Self {
        Self::from_exchanges(vec![RecordedExchange {
            messages: Vec::new(),
            response,
            tools: None,
        }])
    }

    /// 设置回放策略(默认 FIFO;`ByToolName` 按工具名路由)。
    pub fn with_strategy(mut self, strategy: ReplayStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// 绑定工具:返回携带该工具集的新实例(共享同一回放队列)。
    ///
    /// 回放侧绑工具只是让 agent"绑了工具的循环"前提成立——录播里已含
    /// 请求侧 `tools` 与响应侧 `tool_calls`,回放逻辑无需改。`ByToolName`
    /// 策略下,`tools` 同时是路由的匹配键。
    pub fn bind_tools(&self, tools: Vec<ToolDefinition>) -> Self {
        Self {
            queue: self.queue.clone(),
            model_name: self.model_name.clone(),
            strategy: self.strategy,
            tools: Some(tools),
        }
    }

    /// 剩余录播条数。
    pub fn len(&self) -> usize {
        self.queue.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// 是否已无剩余录播。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// `ByToolName` 的匹配:请求侧绑定的工具名或响应侧请求的工具调用名命中即真。
fn exchange_matches(exchange: &RecordedExchange, tool_name: &str) -> bool {
    if let Some(tools) = &exchange.tools {
        if tools.iter().any(|t| t.function.name == tool_name) {
            return true;
        }
    }
    if let Some(calls) = &exchange.response.tool_calls {
        if calls.iter().any(|c| c.name() == tool_name) {
            return true;
        }
    }
    false
}

/// `Exact` 的匹配:请求消息序列与录播消息序列**逐条完全相等**。
///
/// `Vec<Message>` 的 `PartialEq` 按长度 + 逐元素比较,等价于"消息条数与每条
/// 消息的全部字段(含 `id` / `name` / `additional_kwargs` / `tool_calls`)一致"。
fn messages_match(request: &[Message], recorded: &[Message]) -> bool {
    request == recorded
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for ReplayProvider {
    type Error = TestkitError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, config).await
    }
}

impl BaseLanguageModel<Vec<Message>, LLMResult> for ReplayProvider {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        // 估算:约 4 字符/ token。
        text.chars().count() / 4 + 1
    }

    fn temperature(&self) -> Option<f32> {
        None
    }

    fn max_tokens(&self) -> Option<usize> {
        None
    }

    fn with_temperature(self, _temp: f32) -> Self {
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self {
        self
    }
}

#[async_trait]
impl BaseChatModel for ReplayProvider {
    async fn chat(
        &self,
        messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let exchange = match self.strategy {
            ReplayStrategy::Fifo => queue.pop_front(),
            ReplayStrategy::ByToolName => {
                let want = self
                    .tools
                    .as_ref()
                    .and_then(|tools| tools.first().map(|t| t.function.name.clone()));
                match want {
                    // 按工具名从队列里挑匹配录播;未绑定工具 → 退化为 FIFO。
                    Some(name) => queue
                        .iter()
                        .position(|ex| exchange_matches(ex, &name))
                        .map(|i| queue.remove(i).expect("position 必有元素")),
                    None => queue.pop_front(),
                }
            }
            ReplayStrategy::Exact => {
                // 按完整消息签名精确匹配,允许并行乱序;无匹配 → 明确报错,
                // 绝不静默 FIFO 兜底(否则"拿错响应"会伪装成成功)。
                match queue
                    .iter()
                    .position(|ex| messages_match(&messages, &ex.messages))
                {
                    Some(i) => Some(queue.remove(i).expect("position 必有元素")),
                    None => {
                        return Err(TestkitError::ReplayNoMatch { left: queue.len() });
                    }
                }
            }
        };
        let Some(exchange) = exchange else {
            return Err(TestkitError::ReplayExhausted {
                requested: messages.len(),
            });
        };
        Ok(exchange.response)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        let response = self.chat(messages, config).await?;
        let stream = futures_util::stream::iter(vec![Ok(StreamChunk {
            text: response.content,
            token_usage: response.token_usage,
        })]);
        Ok(Box::pin(stream))
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        // 委托给 inherent `bind_tools`:共享队列、记录工具集、返回新实例。
        Some(Box::new(self.bind_tools(tools)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::language_models::TokenUsage;
    use lc_core::tools::{ToolCall, ToolDefinition};

    fn exchange(content: &str) -> RecordedExchange {
        RecordedExchange {
            messages: vec![Message::system("ping")],
            response: LLMResult {
                content: content.to_string(),
                model: "replay".to_string(),
                token_usage: Some(TokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 2,
                    total_tokens: 3,
                }),
                ..Default::default()
            },
            tools: None,
        }
    }

    /// 带响应侧工具调用名的录播(模拟模型请求调用某工具)。
    fn exchange_with_tool_call(tool_name: &str, content: &str) -> RecordedExchange {
        let mut response = exchange(content).response;
        response.tool_calls = Some(vec![ToolCall::builder("call_1")
            .name(tool_name)
            .arguments("{}".to_string())
            .build()]);
        RecordedExchange {
            response,
            ..exchange(content)
        }
    }

    /// 带请求侧工具定义名的录播(模拟绑定了某工具后发起请求)。
    fn exchange_with_bound_tool(tool_name: &str, content: &str) -> RecordedExchange {
        RecordedExchange {
            tools: Some(vec![ToolDefinition::new(tool_name, "a tool")]),
            ..exchange(content)
        }
    }

    #[tokio::test]
    async fn single_returns_fixed_response_for_any_request() {
        let provider = ReplayProvider::single(exchange("hello").response);
        let result = provider
            .chat(vec![Message::system("any")], None)
            .await
            .unwrap();
        assert_eq!(result.content, "hello");
    }

    #[tokio::test]
    async fn replay_is_fifo_ordered() {
        let provider = ReplayProvider::from_exchanges(vec![exchange("first"), exchange("second")]);
        let first = provider
            .chat(vec![Message::system("a")], None)
            .await
            .unwrap();
        let second = provider
            .chat(vec![Message::system("b")], None)
            .await
            .unwrap();
        assert_eq!(first.content, "first");
        assert_eq!(second.content, "second");
    }

    #[tokio::test]
    async fn replay_exhausted_returns_error() {
        let provider = ReplayProvider::from_exchanges(vec![exchange("only")]);
        provider
            .chat(vec![Message::system("a")], None)
            .await
            .unwrap();
        let err = provider
            .chat(vec![Message::system("b")], None)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            TestkitError::ReplayExhausted { requested: 1 }
        ));
    }

    #[test]
    fn bind_tools_returns_some_and_carries_tools() {
        let provider = ReplayProvider::from_exchanges(vec![exchange("x")]);
        // inherent bind_tools 返回携带工具集的新实例(共享队列)。
        let bound = provider.bind_tools(vec![ToolDefinition::new("calculator", "calc")]);
        assert!(bound.tools.is_some());
        assert_eq!(bound.tools.as_ref().unwrap()[0].function.name, "calculator");
        assert_eq!(provider.len(), 1);
        assert_eq!(bound.len(), 1);
        // trait bind_tools(agent 通过 `Box<dyn BaseChatModel>` 调用)恒为 Some。
        let trait_bound = BaseChatModel::bind_tools(&provider, vec![ToolDefinition::new("x", "y")]);
        assert!(trait_bound.is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn parallel_replay_fifo_is_order_independent() {
        // 并发 FIFO:两条请求同时到达,哪条拿哪条响应不确定,但都能拿到、
        // 且队列最终耗尽。断言用"集合相等"而非"第 N 条是 xx"。
        let provider = ReplayProvider::from_exchanges(vec![
            exchange("first"),
            exchange("second"),
            exchange("third"),
        ]);
        let provider = std::sync::Arc::new(provider);

        let mut handles = Vec::new();
        for _ in 0..3 {
            let p = provider.clone();
            handles.push(tokio::spawn(async move {
                p.chat(vec![Message::system("parallel")], None)
                    .await
                    .expect("并发回放不应失败")
            }));
        }
        let mut contents: Vec<String> = Vec::new();
        for handle in handles {
            contents.push(handle.await.unwrap().content);
        }
        contents.sort();
        assert_eq!(
            contents,
            vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string()
            ]
        );
        assert!(provider.is_empty(), "并发回放应恰好耗尽全部录播");
    }

    #[tokio::test]
    async fn by_tool_name_routes_to_matching_exchange() {
        // 每个并行分支各绑各的工具,应各拿到各自正确的响应——顺序无关。
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_tool_call("search", "search result"),
            exchange_with_tool_call("calc", "calc result"),
        ])
        .with_strategy(ReplayStrategy::ByToolName);

        let search =
            BaseChatModel::bind_tools(&provider, vec![ToolDefinition::new("search", "s")]).unwrap();
        let calc =
            BaseChatModel::bind_tools(&provider, vec![ToolDefinition::new("calc", "c")]).unwrap();

        let calc_res = calc.chat(vec![Message::system("q")], None).await.unwrap();
        let search_res = search.chat(vec![Message::system("q")], None).await.unwrap();

        assert_eq!(search_res.content, "search result");
        assert_eq!(calc_res.content, "calc result");
        assert!(provider.is_empty());
    }

    #[tokio::test]
    async fn by_tool_name_matches_request_side_tools() {
        // 请求侧 `exchange.tools` 也参与匹配(录制自绑定工具后发起的请求)。
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_bound_tool("weather", "sunny"),
            exchange_with_bound_tool("news", "headlines"),
        ])
        .with_strategy(ReplayStrategy::ByToolName);

        let weather =
            BaseChatModel::bind_tools(&provider, vec![ToolDefinition::new("weather", "w")])
                .unwrap();
        let res = weather
            .chat(vec![Message::system("q")], None)
            .await
            .unwrap();
        assert_eq!(res.content, "sunny");
    }

    /// 指定请求消息序列的录播(供 `Exact` 策略按完整签名匹配)。
    fn exchange_with_messages(messages: Vec<Message>, content: &str) -> RecordedExchange {
        RecordedExchange {
            messages,
            ..exchange(content)
        }
    }

    #[tokio::test]
    async fn exact_strategy_matches_by_full_signature_out_of_order() {
        // 两条请求消息序列不同;乱序到达,各自精确取到自己的响应。
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_messages(vec![Message::system("ping")], "pong"),
            exchange_with_messages(vec![Message::human("hello")], "hi"),
        ])
        .with_strategy(ReplayStrategy::Exact);

        let hello_res = provider
            .chat(vec![Message::human("hello")], None)
            .await
            .unwrap();
        let ping_res = provider
            .chat(vec![Message::system("ping")], None)
            .await
            .unwrap();

        assert_eq!(hello_res.content, "hi");
        assert_eq!(ping_res.content, "pong");
        assert!(provider.is_empty(), "两条请求应精确消耗两条录播");
    }

    #[tokio::test]
    async fn exact_strategy_matches_full_message_sequence() {
        // 多轮对话历史(系统 + 用户 + AI)作为整体签名,按完整序列命中。
        let msgs = vec![
            Message::system("You are a calculator."),
            Message::human("2 + 2"),
            Message::ai("I'll compute that."),
        ];
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_messages(vec![Message::human("other")], "wrong"),
            exchange_with_messages(msgs.clone(), "42"),
        ])
        .with_strategy(ReplayStrategy::Exact);

        let res = provider.chat(msgs, None).await.unwrap();
        assert_eq!(res.content, "42");
    }

    #[tokio::test]
    async fn exact_strategy_no_match_returns_explicit_error() {
        // 请求签名在录播中无匹配 → 明确报错(不是静默 FIFO 取错响应)。
        let provider = ReplayProvider::from_exchanges(vec![exchange_with_messages(
            vec![Message::system("ping")],
            "pong",
        )])
        .with_strategy(ReplayStrategy::Exact);

        let err = provider
            .chat(vec![Message::system("different")], None)
            .await
            .unwrap_err();
        assert!(
            matches!(err, TestkitError::ReplayNoMatch { left: 1 }),
            "无匹配应返回 ReplayNoMatch,剩余录播保留"
        );
    }

    #[tokio::test]
    async fn exact_strategy_distinguishes_message_type() {
        // 相同文本、不同消息类型 = 不同签名。
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_messages(vec![Message::human("q")], "human response"),
            exchange_with_messages(vec![Message::system("q")], "system response"),
        ])
        .with_strategy(ReplayStrategy::Exact);

        let human = provider
            .chat(vec![Message::human("q")], None)
            .await
            .unwrap();
        assert_eq!(human.content, "human response");

        let system = provider
            .chat(vec![Message::system("q")], None)
            .await
            .unwrap();
        assert_eq!(system.content, "system response");
    }

    #[tokio::test]
    async fn exact_strategy_distinguishes_tool_calls() {
        // 携带工具调用的消息与纯文本消息 = 不同签名。
        let call = ToolCall::builder("call_1")
            .name("weather")
            .arguments("{}".to_string())
            .build();
        let provider = ReplayProvider::from_exchanges(vec![
            exchange_with_messages(vec![Message::ai("q")], "plain"),
            exchange_with_messages(
                vec![Message::ai_with_tool_calls("q", vec![call.clone()])],
                "with tool",
            ),
        ])
        .with_strategy(ReplayStrategy::Exact);

        let plain = provider.chat(vec![Message::ai("q")], None).await.unwrap();
        assert_eq!(plain.content, "plain");

        let with_tool = provider
            .chat(vec![Message::ai_with_tool_calls("q", vec![call])], None)
            .await
            .unwrap();
        assert_eq!(with_tool.content, "with tool");
    }
}
