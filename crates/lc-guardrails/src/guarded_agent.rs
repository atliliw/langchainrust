//! GuardedAgent - 带 Guardrails 的 Agent 包装器

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use lc_agents::AgentExecutor;
use lc_chains::BaseChain;

use super::guardrail::{ChunkAction, GuardrailError, GuardrailsConfig};
use super::runner::{GuardrailRunner, GuardrailViolation, OutputValidation};

/// Guardable 执行单元的错误类型。
type DynError = Box<dyn std::error::Error + Send + Sync>;

/// 流式输出块:单次吐出的文本 + 是否结尾。
#[derive(Debug)]
pub struct GuardableChunk {
    /// 单次吐出的文本
    pub token: String,
    /// 是否为结尾块
    pub is_final: bool,
}

/// GuardedAgent 可包装的执行单元(P1-3 解耦)。
///
/// 只依赖此 trait,不直接耦合 `AgentExecutor`:
/// - [`AgentExecutor`] 直接实现
/// - 任意 [`BaseChain`] 通过 [`ChainGuardable`] 适配器获得实现
#[async_trait]
pub trait Guardable: Send + Sync {
    /// 字符串进、字符串出。
    async fn invoke_str(&self, input: &str) -> Result<String, DynError>;

    /// 流式输出;不支持的实现返回错误,`GuardedAgent` 回退为一次性 `invoke_str`。
    async fn stream_str(
        &self,
        input: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GuardableChunk, DynError>> + Send>>, DynError>;
}

#[async_trait]
impl Guardable for AgentExecutor {
    async fn invoke_str(&self, input: &str) -> Result<String, DynError> {
        self.invoke(input.to_string())
            .await
            .map_err(|e| Box::new(e) as DynError)
    }

    async fn stream_str(
        &self,
        input: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GuardableChunk, DynError>> + Send>>, DynError>
    {
        use futures_util::StreamExt;
        use lc_agents::streaming::state::AgentStreamEvent;

        // 把 Agent 事件流映射为面向用户的输出块:只保留 FinalAnswer,
        // ToolStart / ToolEnd 属中间过程,不进入护栏检查面。
        let stream = self
            .stream(input.to_string())
            .filter_map(|event| async move {
                match event {
                    Ok(AgentStreamEvent::FinalAnswer { content }) => Some(Ok(GuardableChunk {
                        token: content,
                        is_final: true,
                    })),
                    Ok(AgentStreamEvent::Error { message }) => Some(Err(DynError::from(message))),
                    Ok(_) => None, // ToolStart / ToolEnd
                    Err(e) => Some(Err(Box::new(e) as DynError)),
                }
            });
        Ok(Box::pin(stream))
    }
}

/// `BaseChain` 的 Guardable 适配器(P1-3 解耦)。
///
/// 用 `input_keys()[0]` / `output_keys()[0]` 做字符串收发;
/// 输出键缺失或非字符串时返回显式错误,不静默吞掉。
///
/// 为什么需要 adapter 而非 `impl Guardable for dyn BaseChain` / blanket:
/// - blanket `impl<T: BaseChain> Guardable for T` 与 `impl Guardable for AgentExecutor`
///   因相干性冲突(rustc 无法排除 AgentExecutor 未来实现 BaseChain)而无法共存;
/// - `dyn BaseChain` → `dyn Guardable` 的非 supertrait 强转不存在(`Unsize` 不满足),
///   无法自动把 `Arc<dyn BaseChain>` 塞进 `Arc<dyn Guardable>`。
///
/// 因此用 `ChainGuardable` 显式桥接,`GuardedAgent::from_chain` 提供入口。
pub struct ChainGuardable(pub Arc<dyn BaseChain>);

#[async_trait]
impl Guardable for ChainGuardable {
    async fn invoke_str(&self, input: &str) -> Result<String, DynError> {
        let chain: &dyn BaseChain = self.0.as_ref();
        let input_key = chain
            .input_keys()
            .first()
            .ok_or_else(|| DynError::from("chain has no input key"))?
            .to_string();
        let output_key = chain
            .output_keys()
            .first()
            .ok_or_else(|| DynError::from("chain has no output key"))?
            .to_string();

        let mut inputs = HashMap::new();
        inputs.insert(input_key, serde_json::Value::String(input.to_string()));

        let result = chain
            .invoke(inputs)
            .await
            .map_err(|e| Box::new(e) as DynError)?;

        let value = result
            .get(&output_key)
            .ok_or_else(|| DynError::from(format!("chain output has no key {:?}", output_key)))?;
        value.as_str().map(|s| s.to_string()).ok_or_else(|| {
            DynError::from(format!("chain output key {:?} is not a string", output_key))
        })
    }

    async fn stream_str(
        &self,
        input: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GuardableChunk, DynError>> + Send>>, DynError>
    {
        use futures_util::StreamExt;

        let chain: &dyn BaseChain = self.0.as_ref();
        let input_key = chain
            .input_keys()
            .first()
            .ok_or_else(|| DynError::from("chain has no input key"))?
            .to_string();

        let mut inputs = HashMap::new();
        inputs.insert(input_key, serde_json::Value::String(input.to_string()));

        let stream = chain
            .stream(inputs)
            .await
            .map_err(|e| Box::new(e) as DynError)?;

        let mapped = stream.map(|item| {
            item.map(|t| GuardableChunk {
                token: t.token,
                is_final: t.is_final,
            })
            .map_err(|e| Box::new(e) as DynError)
        });
        Ok(Box::pin(mapped))
    }
}

/// 带 Guardrails 的 Agent 包装器
///
/// `invoke` 时:验证输入 -> 执行 Guardable -> 验证输出。
/// `invoke_stream` 时:两阶段流式护栏(P1-4)。
pub struct GuardedAgent {
    inner: Arc<dyn Guardable>,
    runner: GuardrailRunner,
}

impl GuardedAgent {
    /// 用任意 [`Guardable`] 构造。`Arc<AgentExecutor>` 会自动强转为 `Arc<dyn Guardable>`。
    pub fn new(inner: Arc<dyn Guardable>, config: GuardrailsConfig) -> Self {
        Self {
            inner,
            runner: GuardrailRunner::new(config),
        }
    }

    /// 用任意 [`BaseChain`] 构造(P1-3 解耦)。
    ///
    /// `Arc<dyn BaseChain>` 无法直接强转为 `Arc<dyn Guardable>`(非 supertrait),
    /// 因此经 [`ChainGuardable`] 适配器桥接。`Arc<EchoChain>` 之类的具体链
    /// 会先自动强转为 `Arc<dyn BaseChain>` 再包装。
    pub fn from_chain(chain: Arc<dyn BaseChain>, config: GuardrailsConfig) -> Self {
        Self::new(Arc::new(ChainGuardable(chain)), config)
    }

    /// 执行:验输入 -> Guardable -> 验输出。
    ///
    /// 拦截时返回带部分输出 + 用户建议的 [`GuardrailError::Blocked`](P1-1/P1-6)。
    pub async fn invoke(&mut self, input: String) -> Result<String, GuardrailError> {
        if let Err(e) = self.runner.validate_input(&input).await {
            return Err(match e {
                GuardrailError::Blocked { reason, .. } => GuardrailError::Blocked {
                    reason,
                    partial: Some(input),
                    suggestion: Some("please adjust your input and retry".to_string()),
                },
                other => other,
            });
        }

        let output = self
            .inner
            .invoke_str(&input)
            .await
            .map_err(|e| GuardrailError::AgentError(e.to_string()))?;

        match self.runner.validate_output(&output).await {
            OutputValidation::Passed(value) => Ok(value),
            OutputValidation::Blocked { reason, partial } => Err(GuardrailError::from_blocked(
                reason,
                partial,
                "output was blocked by a safety guardrail; please adjust your request and retry, or omit sensitive content"
                    .to_string(),
            )),
        }
    }

    /// 两阶段流式执行(P1-4)。
    ///
    /// 阶段一:每个输出块过流式护栏(带滑动窗口,防跨块切断关键词);
    /// 阶段二:流结束后对完整输出复查 [`GuardrailRunner::validate_output`]。
    /// 输入护栏在流启动前同步验证。
    pub async fn invoke_stream(
        &mut self,
        input: String,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<GuardableChunk, GuardrailError>> + Send>>,
        GuardrailError,
    > {
        use futures_util::StreamExt;

        self.runner.validate_input(&input).await?;

        let inner = self.inner.clone();
        let raw_stream = inner
            .stream_str(&input)
            .await
            .map_err(|e| GuardrailError::AgentError(e.to_string()))?;

        // 两阶段各自持有一份 runner 克隆,避免把 `self` 借进返回的流。
        let mut phase2_runner = self.runner.clone();
        // 阶段一的状态(滑动窗口 tail、违规累积 runner、累积输出 full)跨 chunk 共享。
        // `then` 的闭包是 FnMut:每次调用同步 clone 一份 Arc move 进 async 块,
        // 状态本体留在 Arc 里,跨 chunk 持久。
        let tail = Arc::new(tokio::sync::Mutex::new(String::new()));
        let phase1_runner = Arc::new(tokio::sync::Mutex::new(self.runner.clone()));
        let full = Arc::new(tokio::sync::Mutex::new(String::new()));
        let finalize_full = full.clone();
        const TAIL_WINDOW: usize = 24;

        let phase1 = raw_stream.then(move |item| {
            let tail = tail.clone();
            let runner = phase1_runner.clone();
            let full = full.clone();
            async move {
                let chunk = item.map_err(|e| GuardrailError::AgentError(e.to_string()))?;
                let token = chunk.token;
                // 滑动窗口探测:tail + chunk,跨块切断的关键词也能被命中。
                let probe = {
                    let t = tail.lock().await;
                    format!("{}{}", *t, token)
                };
                let action = runner.lock().await.validate_stream_chunk(&probe).await;
                let emitted = match action {
                    ChunkAction::Pass => token,
                    ChunkAction::Replace(new_value) => new_value,
                    ChunkAction::Block => {
                        let partial = tail.lock().await.clone();
                        return Err(GuardrailError::Blocked {
                            reason: "streaming output was blocked by a guardrail".to_string(),
                            partial: Some(partial),
                            suggestion: Some(
                                "output was blocked by a safety guardrail; please adjust your request and retry"
                                    .to_string(),
                            ),
                        });
                    }
                };
                full.lock().await.push_str(&emitted);
                // 更新滑动窗口:只保留最近 TAIL_WINDOW 个字符。
                let new_tail: String = emitted
                    .chars()
                    .rev()
                    .take(TAIL_WINDOW)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                *tail.lock().await = new_tail;
                Ok(GuardableChunk {
                    token: emitted,
                    is_final: false,
                })
            }
        });

        let finalize = futures_util::stream::once(async move {
            let full_text = finalize_full.lock().await.clone();
            match phase2_runner.validate_output(&full_text).await {
                // 阶段一已把(可能改写过的)全部块发完,消费者拼接即得完整输出;
                // 阶段二通过时只发空 token 的结束标记,不重复输出。
                OutputValidation::Passed(_value) => Ok(GuardableChunk {
                    token: String::new(),
                    is_final: true,
                }),
                OutputValidation::Blocked { reason, partial } => Err(GuardrailError::from_blocked(
                    reason,
                    partial,
                    "final output re-check failed; please adjust your request and retry, or omit sensitive content"
                        .to_string(),
                )),
            }
        });

        Ok(Box::pin(phase1.chain(finalize)))
    }

    /// 获取违规记录快照(含流式路径的记录:两阶段 runner 与 `self.runner` 共享日志)。
    pub fn violations(&self) -> Vec<GuardrailViolation> {
        self.runner.violations()
    }

    /// 清理违规记录(P1-2)。
    pub fn clear_violations(&mut self) {
        self.runner.clear_violations();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validators::MaxLengthGuardrail;
    use lc_agents::{BaseAgent, FunctionCallingAgent};
    use lc_chains::base::{ChainError, ChainResult, ChainStream, StreamToken};
    use lc_providers::{OpenAIChat, OpenAIConfig};
    use serde_json::Value;

    fn guarded_with_maxlen(max: usize) -> GuardedAgent {
        let llm = OpenAIChat::new(OpenAIConfig::default());
        let agent = FunctionCallingAgent::new(llm, vec![], None);
        let executor = Arc::new(AgentExecutor::new(
            Arc::new(agent) as Arc<dyn BaseAgent>,
            vec![],
        ));
        let config =
            GuardrailsConfig::new().with_input(Arc::new(MaxLengthGuardrail::new(max)) as Arc<_>);
        GuardedAgent::new(executor, config)
    }

    #[tokio::test]
    async fn test_blocks_long_input_before_agent() {
        // 输入超过 3 字符,被 MaxLength 拦截,不调用 Agent(不触网)
        let mut g = guarded_with_maxlen(3);
        let result = g.invoke("this is too long input".to_string()).await;
        assert!(result.is_err());
        assert_eq!(g.violations().len(), 1);
        // 确认是 Blocked(携带 partial + suggestion)而非 AgentError
        match result.unwrap_err() {
            GuardrailError::Blocked {
                partial,
                suggestion,
                ..
            } => {
                assert!(partial.is_some());
                assert!(suggestion.is_some());
            }
            other => panic!("应为 Blocked, 实际: {:?}", other),
        }
    }

    /// 简单的 echo Chain:input -> "echo:{input}"。
    struct EchoChain;
    #[async_trait]
    impl BaseChain for EchoChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["input"]
        }
        fn output_keys(&self) -> Vec<&str> {
            vec!["output"]
        }
        async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            let mut out = HashMap::new();
            if let Some(Value::String(s)) = inputs.get("input") {
                out.insert("output".to_string(), Value::String(format!("echo:{}", s)));
            }
            Ok(out)
        }
    }

    /// 分块输出的 Chain:模拟逐 token 流。
    struct TokenChain;
    #[async_trait]
    impl BaseChain for TokenChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["input"]
        }
        fn output_keys(&self) -> Vec<&str> {
            vec!["output"]
        }
        async fn invoke(&self, _inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            Ok(HashMap::new())
        }
        async fn stream(&self, _inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
            let tokens = vec![
                Ok(StreamToken {
                    token: "Hello ".to_string(),
                    is_final: false,
                }),
                Ok(StreamToken {
                    token: "world".to_string(),
                    is_final: false,
                }),
            ];
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
    }

    #[tokio::test]
    async fn test_guardable_chain_invoke() {
        // 任意 Arc<dyn BaseChain> 经 ChainGuardable 适配(P1-3 解耦)。
        let chain: Arc<dyn BaseChain> = Arc::new(EchoChain);
        let mut g = GuardedAgent::from_chain(chain, GuardrailsConfig::new());
        let result = g.invoke("hi".to_string()).await.unwrap();
        assert_eq!(result, "echo:hi");
    }

    #[tokio::test]
    async fn test_invoke_stream_two_phase_passes() {
        // 无护栏:两阶段流式正常走完,最终输出 = 所有块拼接。
        let chain: Arc<dyn BaseChain> = Arc::new(TokenChain);
        let mut g = GuardedAgent::from_chain(chain, GuardrailsConfig::new());
        let mut stream = g.invoke_stream("q".to_string()).await.unwrap();

        use futures_util::StreamExt;
        let mut collected = String::new();
        let mut finals = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            if chunk.is_final {
                finals += 1;
            }
            collected.push_str(&chunk.token);
        }
        assert_eq!(collected, "Hello world");
        assert_eq!(finals, 1);
    }

    /// 命中 "world" 即 Block 的本地流式护栏。
    struct BlockOnWorld;
    #[async_trait]
    impl crate::guardrail::StreamingOutputGuardrail for BlockOnWorld {
        fn name(&self) -> &str {
            "BlockOnWorld"
        }
        async fn validate_chunk(&self, chunk: &str) -> crate::guardrail::ChunkAction {
            if chunk.contains("world") {
                crate::guardrail::ChunkAction::Block
            } else {
                crate::guardrail::ChunkAction::Pass
            }
        }
    }

    #[tokio::test]
    async fn test_invoke_stream_blocks_keyword() {
        // 阶段一流式护栏命中关键词 → 流中途返回 Blocked(携带已输出部分)。
        let chain: Arc<dyn BaseChain> = Arc::new(TokenChain);
        let config = GuardrailsConfig::new().with_streaming(
            Arc::new(BlockOnWorld) as Arc<dyn crate::guardrail::StreamingOutputGuardrail>
        );
        let mut g = GuardedAgent::from_chain(chain, config);
        let mut stream = g.invoke_stream("q".to_string()).await.unwrap();

        use futures_util::StreamExt;
        let mut saw_error = false;
        while let Some(chunk) = stream.next().await {
            if chunk.is_err() {
                saw_error = true;
            }
        }
        assert!(saw_error);
        assert!(!g.violations().is_empty());
    }

    #[tokio::test]
    async fn test_clear_violations_passthrough() {
        let mut g = guarded_with_maxlen(3);
        let _ = g.invoke("this is too long input".to_string()).await;
        assert!(!g.violations().is_empty());
        g.clear_violations();
        assert!(g.violations().is_empty());
    }
}
