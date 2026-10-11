// lc-chains/src/structured_output.rs
//! 结构化输出免 schema 链(T3, v0.27.0)
//!
//! 与 `LLMChain`(纯文本)和需调用方提供 JSON-Schema 的 `JsonSchemaOutput` 不同,
//! 本链**不需要任何 schema**:由模型根据提示词里用自然语言描述的目标形状自推导
//! 输出结构,并把 LLM 返回的 JSON 解析进结果。
//!
//! 典型用法(可配合 provider 的 json-object 模式强化,非必需):
//! ```ignore
//! use lc_chains::StructuredOutputChain;
//!
//! let chain = StructuredOutputChain::new(
//!     llm,
//!     "为「{topic}」生成摘要,输出 JSON 对象,键为 summaries(数组)与 meta(对象)。",
//! );
//!
//! let mut inputs = HashMap::new();
//! inputs.insert("topic".to_string(), "Rust 内存安全".into());
//! let result = chain.invoke(inputs).await?;   // result["output"] 为解析后的 Value
//! ```
//!
//! 解析失败返回 `ChainError::ExecutionError`(显式,不静默),调用方可感知模型未按
//! 预期输出合法 JSON。解析按 直接解析 → 剥离 ```json 围栏 → 截取首尾花括号 兜底。

use async_trait::async_trait;
use lc_core::runnables::RunnableConfig;
use lc_core::BaseChatModel;
use lc_providers::{wrap_chat_model, ProviderError};
use lc_schema::Message;
use serde_json::Value;
use std::collections::HashMap;

use crate::base::{
    run_chain_with_callbacks, substitute_template, BaseChain, ChainError, ChainResult,
};

/// 追加在提示词末尾的 JSON 单对象指令;默认开启,可用 [`with_json_directive`] 关闭。
///
/// [`with_json_directive`]: StructuredOutputChain::with_json_directive
const JSON_ONLY_DIRECTIVE: &str = "回复必须是单个合法 JSON 对象,不要用 markdown 围栏,不要写任何解释性文字。";

/// 结构化输出免 schema 链。
///
/// 把提示词(模板里 `{var}` 处放目标形状的自然语言描述)交给 LLM,要求其直接输出
/// JSON;本链负责解析成 [`serde_json::Value`] 塞进结果。
#[derive(Clone)]
pub struct StructuredOutputChain {
    /// LLM client。
    llm: crate::BoxedChatModel,
    /// 提示词模板(描述目标输出形状)。
    prompt_template: String,
    /// 输入键名。
    input_key: String,
    /// 输出键名。
    output_key: String,
    /// 链名。
    name: String,
    /// 是否追加 [`JSON_ONLY_DIRECTIVE`]。
    json_directive: bool,
}

impl StructuredOutputChain {
    /// 建链:`llm`(任何 `BaseChatModel`) + 描述输出形状的提示词模板。
    pub fn new<L>(llm: L, prompt_template: impl Into<String>) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: wrap_chat_model(llm),
            prompt_template: prompt_template.into(),
            input_key: "question".to_string(),
            output_key: "output".to_string(),
            name: "structured_output_chain".to_string(),
            json_directive: true,
        }
    }

    /// 设置输入键名。
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// 设置输出键名。
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// 设置链名。
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// 开/关末尾 JSON 单对象指令。默认开;对已内置系统指令的模型可关闭。
    pub fn with_json_directive(mut self, on: bool) -> Self {
        self.json_directive = on;
        self
    }

    /// 渲染提示词:模板替换 + 可选 JSON 指令。
    fn render_prompt(&self, inputs: &HashMap<String, Value>) -> Result<String, ChainError> {
        let mut vars = HashMap::with_capacity(inputs.len());
        for (key, value) in inputs {
            let value_str = match value {
                Value::String(s) => s.clone(),
                _ => value.to_string(),
            };
            vars.insert(key.clone(), value_str);
        }
        let (prompt, missing) = substitute_template(&self.prompt_template, &vars);
        if !missing.is_empty() {
            return Err(ChainError::ExecutionError(format!(
                "Prompt template has unreplaced variable(s): {}",
                missing.join(", ")
            )));
        }
        if self.json_directive {
            Ok(format!("{prompt}\n\n{JSON_ONLY_DIRECTIVE}"))
        } else {
            Ok(prompt)
        }
    }

    /// 实际执行:渲染 → 调 LLM → 解析 JSON。不做回调(纯逻辑)。
    async fn execute(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.validate_inputs(&inputs)?;
        let prompt = self.render_prompt(&inputs)?;
        let messages = vec![Message::human(&prompt)];
        let result = self
            .llm
            .invoke(messages, None)
            .await
            .map_err(|e| ChainError::ExecutionError(format!("LLM call failed: {e}")))?;

        let value = extract_json(&result.content).ok_or_else(|| {
            ChainError::ExecutionError(format!(
                "structured output: model did not return parseable JSON; raw: {:?}",
                truncate(&result.content, 120)
            ))
        })?;

        let mut output = HashMap::new();
        output.insert(self.output_key.clone(), value);
        Ok(output)
    }
}

#[async_trait]
impl BaseChain for StructuredOutputChain {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.execute(inputs).await
    }

    /// 带回调执行:用 `run_chain_with_callbacks` 包链级 start/end/error 事件。
    async fn invoke_with_config(
        &self,
        inputs: HashMap<String, Value>,
        config: Option<RunnableConfig>,
    ) -> Result<ChainResult, ChainError> {
        let chain = self.clone();
        run_chain_with_callbacks(self.name(), inputs, config, move |inputs| {
            let chain = chain.clone();
            async move { chain.execute(inputs).await }
        })
        .await
    }
}

/// 把 LLM 返回的文本解析成 JSON:依次尝试 直接解析 → 剥离 ```json 围栏 →
/// 截取首个 `{` 到末个 `}` 之间(容忍围栏 / 前后解释文字)。
fn extract_json(text: &str) -> Option<Value> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        return Some(v);
    }
    let t = t
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        return Some(v);
    }
    if let (Some(a), Some(b)) = (t.find('{'), t.rfind('}')) {
        if b > a {
            if let Ok(v) = serde_json::from_str::<Value>(&t[a..=b]) {
                return Some(v);
            }
        }
    }
    None
}

/// 截断长文本用于错误信息。
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use serde_json::json;
    use lc_core::language_models::{LLMResult, StreamChunk};
    use lc_core::runnables::RunnableConfig;
    use lc_core::{BaseLanguageModel, Runnable};
    use std::pin::Pin;

    /// 桩 LLM:返回可配置的 `reply`,其它三态全部走打包模式(复制自 stuff.rs 的
    /// `CountingLLM`,锁 runnable 双通道:invoke 与 chat 都先经 `Runnable::invoke`)。
    struct StubChat {
        reply: &'static str,
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for StubChat {
        type Error = ProviderError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: self.reply.to_string(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for StubChat {
        fn model_name(&self) -> &str {
            "mock"
        }
        fn get_num_tokens(&self, t: &str) -> usize {
            t.len()
        }
        fn with_temperature(self, _: f32) -> Self {
            self
        }
        fn with_max_tokens(self, _: usize) -> Self {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for StubChat {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: self.reply.to_string(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            let tokens = [Ok(StreamChunk::new(self.reply))];
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
    }

    /// 围栏包裹的 JSON 应被解析进结果(免 schema,形状由模型在 reply 里给出)。
    #[tokio::test]
    async fn parses_fenced_json_into_output() {
        let stub = StubChat {
            reply: "```json\n{\"summaries\":[\"a\",\"b\"],\"meta\":{\"k\":1}}\n```",
        };
        let chain =
            StructuredOutputChain::new(stub, "输出关于「{topic}」的 JSON。").with_input_key("topic");

        let mut inputs = HashMap::new();
        inputs.insert("topic".to_string(), "Rust".into());
        let result = chain.invoke(inputs).await.expect("链调用失败");

        let out = result.get("output").expect("应有 output 键");
        assert_eq!(out["summaries"][0], json!("a"));
        assert_eq!(out["meta"]["k"], json!(1));
    }

    /// 夹带解释文字(前有散文、后有围栏)也应兜底截取到 JSON。
    #[tokio::test]
    async fn extracts_json_from_prose() {
        let stub = StubChat {
            reply: "好的,结果如下: {\"answer\": 42} 结束。",
        };
        let chain = StructuredOutputChain::new(stub, "按形状输出:{shape}").with_input_key("shape");

        let mut inputs = HashMap::new();
        inputs.insert("shape".to_string(), "一个含 answer 的对象".into());
        let result = chain.invoke(inputs).await.expect("链调用失败");
        assert_eq!(result.get("output").unwrap()["answer"], json!(42));
    }

    /// 模型返回非 JSON → 显式报错(不静默给空)。
    #[tokio::test]
    async fn errors_on_non_json() {
        let stub = StubChat {
            reply: "抱歉,我说不清楚。",
        };
        let chain = StructuredOutputChain::new(stub, "输出 JSON。");

        let mut inputs = HashMap::new();
        inputs.insert("question".to_string(), "x".into());
        let err = chain.invoke(inputs).await.expect_err("应失败");
        assert!(matches!(err, ChainError::ExecutionError(_)));
    }
}