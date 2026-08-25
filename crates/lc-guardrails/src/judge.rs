//! 敏感泄露 LLM 裁判(P2-3)
//!
//! `SensitiveInfoGuardrail` 的高误报"提及"词(如 password/密码)上下文敏感命中后,
//! 是否真泄露由 LLM 裁判二次判断——"真实的密钥/凭证值"拦截,"如何安全保存密码"
//! 这类正常提及放行,以此降低误报。
//!
//! 复用 [`lc_core::judge::structured_call`] 这条共享裁判基础设施(与 lc-evaluation
//! 的 Faithfulness / Pairwise 裁判同源,evaluation P2-6):优先走 `bind_tools`
//! 拿 `tool_calls` 结构化参数,模型不支持工具绑定或返回纯文本时回落文本解析。
//!
//! 这也是 P2-4 死依赖处理的一部分:`lc-core` 此前在 lc-guardrails 的 src 零引用,
//! 经此真实连接(计划书"要么移除,要么实际连接"的实际连接路线)。

use async_trait::async_trait;

use crate::guardrail::GuardrailError;

use lc_core::judge::{structured_call, StructuredJudgeError};
use lc_core::tools::ToolDefinition;
use lc_core::BaseChatModel;
use lc_schema::Message;

/// 敏感泄露裁判:对疑似敏感输出做"真实泄露 vs 正常提及"的二次判断。
///
/// `judge` 返回 `true` = 判定为真实泄露(应拦截);`false` = 正常提及(应放行)。
#[async_trait]
pub trait SensitiveJudge: Send + Sync {
    /// 裁判名称。
    fn name(&self) -> &str;

    /// 判断一段文本是否泄露真实的敏感信息。
    async fn judge(&self, text: &str) -> Result<bool, GuardrailError>;
}

/// 基于共享 LLM 裁判基础设施的敏感泄露裁判。
///
/// 用 [`structured_call`] 让裁判以结构化参数提交 `{"is_leak": bool, "reason": "..."}`;
/// 模型不支持工具绑定或返回纯文本时,回落 [`parse_leak_text`] 的文本解析。
pub struct LlmSensitiveJudge<M: BaseChatModel> {
    judge: M,
}

impl<M: BaseChatModel> LlmSensitiveJudge<M> {
    /// 用给定的 LLM 创建裁判。
    pub fn new(judge: M) -> Self {
        Self { judge }
    }
}

#[async_trait]
impl<M: BaseChatModel> SensitiveJudge for LlmSensitiveJudge<M> {
    fn name(&self) -> &str {
        "llm_sensitive_judge"
    }

    async fn judge(&self, text: &str) -> Result<bool, GuardrailError> {
        let system =
            "你是安全审查员。判断给定文本是否泄露了真实的敏感信息(真实的密码、密钥、凭证值等)。\
                      正常提及、教学或讨论场景(如'如何安全保存密码')不算泄露。\
                      调用 check_leak 工具提交判定。"
                .to_string();
        let user = format!("待审查文本:\n{text}\n\n这段文本是否泄露了真实的敏感信息?");
        let messages = vec![Message::system(system), Message::human(user)];

        let args: LeakArgs = structured_call(&self.judge, leak_tool(), messages, |raw| {
            let is_leak = parse_leak_text(raw).ok_or_else(|| {
                StructuredJudgeError::Parse(format!(
                    "failed to parse leak verdict from judge reply: {}",
                    lc_core::judge::truncate(raw, 200)
                ))
            })?;
            Ok(LeakArgs {
                is_leak,
                reason: String::new(),
            })
        })
        .await
        .map_err(|e| GuardrailError::Judge(e.to_string()))?;
        Ok(args.is_leak)
    }
}

/// 结构化判定参数(经 tool_calls 返回)。
#[derive(Debug, serde::Deserialize)]
struct LeakArgs {
    #[serde(default)]
    is_leak: bool,
    /// 让 LLM 附上简短理由(改善判定质量),当前不消费。
    #[serde(default)]
    #[allow(dead_code)]
    reason: String,
}

/// 构建判定工具:让 LLM 以 `{"is_leak": bool, "reason": "..."}` 提交判定。
fn leak_tool() -> ToolDefinition {
    ToolDefinition::new(
        "check_leak",
        "判断文本是否泄露真实的敏感信息,提交布尔判定。",
    )
    .with_parameters(serde_json::json!({
        "type": "object",
        "properties": {
            "is_leak": { "type": "boolean", "description": "是否真实泄露敏感信息" },
            "reason": { "type": "string", "description": "简短依据" }
        },
        "required": ["is_leak", "reason"]
    }))
}

/// 解析"是/否泄露"。无任何是/否标记时返回 `None`(解析失败,由调用方报错),
/// 而非静默默认——避免 LLM 跑题回复被当成"未泄露"。
fn parse_leak_text(raw: &str) -> Option<bool> {
    let lower = raw.to_lowercase();
    // 先判否定(避免"不是""不能"被"是""能"误判;否定词优先于肯定词)。
    if lower.contains("否")
        || lower.contains("no")
        || lower.contains("不能")
        || lower.contains("不是")
        || lower.contains("false")
    {
        return Some(false);
    }
    if lower.contains("是")
        || lower.contains("yes")
        || lower.contains("能")
        || lower.contains("true")
    {
        return Some(true);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::Stream;
    use lc_core::language_models::LLMResult;
    use lc_core::{BaseLanguageModel, Runnable, RunnableConfig};
    use lc_schema::MessageType;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Debug)]
    struct MockJudgeError(String);
    impl std::fmt::Display for MockJudgeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::error::Error for MockJudgeError {}

    /// 依次返回预设回复的 mock 裁判:不实现 `bind_tools`,
    /// 走 `structured_call` 的文本回落路径(P2-3 与 evaluation 同源的可测路径)。
    struct SeqMockJudge {
        replies: Vec<String>,
        call: Arc<AtomicUsize>,
        last_user: Arc<Mutex<Option<String>>>,
    }
    impl SeqMockJudge {
        fn new(replies: Vec<String>) -> Self {
            Self {
                replies,
                call: Arc::new(AtomicUsize::new(0)),
                last_user: Arc::new(Mutex::new(None)),
            }
        }
        fn last_user_content(&self) -> String {
            self.last_user
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
                .unwrap_or_default()
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for SeqMockJudge {
        type Error = MockJudgeError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(MockJudgeError("use chat".into()))
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for SeqMockJudge {
        fn model_name(&self) -> &str {
            "seq-mock"
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
    impl BaseChatModel for SeqMockJudge {
        async fn chat(
            &self,
            messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let idx = self.call.fetch_add(1, Ordering::SeqCst);
            let reply = self.replies.get(idx).cloned().unwrap_or_default();
            if let Some(human) = messages
                .iter()
                .find(|m| m.message_type == MessageType::Human)
            {
                *self.last_user.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(human.content.clone());
            }
            Ok(LLMResult {
                content: reply,
                model: "seq-mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
        {
            Err(MockJudgeError("not supported".into()))
        }
    }

    #[tokio::test]
    async fn test_judge_returns_leak_on_yes() {
        let mock = SeqMockJudge::new(vec!["是".into()]);
        let judge = LlmSensitiveJudge::new(mock);
        let result = judge.judge("密码是 abc123456").await.unwrap();
        assert!(result, "裁判判为是 → 应判定为泄露");
    }

    #[tokio::test]
    async fn test_judge_returns_no_leak_on_no() {
        let mock = SeqMockJudge::new(vec!["否".into()]);
        let judge = LlmSensitiveJudge::new(mock);
        let result = judge.judge("如何安全保存密码").await.unwrap();
        assert!(!result, "裁判判为否 → 应判定为正常提及");
    }

    #[tokio::test]
    async fn test_judge_parse_failure_errors() {
        // 文本回落解析失败 → 显式 Err,不静默默认。
        let mock = SeqMockJudge::new(vec!["无法判断".into()]);
        let judge = LlmSensitiveJudge::new(mock);
        let result = judge.judge("text").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_judge_sends_text_to_model() {
        let mock = SeqMockJudge::new(vec!["是".into()]);
        let judge = LlmSensitiveJudge::new(mock);
        judge.judge("我的 token 是 abc").await.unwrap();
        let sent = judge.judge.last_user_content();
        assert!(
            sent.contains("我的 token 是 abc"),
            "裁判应收到待审查文本, 实际: {sent}"
        );
    }

    #[test]
    fn test_parse_leak_text() {
        assert_eq!(parse_leak_text("是"), Some(true));
        assert_eq!(parse_leak_text("yes"), Some(true));
        assert_eq!(parse_leak_text("是,泄露了"), Some(true));
        assert_eq!(parse_leak_text("否"), Some(false));
        assert_eq!(parse_leak_text("no"), Some(false));
        assert_eq!(parse_leak_text("不是"), Some(false));
        // 无任何是/否标记 = 解析失败,不应静默默认
        assert_eq!(parse_leak_text("我看不出"), None);
    }
}
