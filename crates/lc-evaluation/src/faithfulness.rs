//! 忠实度评测器:检测回答是否忠于参考上下文(幻觉检测)。
//!
//! 思路来自 Ragas 的 faithfulness:把回答拆成原子陈述,
//! 逐条判断能否从参考上下文推导出来,通过率即为忠实度。
//! 这里 reference 充当"上下文/检索内容",prediction 是待检测的回答。

use async_trait::async_trait;
use futures_util::stream::{self, StreamExt};
use serde::Deserialize;

use lc_core::judge::{structured_call, truncate, StructuredJudgeError};
use lc_core::tools::ToolDefinition;
use lc_core::BaseChatModel;
use lc_schema::Message;

use super::{EvalError, Evaluator, Score};

/// P1-5: 单次评测最多同时打给 judge 的陈述验证并发数(防限流 N 路全挂)。
const MAX_CONCURRENT_VERIFY: usize = 4;

/// P2-5: 单条陈述的裁判 prompt 里参考上下文的字符上限。
/// 完整长参考只截一次、N 条陈述复用,避免每条重复传输整段上下文。
const DEFAULT_MAX_CONTEXT_CHARS: usize = 2000;

/// 忠实度评测器(幻觉检测):回答有多忠于参考上下文。
///
/// 把 prediction 拆成原子陈述,逐条问裁判能否从 reference 推导,
/// 通过率 = 能推导的陈述数 / 总陈述数。
pub struct Faithfulness<M: BaseChatModel> {
    judge: M,
    /// 用 LLM 拆原子陈述(默认 false,用规则按标点拆)
    llm_split: bool,
    /// 空预测(无可验证陈述)的得分,默认 0.0(没回答=不忠实)。
    empty_score: f64,
    /// 参考上下文单条传输上限(字符,默认 [`DEFAULT_MAX_CONTEXT_CHARS`])。
    max_context_chars: usize,
}

/// 把回答拆成原子陈述(按句号、问号、感叹号、分号、换行切分)。
fn split_claims(prediction: &str) -> Vec<String> {
    prediction
        .split(['。', '.', '!', '?', '；', ';', '\n'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

impl<M: BaseChatModel> Faithfulness<M> {
    /// 创建忠实度评测器。
    pub fn new(judge: M) -> Self {
        Self {
            judge,
            llm_split: false,
            empty_score: 0.0, // P0-2: 空预测默认 0 分(没回答=不忠实)
            max_context_chars: DEFAULT_MAX_CONTEXT_CHARS,
        }
    }

    /// 用 LLM 拆原子陈述(默认按标点规则拆;LLM 拆能处理逗号复合句)
    pub fn with_llm_split(mut self, v: bool) -> Self {
        self.llm_split = v;
        self
    }

    /// 空预测的得分:默认 0.0(没回答=不忠实),可设 1.0 表示"没编造即忠实"
    pub fn with_empty_score(mut self, score: f64) -> Self {
        self.empty_score = score;
        self
    }

    /// 参考上下文单条传输上限(字符)。P2-5: 默认 2000,防止长参考被每条陈述重复完整塞进 prompt。
    pub fn with_max_context_chars(mut self, max: usize) -> Self {
        self.max_context_chars = max;
        self
    }

    /// 问裁判:单条陈述能否从上下文推导。
    async fn verify_claim(&self, context: &str, claim: &str) -> Result<bool, EvalError> {
        let system =
            "你是事实核查员。判断给定的陈述能否从参考上下文中推导出来。调用 check_claim 工具提交判定。"
                .to_string();
        let user =
            format!("参考上下文:\n{context}\n\n陈述:\n{claim}\n\n这条陈述能从上下文推导出来吗?");
        let messages = vec![Message::system(system), Message::human(user)];

        // P0-1: 优先结构化输出(verdict 布尔);不支持工具绑定的模型走文本解析回落。
        let args: VerdictArgs = structured_call(&self.judge, verdict_tool(), messages, |raw| {
            let verdict = parse_yes_no(raw).ok_or_else(|| {
                StructuredJudgeError::Parse(format!(
                    "failed to parse yes/no from judge reply: {}",
                    truncate(raw, 200)
                ))
            })?;
            Ok(VerdictArgs {
                verdict,
                reason: String::new(),
            })
        })
        .await?;
        Ok(args.verdict)
    }

    /// 用 LLM 把回答拆成原子陈述(每行一条),处理规则拆不动的复合句。
    async fn split_claims_llm(&self, prediction: &str) -> Result<Vec<String>, EvalError> {
        let system =
            "你是文本分析助手。把回答拆成原子陈述,每条一行,只输出陈述本身,不要编号不要解释。"
                .to_string();
        let user = format!("回答:\n{prediction}\n\n把它拆成原子陈述,每行一条:");
        let result = self
            .judge
            .chat_with_system(system, vec![Message::human(user)])
            .await
            .map_err(|e| EvalError::PredictorError(e.to_string()))?;
        Ok(result
            .content
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect())
    }
}

#[async_trait]
impl<M: BaseChatModel> Evaluator for Faithfulness<M> {
    async fn eval(
        &self,
        _input: &str,
        prediction: &str,
        reference: &str,
    ) -> Result<Score, EvalError> {
        let claims = if self.llm_split {
            self.split_claims_llm(prediction).await?
        } else {
            split_claims(prediction)
        };
        if claims.is_empty() {
            return Ok(Score::new(self.empty_score).with_label("no_claims"));
        }
        // P2-5: 参考上下文只截取一次,各陈述复用同一份(避免完整长参考被 N 条重复传输)。
        let context = truncate(reference, self.max_context_chars);
        // 并发验证各陈述(每条一次 LLM 调用),但用 buffer_unordered 限流:
        // P1-5——join_all 无上限并发打同一个 judge,命中限流会 N 路全挂。
        // `ctx` 是 Copy 的引用,闭包可反复捕获;若直接捕获 `context` 会被 async move
        // 逐条 move 出去,map(FnMut) 编译不过。
        let ctx = &context;
        let total = claims.len();
        let results: Vec<Result<bool, EvalError>> = stream::iter(claims)
            .map(|claim| async move { self.verify_claim(ctx, &claim).await })
            .buffer_unordered(MAX_CONCURRENT_VERIFY)
            .collect()
            .await;
        let mut supported = 0usize;
        for r in results {
            if r? {
                supported += 1;
            }
        }
        let value = supported as f64 / total as f64;
        Ok(Score::new(value).with_label("faithfulness"))
    }

    fn name(&self) -> &str {
        "faithfulness"
    }
}

/// 结构化判定参数(经 tool_calls 返回)。
#[derive(Debug, Deserialize)]
struct VerdictArgs {
    verdict: bool,
    /// 让 LLM 附上简短理由(改善判定质量),当前不消费。
    #[serde(default)]
    #[allow(dead_code)]
    reason: String,
}

/// 构建判定工具:让 LLM 以 `{"verdict": bool, "reason": "..."}` 提交判定。
fn verdict_tool() -> ToolDefinition {
    ToolDefinition::new(
        "check_claim",
        "判断陈述能否从参考上下文推导出来,提交布尔判定。",
    )
    .with_parameters(serde_json::json!({
        "type": "object",
        "properties": {
            "verdict": { "type": "boolean", "description": "能否从上下文推导" },
            "reason": { "type": "string", "description": "简短依据" }
        },
        "required": ["verdict", "reason"]
    }))
}

/// 解析"是/否"。无任何是/否标记时返回 `None`(解析失败,由调用方报错),
/// 而非静默默认 false——避免 LLM 跑题回复被当成"不忠实"。
fn parse_yes_no(raw: &str) -> Option<bool> {
    let lower = raw.to_lowercase();
    // 先判否定(避免"不是""不能"被"是""能"误判;否定词优先于肯定词)
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
    use lc_core::language_models::{LLMResult, StreamChunk};
    use lc_core::{BaseLanguageModel, Runnable, RunnableConfig};
    use lc_schema::MessageType;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Debug)]
    struct JudgeError(String);
    impl std::fmt::Display for JudgeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::error::Error for JudgeError {}

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
        type Error = JudgeError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(JudgeError("use chat".into()))
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
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            Err(JudgeError("not supported".into()))
        }
    }

    #[test]
    fn test_split_claims() {
        let claims = split_claims("巴黎是法国首都。伦敦是英国首都。");
        assert_eq!(claims.len(), 2);
        assert_eq!(claims[0], "巴黎是法国首都");
        assert_eq!(claims[1], "伦敦是英国首都");
    }

    #[test]
    fn test_split_claims_empty() {
        assert!(split_claims("").is_empty());
        assert!(split_claims("。。。").is_empty());
    }

    #[tokio::test]
    async fn test_faithfulness_all_supported() {
        let judge = Faithfulness::new(SeqMockJudge::new(vec!["是".into(), "是".into()]));
        let s = judge
            .eval("", "巴黎是法国首都。伦敦是英国首都。", "ctx")
            .await
            .unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_faithfulness_half_supported() {
        let judge = Faithfulness::new(SeqMockJudge::new(vec!["是".into(), "否".into()]));
        let s = judge
            .eval("", "巴黎是法国首都。伦敦是英国首都。", "ctx")
            .await
            .unwrap();
        assert!((s.value - 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_faithfulness_none_supported() {
        let judge = Faithfulness::new(SeqMockJudge::new(vec!["否".into(), "否".into()]));
        let s = judge
            .eval("", "巴黎是法国首都。伦敦是英国首都。", "ctx")
            .await
            .unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_faithfulness_empty_prediction() {
        // P0-2: 空预测默认 0 分(没回答=不忠实)
        let judge = Faithfulness::new(SeqMockJudge::new(vec![]));
        let s = judge.eval("", "", "ctx").await.unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
        assert_eq!(s.label.as_deref(), Some("no_claims"));
    }

    #[tokio::test]
    async fn test_faithfulness_empty_score_configurable() {
        // 可显式配成 1.0(没编造即忠实)
        let judge = Faithfulness::new(SeqMockJudge::new(vec![])).with_empty_score(1.0);
        let s = judge.eval("", "", "ctx").await.unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_faithfulness_llm_split() {
        // 逗号复合句规则拆只算 1 条;LLM 拆能拆成 2 条分别验证
        let judge = Faithfulness::new(SeqMockJudge::new(vec![
            "巴黎是法国首都\n伦敦是英国首都".into(),
            "是".into(),
            "是".into(),
        ]))
        .with_llm_split(true);
        let s = judge
            .eval("", "巴黎是法国首都,伦敦是英国首都。", "ctx")
            .await
            .unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_yes_no() {
        assert_eq!(parse_yes_no("是"), Some(true));
        assert_eq!(parse_yes_no("yes"), Some(true));
        assert_eq!(parse_yes_no("否"), Some(false));
        assert_eq!(parse_yes_no("no"), Some(false));
        assert_eq!(parse_yes_no("不是"), Some(false));
        assert_eq!(parse_yes_no("不能"), Some(false));
        // 无任何是/否标记 = 解析失败,不应静默默认
        assert_eq!(parse_yes_no("我不会告诉你"), None);
    }

    /// P0-1: 支持 bind_tools 的模型走结构化输出(verdict 布尔),不再依赖文本解析。
    #[tokio::test]
    async fn test_faithfulness_structured_verdict() {
        use crate::test_support::ToolJudge;
        // 两条陈述:一条支持、一条不支持 → 忠实度 0.5
        let judge = Faithfulness::new(ToolJudge::sequence(vec![
            r#"{"verdict": true, "reason": "能从上下文推导"}"#.into(),
            r#"{"verdict": false, "reason": "无法推导"}"#.into(),
        ]));
        let s = judge
            .eval("", "巴黎是法国首都。伦敦是英国首都。", "巴黎是法国首都")
            .await
            .unwrap();
        assert!((s.value - 0.5).abs() < 1e-9);
    }

    /// P0-1: 全部不支持 → 0 分。
    #[tokio::test]
    async fn test_faithfulness_structured_all_false() {
        use crate::test_support::ToolJudge;
        let judge = Faithfulness::new(ToolJudge::new(
            r#"{"verdict": false, "reason": "均无法推导"}"#,
        ));
        let s = judge
            .eval("", "巴黎是法国首都。伦敦是英国首都。", "巴黎是法国首都")
            .await
            .unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
    }

    /// P2-5: 长参考上下文只截取一次,N 条陈述复用同一份,不重复整段发送。
    #[tokio::test]
    async fn test_faithfulness_reference_truncated_once() {
        let judge = SeqMockJudge::new(vec!["是".into(), "是".into()]);
        let f = Faithfulness::new(judge).with_max_context_chars(10);
        let long_ref =
            "这是一段非常长的参考上下文,远超默认的单条传输上限,里面藏了一个不该被完整发送的尾巴"
                .to_string();
        let s = f
            .eval("", "巴黎是首都。伦敦是首都。", &long_ref)
            .await
            .unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
        let sent = f.judge.last_user_content();
        // 参考上下文被截到预算内:首部仍在、远在预算外的尾巴不会被发出去
        assert!(sent.contains("这是一段非常长"), "actual sent: {sent}");
        assert!(
            !sent.contains("不该被完整发送"),
            "full long reference was sent repeatedly"
        );
    }
}
