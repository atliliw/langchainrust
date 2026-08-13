//! MCP Sampling - 采样类型与 `sampling/createMessage` 处理
//!
//! MCP Sampling 允许 Server 请求 Host(即 LLM Client)执行 LLM 推理,
//! Server 可借此利用 Host 的模型能力完成子任务。

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// 采样消息内容(内联枚举)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SamplingContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
}

/// 采样消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingMessage {
    pub role: SamplingRole,
    pub content: SamplingContent,
}

/// 消息角色
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SamplingRole {
    User,
    Assistant,
}

/// 模型偏好提示
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPreferences {
    #[serde(rename = "costPriority", skip_serializing_if = "Option::is_none")]
    pub cost_priority: Option<f64>,
    #[serde(rename = "speedPriority", skip_serializing_if = "Option::is_none")]
    pub speed_priority: Option<f64>,
    #[serde(
        rename = "intelligencePriority",
        skip_serializing_if = "Option::is_none"
    )]
    pub intelligence_priority: Option<f64>,
    #[serde(rename = "hints", skip_serializing_if = "Option::is_none")]
    pub hints: Option<Vec<ModelHint>>,
}

/// 模型提示
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHint {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// `sampling/createMessage` 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingRequest {
    pub messages: Vec<SamplingMessage>,
    #[serde(rename = "maxTokens")]
    pub max_tokens: usize,
    #[serde(rename = "systemPrompt", skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(rename = "modelPreferences", skip_serializing_if = "Option::is_none")]
    pub model_preferences: Option<ModelPreferences>,
    #[serde(rename = "temperature", skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(rename = "includeContext", skip_serializing_if = "Option::is_none")]
    pub include_context: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// `sampling/createMessage` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingResult {
    pub role: SamplingRole,
    pub content: SamplingContent,
    #[serde(rename = "model", skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(rename = "stopReason", skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

/// Sampling 递归防护错误(P2-7)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingGuardError {
    /// 嵌套深度超过 `max_depth`(默认 3)。
    TooDeep { depth: usize, max_depth: usize },
    /// 整条采样链累计 token 超过总预算。
    TokenBudgetExceeded {
        tokens_used: usize,
        total_budget: usize,
    },
    /// 整条采样链超过总时长(超时)。
    Timeout,
}

impl std::fmt::Display for SamplingGuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplingGuardError::TooDeep { depth, max_depth } => {
                write!(f, "Sampling 递归深度 {depth} 超过上限 {max_depth}")
            }
            SamplingGuardError::TokenBudgetExceeded {
                tokens_used,
                total_budget,
            } => write!(
                f,
                "Sampling 累计 token {tokens_used} 超过总预算 {total_budget}"
            ),
            SamplingGuardError::Timeout => write!(f, "Sampling 超过总时长上限(超时)"),
        }
    }
}

impl std::error::Error for SamplingGuardError {}

/// Sampling 递归防护(P2-7)。
///
/// MCP 场景存在"Agent 调工具 → 工具请求 Sampling → LLM 调工具 → 工具请求
/// Sampling"的递归环,`SamplingRequest` 本身无防护,递归可以无限加深。本结构在
/// Host 侧给整条采样链加三重约束:
///
/// - **深度限制**:嵌套 Sampling 不得超过 `max_depth`(默认 3)层;
/// - **总 token 预算**:整条链按每次请求的 `max_tokens` 累计,超预算即拒绝;
/// - **超时**:整条链总时长不得超过 `total_timeout`(或显式 `deadline`)。
///
/// 用法:每次执行 `sampling/createMessage` 前 `enter(request.max_tokens)`,拿到的
/// [`SamplingLease`] 在采样调用期间持有(跨 `await` 安全),结束自动释放深度。
///
/// ```no_run
/// use lc_mcp::{SamplingGuard, SamplingRequest};
/// use std::time::Duration;
///
/// # async fn handle(req: SamplingRequest, guard: SamplingGuard) -> Result<(), ()> {
/// let _lease = guard
///     .enter(req.max_tokens)
///     .map_err(|_| ())?;            // 深度 / 预算 / 超时任一超限即拒绝
/// // ... 执行 LLM 推理 ...
/// # Ok(())
/// # }
/// ```
pub struct SamplingGuard {
    max_depth: usize,
    total_token_budget: usize,
    deadline: Option<Instant>,
    /// 当前嵌套深度(原子,跨 await 安全)。
    depth: AtomicUsize,
    /// 已累计 token。
    tokens_used: AtomicUsize,
}

impl SamplingGuard {
    /// 创建采样递归防护。
    ///
    /// 默认无超时;`total_token_budget` 为整条链累计上限。
    pub fn new(max_depth: usize, total_token_budget: usize) -> Self {
        Self {
            max_depth: max_depth.max(1),
            total_token_budget,
            deadline: None,
            depth: AtomicUsize::new(0),
            tokens_used: AtomicUsize::new(0),
        }
    }

    /// 整条采样链总时长上限(从创建时刻起算)。
    pub fn with_timeout(mut self, total_timeout: Duration) -> Self {
        self.deadline = Some(Instant::now() + total_timeout);
        self
    }

    /// 显式设置总时长截止时刻(更精确的绝对时间)。
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// 进入一次采样:校验超时 / 深度 / token 预算。
    ///
    /// 成功返回 [`SamplingLease`],在采样调用期间持有、结束 Drop 自动释放深度;
    /// 任一约束超限返回对应错误,不占用深度与预算。
    pub fn enter(&self, request_tokens: usize) -> Result<SamplingLease<'_>, SamplingGuardError> {
        // 超时:整条链总时长。
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                return Err(SamplingGuardError::Timeout);
            }
        }
        // 深度:嵌套层数 +1,超限回滚。
        let depth = self.depth.fetch_add(1, Ordering::SeqCst) + 1;
        if depth > self.max_depth {
            self.depth.fetch_sub(1, Ordering::SeqCst);
            return Err(SamplingGuardError::TooDeep {
                depth,
                max_depth: self.max_depth,
            });
        }
        // token 预算:累计 +1 次请求的 max_tokens,超限回滚(深度一并回滚)。
        let used = self.tokens_used.fetch_add(request_tokens, Ordering::SeqCst) + request_tokens;
        if used > self.total_token_budget {
            self.tokens_used.fetch_sub(request_tokens, Ordering::SeqCst);
            self.depth.fetch_sub(1, Ordering::SeqCst);
            return Err(SamplingGuardError::TokenBudgetExceeded {
                tokens_used: used,
                total_budget: self.total_token_budget,
            });
        }
        Ok(SamplingLease { guard: self })
    }

    /// 当前嵌套深度。
    pub fn depth(&self) -> usize {
        self.depth.load(Ordering::SeqCst)
    }

    /// 已累计的 token 预算。
    pub fn tokens_used(&self) -> usize {
        self.tokens_used.load(Ordering::SeqCst)
    }
}

impl std::fmt::Debug for SamplingGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplingGuard")
            .field("max_depth", &self.max_depth)
            .field("total_token_budget", &self.total_token_budget)
            .field("depth", &self.depth())
            .field("tokens_used", &self.tokens_used())
            .finish()
    }
}

/// 一次采样占用的防护令牌(P2-7)。
///
/// 持有期间占用一层嵌套深度;Drop 时自动释放,使兄弟/后续采样可继续进入。
#[derive(Debug)]
pub struct SamplingLease<'a> {
    guard: &'a SamplingGuard,
}

impl Drop for SamplingLease<'_> {
    fn drop(&mut self) {
        self.guard.depth.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_content_text() {
        let content = SamplingContent::Text {
            text: "Hello".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"Hello\""));
    }

    #[test]
    fn test_sampling_content_image() {
        let content = SamplingContent::Image {
            data: "base64data".to_string(),
            mime_type: "image/png".to_string(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"image\""));
    }

    #[test]
    fn test_sampling_message_user() {
        let msg = SamplingMessage {
            role: SamplingRole::User,
            content: SamplingContent::Text {
                text: "What is Rust?".to_string(),
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
    }

    #[test]
    fn test_sampling_message_assistant() {
        let msg = SamplingMessage {
            role: SamplingRole::Assistant,
            content: SamplingContent::Text {
                text: "Rust is a systems language".to_string(),
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"assistant\""));
    }

    #[test]
    fn test_sampling_request_minimal() {
        let req = SamplingRequest {
            messages: vec![SamplingMessage {
                role: SamplingRole::User,
                content: SamplingContent::Text {
                    text: "Hello".to_string(),
                },
            }],
            max_tokens: 100,
            system_prompt: None,
            model_preferences: None,
            temperature: None,
            stop_sequences: None,
            include_context: None,
            metadata: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"maxTokens\":100"));
        assert!(!json.contains("systemPrompt"));
        assert!(!json.contains("temperature"));
    }

    #[test]
    fn test_sampling_request_full() {
        let req = SamplingRequest {
            messages: vec![SamplingMessage {
                role: SamplingRole::User,
                content: SamplingContent::Text {
                    text: "Hello".to_string(),
                },
            }],
            max_tokens: 100,
            system_prompt: Some("You are helpful".to_string()),
            model_preferences: Some(ModelPreferences {
                cost_priority: Some(0.5),
                speed_priority: None,
                intelligence_priority: None,
                hints: Some(vec![ModelHint {
                    name: Some("claude-3".to_string()),
                }]),
            }),
            temperature: Some(0.7),
            stop_sequences: Some(vec!["\n".to_string()]),
            include_context: None,
            metadata: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"systemPrompt\":\"You are helpful\""));
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"costPriority\":0.5"));
    }

    #[test]
    fn test_sampling_result_deserialization() {
        let json = r#"{"role":"assistant","content":{"type":"text","text":"Hi there"},"model":"claude-3","stopReason":"endTurn"}"#;
        let result: SamplingResult = serde_json::from_str(json).unwrap();
        assert!(matches!(result.role, SamplingRole::Assistant));
        assert_eq!(result.model.as_deref(), Some("claude-3"));
        assert_eq!(result.stop_reason.as_deref(), Some("endTurn"));
    }

    #[test]
    fn test_model_preferences_serialization() {
        let prefs = ModelPreferences {
            cost_priority: Some(0.3),
            speed_priority: Some(0.7),
            intelligence_priority: None,
            hints: None,
        };
        let json = serde_json::to_string(&prefs).unwrap();
        assert!(json.contains("\"costPriority\":0.3"));
        assert!(json.contains("\"speedPriority\":0.7"));
        assert!(!json.contains("intelligencePriority"));
    }

    /// 深度限制:超过 max_depth 层的嵌套采样被拒绝,已占用的层数不变。
    #[test]
    fn test_guard_limits_depth() {
        let guard = SamplingGuard::new(3, 1000);
        let _l1 = guard.enter(10).expect("第一层进入成功");
        let _l2 = guard.enter(10).expect("第二层进入成功");
        let _l3 = guard.enter(10).expect("第三层进入成功");
        assert_eq!(guard.depth(), 3);
        let err = guard.enter(10).unwrap_err();
        assert_eq!(
            err,
            SamplingGuardError::TooDeep {
                depth: 4,
                max_depth: 3
            },
            "第 4 层应拒绝"
        );
        assert_eq!(guard.depth(), 3, "被拒的进入不应占用深度");
    }

    /// Lease Drop 释放深度:并行/顺序的兄弟采样仍可进入。
    #[test]
    fn test_lease_drop_releases_depth() {
        let guard = SamplingGuard::new(1, 1000);
        let lease = guard.enter(10).expect("第一层进入成功");
        assert_eq!(guard.depth(), 1);
        drop(lease);
        assert_eq!(guard.depth(), 0, "Drop 后深度应释放");
        guard.enter(10).expect("释放后可再次进入");
    }

    /// 总 token 预算:整条链按每次请求的 max_tokens 累计,超预算拒绝。
    #[test]
    fn test_token_budget_accumulates() {
        let guard = SamplingGuard::new(5, 30);
        guard.enter(20).expect("20 token 在预算内");
        assert_eq!(guard.tokens_used(), 20);
        guard.enter(10).expect("累计 30,恰在预算内");
        assert_eq!(guard.tokens_used(), 30);
        let err = guard.enter(1).unwrap_err();
        assert_eq!(
            err,
            SamplingGuardError::TokenBudgetExceeded {
                tokens_used: 31,
                total_budget: 30
            }
        );
        assert_eq!(guard.tokens_used(), 30, "被拒的进入不应占用预算");
    }

    /// 超时:整条链超过总时长后拒绝新采样。
    #[test]
    fn test_timeout_rejects_after_deadline() {
        let guard =
            SamplingGuard::new(3, 1000).with_deadline(Instant::now() - Duration::from_secs(1));
        assert_eq!(guard.enter(10).unwrap_err(), SamplingGuardError::Timeout);
        assert_eq!(guard.depth(), 0);
    }

    /// 错误 Display:三类错误信息可读。
    #[test]
    fn test_error_display() {
        assert!(SamplingGuardError::TooDeep {
            depth: 4,
            max_depth: 3
        }
        .to_string()
        .contains("深度"));
        assert!(SamplingGuardError::TokenBudgetExceeded {
            tokens_used: 40,
            total_budget: 30
        }
        .to_string()
        .contains("预算"));
        assert!(SamplingGuardError::Timeout.to_string().contains("超时"));
    }

    /// 充足预算下,释放后可反复进入(深度不泄漏)。
    #[test]
    fn test_reenter_after_completion() {
        let guard = SamplingGuard::new(3, 1000);
        for _ in 0..5 {
            let lease = guard.enter(10).expect("释放后应可再次进入");
            assert_eq!(guard.depth(), 1, "持有 lease 期间占用一层");
            drop(lease);
        }
        assert_eq!(guard.depth(), 0, "所有 lease 均已释放");
    }
}
