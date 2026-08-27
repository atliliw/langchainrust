//! MCP Sampling - sampling types and `sampling/createMessage` handling
//!
//! MCP Sampling lets a Server ask the Host (i.e. the LLM Client) to run LLM inference,
//! letting the Server use the Host's model capabilities to complete sub-tasks.

use crate::protocol::MCPError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Sampling message content (inline enum)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SamplingContent {
    /// Text content
    #[serde(rename = "text")]
    Text {
        /// Text data
        text: String,
    },
    /// Image content
    #[serde(rename = "image")]
    Image {
        /// Image data (base64-encoded)
        data: String,
        /// Image MIME type
        mime_type: String,
    },
}

/// Sampling message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingMessage {
    /// Message role
    pub role: SamplingRole,
    /// Message content
    pub content: SamplingContent,
}

/// Message role
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SamplingRole {
    /// User role
    User,
    /// Assistant role
    Assistant,
}

/// Model preference hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPreferences {
    /// Cost priority (0~1)
    #[serde(rename = "costPriority", skip_serializing_if = "Option::is_none")]
    pub cost_priority: Option<f64>,
    /// Speed priority (0~1)
    #[serde(rename = "speedPriority", skip_serializing_if = "Option::is_none")]
    pub speed_priority: Option<f64>,
    /// Intelligence priority (0~1)
    #[serde(
        rename = "intelligencePriority",
        skip_serializing_if = "Option::is_none"
    )]
    pub intelligence_priority: Option<f64>,
    /// Model hint list (optional)
    #[serde(rename = "hints", skip_serializing_if = "Option::is_none")]
    pub hints: Option<Vec<ModelHint>>,
}

/// Model hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHint {
    /// Suggested model name (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// `sampling/createMessage` request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingRequest {
    /// List of sampling messages
    pub messages: Vec<SamplingMessage>,
    /// Maximum number of tokens to generate
    #[serde(rename = "maxTokens")]
    pub max_tokens: usize,
    /// Optional system prompt
    #[serde(rename = "systemPrompt", skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Optional model preferences
    #[serde(rename = "modelPreferences", skip_serializing_if = "Option::is_none")]
    pub model_preferences: Option<ModelPreferences>,
    /// Optional sampling temperature
    #[serde(rename = "temperature", skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Optional stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Optional context inclusion policy (see the MCP spec)
    #[serde(rename = "includeContext", skip_serializing_if = "Option::is_none")]
    pub include_context: Option<Value>,
    /// Optional extra metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// `sampling/createMessage` response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingResult {
    /// Role of the generated message
    pub role: SamplingRole,
    /// Content of the generated message
    pub content: SamplingContent,
    /// Name of the model used (optional)
    #[serde(rename = "model", skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Stop reason (optional)
    #[serde(rename = "stopReason", skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

/// Sampling handler (server→host direction).
///
/// Per MCP semantics, `sampling/createMessage` is initiated by the Server and the Host runs the LLM inference.
/// The framework layer does not connect to any concrete transport; the host injects this callback, which is
/// responsible for delivering the request to the Host and retrieving the response. Without an injected handler,
/// [`crate::MCPServer::create_message`] returns a clear error.
#[async_trait]
pub trait SamplingHandler: Send + Sync {
    /// Runs one sampling call, returning the Host's inference result.
    async fn create_message(&self, request: &SamplingRequest) -> Result<SamplingResult, MCPError>;
}

/// Sampling recursion guard error (P2-7).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SamplingGuardError {
    /// Nesting depth exceeds `max_depth` (default 3).
    TooDeep {
        /// Current nesting depth
        depth: usize,
        /// Maximum allowed nesting depth
        max_depth: usize,
    },
    /// Cumulative tokens across the whole sampling chain exceed the total budget.
    TokenBudgetExceeded {
        /// Tokens used so far
        tokens_used: usize,
        /// Total token budget
        total_budget: usize,
    },
    /// The whole sampling chain exceeded the total duration (timeout).
    Timeout,
}

impl std::fmt::Display for SamplingGuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplingGuardError::TooDeep { depth, max_depth } => {
                write!(
                    f,
                    "Sampling recursion depth {depth} exceeds limit {max_depth}"
                )
            }
            SamplingGuardError::TokenBudgetExceeded {
                tokens_used,
                total_budget,
            } => write!(
                f,
                "Sampling cumulative tokens {tokens_used} exceed total budget {total_budget}"
            ),
            SamplingGuardError::Timeout => {
                write!(f, "Sampling exceeds the total duration limit (timeout)")
            }
        }
    }
}

impl std::error::Error for SamplingGuardError {}

/// Sampling recursion guard (P2-7).
///
/// MCP scenarios have a recursion loop of "Agent calls tool → tool requests Sampling → LLM calls tool → tool
/// requests Sampling"; `SamplingRequest` itself has no protection, so recursion can deepen without bound. This
/// structure adds three constraints on the Host side over the whole sampling chain:
///
/// - **Depth limit**: nested Sampling must not exceed `max_depth` (default 3) levels;
/// - **Total token budget**: the chain accumulates each request's `max_tokens`; over-budget is rejected;
/// - **Timeout**: the chain's total duration must not exceed `total_timeout` (or an explicit `deadline`).
///
/// Usage: call `enter(request.max_tokens)` before each `sampling/createMessage`; the returned [`SamplingLease`]
/// is held during the sampling call (safe across `await`) and releases the depth automatically when it ends.
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
    /// Current nesting depth (atomic, safe across `await`).
    depth: AtomicUsize,
    /// Accumulated tokens.
    tokens_used: AtomicUsize,
}

impl SamplingGuard {
    /// Creates a sampling recursion guard.
    ///
    /// No timeout by default; `total_token_budget` is the chain-wide cumulative cap.
    pub fn new(max_depth: usize, total_token_budget: usize) -> Self {
        Self {
            max_depth: max_depth.max(1),
            total_token_budget,
            deadline: None,
            depth: AtomicUsize::new(0),
            tokens_used: AtomicUsize::new(0),
        }
    }

    /// Cap on the whole sampling chain's total duration (counted from creation time).
    pub fn with_timeout(mut self, total_timeout: Duration) -> Self {
        self.deadline = Some(Instant::now() + total_timeout);
        self
    }

    /// Explicitly sets the total-duration deadline (a more precise absolute time).
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Enters one sampling call: checks timeout / depth / token budget.
    ///
    /// On success returns a [`SamplingLease`] held during the sampling call; the depth is released automatically
    /// on Drop; any constraint exceeded returns the matching error without consuming depth or budget.
    pub fn enter(&self, request_tokens: usize) -> Result<SamplingLease<'_>, SamplingGuardError> {
        // Timeout: the chain's total duration.
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                return Err(SamplingGuardError::Timeout);
            }
        }
        // Depth: nesting level +1, rolled back when over the limit.
        let depth = self.depth.fetch_add(1, Ordering::SeqCst) + 1;
        if depth > self.max_depth {
            self.depth.fetch_sub(1, Ordering::SeqCst);
            return Err(SamplingGuardError::TooDeep {
                depth,
                max_depth: self.max_depth,
            });
        }
        // Token budget: accumulate the request's max_tokens; over the limit, roll back (depth rolls back too).
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

    /// Current nesting depth.
    pub fn depth(&self) -> usize {
        self.depth.load(Ordering::SeqCst)
    }

    /// Accumulated token budget.
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

/// A guard token held by one sampling call (P2-7).
///
/// Occupies one nesting level while held; released automatically on Drop, letting sibling/subsequent samplings continue.
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

    /// Depth limit: nested sampling beyond max_depth layers is rejected, and the occupied depth is unchanged.
    #[test]
    fn test_guard_limits_depth() {
        let guard = SamplingGuard::new(3, 1000);
        let _l1 = guard.enter(10).expect("first level enter should succeed");
        let _l2 = guard.enter(10).expect("second level enter should succeed");
        let _l3 = guard.enter(10).expect("third level enter should succeed");
        assert_eq!(guard.depth(), 3);
        let err = guard.enter(10).unwrap_err();
        assert_eq!(
            err,
            SamplingGuardError::TooDeep {
                depth: 4,
                max_depth: 3
            },
            "4th level should be rejected"
        );
        assert_eq!(
            guard.depth(),
            3,
            "a rejected enter should not consume depth"
        );
    }

    /// Lease Drop releases depth: parallel/sequential sibling samplings can still enter.
    #[test]
    fn test_lease_drop_releases_depth() {
        let guard = SamplingGuard::new(1, 1000);
        let lease = guard.enter(10).expect("first level enter should succeed");
        assert_eq!(guard.depth(), 1);
        drop(lease);
        assert_eq!(guard.depth(), 0, "depth should be released after Drop");
        guard
            .enter(10)
            .expect("should be able to enter again after release");
    }

    /// Total token budget: the chain accumulates each request's max_tokens; over-budget is rejected.
    #[test]
    fn test_token_budget_accumulates() {
        let guard = SamplingGuard::new(5, 30);
        guard.enter(20).expect("20 tokens within budget");
        assert_eq!(guard.tokens_used(), 20);
        guard.enter(10).expect("cumulative 30, exactly at budget");
        assert_eq!(guard.tokens_used(), 30);
        let err = guard.enter(1).unwrap_err();
        assert_eq!(
            err,
            SamplingGuardError::TokenBudgetExceeded {
                tokens_used: 31,
                total_budget: 30
            }
        );
        assert_eq!(
            guard.tokens_used(),
            30,
            "a rejected enter should not consume budget"
        );
    }

    /// Timeout: new samplings are rejected after the chain exceeds the total duration.
    #[test]
    fn test_timeout_rejects_after_deadline() {
        let guard =
            SamplingGuard::new(3, 1000).with_deadline(Instant::now() - Duration::from_secs(1));
        assert_eq!(guard.enter(10).unwrap_err(), SamplingGuardError::Timeout);
        assert_eq!(guard.depth(), 0);
    }

    /// Error Display: all three error kinds read clearly.
    #[test]
    fn test_error_display() {
        assert!(SamplingGuardError::TooDeep {
            depth: 4,
            max_depth: 3
        }
        .to_string()
        .contains("depth"));
        assert!(SamplingGuardError::TokenBudgetExceeded {
            tokens_used: 40,
            total_budget: 30
        }
        .to_string()
        .contains("budget"));
        assert!(SamplingGuardError::Timeout.to_string().contains("timeout"));
    }

    /// With enough budget, one can keep re-entering after release (no depth leak).
    #[test]
    fn test_reenter_after_completion() {
        let guard = SamplingGuard::new(3, 1000);
        for _ in 0..5 {
            let lease = guard
                .enter(10)
                .expect("should be able to enter again after release");
            assert_eq!(guard.depth(), 1, "holding a lease consumes one level");
            drop(lease);
        }
        assert_eq!(guard.depth(), 0, "all leases are released");
    }
}
