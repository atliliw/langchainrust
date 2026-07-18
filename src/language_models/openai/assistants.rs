//! OpenAI Assistants API 封装
//!
//! 封装 OpenAI 官方 Assistants / Threads / Run,支持服务端会话状态。
//! 注:需使用支持 Assistants API 的端点(OpenAI 官方);部分 compatible-mode 端点可能不支持。

use crate::OpenAIConfig;
use serde_json::Value;
use std::time::Duration;

/// Assistants 错误
#[derive(Debug)]
pub enum AssistantError {
    Http(String),
    Api(String),
    Parse(String),
    /// Run 处于 requires_action(工具调用),当前精简版未实现工具调度
    RequiresAction,
    /// Run 终止于非完成状态
    RunFailed(String),
    Timeout,
}

impl std::fmt::Display for AssistantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssistantError::Http(m) => write!(f, "HTTP 错误: {}", m),
            AssistantError::Api(m) => write!(f, "API 错误: {}", m),
            AssistantError::Parse(m) => write!(f, "解析错误: {}", m),
            AssistantError::RequiresAction => write!(f, "Run 需要工具调用(当前未实现)"),
            AssistantError::RunFailed(s) => write!(f, "Run 失败, 状态: {}", s),
            AssistantError::Timeout => write!(f, "轮询超时"),
        }
    }
}

impl std::error::Error for AssistantError {}

/// OpenAI Assistant 封装
pub struct OpenAIAssistant {
    client: reqwest::Client,
    config: OpenAIConfig,
    assistant_id: String,
}

/// 判断 Run 状态是否为终态
pub fn is_terminal_status(status: &str) -> bool {
    matches!(
        status,
        "completed" | "failed" | "cancelled" | "expired" | "incomplete"
    )
}

impl OpenAIAssistant {
    /// 创建 Assistant
    pub async fn create(
        config: OpenAIConfig,
        model: &str,
        instructions: &str,
    ) -> Result<Self, AssistantError> {
        let client = reqwest::Client::new();
        let url = format!("{}/assistants", config.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": model,
            "instructions": instructions,
        });
        let resp = Self::post(&client, &config, &url, body).await?;
        let id = resp
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AssistantError::Parse("缺少 assistant id".into()))?
            .to_string();
        Ok(Self {
            client,
            config,
            assistant_id: id,
        })
    }

    /// 用已有 assistant id 构造(跳过 create)
    pub fn from_id(config: OpenAIConfig, assistant_id: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            assistant_id: assistant_id.into(),
        }
    }

    pub fn assistant_id(&self) -> &str {
        &self.assistant_id
    }

    /// 跑一轮对话:创建 thread + 加消息 + 创建 run + 轮询 + 取最终消息
    pub async fn run_once(&self, user_msg: &str) -> Result<String, AssistantError> {
        let base = self.config.base_url.trim_end_matches('/');

        // 1. 创建 thread
        let thread = Self::post(
            &self.client,
            &self.config,
            &format!("{}/threads", base),
            serde_json::json!({}),
        )
        .await?;
        let thread_id = thread
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AssistantError::Parse("缺少 thread id".into()))?;

        // 2. 加用户消息
        Self::post(
            &self.client,
            &self.config,
            &format!("{}/threads/{}/messages", base, thread_id),
            serde_json::json!({ "role": "user", "content": user_msg }),
        )
        .await?;

        // 3. 创建 run
        let run = Self::post(
            &self.client,
            &self.config,
            &format!("{}/threads/{}/runs", base, thread_id),
            serde_json::json!({ "assistant_id": self.assistant_id }),
        )
        .await?;
        let run_id = run
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AssistantError::Parse("缺少 run id".into()))?;

        // 4. 轮询(最多 ~60s)
        let mut attempts = 0u32;
        loop {
            if attempts > 120 {
                return Err(AssistantError::Timeout);
            }
            attempts += 1;
            let run_state = Self::get(
                &self.client,
                &self.config,
                &format!("{}/threads/{}/runs/{}", base, thread_id, run_id),
            )
            .await?;
            let status = run_state
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            match status {
                "completed" => break,
                "requires_action" => return Err(AssistantError::RequiresAction),
                s if is_terminal_status(s) => {
                    return Err(AssistantError::RunFailed(s.to_string()))
                }
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }

        // 5. 取最终 assistant 消息(data[0] 为最新)
        let messages = Self::get(
            &self.client,
            &self.config,
            &format!("{}/threads/{}/messages", base, thread_id),
        )
        .await?;
        let data = messages
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| AssistantError::Parse("缺少 messages data".into()))?;
        let first = data
            .first()
            .ok_or_else(|| AssistantError::Parse("无消息".into()))?;
        let text = first
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("text"))
            .and_then(|t| t.get("value"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| AssistantError::Parse("无法解析消息内容".into()))?;
        Ok(text.to_string())
    }

    async fn post(
        client: &reqwest::Client,
        config: &OpenAIConfig,
        url: &str,
        body: Value,
    ) -> Result<Value, AssistantError> {
        let resp = client
            .post(url)
            .header("Authorization", format!("Bearer {}", config.api_key))
            .header("OpenAI-Beta", "assistants=v2")
            .json(&body)
            .send()
            .await
            .map_err(|e| AssistantError::Http(e.to_string()))?;
        Self::parse(resp).await
    }

    async fn get(
        client: &reqwest::Client,
        config: &OpenAIConfig,
        url: &str,
    ) -> Result<Value, AssistantError> {
        let resp = client
            .get(url)
            .header("Authorization", format!("Bearer {}", config.api_key))
            .header("OpenAI-Beta", "assistants=v2")
            .send()
            .await
            .map_err(|e| AssistantError::Http(e.to_string()))?;
        Self::parse(resp).await
    }

    async fn parse(resp: reqwest::Response) -> Result<Value, AssistantError> {
        let status = resp.status();
        let json: Value = resp
            .json()
            .await
            .map_err(|e| AssistantError::Parse(e.to_string()))?;
        if !status.is_success() {
            let msg = json
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("未知错误");
            return Err(AssistantError::Api(msg.to_string()));
        }
        Ok(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_terminal_status() {
        assert!(is_terminal_status("completed"));
        assert!(is_terminal_status("failed"));
        assert!(is_terminal_status("cancelled"));
        assert!(is_terminal_status("expired"));
        assert!(!is_terminal_status("queued"));
        assert!(!is_terminal_status("in_progress"));
    }

    #[test]
    fn test_from_id() {
        let config = OpenAIConfig::new("sk-test");
        let a = OpenAIAssistant::from_id(config, "asst_123");
        assert_eq!(a.assistant_id(), "asst_123");
    }

    #[tokio::test]
    #[ignore = "需 OpenAI Assistants API key 与支持端点"]
    async fn test_create_and_run() {
        let config = OpenAIConfig::from_env();
        let assistant = OpenAIAssistant::create(config, "gpt-4o", "你是一个助手")
            .await
            .unwrap();
        let answer = assistant.run_once("你好").await.unwrap();
        assert!(!answer.is_empty());
    }
}
