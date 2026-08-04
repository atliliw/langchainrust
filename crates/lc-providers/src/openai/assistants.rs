//! OpenAI Assistants API 封装
//!
//! 封装 OpenAI 官方 Assistants / Threads / Run,支持服务端会话状态。
//! v0.4.1: 支持 `requires_action` 工具调度 -- 当 run 需要工具调用时,
//! 解析 tool_calls,经 ToolRegistry 执行,submit_tool_outputs 回传,继续轮询至完成。
//! 注:需使用支持 Assistants API 的端点(OpenAI 官方);部分 compatible-mode 端点可能不支持。

use super::config::OpenAIConfig;
use lc_core::tools::ToolRegistry;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// Assistants 错误
#[derive(Debug)]
pub enum AssistantError {
    /// HTTP 请求错误
    Http(String),
    /// API 返回错误
    Api(String),
    /// 响应解析错误
    Parse(String),
    /// Run 终止于非完成状态
    RunFailed(String),
    /// 工具执行错误
    ToolExecution { tool_name: String, error: String },
    /// 轮询超时
    Timeout,
}

impl std::fmt::Display for AssistantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssistantError::Http(m) => write!(f, "HTTP 错误: {}", m),
            AssistantError::Api(m) => write!(f, "API 错误: {}", m),
            AssistantError::Parse(m) => write!(f, "解析错误: {}", m),
            AssistantError::RunFailed(s) => write!(f, "Run 失败, 状态: {}", s),
            AssistantError::ToolExecution { tool_name, error } => {
                write!(f, "工具 '{}' 执行失败: {}", tool_name, error)
            }
            AssistantError::Timeout => write!(f, "轮询超时"),
        }
    }
}

impl std::error::Error for AssistantError {}

/// 轮询配置
#[derive(Debug, Clone)]
pub struct PollConfig {
    /// 轮询间隔
    pub interval: Duration,
    /// 最大轮询次数(0 = 无限,直到超时)
    pub max_attempts: u32,
    /// 总超时
    pub timeout: Duration,
}

impl Default for PollConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_millis(500),
            max_attempts: 0, // 无限,靠 timeout
            timeout: Duration::from_secs(120),
        }
    }
}

/// OpenAI Assistant 封装
pub struct OpenAIAssistant {
    client: reqwest::Client,
    config: OpenAIConfig,
    assistant_id: String,
    tools: ToolRegistry,
    poll_config: PollConfig,
}

/// 判断 Run 状态是否为终态
pub fn is_terminal_status(status: &str) -> bool {
    matches!(
        status,
        "completed" | "failed" | "cancelled" | "expired" | "incomplete"
    )
}

impl OpenAIAssistant {
    /// 创建 Assistant(不带工具)
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
            tools: ToolRegistry::new(),
            poll_config: PollConfig::default(),
        })
    }

    /// 创建带工具的 Assistant
    pub async fn create_with_tools(
        config: OpenAIConfig,
        model: &str,
        instructions: &str,
        tools: ToolRegistry,
    ) -> Result<Self, AssistantError> {
        let client = reqwest::Client::new();
        let url = format!("{}/assistants", config.base_url.trim_end_matches('/'));

        // 构建 tools JSON(Assistants API 格式)
        let tools_json: Vec<Value> = tools
            .tools()
            .iter()
            .map(|t: &&Arc<dyn lc_core::tools::BaseTool>| {
                let schema = t
                    .args_schema()
                    .unwrap_or(serde_json::json!({"type": "object"}));
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name(),
                        "description": t.description(),
                        "parameters": schema,
                    }
                })
            })
            .collect();

        let body = serde_json::json!({
            "model": model,
            "instructions": instructions,
            "tools": tools_json,
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
            tools,
            poll_config: PollConfig::default(),
        })
    }

    /// 用已有 assistant id 构造(跳过 create)
    pub fn from_id(config: OpenAIConfig, assistant_id: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            assistant_id: assistant_id.into(),
            tools: ToolRegistry::new(),
            poll_config: PollConfig::default(),
        }
    }

    /// 用已有 assistant id 构造,带工具注册表
    pub fn from_id_with_tools(
        config: OpenAIConfig,
        assistant_id: impl Into<String>,
        tools: ToolRegistry,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            assistant_id: assistant_id.into(),
            tools,
            poll_config: PollConfig::default(),
        }
    }

    pub fn assistant_id(&self) -> &str {
        &self.assistant_id
    }

    /// 设置轮询配置
    pub fn with_poll_config(mut self, config: PollConfig) -> Self {
        self.poll_config = config;
        self
    }

    /// 注册工具(用于 requires_action 工具调度)
    pub fn register_tool(&mut self, tool: Arc<dyn lc_core::BaseTool>) {
        self.tools.register(tool);
    }

    /// 跑一轮对话:创建 thread + 加消息 + 创建 run + 轮询 + 处理 requires_action + 取最终消息
    ///
    /// 当 run 状态为 `requires_action` 时:
    /// 1. 解析 `required_action.submit_tool_outputs.tool_calls`
    /// 2. 经 `ToolRegistry` 执行每个 tool_call
    /// 3. `submit_tool_outputs` 回传执行结果
    /// 4. 继续轮询至 `completed`
    pub async fn run_once(&self, user_msg: &str) -> Result<String, AssistantError> {
        let base = self.config.base_url.trim_end_matches('/');
        let start = std::time::Instant::now();

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

        // 4. 轮询 + 处理 requires_action
        let mut attempts = 0u32;
        loop {
            // 超时检查
            if start.elapsed() > self.poll_config.timeout {
                return Err(AssistantError::Timeout);
            }
            // 最大次数检查(0 = 无限)
            if self.poll_config.max_attempts > 0 && attempts >= self.poll_config.max_attempts {
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
                "requires_action" => {
                    self.handle_requires_action(base, thread_id, run_id, &run_state)
                        .await?;
                }
                s if is_terminal_status(s) => return Err(AssistantError::RunFailed(s.to_string())),
                _ => tokio::time::sleep(self.poll_config.interval).await,
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

    /// 处理 requires_action:解析 tool_calls,执行,submit_tool_outputs
    async fn handle_requires_action(
        &self,
        base: &str,
        thread_id: &str,
        run_id: &str,
        run_state: &Value,
    ) -> Result<(), AssistantError> {
        // 解析 tool_calls
        let tool_calls = run_state
            .get("required_action")
            .and_then(|ra| ra.get("submit_tool_outputs"))
            .and_then(|sto| sto.get("tool_calls"))
            .and_then(|tc| tc.as_array())
            .ok_or_else(|| {
                AssistantError::Parse(
                    "requires_action 但缺少 required_action.submit_tool_outputs.tool_calls".into(),
                )
            })?;

        // 逐个执行并收集 tool_outputs
        let mut tool_outputs = Vec::new();
        for tc in tool_calls {
            let call_id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AssistantError::Parse("tool_call 缺少 id".into()))?;
            let function = tc
                .get("function")
                .ok_or_else(|| AssistantError::Parse("tool_call 缺少 function".into()))?;
            let fn_name = function
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AssistantError::Parse("tool_call.function 缺少 name".into()))?;
            let fn_args = function
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");

            // 执行工具
            let output = match self.tools.get(fn_name) {
                Some(tool) => match tool.run(fn_args.to_string()).await as Result<String, _> {
                    Ok(result) => result,
                    Err(e) => {
                        // 工具执行失败,返回错误信息给 Assistant
                        format!("工具执行错误: {}", e)
                    }
                },
                None => {
                    format!("未找到工具: {}", fn_name)
                }
            };

            tool_outputs.push(serde_json::json!({
                "tool_call_id": call_id,
                "output": output,
            }));
        }

        // submit_tool_outputs
        Self::post(
            &self.client,
            &self.config,
            &format!(
                "{}/threads/{}/runs/{}/submit_tool_outputs",
                base, thread_id, run_id
            ),
            serde_json::json!({ "tool_outputs": tool_outputs }),
        )
        .await?;

        Ok(())
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
    use lc_core::{BaseTool, ToolError};

    #[test]
    fn test_is_terminal_status() {
        assert!(is_terminal_status("completed"));
        assert!(is_terminal_status("failed"));
        assert!(is_terminal_status("cancelled"));
        assert!(is_terminal_status("expired"));
        assert!(is_terminal_status("incomplete"));
        assert!(!is_terminal_status("queued"));
        assert!(!is_terminal_status("in_progress"));
        assert!(!is_terminal_status("requires_action"));
    }

    #[test]
    fn test_from_id() {
        let config = OpenAIConfig::new("sk-test");
        let a = OpenAIAssistant::from_id(config, "asst_123");
        assert_eq!(a.assistant_id(), "asst_123");
    }

    #[test]
    fn test_from_id_with_tools() {
        let config = OpenAIConfig::new("sk-test");
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(MockTool::new("mock", "mock response")));
        let a = OpenAIAssistant::from_id_with_tools(config, "asst_456", registry);
        assert_eq!(a.assistant_id(), "asst_456");
    }

    #[test]
    fn test_poll_config_default() {
        let config = PollConfig::default();
        assert_eq!(config.interval, Duration::from_millis(500));
        assert_eq!(config.max_attempts, 0);
        assert_eq!(config.timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_poll_config_custom() {
        let config = PollConfig {
            interval: Duration::from_millis(200),
            max_attempts: 50,
            timeout: Duration::from_secs(30),
        };
        assert_eq!(config.interval, Duration::from_millis(200));
        assert_eq!(config.max_attempts, 50);
    }

    #[test]
    fn test_with_poll_config() {
        let config = OpenAIConfig::new("sk-test");
        let a = OpenAIAssistant::from_id(config, "asst_789");
        let custom = PollConfig {
            interval: Duration::from_secs(1),
            max_attempts: 10,
            timeout: Duration::from_secs(60),
        };
        let a = a.with_poll_config(custom.clone());
        assert_eq!(a.poll_config.interval, Duration::from_secs(1));
        assert_eq!(a.poll_config.max_attempts, 10);
    }

    /// 用于测试的 mock 工具
    struct MockTool {
        name: String,
        description: String,
        response: String,
    }

    impl MockTool {
        fn new(name: &str, response: &str) -> Self {
            Self {
                name: name.to_string(),
                description: format!("Mock tool: {}", name),
                response: response.to_string(),
            }
        }
    }

    #[async_trait::async_trait]
    impl BaseTool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            &self.description
        }
        async fn run(&self, _input: String) -> Result<String, ToolError> {
            Ok(self.response.clone())
        }
    }

    #[test]
    fn test_register_tool() {
        let config = OpenAIConfig::new("sk-test");
        let mut a = OpenAIAssistant::from_id(config, "asst_tool");
        a.register_tool(Arc::new(MockTool::new("weather", "sunny")));
        assert!(a.tools.contains("weather"));
    }

    #[test]
    fn test_error_display() {
        let e = AssistantError::ToolExecution {
            tool_name: "calc".into(),
            error: "overflow".into(),
        };
        assert!(e.to_string().contains("calc"));
        assert!(e.to_string().contains("overflow"));

        let e = AssistantError::RunFailed("expired".into());
        assert!(e.to_string().contains("expired"));

        let e = AssistantError::Timeout;
        assert!(e.to_string().contains("超时"));
    }
}
