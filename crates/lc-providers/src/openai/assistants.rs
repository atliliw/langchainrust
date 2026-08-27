//! OpenAI Assistants API wrapper
//!
//! Wraps OpenAI's official Assistants / Threads / Run, supporting server-side conversation state.
//! v0.4.1: supports `requires_action` tool dispatch -- when a run needs tool calls,
//! parse the tool_calls, execute them via the ToolRegistry, report back via submit_tool_outputs,
//! and keep polling until completion.
//! Note: an endpoint supporting the Assistants API (official OpenAI) is required; some compatible-mode endpoints may not support it.

use super::config::OpenAIConfig;
use lc_core::tools::ToolRegistry;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// Assistants error
#[derive(Debug)]
#[non_exhaustive]
pub enum AssistantError {
    /// HTTP request error
    Http(String),
    /// API-returned error
    Api(String),
    /// Response parse error
    Parse(String),
    /// Run ended in a non-terminal state
    RunFailed(String),
    /// Tool execution error
    ToolExecution {
        /// Tool name
        tool_name: String,
        /// Error message
        error: String,
    },
    /// Polling timed out
    Timeout,
}

impl std::fmt::Display for AssistantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssistantError::Http(m) => write!(f, "HTTP error: {}", m),
            AssistantError::Api(m) => write!(f, "API error: {}", m),
            AssistantError::Parse(m) => write!(f, "Parse error: {}", m),
            AssistantError::RunFailed(s) => write!(f, "Run failed, status: {}", s),
            AssistantError::ToolExecution { tool_name, error } => {
                write!(f, "Tool '{}' execution failed: {}", tool_name, error)
            }
            AssistantError::Timeout => write!(f, "Polling timeout"),
        }
    }
}

impl std::error::Error for AssistantError {}

/// Polling configuration
#[derive(Debug, Clone)]
pub struct PollConfig {
    /// Polling interval
    pub interval: Duration,
    /// Maximum number of poll attempts (0 = unlimited, until timeout)
    pub max_attempts: u32,
    /// Overall timeout
    pub timeout: Duration,
}

impl Default for PollConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_millis(500),
            max_attempts: 0, // unlimited, bounded by timeout
            timeout: Duration::from_secs(120),
        }
    }
}

/// OpenAI Assistant wrapper
pub struct OpenAIAssistant {
    client: reqwest::Client,
    config: OpenAIConfig,
    assistant_id: String,
    tools: ToolRegistry,
    poll_config: PollConfig,
}

/// Whether the run status is terminal
pub fn is_terminal_status(status: &str) -> bool {
    matches!(
        status,
        "completed" | "failed" | "cancelled" | "expired" | "incomplete"
    )
}

impl OpenAIAssistant {
    /// Creates an Assistant (without tools)
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
            .ok_or_else(|| AssistantError::Parse("missing assistant id".into()))?
            .to_string();
        Ok(Self {
            client,
            config,
            assistant_id: id,
            tools: ToolRegistry::new(),
            poll_config: PollConfig::default(),
        })
    }

    /// Creates an Assistant with tools
    pub async fn create_with_tools(
        config: OpenAIConfig,
        model: &str,
        instructions: &str,
        tools: ToolRegistry,
    ) -> Result<Self, AssistantError> {
        let client = reqwest::Client::new();
        let url = format!("{}/assistants", config.base_url.trim_end_matches('/'));

        // Build the tools JSON (Assistants API format)
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
            .ok_or_else(|| AssistantError::Parse("missing assistant id".into()))?
            .to_string();
        Ok(Self {
            client,
            config,
            assistant_id: id,
            tools,
            poll_config: PollConfig::default(),
        })
    }

    /// Constructs from an existing assistant id (skips create)
    pub fn from_id(config: OpenAIConfig, assistant_id: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            assistant_id: assistant_id.into(),
            tools: ToolRegistry::new(),
            poll_config: PollConfig::default(),
        }
    }

    /// Constructs from an existing assistant id, with a tool registry
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

    /// Returns the assistant id
    pub fn assistant_id(&self) -> &str {
        &self.assistant_id
    }

    /// Sets the polling configuration
    pub fn with_poll_config(mut self, config: PollConfig) -> Self {
        self.poll_config = config;
        self
    }

    /// Registers a tool (used for requires_action tool dispatch)
    pub fn register_tool(&mut self, tool: Arc<dyn lc_core::BaseTool>) {
        self.tools.register(tool);
    }

    /// Runs one conversation turn: create a thread + add a message + create a run + poll +
    /// handle requires_action + fetch the final message.
    ///
    /// When the run status is `requires_action`:
    /// 1. Parse `required_action.submit_tool_outputs.tool_calls`
    /// 2. Execute each tool_call via the `ToolRegistry`
    /// 3. Report results back via `submit_tool_outputs`
    /// 4. Keep polling until `completed`
    pub async fn run_once(&self, user_msg: &str) -> Result<String, AssistantError> {
        let base = self.config.base_url.trim_end_matches('/');
        let start = std::time::Instant::now();

        // 1. Create a thread
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
            .ok_or_else(|| AssistantError::Parse("missing thread id".into()))?;

        // 2. Add the user message
        Self::post(
            &self.client,
            &self.config,
            &format!("{}/threads/{}/messages", base, thread_id),
            serde_json::json!({ "role": "user", "content": user_msg }),
        )
        .await?;

        // 3. Create a run
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
            .ok_or_else(|| AssistantError::Parse("missing run id".into()))?;

        // 4. Poll + handle requires_action
        let mut attempts = 0u32;
        loop {
            // Timeout check
            if start.elapsed() > self.poll_config.timeout {
                return Err(AssistantError::Timeout);
            }
            // Max-attempts check (0 = unlimited)
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

        // 5. Fetch the final assistant message (data[0] is the newest)
        let messages = Self::get(
            &self.client,
            &self.config,
            &format!("{}/threads/{}/messages", base, thread_id),
        )
        .await?;
        let data = messages
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| AssistantError::Parse("missing messages data".into()))?;
        let first = data
            .first()
            .ok_or_else(|| AssistantError::Parse("no messages".into()))?;
        let text = first
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("text"))
            .and_then(|t| t.get("value"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| AssistantError::Parse("failed to parse message content".into()))?;
        Ok(text.to_string())
    }

    /// Handles requires_action: parse the tool_calls, execute them, submit_tool_outputs
    async fn handle_requires_action(
        &self,
        base: &str,
        thread_id: &str,
        run_id: &str,
        run_state: &Value,
    ) -> Result<(), AssistantError> {
        // Parse the tool_calls
        let tool_calls = run_state
            .get("required_action")
            .and_then(|ra| ra.get("submit_tool_outputs"))
            .and_then(|sto| sto.get("tool_calls"))
            .and_then(|tc| tc.as_array())
            .ok_or_else(|| {
                AssistantError::Parse(
                    "requires_action but missing required_action.submit_tool_outputs.tool_calls"
                        .into(),
                )
            })?;

        // Execute each one and collect the tool_outputs
        let mut tool_outputs = Vec::new();
        for tc in tool_calls {
            let call_id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AssistantError::Parse("tool_call missing id".into()))?;
            let function = tc
                .get("function")
                .ok_or_else(|| AssistantError::Parse("tool_call missing function".into()))?;
            let fn_name = function
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AssistantError::Parse("tool_call.function missing name".into()))?;
            let fn_args = function
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");

            // Execute the tool
            let output = match self.tools.get(fn_name) {
                Some(tool) => match tool.run(fn_args.to_string()).await as Result<String, _> {
                    Ok(result) => result,
                    Err(e) => {
                        // Tool execution failed, report the error back to the Assistant
                        format!("Tool execution error: {}", e)
                    }
                },
                None => {
                    format!("Tool not found: {}", fn_name)
                }
            };

            tool_outputs.push(serde_json::json!({
                "tool_call_id": call_id,
                "output": output,
            }));
        }

        // Submit the tool outputs
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
                .unwrap_or("unknown error");
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

    /// Mock tool used for testing
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
        assert!(e.to_string().contains("timeout"));
    }
}
