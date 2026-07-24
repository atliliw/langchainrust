//! Computer Use tool for screen interaction via Anthropic API or native mode.
//!
//! Supports actions: screenshot, click, type, scroll, key_press, wait.
//! The `AnthropicApi` mode constructs tool-call payloads compatible with
//! Anthropic's computer-use beta. The `Native` mode (feature-gated) is a
//! placeholder for local screenshot + input simulation.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::core::tools::ToolError;
use crate::BaseTool;

// ---------------------------------------------------------------------------
// Mode enum
// ---------------------------------------------------------------------------

/// Operating mode for `ComputerUseTool`.
#[derive(Debug, Clone)]
pub enum ComputerMode {
    /// Forward actions to the Anthropic computer-use API.
    /// The tool sends a request to the Anthropic messages endpoint with the
    /// appropriate `computer-use` tool-use block and returns the response.
    AnthropicApi,
    /// Local screenshot + input simulation (behind `native-computer` feature gate).
    #[cfg(feature = "native-computer")]
    Native,
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// Typed input for the computer-use tool (used for schema generation).
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ComputerUseInput {
    /// Action to perform.
    ///
    /// One of: `screenshot`, `click`, `type`, `scroll`, `key_press`, `wait`.
    pub action: String,

    /// Coordinate as `[x, y]` for actions that need a position
    /// (`click`, `scroll`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coordinate: Option<Vec<i32>>,

    /// Text to type (for `type` action).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Key names to press simultaneously (for `key_press` action).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keys: Option<Vec<String>>,

    /// Scroll direction: `up` or `down` (for `scroll` action).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<String>,

    /// Scroll amount in clicks (for `scroll` action).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amount: Option<i32>,

    /// Wait duration in milliseconds (for `wait` action).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

/// Typed output from the computer-use tool.
#[derive(Debug, serde::Serialize)]
pub struct ComputerUseOutput {
    /// The action that was executed.
    pub action: String,
    /// Human-readable result message.
    pub result: String,
    /// Optional base64-encoded screenshot data (for `screenshot` action).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screenshot_base64: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool struct
// ---------------------------------------------------------------------------

/// Computer Use tool that can be registered with `ToolRegistry` and used by
/// any agent.
///
/// # Modes
///
/// - `AnthropicApi`: Sends the action to the Anthropic messages API with the
///   `computer-use` tool type. The LLM decides what to do; this tool
///   constructs the proper request body and returns the result.
/// - `Native` (feature-gated): Local screenshot + input simulation.
///
/// # Example
///
/// ```rust,no_run
/// use langchainrust::ComputerUseTool;
///
/// let tool = ComputerUseTool::new_anthropic(
///     "sk-ant-...".to_string(),
///     1024,
///     768,
/// );
/// ```
pub struct ComputerUseTool {
    mode: ComputerMode,
    api_key: String,
    base_url: String,
    display_width: u32,
    display_height: u32,
    client: reqwest::Client,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl ComputerUseTool {
    /// Create a new tool in `AnthropicApi` mode.
    pub fn new_anthropic(
        api_key: impl Into<String>,
        display_width: u32,
        display_height: u32,
    ) -> Self {
        Self {
            mode: ComputerMode::AnthropicApi,
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            display_width,
            display_height,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Create with a custom Anthropic-compatible base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Create with a custom HTTP timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        self
    }

    /// Create a new tool in `Native` mode (requires `native-computer` feature).
    #[cfg(feature = "native-computer")]
    pub fn new_native(display_width: u32, display_height: u32) -> Self {
        Self {
            mode: ComputerMode::Native,
            api_key: String::new(),
            base_url: String::new(),
            display_width,
            display_height,
            client: reqwest::Client::new(),
        }
    }

    /// Return the current mode.
    pub fn mode(&self) -> &ComputerMode {
        &self.mode
    }
}

impl Default for ComputerUseTool {
    fn default() -> Self {
        Self::new_anthropic(String::new(), 1024, 768)
    }
}

// ---------------------------------------------------------------------------
// Action helpers (AnthropicApi mode)
// ---------------------------------------------------------------------------

impl ComputerUseTool {
    /// Build the Anthropic messages API request body for a computer-use action.
    fn build_anthropic_request(&self, action_input: &ComputerUseInput) -> Result<Value, ToolError> {
        let tool_input = self.build_tool_input(action_input)?;
        let tool_input_str = serde_json::to_string(&tool_input).unwrap_or_default();

        Ok(serde_json::json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "tools": [
                {
                    "type": "computer_20250124",
                    "name": "computer",
                    "display_width_px": self.display_width,
                    "display_height_px": self.display_height,
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": format!(
                        "Use the computer tool with these parameters: {}",
                        tool_input_str
                    )
                }
            ],
            "tool_choice": {
                "type": "tool",
                "name": "computer"
            }
        }))
    }

    /// Build the tool-specific input dict that Anthropic's computer-use API
    /// expects for each action type.
    fn build_tool_input(&self, input: &ComputerUseInput) -> Result<Value, ToolError> {
        match input.action.as_str() {
            "screenshot" => Ok(serde_json::json!({
                "action": "screenshot"
            })),
            "click" => {
                let coord = input.coordinate.as_deref().unwrap_or(&[0, 0]);
                let button = if input
                    .text
                    .as_deref()
                    .unwrap_or("left")
                    .eq_ignore_ascii_case("right")
                {
                    "right"
                } else {
                    "left"
                };
                Ok(serde_json::json!({
                    "action": "mouse_click",
                    "coordinate": coord,
                    "text": button
                }))
            }
            "type" => Ok(serde_json::json!({
                "action": "type",
                "text": input.text.as_deref().unwrap_or("")
            })),
            "scroll" => {
                let coord = input.coordinate.as_deref().unwrap_or(&[0, 0]);
                let direction = input.direction.as_deref().unwrap_or("down");
                let scroll_direction = if direction.eq_ignore_ascii_case("up") {
                    "up"
                } else if direction.eq_ignore_ascii_case("left") {
                    "left"
                } else if direction.eq_ignore_ascii_case("right") {
                    "right"
                } else {
                    "down"
                };
                Ok(serde_json::json!({
                    "action": "scroll",
                    "coordinate": coord,
                    "direction": scroll_direction,
                    "amount": input.amount.unwrap_or(3)
                }))
            }
            "key_press" => Ok(serde_json::json!({
                "action": "key",
                "text": input.keys.as_deref().unwrap_or(&[]).join("+")
            })),
            "wait" => Ok(serde_json::json!({
                "action": "wait",
                "duration_ms": input.duration_ms.unwrap_or(1000)
            })),
            _ => Err(ToolError::InvalidInput(format!(
                "Unknown action: '{}'. Valid actions: screenshot, click, type, scroll, key_press, wait",
                input.action
            ))),
        }
    }

    /// Send a request to the Anthropic messages API and return the response.
    async fn call_anthropic_api(&self, body: &Value) -> Result<String, ToolError> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "computer-use-2025-01-24")
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Anthropic API request failed: {}", e))
            })?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to read response body: {}", e))
        })?;

        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(format!(
                "Anthropic API returned status {}: {}",
                status, text
            )));
        }

        Ok(text)
    }

    /// Execute an action in AnthropicApi mode.
    async fn execute_anthropic(
        &self,
        input: &ComputerUseInput,
    ) -> Result<ComputerUseOutput, ToolError> {
        if self.api_key.is_empty() {
            return Err(ToolError::InvalidInput(
                "API key is required for AnthropicApi mode".to_string(),
            ));
        }

        self.validate_input(input)?;

        let body = self.build_anthropic_request(input)?;
        let response_text = self.call_anthropic_api(&body).await?;

        // Try to extract screenshot base64 from the response if this was a
        // screenshot action.
        let screenshot_base64 = if input.action == "screenshot" {
            serde_json::from_str::<Value>(&response_text)
                .ok()
                .and_then(|v| {
                    v.get("content")?.as_array()?.iter().find_map(|block| {
                        if block.get("type")?.as_str()? == "image" {
                            block.get("source")?.get("data")?.as_str().map(String::from)
                        } else {
                            None
                        }
                    })
                })
        } else {
            None
        };

        Ok(ComputerUseOutput {
            action: input.action.clone(),
            result: response_text,
            screenshot_base64,
        })
    }

    /// Execute an action in Native mode (placeholder).
    #[cfg(feature = "native-computer")]
    async fn execute_native(
        &self,
        input: &ComputerUseInput,
    ) -> Result<ComputerUseOutput, ToolError> {
        self.validate_input(input)?;

        match input.action.as_str() {
            "screenshot" => Ok(ComputerUseOutput {
                action: "screenshot".to_string(),
                result: "Screenshot captured (native mode)".to_string(),
                screenshot_base64: None,
            }),
            "click" => {
                let coord = input.coordinate.as_deref().unwrap_or(&[0, 0]);
                Ok(ComputerUseOutput {
                    action: "click".to_string(),
                    result: format!("Clicked at ({}, {}) (native mode)", coord[0], coord[1]),
                    screenshot_base64: None,
                })
            }
            "type" => Ok(ComputerUseOutput {
                action: "type".to_string(),
                result: format!(
                    "Typed '{}' (native mode)",
                    input.text.as_deref().unwrap_or("")
                ),
                screenshot_base64: None,
            }),
            "scroll" => {
                let coord = input.coordinate.as_deref().unwrap_or(&[0, 0]);
                Ok(ComputerUseOutput {
                    action: "scroll".to_string(),
                    result: format!(
                        "Scrolled {} by {} at ({}, {}) (native mode)",
                        input.direction.as_deref().unwrap_or("down"),
                        input.amount.unwrap_or(3),
                        coord[0],
                        coord[1]
                    ),
                    screenshot_base64: None,
                })
            }
            "key_press" => Ok(ComputerUseOutput {
                action: "key_press".to_string(),
                result: format!(
                    "Pressed keys: {} (native mode)",
                    input.keys.as_deref().unwrap_or(&[]).join("+")
                ),
                screenshot_base64: None,
            }),
            "wait" => {
                let ms = input.duration_ms.unwrap_or(1000);
                tokio::time::sleep(Duration::from_millis(ms)).await;
                Ok(ComputerUseOutput {
                    action: "wait".to_string(),
                    result: format!("Waited {}ms (native mode)", ms),
                    screenshot_base64: None,
                })
            }
            other => Err(ToolError::InvalidInput(format!(
                "Unknown action: {}",
                other
            ))),
        }
    }

    /// Validate the input for the given action.
    fn validate_input(&self, input: &ComputerUseInput) -> Result<(), ToolError> {
        let valid_actions = ["screenshot", "click", "type", "scroll", "key_press", "wait"];

        if !valid_actions.contains(&input.action.as_str()) {
            return Err(ToolError::InvalidInput(format!(
                "Unknown action: '{}'. Valid actions: {:?}",
                input.action, valid_actions
            )));
        }

        match input.action.as_str() {
            "click" => {
                if input.coordinate.is_none() {
                    return Err(ToolError::InvalidInput(
                        "'click' action requires 'coordinate' field as [x, y]".to_string(),
                    ));
                }
            }
            "scroll" => {
                if input.coordinate.is_none() {
                    return Err(ToolError::InvalidInput(
                        "'scroll' action requires 'coordinate' field as [x, y]".to_string(),
                    ));
                }
                if input.direction.is_none() {
                    return Err(ToolError::InvalidInput(
                        "'scroll' action requires 'direction' field (up/down/left/right)"
                            .to_string(),
                    ));
                }
            }
            "type" => {
                if input.text.is_none() {
                    return Err(ToolError::InvalidInput(
                        "'type' action requires 'text' field".to_string(),
                    ));
                }
            }
            "key_press" => {
                if input.keys.is_none() {
                    return Err(ToolError::InvalidInput(
                        "'key_press' action requires 'keys' field".to_string(),
                    ));
                }
            }
            _ => {}
        }

        // Validate coordinate bounds if provided.
        if let Some(ref coord) = input.coordinate {
            if coord.len() != 2 {
                return Err(ToolError::InvalidInput(
                    "coordinate must be exactly [x, y]".to_string(),
                ));
            }
            if coord[0] < 0 || coord[0] as u32 > self.display_width {
                return Err(ToolError::InvalidInput(format!(
                    "x coordinate {} out of bounds (0-{})",
                    coord[0], self.display_width
                )));
            }
            if coord[1] < 0 || coord[1] as u32 > self.display_height {
                return Err(ToolError::InvalidInput(format!(
                    "y coordinate {} out of bounds (0-{})",
                    coord[1], self.display_height
                )));
            }
        }

        Ok(())
    }

    /// Dispatch to the appropriate mode handler.
    async fn dispatch(&self, input: &ComputerUseInput) -> Result<ComputerUseOutput, ToolError> {
        match self.mode {
            ComputerMode::AnthropicApi => self.execute_anthropic(input).await,
            #[cfg(feature = "native-computer")]
            ComputerMode::Native => self.execute_native(input).await,
        }
    }
}

// ---------------------------------------------------------------------------
// BaseTool implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl BaseTool for ComputerUseTool {
    fn name(&self) -> &str {
        "computer_use"
    }

    fn description(&self) -> &str {
        "Computer use tool for screen interaction. \
         Input JSON: {\"action\": \"screenshot|click|type|scroll|key_press|wait\", \
         \"coordinate\": [x, y], \"text\": \"...\", \"keys\": [\"...\"], \
         \"direction\": \"up|down|left|right\", \"amount\": N, \"duration_ms\": N}. \
         - screenshot: capture the current screen. \
         - click: click at (x, y). \
         - type: type text string. \
         - scroll: scroll at (x, y) in direction by amount. \
         - key_press: press key combination. \
         - wait: wait for duration_ms milliseconds."
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: ComputerUseInput =
            serde_json::from_str(&input).map_err(|e| ToolError::InvalidInput(e.to_string()))?;

        let output = self.dispatch(&parsed).await?;

        serde_json::to_string(&output).map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    }

    fn args_schema(&self) -> Option<Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(ComputerUseInput)).ok()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> ComputerUseTool {
        ComputerUseTool::new_anthropic("test-key", 1024, 768)
    }

    // -- name / description / schema --

    #[test]
    fn test_name() {
        let t = tool();
        assert_eq!(t.name(), "computer_use");
    }

    #[test]
    fn test_description_contains_actions() {
        let t = tool();
        let desc = t.description();
        for action in &["screenshot", "click", "type", "scroll", "key_press", "wait"] {
            assert!(
                desc.contains(action),
                "description should mention '{}'",
                action
            );
        }
    }

    #[test]
    fn test_args_schema_has_action_field() {
        let t = tool();
        let schema = t.args_schema().expect("schema should be present");
        let props = schema
            .get("properties")
            .expect("schema should have properties");
        assert!(
            props.get("action").is_some(),
            "schema should define 'action' property"
        );
    }

    // -- input validation --

    #[tokio::test]
    async fn test_run_invalid_json() {
        let t = tool();
        let result = t.run("not json".to_string()).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_run_unknown_action() {
        let t = tool();
        let input = r#"{"action":"fly"}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("Unknown action"),
            "error should mention unknown action, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_click_missing_coordinate() {
        let t = tool();
        let input = r#"{"action":"click"}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("coordinate"),
            "error should mention missing coordinate, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_type_missing_text() {
        let t = tool();
        let input = r#"{"action":"type"}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("text"),
            "error should mention missing text, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_key_press_missing_keys() {
        let t = tool();
        let input = r#"{"action":"key_press"}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("keys"),
            "error should mention missing keys, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_scroll_missing_direction() {
        let t = tool();
        let input = r#"{"action":"scroll","coordinate":[100,200]}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("direction"),
            "error should mention missing direction, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_coordinate_out_of_bounds() {
        let t = tool();
        let input = r#"{"action":"click","coordinate":[2000,500]}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("out of bounds"),
            "error should mention out of bounds, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_coordinate_wrong_length() {
        let t = tool();
        let input = r#"{"action":"click","coordinate":[100]}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("exactly [x, y]"),
            "error should mention coordinate format, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_run_empty_api_key() {
        let t = ComputerUseTool::new_anthropic("", 1024, 768);
        let input = r#"{"action":"screenshot"}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(
            err.to_string().contains("API key"),
            "error should mention API key, got: {}",
            err
        );
    }

    // -- build_tool_input (unit tests) --

    #[test]
    fn test_build_tool_input_screenshot() {
        let t = tool();
        let input = ComputerUseInput {
            action: "screenshot".to_string(),
            coordinate: None,
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "screenshot");
    }

    #[test]
    fn test_build_tool_input_click() {
        let t = tool();
        let input = ComputerUseInput {
            action: "click".to_string(),
            coordinate: Some(vec![100, 200]),
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "mouse_click");
        assert_eq!(result["coordinate"], serde_json::json!([100, 200]));
        assert_eq!(result["text"], "left");
    }

    #[test]
    fn test_build_tool_input_click_right_button() {
        let t = tool();
        let input = ComputerUseInput {
            action: "click".to_string(),
            coordinate: Some(vec![50, 75]),
            text: Some("right".to_string()),
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["text"], "right");
    }

    #[test]
    fn test_build_tool_input_type() {
        let t = tool();
        let input = ComputerUseInput {
            action: "type".to_string(),
            coordinate: None,
            text: Some("hello world".to_string()),
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "type");
        assert_eq!(result["text"], "hello world");
    }

    #[test]
    fn test_build_tool_input_scroll() {
        let t = tool();
        let input = ComputerUseInput {
            action: "scroll".to_string(),
            coordinate: Some(vec![500, 300]),
            text: None,
            keys: None,
            direction: Some("up".to_string()),
            amount: Some(5),
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "scroll");
        assert_eq!(result["coordinate"], serde_json::json!([500, 300]));
        assert_eq!(result["direction"], "up");
        assert_eq!(result["amount"], 5);
    }

    #[test]
    fn test_build_tool_input_key_press() {
        let t = tool();
        let input = ComputerUseInput {
            action: "key_press".to_string(),
            coordinate: None,
            text: None,
            keys: Some(vec!["ctrl".to_string(), "c".to_string()]),
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "key");
        assert_eq!(result["text"], "ctrl+c");
    }

    #[test]
    fn test_build_tool_input_wait() {
        let t = tool();
        let input = ComputerUseInput {
            action: "wait".to_string(),
            coordinate: None,
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: Some(2000),
        };
        let result = t.build_tool_input(&input).unwrap();
        assert_eq!(result["action"], "wait");
        assert_eq!(result["duration_ms"], 2000);
    }

    #[test]
    fn test_build_tool_input_unknown_action() {
        let t = tool();
        let input = ComputerUseInput {
            action: "fly".to_string(),
            coordinate: None,
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let result = t.build_tool_input(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown action"));
    }

    // -- constructor tests --

    #[test]
    fn test_default() {
        let t = ComputerUseTool::default();
        assert_eq!(t.name(), "computer_use");
        assert_eq!(t.display_width, 1024);
        assert_eq!(t.display_height, 768);
    }

    #[test]
    fn test_with_base_url() {
        let t = ComputerUseTool::new_anthropic("key", 1920, 1080)
            .with_base_url("https://custom.api.com");
        assert_eq!(t.base_url, "https://custom.api.com");
    }

    #[test]
    fn test_mode_returns_anthropic() {
        let t = tool();
        assert!(matches!(t.mode(), ComputerMode::AnthropicApi));
    }

    // -- build_anthropic_request --

    #[test]
    fn test_build_anthropic_request_contains_model() {
        let t = tool();
        let input = ComputerUseInput {
            action: "screenshot".to_string(),
            coordinate: None,
            text: None,
            keys: None,
            direction: None,
            amount: None,
            duration_ms: None,
        };
        let body = t.build_anthropic_request(&input).unwrap();
        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert!(body["tools"].is_array());
        assert_eq!(body["tools"][0]["type"], "computer_20250124");
    }

    // -- integration test (requires network, ignored by default) --

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY environment variable"]
    async fn test_screenshot_with_real_api() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
        let t = ComputerUseTool::new_anthropic(api_key, 1024, 768);
        let input = r#"{"action":"screenshot"}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_ok(), "screenshot should succeed: {:?}", result);
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY environment variable"]
    async fn test_click_with_real_api() {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
        let t = ComputerUseTool::new_anthropic(api_key, 1024, 768);
        let input = r#"{"action":"click","coordinate":[500,400]}"#.to_string();
        let result = t.run(input).await;
        assert!(result.is_ok(), "click should succeed: {:?}", result);
    }
}
