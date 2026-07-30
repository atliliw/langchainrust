//! Action types and helpers for the computer-use tool.
//!
//! Defines the input/output structures and the operating mode enum used by
//! `ComputerUseTool`, plus the action-building and
//! validation helpers that translate user input into Anthropic API payloads.

use serde_json::Value;

#[cfg(feature = "native-computer")]
use std::time::Duration;

use crate::core::tools::ToolError;

use super::screen::ComputerUseTool;

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
// Action helpers (AnthropicApi mode)
// ---------------------------------------------------------------------------

impl ComputerUseTool {
    /// Build the Anthropic messages API request body for a computer-use action.
    pub(super) fn build_anthropic_request(
        &self,
        action_input: &ComputerUseInput,
    ) -> Result<Value, ToolError> {
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
    pub(super) fn build_tool_input(&self, input: &ComputerUseInput) -> Result<Value, ToolError> {
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
    pub(super) async fn call_anthropic_api(&self, body: &Value) -> Result<String, ToolError> {
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
    pub(super) async fn execute_anthropic(
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
    pub(super) async fn execute_native(
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
    pub(super) fn validate_input(&self, input: &ComputerUseInput) -> Result<(), ToolError> {
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
}
