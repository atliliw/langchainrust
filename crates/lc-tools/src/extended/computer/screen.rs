//! Computer Use tool — struct, constructors, and [`BaseTool`] implementation.
//!
//! Supports actions: screenshot, click, type, scroll, key_press, wait.
//! The `AnthropicApi` mode constructs tool-call payloads compatible with
//! Anthropic's computer-use beta.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use lc_core::tools::ToolError;
use lc_core::BaseTool;

use super::actions::{ComputerMode, ComputerUseInput, ComputerUseOutput};

// ---------------------------------------------------------------------------
// Tool struct
// ---------------------------------------------------------------------------

/// Computer Use tool that can be registered with `ToolRegistry` and used by
/// any agent.
pub struct ComputerUseTool {
    pub(super) mode: ComputerMode,
    pub(super) api_key: String,
    pub(super) base_url: String,
    pub(super) display_width: u32,
    pub(super) display_height: u32,
    pub(super) client: reqwest::Client,
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
// Dispatch
// ---------------------------------------------------------------------------

impl ComputerUseTool {
    /// Dispatch to the appropriate mode handler.
    pub(super) async fn dispatch(
        &self,
        input: &ComputerUseInput,
    ) -> Result<ComputerUseOutput, ToolError> {
        match self.mode {
            ComputerMode::AnthropicApi => self.execute_anthropic(input).await,
        }
    }
}
