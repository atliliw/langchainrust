// lc-providers/src/providers/azure/types.rs
//! Private response types for the Azure OpenAI API.

use serde::Deserialize;

use lc_core::tools::ToolCall;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct AzureChatResponse {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) created: i64,
    pub(crate) model: String,
    pub(crate) choices: Vec<AzureChoice>,
    pub(crate) usage: Option<AzureUsage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct AzureChoice {
    pub(crate) index: i32,
    pub(crate) message: AzureMessage,
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct AzureMessage {
    pub(crate) role: String,
    pub(crate) content: Option<String>,
    pub(crate) tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AzureUsage {
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
    pub(crate) total_tokens: usize,
}
