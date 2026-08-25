// lc-providers/src/providers/gemini/types.rs
//! Private request/response models for the Gemini API.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiRequest {
    pub(crate) contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<GeminiToolDeclaration>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_config: Option<GeminiToolConfig>,
}

/// Gemini tool declaration wrapper (contains functionDeclarations array).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiToolDeclaration {
    pub(crate) function_declarations: Vec<GeminiFunctionDeclaration>,
}

/// A single function declaration in Gemini format.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionDeclaration {
    pub(crate) name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) parameters: Option<serde_json::Value>,
}

/// Gemini tool configuration (controls tool choice behavior).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiToolConfig {
    pub(crate) function_calling_config: GeminiFunctionCallingConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionCallingConfig {
    /// "AUTO", "ANY", or "NONE"
    pub(crate) mode: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<String>,
    pub(crate) parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) function_response: Option<GeminiFunctionResponse>,
}

/// A function call returned by the model (in the response).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionCall {
    pub(crate) name: String,
    pub(crate) args: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiFunctionResponse {
    pub(crate) name: String,
    pub(crate) response: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiSystemInstruction {
    pub(crate) parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_k: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiResponse {
    pub(crate) candidates: Option<Vec<GeminiCandidate>>,
    pub(crate) usage_metadata: Option<GeminiUsageMetadata>,
    #[serde(default)]
    pub(crate) prompt_feedback: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct GeminiCandidate {
    pub(crate) content: Option<GeminiContent>,
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiUsageMetadata {
    pub(crate) prompt_token_count: Option<i32>,
    pub(crate) candidates_token_count: Option<i32>,
    pub(crate) total_token_count: Option<i32>,
}
