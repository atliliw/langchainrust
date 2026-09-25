// lc-providers/src/providers/gemini/types.rs
//! Private request/response models for the Gemini API.

use serde::{Deserialize, Serialize};

// 0.25.0 B2: the Generative Language API wire JSON is camelCase. The Rust
// field names stay snake_case (internal ergonomics) but every multi-word field
// carries its explicit wire name; previously the snake_case keys were unknown
// to the API and silently dropped, so system prompts, sampling parameters,
// tools and multimodal parts never reached a real endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiRequest {
    pub(crate) contents: Vec<GeminiContent>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    pub(crate) system_instruction: Option<GeminiSystemInstruction>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    pub(crate) generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<GeminiToolDeclaration>>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub(crate) tool_config: Option<GeminiToolConfig>,
}

/// Gemini tool declaration wrapper (contains functionDeclarations array).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiToolDeclaration {
    #[serde(rename = "functionDeclarations")]
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
    #[serde(rename = "functionCallingConfig")]
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
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) text: Option<String>,
    #[serde(
        rename = "functionCall",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub(crate) function_call: Option<GeminiFunctionCall>,
    #[serde(
        rename = "functionResponse",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub(crate) function_response: Option<GeminiFunctionResponse>,
    /// M-9: reasoning models mark thought (chain-of-thought) parts with
    /// `thought: true`; such text is surfaced as `thinking_content` instead of
    /// being mixed into the visible reply. (`skip_serializing_if` keeps it off
    /// outgoing request parts, where it is always `None`.)
    #[serde(rename = "thought", skip_serializing_if = "Option::is_none", default)]
    pub(crate) thought: Option<bool>,
    /// B7: inline base64 media (`inlineData`: image/audio/video/PDF bytes).
    #[serde(
        rename = "inlineData",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub(crate) inline_data: Option<GeminiInlineData>,
    /// B7: hosted media reference (`fileData`: `gs://` File API URIs).
    #[serde(rename = "fileData", skip_serializing_if = "Option::is_none", default)]
    pub(crate) file_data: Option<GeminiFileData>,
}

/// Inline base64 media payload (Gemini `inlineData`).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiInlineData {
    /// MIME type of the media (e.g. `image/png`, `audio/wav`, `video/mp4`,
    /// `application/pdf`).
    #[serde(rename = "mimeType")]
    pub(crate) mime_type: String,
    /// Raw base64-encoded media bytes (data-URI payload without its header).
    pub(crate) data: String,
}

/// Hosted media reference (Gemini `fileData`, typically a `gs://` URI).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GeminiFileData {
    /// File API URI (e.g. `gs://bucket/file.png`).
    #[serde(rename = "fileUri")]
    pub(crate) file_uri: String,
    /// MIME type of the referenced media.
    #[serde(rename = "mimeType")]
    pub(crate) mime_type: String,
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
#[serde(rename_all = "camelCase")]
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

// 0.25.0 B2: response-only structs. The Generative Language API answers in
// camelCase (`usageMetadata`/`promptTokenCount`/`finishReason`); without the
// rename every field below deserialized to None, so non-stream token counting
// (mod.rs parse_response) and the streaming usage chunk silently got nothing.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiResponse {
    pub(crate) candidates: Option<Vec<GeminiCandidate>>,
    pub(crate) usage_metadata: Option<GeminiUsageMetadata>,
    #[serde(default)]
    pub(crate) prompt_feedback: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub(crate) struct GeminiCandidate {
    pub(crate) content: Option<GeminiContent>,
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeminiUsageMetadata {
    pub(crate) prompt_token_count: Option<i32>,
    pub(crate) candidates_token_count: Option<i32>,
    pub(crate) total_token_count: Option<i32>,
}
