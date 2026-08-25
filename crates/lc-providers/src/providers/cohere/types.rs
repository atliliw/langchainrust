// lc-providers/src/providers/cohere/types.rs
//! Private response types for the Cohere v2 API.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct CohereChatResponse {
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) message: Option<CohereMessage>,
    pub(crate) usage: Option<CohereUsage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct CohereMessage {
    pub(crate) role: String,
    pub(crate) content: Vec<CohereContentPart>,
    #[serde(default)]
    pub(crate) tool_calls: Vec<CohereToolCall>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct CohereContentPart {
    pub(crate) r#type: String,
    pub(crate) text: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct CohereToolCall {
    pub(crate) id: String,
    pub(crate) r#type: String,
    pub(crate) function: CohereFunctionCall,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CohereFunctionCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CohereUsage {
    pub(crate) tokens: CohereTokenUsage,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CohereTokenUsage {
    pub(crate) input_tokens: usize,
    pub(crate) output_tokens: usize,
}
