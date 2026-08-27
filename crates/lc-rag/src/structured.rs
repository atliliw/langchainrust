// lc-rag/src/structured.rs
//! Structured-output helpers (P2-1)
//!
//! GraphRAG entity extraction / MultiQuery query generation previously relied on LLM text
//! JSON/line parsing, which failed when the LLM occasionally smuggled in explanations,
//! numbering, or code blocks. This unifies that (reusing the bind_tools path proven in
//! lc-agents, applying the same lesson across crates):
//!
//! - Prefer `bind_tools` to force structured output (the tool_calls arguments are the
//!   structured result).
//! - When the provider does not support tool binding (e.g. mock / plain-text models), fall
//!   back to the same response's text, leaving the caller to reuse its existing text-parsing
//!   logic. One LLM call, no retry.

use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_core::tools::ToolDefinition;
use lc_schema::Message;
use serde_json::Value;

/// Structured call result: either carries tool arguments (preferred) or text (fallback).
#[derive(Debug, Clone)]
pub(crate) struct StructuredChatResult {
    /// The text content returned by the LLM (the fallback parse source when there are no tool_calls).
    pub content: String,
    /// The first tool_call's argument JSON (used in preference when present).
    pub tool_args: Option<Value>,
}

/// Structured call error.
#[derive(Debug, thiserror::Error)]
pub(crate) enum StructuredError {
    #[error("{0}")]
    Msg(String),
}

/// Calls the LLM with `tool` bound, preferring structured output; falls back to a plain text call when binding is unsupported.
///
/// Makes a single request: if the model returns tool_calls, extract the arguments; otherwise
/// hand `content` to the caller for text parsing. Errors are uniformly wrapped as
/// `StructuredError::Msg` (each caller maps them to its own error type).
pub(crate) async fn chat_structured<M>(
    llm: &M,
    tool: Option<ToolDefinition>,
    messages: Vec<Message>,
) -> Result<StructuredChatResult, StructuredError>
where
    M: BaseChatModel + ?Sized,
{
    let result = match tool {
        Some(tool) => match llm.bind_tools(vec![tool]) {
            Some(bound) => bound
                .chat(messages, None)
                .await
                .map_err(|e| StructuredError::Msg(e.to_string()))?,
            None => llm
                .chat(messages, None)
                .await
                .map_err(|e| StructuredError::Msg(e.to_string()))?,
        },
        None => llm
            .chat(messages, None)
            .await
            .map_err(|e| StructuredError::Msg(e.to_string()))?,
    };
    Ok(extract_structured(&result))
}

/// Extracts the first tool_call's argument JSON from an LLM result.
pub(crate) fn extract_structured(result: &LLMResult) -> StructuredChatResult {
    let tool_args = result
        .tool_calls
        .as_ref()
        .and_then(|calls| calls.first())
        .and_then(|call| call.parse_arguments::<Value>().ok());
    StructuredChatResult {
        content: result.content.clone(),
        tool_args,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::tools::ToolCall;

    #[test]
    fn test_extract_structured_no_tool_calls() {
        let result = LLMResult {
            content: "text answer".to_string(),
            model: "mock".to_string(),
            token_usage: None,
            tool_calls: None,
            thinking_content: None,
        };
        let structured = extract_structured(&result);
        assert_eq!(structured.content, "text answer");
        assert!(structured.tool_args.is_none());
    }

    #[test]
    fn test_extract_structured_with_tool_call() {
        let result = LLMResult {
            content: String::new(),
            model: "mock".to_string(),
            token_usage: None,
            tool_calls: Some(vec![ToolCall::builder("call_1")
                .name("extract_entities_relations")
                .arguments(r#"{"entities": [], "relations": []}"#.to_string())
                .build()]),
            thinking_content: None,
        };
        let structured = extract_structured(&result);
        assert!(structured.tool_args.is_some());
        assert!(structured.tool_args.unwrap()["entities"].is_array());
    }

    #[test]
    fn test_extract_structured_invalid_tool_args() {
        let result = LLMResult {
            content: "fallback text".to_string(),
            model: "mock".to_string(),
            token_usage: None,
            tool_calls: Some(vec![ToolCall::builder("call_2")
                .name("f")
                .arguments("not-json".to_string())
                .build()]),
            thinking_content: None,
        };
        let structured = extract_structured(&result);
        // Argument parsing failed -> fall back to content
        assert!(structured.tool_args.is_none());
        assert_eq!(structured.content, "fallback text");
    }
}
