//! Structured-output tool-call helper (P1-3)
//!
//! The planner / router / grader previously relied on "output JSON only"
//! prompts + regex parsing, which failed whenever the LLM occasionally
//! included explanations or code fences. This unifies the approach:
//!
//! - Prefer `bind_tools` to force structured output (tool_calls arguments are
//!   the structured result).
//! - When the provider cannot bind tools (e.g. mocks / plain-text models),
//!   fall back to the same response's text and let the caller keep its existing
//!   text-parsing logic. One LLM call, no retry/fallback round trip.

use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_core::runnables::RunnableConfig;
use lc_core::tools::ToolDefinition;
use lc_schema::Message;
use serde_json::Value;

use crate::retry::{retry_chat, RetryConfig};

/// Structured call result: either with tool arguments (preferred) or text (fallback).
#[derive(Debug, Clone)]
pub(crate) struct StructuredChatResult {
    /// Text content returned by the LLM (fallback parse source when there are no tool_calls).
    pub content: String,
    /// Argument JSON of the first tool_call (used with priority when present).
    pub tool_args: Option<Value>,
}

/// Calls the LLM with `tool` bound, preferring structured output; falls back to
/// a plain text call when binding is unsupported.
///
/// Only one request: if the model returns tool_calls, extract the arguments;
/// otherwise hand `content` to the caller for text parsing.
pub(crate) async fn chat_structured<M>(
    llm: &M,
    tool: Option<ToolDefinition>,
    messages: Vec<Message>,
    config: Option<RunnableConfig>,
    retry: &RetryConfig,
) -> Result<StructuredChatResult, M::Error>
where
    M: BaseChatModel + ?Sized,
{
    let result = match tool {
        Some(tool) => match llm.bind_tools(vec![tool]) {
            Some(bound) => retry_chat(bound.as_ref(), messages, config, retry).await?,
            None => retry_chat(llm, messages, config, retry).await?,
        },
        None => retry_chat(llm, messages, config, retry).await?,
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
        use lc_core::tools::ToolCall;
        let result = LLMResult {
            content: String::new(),
            model: "mock".to_string(),
            token_usage: None,
            tool_calls: Some(vec![ToolCall::builder("call_1")
                .name("generate_plan")
                .arguments(r#"{"steps": ["a", "b"]}"#.to_string())
                .build()]),
            thinking_content: None,
        };
        let structured = extract_structured(&result);
        assert!(structured.tool_args.is_some());
        let steps = structured.tool_args.unwrap();
        assert_eq!(steps["steps"][0], "a");
    }

    #[test]
    fn test_extract_structured_invalid_tool_args() {
        use lc_core::tools::ToolCall;
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
        // Argument parsing failed → fall back to content
        assert!(structured.tool_args.is_none());
        assert_eq!(structured.content, "fallback text");
    }
}
