//! 结构化输出工具调用辅助(P1-3)
//!
//! planner / router / grader 此前靠提示词"只输出 JSON"+ 正则解析,LLM 偶发
//! 夹带解释/代码块就解析失败。这里统一:
//!
//! - 优先 `bind_tools` 强制结构化输出(tool_calls 参数即结构化结果)。
//! - Provider 不支持工具绑定(如 mock / 纯文本模型)时,回落同一响应的文本,
//!   由调用方沿用原有文本解析逻辑。一次 LLM 调用,不回退重试。

use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_core::runnables::RunnableConfig;
use lc_core::tools::ToolDefinition;
use lc_schema::Message;
use serde_json::Value;

use crate::retry::{retry_chat, RetryConfig};

/// 结构化调用结果:要么带工具参数(首选),要么带文本(回落)。
#[derive(Debug, Clone)]
pub(crate) struct StructuredChatResult {
    /// LLM 返回的文本内容(无 tool_calls 时的回落解析源)。
    pub content: String,
    /// 首个 tool_call 的参数 JSON(存在则优先使用)。
    pub tool_args: Option<Value>,
}

/// 绑定 `tool` 后调用 LLM,优先结构化输出;不支持绑定则普通文本调用。
///
/// 只发一次请求:模型返回 tool_calls 就提取参数,否则把 `content` 交给
/// 调用方做文本解析。
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

/// 从 LLM 结果提取首个 tool_call 的参数 JSON。
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
            tool_calls: Some(vec![ToolCall::new(
                "call_1",
                "generate_plan",
                r#"{"steps": ["a", "b"]}"#.to_string(),
            )]),
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
            tool_calls: Some(vec![ToolCall::new("call_2", "f", "not-json".to_string())]),
            thinking_content: None,
        };
        let structured = extract_structured(&result);
        // 参数解析失败 → 回落 content
        assert!(structured.tool_args.is_none());
        assert_eq!(structured.content, "fallback text");
    }
}
