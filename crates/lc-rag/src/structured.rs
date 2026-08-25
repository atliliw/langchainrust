// lc-rag/src/structured.rs
//! 结构化输出辅助(P2-1)
//!
//! GraphRAG 实体提取 / MultiQuery 查询生成此前依赖 LLM 文本 JSON/行解析,
//! LLM 偶发夹带解释/编号/代码块就解析失败。这里统一(复用 lc-agents 已证明
//! 的 bind_tools 路径,跨 crate 落地同款教训):
//!
//! - 优先 `bind_tools` 强制结构化输出(tool_calls 参数即结构化结果)。
//! - Provider 不支持工具绑定(如 mock / 纯文本模型)时,回落同一响应的文本,
//!   由调用方沿用原有文本解析逻辑。一次 LLM 调用,不回退重试。

use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_core::tools::ToolDefinition;
use lc_schema::Message;
use serde_json::Value;

/// 结构化调用结果:要么带工具参数(首选),要么带文本(回落)。
#[derive(Debug, Clone)]
pub(crate) struct StructuredChatResult {
    /// LLM 返回的文本内容(无 tool_calls 时的回落解析源)。
    pub content: String,
    /// 首个 tool_call 的参数 JSON(存在则优先使用)。
    pub tool_args: Option<Value>,
}

/// 结构化调用错误。
#[derive(Debug, thiserror::Error)]
pub(crate) enum StructuredError {
    #[error("{0}")]
    Msg(String),
}

/// 绑定 `tool` 后调用 LLM,优先结构化输出;不支持绑定则普通文本调用。
///
/// 只发一次请求:模型返回 tool_calls 就提取参数,否则把 `content` 交给
/// 调用方做文本解析。错误统一包成 `StructuredError::Msg`(调用方各自映射到自身错误类型)。
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
        // 参数解析失败 → 回落 content
        assert!(structured.tool_args.is_none());
        assert_eq!(structured.content, "fallback text");
    }
}
