use std::collections::HashMap;

use serde_json::Value;

use lc_chains::base::{BaseChain, ChainError, ChainResult};

use crate::protocol::A2AMessage;

/// Extract the `A2AMessage` from `tasks/send` params.
///
/// Reads `params.message` (a message object); when absent, the whole params
/// become the input content.
pub(crate) fn extract_message(params: &Value) -> A2AMessage {
    match params.get("message") {
        Some(msg_val) => serde_json::from_value(msg_val.clone()).unwrap_or_else(|_| {
            let content = extract_content_text(msg_val.get("content"));
            if content.is_empty() {
                log::warn!(
                    "tasks/send: message content is not a plain string and yielded no text; \
                     role={:?} content={:?}",
                    msg_val.get("role"),
                    msg_val.get("content")
                );
            }
            A2AMessage::new(
                msg_val
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("user"),
                content,
            )
        }),
        None => A2AMessage::user(params.to_string()),
    }
}

/// 从 message 的 `content` 里提取可读文本。
///
/// 兼容纯字符串 `"hi"` 与 A2A 2.0 结构化内容对象 `{"type":"text","text":"hi"}`:
/// 对象优先取 `text` / `content` 字段;仍取不到时整体 JSON 序列化作为内容,
/// 避免把结构化 content 静默当成空串。
pub(crate) fn extract_content_text(content: Option<&Value>) -> String {
    match content {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(map)) => map
            .get("text")
            .or_else(|| map.get("content"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| serde_json::to_string(map).unwrap_or_default()),
        Some(v) => serde_json::to_string(v).unwrap_or_default(),
    }
}

/// Build the chain input map from a full message history (P2-2).
///
/// A single-message history keeps its original content (backward compatible);
/// multi-turn histories are joined as `role: content` lines so the chain sees
/// the whole conversation.
pub(crate) fn build_chain_input_from_history(
    history: &[A2AMessage],
    chain: &dyn BaseChain,
) -> HashMap<String, Value> {
    let content = if history.len() == 1 {
        history[0].content.clone()
    } else {
        history
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    };
    build_chain_input(&content, chain)
}

/// Build the chain input map from message content, using the chain's first
/// declared input key (or a fallback `"input"` key).
pub(crate) fn build_chain_input(content: &str, chain: &dyn BaseChain) -> HashMap<String, Value> {
    let mut map = HashMap::new();
    let input_keys = chain.input_keys();
    if let Some(first_key) = input_keys.first() {
        map.insert(first_key.to_string(), Value::String(content.to_string()));
    } else {
        map.insert("input".to_string(), Value::String(content.to_string()));
    }
    map
}

/// Extract the output text from a chain result (first value).
pub(crate) fn extract_output(result: &ChainResult) -> String {
    result
        .values()
        .next()
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Whether a chain error signals that more input is needed (P2-3).
///
/// `MissingInput` (a required key is absent) and `InputError` (the input is
/// present but incomplete/malformed) are mapped to the `input-required` task
/// state so the client can resume the conversation.
pub(crate) fn is_input_required(e: &ChainError) -> bool {
    matches!(e, ChainError::MissingInput(_) | ChainError::InputError(_))
}
