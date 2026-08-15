// src/core/json_parse.rs
//! Tolerant JSON parsing for LLM outputs.
//!
//! LLMs frequently produce malformed JSON: trailing commas, unescaped quotes,
//! incomplete brackets, or markdown-wrapped code blocks. The repair pipeline
//! lives in [`lc_shared::json_repair`] so that every parser of LLM JSON —
//! including `ToolCall::parse_arguments` in `lc-shared` — shares one tolerant
//! parser. This module keeps the historical public API (`parse_llm_json`,
//! `parse_llm_json_with_retry`, `LlmJsonParseError`) and delegates to it.

use lc_shared::json_repair::{parse_tolerant_json, JsonRepairError};
use serde::de::DeserializeOwned;

/// Error types for LLM JSON parsing.
#[derive(Debug, thiserror::Error)]
pub enum LlmJsonParseError {
    /// The raw text could not be repaired into valid JSON.
    #[error("JSON repair failed: {0}")]
    RepairFailed(String),

    /// The repaired JSON could not be deserialized into the target type.
    #[error("Deserialization failed: {details}")]
    DeserializationFailed { details: String },

    /// All retry attempts failed.
    #[error("All {attempts} retry attempts failed")]
    RetryExhausted { attempts: usize },
}

impl From<JsonRepairError> for LlmJsonParseError {
    fn from(e: JsonRepairError) -> Self {
        match e {
            JsonRepairError::RepairFailed(msg) => LlmJsonParseError::RepairFailed(msg),
            JsonRepairError::DeserializationFailed { details } => {
                LlmJsonParseError::DeserializationFailed { details }
            }
        }
    }
}

/// Parses LLM JSON output with automatic repair of common errors.
///
/// Applies these repairs in order:
/// 1. Strip markdown code fences (```json ... ```)
/// 2. Find the outermost JSON bracket pair
/// 3. Fix unescaped inner quotes within string values (heuristic)
/// 4. Remove trailing commas before `}` or `]`
/// 5. Truncate to the last matching `}` or `]` if there's trailing garbage
///
/// Returns the deserialized value or an error if repair fails.
pub fn parse_llm_json<T: DeserializeOwned>(raw: &str) -> Result<T, LlmJsonParseError> {
    parse_tolerant_json::<T>(raw).map_err(Into::into)
}

/// Parses LLM JSON with retry: if parsing fails, calls the provided callback
/// to get a corrected response from the LLM, and tries again.
///
/// The callback receives the original raw text and the parse error message,
/// and should return the LLM's corrected output. Up to `max_retries` attempts
/// are made.
pub async fn parse_llm_json_with_retry<T, F, Fut>(
    raw: &str,
    max_retries: usize,
    retry_callback: F,
) -> Result<T, LlmJsonParseError>
where
    T: DeserializeOwned,
    F: Fn(&str, &str) -> Fut,
    Fut: std::future::Future<Output = Result<String, String>>,
{
    let mut current_raw = raw.to_string();

    for attempt in 0..=max_retries {
        match parse_llm_json::<T>(&current_raw) {
            Ok(value) => return Ok(value),
            Err(e) if attempt < max_retries => {
                let error_msg = e.to_string();
                let corrected = retry_callback(&current_raw, &error_msg)
                    .await
                    .map_err(|_| LlmJsonParseError::RetryExhausted {
                        attempts: attempt + 1,
                    })?;
                current_raw = corrected;
            }
            Err(_) => {
                return Err(LlmJsonParseError::RetryExhausted {
                    attempts: attempt + 1,
                });
            }
        }
    }

    Err(LlmJsonParseError::RetryExhausted {
        attempts: max_retries + 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestStruct {
        name: String,
        value: i32,
    }

    #[test]
    fn test_parse_valid_json() {
        let raw = r#"{"name": "test", "value": 42}"#;
        let result: TestStruct = parse_llm_json(raw).unwrap();
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_parse_json_with_code_fence() {
        let raw = "```json\n{\"name\": \"test\", \"value\": 42}\n```";
        let result: TestStruct = parse_llm_json(raw).unwrap();
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_parse_json_with_trailing_comma() {
        let raw = r#"{"name": "test", "value": 42,}"#;
        let result: TestStruct = parse_llm_json(raw).unwrap();
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_parse_json_with_surrounding_text() {
        let raw = "Here is the result: {\"name\": \"test\", \"value\": 42} done.";
        let result: TestStruct = parse_llm_json(raw).unwrap();
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_parse_json_with_trailing_garbage() {
        let raw = r#"{"name": "test", "value": 42} and some extra text"#;
        let result: TestStruct = parse_llm_json(raw).unwrap();
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_parse_json_with_unescaped_inner_quotes() {
        let raw = r#"{"name": "He said "hi" and left", "value": 1}"#;
        let result: TestStruct = parse_llm_json(raw).unwrap();
        assert_eq!(result.name, "He said \"hi\" and left");
        assert_eq!(result.value, 1);
    }

    #[test]
    fn test_parse_json_array_with_trailing_comma() {
        let raw = r#"[{"name": "a", "value": 1}, {"name": "b", "value": 2},]"#;
        let result: Vec<TestStruct> = parse_llm_json(raw).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_empty_text_fails() {
        let result: Result<TestStruct, _> = parse_llm_json("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_no_json_content_fails() {
        let result: Result<TestStruct, _> = parse_llm_json("just some plain text");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_no_closing_fence() {
        let raw = "```json\n{\"name\": \"test\", \"value\": 42}";
        let result: TestStruct = parse_llm_json(raw).unwrap();
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_error_display() {
        let err = LlmJsonParseError::RepairFailed("no json".to_string());
        assert!(err.to_string().contains("no json"));

        let err = LlmJsonParseError::RetryExhausted { attempts: 3 };
        assert!(err.to_string().contains("3"));
    }

    #[tokio::test]
    async fn test_parse_with_retry_succeeds_on_first_try() {
        let raw = r#"{"name": "test", "value": 42}"#;
        let result: TestStruct = parse_llm_json_with_retry(raw, 2, |_raw, _err| async {
            Ok("should not be called".to_string())
        })
        .await
        .unwrap();
        assert_eq!(result.name, "test");
    }

    #[tokio::test]
    async fn test_parse_with_retry_succeeds_on_second_try() {
        let bad_raw = "not json at all";
        let good_raw = r#"{"name": "retry", "value": 7}"#;
        let result: TestStruct =
            parse_llm_json_with_retry(bad_raw, 2, |_raw, _err| async { Ok(good_raw.to_string()) })
                .await
                .unwrap();
        assert_eq!(result.name, "retry");
        assert_eq!(result.value, 7);
    }

    #[tokio::test]
    async fn test_parse_with_retry_fails_all_attempts() {
        let bad_raw = "not json";
        let result: Result<TestStruct, _> =
            parse_llm_json_with_retry(bad_raw, 1, |_raw, _err| async {
                Ok("still not json".to_string())
            })
            .await;
        assert!(result.is_err());
    }
}
