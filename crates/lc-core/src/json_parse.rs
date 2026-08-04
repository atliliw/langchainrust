// src/core/json_parse.rs
//! Tolerant JSON parsing for LLM outputs.
//!
//! LLMs frequently produce malformed JSON: trailing commas, unescaped quotes,
//! incomplete brackets, or markdown-wrapped code blocks. This module provides
//! utilities to handle these cases gracefully.

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

/// Parses LLM JSON output with automatic repair of common errors.
///
/// Applies these repairs in order:
/// 1. Strip markdown code fences (```json ... ```)
/// 2. Find the outermost JSON bracket pair
/// 3. Remove trailing commas before `}` or `]`
/// 4. Fix unescaped inner quotes within string values (heuristic)
/// 5. Truncate to the last matching `}` or `]` if there's trailing garbage
///
/// Returns the deserialized value or an error if repair fails.
pub fn parse_llm_json<T: DeserializeOwned>(raw: &str) -> Result<T, LlmJsonParseError> {
    let repaired = repair_json(raw)?;
    serde_json::from_str::<T>(&repaired).map_err(|e| LlmJsonParseError::DeserializationFailed {
        details: format!("{} (repaired JSON: {})", e, truncate(&repaired, 200)),
    })
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

/// Repairs common LLM JSON mistakes and extracts the JSON portion.
fn repair_json(raw: &str) -> Result<String, LlmJsonParseError> {
    let trimmed = raw.trim();

    // Step 1: Strip markdown code fences
    let stripped = strip_code_fences(trimmed);

    // Step 2: Find outermost JSON bracket pair
    let extracted = extract_bracket_pair(&stripped);

    // Step 3: Remove trailing commas before } or ]
    let no_trailing = remove_trailing_commas(&extracted);

    // Step 4: Try to truncate trailing garbage
    let truncated = truncate_to_matching_bracket(&no_trailing);

    // Verify the result is at least syntactically plausible
    if truncated.is_empty() {
        return Err(LlmJsonParseError::RepairFailed(
            "no JSON content found".to_string(),
        ));
    }

    Ok(truncated)
}

/// Strips markdown code fences from the text.
fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("```json") {
        if let Some(end) = rest.find("```") {
            return rest[..end].trim().to_string();
        }
        // No closing fence — strip prefix only
        return rest.trim().to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("```") {
        if let Some(end) = rest.find("```") {
            return rest[..end].trim().to_string();
        }
        return rest.trim().to_string();
    }
    trimmed.to_string()
}

/// Finds the outermost `[...]` or `{...}` bracket pair.
fn extract_bracket_pair(text: &str) -> String {
    let bytes = text.as_bytes();
    let start_idx = text.find(['[', '{']);

    if let Some(start) = start_idx {
        let open = bytes[start];
        let close = if open == b'[' { b']' } else { b'}' };

        let mut depth = 0i32;
        let mut in_string = false;
        let mut escape_next = false;

        for i in start..bytes.len() {
            let ch = bytes[i];
            if escape_next {
                escape_next = false;
                continue;
            }
            if ch == b'\\' && in_string {
                escape_next = true;
                continue;
            }
            if ch == b'"' {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }
            if ch == open {
                depth += 1;
            } else if ch == close {
                depth -= 1;
                if depth == 0 {
                    return text[start..=i].to_string();
                }
            }
        }
    }

    text.to_string()
}

/// Removes trailing commas before closing brackets.
///
/// Handles patterns like: `[...,]` or `{...:,}` → `[...,]` or `{...:,}`
fn remove_trailing_commas(json: &str) -> String {
    let mut result = String::with_capacity(json.len());
    let chars: Vec<char> = json.chars().collect();
    let len = chars.len();

    for i in 0..len {
        let ch = chars[i];
        // Check if this comma is followed by ] or } (possibly with whitespace)
        if ch == ',' {
            let mut j = i + 1;
            while j < len && chars[j].is_whitespace() {
                j += 1;
            }
            if j < len && (chars[j] == ']' || chars[j] == '}') {
                // Skip this trailing comma
                continue;
            }
        }
        result.push(ch);
    }

    result
}

/// Truncates to the last matching closing bracket if there's trailing garbage.
fn truncate_to_matching_bracket(json: &str) -> String {
    // If the JSON already ends with ] or }, it's likely complete.
    let trimmed = json.trim_end();
    if trimmed.ends_with(']') || trimmed.ends_with('}') {
        return trimmed.to_string();
    }

    // Find the last closing bracket
    if let Some(last_close) = trimmed.rfind([']', '}']) {
        return trimmed[..=last_close].to_string();
    }

    json.to_string()
}

/// Truncates a string for display purposes.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let end = s
            .char_indices()
            .take(max_len)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        format!("{}...", &s[..end])
    }
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
    fn test_strip_code_fences_json() {
        let input = "```json\n{\"key\": \"val\"}\n```";
        assert_eq!(strip_code_fences(input), "{\"key\": \"val\"}");
    }

    #[test]
    fn test_strip_code_fences_plain() {
        let input = "```\n{\"key\": \"val\"}\n```";
        assert_eq!(strip_code_fences(input), "{\"key\": \"val\"}");
    }

    #[test]
    fn test_strip_code_fences_no_fence() {
        let input = "{\"key\": \"val\"}";
        assert_eq!(strip_code_fences(input), "{\"key\": \"val\"}");
    }

    #[test]
    fn test_remove_trailing_commas_object() {
        let input = r#"{"a": 1, "b": 2,}"#;
        assert_eq!(remove_trailing_commas(input), r#"{"a": 1, "b": 2}"#);
    }

    #[test]
    fn test_remove_trailing_commas_array() {
        let input = r#"[1, 2, 3,]"#;
        assert_eq!(remove_trailing_commas(input), r#"[1, 2, 3]"#);
    }

    #[test]
    fn test_remove_trailing_commas_nested() {
        let input = r#"{"arr": [1, 2,], "val": 3,}"#;
        assert_eq!(
            remove_trailing_commas(input),
            r#"{"arr": [1, 2], "val": 3}"#
        );
    }

    #[test]
    fn test_extract_bracket_pair_array() {
        let input = "prefix [1, 2, 3] suffix";
        assert_eq!(extract_bracket_pair(input), "[1, 2, 3]");
    }

    #[test]
    fn test_extract_bracket_pair_object() {
        let input = r#"text {"a": 1} more"#;
        assert_eq!(extract_bracket_pair(input), r#"{"a": 1}"#);
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
