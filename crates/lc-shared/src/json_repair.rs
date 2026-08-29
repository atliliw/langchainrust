// lc-shared/src/json_repair.rs
//! Tolerant JSON parsing for LLM outputs.
//!
//! LLMs frequently produce malformed JSON: trailing commas, unescaped quotes,
//! incomplete brackets, or markdown-wrapped code blocks. Historically this
//! logic lived in `lc-core`, but `ToolCall::parse_arguments` (in this crate)
//! parses the same kind of LLM-generated JSON and cannot depend on `lc-core`.
//! So the repair pipeline lives here so that every parser of LLM JSON — in
//! `lc-shared` and `lc-core` alike — converges on a single tolerant parser.

use serde::de::DeserializeOwned;

/// Error types for tolerant JSON parsing.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JsonRepairError {
    /// The raw text could not be repaired into valid JSON.
    #[error("JSON repair failed: {0}")]
    RepairFailed(String),

    /// The repaired JSON could not be deserialized into the target type.
    #[error("Deserialization failed: {details}")]
    DeserializationFailed {
        /// Details of the deserialization failure.
        details: String,
    },
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
pub fn parse_tolerant_json<T: DeserializeOwned>(raw: &str) -> Result<T, JsonRepairError> {
    let repaired = repair_json(raw)?;
    serde_json::from_str::<T>(&repaired).map_err(|e| JsonRepairError::DeserializationFailed {
        details: format!("{} (repaired JSON: {})", e, truncate(&repaired, 200)),
    })
}

/// Repairs common LLM JSON mistakes and extracts the JSON portion.
///
/// Returns the repaired JSON string, or an error if no JSON content is found.
pub fn repair_json(raw: &str) -> Result<String, JsonRepairError> {
    let trimmed = raw.trim();

    // Step 1: Strip markdown code fences
    let stripped = strip_code_fences(trimmed);

    // Step 2: Find outermost JSON bracket pair
    let extracted = extract_bracket_pair(&stripped);

    // Step 3: Fix unescaped inner quotes inside string values
    let quotes_fixed = fix_unescaped_quotes(&extracted);

    // Step 4: Remove trailing commas before } or ]
    let no_trailing = remove_trailing_commas(&quotes_fixed);

    // Step 5: Try to truncate trailing garbage
    let truncated = truncate_to_matching_bracket(&no_trailing);

    // Verify the result is at least syntactically plausible
    if truncated.is_empty() {
        return Err(JsonRepairError::RepairFailed(
            "no JSON content found".to_string(),
        ));
    }

    Ok(truncated)
}

/// Strips markdown code fences from the text.
pub fn strip_code_fences(text: &str) -> String {
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
pub fn extract_bracket_pair(text: &str) -> String {
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

/// Escapes `"` characters that appear inside string values but are not the
/// string's closing quote.
///
/// The heuristic: while inside a string, a `"` is treated as the closing
/// quote only if the next non-whitespace character is a JSON structural
/// character (`:`, `,`, `]`, `}`) or end of input — that is, a quote that
/// closes a key or a value. Any other `"` (e.g. `He said "hi"`) is an inner
/// quote and is escaped. This is a best-effort repair for the common LLM
/// failure of emitting unescaped quotes inside string values.
pub fn fix_unescaped_quotes(json: &str) -> String {
    let chars: Vec<char> = json.chars().collect();
    let n = chars.len();
    let mut result = String::with_capacity(json.len());
    let mut in_string = false;
    let mut escape_next = false;

    for i in 0..n {
        let c = chars[i];

        if in_string {
            if escape_next {
                result.push(c);
                escape_next = false;
                continue;
            }
            if c == '\\' {
                result.push(c);
                escape_next = true;
                continue;
            }
            if c == '"' {
                // Look ahead: is this the closing quote of a key or value?
                let mut j = i + 1;
                while j < n && chars[j].is_whitespace() {
                    j += 1;
                }
                let is_closing = j >= n || matches!(chars[j], ':' | ',' | ']' | '}');
                if is_closing {
                    in_string = false;
                    result.push(c);
                } else {
                    // Inner quote — escape it
                    result.push('\\');
                    result.push(c);
                }
                continue;
            }
            result.push(c);
        } else if c == '"' {
            in_string = true;
            result.push(c);
        } else {
            result.push(c);
        }
    }

    result
}

/// Removes trailing commas before closing brackets.
///
/// Handles patterns like: `{"a": 1, "b": 2,}` → `{"a": 1, "b": 2}`.
///
/// 0.20.0 K1: tracks string state so a comma inside a string literal (e.g.
/// `{"a": "x, }"}`) is kept — it is content, not a trailing comma. Mirrors the
/// in-string/escape state machine used by [`extract_bracket_pair`] /
/// [`fix_unescaped_quotes`]; without it the old version corrupted string values
/// whose text ended with a comma followed by `}`/`]`.
pub fn remove_trailing_commas(json: &str) -> String {
    let mut result = String::with_capacity(json.len());
    let chars: Vec<char> = json.chars().collect();
    let len = chars.len();
    let mut in_string = false;
    let mut escape_next = false;

    for i in 0..len {
        let ch = chars[i];
        if in_string {
            result.push(ch);
            if escape_next {
                escape_next = false;
            } else if ch == '\\' {
                escape_next = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            result.push(ch);
            continue;
        }
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
pub fn truncate_to_matching_bracket(json: &str) -> String {
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
        let result: TestStruct = parse_tolerant_json(raw).unwrap();
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_parse_json_with_code_fence() {
        let raw = "```json\n{\"name\": \"test\", \"value\": 42}\n```";
        let result: TestStruct = parse_tolerant_json(raw).unwrap();
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_parse_json_with_trailing_comma() {
        let raw = r#"{"name": "test", "value": 42,}"#;
        let result: TestStruct = parse_tolerant_json(raw).unwrap();
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_parse_json_with_surrounding_text() {
        let raw = "Here is the result: {\"name\": \"test\", \"value\": 42} done.";
        let result: TestStruct = parse_tolerant_json(raw).unwrap();
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_parse_json_with_trailing_garbage() {
        let raw = r#"{"name": "test", "value": 42} and some extra text"#;
        let result: TestStruct = parse_tolerant_json(raw).unwrap();
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_parse_json_with_unescaped_inner_quotes() {
        // LLM output with unescaped quotes inside a string value
        let raw = r#"{"name": "He said "hi" and left", "value": 1}"#;
        let result: TestStruct = parse_tolerant_json(raw).unwrap();
        assert_eq!(result.name, "He said \"hi\" and left");
        assert_eq!(result.value, 1);
    }

    #[test]
    fn test_parse_json_array_with_trailing_comma() {
        let raw = r#"[{"name": "a", "value": 1}, {"name": "b", "value": 2},]"#;
        let result: Vec<TestStruct> = parse_tolerant_json(raw).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_empty_text_fails() {
        let result: Result<TestStruct, _> = parse_tolerant_json("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_no_json_content_fails() {
        let result: Result<TestStruct, _> = parse_tolerant_json("just some plain text");
        assert!(result.is_err());
    }

    #[test]
    fn test_fix_unescaped_quotes_inner_only() {
        let input = r#"{"a": "He said "hi"", "b": 2}"#;
        assert_eq!(
            fix_unescaped_quotes(input),
            r#"{"a": "He said \"hi\"", "b": 2}"#
        );
    }

    #[test]
    fn test_fix_unescaped_quotes_leaves_closing_quotes() {
        let input = r#"{"key": "value", "n": 1}"#;
        assert_eq!(fix_unescaped_quotes(input), input);
    }

    #[test]
    fn test_fix_unescaped_quotes_empty_string() {
        let input = r#"{"a": ""}"#;
        assert_eq!(fix_unescaped_quotes(input), input);
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
    fn test_remove_trailing_commas_preserves_string_literals() {
        // 0.20.0 K1: commas inside string values are content, not trailing commas.
        let input = r#"{"a": "text, } more"}"#;
        assert_eq!(remove_trailing_commas(input), input);

        let input = r#"{"a": "x,]", "b": 1}"#;
        assert_eq!(remove_trailing_commas(input), input);

        let input = r#"{"a": "he said \"hi, \"", "b": 1,}"#;
        assert_eq!(
            remove_trailing_commas(input),
            r#"{"a": "he said \"hi, \"", "b": 1}"#
        );

        let input = r#"{"a": "1, 2, 3", "b": [1,]}"#;
        assert_eq!(
            remove_trailing_commas(input),
            r#"{"a": "1, 2, 3", "b": [1]}"#
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
        let result: TestStruct = parse_tolerant_json(raw).unwrap();
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_error_display() {
        let err = JsonRepairError::RepairFailed("no json".to_string());
        assert!(err.to_string().contains("no json"));
    }
}
