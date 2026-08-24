// src/core/structured_output/parser.rs
//! Incremental JSON parser for handling partial/incomplete JSON from streaming LLM output.

use serde_json::Value;

/// Errors produced by `PartialJsonParser`.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PartialJsonError {
    /// The buffer does not yet contain parseable JSON.
    #[error("Incomplete JSON: {0}")]
    Incomplete(String),

    /// The accumulated text is not valid JSON even after repair attempts.
    #[error("Invalid JSON: {0}")]
    Invalid(String),
}

/// Incremental JSON parser that can handle partial/incomplete JSON.
///
/// Builds up a string token by token and attempts to parse at each step,
/// returning the best partial result possible. This is designed for streaming
/// LLM output where JSON arrives in small chunks and may be incomplete until
/// the stream finishes.
///
/// # Strategy
///
/// 1. Accumulate tokens into an internal buffer.
/// 2. On each `push_and_parse`, attempt to parse the buffer as complete JSON.
/// 3. If that fails, try to repair the partial JSON by closing unclosed
///    brackets/braces and truncating incomplete string values.
/// 4. If repair yields valid JSON, return it; otherwise return
///    `PartialJsonError::Incomplete`.
///
/// # Example
///
/// ```ignore
/// let mut parser = PartialJsonParser::new();
/// // Simulating token-by-token LLM output
/// let _ = parser.push_and_parse(r#"{"name":"#); // Incomplete
/// let v = parser.push_and_parse(r#""Alice","age":30}"#); // Ok({"name":"Alice","age":30})
/// ```
pub struct PartialJsonParser {
    buffer: String,
    depth: usize,
    in_string: bool,
    escape_next: bool,
}

impl PartialJsonParser {
    /// Create a new, empty parser.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            depth: 0,
            in_string: false,
            escape_next: false,
        }
    }

    /// Push a new token and attempt to parse the accumulated buffer.
    ///
    /// Returns `Ok(value)` if the buffer (after optional repair) yields valid
    /// JSON, or `Err(PartialJsonError::Incomplete)` if it does not yet form
    /// any parseable JSON.
    pub fn push_and_parse(&mut self, token: &str) -> Result<Value, PartialJsonError> {
        // Update parser state by scanning the new token
        for ch in token.chars() {
            if self.escape_next {
                self.escape_next = false;
                continue;
            }
            if ch == '\\' && self.in_string {
                self.escape_next = true;
                continue;
            }
            if ch == '"' {
                self.in_string = !self.in_string;
                continue;
            }
            if !self.in_string {
                match ch {
                    '{' | '[' => self.depth += 1,
                    '}' | ']' => {
                        if self.depth > 0 {
                            self.depth -= 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Ensure we only push at character boundaries (M37: UTF-8 boundary check)
        if token.is_char_boundary(0) {
            self.buffer.push_str(token);
        } else {
            // Find the first valid char boundary
            let mut pos = 0;
            while pos < token.len() && !token.is_char_boundary(pos) {
                pos += 1;
            }
            self.buffer.push_str(&token[pos..]);
        }

        // H4: 模型常把 JSON 包在 ```json ... ``` 围栏里(或先输出"结果是:"等
        // 前导文本)。解析时先剥掉围栏/前导文本,只看真正的 JSON 值;否则带
        // 围栏的合法 JSON 会被判为无法解析,流式结构化输出整条路径不可用。
        let json = Self::strip_markdown_fence(&self.buffer);

        // Fast path: try full parse first
        if let Ok(value) = serde_json::from_str::<Value>(json) {
            return Ok(value);
        }

        // Only attempt repair if we have at least opened a structure
        let trimmed = json.trim();
        if self.depth > 0 || trimmed.starts_with('{') || trimmed.starts_with('[') {
            let repaired = Self::repair_partial_json(json);
            if let Ok(value) = serde_json::from_str::<Value>(&repaired) {
                return Ok(value);
            }
        }

        Err(PartialJsonError::Incomplete(format!(
            "Buffer has {} chars, depth={}",
            self.buffer.len(),
            self.depth
        )))
    }

    /// Get the final complete value.
    ///
    /// Call this when the stream has ended. It first tries to parse the
    /// fence-stripped buffer, then falls back to the repaired version.
    pub fn finalize(self) -> Result<Value, PartialJsonError> {
        // H4: 同样先剥 ```json 围栏,否则带围栏的完整 JSON 会被误判为 Invalid。
        let json = Self::strip_markdown_fence(&self.buffer);

        // Try full parse
        if let Ok(value) = serde_json::from_str::<Value>(json) {
            return Ok(value);
        }

        // Try repaired
        let repaired = Self::repair_partial_json(json);
        serde_json::from_str::<Value>(&repaired).map_err(|e| {
            PartialJsonError::Invalid(format!(
                "Failed to parse final buffer ({} chars): {}. Buffer: {}",
                self.buffer.len(),
                e,
                &self.buffer[..std::cmp::min(200, self.buffer.len())]
            ))
        })
    }

    /// Return a reference to the current buffer contents.
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Whether the parser is currently inside a JSON string.
    pub fn is_in_string(&self) -> bool {
        self.in_string
    }

    /// Current nesting depth of brackets/braces.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Strip a markdown code fence and any leading/trailing non-JSON text,
    /// returning the slice that holds the top-level JSON value.
    ///
    /// Streaming-safe: only reads what has been accumulated so far, so a
    /// partially-delivered object still yields its partial JSON (e.g. while the
    /// model is still emitting the closing brace).
    ///
    /// # Rules
    ///
    /// - Leading text up to the first `{` or `[` is dropped (covers a ```json
    ///   fence line, "结果是:" prose, and whitespace).
    /// - Trailing text after the top-level structure closes is dropped (covers
    ///   the closing ``` fence).
    /// - Returns `""` when no `{`/`[` has been seen yet (e.g. the buffer is
    ///   still just "```json").
    pub(crate) fn strip_markdown_fence(buffer: &str) -> &str {
        let bytes = buffer.as_bytes();
        // First byte that opens the top-level JSON value. `{`/`[` are ASCII, so
        // this byte index is always a UTF-8 char boundary.
        let start = match bytes.iter().position(|b| *b == b'{' || *b == b'[') {
            Some(i) => i,
            None => return "",
        };

        // Walk from `start` tracking string/escape state; the top-level value
        // ends where depth returns to 0. Everything after it (the closing ```
        // fence) is dropped. Multi-byte UTF-8 never matches the structural
        // ASCII bytes below, so byte-wise scanning is safe.
        let mut depth: i64 = 0;
        let mut in_string = false;
        let mut escape_next = false;
        let mut end = bytes.len();
        let mut idx = start;
        while idx < bytes.len() {
            let b = bytes[idx];
            if escape_next {
                escape_next = false;
            } else if b == b'\\' && in_string {
                escape_next = true;
            } else if b == b'"' {
                in_string = !in_string;
            } else if !in_string {
                match b {
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            end = idx + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            idx += 1;
        }
        &buffer[start..end]
    }

    /// Repair a partial JSON string by closing unclosed structures and
    /// truncating incomplete values.
    pub(crate) fn repair_partial_json(text: &str) -> String {
        let mut repaired = text.trim().to_string();

        // Scan the text tracking string state to correctly count braces/brackets
        // and quotes outside of strings (C20 + C21).
        let mut in_string = false;
        let mut escape_next = false;
        let mut open_braces = 0usize;
        let mut close_braces = 0usize;
        let mut open_brackets = 0usize;
        let mut close_brackets = 0usize;
        let mut unescaped_quote_count = 0usize;

        for ch in repaired.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }
            if ch == '\\' && in_string {
                escape_next = true;
                continue;
            }
            if ch == '"' {
                unescaped_quote_count += 1;
                in_string = !in_string;
                continue;
            }
            if !in_string {
                match ch {
                    '{' => open_braces += 1,
                    '}' => close_braces += 1,
                    '[' => open_brackets += 1,
                    ']' => close_brackets += 1,
                    _ => {}
                }
            }
        }

        // If we are in the middle of a string value, close it.
        // Heuristic: odd number of unescaped quotes means an unclosed string.
        if unescaped_quote_count % 2 != 0 {
            repaired.push('"');
        }

        // Close unclosed braces first (before removing trailing commas,
        // so that commas before the newly-added braces get removed)
        for _ in close_braces..open_braces {
            repaired.push('}');
        }

        // Close unclosed brackets
        for _ in close_brackets..open_brackets {
            repaired.push(']');
        }

        // Remove trailing commas before closing brackets/braces
        // (must come after closing braces/brackets so we can detect them)
        repaired = Self::remove_trailing_commas(&repaired);

        repaired
    }

    /// Remove trailing commas before closing braces/brackets (invalid in strict JSON).
    pub(crate) fn remove_trailing_commas(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let chars: Vec<char> = s.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == ',' && i + 1 < chars.len() {
                let next_non_ws = chars[i + 1..].iter().find(|c| !c.is_whitespace());
                if next_non_ws == Some(&'}') || next_non_ws == Some(&']') {
                    // Skip the trailing comma
                    i += 1;
                    continue;
                }
            }
            result.push(chars[i]);
            i += 1;
        }
        result
    }
}

impl Default for PartialJsonParser {
    fn default() -> Self {
        Self::new()
    }
}
