// src/core/structured_output/parser.rs
//! Incremental JSON parser for handling partial/incomplete JSON from streaming LLM output.

use serde_json::Value;

use crate::text::truncate_at_char_boundary;

/// Errors produced by `PartialJsonParser`.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
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
                    '}' | ']' if self.depth > 0 => self.depth -= 1,
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

        // F-03: the fence-stripped slice is empty iff no top-level `{`/`[` has
        // appeared yet — everything accumulated is pre-JSON garbage (markdown
        // fence line, "答案是:", plain chat text). Drop it now so a model that
        // never opens a JSON structure cannot grow the buffer unboundedly across
        // a single stream (which previously hung `stream.next()` and leaked
        // memory). Once a structural open appears we stop dropping and resume
        // normal partial accumulation.
        if json.is_empty() {
            self.buffer.clear();
            self.depth = 0;
            self.in_string = false;
            self.escape_next = false;
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
                truncate_at_char_boundary(&self.buffer, 200)
            ))
        })
    }

    /// Drop all non-JSON leading text, re-anchoring the internal buffer at the
    /// latest top-level `{`/`[` (F-03).
    ///
    /// Bounds the buffer when an *invalid* fragment is delivered repeatedly:
    /// instead of accumulating malformed text forever, retain only the tail from
    /// the last structural open. When no structural open has ever appeared the
    /// buffer becomes empty (all pre-JSON garbage dropped). Parser state
    /// (bracket depth / string / escape) is rescanned from the re-anchored slice
    /// so subsequent `push_and_parse` calls stay consistent.
    pub fn reanchor(&mut self) {
        let anchored = Self::strip_markdown_fence(&self.buffer).to_string();
        self.buffer = anchored;
        self.depth = 0;
        self.in_string = false;
        self.escape_next = false;
        for ch in self.buffer.chars() {
            if self.escape_next {
                self.escape_next = false;
            } else if ch == '\\' && self.in_string {
                self.escape_next = true;
            } else if ch == '"' {
                self.in_string = !self.in_string;
            } else if !self.in_string {
                match ch {
                    '{' | '[' => self.depth += 1,
                    '}' | ']' if self.depth > 0 => self.depth -= 1,
                    _ => {}
                }
            }
        }
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
    ///
    /// v0.27.0 (T2): delegates to the canonical streaming repairer in
    /// [`lc_shared::json_repair::repair_partial_json`], so the two historical
    /// "partial JSON repair" logic sets no longer drift (lc-shared owns the
    /// one string-aware implementation of `remove_trailing_commas` too).
    ///
    /// 0.21.0 BUG-1: closing order is a LIFO stack (last opened, first closed),
    /// not two independent counters. The old counter-based repair closed all
    /// braces first, then all brackets — independent of actual nesting — so a
    /// stream truncated at the end of an array produced illegal JSON
    /// (`{"a": [1, 2` → `{"a": [1, 2}]` instead of `{"a": [1, 2]}`), failing
    /// the whole streaming structured-output path in that scenario.
    pub(crate) fn repair_partial_json(text: &str) -> String {
        lc_shared::json_repair::repair_partial_json(text)
    }
}

impl Default for PartialJsonParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F-03: a model that treats JSON as chat text emits ordinary tokens with no
    /// top-level `{`/`[`. Each stays `Incomplete`, but the buffer must not grow
    /// with every token (that previously hung `stream.next()` and leaked memory).
    #[test]
    fn plain_chat_tokens_keep_buffer_bounded() {
        let mut parser = PartialJsonParser::new();
        for _ in 0..10_000 {
            let r = parser.push_and_parse("hello world, this is chat, not JSON ");
            assert!(matches!(r, Err(PartialJsonError::Incomplete(_))));
            assert!(
                parser.buffer().len() < 64,
                "buffer grew unboundedly to {} chars",
                parser.buffer().len()
            );
        }
    }

    /// F-03: pre-structure garbage is dropped, but once a `{` arrives normal
    /// partial-JSON accumulation resumes and the value parses (健康恢复).
    #[test]
    fn garbage_then_valid_json_recovers() {
        let mut parser = PartialJsonParser::new();
        assert!(parser.push_and_parse("答案是：" ).is_err());
        assert!(parser.push_and_parse("那么我们来聊：").is_err());
        let v = parser.push_and_parse(r#"{"name":"Alice","age":30}"#).unwrap();
        assert_eq!(v, serde_json::json!({"name": "Alice", "age": 30}));
    }

    /// F-03: `reanchor` drops leading junk and retains only the tail from the
    /// last top-level `{`/`[`, keeping the buffer bounded after an invalid
    /// fragment and leaving the stream recoverable.
    #[test]
    fn reanchor_bounds_buffer_and_keeps_recoverable() {
        let mut parser = PartialJsonParser::new();
        for _ in 0..3 {
            let _ = parser.push_and_parse("junk ");
        }
        // a malformed fragment that *has* opened a structure (so it is not
        // auto-cleared by the pre-structure guard) accumulates leading junk.
        let _ = parser.push_and_parse(r#"{"k": "#);
        assert!(parser.buffer().contains('{'));

        parser.reanchor();
        assert!(parser.buffer().is_empty() || parser.buffer().starts_with('{'));
        assert!(parser.buffer().len() < 64, "re-anchored buffer stayed {} chars", parser.buffer().len());
    }
}
