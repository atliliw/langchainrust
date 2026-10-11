// lc-agents/src/hooks/content_filter.rs
//! ContentFilterHook — filters or replaces sensitive words in stream output.
//!
//! Scans each streaming token for sensitive words and either drops the token
//! or replaces the sensitive content with a placeholder.

use async_trait::async_trait;

use super::{AgentHook, StreamAction};

/// A hook that filters sensitive words from streaming output.
///
/// When `on_stream_chunk` is invoked, it checks the token for any of the
/// configured sensitive words. If found, the token is either filtered (dropped)
/// or replaced with a placeholder.
///
/// # Example
///
/// ```rust,ignore
/// use lc_agents::hooks::ContentFilterHook;
///
/// let hook = ContentFilterHook::new(vec!["secret".to_string(), "password".to_string()]);
/// let executor = AgentExecutor::new(agent, tools).hook(hook);
/// ```
pub struct ContentFilterHook {
    /// Words to filter from the stream.
    sensitive_words: Vec<String>,
    /// Placeholder to replace sensitive words with.
    placeholder: String,
    /// If true, drop the entire token if it contains a sensitive word.
    /// If false, replace the sensitive word with the placeholder.
    drop_token: bool,
}

impl ContentFilterHook {
    /// Creates a new ContentFilterHook with the given sensitive words.
    pub fn new(sensitive_words: Vec<String>) -> Self {
        Self {
            sensitive_words,
            placeholder: "[REDACTED]".to_string(),
            drop_token: false,
        }
    }

    /// Sets the placeholder for replaced words.
    pub fn with_placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    /// Sets whether to drop entire tokens containing sensitive words.
    pub fn with_drop_token(mut self, drop: bool) -> Self {
        self.drop_token = drop;
        self
    }

    /// Checks if the text contains any sensitive word.
    fn contains_sensitive(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        self.sensitive_words
            .iter()
            .any(|word| text_lower.contains(&word.to_lowercase()))
    }

    /// Replaces sensitive words in the text with the placeholder.
    ///
    /// 0.26.0 (MEDIUM): the previous implementation computed byte offsets on a
    /// lowercase copy of the *original* text, then applied them as slice indexes
    /// to `result` — which the placeholder had already made longer. After the
    /// first replacement (or on a multi-word hit) the offsets desynced and a
    /// boundary could land in the middle of a UTF‑8 char, panicking
    /// `on_stream_chunk`. This version scans character by character and always
    /// slices the *current* string, so replacements never run ahead of the
    /// source they index. (Case-insensitive for ASCII; the offset math assumes
    /// the lowercased form of each slice has the same byte length as the
    /// original, which holds except for a handful of expansion-case Unicode
    /// codepoints — a corner this bound-reader-style replace tolerates.)
    fn replace_sensitive(&self, text: &str) -> String {
        let mut result = text.to_string();
        for word in &self.sensitive_words {
            let needle = word.to_lowercase();
            // Collect byte ranges of every non-overlapping hit for THIS word in
            // the current string, then rebuild from the tail so earlier offsets
            // stay valid while we shrink the string.
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            let mut cursor = 0;
            while cursor < result.len() {
                let remaining = result[cursor..].to_lowercase();
                let Some(rel) = remaining.find(&needle) else {
                    break;
                };
                let hit_start = cursor + rel;
                let hit_end = hit_start + needle.len();
                ranges.push((hit_start, hit_end));
                cursor = hit_end;
            }
            if ranges.is_empty() {
                continue;
            }
            // Rebuild tail-first: replacing later hits first keeps earlier
            // byte offsets intact while the string shrinks.
            let mut rebuilt = result;
            for (start, end) in ranges.into_iter().rev() {
                rebuilt = format!(
                    "{}{}{}",
                    &rebuilt[..start],
                    self.placeholder,
                    &rebuilt[end..]
                );
            }
            result = rebuilt;
        }
        result
    }
}

#[async_trait]
impl AgentHook for ContentFilterHook {
    fn on_stream_chunk(&self, chunk: &str) -> StreamAction {
        if !self.contains_sensitive(chunk) {
            return StreamAction::Forward(chunk.to_string());
        }

        if self.drop_token {
            StreamAction::Filter
        } else {
            StreamAction::Replace(self.replace_sensitive(chunk))
        }
    }
}
