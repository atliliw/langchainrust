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
    fn replace_sensitive(&self, text: &str) -> String {
        let mut result = text.to_string();
        for word in &self.sensitive_words {
            // Case-insensitive replacement
            let lower = text.to_lowercase();
            let mut start = 0;
            while let Some(pos) = lower[start..].find(&word.to_lowercase()) {
                let actual_pos = start + pos;
                let end = actual_pos + word.len();
                result = format!(
                    "{}{}{}",
                    &result[..actual_pos],
                    self.placeholder,
                    &result[end..]
                );
                start = actual_pos + self.placeholder.len();
                // Re-check the modified string
                let new_lower = result.to_lowercase();
                if start >= new_lower.len() {
                    break;
                }
                if !new_lower[start..].contains(&word.to_lowercase()) {
                    break;
                }
            }
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
