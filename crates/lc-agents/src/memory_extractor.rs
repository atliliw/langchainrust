// lc-agents/src/memory_extractor.rs
//! B4 (v0.22.4): LLM-backed [`lc_memory::MemoryExtractor`].
//!
//! [`LlmMemoryExtractor`] asks any [`lc_core::language_models::BaseChatModel`] to
//! distill a completed turn into durable facts as a strict JSON array, then converts
//! the reply into [`lc_memory::MemoryItem`]s for the two-tier semantic memory (see
//! `lc_memory::semantic`).
//!
//! Design notes:
//! - **Parsing is tolerant, extraction is lossy-safe.** Fenced code blocks and prose
//!   around the JSON are stripped via [`lc_core::json_parse::parse_llm_json`]; both a bare array and an
//!   object envelope (`{"memories": [...]}`) are accepted; malformed individual
//!   entries are skipped, never fatal to the turn.
//! - **The model chooses importance.** The prompt requests a `[0, 1]` float; missing
//!   values default to 0.5, out-of-range values are clamped by the store.
//! - **Keys are stable.** A missing key is derived from the fact text itself
//!   (normalized, truncated), so the same fact extracted twice updates one entry
//!   instead of duplicating.
//! - **No network in tests.** The extractor is generic over the model; the module
//!   tests use a fixed-reply fake.

use async_trait::async_trait;
use lc_core::json_parse::parse_llm_json;
use lc_core::language_models::BaseChatModel;
use lc_memory::{MemoryError, MemoryExtractor, MemoryItem};
use lc_schema::Message;
use std::sync::Arc;

/// Default importance assigned when the model omits the field.
const DEFAULT_IMPORTANCE: f64 = 0.5;
/// Maximum number of memories accepted from one turn (guards a runaway reply).
const MAX_EXTRACTIONS_PER_TURN: usize = 10;
/// Truncation length for derived keys.
const KEY_MAX_CHARS: usize = 60;

const SYSTEM_PROMPT: &str = "You are a memory extraction engine for a personal AI assistant. \
From the conversation turn, extract only durable facts worth remembering across future sessions: \
user preferences, identities, project context, goals, constraints, explicitly stated corrections. \
Do NOT extract ephemeral task chatter, greetings, or information already obvious in the current reply. \
Respond with a single JSON array and nothing else. Each element must be an object: \
{\"key\": \"short_stable_snake_case_id\", \"text\": \"one self-contained fact\", \"importance\": 0.0-1.0}. \
Higher importance means stable and broadly useful. If nothing is worth remembering, output [].";

/// `lc_memory::MemoryExtractor` backed by any chat model.
pub struct LlmMemoryExtractor<M: BaseChatModel + Send + Sync> {
    model: Arc<M>,
}

impl<M: BaseChatModel + Send + Sync> LlmMemoryExtractor<M> {
    /// Wraps an `Arc`-shared model.
    pub fn new(model: Arc<M>) -> Self {
        Self { model }
    }

    /// Wraps an owned model.
    pub fn from_model(model: M) -> Self {
        Self {
            model: Arc::new(model),
        }
    }

    fn user_prompt(namespace: &str, user_input: &str, assistant_output: &str) -> String {
        format!(
            "Memory namespace: {namespace}\n\nUser: {user_input}\n\nAssistant: {assistant_output}\n\n\
             JSON array of durable facts:"
        )
    }

    fn parse_items(raw: &str) -> Vec<MemoryItem> {
        // Accept a bare array first, then an object envelope carrying the array.
        let values: Vec<serde_json::Value> = match parse_llm_json::<serde_json::Value>(raw) {
            Ok(serde_json::Value::Array(v)) => v,
            Ok(serde_json::Value::Object(map)) => map
                .get("memories")
                .or_else(|| map.get("items"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default(),
            Ok(_) => Vec::new(),
            Err(_) => Vec::new(),
        };

        let mut items = Vec::new();
        for value in values.into_iter().take(MAX_EXTRACTIONS_PER_TURN) {
            let Some(map) = value.as_object() else {
                continue;
            };
            let Some(text) = map.get("text").and_then(|v| v.as_str()) else {
                continue;
            };
            if text.trim().is_empty() {
                continue;
            }
            let importance = map
                .get("importance")
                .and_then(|v| v.as_f64())
                .unwrap_or(DEFAULT_IMPORTANCE);
            let key = match map.get("key").and_then(|v| v.as_str()) {
                Some(k) if !k.trim().is_empty() => k.trim().to_string(),
                _ => derive_key(text),
            };
            items.push(MemoryItem::new(key, text.trim().to_string()).with_importance(importance));
        }
        items
    }
}

#[async_trait]
impl<M> MemoryExtractor for LlmMemoryExtractor<M>
where
    M: BaseChatModel + Send + Sync,
{
    async fn extract(
        &self,
        namespace: &str,
        user_input: &str,
        assistant_output: &str,
    ) -> Result<Vec<MemoryItem>, MemoryError> {
        if assistant_output.trim().is_empty() {
            return Ok(Vec::new());
        }
        let messages = vec![Message::human(Self::user_prompt(
            namespace,
            user_input,
            assistant_output,
        ))];
        let result = self
            .model
            .chat_with_system(SYSTEM_PROMPT.to_string(), messages)
            .await
            .map_err(|e| {
                MemoryError::Other(format!("memory extraction model call failed: {e:?}"))
            })?;
        Ok(Self::parse_items(&result.content))
    }
}

/// Derives a stable key from fact text: lowercase, non-alphanumeric runs → `_`,
/// truncated to [`KEY_MAX_CHARS`] (chars, not bytes).
fn derive_key(text: &str) -> String {
    let mut key = String::new();
    let mut prev_sep = true;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            key.extend(ch.to_lowercase());
            prev_sep = false;
        } else if !prev_sep {
            key.push('_');
            prev_sep = true;
        }
        if key.chars().count() >= KEY_MAX_CHARS {
            break;
        }
    }
    let key = key.trim_end_matches('_').to_string();
    if key.is_empty() {
        "memory".to_string()
    } else {
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bare_array_fenced_with_prose() {
        let raw = "Here are the facts:\n```json\n[\
            {\"key\": \"lang\", \"text\": \"User prefers Rust.\", \"importance\": 0.9},\
            {\"text\": \"Works on the agent framework.\"}\
        ]\n```";
        let items = LlmMemoryExtractor::<FakeModel>::parse_items(raw);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].key, "lang");
        assert!((items[0].importance - 0.9).abs() < 1e-9);
        // Missing key → derived from text; missing importance → 0.5.
        assert!(items[1].key.starts_with("works_on_the_agent"));
        assert!((items[1].importance - 0.5).abs() < 1e-9);
    }

    #[test]
    fn parses_envelope_and_skips_bad_entries() {
        let raw = r#"{"memories":[
            {"key":"ok","text":"Team uses trunk-based development.","importance":0.7},
            {"key":"no_text"},
            "garbage",
            {"key":"empty","text":"   "}
        ]}"#;
        let items = LlmMemoryExtractor::<FakeModel>::parse_items(raw);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].key, "ok");
    }

    #[test]
    fn non_json_reply_yields_no_items_not_error() {
        assert!(
            LlmMemoryExtractor::<FakeModel>::parse_items("I could not find any facts.").is_empty()
        );
        assert!(LlmMemoryExtractor::<FakeModel>::parse_items("[]").is_empty());
    }

    #[test]
    fn derived_key_is_stable_and_bounded() {
        let a = derive_key("User's favorite IDE is (NeoVim)!");
        let b = derive_key("user's favorite IDE is (NeoVim)!!");
        assert_eq!(a, b);
        assert!(a.chars().count() <= KEY_MAX_CHARS);
        assert_eq!(derive_key("!!! ???"), "memory");
    }

    #[tokio::test]
    async fn extract_calls_model_and_converts_reply() {
        let model = FakeModel::new(json_array());
        let extractor = LlmMemoryExtractor::from_model(model);
        let items = extractor
            .extract("user-1", "I work on langchainrust", "Noted.")
            .await
            .unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "User works on langchainrust.");
    }

    #[tokio::test]
    async fn empty_output_short_circuits_without_model_call() {
        let model = FakeModel::new(json_array());
        let extractor = LlmMemoryExtractor::from_model(model);
        let items = extractor.extract("ns", "hi", "   ").await.unwrap();
        assert!(items.is_empty());
    }

    fn json_array() -> String {
        r#"[{"key":"project","text":"User works on langchainrust.","importance":0.8}]"#.into()
    }

    /// Minimal fixed-reply chat model (only `chat` is exercised by the extractor).
    #[derive(Clone)]
    struct FakeModel {
        reply: String,
    }

    impl FakeModel {
        fn new(reply: String) -> Self {
            Self { reply }
        }
    }

    #[async_trait]
    impl lc_core::runnables::Runnable<Vec<Message>, lc_core::language_models::LLMResult> for FakeModel {
        type Error = lc_core::LcelError;

        async fn invoke(
            &self,
            _messages: Vec<Message>,
            _config: Option<lc_core::runnables::RunnableConfig>,
        ) -> Result<lc_core::language_models::LLMResult, Self::Error> {
            Ok(lc_core::language_models::LLMResult {
                content: self.reply.clone(),
                ..Default::default()
            })
        }
    }

    impl
        lc_core::language_models::BaseLanguageModel<
            Vec<Message>,
            lc_core::language_models::LLMResult,
        > for FakeModel
    {
        fn model_name(&self) -> &str {
            "fake-extractor"
        }
        fn get_num_tokens(&self, _text: &str) -> usize {
            0
        }
        fn with_temperature(self, _temp: f32) -> Self {
            self
        }
        fn with_max_tokens(self, _max: usize) -> Self {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for FakeModel {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<lc_core::runnables::RunnableConfig>,
        ) -> Result<lc_core::language_models::LLMResult, lc_core::LcelError> {
            Ok(lc_core::language_models::LLMResult {
                content: self.reply.clone(),
                ..Default::default()
            })
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<lc_core::runnables::RunnableConfig>,
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures_util::Stream<
                            Item = Result<
                                lc_core::language_models::StreamChunk,
                                lc_core::LcelError,
                            >,
                        > + Send,
                >,
            >,
            lc_core::LcelError,
        > {
            unimplemented!("extractor never streams")
        }
    }
}
