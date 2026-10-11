// src/language_models/providers/anthropic/config.rs
//! Anthropic configuration types and constants.

use lc_core::tools::ToolDefinition;
use lc_core::{JsonSchemaOutput, PromptCache, SensitiveString};
use serde::{Deserialize, Serialize};
use std::env;

use crate::ProviderError;

/// Anthropic API endpoint.
pub const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";

/// Claude model list (B5, v0.22.4 — refreshed to the Claude 4.x aliases).
///
/// Non-exhaustive: the `model` field accepts any provider string, including
/// dated snapshots such as `claude-sonnet-4-5-20250929`.
pub const CLAUDE_MODELS: [&str; 5] = [
    "claude-opus-4-1",          // Claude Opus 4.1
    "claude-sonnet-4-5",        // Claude Sonnet 4.5
    "claude-haiku-4-5",         // Claude Haiku 4.5
    "claude-3-5-sonnet-latest", // Claude 3.5 Sonnet (legacy alias)
    "claude-3-5-haiku-latest",  // Claude 3.5 Haiku (legacy alias)
];

/// Type of extended thinking mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ThinkingType {
    /// Extended thinking enabled with a token budget.
    Enabled,
    /// Extended thinking disabled (default).
    #[default]
    Disabled,
}

/// Configuration for Anthropic extended thinking.
///
/// When enabled, the model emits a "thinking" content block before the
/// final text answer, allowing callers to observe the reasoning process.
#[derive(Debug, Clone)]
pub struct ThinkingConfig {
    /// Maximum number of tokens the model may spend on thinking.
    pub budget_tokens: usize,
    /// Whether thinking is enabled or disabled.
    pub r#type: ThinkingType,
}

impl ThinkingConfig {
    /// Create a new enabled thinking config with the given budget.
    pub fn enabled(budget_tokens: usize) -> Self {
        Self {
            budget_tokens,
            r#type: ThinkingType::Enabled,
        }
    }

    /// Create a disabled thinking config.
    pub fn disabled() -> Self {
        Self {
            budget_tokens: 0,
            r#type: ThinkingType::Disabled,
        }
    }

    /// Returns true if thinking is enabled.
    pub fn is_enabled(&self) -> bool {
        self.r#type == ThinkingType::Enabled
    }
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Anthropic Claude configuration.
#[derive(Clone)]
pub struct AnthropicConfig {
    /// Anthropic API key.
    pub api_key: SensitiveString,
    /// Base URL of the Anthropic API endpoint.
    pub base_url: String,
    /// Model name to use.
    pub model: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Optional system prompt.
    pub system_prompt: Option<String>,
    /// Extended thinking configuration.
    pub thinking: ThinkingConfig,
    /// Tool definitions for function calling.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Tool choice strategy (e.g. "auto", "any", or a specific tool name).
    pub tool_choice: Option<String>,
    /// When true, emit Anthropic `cache_control: {"type":"ephemeral"}` on the system
    /// prompt and the last tool definition, marking the stable prefix for prompt caching
    /// (B1 transparent passthrough). Defaults to off so behavior is unchanged unless opted in.
    pub prompt_caching: bool,
    /// T5 (v0.23.0): an explicit cache TTL for the per-breakpoint `cache_control`
    /// markers. When set (e.g. `"1h"`), every caching breakpoint emits
    /// `{"type":"ttl","ttl":<ttl>}` instead of the ephemeral marker — Anthropic's
    /// long-lived prompt cache, useful for a stable system prefix that outlives the
    /// 5-minute ephemeral window. Applied to the same breakpoints `prompt_caching`
    /// controls (system prompt + last tool definition) so the static prefix is cached
    /// independently of the conversation tail. `None` (default) falls back to
    /// `prompt_caching`'s ephemeral marker.
    pub prompt_cache_ttl: Option<String>,
    /// Provider-agnostic prompt-cache request (see [`lc_core::PromptCache`]).
    ///
    /// Maps onto Anthropic's `cache_control` breakpoints: the stable prefix
    /// (system prompt + last tool definition) is always marked, and the last
    /// message content block is marked when
    /// [`last_message_breakpoint`](lc_core::PromptCache::with_last_message_breakpoint)
    /// is set. [`ttl`](lc_core::PromptCache::with_ttl) selects the long-lived
    /// `{"type":"ephemeral","ttl":…}` marker. `key` has no Anthropic equivalent
    /// and is ignored. `None` (default) keeps the current behavior.
    pub prompt_cache: Option<PromptCache>,
    /// Provider-agnostic raw-JSON-Schema structured-output constraint (see
    /// [`lc_core::JsonSchemaOutput`]).
    ///
    /// Mapped to Anthropic native structured outputs
    /// (`output_config.format: {"type":"json_schema","schema":…}`). Anthropic
    /// has no `name`/`strict` fields, so those are dropped. `None` (default)
    /// sends no constraint.
    pub json_schema: Option<JsonSchemaOutput>,
}

impl std::fmt::Debug for AnthropicConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A13: redact the API key so `{:?}` never leaks the secret.
        f.debug_struct("AnthropicConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("system_prompt", &self.system_prompt)
            .field("thinking", &self.thinking)
            .field("tools", &self.tools)
            .field("tool_choice", &self.tool_choice)
            .field("prompt_caching", &self.prompt_caching)
            .field("prompt_cache_ttl", &self.prompt_cache_ttl)
            .field("prompt_cache", &self.prompt_cache)
            .field("json_schema", &self.json_schema)
            .finish()
    }
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: SensitiveString::default(),
            base_url: ANTHROPIC_BASE_URL.to_string(),
            model: "claude-3-5-sonnet-20241022".to_string(),
            max_tokens: 4096,
            temperature: None,
            system_prompt: None,
            thinking: ThinkingConfig::default(),
            tools: None,
            tool_choice: None,
            prompt_caching: false,
            prompt_cache_ttl: None,
            prompt_cache: None,
            json_schema: None,
        }
    }
}

impl AnthropicConfig {
    /// Creates a new AnthropicConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: SensitiveString::new(api_key),
            ..Default::default()
        }
    }

    /// Creates an AnthropicConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `ANTHROPIC_API_KEY`: API key (required)
    /// - `ANTHROPIC_BASE_URL`: API endpoint (optional)
    /// - `ANTHROPIC_MODEL`: Model name (optional)
    /// - `ANTHROPIC_MAX_TOKENS`: Max tokens (optional)
    pub fn from_env_result() -> Result<Self, ProviderError> {
        let api_key = env::var("ANTHROPIC_API_KEY").map_err(|_| {
            ProviderError::Config("ANTHROPIC_API_KEY environment variable not set".to_string())
        })?;

        let base_url =
            env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| ANTHROPIC_BASE_URL.to_string());

        let model = env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-3-5-sonnet-20241022".to_string());

        let max_tokens = env::var("ANTHROPIC_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);

        Ok(Self {
            api_key: SensitiveString::new(api_key),
            base_url,
            model,
            max_tokens,
            ..Default::default()
        })
    }

    /// Sets the Claude model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets a custom API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the max tokens limit.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Sets the temperature parameter.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets a custom system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Enables extended thinking with the given token budget.
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = thinking;
        self
    }

    /// Enables (`true`) or disables (`false`) explicit prompt caching (B1): the system
    /// prompt and last tool carry a `cache_control` breakpoint so the stable prefix is
    /// cached across turns.
    pub fn with_prompt_caching(mut self, on: bool) -> Self {
        self.prompt_caching = on;
        self.prompt_cache_ttl = None;
        self
    }

    /// T5 (v0.23.0): enables explicit prompt caching with a long-lived TTL on every
    /// caching breakpoint. Emits `cache_control: {"type":"ttl","ttl":<ttl>}` (Anthropic's
    /// long-lived prompt cache) instead of the ephemeral marker, applied to the same
    /// breakpoints as [`Self::with_prompt_caching`] (system prefix + last tool). Use e.g.
    /// `"1h"` for a stable system prompt that should outlive the 5-minute ephemeral window;
    /// the conversation tail is unaffected, letting the static prefix be cached
    /// independently.
    ///
    /// Setting a TTL also turns `prompt_caching` on; call `with_prompt_caching(false)` after
    /// to clear the TTL and stay un-cached.
    pub fn with_prompt_cache_ttl(mut self, ttl: impl Into<String>) -> Self {
        self.prompt_caching = true;
        self.prompt_cache_ttl = Some(ttl.into());
        self
    }

    /// Sets the provider-agnostic prompt-cache request (see [`PromptCache`]).
    ///
    /// Emits `cache_control` breakpoints on the stable prefix (system prompt +
    /// last tool definition), and — when
    /// [`PromptCache::with_last_message_breakpoint`] is on — on the last
    /// message content block too. [`PromptCache::ttl`] yields the long-lived
    /// `{"type":"ephemeral","ttl":…}` marker; [`PromptCache::key`] is ignored
    /// because Anthropic has no `prompt_cache_key`.
    ///
    /// This complements (and takes precedence over) the legacy
    /// [`Self::with_prompt_caching`] / [`Self::with_prompt_cache_ttl`] knobs.
    pub fn with_prompt_cache(mut self, cache: PromptCache) -> Self {
        self.prompt_cache = Some(cache);
        self
    }

    /// Sets a raw-JSON-Schema structured-output constraint from a
    /// [`JsonSchemaOutput`].
    ///
    /// Mapped to Anthropic native structured outputs
    /// (`output_config.format: {"type":"json_schema","schema":…}`). The parsed
    /// JSON arrives in [`LLMResult::content`](lc_core::language_models::LLMResult)
    /// and can be decoded with `serde_json::from_str`. Requires a model / API
    /// version that supports structured outputs; unsupported ones return 4xx.
    pub fn with_json_schema(mut self, output: JsonSchemaOutput) -> Self {
        self.json_schema = Some(output);
        self
    }
}
