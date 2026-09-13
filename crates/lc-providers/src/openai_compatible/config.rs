// lc-providers/src/openai_compatible/config.rs
//! Configuration for the generic OpenAI-compatible chat client (B5, v0.22.4).
//!
//! Every endpoint speaking the OpenAI Chat Completions protocol is reached
//! through one config type: hosted routers ([`GROQ_BASE_URL`],
//! [`OPENROUTER_BASE_URL`], [`XAI_BASE_URL`]) as well as self-hosted/private
//! deployments (vLLM, LM Studio, SGLang, Ollama's `/v1` shim, internal
//! gateways). The model string is always free-form — new provider flagship
//! models need no framework upgrade.

use crate::error::ProviderError;
use crate::openai::OpenAIConfig;
use std::env;

/// Generic endpoint label used when no preset is selected.
pub(crate) const LABEL_GENERIC: &str = "openai-compatible";
/// Groq endpoint label (surfaced in [`ProviderError`](crate::ProviderError)).
pub(crate) const LABEL_GROQ: &str = "groq";
/// OpenRouter endpoint label.
pub(crate) const LABEL_OPENROUTER: &str = "openrouter";
/// xAI endpoint label.
pub(crate) const LABEL_XAI: &str = "xai";

/// Groq OpenAI-compatible endpoint.
pub const GROQ_BASE_URL: &str = "https://api.groq.com/openai/v1";
/// OpenRouter OpenAI-compatible endpoint.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// xAI (Grok) OpenAI-compatible endpoint.
pub const XAI_BASE_URL: &str = "https://api.x.ai/v1";

/// Default model for the Groq preset (their long-lived production ID).
pub const DEFAULT_GROQ_MODEL: &str = "llama-3.3-70b-versatile";
/// Default model for the xAI preset.
pub const DEFAULT_XAI_MODEL: &str = "grok-4";

/// A **non-exhaustive** hint list of Groq production model IDs.
///
/// The `model` field accepts any string the endpoint serves; check the
/// provider's docs for the current list.
pub const GROQ_MODELS: [&str; 6] = [
    "llama-3.3-70b-versatile",                   // general-purpose 70B
    "llama-3.1-8b-instant",                      // fast 8B
    "openai/gpt-oss-120b",                       // open-weight GPT-OSS 120B
    "openai/gpt-oss-20b",                        // open-weight GPT-OSS 20B
    "meta-llama/llama-4-scout-17b-16e-instruct", // Llama 4 Scout
    "moonshotai/kimi-k2-instruct",               // Kimi K2
];

/// A **non-exhaustive** hint list of xAI (Grok) model IDs.
pub const XAI_MODELS: [&str; 4] = [
    "grok-4",      // Grok 4 flagship
    "grok-4-fast", // Grok 4 fast tier
    "grok-3",      // Grok 3
    "grok-3-mini", // Grok 3 mini
];

/// Configuration for a generic OpenAI-compatible chat endpoint.
///
/// Construct either explicitly for a private/self-hosted endpoint
/// ([`OpenAICompatibleConfig::new`], keyless by default — opt in with
/// [`with_api_key`](Self::with_api_key)), or through a hosted preset
/// ([`groq`](Self::groq) / [`openrouter`](Self::openrouter) /
/// [`xai`](Self::xai)).
#[derive(Clone)]
pub struct OpenAICompatibleConfig {
    /// Bearer token. `None` sends no `Authorization` header at all
    /// (keyless local servers).
    pub api_key: Option<String>,
    /// Base URL ending at the `/v1`-style root; `/chat/completions` is appended.
    pub base_url: String,
    /// Free-form model id served by the endpoint.
    pub model: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<usize>,
    /// Nucleus sampling probability mass.
    pub top_p: Option<f32>,
    /// Additional HTTP headers on every request (tenant headers, attribution).
    pub extra_headers: Vec<(String, String)>,
    /// Endpoint label surfaced in errors (preset name, generic by default).
    pub(crate) provider: &'static str,
}

impl std::fmt::Debug for OpenAICompatibleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A13 parity: never leak the token through `{:?}`.
        f.debug_struct("OpenAICompatibleConfig")
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("top_p", &self.top_p)
            .field("extra_headers", &self.extra_headers)
            .field("provider", &self.provider)
            .finish()
    }
}

impl OpenAICompatibleConfig {
    /// Config for a custom (often self-hosted/private) endpoint.
    ///
    /// Keyless by default — many local servers ignore auth, and no
    /// `Authorization` header is sent until
    /// [`with_api_key`](Self::with_api_key) is called.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: None,
            base_url: normalize_base_url(base_url.into()),
            model: model.into(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            extra_headers: Vec::new(),
            provider: LABEL_GENERIC,
        }
    }

    /// Groq preset: `https://api.groq.com/openai/v1`.
    pub fn groq(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(GROQ_BASE_URL, model)
            .with_provider(LABEL_GROQ)
            .with_api_key(api_key)
    }

    /// OpenRouter preset: `https://openrouter.ai/api/v1`.
    ///
    /// `model` is a vendor-qualified id such as `"anthropic/claude-sonnet-4-5"`
    /// or `"openai/gpt-5"`.
    pub fn openrouter(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(OPENROUTER_BASE_URL, model)
            .with_provider(LABEL_OPENROUTER)
            .with_api_key(api_key)
    }

    /// xAI (Grok) preset: `https://api.x.ai/v1`.
    pub fn xai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(XAI_BASE_URL, model)
            .with_provider(LABEL_XAI)
            .with_api_key(api_key)
    }

    /// Generic config from environment variables.
    ///
    /// - `OPENAI_COMPATIBLE_BASE_URL` (required)
    /// - `OPENAI_COMPATIBLE_MODEL` (required)
    /// - `OPENAI_COMPATIBLE_API_KEY` (optional; unset ⇒ keyless local endpoint)
    pub fn from_env_result() -> Result<Self, ProviderError> {
        let base_url = require_env("OPENAI_COMPATIBLE_BASE_URL")?;
        let model = require_env("OPENAI_COMPATIBLE_MODEL")?;
        let mut config = Self::new(base_url, model);
        if let Ok(key) = env::var("OPENAI_COMPATIBLE_API_KEY") {
            config = config.with_api_key(key);
        }
        Ok(config)
    }

    /// Groq preset from `GROQ_API_KEY` (+ optional `GROQ_MODEL`, `GROQ_BASE_URL`).
    pub fn groq_from_env() -> Result<Self, ProviderError> {
        let key = require_env("GROQ_API_KEY")?;
        let base_url = env::var("GROQ_BASE_URL").unwrap_or_else(|_| GROQ_BASE_URL.to_string());
        let model = env::var("GROQ_MODEL").unwrap_or_else(|_| DEFAULT_GROQ_MODEL.to_string());
        Ok(Self::groq(key, model).with_base_url(base_url))
    }

    /// OpenRouter preset from `OPENROUTER_API_KEY` + `OPENROUTER_MODEL`.
    ///
    /// Optional attribution: `OPENROUTER_SITE_URL`, `OPENROUTER_SITE_NAME`.
    pub fn openrouter_from_env() -> Result<Self, ProviderError> {
        let key = require_env("OPENROUTER_API_KEY")?;
        // OpenRouter has no sane default model — vendor/model is the user's choice.
        let model = require_env("OPENROUTER_MODEL")?;
        let mut config = Self::openrouter(key, model);
        if let Ok(site_url) = env::var("OPENROUTER_SITE_URL") {
            if let Ok(site_name) = env::var("OPENROUTER_SITE_NAME") {
                config = config.with_openrouter_attribution(site_url, site_name);
            }
        }
        Ok(config)
    }

    /// xAI preset from `XAI_API_KEY` (+ optional `XAI_MODEL`, `XAI_BASE_URL`).
    pub fn xai_from_env() -> Result<Self, ProviderError> {
        let key = require_env("XAI_API_KEY")?;
        let base_url = env::var("XAI_BASE_URL").unwrap_or_else(|_| XAI_BASE_URL.to_string());
        let model = env::var("XAI_MODEL").unwrap_or_else(|_| DEFAULT_XAI_MODEL.to_string());
        Ok(Self::xai(key, model).with_base_url(base_url))
    }

    /// Sets the bearer token (enables the `Authorization` header).
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Overrides the base URL (preset users can point at a mirror/gateway).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = normalize_base_url(base_url.into());
        self
    }

    /// Sets the model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the temperature parameter.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets the max tokens limit.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Sets the `top_p` parameter.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Appends one extra HTTP header sent on every request.
    pub fn with_extra_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    /// OpenRouter attribution headers (`HTTP-Referer`, `X-Title`) shown on the
    /// router's analytics/leaderboard for this app.
    pub fn with_openrouter_attribution(
        mut self,
        site_url: impl Into<String>,
        site_name: impl Into<String>,
    ) -> Self {
        self.extra_headers
            .push(("HTTP-Referer".to_string(), site_url.into()));
        self.extra_headers
            .push(("X-Title".to_string(), site_name.into()));
        self
    }

    /// Endpoint label used in error messages.
    pub fn provider(&self) -> &str {
        self.provider
    }

    /// Configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Configured model id.
    pub fn model(&self) -> &str {
        &self.model
    }

    fn with_provider(mut self, provider: &'static str) -> Self {
        self.provider = provider;
        self
    }

    pub(crate) fn into_openai_config(self) -> OpenAIConfig {
        // Keyless endpoints must not receive a bare `Bearer ` header.
        let send_auth = self.api_key.is_some();
        OpenAIConfig {
            api_key: self.api_key.unwrap_or_default(),
            base_url: self.base_url,
            model: self.model,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            frequency_penalty: None,
            presence_penalty: None,
            streaming: false,
            organization: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            extra_headers: self.extra_headers,
            send_auth,
        }
    }
}

fn require_env(key: &str) -> Result<String, ProviderError> {
    env::var(key).map_err(|_| ProviderError::Config(format!("{key} environment variable not set")))
}

/// Drops a trailing `/` so the appended `/chat/completions` never doubles it.
fn normalize_base_url(mut url: String) -> String {
    while url.ends_with('/') {
        url.pop();
    }
    url
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ENV_TEST_LOCK;

    fn save_set_restore<'a>(vars: &'a [(&'a str, &'a str)]) -> Vec<(&'a str, Option<String>)> {
        let saved: Vec<_> = vars
            .iter()
            .map(|(k, v)| {
                let old = env::var(k).ok();
                env::set_var(k, v);
                (*k, old)
            })
            .collect();
        saved
    }

    fn restore(saved: Vec<(&str, Option<String>)>) {
        for (k, old) in saved {
            match old {
                Some(v) => env::set_var(k, v),
                None => env::remove_var(k),
            }
        }
    }

    fn remove_env<'a>(keys: &'a [&'a str]) -> Vec<(&'a str, Option<String>)> {
        keys.iter()
            .map(|k| {
                let old = env::var(k).ok();
                env::remove_var(k);
                (*k, old)
            })
            .collect()
    }

    #[test]
    fn new_is_keyless_and_trims_trailing_slash() {
        let config = OpenAICompatibleConfig::new("http://127.0.0.1:8000/v1/", "local-model");
        assert_eq!(config.base_url(), "http://127.0.0.1:8000/v1");
        assert_eq!(config.model(), "local-model");
        assert_eq!(config.provider(), LABEL_GENERIC);
        let converted = config.into_openai_config();
        assert!(!converted.send_auth, "keyless config sends no auth header");
        assert!(converted.extra_headers.is_empty());
    }

    #[test]
    fn api_key_enables_auth_and_extra_headers_are_carried() {
        let config = OpenAICompatibleConfig::new("https://gw.example/v1", "m")
            .with_api_key("sk-x")
            .with_extra_header("X-Tenant", "acme");
        let converted = config.into_openai_config();
        assert!(converted.send_auth);
        assert_eq!(converted.api_key, "sk-x");
        assert_eq!(
            converted.extra_headers,
            vec![("X-Tenant".to_string(), "acme".to_string())]
        );
    }

    #[test]
    fn hosted_presets_pin_endpoint_and_label() {
        let groq = OpenAICompatibleConfig::groq("k", "llama-3.3-70b-versatile");
        assert_eq!(groq.base_url(), GROQ_BASE_URL);
        assert_eq!(groq.provider(), LABEL_GROQ);

        let router = OpenAICompatibleConfig::openrouter("k", "anthropic/claude-sonnet-4-5");
        assert_eq!(router.base_url(), OPENROUTER_BASE_URL);
        assert_eq!(router.provider(), LABEL_OPENROUTER);

        let xai = OpenAICompatibleConfig::xai("k", "grok-4");
        assert_eq!(xai.base_url(), XAI_BASE_URL);
        assert_eq!(xai.provider(), LABEL_XAI);
        assert!(xai.into_openai_config().send_auth);
    }

    #[test]
    fn openrouter_attribution_headers() {
        let config = OpenAICompatibleConfig::openrouter("k", "openai/gpt-5")
            .with_openrouter_attribution("https://app.example", "Example App");
        let names: Vec<_> = config
            .extra_headers
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert!(names.contains(&"HTTP-Referer"));
        assert!(names.contains(&"X-Title"));
    }

    #[test]
    fn from_env_generic_requires_base_url_and_model() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = remove_env(&[
            "OPENAI_COMPATIBLE_BASE_URL",
            "OPENAI_COMPATIBLE_MODEL",
            "OPENAI_COMPATIBLE_API_KEY",
        ]);

        let err = OpenAICompatibleConfig::from_env_result().unwrap_err();
        assert!(err.to_string().contains("OPENAI_COMPATIBLE_BASE_URL"));

        let only_base =
            save_set_restore(&[("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1234/v1")]);
        let err = OpenAICompatibleConfig::from_env_result().unwrap_err();
        assert!(err.to_string().contains("OPENAI_COMPATIBLE_MODEL"));
        restore(only_base);

        // Keyless local endpoint: no API key env ⇒ still valid, no auth header.
        let base_and_model = save_set_restore(&[
            ("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1234/v1/"),
            ("OPENAI_COMPATIBLE_MODEL", "local"),
        ]);
        let config = OpenAICompatibleConfig::from_env_result().unwrap();
        assert_eq!(config.base_url(), "http://localhost:1234/v1");
        assert!(!config.into_openai_config().send_auth);

        // With a key set on top of base/model, auth is enabled.
        let only_key = save_set_restore(&[("OPENAI_COMPATIBLE_API_KEY", "secret")]);
        let config = OpenAICompatibleConfig::from_env_result().unwrap();
        assert!(config.into_openai_config().send_auth);
        restore(only_key);
        restore(base_and_model);

        restore(saved);
    }

    #[test]
    fn groq_from_env_with_defaults_and_base_override() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = remove_env(&["GROQ_API_KEY", "GROQ_MODEL", "GROQ_BASE_URL"]);

        let err = OpenAICompatibleConfig::groq_from_env().unwrap_err();
        assert!(err.to_string().contains("GROQ_API_KEY"));

        let set = save_set_restore(&[
            ("GROQ_API_KEY", "gsk-x"),
            ("GROQ_BASE_URL", "https://groq.mirror.example/v1/"),
        ]);
        let config = OpenAICompatibleConfig::groq_from_env().unwrap();
        assert_eq!(config.model(), DEFAULT_GROQ_MODEL);
        assert_eq!(config.base_url(), "https://groq.mirror.example/v1");
        assert_eq!(config.provider(), LABEL_GROQ);
        restore(set);
        restore(saved);
    }

    #[test]
    fn openrouter_from_env_requires_model_and_reads_attribution() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = remove_env(&[
            "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL",
            "OPENROUTER_SITE_URL",
            "OPENROUTER_SITE_NAME",
        ]);

        let set = save_set_restore(&[
            ("OPENROUTER_API_KEY", "or-x"),
            ("OPENROUTER_MODEL", "x-ai/grok-4"),
            ("OPENROUTER_SITE_URL", "https://app.example"),
            ("OPENROUTER_SITE_NAME", "Example"),
        ]);
        let config = OpenAICompatibleConfig::openrouter_from_env().unwrap();
        assert_eq!(config.model(), "x-ai/grok-4");
        assert_eq!(config.extra_headers.len(), 2);
        restore(set);

        let set = save_set_restore(&[("OPENROUTER_API_KEY", "or-x")]);
        assert!(OpenAICompatibleConfig::openrouter_from_env().is_err());
        restore(set);
        restore(saved);
    }

    #[test]
    fn xai_from_env_defaults_to_grok_4() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = remove_env(&["XAI_API_KEY", "XAI_MODEL", "XAI_BASE_URL"]);
        let set = save_set_restore(&[("XAI_API_KEY", "xai-x")]);
        let config = OpenAICompatibleConfig::xai_from_env().unwrap();
        assert_eq!(config.model(), DEFAULT_XAI_MODEL);
        assert_eq!(config.base_url(), XAI_BASE_URL);
        assert_eq!(config.provider(), LABEL_XAI);
        restore(set);
        restore(saved);
    }

    #[test]
    fn debug_redacts_api_key() {
        let config = OpenAICompatibleConfig::groq("gsk-secret", "m");
        let debug = format!("{config:?}");
        assert!(!debug.contains("gsk-secret"));
        assert!(debug.contains("***"));

        let keyless = OpenAICompatibleConfig::new("http://localhost/v1", "m");
        assert!(format!("{keyless:?}").contains("api_key: None"));
    }
}
