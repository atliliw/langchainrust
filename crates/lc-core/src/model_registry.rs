//! Dynamic model registry (B3, 0.22.4).
//!
//! A [`ModelRegistry`] catalogs models independently of concrete clients:
//! provider slug, model id, context window, output cap, feature flags and
//! [`ModelPrice`]. Routers ([`crate::router_llm::RouterLLM`]) read prices from
//! a shared registry; callers can `register` custom/self-hosted models or
//! [`ModelRegistry::fetch`] a JSON catalog over HTTP, so price/capability
//! updates need no library release.
//!
//! Remote catalog JSON shape:
//!
//! ```json
//! {
//!   "version": 1,
//!   "models": [
//!     {
//!       "provider": "openai",
//!       "id": "gpt-4o-mini",
//!       "context_window": 128000,
//!       "max_output_tokens": 16384,
//!       "capabilities": {
//!         "tools": true,
//!         "vision": false,
//!         "json_mode": true,
//!         "reasoning": false,
//!         "audio": false
//!       },
//!       "price": { "input_per_1k": 0.15, "output_per_1k": 0.60 }
//!     }
//!   ]
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::cost::{CostError, ModelPrice};

/// Feature flags advertised by a model.
///
/// All fields default to `false` when omitted from a remote payload, so newly
/// added capabilities parse safely against older catalogs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Native function/tool calling.
    #[serde(default)]
    pub tools: bool,
    /// Image (vision) inputs.
    #[serde(default)]
    pub vision: bool,
    /// Strict JSON / structured output mode.
    #[serde(default)]
    pub json_mode: bool,
    /// Reasoning/thinking model.
    #[serde(default)]
    pub reasoning: bool,
    /// Audio input/output.
    #[serde(default)]
    pub audio: bool,
}

/// One catalog entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Provider slug (`"openai"`, `"anthropic"`, `"groq"`, `"custom"` ...).
    pub provider: String,
    /// Model id as used in API calls.
    pub id: String,
    /// Maximum context window in tokens.
    pub context_window: usize,
    /// Maximum output tokens the model accepts; `None` when undeclared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
    /// Capability flags.
    #[serde(default)]
    pub capabilities: ModelCapabilities,
    /// USD price per 1K tokens.
    pub price: ModelPrice,
}

impl ModelInfo {
    /// Creates an entry with the given essentials and no capability flags.
    pub fn new(
        provider: impl Into<String>,
        id: impl Into<String>,
        context_window: usize,
        price: ModelPrice,
    ) -> Self {
        Self {
            provider: provider.into(),
            id: id.into(),
            context_window,
            max_output_tokens: None,
            capabilities: ModelCapabilities::default(),
            price,
        }
    }

    /// Builder-style setter for the output cap.
    pub fn with_max_output(mut self, max_output_tokens: usize) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    /// Builder-style setter for capability flags.
    pub fn with_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// The canonical `"<provider>/<id>"` key.
    pub fn key(&self) -> String {
        format!("{}/{}", self.provider, self.id)
    }
}

/// Wire format of a remote catalog.
#[derive(Debug, Clone, Deserialize)]
struct CatalogEnvelope {
    #[serde(default)]
    #[allow(dead_code)]
    version: Option<u32>,
    models: Vec<ModelInfo>,
}

/// Provider/model catalog. Cheap to clone (entries shared via `Arc`).
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    models: HashMap<String, Arc<ModelInfo>>,
}

impl ModelRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds/overwrites one entry.
    pub fn register(&mut self, info: ModelInfo) {
        self.models.insert(info.key(), Arc::new(info));
    }

    /// Builder-style entry registration.
    pub fn with(mut self, info: ModelInfo) -> Self {
        self.register(info);
        self
    }

    /// Merges every entry of `other` into this registry (other wins on key
    /// collision — a remote catalog can therefore override built-in prices).
    pub fn merge(&mut self, other: ModelRegistry) {
        for (key, info) in other.models {
            self.models.insert(key, info);
        }
    }

    /// Lookup by provider + model id.
    pub fn get(&self, provider: &str, model: &str) -> Option<&Arc<ModelInfo>> {
        self.models.get(&format!("{provider}/{model}"))
    }

    /// Lookup by canonical `"<provider>/<id>"` key.
    pub fn get_by_key(&self, key: &str) -> Option<&Arc<ModelInfo>> {
        self.models.get(key)
    }

    /// All registered entries (arbitrary order).
    pub fn models(&self) -> impl Iterator<Item = &Arc<ModelInfo>> {
        self.models.values()
    }

    /// Entries of one provider.
    pub fn models_of<'a>(
        &'a self,
        provider: &'a str,
    ) -> impl Iterator<Item = &'a Arc<ModelInfo>> + 'a {
        self.models.values().filter(move |m| m.provider == provider)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Parses a remote/JSON catalog (`{"version":..,"models":[...]}` or a bare
    /// `[ModelInfo]` array).
    pub fn from_json(json: &str) -> Result<Self, CostError> {
        // Accept both the enveloped object and a bare array.
        if let Ok(envelope) = serde_json::from_str::<CatalogEnvelope>(json) {
            return Ok(Self::from_iter(envelope.models));
        }
        let models: Vec<ModelInfo> = serde_json::from_str(json)
            .map_err(|e| CostError::Payload(format!("catalog JSON parse failed: {e}")))?;
        Ok(Self::from_iter(models))
    }

    /// Serializes the registry to the enveloped catalog JSON.
    pub fn to_json(&self) -> Result<String, CostError> {
        let models: Vec<&ModelInfo> = self.models.values().map(AsRef::as_ref).collect();
        Ok(serde_json::json!({ "version": 1, "models": models }).to_string())
    }

    /// Downloads and parses a remote catalog with a fresh blocking-capable
    /// `reqwest` client.
    pub async fn fetch(url: &str) -> Result<Self, CostError> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| CostError::Fetch(e.to_string()))?;
        Self::fetch_with_client(&client, url).await
    }

    /// Downloads and parses a remote catalog with a caller-provided client
    /// (shared connection pool, custom timeouts/proxies/auth headers).
    pub async fn fetch_with_client(client: &reqwest::Client, url: &str) -> Result<Self, CostError> {
        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| CostError::Fetch(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(CostError::Fetch(format!(
                "catalog GET {url} returned HTTP {status}"
            )));
        }
        let body = resp
            .text()
            .await
            .map_err(|e| CostError::Fetch(e.to_string()))?;
        Self::from_json(&body)
    }

    fn from_iter(iter: impl IntoIterator<Item = ModelInfo>) -> Self {
        let mut registry = Self::new();
        for info in iter {
            registry.register(info);
        }
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ModelInfo {
        ModelInfo::new("openai", "gpt-x", 128_000, ModelPrice::new(1.0, 4.0))
            .with_max_output(16_384)
            .with_capabilities(ModelCapabilities {
                tools: true,
                vision: false,
                json_mode: true,
                reasoning: false,
                audio: false,
            })
    }

    #[test]
    fn register_and_lookup() {
        let registry = ModelRegistry::new().with(sample());
        assert_eq!(registry.len(), 1);
        let info = registry.get("openai", "gpt-x").expect("registered");
        assert_eq!(info.context_window, 128_000);
        assert_eq!(info.max_output_tokens, Some(16_384));
        assert!(info.capabilities.tools);
        assert!(registry.get_by_key("openai/gpt-x").is_some());
        assert!(registry.get("anthropic", "gpt-x").is_none());
        assert_eq!(registry.models_of("openai").count(), 1);
        assert_eq!(registry.models_of("google").count(), 0);
    }

    #[test]
    fn merge_other_wins_on_collision() {
        let mut base =
            ModelRegistry::new().with(ModelInfo::new("p", "m", 1000, ModelPrice::new(1.0, 1.0)));
        let newer =
            ModelRegistry::new().with(ModelInfo::new("p", "m", 2000, ModelPrice::new(2.0, 2.0)));
        base.merge(newer);
        assert_eq!(base.get("p", "m").unwrap().context_window, 2000);
    }

    #[test]
    fn parses_enveloped_json_with_defaults() {
        let json = serde_json::json!({
            "version": 1,
            "models": [
                {
                    "provider": "custom",
                    "id": "local-llm",
                    "context_window": 32768,
                    "price": { "input_per_1k": 0.0, "output_per_1k": 0.0 }
                }
            ]
        })
        .to_string();
        let registry = ModelRegistry::from_json(&json).unwrap();
        let info = registry.get("custom", "local-llm").unwrap();
        assert_eq!(info.context_window, 32768);
        assert_eq!(info.max_output_tokens, None);
        assert!(!info.capabilities.tools);
        assert_eq!(info.price, ModelPrice::free());
    }

    #[test]
    fn parses_bare_array_json() {
        let json = serde_json::json!([
            {
                "provider": "groq",
                "id": "llama-x",
                "context_window": 131072,
                "capabilities": { "tools": true },
                "price": { "input_per_1k": 0.59, "output_per_1k": 0.79 }
            }
        ])
        .to_string();
        let registry = ModelRegistry::from_json(&json).unwrap();
        assert_eq!(registry.len(), 1);
        assert!(registry.get("groq", "llama-x").unwrap().capabilities.tools);
    }

    #[test]
    fn invalid_json_is_payload_error() {
        let err = ModelRegistry::from_json("{not json").unwrap_err();
        assert!(matches!(err, CostError::Payload(_)));
    }

    #[test]
    fn round_trips_through_json() {
        let registry = ModelRegistry::new().with(sample());
        let json = registry.to_json().unwrap();
        let parsed = ModelRegistry::from_json(&json).unwrap();
        assert_eq!(
            parsed.get("openai", "gpt-x").unwrap().as_ref(),
            registry.get("openai", "gpt-x").unwrap().as_ref()
        );
    }

    #[tokio::test]
    async fn fetch_reports_http_errors_as_fetch_error() {
        // Unroutable port → connection error, surfaced as Fetch (not a panic).
        let err = ModelRegistry::fetch("http://127.0.0.1:1/catalog.json").await;
        assert!(matches!(err, Err(CostError::Fetch(_))));
    }
}
