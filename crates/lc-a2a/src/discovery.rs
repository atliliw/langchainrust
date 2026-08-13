//! P2-8: agent discovery.
//!
//! A caller that can't enumerate 1000 agent URLs needs a directory to ask:
//! "who can do X?" [`AgentRegistry`] is a local in-memory catalog of
//! [`AgentCard`]s with skill- and data-boundary-aware lookup.
//! [`RegistryClient`] pulls such a catalog from a remote HTTP registry
//! (`GET /registry.json`).
//!
//! DNS-SD / mDNS is a possible future transport for the same directory; the
//! catalog shape (a `Vec<AgentCard>`) is transport-agnostic so a DNS-SD-backed
//! provider could slot in behind the same lookup helpers.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::protocol::{AgentCard, AgentSkill};

/// Errors raised by the discovery components.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// The agent is not in the registry.
    #[error("agent `{0}` is not registered")]
    UnknownAgent(String),
    /// `register` was called twice with the same URL (use `upsert` to replace).
    #[error("agent `{0}` is already registered")]
    AlreadyRegistered(String),
    /// A remote registry request failed at the transport level.
    #[error("registry request failed: {0}")]
    Http(String),
    /// A remote registry returned an unparseable catalog.
    #[error("registry payload malformed: {0}")]
    Parse(String),
}

/// In-memory agent directory keyed by agent URL (P2-8).
///
/// Register every agent's [`AgentCard`], then discover by skill
/// ([`AgentRegistry::search_skill`]) or data boundary
/// ([`AgentRegistry::filter_data_class`]) instead of hardcoding URLs. All
/// operations are cheap hash lookups behind a short-lived mutex.
#[derive(Debug, Default)]
pub struct AgentRegistry {
    by_url: Mutex<HashMap<String, AgentCard>>,
}

impl AgentRegistry {
    /// An empty directory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an agent. Fails if the URL is already registered — use
    /// [`AgentRegistry::upsert`] to replace a card in place.
    pub fn register(&self, card: AgentCard) -> Result<(), RegistryError> {
        let mut by_url = self.by_url.lock().expect("registry lock poisoned");
        if by_url.contains_key(&card.url) {
            return Err(RegistryError::AlreadyRegistered(card.url));
        }
        by_url.insert(card.url.clone(), card);
        Ok(())
    }

    /// Register or replace an agent card in place (idempotent).
    pub fn upsert(&self, card: AgentCard) {
        self.by_url
            .lock()
            .expect("registry lock poisoned")
            .insert(card.url.clone(), card);
    }

    /// Remove an agent by URL. Returns `true` if it was present.
    pub fn unregister(&self, url: &str) -> bool {
        self.by_url
            .lock()
            .expect("registry lock poisoned")
            .remove(url)
            .is_some()
    }

    /// Look up an agent card by URL.
    pub fn lookup(&self, url: &str) -> Option<AgentCard> {
        self.by_url
            .lock()
            .expect("registry lock poisoned")
            .get(url)
            .cloned()
    }

    /// Every registered agent card, in arbitrary order.
    pub fn agents(&self) -> Vec<AgentCard> {
        self.by_url
            .lock()
            .expect("registry lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    /// Discover agents advertising a skill whose id/name/description contains
    /// `query` (case-insensitive substring match).
    pub fn search_skill(&self, query: &str) -> Vec<AgentCard> {
        let q = query.to_lowercase();
        self.by_url
            .lock()
            .expect("registry lock poisoned")
            .values()
            .filter(|card| card.skills.iter().any(|s| skill_matches(s, &q)))
            .cloned()
            .collect()
    }

    /// Discover agents whose card declares exactly `class` as its data class
    /// (data boundary, P2-8).
    pub fn filter_data_class(&self, class: &str) -> Vec<AgentCard> {
        self.by_url
            .lock()
            .expect("registry lock poisoned")
            .values()
            .filter(|card| card.data_class.as_deref() == Some(class))
            .cloned()
            .collect()
    }

    /// How many agents are registered.
    pub fn len(&self) -> usize {
        self.by_url.lock().expect("registry lock poisoned").len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Client that fetches an agent catalog from a remote HTTP registry (P2-8).
///
/// The registry is expected to serve a JSON array of [`AgentCard`]s at
/// `GET {base}/registry.json`. Discovery helpers (`search_skill`) filter the
/// catalog locally so a remote directory needs zero extra endpoints.
pub struct RegistryClient {
    base_url: String,
    http: reqwest::Client,
}

impl RegistryClient {
    /// A client for a remote registry at `base_url`.
    ///
    /// The client disables proxy usage: registries are typically on a private
    /// network, and an environment proxy must not intercept catalog fetches.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            http: reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("reqwest client builds"),
        }
    }

    /// Fetch the full catalog from the remote registry.
    pub async fn fetch_catalog(&self) -> Result<Vec<AgentCard>, RegistryError> {
        let url = format!("{}/registry.json", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| RegistryError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(RegistryError::Http(format!(
                "registry returned {}",
                resp.status()
            )));
        }
        resp.json::<Vec<AgentCard>>()
            .await
            .map_err(|e| RegistryError::Parse(e.to_string()))
    }

    /// Fetch the catalog and filter to agents advertising `query` as a skill.
    pub async fn search_skill(&self, query: &str) -> Result<Vec<AgentCard>, RegistryError> {
        let q = query.to_lowercase();
        Ok(self
            .fetch_catalog()
            .await?
            .into_iter()
            .filter(|card| card.skills.iter().any(|s| skill_matches(s, &q)))
            .collect())
    }
}

/// Whether a skill's id/name/description matches a lowercased query.
fn skill_matches(skill: &AgentSkill, query: &str) -> bool {
    skill.id.to_lowercase().contains(query)
        || skill.name.to_lowercase().contains(query)
        || skill.description.to_lowercase().contains(query)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn card(url: &str, skill: &str, data_class: Option<&str>) -> AgentCard {
        let mut card = AgentCard::new("agent", "a test agent", url).with_skill(AgentSkill::new(
            skill,
            skill,
            format!("provides {skill}"),
        ));
        if let Some(class) = data_class {
            card = card.with_data_class(class);
        }
        card
    }

    #[test]
    fn registry_register_lookup_unregister() {
        let reg = AgentRegistry::new();
        assert!(reg.is_empty());
        reg.register(card("http://a", "summarize", None)).unwrap();
        assert_eq!(reg.len(), 1);
        assert!(reg.lookup("http://a").is_some());
        assert!(reg.unregister("http://a"));
        assert!(!reg.unregister("http://a"));
        assert!(reg.is_empty());
    }

    #[test]
    fn registry_rejects_duplicate_register_but_upsert_replaces() {
        let reg = AgentRegistry::new();
        reg.register(card("http://a", "summarize", None)).unwrap();
        assert!(matches!(
            reg.register(card("http://a", "translate", None)),
            Err(RegistryError::AlreadyRegistered(_))
        ));
        reg.upsert(card("http://a", "translate", None));
        assert_eq!(reg.agents()[0].skills[0].id, "translate");
    }

    #[test]
    fn registry_searches_skill_case_insensitively() {
        let reg = AgentRegistry::new();
        reg.upsert(card("http://sum", "summarize", None));
        reg.upsert(card("http://translate", "translate", None));

        let hits = reg.search_skill("Summ");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].url, "http://sum");

        assert!(reg.search_skill("nothing").is_empty());
    }

    #[test]
    fn registry_filters_by_data_class() {
        let reg = AgentRegistry::new();
        reg.upsert(card("http://a", "summarize", Some("public")));
        reg.upsert(card("http://b", "translate", Some("confidential")));

        let public = reg.filter_data_class("public");
        assert_eq!(public.len(), 1);
        assert_eq!(public[0].url, "http://a");
        assert!(reg.filter_data_class("internal").is_empty());
    }

    #[tokio::test]
    async fn registry_client_fetches_remote_catalog() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // Serve the same catalog on every connection so a client that issues
        // several fetches (e.g. `search_skill` after `fetch_catalog`) succeeds.
        tokio::spawn(async move {
            let body = serde_json::json!([
                {
                    "name": "summarizer",
                    "description": "summarizes",
                    "url": "http://sum",
                    "skills": [
                        { "id": "summarize", "name": "summarize", "description": "provides summarize" }
                    ],
                    "protocolVersion": "0.3.0",
                    "interfaces": {},
                    "securitySchemes": []
                }
            ])
            .to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
                body.len(),
                body
            );
            while let Ok((mut stream, _)) = listener.accept().await {
                let resp = resp.clone();
                tokio::spawn(async move {
                    // Best-effort drain of the request head: a GET has no body,
                    // so we must not block waiting for bytes that will never
                    // come. A brief timeout lets the write below proceed
                    // regardless of how the OS chunks the small request.
                    let mut chunk = [0u8; 4096];
                    let _ = tokio::time::timeout(
                        std::time::Duration::from_millis(200),
                        stream.read(&mut chunk),
                    )
                    .await;
                    let _ = stream.write_all(resp.as_bytes()).await;
                });
            }
        });

        let client = RegistryClient::new(format!("http://{addr}"));
        let catalog = client.fetch_catalog().await.unwrap();
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog[0].url, "http://sum");

        let hits = client.search_skill("summ").await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "summarizer");
    }
}
