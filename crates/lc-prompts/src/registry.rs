//! Versioned prompt registry.
//!
//! [`PromptRegistry`] stores immutable prompt versions under slash-separated **namespaces**
//! (e.g. `"customer/support/zh"`). Every registration is content-addressed with a SHA-256
//! hash and receives a monotonic 1-based **version number**. Stored versions are never
//! mutated; a rollback is implemented as a *new* version that re-publishes historical
//! content, so audit trails (and concurrent consumers pinning an old number) stay intact.
//!
//! ```
//! use lc_prompts::PromptRegistry;
//!
//! let registry = PromptRegistry::new();
//! let v1 = registry.register("greeting", "Hello, {name}!").unwrap();
//! let v2 = registry.register("greeting", "Hi {name}, welcome!").unwrap();
//! assert_eq!(v1.version, 1);
//! assert_eq!(v2.version, 2);
//!
//! // pin an exact version, follow latest, or address by content hash
//! let pinned = registry.resolve("greeting", &1.into()).unwrap();
//! let latest = registry.get("greeting").unwrap();
//! let by_hash = registry.resolve("greeting", &v1.hash.clone().into()).unwrap();
//! assert_eq!(pinned.template, "Hello, {name}!");
//! assert_eq!(latest.template, "Hi {name}, welcome!");
//! assert_eq!(by_hash.version, 1);
//!
//! // a rollback re-publishes the old content as a new version
//! let v3 = registry.rollback("greeting", 1).unwrap();
//! assert_eq!(v3.version, 3);
//! assert_eq!(registry.get("greeting").unwrap().template, "Hello, {name}!");
//! ```

use std::collections::BTreeMap;
use std::sync::RwLock;

use sha2::{Digest, Sha256};

use crate::error::PromptsError;
use crate::prompt_template::PromptTemplate;

/// Minimum accepted length for a hash-prefix lookup. Shorter prefixes are rejected
/// outright to keep accidental ambiguity rare.
pub const MIN_HASH_PREFIX: usize = 7;

/// Selector for one registered prompt version.
///
/// Conversions are provided: `u32` ([`VersionSpec::Number`]), `&str`/`String`
/// ([`VersionSpec::Hash`]), and [`VersionSpec::latest`](Self::latest).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionSpec {
    /// The most recently registered version.
    Latest,
    /// An exact 1-based version number.
    Number(u32),
    /// A full SHA-256 hex hash or a unique prefix of at least
    /// [`MIN_HASH_PREFIX`] characters.
    Hash(String),
}

impl VersionSpec {
    /// Convenience constructor for [`VersionSpec::Latest`].
    pub fn latest() -> Self {
        VersionSpec::Latest
    }
}

impl From<u32> for VersionSpec {
    fn from(version: u32) -> Self {
        VersionSpec::Number(version)
    }
}

impl From<&str> for VersionSpec {
    fn from(hash: &str) -> Self {
        VersionSpec::Hash(hash.to_string())
    }
}

impl From<String> for VersionSpec {
    fn from(hash: String) -> Self {
        VersionSpec::Hash(hash)
    }
}

/// One immutable, registered prompt version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredPrompt {
    /// Namespace the prompt was registered under.
    pub namespace: String,
    /// Monotonic 1-based version number within the namespace.
    pub version: u32,
    /// SHA-256 hex hash of the exact template bytes.
    pub hash: String,
    /// Raw template text.
    pub template: String,
    /// Variable names parsed from the template at registration time.
    pub variables: Vec<String>,
}

impl RegisteredPrompt {
    /// Builds a fresh [`PromptTemplate`] from the stored text, ready for
    /// [`PromptTemplate::format`] or piping into an LCEL chain.
    pub fn to_template(&self) -> PromptTemplate {
        PromptTemplate::new(self.template.clone())
    }
}

/// Summary row of one stored version (for listings / audit UIs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptVersionInfo {
    /// Version number.
    pub version: u32,
    /// Content hash of that version.
    pub hash: String,
}

#[derive(Debug, Default)]
struct Namespace {
    /// version number -> record (BTreeMap keeps versions ordered).
    versions: BTreeMap<u32, RegisteredPrompt>,
    /// full content hash -> every version carrying that content
    /// (a rollback legitimately re-publishes an older hash).
    hashes: BTreeMap<String, Vec<u32>>,
}

/// Thread-safe, in-memory registry of versioned prompts.
///
/// Cheap to clone-contained-share: wrap in `Arc` when several components need the same
/// registry. All operations take `&self`; a `RwLock` guards the store.
#[derive(Debug, Default)]
pub struct PromptRegistry {
    namespaces: RwLock<BTreeMap<String, Namespace>>,
}

impl PromptRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new prompt version under `namespace`.
    ///
    /// Registration is **idempotent on identical consecutive content**: re-registering the
    /// current latest text returns the existing record instead of bumping the version
    /// (nothing changed). Distinct content always creates a new monotonic version;
    /// historical content can legitimately recur later via [`rollback`](Self::rollback).
    ///
    /// # Errors
    /// Returns [`PromptsError::InvalidNamespace`] if the namespace is malformed.
    pub fn register(
        &self,
        namespace: &str,
        template: impl Into<String>,
    ) -> Result<RegisteredPrompt, PromptsError> {
        validate_namespace(namespace)?;
        let template = template.into();
        let hash = content_hash(&template);
        let variables = PromptTemplate::new(&template).variables();

        let mut store = self.namespaces.write().unwrap_or_else(|e| e.into_inner());
        let entry = store.entry(namespace.to_string()).or_default();

        // consecutive no-op publish: same content as current latest
        if let Some((_, latest)) = entry.versions.last_key_value() {
            if latest.hash == hash {
                return Ok(latest.clone());
            }
        }
        let version = entry
            .versions
            .last_key_value()
            .map(|(v, _)| v + 1)
            .unwrap_or(1);
        let record = RegisteredPrompt {
            namespace: namespace.to_string(),
            version,
            hash: hash.clone(),
            template,
            variables,
        };
        entry.hashes.entry(hash).or_default().push(version);
        entry.versions.insert(version, record.clone());
        Ok(record)
    }

    /// Resolves one prompt version.
    ///
    /// Hash addressing matches the full hash or a unique prefix (≥
    /// [`MIN_HASH_PREFIX`]); historical content re-published by a rollback resolves to its
    /// newest version (the content is identical).
    ///
    /// # Errors
    /// - [`PromptsError::NamespaceNotFound`] when the namespace has no versions;
    /// - [`PromptsError::VersionNotFound`] for a missing exact number;
    /// - [`PromptsError::HashNotFound`] / [`PromptsError::AmbiguousHash`] for hash selectors.
    pub fn resolve(
        &self,
        namespace: &str,
        spec: &VersionSpec,
    ) -> Result<RegisteredPrompt, PromptsError> {
        let store = self.namespaces.read().unwrap_or_else(|e| e.into_inner());
        let entry = store
            .get(namespace)
            .ok_or_else(|| PromptsError::NamespaceNotFound(namespace.to_string()))?;
        let version = match spec {
            VersionSpec::Latest => *entry
                .versions
                .last_key_value()
                .map(|(v, _)| v)
                .expect("namespace always has at least one version"),
            VersionSpec::Number(n) => {
                if entry.versions.contains_key(n) {
                    *n
                } else {
                    return Err(PromptsError::VersionNotFound {
                        namespace: namespace.to_string(),
                        version: *n,
                    });
                }
            }
            VersionSpec::Hash(prefix) => resolve_hash_version(entry, prefix)?,
        };
        Ok(entry.versions[&version].clone())
    }

    /// Resolves the current latest version (see [`resolve`](Self::resolve)).
    pub fn get(&self, namespace: &str) -> Result<RegisteredPrompt, PromptsError> {
        self.resolve(namespace, &VersionSpec::Latest)
    }

    /// Parses and resolves a textual prompt reference.
    ///
    /// Accepted forms:
    /// - `"namespace"` — current latest;
    /// - `"namespace@latest"`;
    /// - `"namespace@3"` — exact version number;
    /// - `"namespace@hash:abcdef0"` (or a bare `"namespace@abcdef0"`) — unique hash prefix.
    pub fn resolve_ref(&self, reference: &str) -> Result<RegisteredPrompt, PromptsError> {
        let (namespace, spec) = parse_reference(reference)?;
        self.resolve(&namespace, &spec)
    }

    /// Fetches a reference (see [`resolve_ref`](Self::resolve_ref)) and returns a ready-to-run
    /// [`PromptTemplate`] — the "pull template" entry point for call sites that only need to
    /// render prompts.
    pub fn fetch(&self, reference: &str) -> Result<PromptTemplate, PromptsError> {
        Ok(self.resolve_ref(reference)?.to_template())
    }

    /// Rolls the namespace back to historical content by re-publishing it as a new version.
    ///
    /// Versions are immutable, so this never overwrites anything: if the target already is
    /// the current latest content, its record is returned unchanged; otherwise a fresh
    /// version carrying the target's text is appended and becomes latest.
    ///
    /// # Errors
    /// Propagates the same errors as [`resolve`](Self::resolve).
    pub fn rollback(
        &self,
        namespace: &str,
        target_version: u32,
    ) -> Result<RegisteredPrompt, PromptsError> {
        let target = self.resolve(namespace, &VersionSpec::Number(target_version))?;
        // publish even though the hash may already exist in history; `register`'s
        // consecutive-duplicate guard returns latest directly if nothing would change.
        self.register(namespace, target.template)
    }

    /// Lists all stored versions of a namespace in registration order.
    ///
    /// # Errors
    /// [`PromptsError::NamespaceNotFound`] when nothing was registered there.
    pub fn history(&self, namespace: &str) -> Result<Vec<PromptVersionInfo>, PromptsError> {
        let store = self.namespaces.read().unwrap_or_else(|e| e.into_inner());
        let entry = store
            .get(namespace)
            .ok_or_else(|| PromptsError::NamespaceNotFound(namespace.to_string()))?;
        Ok(entry
            .versions
            .values()
            .map(|r| PromptVersionInfo {
                version: r.version,
                hash: r.hash.clone(),
            })
            .collect())
    }

    /// Lists every namespace that currently has at least one version, sorted alphabetically.
    pub fn namespaces(&self) -> Vec<String> {
        let store = self.namespaces.read().unwrap_or_else(|e| e.into_inner());
        store.keys().cloned().collect()
    }

    /// Number of namespaces currently held.
    pub fn len(&self) -> usize {
        self.namespaces
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Whether the registry holds no namespaces at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Test-only seam: inserts a record with an arbitrary hash so hash-prefix ambiguity can
    /// be exercised deterministically without brute-forcing SHA-256.
    #[cfg(test)]
    fn insert_test_record(&self, namespace: &str, version: u32, hash: &str, template: &str) {
        let mut store = self.namespaces.write().unwrap_or_else(|e| e.into_inner());
        let entry = store.entry(namespace.to_string()).or_default();
        entry
            .hashes
            .entry(hash.to_string())
            .or_default()
            .push(version);
        entry.versions.insert(
            version,
            RegisteredPrompt {
                namespace: namespace.to_string(),
                version,
                hash: hash.to_string(),
                template: template.to_string(),
                variables: PromptTemplate::new(template).variables(),
            },
        );
    }
}

/// SHA-256 hex of the exact template bytes.
fn content_hash(template: &str) -> String {
    use std::fmt::Write;
    let mut hasher = Sha256::new();
    hasher.update(template.as_bytes());
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Validates `team/name` style namespaces: slash-separated non-empty segments, each
/// limited to `[A-Za-z0-9._-]`.
fn validate_namespace(namespace: &str) -> Result<(), PromptsError> {
    let valid = !namespace.is_empty()
        && namespace.len() <= 128
        && namespace
            .split('/')
            .all(|segment| !segment.is_empty() && segment.chars().all(is_namespace_char));
    if valid {
        Ok(())
    } else {
        Err(PromptsError::InvalidNamespace(namespace.to_string()))
    }
}

fn is_namespace_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')
}

/// Resolves a hash/full-or-prefix selector inside one namespace, returning the newest
/// matching version when a hash was re-published (identical content).
fn resolve_hash_version(entry: &Namespace, prefix: &str) -> Result<u32, PromptsError> {
    if prefix.len() < MIN_HASH_PREFIX {
        return Err(PromptsError::HashNotFound(prefix.to_string()));
    }
    let mut exact: Option<u32> = None;
    let mut prefix_matches: Vec<u32> = Vec::new();
    for (hash, versions) in &entry.hashes {
        if hash == prefix {
            exact = Some(*versions.last().expect("hash index never empty"));
        } else if hash.starts_with(prefix) {
            prefix_matches.push(*versions.last().expect("hash index never empty"));
        }
    }
    if let Some(version) = exact {
        return Ok(version);
    }
    match prefix_matches.len() {
        0 => Err(PromptsError::HashNotFound(prefix.to_string())),
        1 => Ok(prefix_matches[0]),
        _ => {
            prefix_matches.sort_unstable();
            Err(PromptsError::AmbiguousHash {
                prefix: prefix.to_string(),
                versions: prefix_matches,
            })
        }
    }
}

/// Splits a `"namespace@spec"` reference into a namespace and a [`VersionSpec`].
fn parse_reference(reference: &str) -> Result<(String, VersionSpec), PromptsError> {
    let (namespace, raw_spec) = match reference.rsplit_once('@') {
        Some((ns, spec)) if !ns.is_empty() && !spec.is_empty() => (ns, Some(spec)),
        // no '@' at all -> whole string is the namespace, latest implied
        None if !reference.is_empty() && !reference.contains('@') => (reference, None),
        _ => return Err(PromptsError::InvalidReference(reference.to_string())),
    };
    validate_namespace(namespace)?;
    let spec = match raw_spec {
        None | Some("latest") => VersionSpec::Latest,
        Some(raw) => {
            if let Ok(number) = raw.parse::<u32>() {
                VersionSpec::Number(number)
            } else if let Some(hex) = raw.strip_prefix("hash:") {
                VersionSpec::Hash(hex.to_string())
            } else if !raw.is_empty() && raw.chars().all(|c| c.is_ascii_hexdigit()) {
                VersionSpec::Hash(raw.to_string())
            } else {
                return Err(PromptsError::InvalidReference(reference.to_string()));
            }
        }
    };
    Ok((namespace.to_string(), spec))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_sequential_versions_and_resolves_latest() {
        let registry = PromptRegistry::new();
        let v1 = registry.register("greeting", "Hello, {name}!").unwrap();
        let v2 = registry
            .register("greeting", "Hi {name}, welcome!")
            .unwrap();

        assert_eq!(v1.version, 1);
        assert_eq!(v1.hash.len(), 64);
        assert_eq!(v1.variables, vec!["name"]);
        assert_eq!(v2.version, 2);
        assert_ne!(v1.hash, v2.hash);

        let latest = registry.get("greeting").unwrap();
        assert_eq!(latest.version, 2);
        assert_eq!(latest.template, "Hi {name}, welcome!");

        assert_eq!(registry.len(), 1);
        let history = registry.history("greeting").unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].version, 1);
        assert_eq!(history[1].version, 2);
    }

    #[test]
    fn re_registering_identical_consecutive_content_is_noop() {
        let registry = PromptRegistry::new();
        let v1 = registry.register("greeting", "Hello, {name}!").unwrap();
        let again = registry.register("greeting", "Hello, {name}!").unwrap();

        assert_eq!(again.version, 1);
        assert_eq!(again.hash, v1.hash);
        assert_eq!(registry.history("greeting").unwrap().len(), 1);
    }

    #[test]
    fn same_content_under_different_namespaces_shares_hash_independently() {
        let registry = PromptRegistry::new();
        let a = registry.register("team-a/greet", "Hello, {name}!").unwrap();
        let b = registry.register("team-b/greet", "Hello, {name}!").unwrap();

        // identical bytes -> identical content hash across namespaces ...
        assert_eq!(a.hash, b.hash);
        // ... but the namespaces stay independent version lines
        assert_eq!(a.namespace, "team-a/greet");
        assert_eq!(b.namespace, "team-b/greet");
        assert_eq!(registry.namespaces(), vec!["team-a/greet", "team-b/greet"]);
        assert_eq!(
            registry
                .resolve("team-a/greet", &a.hash.clone().into())
                .unwrap()
                .namespace,
            "team-a/greet"
        );
    }

    #[test]
    fn resolves_by_number_and_reports_missing_version() {
        let registry = PromptRegistry::new();
        registry.register("greeting", "v1").unwrap();
        registry.register("greeting", "v2").unwrap();

        let pinned = registry.resolve("greeting", &1.into()).unwrap();
        assert_eq!(pinned.template, "v1");

        let err = registry.resolve("greeting", &9.into()).unwrap_err();
        assert!(matches!(
            err,
            PromptsError::VersionNotFound { version: 9, .. }
        ));

        let err = registry.get("missing").unwrap_err();
        assert!(matches!(err, PromptsError::NamespaceNotFound(_)));
    }

    #[test]
    fn resolves_by_full_hash_and_unique_prefix() {
        let registry = PromptRegistry::new();
        let v1 = registry.register("greeting", "v1 text").unwrap();
        registry.register("greeting", "v2 text").unwrap();

        let by_full = registry
            .resolve("greeting", &VersionSpec::Hash(v1.hash.clone()))
            .unwrap();
        assert_eq!(by_full.version, 1);

        let by_prefix = registry
            .resolve("greeting", &v1.hash[..MIN_HASH_PREFIX].into())
            .unwrap();
        assert_eq!(by_prefix.version, 1);

        let err = registry.resolve("greeting", &"abc".into()).unwrap_err();
        assert!(matches!(err, PromptsError::HashNotFound(_)));
        let err = registry
            .resolve("greeting", &"deadbeefdeadbeef".into())
            .unwrap_err();
        assert!(matches!(err, PromptsError::HashNotFound(_)));
    }

    #[test]
    fn ambiguous_hash_prefix_lists_matching_versions() {
        let registry = PromptRegistry::new();
        // two distinct contents forced to share a hash prefix via the test seam
        registry.insert_test_record(
            "ns",
            1,
            "abcdef0111111111111111111111111111111111111111111111111111111111",
            "one",
        );
        registry.insert_test_record(
            "ns",
            2,
            "abcdef0222222222222222222222222222222222222222222222222222222222",
            "two",
        );

        let err = registry.resolve("ns", &"abcdef0".into()).unwrap_err();
        match err {
            PromptsError::AmbiguousHash { versions, .. } => assert_eq!(versions, vec![1, 2]),
            other => panic!("expected AmbiguousHash, got {other:?}"),
        }
        // a longer prefix disambiguates
        let resolved = registry.resolve("ns", &"abcdef01".into()).unwrap();
        assert_eq!(resolved.version, 1);
    }

    #[test]
    fn rollback_appends_old_content_as_new_version() {
        let registry = PromptRegistry::new();
        let v1 = registry.register("greeting", "v1").unwrap();
        registry.register("greeting", "v2").unwrap();

        let v3 = registry.rollback("greeting", 1).unwrap();
        assert_eq!(v3.version, 3);
        assert_eq!(v3.template, "v1");
        assert_eq!(v3.hash, v1.hash);
        assert_eq!(registry.get("greeting").unwrap().version, 3);

        // history is append-only: v1 and v2 are still addressable
        assert_eq!(
            registry.resolve("greeting", &2.into()).unwrap().template,
            "v2"
        );
        // rolling back to content that already is latest is a no-op
        let again = registry.rollback("greeting", 3).unwrap();
        assert_eq!(again.version, 3);
        assert_eq!(registry.history("greeting").unwrap().len(), 3);

        let err = registry.rollback("greeting", 99).unwrap_err();
        assert!(matches!(err, PromptsError::VersionNotFound { .. }));
    }

    #[test]
    fn parses_references_and_fetch_renders_template() {
        let registry = PromptRegistry::new();
        registry
            .register("customer/support", "Hello, {name}!")
            .unwrap();
        registry.register("customer/support", "Hi {name}!").unwrap();

        let mut vars = std::collections::HashMap::new();
        vars.insert("name", "Sam");

        let template = registry.fetch("customer/support").unwrap();
        assert_eq!(template.format(&vars).unwrap(), "Hi Sam!");

        let template = registry.fetch("customer/support@latest").unwrap();
        assert_eq!(template.format(&vars).unwrap(), "Hi Sam!");

        let pinned = registry.resolve_ref("customer/support@1").unwrap();
        assert_eq!(pinned.version, 1);
        assert_eq!(pinned.to_template().format(&vars).unwrap(), "Hello, Sam!");

        let hash_ref = format!("customer/support@{}", &pinned.hash[..12]);
        assert_eq!(registry.resolve_ref(&hash_ref).unwrap().version, 1);
        let hash_kw_ref = format!("customer/support@hash:{}", &pinned.hash[..12]);
        assert_eq!(registry.resolve_ref(&hash_kw_ref).unwrap().version, 1);

        for bad in ["", "@1", "ns@", "ns@v2", "bad namespace@1"] {
            assert!(
                registry.resolve_ref(bad).is_err(),
                "expected error for {bad:?}"
            );
        }
    }

    #[test]
    fn validates_namespaces() {
        let registry = PromptRegistry::new();
        for good in ["a", "team/greet", "team-a/greet_b.v2/zh"] {
            assert!(registry.register(good, "x").is_ok(), "should accept {good}");
        }
        for bad in ["", "/a", "a/", "a//b", "a b", "a@b"] {
            assert!(registry.register(bad, "x").is_err(), "should reject {bad}");
        }
    }
}
