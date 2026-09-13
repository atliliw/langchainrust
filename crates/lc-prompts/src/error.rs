// crates/lc-prompts/src/error.rs
//! Error types for prompt templates.

/// Error type for prompt template operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PromptsError {
    /// A template referenced a variable that was not provided.
    #[error("Missing variable: {0}")]
    MissingVariable(String),

    /// A namespace did not satisfy the naming rules
    /// (`namespace/sub-namespace`, segments of `[A-Za-z0-9._-]`).
    #[error("invalid prompt namespace {0:?}: use slash-separated segments of [A-Za-z0-9._-]")]
    InvalidNamespace(String),

    /// No template was ever registered under the namespace.
    #[error("prompt namespace not found: {0}")]
    NamespaceNotFound(String),

    /// A specific version number does not exist under the namespace.
    #[error("version {version} not found in namespace {namespace:?}")]
    VersionNotFound {
        /// Namespace that was queried.
        namespace: String,
        /// Version number that was requested.
        version: u32,
    },

    /// No stored version has a content hash matching the prefix.
    #[error("no prompt version with hash prefix {0:?}")]
    HashNotFound(String),

    /// More than one stored version shares the hash prefix. Content is identical across them,
    /// but callers must lengthen the prefix to disambiguate the version.
    #[error("ambiguous hash prefix {prefix:?}: versions {versions:?}")]
    AmbiguousHash {
        /// Hash prefix that matched multiple versions.
        prefix: String,
        /// Versions that matched.
        versions: Vec<u32>,
    },

    /// A `namespace@spec` reference string could not be parsed.
    #[error("invalid prompt reference {0:?}: expected namespace, namespace@<version>, or namespace@hash:<hex>")]
    InvalidReference(String),
}
