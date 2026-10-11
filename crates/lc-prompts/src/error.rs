// crates/lc-prompts/src/error.rs
//! Error types for prompt templates.

/// Error type for prompt template operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PromptsError {
    /// A template referenced a variable that was not provided.
    #[error("Missing variable: {0}")]
    MissingVariable(String),

    /// A template contained a malformed placeholder — an opening `{` with no
    /// matching `}`, or a name inside `{...}` that is not a valid variable
    /// name. Reported by the strict
    /// [`validate`](crate::PromptTemplate::validate) so a template typo fails
    /// loudly instead of rendering `{` literally.
    #[error("malformed placeholder in template {0:?}")]
    MalformedTemplate(String),

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

    /// A chat-template message failed to format.
    ///
    /// The underlying error is kept as the
    /// [`source`](std::error::Error::source), and the message's position is
    /// carried in fields. The previous shape flattened both into one string and
    /// wrapped it in [`MissingVariable`](Self::MissingVariable), so that
    /// variant's payload was a prose sentence rather than a variable name —
    /// callers matching on the variant got a name they could not use, and the
    /// real cause was one string-parse away.
    #[error("failed to format {role} message {index}: {source}")]
    MessageFormat {
        /// Zero-based index of the message in the template.
        index: usize,
        /// Role of the message (`"System"`, `"Human"`, `"AI"`, `"Tool"`).
        role: &'static str,
        /// The underlying formatting error, preserved.
        #[source]
        source: Box<PromptsError>,
    },

    /// A strict format was asked to run and the caller supplied variables the
    /// template never uses — usually a typo in a variable name, which a lenient
    /// format would ignore silently.
    #[error("unused variables for template {template:?}: {variables:?}")]
    UnusedVariables {
        /// The template the variables were supplied for.
        template: String,
        /// Supplied variables that appear nowhere in the template.
        variables: Vec<String>,
    },

    /// A few-shot example does not supply a variable the example prompt needs.
    ///
    /// Such an example cannot be rendered. Without a selector it fails the whole
    /// `format`; with one it was silently *skipped*, quietly shrinking the
    /// example pool, so it is worth catching when the template is built.
    #[error("example {index} is missing variable {variable:?} required by the example prompt")]
    ExampleMissingVariable {
        /// Zero-based index of the offending example.
        index: usize,
        /// The variable the example prompt needs but the example lacks.
        variable: String,
    },

    /// A few-shot template's declared `input_variables` disagree with the
    /// variables its suffix actually uses.
    ///
    /// Both directions are reported together: `undeclared` (the suffix needs a
    /// variable nobody declared, so `format` can only fail at call time) and
    /// `unused` (a declared variable the suffix never references, so `format`
    /// demands a value it will not use).
    #[error("few-shot template variable mismatch: undeclared {undeclared:?}, unused {unused:?}")]
    FewShotVariableMismatch {
        /// Suffix variables that are not in `input_variables`.
        undeclared: Vec<String>,
        /// `input_variables` entries the suffix never uses.
        unused: Vec<String>,
    },
}
