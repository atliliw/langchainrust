// lc-tools/src/sandbox/e2b.rs
//! E2B cloud sandbox backend.

use async_trait::async_trait;

use super::{CodeSandbox, Language, RunResult, SandboxError};

/// E2B cloud sandbox backend.
pub struct E2BSandbox {
    api_key: String,
    /// Base URL for the E2B API.
    base_url: String,
}

impl E2BSandbox {
    /// Create a new E2B sandbox, reading the API key from the `E2B_API_KEY` environment variable.
    pub fn new() -> Result<Self, SandboxError> {
        let api_key = std::env::var("E2B_API_KEY").map_err(|_| {
            SandboxError::Runtime("E2B_API_KEY environment variable is not set".to_string())
        })?;
        Ok(Self {
            api_key,
            base_url: "https://api.e2b.dev".to_string(),
        })
    }

    /// Create an E2B sandbox with a custom API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.e2b.dev".to_string(),
        }
    }

    /// Use a custom base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[async_trait]
impl CodeSandbox for E2BSandbox {
    async fn run(
        &self,
        code: &str,
        language: Language,
        timeout_ms: u64,
    ) -> Result<RunResult, SandboxError> {
        let _ = (code, language, timeout_ms);
        Err(SandboxError::Runtime(
            "E2B sandbox backend is not yet implemented. \
             This is a stub behind the `sandbox-e2b` feature gate. \
             Contributions welcome!"
                .to_string(),
        ))
    }
}
