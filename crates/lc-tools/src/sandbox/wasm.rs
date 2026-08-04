// lc-tools/src/sandbox/wasm.rs
//! WASM sandbox backend.

use async_trait::async_trait;

use super::{CodeSandbox, Language, RunResult, SandboxError};

/// WASM-based sandbox backend.
pub struct WasmSandbox {
    /// Maximum memory in bytes available to the WASM module.
    max_memory_bytes: u64,
}

impl WasmSandbox {
    /// Create a new WASM sandbox with default settings.
    pub fn new() -> Self {
        Self {
            max_memory_bytes: 64 * 1024 * 1024, // 64 MB default
        }
    }

    /// Set the maximum memory available to the WASM module.
    pub fn with_max_memory(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = bytes;
        self
    }
}

impl Default for WasmSandbox {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CodeSandbox for WasmSandbox {
    async fn run(
        &self,
        code: &str,
        language: Language,
        timeout_ms: u64,
    ) -> Result<RunResult, SandboxError> {
        let _ = (code, language, timeout_ms);
        Err(SandboxError::Runtime(
            "WASM sandbox backend is not yet implemented. \
             This is a stub behind the `sandbox-wasm` feature gate. \
             Contributions welcome!"
                .to_string(),
        ))
    }
}
