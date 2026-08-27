// src/agents/builder.rs
//! AgentBuilder — create an Agent in 3 lines
//!
//! Provides a fluent Builder API, modeled on rig's
//! `client.agent(model).preamble(...).build()`.
//!
//! # Example
//!
//! ```ignore
//! let agent = AgentBuilder::new()
//!     .llm(OpenAIChat::new(OpenAIConfig::new("sk-...")))
//!     .system("You are a helpful assistant.")
//!     .tool(Calculator::new())
//!     .build()?;
//! ```

use crate::{AgentError, BaseAgent, FunctionCallingAgent};
use lc_core::language_models::BaseChatModel;
use lc_core::tools::BaseTool;
use lc_providers::ProviderError;
use std::sync::Arc;

/// Agent Builder — fluent API to create a FunctionCallingAgent
///
/// Uses the Builder pattern so users can create and run an Agent in just 3
/// lines.
///
/// # Basic usage
///
/// ```ignore
/// let agent = AgentBuilder::new()
///     .llm(OpenAIChat::new(OpenAIConfig::new("sk-...")))
///     .system("You are a helpful assistant.")
///     .tool(Calculator::new())
///     .build()?;
/// ```
///
/// # Using `Arc<dyn BaseChatModel>`
///
/// ```ignore
/// let llm = wrap_chat_model(OpenAIChat::new(config));
/// let agent = AgentBuilder::new()
///     .llm_from_arc(llm)
///     .system("You are a helpful assistant.")
///     .build()?;
/// ```
pub struct AgentBuilder {
    llm: Option<Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>>,
    system_prompt: Option<String>,
    tools: Vec<Arc<dyn BaseTool>>,
    max_iterations: usize,
}

impl AgentBuilder {
    /// Creates a new AgentBuilder
    pub fn new() -> Self {
        Self {
            llm: None,
            system_prompt: None,
            tools: Vec::new(),
            max_iterations: 10,
        }
    }

    /// Sets the LLM (any type implementing `BaseChatModel`)
    ///
    /// Automatically wraps it as `Arc<dyn BaseChatModel<Error = ProviderError>>`.
    pub fn llm<L>(mut self, llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        self.llm = Some(lc_providers::wrap_chat_model(llm));
        self
    }

    /// Sets the LLM (from an already-wrapped `Arc<dyn BaseChatModel>`)
    ///
    /// For LLM instances already created via `wrap_chat_model()` or `LLMClient`.
    pub fn llm_from_arc(
        mut self,
        llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    ) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Sets the system prompt
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Adds a single tool
    pub fn tool<T: BaseTool + 'static>(mut self, tool: T) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Adds multiple tools
    pub fn tools(mut self, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Sets max iterations (clamped to [1, 100] to prevent 0 or runaway limits)
    pub fn max_iterations(mut self, n: usize) -> Self {
        const MIN_MAX_ITERATIONS: usize = 1;
        const MAX_MAX_ITERATIONS: usize = 100;
        self.max_iterations = n.clamp(MIN_MAX_ITERATIONS, MAX_MAX_ITERATIONS);
        if n > MAX_MAX_ITERATIONS {
            log::warn!("max_iterations {} clamped to {}", n, MAX_MAX_ITERATIONS);
        }
        self
    }

    /// Builds a FunctionCallingAgent
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Other` if no LLM was set.
    pub fn build(self) -> Result<FunctionCallingAgent, AgentError> {
        let llm = self.llm.ok_or_else(|| {
            AgentError::Other("AgentBuilder: LLM is required. Call .llm() first.".into())
        })?;

        Ok(FunctionCallingAgent::from_arc(
            llm,
            self.tools,
            self.system_prompt,
        ))
    }

    /// Builds and wraps as `Arc<dyn BaseAgent>` for direct use with `AgentExecutor`
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Other` if no LLM was set.
    pub fn build_as_agent(self) -> Result<Arc<dyn BaseAgent>, AgentError> {
        let agent = self.build()?;
        Ok(Arc::new(agent) as Arc<dyn BaseAgent>)
    }

    /// Returns the max iterations
    pub fn get_max_iterations(&self) -> usize {
        self.max_iterations
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_providers::{OpenAIChat, OpenAIConfig};
    use lc_tools::Calculator;

    #[test]
    fn test_builder_with_openai() {
        let config = OpenAIConfig::new("test_key").with_base_url("http://localhost:8080/v1");
        let agent = AgentBuilder::new()
            .llm(OpenAIChat::new(config))
            .system("You are a test assistant.")
            .tool(Calculator::new())
            .build()
            .unwrap();

        assert_eq!(agent.tools_count(), 1);
        assert_eq!(agent.system_prompt(), Some("You are a test assistant."));
    }

    #[test]
    fn test_builder_missing_llm() {
        let result = AgentBuilder::new().system("test").build();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("LLM is required"));
    }

    #[test]
    fn test_builder_multiple_tools() {
        let config = OpenAIConfig::new("test_key").with_base_url("http://localhost:8080/v1");
        let agent = AgentBuilder::new()
            .llm(OpenAIChat::new(config))
            .tool(Calculator::new())
            .tool(Calculator::new())
            .build()
            .unwrap();

        assert_eq!(agent.tools_count(), 2);
    }

    #[test]
    fn test_builder_tools_vec() {
        let config = OpenAIConfig::new("test_key").with_base_url("http://localhost:8080/v1");
        let tools: Vec<Arc<dyn BaseTool>> =
            vec![Arc::new(Calculator::new()), Arc::new(Calculator::new())];
        let agent = AgentBuilder::new()
            .llm(OpenAIChat::new(config))
            .tools(tools)
            .build()
            .unwrap();

        assert_eq!(agent.tools_count(), 2);
    }

    #[test]
    fn test_builder_build_as_agent() {
        let config = OpenAIConfig::new("test_key").with_base_url("http://localhost:8080/v1");
        let agent = AgentBuilder::new()
            .llm(OpenAIChat::new(config))
            .system("test")
            .build_as_agent()
            .unwrap();

        let allowed = agent.get_allowed_tools();
        assert!(allowed.is_some());
    }

    #[test]
    fn test_builder_max_iterations() {
        let builder = AgentBuilder::new().max_iterations(5);
        assert_eq!(builder.get_max_iterations(), 5);
    }

    #[test]
    fn test_builder_max_iterations_clamped_to_min() {
        let builder = AgentBuilder::new().max_iterations(0);
        assert_eq!(builder.get_max_iterations(), 1);
    }

    #[test]
    fn test_builder_max_iterations_clamped_to_max() {
        let builder = AgentBuilder::new().max_iterations(1_000_000_000);
        assert_eq!(builder.get_max_iterations(), 100);
    }

    #[test]
    fn test_builder_default() {
        let builder = AgentBuilder::default();
        assert_eq!(builder.get_max_iterations(), 10);
        assert!(builder.llm.is_none());
    }
}
