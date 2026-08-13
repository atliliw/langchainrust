// src/agents/builder.rs
//! AgentBuilder — 3 行代码创建 Agent
//!
//! 提供流畅的 Builder API，对标 rig 的 `client.agent(model).preamble(...).build()`。
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

/// Agent Builder — 流畅 API 创建 FunctionCallingAgent
///
/// 使用 Builder 模式，让用户只需 3 行代码就能创建并运行 Agent。
///
/// # 基本用法
///
/// ```ignore
/// let agent = AgentBuilder::new()
///     .llm(OpenAIChat::new(OpenAIConfig::new("sk-...")))
///     .system("You are a helpful assistant.")
///     .tool(Calculator::new())
///     .build()?;
/// ```
///
/// # 使用 Arc<dyn BaseChatModel>
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
    /// 创建新的 AgentBuilder
    pub fn new() -> Self {
        Self {
            llm: None,
            system_prompt: None,
            tools: Vec::new(),
            max_iterations: 10,
        }
    }

    /// 设置 LLM（任何实现了 `BaseChatModel` 的类型）
    ///
    /// 自动包装为 `Arc<dyn BaseChatModel<Error = ProviderError>>`。
    pub fn llm<L>(mut self, llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        self.llm = Some(lc_providers::wrap_chat_model(llm));
        self
    }

    /// 设置 LLM（从已包装的 `Arc<dyn BaseChatModel>`）
    ///
    /// 适用于已通过 `wrap_chat_model()` 或 `LLMClient` 创建的 LLM 实例。
    pub fn llm_from_arc(
        mut self,
        llm: Arc<dyn BaseChatModel<Error = ProviderError> + Send + Sync>,
    ) -> Self {
        self.llm = Some(llm);
        self
    }

    /// 设置系统提示词
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// 添加单个工具
    pub fn tool<T: BaseTool + 'static>(mut self, tool: T) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// 添加多个工具
    pub fn tools(mut self, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// 设置最大迭代次数（clamp 到 [1, 100]，防止 0 或失控上限）
    pub fn max_iterations(mut self, n: usize) -> Self {
        const MIN_MAX_ITERATIONS: usize = 1;
        const MAX_MAX_ITERATIONS: usize = 100;
        self.max_iterations = n.clamp(MIN_MAX_ITERATIONS, MAX_MAX_ITERATIONS);
        if n > MAX_MAX_ITERATIONS {
            log::warn!("max_iterations {} clamped to {}", n, MAX_MAX_ITERATIONS);
        }
        self
    }

    /// 构建 FunctionCallingAgent
    ///
    /// # Errors
    ///
    /// 如果没有设置 LLM，返回 `AgentError::Other`。
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

    /// 构建并包装为 `Arc<dyn BaseAgent>`，可直接传给 `AgentExecutor`
    ///
    /// # Errors
    ///
    /// 如果没有设置 LLM，返回 `AgentError::Other`。
    pub fn build_as_agent(self) -> Result<Arc<dyn BaseAgent>, AgentError> {
        let agent = self.build()?;
        Ok(Arc::new(agent) as Arc<dyn BaseAgent>)
    }

    /// 获取最大迭代次数
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
