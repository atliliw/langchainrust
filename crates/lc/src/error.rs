// crates/lc/src/error.rs
//! Unified error type for the `langchainrust` crate.
//!
//! All sub-module error types can be converted into [`Error`] via `From` impls,
//! so the `?` operator works seamlessly across module boundaries.
//!
//! # Example
//!
//! ```ignore
//! use langchainrust::Error;
//!
//! fn do_work() -> Result<(), Error> {
//!     let chat = OpenAIChat::new(config);
//!     let result = chat.invoke(messages, None).await?; // OpenAIError → Error
//!     let docs = retriever.retrieve(query, 4).await?;  // RetrieverError → Error
//!     Ok(())
//! }
//! ```

// ---- LLM Provider Errors ----
pub use lc_providers::ollama::OllamaError;
pub use lc_providers::openai::responses::types::ResponsesError;
pub use lc_providers::openai::AssistantError;
pub use lc_providers::openai::OpenAIError;
pub use lc_providers::providers::anthropic::error::AnthropicError;
pub use lc_providers::providers::gemini::GeminiError;

// ---- Agent Errors ----
pub use lc_agents::crag::grader::GraderError;
pub use lc_agents::crag::rewriter::RewriterError;
pub use lc_agents::handoffs::handoff::HandoffError;
pub use lc_agents::AdaptiveRAGError;
pub use lc_agents::AgentError;
pub use lc_agents::CRAGError;
pub use lc_agents::PlanExecuteError;
pub use lc_agents::ResearchError;

// ---- Chain Errors ----
pub use lc_chains::ChainError;

// ---- Core Errors ----
pub use lc_core::batch::BatchError;
pub use lc_core::json_parse::LlmJsonParseError;
pub use lc_core::math::MathError;
pub use lc_core::output_parsers::OutputParserError;
pub use lc_core::router_llm::RouterError;
pub use lc_core::structured_output::extract::StructuredOutputError;
pub use lc_core::structured_output::parser::PartialJsonError;
pub use lc_core::tools::ToolError;

// ---- Embedding Errors ----
pub use lc_embeddings::EmbeddingError;

// ---- Memory Errors ----
pub use lc_memory::base::MemoryError;

// ---- Retrieval Errors ----
pub use lc_rag::graph_rag::GraphRAGError;
pub use lc_rag::hyde::HyDEError;
pub use lc_rag::multi_query::MultiQueryError;
pub use lc_rag::reranking::RerankingError;
pub use lc_rag::LoaderError;
pub use lc_rag::RetrieverError;

// ---- Vector Store Errors ----
pub use lc_vector_stores::VectorStoreError;

// ---- Tool/Sandbox Errors ----
pub use lc_tools::sandbox::SandboxError;

// ---- Callback Errors ----
pub use lc_callbacks::LangSmithError;

// ---- Graph Errors ----
pub use lc_langgraph::errors::GraphError;
pub use lc_langgraph::PersistenceError;

// ---- Guardrail Errors ----
pub use lc_guardrails::GuardrailError;

// ---- Evaluation Errors ----
pub use lc_evaluation::EvalError;

// ---- Session Errors ----
pub use lc_sessions::store::SessionError;

// ---- A2A Errors ----
pub use lc_a2a::client::A2AError;

/// Unified error type that aggregates all sub-module errors.
///
/// This allows using `?` across module boundaries without manually
/// mapping error types. Each variant wraps the original sub-module
/// error, preserving full context.
///
/// 0.21.0 S3.4: `Display` / `std::error::Error` (with `source()`) / `From` are
/// derived via `thiserror` (`#[error]` / `#[from]`) — behaviorally equivalent
/// to the previous 40+ hand-written impls (~300 lines of boilerplate removed):
/// each variant renders as `"<Name> error: <inner>"` and its `source()` is the
/// wrapped sub-module error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    // ---- LLM Provider ----
    /// OpenAI API error.
    #[error("OpenAI error: {0}")]
    OpenAI(#[from] OpenAIError),
    /// Anthropic API error.
    #[error("Anthropic error: {0}")]
    Anthropic(#[from] AnthropicError),
    /// Gemini API error.
    #[error("Gemini error: {0}")]
    Gemini(#[from] GeminiError),
    /// Ollama API error.
    #[error("Ollama error: {0}")]
    Ollama(#[from] OllamaError),
    /// OpenAI Assistants API error.
    #[error("Assistant error: {0}")]
    Assistant(#[from] AssistantError),
    /// OpenAI Responses API error.
    #[error("Responses error: {0}")]
    Responses(#[from] ResponsesError),
    /// Provider error (from lc-providers crate).
    #[error("Provider error: {0}")]
    Provider(#[from] lc_providers::ProviderError),

    // ---- Agents ----
    /// Adaptive RAG agent error.
    #[error("AdaptiveRAG error: {0}")]
    AdaptiveRAG(#[from] AdaptiveRAGError),
    /// Agent execution error.
    #[error("Agent error: {0}")]
    Agent(#[from] AgentError),
    /// Corrective RAG agent error.
    #[error("CRAG error: {0}")]
    CRAG(#[from] CRAGError),
    /// Document grading error.
    #[error("Grader error: {0}")]
    Grader(#[from] GraderError),
    /// Query rewriting error.
    #[error("Rewriter error: {0}")]
    Rewriter(#[from] RewriterError),
    /// Deep research agent error.
    #[error("Research error: {0}")]
    Research(#[from] ResearchError),
    /// Agent handoff error.
    #[error("Handoff error: {0}")]
    Handoff(#[from] HandoffError),
    /// Plan-execute agent error.
    #[error("PlanExecute error: {0}")]
    PlanExecute(#[from] PlanExecuteError),

    // ---- Chains ----
    /// Chain execution error.
    #[error("Chain error: {0}")]
    Chain(#[from] ChainError),

    // ---- Core ----
    /// Batch processing error.
    #[error("Batch error: {0}")]
    Batch(#[from] BatchError),
    /// LLM JSON parsing error.
    #[error("LlmJsonParse error: {0}")]
    LlmJsonParse(#[from] LlmJsonParseError),
    /// Math operation error.
    #[error("Math error: {0}")]
    Math(#[from] MathError),
    /// Output parser error.
    #[error("OutputParser error: {0}")]
    OutputParser(#[from] OutputParserError),
    /// LLM router error.
    #[error("Router error: {0}")]
    Router(#[from] RouterError),
    /// Structured output extraction error.
    #[error("StructuredOutput error: {0}")]
    StructuredOutput(#[from] StructuredOutputError),
    /// Partial JSON parsing error.
    #[error("PartialJson error: {0}")]
    PartialJson(#[from] PartialJsonError),
    /// Tool execution error.
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    // ---- Embeddings ----
    /// Embedding model error.
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    // ---- Memory ----
    /// Memory operation error.
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),

    // ---- Retrieval ----
    /// Graph RAG error.
    #[error("GraphRAG error: {0}")]
    GraphRAG(#[from] GraphRAGError),
    /// HyDE retriever error.
    #[error("HyDE error: {0}")]
    HyDE(#[from] HyDEError),
    /// Document loader error.
    #[error("Loader error: {0}")]
    Loader(#[from] LoaderError),
    /// Multi-query retriever error.
    #[error("MultiQuery error: {0}")]
    MultiQuery(#[from] MultiQueryError),
    /// Reranking error.
    #[error("Reranking error: {0}")]
    Reranking(#[from] RerankingError),
    /// Retriever error.
    #[error("Retriever error: {0}")]
    Retriever(#[from] RetrieverError),

    // ---- Vector Stores ----
    /// Vector store error.
    #[error("VectorStore error: {0}")]
    VectorStore(#[from] VectorStoreError),

    // ---- Tools / Sandbox ----
    /// Code sandbox error.
    #[error("Sandbox error: {0}")]
    Sandbox(#[from] SandboxError),

    // ---- Callbacks ----
    /// LangSmith callback error.
    #[error("LangSmith error: {0}")]
    LangSmith(#[from] LangSmithError),

    // ---- LangGraph ----
    /// Graph execution error.
    #[error("Graph error: {0}")]
    Graph(#[from] GraphError),
    /// Graph persistence error.
    #[error("Persistence error: {0}")]
    Persistence(#[from] PersistenceError),

    // ---- Guardrails ----
    /// Guardrail validation error.
    #[error("Guardrail error: {0}")]
    Guardrail(#[from] GuardrailError),

    // ---- Evaluation ----
    /// Evaluation error.
    #[error("Eval error: {0}")]
    Eval(#[from] EvalError),

    // ---- Sessions ----
    /// Session store error.
    #[error("Session error: {0}")]
    Session(#[from] SessionError),

    // ---- A2A ----
    /// Agent-to-Agent protocol error.
    #[error("A2A error: {0}")]
    A2A(#[from] A2AError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_openai_error() {
        let err = Error::from(OpenAIError::Http("timeout".to_string()));
        assert!(matches!(err, Error::OpenAI(OpenAIError::Http(_))));
        assert!(err.to_string().contains("OpenAI"));
    }

    #[test]
    fn test_from_agent_error() {
        let err = Error::from(AgentError::ToolExecutionError("tool failed".to_string()));
        assert!(matches!(
            err,
            Error::Agent(AgentError::ToolExecutionError(_))
        ));
    }

    #[test]
    fn test_from_chain_error() {
        let err = Error::from(ChainError::ExecutionError("bad request".to_string()));
        assert!(matches!(err, Error::Chain(ChainError::ExecutionError(_))));
    }

    #[test]
    fn test_from_memory_error() {
        let err = Error::from(MemoryError::LoadError("corrupt".to_string()));
        assert!(matches!(err, Error::Memory(MemoryError::LoadError(_))));
    }

    #[test]
    fn test_from_retriever_error() {
        let err = Error::from(RetrieverError::NoResults);
        assert!(matches!(err, Error::Retriever(RetrieverError::NoResults)));
    }

    #[test]
    fn test_display_format() {
        let err = Error::from(ToolError::ExecutionFailed("timeout".to_string()));
        let msg = err.to_string();
        assert!(msg.contains("Tool error"));
        assert!(msg.contains("timeout"));
    }

    #[test]
    fn test_source_chain() {
        use std::error::Error as StdError;
        let err = Error::from(EmbeddingError::HttpError("invalid".to_string()));
        assert!(StdError::source(&err).is_some());
    }
}
