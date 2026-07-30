// src/error.rs
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
pub use crate::language_models::openai::OpenAIError;
pub use crate::language_models::providers::anthropic::error::AnthropicError;
pub use crate::language_models::providers::gemini::GeminiError;
pub use crate::language_models::ollama::chat::OllamaError;
pub use crate::language_models::openai::AssistantError;
pub use crate::language_models::openai::responses::ResponsesError;

// ---- Agent Errors ----
pub use crate::agents::adaptive_rag::AdaptiveRAGError;
pub use crate::agents::base::AgentError;
pub use crate::agents::crag::CRAGError;
pub use crate::agents::crag::grader::GraderError;
pub use crate::agents::crag::rewriter::RewriterError;
pub use crate::agents::deep_research::ResearchError;
pub use crate::agents::handoffs::handoff::HandoffError;
pub use crate::agents::plan_execute::agent::PlanExecuteError;

// ---- Chain Errors ----
pub use crate::chains::base::ChainError;

// ---- Core Errors ----
pub use crate::core::batch::BatchError;
pub use crate::core::json_parse::LlmJsonParseError;
pub use crate::core::math::MathError;
pub use crate::core::output_parsers::OutputParserError;
pub use crate::core::router_llm::RouterError;
pub use crate::core::structured_output::extract::StructuredOutputError;
pub use crate::core::structured_output::parser::PartialJsonError;
pub use crate::core::tools::ToolError;

// ---- Embedding Errors ----
pub use crate::embeddings::EmbeddingError;

// ---- Memory Errors ----
pub use crate::memory::base::MemoryError;

// ---- Retrieval Errors ----
pub use crate::retrieval::graph_rag::GraphRAGError;
pub use crate::retrieval::hyde::HyDEError;
pub use crate::retrieval::LoaderError;
pub use crate::retrieval::multi_query::MultiQueryError;
pub use crate::retrieval::reranking::RerankingError;
pub use crate::retrieval::RetrieverError;

// ---- Vector Store Errors ----
pub use crate::vector_stores::VectorStoreError;

// ---- Tool/Sandbox Errors ----
pub use crate::tools::sandbox::SandboxError;

// ---- Callback Errors ----
pub use crate::callbacks::LangSmithError;

// ---- Graph Errors ----
pub use crate::langgraph::errors::GraphError;
pub use crate::langgraph::PersistenceError;

// ---- Guardrail Errors ----
pub use crate::guardrails::guardrail::GuardrailError;

// ---- Evaluation Errors ----
pub use crate::evaluation::EvalError;

// ---- Session Errors ----
pub use crate::sessions::store::SessionError;

// ---- A2A Errors ----
pub use crate::a2a::client::A2AError;

/// Unified error type that aggregates all sub-module errors.
///
/// This allows using `?` across module boundaries without manually
/// mapping error types. Each variant wraps the original sub-module
/// error, preserving full context.
#[derive(Debug)]
pub enum Error {
    // ---- LLM Provider ----
    /// OpenAI API error.
    OpenAI(OpenAIError),
    /// Anthropic API error.
    Anthropic(AnthropicError),
    /// Gemini API error.
    Gemini(GeminiError),
    /// Ollama API error.
    Ollama(OllamaError),
    /// OpenAI Assistants API error.
    Assistant(AssistantError),
    /// OpenAI Responses API error.
    Responses(ResponsesError),

    // ---- Agents ----
    /// Adaptive RAG agent error.
    AdaptiveRAG(AdaptiveRAGError),
    /// Agent execution error.
    Agent(AgentError),
    /// Corrective RAG agent error.
    CRAG(CRAGError),
    /// Document grading error.
    Grader(GraderError),
    /// Query rewriting error.
    Rewriter(RewriterError),
    /// Deep research agent error.
    Research(ResearchError),
    /// Agent handoff error.
    Handoff(HandoffError),
    /// Plan-execute agent error.
    PlanExecute(PlanExecuteError),

    // ---- Chains ----
    /// Chain execution error.
    Chain(ChainError),

    // ---- Core ----
    /// Batch processing error.
    Batch(BatchError),
    /// LLM JSON parsing error.
    LlmJsonParse(LlmJsonParseError),
    /// Math operation error.
    Math(MathError),
    /// Output parser error.
    OutputParser(OutputParserError),
    /// LLM router error.
    Router(RouterError),
    /// Structured output extraction error.
    StructuredOutput(StructuredOutputError),
    /// Partial JSON parsing error.
    PartialJson(PartialJsonError),
    /// Tool execution error.
    Tool(ToolError),

    // ---- Embeddings ----
    /// Embedding model error.
    Embedding(EmbeddingError),

    // ---- Memory ----
    /// Memory operation error.
    Memory(MemoryError),

    // ---- Retrieval ----
    /// Graph RAG error.
    GraphRAG(GraphRAGError),
    /// HyDE retriever error.
    HyDE(HyDEError),
    /// Document loader error.
    Loader(LoaderError),
    /// Multi-query retriever error.
    MultiQuery(MultiQueryError),
    /// Reranking error.
    Reranking(RerankingError),
    /// Retriever error.
    Retriever(RetrieverError),

    // ---- Vector Stores ----
    /// Vector store error.
    VectorStore(VectorStoreError),

    // ---- Tools / Sandbox ----
    /// Code sandbox error.
    Sandbox(SandboxError),

    // ---- Callbacks ----
    /// LangSmith callback error.
    LangSmith(LangSmithError),

    // ---- LangGraph ----
    /// Graph execution error.
    Graph(GraphError),
    /// Graph persistence error.
    Persistence(PersistenceError),

    // ---- Guardrails ----
    /// Guardrail validation error.
    Guardrail(GuardrailError),

    // ---- Evaluation ----
    /// Evaluation error.
    Eval(EvalError),

    // ---- Sessions ----
    /// Session store error.
    Session(SessionError),

    // ---- A2A ----
    /// Agent-to-Agent protocol error.
    A2A(A2AError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OpenAI(e) => write!(f, "OpenAI error: {e}"),
            Error::Anthropic(e) => write!(f, "Anthropic error: {e}"),
            Error::Gemini(e) => write!(f, "Gemini error: {e}"),
            Error::Ollama(e) => write!(f, "Ollama error: {e}"),
            Error::Assistant(e) => write!(f, "Assistant error: {e}"),
            Error::Responses(e) => write!(f, "Responses error: {e}"),
            Error::AdaptiveRAG(e) => write!(f, "AdaptiveRAG error: {e}"),
            Error::Agent(e) => write!(f, "Agent error: {e}"),
            Error::CRAG(e) => write!(f, "CRAG error: {e}"),
            Error::Grader(e) => write!(f, "Grader error: {e}"),
            Error::Rewriter(e) => write!(f, "Rewriter error: {e}"),
            Error::Research(e) => write!(f, "Research error: {e}"),
            Error::Handoff(e) => write!(f, "Handoff error: {e}"),
            Error::PlanExecute(e) => write!(f, "PlanExecute error: {e}"),
            Error::Chain(e) => write!(f, "Chain error: {e}"),
            Error::Batch(e) => write!(f, "Batch error: {e}"),
            Error::LlmJsonParse(e) => write!(f, "LlmJsonParse error: {e}"),
            Error::Math(e) => write!(f, "Math error: {e}"),
            Error::OutputParser(e) => write!(f, "OutputParser error: {e}"),
            Error::Router(e) => write!(f, "Router error: {e}"),
            Error::StructuredOutput(e) => write!(f, "StructuredOutput error: {e}"),
            Error::PartialJson(e) => write!(f, "PartialJson error: {e}"),
            Error::Tool(e) => write!(f, "Tool error: {e}"),
            Error::Embedding(e) => write!(f, "Embedding error: {e}"),
            Error::Memory(e) => write!(f, "Memory error: {e}"),
            Error::GraphRAG(e) => write!(f, "GraphRAG error: {e}"),
            Error::HyDE(e) => write!(f, "HyDE error: {e}"),
            Error::Loader(e) => write!(f, "Loader error: {e}"),
            Error::MultiQuery(e) => write!(f, "MultiQuery error: {e}"),
            Error::Reranking(e) => write!(f, "Reranking error: {e}"),
            Error::Retriever(e) => write!(f, "Retriever error: {e}"),
            Error::VectorStore(e) => write!(f, "VectorStore error: {e}"),
            Error::Sandbox(e) => write!(f, "Sandbox error: {e}"),
            Error::LangSmith(e) => write!(f, "LangSmith error: {e}"),
            Error::Graph(e) => write!(f, "Graph error: {e}"),
            Error::Persistence(e) => write!(f, "Persistence error: {e}"),
            Error::Guardrail(e) => write!(f, "Guardrail error: {e}"),
            Error::Eval(e) => write!(f, "Eval error: {e}"),
            Error::Session(e) => write!(f, "Session error: {e}"),
            Error::A2A(e) => write!(f, "A2A error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::OpenAI(e) => Some(e),
            Error::Anthropic(e) => Some(e),
            Error::Gemini(e) => Some(e),
            Error::Ollama(e) => Some(e),
            Error::Assistant(e) => Some(e),
            Error::Responses(e) => Some(e),
            Error::AdaptiveRAG(e) => Some(e),
            Error::Agent(e) => Some(e),
            Error::CRAG(e) => Some(e),
            Error::Grader(e) => Some(e),
            Error::Rewriter(e) => Some(e),
            Error::Research(e) => Some(e),
            Error::Handoff(e) => Some(e),
            Error::PlanExecute(e) => Some(e),
            Error::Chain(e) => Some(e),
            Error::Batch(e) => Some(e),
            Error::LlmJsonParse(e) => Some(e),
            Error::Math(e) => Some(e),
            Error::OutputParser(e) => Some(e),
            Error::Router(e) => Some(e),
            Error::StructuredOutput(e) => Some(e),
            Error::PartialJson(e) => Some(e),
            Error::Tool(e) => Some(e),
            Error::Embedding(e) => Some(e),
            Error::Memory(e) => Some(e),
            Error::GraphRAG(e) => Some(e),
            Error::HyDE(e) => Some(e),
            Error::Loader(e) => Some(e),
            Error::MultiQuery(e) => Some(e),
            Error::Reranking(e) => Some(e),
            Error::Retriever(e) => Some(e),
            Error::VectorStore(e) => Some(e),
            Error::Sandbox(e) => Some(e),
            Error::LangSmith(e) => Some(e),
            Error::Graph(e) => Some(e),
            Error::Persistence(e) => Some(e),
            Error::Guardrail(e) => Some(e),
            Error::Eval(e) => Some(e),
            Error::Session(e) => Some(e),
            Error::A2A(e) => Some(e),
        }
    }
}

// ---- From impls for all sub-error types ----

impl From<OpenAIError> for Error {
    fn from(e: OpenAIError) -> Self { Error::OpenAI(e) }
}
impl From<AnthropicError> for Error {
    fn from(e: AnthropicError) -> Self { Error::Anthropic(e) }
}
impl From<GeminiError> for Error {
    fn from(e: GeminiError) -> Self { Error::Gemini(e) }
}
impl From<OllamaError> for Error {
    fn from(e: OllamaError) -> Self { Error::Ollama(e) }
}
impl From<AssistantError> for Error {
    fn from(e: AssistantError) -> Self { Error::Assistant(e) }
}
impl From<ResponsesError> for Error {
    fn from(e: ResponsesError) -> Self { Error::Responses(e) }
}
impl From<AdaptiveRAGError> for Error {
    fn from(e: AdaptiveRAGError) -> Self { Error::AdaptiveRAG(e) }
}
impl From<AgentError> for Error {
    fn from(e: AgentError) -> Self { Error::Agent(e) }
}
impl From<CRAGError> for Error {
    fn from(e: CRAGError) -> Self { Error::CRAG(e) }
}
impl From<GraderError> for Error {
    fn from(e: GraderError) -> Self { Error::Grader(e) }
}
impl From<RewriterError> for Error {
    fn from(e: RewriterError) -> Self { Error::Rewriter(e) }
}
impl From<ResearchError> for Error {
    fn from(e: ResearchError) -> Self { Error::Research(e) }
}
impl From<HandoffError> for Error {
    fn from(e: HandoffError) -> Self { Error::Handoff(e) }
}
impl From<PlanExecuteError> for Error {
    fn from(e: PlanExecuteError) -> Self { Error::PlanExecute(e) }
}
impl From<ChainError> for Error {
    fn from(e: ChainError) -> Self { Error::Chain(e) }
}
impl From<BatchError> for Error {
    fn from(e: BatchError) -> Self { Error::Batch(e) }
}
impl From<LlmJsonParseError> for Error {
    fn from(e: LlmJsonParseError) -> Self { Error::LlmJsonParse(e) }
}
impl From<MathError> for Error {
    fn from(e: MathError) -> Self { Error::Math(e) }
}
impl From<OutputParserError> for Error {
    fn from(e: OutputParserError) -> Self { Error::OutputParser(e) }
}
impl From<RouterError> for Error {
    fn from(e: RouterError) -> Self { Error::Router(e) }
}
impl From<StructuredOutputError> for Error {
    fn from(e: StructuredOutputError) -> Self { Error::StructuredOutput(e) }
}
impl From<PartialJsonError> for Error {
    fn from(e: PartialJsonError) -> Self { Error::PartialJson(e) }
}
impl From<ToolError> for Error {
    fn from(e: ToolError) -> Self { Error::Tool(e) }
}
impl From<EmbeddingError> for Error {
    fn from(e: EmbeddingError) -> Self { Error::Embedding(e) }
}
impl From<MemoryError> for Error {
    fn from(e: MemoryError) -> Self { Error::Memory(e) }
}
impl From<GraphRAGError> for Error {
    fn from(e: GraphRAGError) -> Self { Error::GraphRAG(e) }
}
impl From<HyDEError> for Error {
    fn from(e: HyDEError) -> Self { Error::HyDE(e) }
}
impl From<LoaderError> for Error {
    fn from(e: LoaderError) -> Self { Error::Loader(e) }
}
impl From<MultiQueryError> for Error {
    fn from(e: MultiQueryError) -> Self { Error::MultiQuery(e) }
}
impl From<RerankingError> for Error {
    fn from(e: RerankingError) -> Self { Error::Reranking(e) }
}
impl From<RetrieverError> for Error {
    fn from(e: RetrieverError) -> Self { Error::Retriever(e) }
}
impl From<VectorStoreError> for Error {
    fn from(e: VectorStoreError) -> Self { Error::VectorStore(e) }
}
impl From<SandboxError> for Error {
    fn from(e: SandboxError) -> Self { Error::Sandbox(e) }
}
impl From<LangSmithError> for Error {
    fn from(e: LangSmithError) -> Self { Error::LangSmith(e) }
}
impl From<GraphError> for Error {
    fn from(e: GraphError) -> Self { Error::Graph(e) }
}
impl From<PersistenceError> for Error {
    fn from(e: PersistenceError) -> Self { Error::Persistence(e) }
}
impl From<GuardrailError> for Error {
    fn from(e: GuardrailError) -> Self { Error::Guardrail(e) }
}
impl From<EvalError> for Error {
    fn from(e: EvalError) -> Self { Error::Eval(e) }
}
impl From<SessionError> for Error {
    fn from(e: SessionError) -> Self { Error::Session(e) }
}
impl From<A2AError> for Error {
    fn from(e: A2AError) -> Self { Error::A2A(e) }
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
        assert!(matches!(err, Error::Agent(AgentError::ToolExecutionError(_))));
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
