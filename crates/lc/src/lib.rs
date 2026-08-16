// crates/lc/src/lib.rs
//! LangChain Rust - A LangChain-compatible framework for building LLM applications.
//!
//! This crate is a Rust implementation inspired by LangChain Python.
//!
//! # Example
//! ```
//! use langchainrust::core::RunnableConfig;
//! use langchainrust::core::runnables::Runnable;
//!
//! let config = RunnableConfig::new()
//!     .with_tag("example")
//!     .with_run_name("my_run");
//! ```

#[cfg(test)]
extern crate tempfile;

// === Core Modules ===

/// Core abstractions: Runnable, BaseTool, BaseLanguageModel, etc.
pub mod core;

/// Message types: Human, AI, System, Tool.
pub mod schema;

/// LLM integrations: OpenAI, Ollama, DeepSeek, Qwen, Anthropic, etc.
pub mod language_models;

/// Built-in tools: Calculator, DateTime, Math, URLFetch.
pub mod tools;

/// Agent implementations: FunctionCallingAgent, ReActAgent.
pub mod agents;

/// Memory management: Buffer, Window, Summary, SummaryBuffer.
pub mod memory;

/// Chain compositions: LLMChain, SequentialChain, RetrievalQA.
pub mod chains;

/// Embedding models: OpenAI, DeepSeek, Qwen, Mock.
pub mod embeddings;

/// Vector stores: InMemory, Qdrant, MongoDB.
pub mod vector_stores;

/// Retrieval: BM25, Hybrid, MultiQuery, HyDE, Reranking.
pub mod retrieval;

/// Prompt templates: PromptTemplate, ChatPromptTemplate.
pub mod prompts;

/// Callbacks: StdOutHandler, LangSmith tracing.
pub mod callbacks;

/// LangGraph: StateGraph, CompiledGraph, Checkpointer.
pub mod langgraph;

/// MCP: Model Context Protocol client.
pub mod mcp;
pub use mcp::MCPServer;

/// Sessions: conversation lifecycle management.
pub mod sessions;

pub mod evaluation;
/// Guardrails: input/output safety validation.
pub mod guardrails;

/// Unified error type aggregating all sub-module errors.
pub mod error;
pub use error::Error;

/// A2A: Agent-to-Agent protocol for inter-agent communication.
pub mod a2a;

// 重新导出常用类型
pub use agents::{
    parse_review_verdict, review_envelope, task_adapter, AdaptiveRAG, AdaptiveRAGError,
    AdaptiveRAGResult, AgentAction, AgentBuilder, AgentError, AgentEventRunnable, AgentExecutor,
    AgentFinish, AgentOutput, AgentRunnable, AgentStep, AgentStreamEvent, AgentTask, BaseAgent,
    CRAGError, CRAGResult, Citation, CorrectiveRAGAgent, DeepResearchAgent, FanOutFanIn,
    FunctionCallingAgent, HandoffManager, Orchestrator, OrchestratorRunnable, PlanExecuteAgent,
    PlanExecuteError, PromptInjectionHook, RagDecision, ReActAgent, ResearchError, ResearchReport,
    ReviewOrchestrator, ReviewVerdict, RunContext, SequentialPipeline,
    StreamingFunctionCallingAgent, TaskAdapter, TokenBudgetHook, ToolInput, ToolPolicy, ToolRisk,
};
pub use core::batch::{
    BatchClient, BatchError, BatchId, BatchProvider, BatchRequest, BatchResult, BatchStatus,
};
pub use core::language_models::wrap_chat_model;
pub use core::language_models::LLMResult;
pub use core::router_llm::{RouterError, RouterLLM, RoutingStrategy};
pub use core::token_counter::{ModelPricing, TiktokenCounter, TokenCounter, TokenTrackingLLM};
pub use core::tools::to_tool_definition;
pub use core::tools::StructuredOutput;
pub use core::{
    // LCEL types
    into_runnable_any,
    BaseChatModel,
    BaseLanguageModel,
    BaseTool,
    FunctionCall,
    FunctionDefinition,
    LcelError,
    Runnable,
    RunnableAny,
    RunnableAnyWrapper,
    RunnableBinding,
    RunnableBranch,
    RunnableConfig,
    RunnableExt,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
    Tool,
    ToolCall,
    ToolCallResult,
    ToolDefinition,
    ToolError,
    ToolRegistry,
};
pub use evaluation::{
    Bleu, ContainsKeyword, Dataset, EmbeddingSimilarity, EvalError, EvalRunner, Evaluator,
    ExactMatch, Example, Faithfulness, LLMAsJudge, LengthCheck, PairwiseEvaluator, PairwiseJudge,
    Predictor, RegexMatch, Report, Score, StringDistance, Verdict,
};
pub use guardrails::{
    ForbiddenWordsGuardrail, GuardedAgent, GuardrailError, GuardrailRunner, GuardrailsConfig,
    InputGuardrail, MaxLengthGuardrail, OutputGuardrail, SensitiveInfoGuardrail,
};
pub use language_models::{
    AnthropicChat, AnthropicConfig, AnthropicError, AnthropicStreamToken,
    AnthropicStructuredOutputMethod, AssistantError, DeepSeekChat, DeepSeekConfig, GeminiChat,
    GeminiConfig, GeminiError, GeminiStructuredOutputMethod, LLMClient, MoonshotChat,
    MoonshotConfig, OllamaChat, OllamaConfig, OpenAIAssistant, OpenAIChat, OpenAIConfig, QwenChat,
    QwenConfig, ThinkingConfig, ThinkingType, ZhipuChat, ZhipuConfig,
};
pub use memory::{
    BaseMemory, ChatMessageHistory, ContextWindow, ConversationBufferMemory,
    ConversationBufferWindowMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory,
    MemoryData, MemoryError, PersistenceConfig, PersistentMemory, Strategy,
};

#[cfg(feature = "vectorstore-memory")]
pub use memory::VectorStoreRetrieverMemory;
pub use schema::{ImageContent, Message, MessageType};
pub use tools::{
    Calculator, CalculatorInput, CodeSandbox, ComputerMode, ComputerUseInput, ComputerUseOutput,
    ComputerUseTool, DateTimeInput, DateTimeTool, DuckDuckGoSearchTool, FileTool, HTTPTool,
    Language, LocalSandbox, MathInput, PythonREPLInput, PythonREPLTool, RunResult, SandboxError,
    SandboxTool, SearchInput, SimpleMathTool, URLFetchInput, URLFetchTool, WikipediaInput,
    WikipediaTool,
};

pub use chains::{
    BaseChain, ChainError, ChainResult, ChainRunnable, ChainStream, ConversationChain,
    ConversationChainBuilder, ConversationRetrievalChain, LLMChain, LLMChainBuilder,
    LLMRouterChain, MapReduceDocumentsChain, MapRerankDocumentsChain, RefineDocumentsChain,
    RetrievalQA, RouteDestination, RouterChain, SequentialChain, StreamToken, StuffDocumentsChain,
};
#[cfg(feature = "mongodb-persistence")]
pub use memory::MongoPersistentMemory;

// Embeddings
// P2-1: 无 `local-embeddings` feature 时 `LocalEmbeddings` 是已弃用的
// BagOfWordsEmbeddings 别名(静默降级);`#[allow(deprecated)]` 豁免重导出警告。
#[allow(deprecated)]
pub use embeddings::{
    cosine_similarity, l2_normalize, BagOfWordsEmbeddings, DeepSeekEmbeddings,
    DeepSeekEmbeddingsConfig, EmbeddingError, Embeddings, LocalEmbeddings, MockEmbeddings,
    OpenAIEmbeddings, OpenAIEmbeddingsConfig, QwenEmbeddings, QwenEmbeddingsConfig,
};

// Vector Stores
pub use vector_stores::{
    ChromaDBConfig, ChromaDBVectorStore, Document, FileVectorStore, InMemoryVectorStore,
    SearchResult, VectorStore, VectorStoreBuilder, VectorStoreError, VectorStoreProvider,
    VectorStoreType,
};

#[cfg(feature = "redis-storage")]
pub use vector_stores::{RedisDocumentStore, RedisStoreConfig};

#[cfg(feature = "sqlite-storage")]
pub use vector_stores::{SQLiteDocumentStore, SQLiteStoreConfig};

// PGVectorStore requires user-configured sqlx + pgvector dependencies.
// See src/vector_stores/pgvector.rs and docs/USAGE_EN.md for setup.
#[cfg(feature = "pgvector-storage")]
pub use vector_stores::pgvector;

pub use vector_stores::PineconeStore;
pub use vector_stores::{
    ChunkDocument, ChunkedDocumentStore, ChunkedDocumentStoreTrait, InMemoryChunkedDocumentStore,
};

#[cfg(feature = "qdrant-integration")]
pub use vector_stores::{QdrantConfig, QdrantVectorStore};

#[cfg(feature = "mongodb-persistence")]
pub use vector_stores::{MongoChunkedDocumentStore, MongoStoreConfig};

// Retrieval
#[allow(deprecated)]
pub use retrieval::{
    filter_by_score, reciprocal_rank_fusion, ChunkedHybridRetriever, RAGPipeline,
    RAGPipelineBuilder, RAGQueryResult, RagRunnable, RetrievalSource, RetrievedDocument,
};
pub use retrieval::{
    AutoMergingConfig, BM25Index, BM25Params, BM25Retriever, ChunkedBM25Retriever,
    ChunkedSearchResult, Tokenizer,
};
pub use retrieval::{
    BM25Reranker, KeywordReranker, Reranker, RerankingConfig, RerankingError, RerankingExecutor,
};
pub use retrieval::{
    CSVLoader, DocumentLoader, DocxLoader, HTMLLoader, JSONLoader, LoaderError, MarkdownLoader,
    PDFLoader, RecursiveCharacterSplitter, Retriever, RetrieverError, RetrieverTrait,
    SemanticSplitter, SimilarityRetriever, SitemapLoader, TextLoader, TextSplitter,
    WebScraperLoader,
};
pub use retrieval::{
    GraphCommunity, GraphEntity, GraphRAG, GraphRAGConfig, GraphRAGError, GraphRAGResult,
    GraphRelation, GraphStore, QueryMode as GraphQueryMode,
};
pub use retrieval::{HyDEConfig, HyDEError, HyDERetriever};
pub use retrieval::{HybridIndexConfig, HybridSearchResult, UnifiedHybridIndex};
pub use retrieval::{MultiQueryConfig, MultiQueryError, MultiQueryRetriever, StaticQueryGenerator};

// Prompts
pub use prompts::{
    ChatPromptTemplate, ExampleSelector, FewShotPromptTemplate, LengthBasedExampleSelector,
    PromptTemplate,
};

// Callbacks
#[cfg(feature = "opentelemetry")]
pub use callbacks::OtelHandler;
pub use callbacks::{
    CallbackHandler, CallbackManager, FileCallbackHandler, LangSmithClient, LangSmithConfig,
    LangSmithError, LangSmithHandler, LogFormat, RunTree, RunType, StdOutHandler,
};

// Tracing
#[cfg(feature = "opentelemetry")]
pub use callbacks::OtelTracingBackend;
pub use callbacks::{
    clear_span_stack, ConsoleTracingBackend, InMemoryTracingBackend, SpanGuard, SpanId, SpanKind,
    SpanStatus, SpanTokenUsage, TraceNode, TraceSpan, Tracer, TracingBackend,
};

// Output Parsers
pub use core::output_parsers::{
    BaseOutputParser, CommaSeparatedListOutputParser, JsonOutputParser, OutputParserError,
    OutputParserResult, StrOutputParser, StructuredOutputParser, TypedOutputParser,
};

// Structured Output
pub use core::structured_output::{
    stream_structured_output, with_structured_output, PartialJsonError, PartialJsonParser,
    StreamingStructuredOutputExt, StructuredOutputError, StructuredOutputExt,
};

// A2A
pub use a2a::{
    A2AClient, A2AError, A2AErrorData, A2AMessage, A2ARequest, A2AResponse, A2AServer, A2ATask,
    A2ATaskResult, AgentCard, TaskStatus,
};

// LangGraph
pub use langgraph::{
    AgentState, AppendMessagesReducer, AppendReducer, AppendStepsReducer, AsyncFn,
    AsyncFunctionRouter, AsyncNode, CheckpointData, Checkpointer, CompiledGraph, ConditionalEdge,
    EdgeDefinition, EdgeTarget, EdgeType, ExecutionStep, FileCheckpointer, FilePersistence,
    FunctionRouter, GraphBuilder, GraphDefinition, GraphEdge, GraphError, GraphExecution,
    GraphInvocation, GraphNode, GraphPersistence, GraphResult, MemoryCheckpointer,
    MemoryPersistence, MessageEntry, MessageRole, NodeConfig, NodeDefinition, NodeResult, NodeType,
    ParallelBranch, ParallelInvocation, Reducer, ReplaceReducer, RouterDefinition, StateGraph,
    StateSchema, StateUpdate, StepEntry, StreamEvent, SubgraphBuilder, SubgraphNode,
    ThreadSafeMemoryCheckpointer, END, START,
};
