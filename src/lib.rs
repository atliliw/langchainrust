// src/lib.rs
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

// 重新导出常用类型
pub use core::{
    Runnable, RunnableConfig, BaseLanguageModel, BaseChatModel, 
    BaseTool, Tool, ToolError, ToolRegistry,
    ToolDefinition, ToolCall, ToolCallResult, FunctionDefinition, FunctionCall,
};
pub use core::tools::StructuredOutput;
pub use schema::{Message, MessageType};
pub use language_models::{
    OpenAIChat, OpenAIConfig, 
    OllamaChat, OllamaConfig,
    DeepSeekChat, DeepSeekConfig,
    MoonshotChat, MoonshotConfig,
    ZhipuChat, ZhipuConfig,
    QwenChat, QwenConfig,
    AnthropicChat, AnthropicConfig, AnthropicError,
};
pub use tools::{Calculator, CalculatorInput, DateTimeTool, DateTimeInput, SimpleMathTool, MathInput, URLFetchTool, URLFetchInput};
pub use agents::{AgentAction, AgentFinish, AgentStep, AgentOutput, ToolInput, BaseAgent, AgentExecutor, AgentError, ReActAgent, FunctionCallingAgent};
pub use core::tools::to_tool_definition;
pub use memory::{BaseMemory, MemoryError, ChatMessageHistory, ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory, PersistentMemory, PersistenceConfig, MemoryData};

#[cfg(feature = "mongodb-persistence")]
pub use memory::MongoPersistentMemory;
pub use chains::{BaseChain, ChainError, ChainResult, LLMChain, LLMChainBuilder, SequentialChain, ConversationChain, ConversationChainBuilder, RouterChain, LLMRouterChain, RouteDestination, RetrievalQA};

// Embeddings
pub use embeddings::{
    Embeddings, EmbeddingError, 
    OpenAIEmbeddings, OpenAIEmbeddingsConfig, 
    MockEmbeddings, 
    DeepSeekEmbeddings, DeepSeekEmbeddingsConfig,
    QwenEmbeddings, QwenEmbeddingsConfig,
    cosine_similarity,
};

// Vector Stores
pub use vector_stores::{Document, SearchResult, VectorStore, VectorStoreError, InMemoryVectorStore, VectorStoreProvider, VectorStoreType, VectorStoreBuilder};
pub use vector_stores::{ChunkDocument, ChunkedDocumentStoreTrait, InMemoryChunkedDocumentStore, ChunkedDocumentStore};

#[cfg(feature = "qdrant-integration")]
pub use vector_stores::{QdrantVectorStore, QdrantConfig};

#[cfg(feature = "mongodb-persistence")]
pub use vector_stores::{MongoChunkedDocumentStore, MongoStoreConfig};

// Retrieval
pub use retrieval::{Retriever, SimilarityRetriever, RetrieverTrait, RetrieverError, TextSplitter, RecursiveCharacterSplitter, PDFLoader, CSVLoader, TextLoader, JSONLoader, MarkdownLoader, DocumentLoader, LoaderError};
pub use retrieval::{BM25Retriever, BM25Index, BM25Params, Tokenizer, ChunkedBM25Retriever, ChunkedSearchResult, AutoMergingConfig};
pub use retrieval::{HybridRetriever, RetrievedDocument, RetrievalSource, reciprocal_rank_fusion, ChunkedHybridRetriever};
pub use retrieval::{UnifiedHybridIndex, HybridIndexConfig, HybridSearchResult};
pub use retrieval::{MultiQueryRetriever, MultiQueryConfig, MultiQueryError, StaticQueryGenerator};
pub use retrieval::{HyDERetriever, HyDEConfig, HyDEError};
pub use retrieval::{Reranker, KeywordReranker, BM25Reranker, RerankingExecutor, RerankingConfig, RerankingError};

// Prompts
pub use prompts::{PromptTemplate, ChatPromptTemplate};

// Callbacks
pub use callbacks::{CallbackHandler, CallbackManager, RunTree, RunType, LangSmithClient, LangSmithConfig, LangSmithError, StdOutHandler, LangSmithHandler, FileCallbackHandler, LogFormat};

// LangGraph
pub use langgraph::{
    StateSchema, StateUpdate, Reducer, ReplaceReducer, AppendReducer, AppendMessagesReducer, AppendStepsReducer,
    GraphNode, NodeResult, NodeConfig, AsyncNode, AsyncFn,
    GraphEdge, ConditionalEdge, EdgeTarget, FunctionRouter, AsyncFunctionRouter,
    StateGraph, GraphBuilder, START, END,
    CompiledGraph, GraphInvocation, StreamEvent, ExecutionStep, GraphExecution, ParallelInvocation, ParallelBranch,
    GraphError, GraphResult,
    Checkpointer, MemoryCheckpointer, ThreadSafeMemoryCheckpointer, FileCheckpointer, CheckpointData,
    AgentState, MessageEntry, MessageRole, StepEntry,
    SubgraphNode, SubgraphBuilder,
    GraphPersistence, GraphDefinition, NodeDefinition, EdgeDefinition, NodeType, EdgeType, RouterDefinition,
    MemoryPersistence, FilePersistence,
};