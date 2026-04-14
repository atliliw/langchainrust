// src/lib.rs
//! LangChain Rust - A LangChain-compatible framework for building LLM applications
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

pub mod core;
pub mod schema;
pub mod language_models;
pub mod tools;
pub mod agents;
pub mod memory;
pub mod chains;
pub mod embeddings;
pub mod vector_stores;
pub mod retrieval;
pub mod prompts;
pub mod callbacks;

// 重新导出常用类型
pub use core::{
    Runnable, RunnableConfig, BaseLanguageModel, BaseChatModel, 
    BaseTool, Tool, ToolError, ToolRegistry,
    ToolDefinition, ToolCall, ToolCallResult, FunctionDefinition, FunctionCall,
};
pub use core::tools::StructuredOutput;
pub use schema::{Message, MessageType};
pub use language_models::{OpenAIChat, OpenAIConfig, OllamaChat, OllamaConfig};
pub use tools::{Calculator, CalculatorInput, DateTimeTool, DateTimeInput, SimpleMathTool, MathInput, URLFetchTool, URLFetchInput};
pub use agents::{AgentAction, AgentFinish, AgentStep, AgentOutput, ToolInput, BaseAgent, AgentExecutor, AgentError, ReActAgent, FunctionCallingAgent};
pub use core::tools::to_tool_definition;
pub use memory::{BaseMemory, MemoryError, ChatMessageHistory, ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory};
pub use chains::{BaseChain, ChainError, ChainResult, LLMChain, LLMChainBuilder, SequentialChain, ConversationChain, ConversationChainBuilder, RouterChain, LLMRouterChain, RouteDestination, RetrievalQA};

// Embeddings
pub use embeddings::{Embeddings, EmbeddingError, OpenAIEmbeddings, OpenAIEmbeddingsConfig, MockEmbeddings, cosine_similarity};

// Vector Stores
pub use vector_stores::{Document, SearchResult, VectorStore, VectorStoreError, InMemoryVectorStore, VectorStoreProvider, VectorStoreType, VectorStoreBuilder};

#[cfg(feature = "qdrant-integration")]
pub use vector_stores::{QdrantVectorStore, QdrantConfig};

// Retrieval
pub use retrieval::{Retriever, SimilarityRetriever, RetrieverTrait, TextSplitter, RecursiveCharacterSplitter, PDFLoader, CSVLoader, DocumentLoader, LoaderError};

// Prompts
pub use prompts::{PromptTemplate, ChatPromptTemplate};

// Callbacks
pub use callbacks::{CallbackHandler, CallbackManager, RunTree, RunType, LangSmithClient, LangSmithConfig, LangSmithError, StdOutHandler, LangSmithHandler, FileCallbackHandler, LogFormat};