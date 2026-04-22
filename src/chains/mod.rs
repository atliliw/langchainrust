// src/chains/mod.rs
//! Chain system for composing operations.
//!
//! Chain is LangChain's core abstraction, representing a sequence of operations.
//!
//! # Core Concepts
//!
//! - **BaseChain**: Base trait for chains.
//! - **LLMChain**: Most basic chain (Prompt + LLM).
//! - **SequentialChain**: Execute multiple chains sequentially.
//!
//! # Example
//!
//! ```ignore
//! use langchainrust::{LLMChain, SequentialChain, OpenAIChat};
//!
//! let llm = OpenAIChat::new(config);
//!
//! // Create LLMChain
//! let chain1 = LLMChain::new(llm.clone(), "Generate a word about {topic}");
//! let chain2 = LLMChain::new(llm, "Make a sentence with word: {word}");
//!
//! // Create SequentialChain
//! let seq_chain = SequentialChain::new()
//!     .add_chain(Arc::new(chain1), vec!["topic"], vec!["word"])
//!     .add_chain(Arc::new(chain2), vec!["word"], vec!["sentence"]);
//!
//! // Execute
//! let inputs = HashMap::from([("topic".into(), "programming".into())]);
//! let result = seq_chain.invoke(inputs).await?;
//! ```

pub mod base;
pub mod llm_chain;
pub mod sequential_chain;
pub mod conversation_chain;
pub mod router_chain;
pub mod retrieval_qa;

pub use base::{BaseChain, ChainError, ChainResult};
pub use llm_chain::{LLMChain, LLMChainBuilder};
pub use sequential_chain::SequentialChain;
pub use conversation_chain::{ConversationChain, ConversationChainBuilder};
pub use router_chain::{RouterChain, LLMRouterChain, RouteDestination};
pub use retrieval_qa::RetrievalQA;