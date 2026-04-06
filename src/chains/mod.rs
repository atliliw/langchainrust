// src/chains/mod.rs
//! Chains 系统
//!
//! Chain 是 LangChain 的核心抽象，表示一系列操作的组合。
//!
//! # 核心概念
//!
//! - **BaseChain**: Chain 的基础 trait
//! - **LLMChain**: 最基础的 Chain（Prompt + LLM）
//! - **SequentialChain**: 顺序执行多个 Chain
//!
//! # 示例
//!
//! ```ignore
//! use langchainrust::{LLMChain, SequentialChain, OpenAIChat};
//!
//! let llm = OpenAIChat::new(config);
//!
//! // 创建 LLMChain
//! let chain1 = LLMChain::new(llm.clone(), "生成一个关于{topic}的词");
//! let chain2 = LLMChain::new(llm, "用这个词造句: {word}");
//!
//! // 创建 SequentialChain
//! let seq_chain = SequentialChain::new()
//!     .add_chain(Arc::new(chain1), vec!["topic"], vec!["word"])
//!     .add_chain(Arc::new(chain2), vec!["word"], vec!["sentence"]);
//!
//! // 执行
//! let inputs = HashMap::from([("topic".into(), "编程".into())]);
//! let result = seq_chain.invoke(inputs).await?;
//! ```

pub mod base;
pub mod llm_chain;
pub mod sequential_chain;

pub use base::{BaseChain, ChainError, ChainResult};
pub use llm_chain::{LLMChain, LLMChainBuilder};
pub use sequential_chain::SequentialChain;