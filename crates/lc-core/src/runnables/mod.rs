// src/core/runnables/mod.rs
//! Runnable module - LangChain Expression Language (LCEL) core
//!
//! The Runnable trait is the foundation of LangChain's composability.
//! Every component (LLM, Prompt, Tool, etc.) implements Runnable,
//! enabling them to be chained together seamlessly.
//!
//! # LCEL Composition
//!
//! Use `pipe()` to chain runnables:
//!
//! ```rust,ignore
//! let chain = prompt.pipe(llm).pipe(parser);
//! let result = chain.invoke("What is Rust?".to_string(), None).await?;
//! ```
//!
//! # Core Types
//!
//! - `Runnable<I, O>`: Base execution trait
//! - `RunnableExt`: Extension providing `pipe()` composition
//! - `RunnableSequence<I, O>`: Pipeline of chained runnables
//! - `RunnableLambda<I, O>`: Closure wrapper
//! - `RunnablePassthrough<I>`: Identity pass-through
//! - `RunnableParallel<I>`: Fan-out/fan-in
//! - `RunnableBranch<I, O>`: Conditional routing
//! - `RunnableBinding<I, O>`: Config/kwargs binding
//! - `LcelError`: Unified error type for pipelines
//! - `StreamEvent`: Fine-grained pipeline events

mod any;
mod assign;
mod binding;
mod branch;
mod cancellation;
mod config;
mod error;
mod events;
mod ext;
mod fallback;
mod lambda;
mod parallel;
mod passthrough;
mod runnable_trait;
mod sequence;

mod retry;

pub use any::{into_runnable_any, RunnableAny, RunnableAnyWrapper};
pub use assign::RunnableAssign;
pub use binding::RunnableBinding;
pub use branch::RunnableBranch;
pub use cancellation::CancellationToken;
pub use config::RunnableConfig;
pub use error::LcelError;
pub use events::LcelStreamEvent;
pub use ext::RunnableExt;
pub use fallback::RunnableWithFallbacks;
pub use lambda::RunnableLambda;
pub use parallel::RunnableParallel;
pub use passthrough::RunnablePassthrough;
pub use retry::{RetryConfig, RetryOn, RunnableRetry};
pub use runnable_trait::Runnable;
pub use sequence::RunnableSequence;
