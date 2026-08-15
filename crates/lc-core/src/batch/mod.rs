// src/core/batch/mod.rs
//! Batch API client for OpenAI and Anthropic batch processing.
//!
//! Provides a unified interface for submitting, polling, and retrieving
//! results from batch API endpoints:
//!
//! - **OpenAI Batch API**: POST /v1/batches (JSONL file upload + batch creation)
//! - **Anthropic Batch API**: POST /v1/messages/batches (inline requests array)
//!
//! # Example
//! ```ignore
//! use langchainrust::core::batch::{BatchClient, BatchProvider, BatchRequest};
//! use langchainrust::Message;
//!
//! let client = BatchClient::new(BatchProvider::OpenAI, "sk-...");
//! let requests = vec![
//!     BatchRequest {
//!         custom_id: "req-1".into(),
//!         messages: vec![Message::human("Hello")],
//!         model: "gpt-4o".into(),
//!         temperature: None,
//!         max_tokens: None,
//!     },
//! ];
//! // let batch_id = client.submit(requests).await?;
//! // let results = client.submit_and_wait(requests, 5000, 300_000).await?;
//! ```

mod client;
#[cfg(test)]
mod tests;
mod types;

pub use client::BatchClient;
pub use types::{BatchError, BatchId, BatchProvider, BatchRequest, BatchResult, BatchStatus};
