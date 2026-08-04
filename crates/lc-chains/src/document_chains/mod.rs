// lc-chains/src/document_chains/mod.rs
//! Document processing chains.
//!
//! Document processing chains that provide LLM processing capabilities for multiple documents:
//! - StuffDocumentsChain: Stuff all documents into a single prompt
//! - RefineDocumentsChain: Iteratively refine the answer document by document
//! - MapReduceDocumentsChain: Process documents in parallel then merge results
//! - MapRerankDocumentsChain: Process documents in parallel then rank by relevance

pub mod map_reduce;
pub mod map_rerank;
pub mod refine;
pub mod stuff;

pub use map_reduce::MapReduceDocumentsChain;
pub use map_rerank::{extract_score, MapRerankDocumentsChain};
pub use refine::RefineDocumentsChain;
pub use stuff::StuffDocumentsChain;
