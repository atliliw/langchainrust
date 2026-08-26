// src/core/language_models/mod.rs
//! Language model base traits.

mod base;
mod chat;
mod multimodal;

pub use base::BaseLanguageModel;
pub use chat::{
    predict_tools, BaseChatModel, LLMResult, PredictToolsError, StreamChunk, TokenUsage,
};
pub use multimodal::{MultimodalError, MultimodalModel};
