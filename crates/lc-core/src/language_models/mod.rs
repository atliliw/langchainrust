// src/core/language_models/mod.rs
//! Language model base traits.

mod base;
mod boxed;
mod chat;
mod multimodal;

pub use base::BaseLanguageModel;
pub use chat::{
    predict_tools, BaseChatModel, LLMResult, LogprobChatModel, LogprobResult, PredictToolsError,
    StreamChunk, TokenAlternative, TokenLogprob, TokenUsage,
};
pub use multimodal::{MultimodalError, MultimodalModel};
