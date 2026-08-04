// lc-schema/src/messages/mod.rs
//! Message types for LangChain
//!
//! Messages are the inputs and outputs of chat models.

mod image;
mod message;

pub use image::ImageContent;
pub use message::{Message, MessageType};
