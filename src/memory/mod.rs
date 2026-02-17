pub mod chat_message_history;
pub use crate::memory::chat_message_history::{Memory, SimpleMemory};
// use std::collections::HashMap;

// pub trait Memory: Send + Sync {
//     fn load_memory_variables(&self) -> HashMap<String, String>;
//     fn save_context(&self, input: &str, output: &str);
// }
