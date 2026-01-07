pub mod chat_message_history;
use std::collections::HashMap;


pub trait Memory: Send + Sync {
    fn load_memory_variables(&self) -> HashMap<String, String>;
    fn save_context(&self, input: &str, output: &str);
}