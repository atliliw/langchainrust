// src/memory/mod.rs
pub mod chat_message_history;
use std::collections::HashMap;


// 定义 Memory trait，所有记忆系统都必须实现它
pub trait Memory {
    fn load_memory_variables(&self) -> HashMap<String, String>;
    fn save_context(&mut self, input: &str, output: &str);
}