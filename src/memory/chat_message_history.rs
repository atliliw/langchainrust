use crate::memory::Memory;
use crate::messages::Message;
use std::collections::HashMap;

use std::sync::{Arc, RwLock};

pub struct ChatMessageHistory {
    messages: Arc<RwLock<Vec<Message>>>,
}

impl ChatMessageHistory {
    pub fn new() -> Self {
        Self { messages: Arc::new(RwLock::new(vec![])) }
    }

    pub fn add_user_message(&self, message: &str) {
        self.messages.write().unwrap().push(Message::human(message.to_string()));
    }

    pub fn add_ai_message(&self, message: &str) {
        self.messages.write().unwrap().push(Message::ai(message.to_string()));
    }

    pub fn get_messages(&self) -> Vec<Message> {
        self.messages.read().unwrap().clone()
    }

    pub fn clear(&self) {
        self.messages.write().unwrap().clear();
    }
}

// 实现 Memory trait
impl Memory for ChatMessageHistory {
    fn load_memory_variables(&self) -> HashMap<String, String> {
        let messages = self.messages.read().unwrap();
        let history_str = messages
            .iter()
            .map(|msg| msg.content())
            .collect::<Vec<&str>>()
            .join("\n");

        let mut variables = HashMap::new();
        variables.insert("chat_history".to_string(), history_str);
        variables
    }

    fn save_context(&self, input: &str, output: &str) {
        self.add_user_message(input);
        self.add_ai_message(output);
    }
}