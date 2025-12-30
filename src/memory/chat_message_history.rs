// use crate::memory::Memory;
// use crate::messages::Message;
// use std::collections::HashMap;
//
// pub struct ChatMessageHistory {
//     messages: Vec<Message>,
// }
//
// impl ChatMessageHistory {
//     pub fn new() -> Self {
//         Self { messages: vec![] }
//     }
//
//     pub fn add_user_message(&mut self, message: &str) {
//         self.messages.push(Message::human(message.to_string()));
//     }
//
//     pub fn add_ai_message(&mut self, message: &str) {
//         self.messages.push(Message::ai(message.to_string()));
//     }
//
//     pub fn get_messages(&self) -> Vec<Message> {
//         self.messages.clone()
//     }
//
//     pub fn clear(&mut self) {
//         self.messages.clear();
//     }
// }
//
// // 实现 Memory trait
// impl Memory for ChatMessageHistory {
//     fn load_memory_variables(&self) -> HashMap<&str, &str> {
//         let mut variables = HashMap::new();
//
//         // 将历史消息序列化为字符串（简化版）
//         let history_str = self
//             .messages
//             .iter()
//             .map(|msg| msg.content())
//             .collect::<Vec<&str>>()
//             .join("\n");
//         variables.insert("chat_history", &*history_str);
//         variables
//     }
//
//     fn save_context(&mut self, input: &str, output: &str) {
//         self.add_user_message(input);
//         self.add_ai_message(output);
//     }
// }


use crate::memory::Memory;
use crate::messages::Message;
use std::collections::HashMap;

pub struct ChatMessageHistory {
    messages: Vec<Message>,
}

impl ChatMessageHistory {
    pub fn new() -> Self {
        Self { messages: vec![] }
    }

    pub fn add_user_message(&mut self, message: &str) {
        self.messages.push(Message::human(message.to_string()));
    }

    pub fn add_ai_message(&mut self, message: &str) {
        self.messages.push(Message::ai(message.to_string()));
    }

    pub fn get_messages(&self) -> Vec<Message> {
        self.messages.clone()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

// 实现 Memory trait
impl Memory for ChatMessageHistory {
    // ✅ 返回 HashMap<String, String>
    fn load_memory_variables(&self) -> HashMap<String, String> {
        let history_str = self
            .messages
            .iter()
            .map(|msg| msg.content()) // 假设 content() 返回 &str
            .collect::<Vec<&str>>()
            .join("\n");

        let mut variables = HashMap::new();
        // ✅ 插入 String，不再用引用
        variables.insert("chat_history".to_string(), history_str);
        variables
    }

    fn save_context(&mut self, input: &str, output: &str) {
        self.add_user_message(input);
        self.add_ai_message(output);
    }
}