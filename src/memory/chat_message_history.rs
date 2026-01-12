// // use crate::memory::Memory;
// // use crate::messages::Message;
// // use std::collections::HashMap;
// //
// // use std::sync::{Arc, RwLock};
// //
// // pub struct ChatMessageHistory {
// //     messages: Arc<RwLock<Vec<Message>>>,
// // }
// //
// // impl ChatMessageHistory {
// //     pub fn new() -> Self {
// //         Self { messages: Arc::new(RwLock::new(vec![])) }
// //     }
// //
// //     pub fn add_user_message(&self, message: &str) {
// //         self.messages.write().unwrap().push(Message::human(message.to_string()));
// //     }
// //
// //     pub fn add_ai_message(&self, message: &str) {
// //         self.messages.write().unwrap().push(Message::ai(message.to_string()));
// //     }
// //
// //     pub fn get_messages(&self) -> Vec<Message> {
// //         self.messages.read().unwrap().clone()
// //     }
// //
// //     pub fn clear(&self) {
// //         self.messages.write().unwrap().clear();
// //     }
// // }
// //
// //
// // impl Memory for ChatMessageHistory {
// //     fn load_memory_variables(&self) -> HashMap<String, String> {
// //         let messages = self.messages.read().unwrap();
// //         let history_str = messages
// //             .iter()
// //             .map(|msg| msg.content())
// //             .collect::<Vec<&str>>()
// //             .join("\n");
// //
// //         let mut variables = HashMap::new();
// //         variables.insert("chat_history".to_string(), history_str);
// //         variables
// //     }
// //
// //     fn save_context(&self, input: &str, output: &str) {
// //         self.add_user_message(input);
// //         self.add_ai_message(output);
// //     }
// // }
//
//
// // src/memory.rs
//
// pub trait Memory {
//     /// 将当前输入和输出存入记忆
//     fn add(&mut self, input: &str, output: &str);
//
//     /// 获取当前完整上下文（用于拼接到 prompt）
//     fn context(&self) -> String;
//
//     /// 清空记忆
//     fn clear(&mut self);
// }
//
//
// // src/memory.rs
// #[derive(Default)]
// pub struct SimpleMemory {
//     history: Vec<(String, String)>, // (input, output)
// }
//
// impl Memory for SimpleMemory {
//     fn add(&mut self, input: &str, output: &str) {
//         self.history.push((input.to_string(), output.to_string()));
//     }
//
//     fn context(&self) -> String {
//         let mut s = String::new();
//         for (inp, out) in &self.history {
//             s.push_str(&format!("Human: {}\nAI: {}\n", inp, out));
//         }
//         s
//     }
//
//     fn clear(&mut self) {
//         self.history.clear();
//     }
// }


// src/memory.rs
use std::collections::VecDeque;

pub trait Memory {
    fn add(&mut self, input: &str, output: &str);
    fn context(&self) -> String;
    fn clear(&mut self);
    fn history(&self) -> Vec<&str>;
}

pub struct SimpleMemory {
    history: Vec<String>,
    max_turns: usize,
}

impl SimpleMemory {
    pub fn new(max_turns: usize) -> Self {
        Self {
            history: Vec::new(),
            max_turns,
        }
    }
}

impl Default for SimpleMemory {
    fn default() -> Self {
        Self::new(5)
    }
}

impl Memory for SimpleMemory {
    // fn add(&mut self, input: &str, _output: &str) {
    //     self.history.push(input.to_string());
    //     if self.history.len() > self.max_turns {
    //         self.history.remove(0);
    //     }
    // }

    fn add(&mut self, _input: &str, output: &str) {
        self.history.push(output.to_string()); // ← 存 output
        if self.history.len() > self.max_turns {
            self.history.remove(0);
        }
    }

    fn context(&self) -> String {
        self.history
            .iter()
            .enumerate()
            .map(|(i, q)| format!("{}. {}", i + 1, q))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn clear(&mut self) {
        self.history.clear();
    }

    fn history(&self) -> Vec<&str> {
        self.history.iter().map(|s| s.as_str()).collect()
    }

}