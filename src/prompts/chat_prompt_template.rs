// src/prompts/chat_prompt_template.rs

use super::PromptTemplate;
use crate::messages::{SystemMessage, HumanMessage, AIMessage};
use std::collections::HashMap;

/// 支持多消息的聊天提示模板
pub struct ChatPromptTemplate {
    messages: Vec<ChatMessage>,
}

#[derive(Clone)]
pub enum ChatMessage {
    System(SystemMessage),
    Human(HumanMessage),
    AI(AIMessage),
}

impl ChatPromptTemplate {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self { messages }
    }

    pub fn from_messages(message_templates: Vec<(String, Vec<String>)>) -> Result<Self, String> {
        let mut messages = Vec::new();
        for (template, vars) in message_templates {
            // 简单解析：假设是 "system" 或 "human"
            if template.starts_with("system:") {
                let content = template.trim_start_matches("system:");
                let prompt = PromptTemplate::new(content, vars);
                messages.push(ChatMessage::System(SystemMessage {
                    content: prompt.format(&HashMap::new()).unwrap(),
                }));
            } else if template.starts_with("human:") {
                let content = template.trim_start_matches("human:");
                let prompt = PromptTemplate::new(content, vars);
                messages.push(ChatMessage::Human(HumanMessage {
                    content: prompt.format(&HashMap::new()).unwrap(),
                }));
            } else {
                return Err("Only 'system:' and 'human:' are supported".to_string());
            }
        }
        Ok(Self { messages })
    }

    pub fn format(&self, values: &HashMap<String, String>) -> Vec<ChatMessage> {
        self.messages
            .iter()
            .map(|msg| match msg {
                ChatMessage::System(sys) => {
                    let content = sys.content.replace("{{", "{").replace("}}", "}"); // 简单占位符处理
                    ChatMessage::System(SystemMessage { content })
                }
                ChatMessage::Human(hum) => {
                    let content = hum.content.replace("{{", "{").replace("}}", "}");
                    ChatMessage::Human(HumanMessage { content })
                }
                ChatMessage::AI(ai) => ChatMessage::AI(ai.clone()),
            })
            .collect()
    }
}