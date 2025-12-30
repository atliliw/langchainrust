use serde::{Deserialize, Serialize};
use crate::messages::{Message as LangMessage, HumanMessage, SystemMessage, AIMessage};

#[derive(Debug, Clone)]
pub struct LLM {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl LLM {
    pub fn new(api_key: &str, base_url: &str, model: &str) -> Self {
        let client = reqwest::Client::new();
        Self {
            client,
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        }
    }

    // Original method for backward compatibility
    pub async fn generate(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let messages = vec![LangMessage::human(prompt)];
        self.generate_with_messages(messages).await
    }

    // New method that accepts a vector of messages
    pub async fn generate_with_messages(
        &self,
        messages: Vec<LangMessage>
    ) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = ChatRequest {
            model: self.model.to_string(),
            messages: messages.into_iter().map(|msg| ChatMessage {
                role: msg.role().to_string(),
                content: msg.content().to_string(),
            }).collect(),
        };

        let response: ChatResponse = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?
            .json()
            .await?;

        Ok(response.choices[0].message.content.clone())
    }

    // Convenience method for chat with system message
    pub async fn chat(
        &self,
        system_message: Option<&str>,
        human_message: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut messages = Vec::new();

        if let Some(system) = system_message {
            messages.push(LangMessage::system(system));
        }

        messages.push(LangMessage::human(human_message));

        self.generate_with_messages(messages).await
    }
}

// OpenAI API 结构体
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}