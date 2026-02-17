use crate::messages::Message as LangMessage;
use crate::prompts::ChatPromptTemplate;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LLMQwen {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl LLMQwen {
    pub fn new(api_key: &str, base_url: &str, model: &str) -> Self {
        let client = reqwest::Client::new();
        Self {
            client,
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        }
    }
    pub async fn generate(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let messages = vec![LangMessage::human(prompt)];
        self.generate_with_messages(messages).await
    }
    pub async fn generate_with_messages(
        &self,
        messages: Vec<LangMessage>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{}/chat/completions", self.base_url);

        let openai_messages: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role(),
                    "content": m.content()
                })
            })
            .collect();

        let body = serde_json::json!({
            "model": self.model,
            "messages": openai_messages,
        });

        let response: ChatResponse = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?
            .json()
            .await?;

        let first_choice = response.choices.first().ok_or("No choices returned")?;
        Ok(first_choice.message.content.clone())
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

    pub async fn invoke_chat_template(
        &self,
        template: &ChatPromptTemplate,
        values: &HashMap<&str, &str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let messages = template
            .format(values)
            .map_err(|e| format!("模板格式化失败: {}", e))?;

        self.generate_with_messages(messages).await
    }
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
