use super::prompts::PromptTemplate;
use super::llms::LLM;
use super::messages::Message;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Chain {
    prompt: PromptTemplate,
    llm: LLM,
    system_message: Option<String>,
}

impl Chain {
    pub fn new(prompt: PromptTemplate, llm: LLM) -> Self {
        Self {
            prompt,
            llm,
            system_message: None,
        }
    }

    pub fn with_system_message(mut self, system_message: impl Into<String>) -> Self {
        self.system_message = Some(system_message.into());
        self
    }

    // Original run method - for backward compatibility
    pub async fn run(&self, inputs: HashMap<&str, &str>) -> Result<String, Box<dyn std::error::Error>> {
        let formatted_prompt = self.prompt.format(&inputs)?;

        if let Some(system) = &self.system_message {
            let messages = vec![
                Message::system(system),
                Message::human(formatted_prompt),
            ];
            self.llm.generate_with_messages(messages).await
        } else {
            self.llm.generate(&formatted_prompt).await
        }
    }

    // New run method with message history
    pub async fn run_with_messages(
        &self,
        inputs: HashMap<&str, &str>,
        history: Vec<Message>
    ) -> Result<String, Box<dyn std::error::Error>> {
        let formatted_prompt = self.prompt.format(&inputs)?;

        let mut messages = history;

        // Add system message if present
        if let Some(system) = &self.system_message {
            messages.push(Message::system(system));
        }

        // Add current human message
        messages.push(Message::human(formatted_prompt));

        self.llm.generate_with_messages(messages).await
    }
}