use crate::messages::Message;
use crate::prompts::PromptTemplate; 
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ChatPromptTemplate {
    templates: Vec<Message>,
}

impl ChatPromptTemplate {
    /// Construct a chat prompt template from message templates.
    pub fn new(templates: Vec<Message>) -> Self {
        Self { templates }
    }

    /// Prepend a message to the template (useful for injecting memory).
    pub fn add_to_front(&mut self, message: Message) {
        self.templates.insert(0, message);
    }


    /// Render the template with the provided variables into concrete messages.
    pub fn format(&self, values: &HashMap<&str, &str>) -> Result<Vec<Message>, String> {
        let mut result = Vec::new();

        for template_msg in &self.templates {
            let template_str = template_msg.content();
            let role = template_msg.role(); // "system", "user", "assistant"
            let prompt = PromptTemplate::new(template_str);
            let content = prompt.format(values)?;
            let new_msg = match role {
                "system" => Message::system(content),
                "user" => Message::human(content),
                "assistant" => Message::ai(content),
                _ => return Err(format!("Unsupported characters: {}", role)),
            };

            result.push(new_msg);
        }

        Ok(result)
    }
}
