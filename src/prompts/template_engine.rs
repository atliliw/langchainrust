// src/prompts/prompt_template.rs

use std::collections::HashMap;

/// 基础提示词模板
pub struct PromptTemplate {
    template: String,
    input_variables: Vec<String>,
}

impl PromptTemplate {
    pub fn new(template: &str, input_variables: Vec<&str>) -> Self {
        let input_vars = input_variables.iter().map(|s| s.to_string()).collect();
        Self {
            template: template.to_string(),
            input_variables: input_vars,
        }
    }

    /// 格式化模板，支持 {var} 占位符
    pub fn format(&self, values: &HashMap<String, String>) -> Result<String, String> {
        let mut result = self.template.clone();
        for (key, value) in values {
            let placeholder = format!("{{{}}}", key);
            if !result.contains(&placeholder) {
                return Err(format!("Missing placeholder: {}", key));
            }
            result = result.replace(&placeholder, value);
        }
        Ok(result)
    }

    pub fn get_input_variables(&self) -> &[String] {
        &self.input_variables
    }
}