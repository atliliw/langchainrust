// src/prompts/prompt_template.rs

use regex::Regex;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template: String,
    input_variables: Vec<String>,
}

impl PromptTemplate {
    // ✅ 只接收模板字符串，自动提取 {xxx} 变量
    pub fn new(template: &str) -> Self {
        let re = Regex::new(r"\{([^}]+)\}").unwrap();
        let mut vars = Vec::new();
        for cap in re.captures_iter(template) {
            vars.push(cap[1].to_string());
        }
        vars.sort_unstable();
        vars.dedup();
        Self {
            template: template.to_string(),
            input_variables: vars,
        }
    }

    pub fn format(&self, values: &HashMap<&str, &str>) -> Result<String, String> {
        for var in &self.input_variables {
            if !values.contains_key(var.as_str()) {
                return Err(format!("缺少变量: {}", var));
            }
        }

        let mut result = self.template.clone();
        for var in &self.input_variables {
            let placeholder = format!("{{{}}}", var);
            if let Some(val) = values.get(var.as_str()) {
                result = result.replace(&placeholder, val);
            }
        }
        Ok(result)
    }
}