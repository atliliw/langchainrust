use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template: String,
    input_variables: Vec<String>,
}

impl PromptTemplate {
    pub fn new(template: &str, input_variables: Vec<&str>) -> Self {
        Self {
            template: template.to_string(),
            input_variables: input_variables.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn format(&self, values: &HashMap<&str, &str>) -> Result<String, String> {
        for var in &self.input_variables {
            if !values.contains_key(var.as_str()) {
                return Err(format!("Missing value for variable: {}", var));
            }
        }

        let mut result = self.template.clone();
        for (key, val) in values {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, val);
        }
        Ok(result)
    }
}


// // src/prompts/mod.rs

// // 1. 声明子模块（告诉编译器存在这些文件）
// mod prompt_template;
// mod chat_prompt_template;

// // 2. 选择性地将内部类型/结构体公开给外部使用者
// //    （这是最关键的一步：决定 API 暴露哪些内容）


// pub use chat_prompt_template::ChatPromptTemplate;

// // 3. （可选）如果将来有通用 trait 或错误类型，也可以在这里定义
// // pub trait Prompt {
// //     fn format(&self, inputs: &HashMap<String, String>) -> Result<String, Box<dyn std::error::Error>>;
// // }