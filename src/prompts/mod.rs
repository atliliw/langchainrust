// src/prompts/mod.rs
//! Prompt template module.

mod chat_prompt_template;
mod prompt_template;
mod few_shot;

pub use chat_prompt_template::ChatPromptTemplate;
pub use prompt_template::PromptTemplate;
pub use few_shot::{FewShotPromptTemplate, ExampleSelector, LengthBasedExampleSelector};
