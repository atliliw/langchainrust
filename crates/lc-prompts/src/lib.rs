#![warn(missing_docs)]
// crates/lc-prompts/src/lib.rs
//! Prompt template module.

mod chat_prompt_template;
mod error;
mod few_shot;
mod prompt_template;
mod registry;
mod template_parser;

pub use chat_prompt_template::ChatPromptTemplate;
pub use error::PromptsError;
pub use few_shot::{ExampleSelector, FewShotPromptTemplate, LengthBasedExampleSelector};
pub use prompt_template::PromptTemplate;
pub use registry::{
    PromptRegistry, PromptVersionInfo, RegisteredPrompt, VersionSpec, MIN_HASH_PREFIX,
};
