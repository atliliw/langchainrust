
use crate::messages::Message as LangMessage;
use crate::prompts::ChatPromptTemplate;
use std::collections::HashMap;

mod qwen;
mod openai;


pub use qwen::{LLMQwen};
pub use openai::LLM;
