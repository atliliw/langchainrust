use crate::llms::{LLM, LLMQwen};
use crate::prompts::ChatPromptTemplate;
use std::collections::HashMap;

/// 统一的 LLM 抽象，支持 OpenAI 和 Qwen
#[derive(Debug, Clone)]
pub enum AnyLLM {
    OpenAI(LLM),
    Qwen(LLMQwen),
}

impl AnyLLM {
    pub async fn invoke_chat_template(
        &self,
        template: &ChatPromptTemplate,
        values: &HashMap<&str, &str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        match self {
            AnyLLM::OpenAI(llm) => llm.invoke_chat_template(template, values).await,
            AnyLLM::Qwen(llm) => llm.invoke_chat_template(template, values).await,
        }
    }
}

/// 模型路由缓存状态
#[derive(Debug, Clone)]
pub struct RoutingState {
    pub key: u64,
    pub llm: AnyLLM,
}
