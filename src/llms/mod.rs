mod openai;
mod qwen;

pub use openai::{LLM, OpenAIConfig};
pub use qwen::{LLMQwen, QwenConfig};

/// 模型配置枚举，支持多种 LLM 提供商
#[derive(Debug, Clone)]
pub enum ModelConfig {
    OpenAI(OpenAIConfig),
    Qwen(QwenConfig),
}
