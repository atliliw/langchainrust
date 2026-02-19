use langchainrust::llms::{LLM, OpenAIConfig};

pub fn create_test_llm() -> LLM {
    LLM::new(llm_config())
}

pub fn llm_config() -> OpenAIConfig {
    OpenAIConfig {
        api_key: "".to_string(),
        base_url: "https://api.openai-proxy.org/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        factor: 1,
    }
}

pub fn create_test_llm_config_streaming() -> OpenAIConfig {
    OpenAIConfig {
        api_key: "".to_string(),
        base_url: "https://api.openai-proxy.org/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: true,
        factor: 1,
    }
}
