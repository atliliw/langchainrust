use langchainrust::llms::{LLM, OpenAIConfig};

pub fn create_test_llm() -> LLM {
    let config = OpenAIConfig {
        api_key: "".to_string(),
        base_url: "".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
    };
    LLM::new(config)
}

pub fn llm_Config() -> OpenAIConfig {
    let config = OpenAIConfig {
        api_key: "".to_string(),
        base_url: "".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
    };
    return config;
}

pub fn create_test_llm_config_streaming() -> OpenAIConfig {
    let config = OpenAIConfig {
        api_key: "".to_string(),
        base_url: "".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: true,
    };
    return config;
}
