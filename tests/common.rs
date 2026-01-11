use langchainrust::llms::{OpenAIConfig, LLM};


pub fn create_test_llm() -> LLM {
    let config = OpenAIConfig {
        api_key: "sk-l0YYMX65mCYRlTJYH0ptf4BFpqJwm8Xo9Z5IMqSZD0yOafl6".to_string(),
        base_url: "https://api.openai-proxy.org/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
    };
    LLM::new(config)
}




pub fn create_test_llm_Config() -> OpenAIConfig {
    let config = OpenAIConfig {
        api_key: "sk-l0YYMX65mCYRlTJYH0ptf4BFpqJwm8Xo9Z5IMqSZD0yOafl6".to_string(),
        base_url: "https://api.openai-proxy.org/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
    };
    return config;
}


pub fn create_test_llm_config_streaming() -> OpenAIConfig {
    let config = OpenAIConfig {
        api_key: "sk-l0YYMX65mCYRlTJYH0ptf4BFpqJwm8Xo9Z5IMqSZD0yOafl6".to_string(),
        base_url: "https://api.openai-proxy.org/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: true,
    };
    return config;
}