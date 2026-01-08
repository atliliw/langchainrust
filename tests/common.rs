use langchainrust::llms::{OpenAIConfig, LLM};


pub fn create_test_llm() -> LLM {
    let config = OpenAIConfig {
        api_key: "".to_string(),
        base_url: "https://api.openai-proxy.org/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
    };
    LLM::new(config)
}




