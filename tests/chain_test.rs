#[cfg(test)]
mod tests {
    use langchainrust::chains::{PromptChain, SequentialChain};
    use langchainrust::llms::{LLM,OpenAIConfig};
    use langchainrust::messages::Message;
    use langchainrust::prompts::ChatPromptTemplate;
    use std::collections::HashMap;
    

    #[tokio::test]
    async fn test_sequential_chain_with_input_output_keys() {
        let config = OpenAIConfig {
            api_key: "your-api-key".to_string(),
            base_url: "https://api.openai-proxy.org/v1".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            streaming: false,
        };

        let llm = LLM::new(config);
        // Chain A: 输入 topic/action，输出 explanation
        let chain_a = PromptChain::new(
            llm.clone(),
            ChatPromptTemplate::new(vec![
                Message::system("你是一位教授"),
                Message::human("请详细计算：{topic}，"),
            ]),
            vec!["topic"   ], // input_keys
            "explanation"   , // output_key
        );

        // Chain B: 输入 explanation，输出 summary
        let chain_b = PromptChain::new(
            llm,
            ChatPromptTemplate::new(vec![
                Message::system("你是数学家"),
                Message::human("请计算{explanation}乘100为多少"),
            ]),
            vec!["explanation"], // input_keys
            "summary",           // output_key
        );

        let seq_chain = SequentialChain::new(
            vec![Box::new(chain_a), Box::new(chain_b)],
            vec!["topic"   ],                  // 整体输入
            vec!["explanation"   , "summary"   ], // 整体输出
        );

        let input: HashMap<&str, &str> =
            HashMap::from([("topic"   , "1+3"   )]);

        let result = seq_chain.call(&input).await.unwrap();

        println!("详细解释:\n{}", result["explanation"]);
        println!("\n总结:\n{}", result["summary"]);
    }
}
