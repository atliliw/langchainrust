#[path = "common.rs"]
mod common;

#[cfg(test)]
mod tests {
    use crate::common::llm_Config;
    use langchainrust::chains::{PromptChain, SequentialChain};
    use langchainrust::llms::LLM;
    use langchainrust::memory::SimpleMemory;
    use langchainrust::messages::Message;
    use langchainrust::prompts::ChatPromptTemplate;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_sequential_chain_with_input_output_keys() {
        let config = llm_Config();
        let llm = LLM::new(config);
        let chain_a = PromptChain::new(
            llm.clone(),
            ChatPromptTemplate::new(vec![
                Message::system("你是一位教授"),
                Message::human("请详细计算：{topic}，"),
            ]),
            vec!["topic"], // input_keys
            "explanation", // output_key
        );
        // Chain B: 输入 explanation，输出 summary
        let chain_b = PromptChain::new(
            llm,
            ChatPromptTemplate::new(vec![
                Message::system("你是数学家小李"),
                Message::human("请计算{explanation}乘100为多少"),
                Message::human("你名字是什么"),
            ]),
            vec!["explanation"], // input_keys
            "summary",           // output_key
        );

        let mut seq_chain = SequentialChain::new(
            vec![Box::new(chain_a), Box::new(chain_b)],
            vec!["topic"],                  // 整体输入
            vec!["explanation", "summary"], // 整体输出
            true,
            Some(Box::new(SimpleMemory::default())), // 传入 memory！
        );

        let input: HashMap<&str, &str> = HashMap::from([("topic", "1+3")]);

        let result = seq_chain.call(&input).await.unwrap();

        println!("详细解释:\n{}", result["explanation"]);
        println!("\n总结:\n{}", result["summary"]);
    }
}
