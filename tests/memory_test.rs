// #[path = "common.rs"]
// mod common;
// use common::create_test_llm;
//
// use std::collections::HashMap;
// use langchainrust::llms::{OpenAIConfig, LLM};
// use langchainrust::memory::{Memory};
// use langchainrust::memory::chat_message_history::ChatMessageHistory;
// use langchainrust::prompts::ChatPromptTemplate;
// use langchainrust::messages::Message;
//
//
// #[tokio::test]
// async fn test_chat_message_history() {
//     let mut history = ChatMessageHistory::new();
//
//     // 添加对话
//     history.add_user_message("你好");
//     history.add_ai_message("你好！我是AI");
//
//     history.add_user_message("今天天气怎么样？");
//     history.add_ai_message("晴天，适合出门。");
//
//     // 获取变量（用于 prompt）
//     let vars = history.load_memory_variables();
//     assert!(vars.contains_key("chat_history"));
//     println!("历史记录:\n{}", vars["chat_history"]);
//
//     // 验证保存上下文功能
//     history.save_context("我饿了", "建议吃点饭");
//     assert_eq!(history.get_messages().len(), 6); // 原来4条 + 新增2条
// }
//
//
// #[tokio::test]
// async fn test_llm_with_chat_history1() {
//     let llm =create_test_llm();
//     let mut history = ChatMessageHistory::new();
//
//     let template = ChatPromptTemplate::new(vec![
//         Message::system("你是一个AI助手，能记住之前的对话。"),
//         Message::human("{chat_history}\n\n用户问: {input}"),
//     ]);
//
//     // === 第一轮 ===
//     let values1 = [("input", "你好")].into_iter().collect::<HashMap<_, _>>();
//
//     // ✅ 关键：先保存 load_memory_variables() 的结果
//     let memory_vars1 = history.load_memory_variables(); // 👈 活到作用域结束
//
//     let all_values1: HashMap<&str, &str> = memory_vars1
//         .iter()
//         .map(|(k, v)| (k.as_str(), v.as_str()))
//         .chain(values1.iter().map(|(k, v)| (*k, *v)))
//         .collect();
//
//     let response1 = llm.invoke_chat_template(&template, &all_values1).await.unwrap();
//     println!("AI 1: {}", response1);
//     history.save_context("你好"  , &response1);
//
//     // === 第二轮 ===
//     let values2 = [("input", "我刚才说了什么？")].into_iter().collect::<HashMap<_, _>>();
//
//     let memory_vars2 = history.load_memory_variables(); // 👈 再次绑定
//
//     let all_values2: HashMap<&str, &str> = memory_vars2
//         .iter()
//         .map(|(k, v)| (k.as_str(), v.as_str()))
//         .chain(values2.iter().map(|(k, v)| (*k, *v)))
//         .collect();
//
//     let response2 = llm.invoke_chat_template(&template, &all_values2).await.unwrap();
//     println!("AI 2: {}", response2);
// }



#[path = "common.rs"]
mod common;

#[cfg(test)]
mod tests {
    use langchainrust::chains::{PromptChain, SequentialChain};
    use langchainrust::llms::{LLM,OpenAIConfig};
    use langchainrust::messages::Message;
    use langchainrust::prompts::ChatPromptTemplate;
    use std::collections::HashMap;
    use crate::common::llm_Config;
    use langchainrust::memory::{SimpleMemory};



    #[tokio::test]
    async fn test_sequential_chain_with_input_output_keys() {
        let config = llm_Config();
        let llm = LLM::new(config);
        let chain_a = PromptChain::new(
            llm.clone(),
            ChatPromptTemplate::new(vec![
                Message::system("你是一位教授你是名字叫小李"),
                Message::human("请详细计算：{topic}，"),
            ]),
            vec!["topic"], // input_keys
            "explanation"   , // output_key
        );
        // Chain B: 输入 explanation，输出 summary
        let chain_b = PromptChain::new(
            llm,
            ChatPromptTemplate::new(vec![
                Message::system("你是数学家"),
                Message::human("请计算{explanation}乘100为多少,然后告诉我你的名字"),
            ]),
            vec!["explanation"], // input_keys
            "summary", // output_key
        );

        let mut seq_chain = SequentialChain::new(
            vec![Box::new(chain_a), Box::new(chain_b)],
            vec!["topic"],
            vec!["explanation", "summary"],
            true,
            Some(Box::new(SimpleMemory::default()))
        );

        let input: HashMap<&str, &str> =
            HashMap::from([("topic", "1+3")]);

        let result = seq_chain.call(&input).await.unwrap();

        println!("详细解释:\n{}", result["explanation"]);
        println!("\n总结:\n{}", result["summary"]);
    }
}
