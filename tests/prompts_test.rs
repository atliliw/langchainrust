use langchainrust::messages::Message;
use langchainrust::prompts::{ChatPromptTemplate, PromptTemplate};
use std::collections::HashMap;
#[path = "common.rs"]
mod common;

#[cfg(test)]
mod tests {
    use crate::common::create_test_llm;
    use langchainrust::messages::Message;
    use langchainrust::prompts::{ChatPromptTemplate, PromptTemplate};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_llm_generate_promptTemplate() {
        let template = PromptTemplate::new("请用{tone}风格解释{topic}。");
        let values = HashMap::from([("tone", "简洁"), ("topic", "Rust的所有权")]);
        let prompt = template.format(&values).unwrap();
        assert_eq!(prompt, "请用简洁风格解释Rust的所有权。");
        //调用他
        let LLM = create_test_llm();
        let result = LLM.invoke(&prompt).await;
        println!("{:?}", result);
    }

    #[test]
    fn test_chat_prompt_format() {
        let chat_prompt = ChatPromptTemplate::new(vec![
            Message::system("你是一个{role}助手。".to_string()),
            Message::human("你好，{name}！".to_string()),
        ]);
        let values = HashMap::from([("role", "编程"), ("name", "Alice")]);
        let messages = chat_prompt.format(&values).unwrap();
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_prompt_from_user_perspective() {
        let template = "Hi, {user}! You have {count} messages.";
        let prompt = PromptTemplate::new(template);

        let mut input = HashMap::new();
        input.insert("user", "Bob");
        input.insert("count", "5");

        let result = prompt.format(&input).unwrap();
        assert_eq!(result, "Hi, Bob! You have 5 messages.");
    }
}
