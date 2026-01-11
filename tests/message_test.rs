// use langchainrust::llms::LLM;
// use langchainrust::messages::{Message, HumanMessage, SystemMessage};
// use langchainrust::chains::Chain;
// use langchainrust::prompts::PromptTemplate;
//
//
// #[path = "common.rs"]
// mod common;
// use common::create_test_llm;
//
//
// #[tokio::test]
// async fn test_llm_with_messages() {
//     let llm = create_test_llm();
//
//     // Test with system and human message
//     let result = llm.chatModel(
//         Some("You are a helpful assistant."),
//         "What is the capital of France?"
//     ).await;
//
//     println!("Chat result: {:?}", result);
//     assert!(result.is_ok());
// }
//
// #[tokio::test]
// async fn test_llm_with_message_history() {
//     let llm = create_test_llm();
//
//     let messages = vec![
//         Message::system("You are a helpful assistant."),
//         Message::human("What is the capital of France?"),
//         Message::ai("The capital of France is Paris."),
//         Message::human("What is its population?"),
//     ];
//
//     let result = llm.generate_with_messages(messages).await;
//     println!("Message history result: {:?}", result);
//     assert!(result.is_ok());
// }
//
// #[tokio::test]
// async fn test_chain_with_system_message() {
//     let llm = create_test_llm();
//     let prompt = PromptTemplate::new(
//         "Tell me a fact about {topic}",
//     );
//
//     let mut inputs = std::collections::HashMap::new();
//     inputs.insert("topic", "cats");
//
// }