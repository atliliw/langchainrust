#[path = "common.rs"]
mod common;

use common::create_test_llm;
use langchainrust::messages::Message;
pub use langchainrust::prompts::ChatPromptTemplate;

#[tokio::test]
#[ignore]
async fn test_with_template() {
    #[tokio::test]
    async fn test_only() {
        let llm = create_test_llm();

        let template = ChatPromptTemplate::new(vec![
            Message::system(
                "你是由{name}开发的AI助手，专精于{field}领域。请用清晰易懂的方式回答问题。",
            ),
            Message::human("请向初学者解释{topic}是什么。"),
        ]);

        let mut values = std::collections::HashMap::new();
        values.insert("name", "阿里云");
        values.insert("field", "人工智能");
        values.insert("topic", "大语言模型");

        let result = llm.invoke_chat_template(&template, &values).await;
        println!("{:?}", result);
        assert!(
            result.is_ok(),
            "Template invocation failed: {:?}",
            result.err()
        );
    }
}

#[tokio::test]
async fn test_batch_limited() {
    let llm = create_test_llm();

    let template1 = ChatPromptTemplate::new(vec![
        Message::system("你是数学老师"),
        Message::human("解释{topic}"),
    ]);

    let template2 = ChatPromptTemplate::new(vec![
        Message::system("你是编程导师"),
        Message::human("用Rust写一个{topic}的例子"),
    ]);

    let mut values1 = std::collections::HashMap::new();
    values1.insert("topic", "导数");

    let mut values2 = std::collections::HashMap::new();
    values2.insert("topic", "斐波那契数列");

    let pairs = vec![(template1, values1), (template2, values2)];

    // 调用方法并处理结果
    match llm.invoke_chat_template_batch_limited(&pairs, 2).await {
        Ok(results) => {
            println!("\n✅ 批量调用成功！共 {} 个结果:\n", results.len());
            for (i, response) in results.iter().enumerate() {
                println!("--- 响应 #{} ---\n{}\n", i + 1, response);
            }
        }
        Err(e) => {
            eprintln!("\n❌ 调用失败: {:?}", e);
            panic!("测试因 LLM 调用失败而终止");
        }
    }
}
