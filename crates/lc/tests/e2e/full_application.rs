//! 端到端测试 - 完整 AI 应用

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::schema::Message;
use langchainrust::BaseChatModel;
use langchainrust::{
    AgentExecutor, BaseAgent, BaseTool, Calculator, ChatMessageHistory, Document,
    InMemoryVectorStore, ReActAgent, RecursiveCharacterSplitter, RetrieverTrait,
    SimilarityRetriever, SimpleMathTool, TextSplitter, VectorStore,
};
use std::sync::Arc;

/// 完整 AI 应用测试 - 所有组件协同工作
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_full_ai_application() {
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());
    let llm = config.openai_chat();
    let mut history = ChatMessageHistory::new();

    let tools: Vec<Arc<dyn BaseTool>> =
        vec![Arc::new(Calculator::new()), Arc::new(SimpleMathTool::new())];

    let agent = ReActAgent::new(config.openai_chat(), tools.clone(), None);
    let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
        .with_max_iterations(5)
        .with_verbose(true);

    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());

    println!("=== 构建知识库 ===");

    let knowledge = [
        Document::new("公司名称：智能科技有限公司。成立年份：2015年。员工人数：500人。"),
        Document::new("主要产品：智能音箱、智能灯泡。年营业额：2亿元。"),
        Document::new("公司地址：北京市海淀区科技园路100号。邮编：100080。"),
    ];

    let splitter = RecursiveCharacterSplitter::new(100, 20);
    let chunks: Vec<Document> = knowledge
        .iter()
        .flat_map(|doc| {
            splitter
                .split_text(doc.page_content())
                .into_iter()
                .map(Document::new)
                .collect::<Vec<_>>()
        })
        .collect();

    retriever.add_documents(chunks).await.unwrap();
    println!("知识库已建立，共 {} 个文档块", store.count().await);

    // 场景1：RAG 知识问答
    println!("\n=== 场景1：RAG 知识问答 ===");

    let query1 = "公司是什么时候成立的？";
    let docs1 = retriever.retrieve(query1, 2).await.unwrap();
    let context1 = docs1
        .iter()
        .map(|d| d.page_content())
        .collect::<Vec<_>>()
        .join("\n");

    let messages1 = vec![
        Message::system("根据资料回答问题。"),
        Message::human(format!("资料：{}\n\n问题：{}", context1, query1)),
    ];

    let answer1 = llm.chat(messages1, None).await.unwrap();
    println!("Q: {}", query1);
    println!("A: {}", answer1.content);

    history.add_message(Message::human(query1));
    history.add_message(Message::ai(&answer1.content));

    // 场景2：工具调用计算
    println!("\n=== 场景2：工具调用计算 ===");

    let query2 = "公司有500人，每人创造40万元产值，总产值是多少？";
    println!("Q: {}", query2);

    let answer2 = executor.invoke(query2.to_string()).await.unwrap();
    println!("A: {}", answer2);

    assert!(answer2.contains("20000") || answer2.contains("2亿"));

    history.add_message(Message::human(query2));
    history.add_message(Message::ai(&answer2));

    // 场景3：记忆测试
    println!("\n=== 场景3：记忆测试 ===");

    let query3 = "公司成立几年了？（假设现在是2024年）";

    let mut messages3 = vec![Message::system("记得之前的对话内容。")];
    messages3.extend(history.messages().iter().cloned());

    let docs3 = retriever.retrieve("成立年份 2015", 1).await.unwrap();
    let context3 = docs3.first().map(|d| d.page_content()).unwrap_or("");

    messages3.push(Message::human(format!(
        "资料：{}\n\n问题：{}",
        context3, query3
    )));

    let answer3 = llm.chat(messages3, None).await.unwrap();
    println!("Q: {}", query3);
    println!("A: {}", answer3.content);

    // 场景4：组合查询
    println!("\n=== 场景4：组合查询（RAG + 工具） ===");

    let query4 = "根据公司邮编100080，计算所有数字之和";
    println!("Q: {}", query4);

    let answer4 = executor.invoke(query4.to_string()).await.unwrap();
    println!("A: {}", answer4);

    println!("\n=== 完整 AI 应用测试通过 ===");
}

/// 智能路由测试
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_intelligent_routing() {
    let config = TestConfig::get();
    let embeddings = Arc::new(config.embeddings());

    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());

    let docs = vec![
        Document::new("产品A的价格是100元。产品B的价格是200元。"),
        Document::new("运费规则：订单满300元免运费。"),
    ];

    retriever.add_documents(docs).await.unwrap();

    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];

    let agent = ReActAgent::new(config.openai_chat(), tools.clone(), None);
    let executor =
        AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools).with_max_iterations(3);

    println!("=== 智能路由测试 ===");

    // 测试1：知识问题
    println!("\n--- 知识问题（RAG）---");
    let q1 = "产品A多少钱？";
    let docs1 = retriever.retrieve(q1, 1).await.unwrap();
    println!("问题: {}", q1);
    println!("检索结果: {}", docs1[0].page_content());

    // 测试2：计算问题
    println!("\n--- 计算问题（Agent）---");
    let q2 = "What is 100 + 200?";
    println!("问题: {}", q2);
    let answer2 = executor.invoke(q2.to_string()).await.unwrap();
    println!("答案: {}", answer2);

    // 测试3：组合问题
    println!("\n--- 组合问题（RAG + Agent）---");
    let q3 = "买产品A和B，需要付运费吗？";
    println!("问题: {}", q3);

    let docs3 = retriever.retrieve("产品 价格", 2).await.unwrap();
    let context3 = docs3
        .iter()
        .map(|d| d.page_content())
        .collect::<Vec<_>>()
        .join(" ");

    let full_q3 = format!(
        "Context: {}. Question: Buy A (100元) and B (200元), free shipping?",
        context3
    );

    let answer3 = executor.invoke(full_q3).await.unwrap();
    println!("答案: {}", answer3);
}
