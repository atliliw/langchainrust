#[path = "common.rs"]
mod common;

use langchainrust::agent::{AgentExecutor, ReActAgent};
use langchainrust::llms::LLM;
use langchainrust::memory::SimpleMemory;
use langchainrust::retrieval::{
    Document, InMemoryVectorStore, MockEmbeddingModel, RecursiveCharacterSplitter,
    Retriever, SimilarityRetriever, TextSplitter,
};
use std::sync::Arc;

/// 创建测试用的检索器
async fn create_test_retriever() -> Arc<dyn Retriever> {
    let docs = vec![
        Document::new(
            "Rust是一种系统编程语言，由Mozilla开发。它注重内存安全、并发性能和执行效率。\
             Rust使用所有权系统来管理内存，不需要垃圾回收器。"
                .to_string(),
        ),
        Document::new(
            "Python是一种解释型、高级编程语言。它的设计哲学强调代码可读性，\
             使用缩进来划分代码块。Python广泛应用于Web开发、数据科学、人工智能等领域。"
                .to_string(),
        ),
        Document::new(
            "JavaScript是一种脚本语言，主要用于网页开发。\
             它可以在浏览器中运行，也可以通过Node.js在服务器端运行。\
             JavaScript是Web开发的三大核心技术之一。"
                .to_string(),
        ),
    ];

    let splitter = RecursiveCharacterSplitter::new(100, 20);
    let mut all_chunks = Vec::new();
    for doc in docs {
        let chunks = splitter.split_document(&doc).unwrap();
        all_chunks.extend(chunks);
    }

    println!("文档被分割为 {} 个块", all_chunks.len());

    let embedding_model = Arc::new(MockEmbeddingModel::new(128));
    let vector_store = Box::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(vector_store, embedding_model);
    retriever.add_documents(all_chunks).await.unwrap();
    println!("文档已添加到向量存储");

    Arc::new(retriever)
}

#[tokio::test]
async fn test_agent_with_retriever() {
    println!("\n=== ReActAgent + Retriever 测试 ===\n");

    let retriever = create_test_retriever().await;
    let llm = LLM::new(common::llm_config());

    // 使用 with_retriever 创建带检索功能的 Agent
    let agent = ReActAgent::with_retriever(
        llm,
        vec![],  // 无工具
        None,    // 无记忆
        retriever,
        3,       // top_k
    );
    let executor = AgentExecutor::new(Box::new(agent), vec![]).with_max_iterations(1);

    let result = executor.run_with_details("什么是 Rust 语言？").await.unwrap();
    println!("回答: {}", result.answer);
    println!("迭代次数: {}", result.iterations);

    assert!(!result.answer.is_empty());
}

#[tokio::test]
async fn test_agent_with_retriever_and_memory() {
    println!("\n=== ReActAgent + Retriever + Memory 测试 ===\n");

    let retriever = create_test_retriever().await;
    let llm = LLM::new(common::llm_config());

    let agent = ReActAgent::with_retriever(
        llm,
        vec![],
        Some(Box::new(SimpleMemory::default())),
        retriever,
        3,
    );
    let executor = AgentExecutor::new(Box::new(agent), vec![]).with_max_iterations(1);

    // 第一轮对话
    println!("第一轮对话:");
    let result1 = executor.run_with_details("介绍 Python 语言").await.unwrap();
    println!("回答: {}", result1.answer);

    // 第二轮对话
    println!("\n第二轮对话:");
    let result2 = executor.run_with_details("它和 Rust 有什么区别？").await.unwrap();
    println!("回答: {}", result2.answer);

    assert!(!result2.answer.is_empty());
}

#[tokio::test]
async fn test_agent_with_retriever_and_template() {
    println!("\n=== ReActAgent + Retriever + 自定义模板 测试 ===\n");

    use langchainrust::messages::Message;
    use langchainrust::prompts::ChatPromptTemplate;

    let retriever = create_test_retriever().await;
    let llm = LLM::new(common::llm_config());

    let template = ChatPromptTemplate::new(vec![
        Message::system(
            "你是一个专业的编程顾问。请根据参考资料回答问题。\n\
            回答要专业、准确、有条理。\n\n\
            参考资料：\n{context}",
        ),
        Message::human("问题：{input}"),
    ]);

    let agent = ReActAgent::with_retriever_and_template(
        llm,
        vec![],
        None,
        retriever,
        3,
        template,
    );
    let executor = AgentExecutor::new(Box::new(agent), vec![]).with_max_iterations(1);

    let result = executor.run_with_details("比较 Python 和 JavaScript").await.unwrap();
    println!("回答: {}", result.answer);

    assert!(!result.answer.is_empty());
}

#[tokio::test]
async fn test_agent_with_retriever_and_tools() {
    println!("\n=== ReActAgent + Retriever + Tools 测试 ===\n");

    use langchainrust::tools::Calculator;

    let retriever = create_test_retriever().await;
    let llm = LLM::new(common::llm_config());

    let tools: Vec<Arc<dyn langchainrust::tools::Tool>> = vec![Arc::new(Calculator)];

    let agent = ReActAgent::with_retriever(
        llm,
        tools.clone(),
        None,
        retriever,
        2,
    );
    let executor = AgentExecutor::new(Box::new(agent), tools).with_max_iterations(3);

    // 问题可能需要检索 + 工具
    let result = executor
        .run_with_details("Rust 有什么特点？另外计算 10 + 20")
        .await
        .unwrap();

    println!("回答: {}", result.answer);
    println!("是否使用工具: {}", result.used_tools);

    assert!(!result.answer.is_empty());
}

#[tokio::test]
async fn test_agent_without_retriever() {
    println!("\n=== ReActAgent 不带 Retriever 测试（普通模式）===\n");

    let llm = LLM::new(common::llm_config());

    // 不传 retriever，走普通逻辑
    let agent = ReActAgent::new(llm, vec![], None);
    let executor = AgentExecutor::new(Box::new(agent), vec![]).with_max_iterations(1);

    let result = executor.run_with_details("你好").await.unwrap();
    println!("回答: {}", result.answer);

    assert!(!result.answer.is_empty());
}

#[tokio::test]
async fn test_full_rag_workflow() {
    println!("\n=== 完整 RAG 工作流演示 ===\n");

    // 1. 创建知识库
    println!("步骤1：创建知识库");
    let knowledge_docs = vec![
        Document::new(
            "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。\
             它提供了标准的接口，可以轻松地与大语言模型（LLM）进行交互。"
                .to_string(),
        ),
        Document::new(
            "RAG（Retrieval-Augmented Generation）是一种将检索和生成结合的技术。\
             它首先从知识库中检索相关文档，然后将这些文档作为上下文提供给大语言模型。"
                .to_string(),
        ),
    ];

    // 2. 分割文档
    println!("步骤2：分割文档");
    let splitter = RecursiveCharacterSplitter::new(100, 20);
    let mut chunks = Vec::new();
    for doc in knowledge_docs {
        let doc_chunks = splitter.split_document(&doc).unwrap();
        chunks.extend(doc_chunks);
    }
    println!("  - 分割后共 {} 个文档块", chunks.len());

    // 3. 创建向量存储和检索器
    println!("步骤3：创建向量存储");
    let embedding_model = Arc::new(MockEmbeddingModel::new(128));
    let vector_store = Box::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(vector_store, embedding_model);
    retriever.add_documents(chunks).await.unwrap();
    println!("  - 文档已添加到向量存储");

    // 4. 创建 Agent
    println!("步骤4：创建 RAG Agent");
    let llm = LLM::new(common::llm_config());
    let agent = ReActAgent::with_retriever(
        llm,
        vec![],
        Some(Box::new(SimpleMemory::default())),
        Arc::new(retriever) as Arc<dyn Retriever>,
        2,
    );
    let executor = AgentExecutor::new(Box::new(agent), vec![]);

    // 5. 查询
    println!("\n步骤5：执行查询\n");
    let result = executor.run_with_details("什么是 RAG？").await.unwrap();
    println!("回答: {}", result.answer);

    assert!(!result.answer.is_empty());
}
