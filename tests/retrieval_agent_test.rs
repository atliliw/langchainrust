#[path = "common.rs"]
mod common;

use langchainrust::agent::{AgentExecutor, RetrievalAgent, SimpleRetrievalAgent};
use langchainrust::llms::LLM;
use langchainrust::memory::SimpleMemory;
use langchainrust::retrieval::{
    Document, DocumentChunk, InMemoryVectorStore, MockEmbeddingModel, RecursiveCharacterSplitter,
    Retriever, SimilarityRetriever, TextSplitter,
};
use std::sync::Arc;

/// 创建测试用的检索器
async fn create_test_retriever() -> Arc<dyn Retriever> {
    // 1. 创建测试文档
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
        Document::new(
            "苹果是一种常见的水果，富含维生素C和膳食纤维。\
             苹果可以生吃，也可以加工成苹果汁、苹果酱等产品。\
             常见的苹果品种有红富士、黄元帅、青苹果等。"
                .to_string(),
        ),
    ];

    // 2. 分割文档
    let splitter = RecursiveCharacterSplitter::new(100, 20);
    let mut all_chunks = Vec::new();
    for doc in docs {
        let chunks = splitter.split_document(&doc).unwrap();
        all_chunks.extend(chunks);
    }

    println!("文档被分割为 {} 个块", all_chunks.len());

    // 3. 创建检索器
    let embedding_model = Arc::new(MockEmbeddingModel::new(128));
    let vector_store = Box::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(vector_store, embedding_model);

    // 4. 添加文档
    retriever.add_documents(all_chunks).await.unwrap();
    println!("文档已添加到向量存储");

    Arc::new(retriever)
}

#[tokio::test]
async fn test_simple_retrieval_agent() {
    println!("\n=== SimpleRetrievalAgent 测试 ===\n");

    let retriever = create_test_retriever().await;
    let llm = LLM::new(common::llm_config());

    let agent = SimpleRetrievalAgent::new(llm, retriever, 3);

    // 测试1：编程相关的问题
    println!("问题1: 什么是 Rust 语言？");
    let answer = agent.query("什么是 Rust 语言？请简要介绍").await.unwrap();
    println!("回答: {}\n", answer);
    assert!(!answer.is_empty());

    // 测试2：另一个问题
    println!("问题2: Python 有什么特点？");
    let answer = agent.query("Python 有什么特点？").await.unwrap();
    println!("回答: {}\n", answer);
    assert!(!answer.is_empty());
}

#[tokio::test]
async fn test_retrieval_agent_with_executor() {
    println!("\n=== RetrievalAgent + AgentExecutor 测试 ===\n");

    let retriever = create_test_retriever().await;
    let llm = LLM::new(common::llm_config());

    let agent = RetrievalAgent::new(llm, retriever, Some(Box::new(SimpleMemory::default())), 3);
    let executor = AgentExecutor::new(Box::new(agent), vec![]).with_max_iterations(1);

    println!("问题: 介绍 JavaScript 的用途");
    let result = executor.run_with_details("介绍 JavaScript 的用途").await.unwrap();

    println!("\n最终答案: {}", result.answer);
    println!("迭代次数: {}", result.iterations);

    assert!(!result.answer.is_empty());
}

#[tokio::test]
async fn test_retrieval_agent_with_memory() {
    println!("\n=== RetrievalAgent 记忆功能测试 ===\n");

    let retriever = create_test_retriever().await;
    let llm = LLM::new(common::llm_config());

    let agent = RetrievalAgent::new(llm, retriever, Some(Box::new(SimpleMemory::default())), 3);
    let executor = AgentExecutor::new(Box::new(agent), vec![]).with_max_iterations(1);

    // 第一次问答
    println!("第一轮对话:");
    let result1 = executor.run_with_details("什么是 Rust？").await.unwrap();
    println!("回答: {}", result1.answer);

    // 第二次问答（应该能记住之前的对话）
    println!("\n第二轮对话:");
    let result2 = executor.run_with_details("它和 Python 有什么区别？").await.unwrap();
    println!("回答: {}", result2.answer);

    assert!(!result2.answer.is_empty());
}

#[tokio::test]
async fn test_retrieval_agent_custom_template() {
    println!("\n=== RetrievalAgent 自定义模板测试 ===\n");

    use langchainrust::messages::Message;
    use langchainrust::prompts::ChatPromptTemplate;

    let retriever = create_test_retriever().await;
    let llm = LLM::new(common::llm_config());

    // 自定义模板
    let template = ChatPromptTemplate::new(vec![
        Message::system(
            "你是一个专业的编程顾问。请根据以下参考资料回答用户问题。\n\
            回答要专业、准确、有条理。\n\n\
            参考资料：\n{context}",
        ),
        Message::human("问题：{input}\n\n请给出详细的回答。"),
    ]);

    let agent = RetrievalAgent::with_template(
        llm,
        retriever,
        None,
        3,
        template,
    );
    let executor = AgentExecutor::new(Box::new(agent), vec![]).with_max_iterations(1);

    let result = executor.run_with_details("比较几种编程语言的特点").await.unwrap();
    println!("回答: {}", result.answer);

    assert!(!result.answer.is_empty());
}

#[tokio::test]
async fn test_retrieval_workflow_demo() {
    println!("\n=== 完整 RAG 工作流演示 ===\n");

    // 步骤1：创建知识库文档
    println!("步骤1：创建知识库文档");
    let knowledge_docs = vec![
        Document::new(
            "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。\
             它提供了标准的接口，可以轻松地与大语言模型（LLM）进行交互。\
             LangChain 支持多种 LLM 提供商，包括 OpenAI、Anthropic、Google 等。"
                .to_string(),
        )
        .with_metadata("category".to_string(), "framework".to_string()),
        Document::new(
            "向量数据库是一种专门用于存储和检索高维向量数据的数据库。\
             常见的向量数据库包括 Pinecone、Weaviate、Milvus、Qdrant 等。\
             向量数据库是构建 RAG（检索增强生成）应用的核心组件。"
                .to_string(),
        )
        .with_metadata("category".to_string(), "database".to_string()),
        Document::new(
            "RAG（Retrieval-Augmented Generation）是一种将检索和生成结合的技术。\
             它首先从知识库中检索相关文档，然后将这些文档作为上下文提供给大语言模型。\
             RAG 可以帮助模型获取最新信息，减少幻觉，提高回答的准确性。"
                .to_string(),
        )
        .with_metadata("category".to_string(), "technique".to_string()),
    ];

    // 步骤2：文档分割
    println!("步骤2：文档分割");
    let splitter = RecursiveCharacterSplitter::new(150, 30);
    let mut chunks = Vec::new();
    for doc in knowledge_docs {
        let doc_chunks = splitter.split_document(&doc).unwrap();
        chunks.extend(doc_chunks);
    }
    println!("  - 分割后共 {} 个文档块", chunks.len());

    // 步骤3：创建向量存储和检索器
    println!("步骤3：创建向量存储和检索器");
    let embedding_model = Arc::new(MockEmbeddingModel::new(256));
    let vector_store = Box::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(vector_store, embedding_model);
    retriever.add_documents(chunks).await.unwrap();
    println!("  - 文档已添加到向量存储");

    // 步骤4：创建 RAG Agent
    println!("步骤4：创建 RAG Agent");
    let llm = LLM::new(common::llm_config());
    let agent = SimpleRetrievalAgent::new(llm, Arc::new(retriever) as Arc<dyn Retriever>, 2);

    // 步骤5：查询
    println!("\n步骤5：执行查询\n");

    let questions = vec![
        "什么是 RAG？",
        "LangChain 是什么？",
        "向量数据库有什么用？",
    ];

    for question in questions {
        println!("Q: {}", question);
        let answer = agent.query(question).await.unwrap();
        println!("A: {}", answer);
        println!();
    }
}
