// examples/advanced/rag_demo.rs
//! 高级示例 1: RAG (检索增强生成)
//!
//! 运行: cargo run --example rag_demo
//!
//! 功能: 演示完整的 RAG 流程：文档分割 -> 向量化 -> 存储 -> 检索 -> 生成答案

use langchainrust::{
    Document, InMemoryVectorStore,
    OpenAIEmbeddings, OpenAIEmbeddingsConfig, Embeddings,
    SimilarityRetriever, RetrieverTrait, TextSplitter, RecursiveCharacterSplitter,
    OpenAIChat, OpenAIConfig, BaseChatModel,
};
use langchainrust::schema::Message;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 高级示例 1: RAG (检索增强生成) ===\n");
    
    // ========== 1. 准备知识库文档 ==========
    println!("--- 步骤 1: 准备知识库文档 ---\n");
    
    let documents = vec![
        Document::new(
            "Rust 是一种系统编程语言，由 Mozilla 研发。\
             它注重内存安全、并发性能和速度。\
             Rust 使用所有权系统来管理内存，无需垃圾回收器。\
             Rust 的语法类似于 C++，但提供了更强的安全保证。"
        ),
        Document::new(
            "Python 是一种高级编程语言，由 Guido van Rossum 创建于 1991 年。\
             Python 以其简洁的语法和丰富的生态系统著称。\
             它广泛用于数据科学、机器学习、Web 开发和自动化脚本。\
             Python 是解释型语言，运行速度相对较慢。"
        ),
        Document::new(
            "JavaScript 是网页开发的核心语言，最初由 Netscape 创建于 1995 年。\
             它可以在浏览器和服务器端(Node.js)运行。\
             JavaScript 是动态类型语言，支持函数式和面向对象编程。\
             现代前端框架如 React、Vue 和 Angular 都基于 JavaScript。"
        ),
        Document::new(
            "Go 是由 Google 开发的编程语言，于 2009 年发布。\
             Go 的设计目标是简洁、高效和易于并发。\
             它常用于云服务、微服务和网络编程。\
             Go 编译速度快，生成的可执行文件性能优秀。"
        ),
    ];
    
    println!("已准备 {} 个文档作为知识库\n", documents.len());
    
    // ========== 2. 文档分割 ==========
    println!("--- 步骤 2: 文档分割 ---\n");
    
    let splitter = RecursiveCharacterSplitter::new(200, 50);
    let mut all_chunks = Vec::new();
    
    for doc in &documents {
        let chunks = splitter.split_document(doc);
        all_chunks.extend(chunks);
    }
    
    println!("分割后共 {} 个文本块\n", all_chunks.len());
    
    // ========== 3. 创建 Embedding 模型和向量存储 ==========
    println!("--- 步骤 3: 创建向量存储 ---\n");
    
    let embedding_config = OpenAIEmbeddingsConfig {
        api_key: std::env::var("OPENAI_API_KEY")
            .unwrap_or_else(|_| "your-api-key-here".to_string()),
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        model: "text-embedding-ada-002".to_string(),
        batch_size: 2048,
    };
    
    let embeddings = Arc::new(OpenAIEmbeddings::new(embedding_config));
    let store = Arc::new(InMemoryVectorStore::new());
    
    println!("Embedding 模型: {}", embeddings.model_name());
    println!("向量维度: {}\n", embeddings.dimension());
    
    // ========== 4. 创建检索器并添加文档 ==========
    println!("--- 步骤 4: 添加文档到向量存储 ---\n");
    
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());
    
    match retriever.add_documents(all_chunks.clone()).await {
        Ok(_) => println!("成功添加 {} 个文档块到向量存储\n", all_chunks.len()),
        Err(e) => {
            eprintln!("添加文档失败: {}", e);
            eprintln!("提示: 请确保设置了正确的 OPENAI_API_KEY 环境变量");
            return Ok(());
        }
    }
    
    // ========== 5. 检索相关文档 ==========
    println!("--- 步骤 5: 检索相关文档 ---\n");
    
    let queries = vec![
        "哪个语言适合数据科学？",
        "Rust 有什么特点？",
        "用于网页开发的语言是什么？",
        "哪种语言编译速度最快？",
    ];
    
    for query in &queries {
        println!("查询: {}", query);
        
        match retriever.retrieve_with_scores(query, 3).await {
            Ok(results) => {
                println!("检索到 {} 个相关文档:", results.len());
                for (i, result) in results.iter().enumerate() {
                    println!("  [{}] 相似度: {:.4}", i + 1, result.score);
                    println!("      内容: {}...", result.document.content.chars().take(60).collect::<String>());
                }
                println!();
            }
            Err(e) => {
                eprintln!("检索失败: {}\n", e);
            }
        }
    }
    
    // ========== 6. RAG: 检索 + 生成 ==========
    println!("--- 步骤 6: RAG 完整流程 (检索 + 生成) ---\n");
    
    let llm_config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")
            .unwrap_or_else(|_| "your-api-key-here".to_string()),
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        temperature: Some(0.3),
        max_tokens: Some(500),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        organization: None,
    };
    
    let llm = OpenAIChat::new(llm_config);
    
    let rag_questions = vec![
        "我应该选择哪个语言来做数据分析？为什么？",
        "Rust 和 Go 有什么区别？",
    ];
    
    for question in &rag_questions {
        println!("问题: {}", question);
        
        // 检索相关文档
        let docs = retriever.retrieve(question, 3).await?;
        
        // 构建上下文
        let context: String = docs.iter()
            .map(|d| d.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        
        // 构建提示词
        let messages = vec![
            Message::system("你是一个知识渊博的助手。请根据提供的参考资料回答问题。\
                           如果参考资料中没有相关信息，请诚实说明。"),
            Message::human(format!("参考资料:\n{}\n\n问题: {}", context, question)),
        ];
        
        println!("\nLLM 回答:");
        match llm.chat(messages, None).await {
            Ok(response) => {
                println!("{}\n", response.content);
            }
            Err(e) => {
                eprintln!("生成失败: {}\n", e);
            }
        }
        
        println!("{}", "-".repeat(60));
        println!();
    }
    
    println!("=== 示例完成 ===");
    Ok(())
}