//! CorrectiveRAG 示例
//!
//! 展示 CorrectiveRAGAgent 的检索-评分-纠错-幻觉检测流程。
//!
//! # 运行
//! ```bash
//! cargo run --example rag_corrective_rag
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)
//! - `OPENAI_BASE_URL`:API 基址(可选)

use langchainrust::embeddings::{Embeddings, MockEmbeddings};
use langchainrust::retrieval::SimilarityRetriever;
use langchainrust::tools::DuckDuckGoSearchTool;
use langchainrust::vector_stores::{InMemoryVectorStore, VectorStore};
use langchainrust::{CorrectiveRAGAgent, Document, OpenAIChat, OpenAIConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 配置 LLM
    let api_key = std::env::var("OPENAI_API_KEY").expect("请设置 OPENAI_API_KEY 环境变量");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let llm = OpenAIChat::new(OpenAIConfig {
        api_key,
        base_url,
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    });

    // 2. 准备检索器（InMemoryVectorStore + MockEmbeddings,演示用）
    let store = Arc::new(InMemoryVectorStore::new());
    let embeddings = Arc::new(MockEmbeddings::new(3));

    // 先添加文档到向量存储
    let docs = vec![
        Document::new("Rust 是一门系统编程语言,由 Mozilla 开发,注重安全和性能。"),
        Document::new("Rust 的核心特性包括所有权系统、借用检查和零成本抽象。"),
        Document::new("Rust 的包管理器叫 Cargo,支持依赖管理和构建自动化。"),
    ];
    let doc_texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
    let doc_embeddings = embeddings.embed_documents(&doc_texts).await?;
    store.add_documents(docs, doc_embeddings).await?;

    let retriever = SimilarityRetriever::new(store, embeddings);

    // 3. 创建 CRAG Agent
    let agent = CorrectiveRAGAgent::new(llm, retriever)
        .with_grade_threshold(0.5) // 文档评分低于 0.5 触发纠错
        .with_web_fallback(Box::new(DuckDuckGoSearchTool::new())) // 可选:Web 搜索 fallback
        .with_hallucination_check(true); // 可选:幻觉检测

    // 4. 查询
    let result = agent.invoke("Rust 语言有哪些核心特性?").await?;

    println!("回答: {}", result.answer);
    println!("是否基于文档: {}", result.grounded);
    println!("来源文档数: {}", result.sources.len());
    for (i, score) in result.grade_scores.iter().enumerate() {
        println!("  文档[{}] 评分: {:.2}", i, score);
    }

    Ok(())
}
