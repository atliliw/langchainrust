//! AdaptiveRAG 示例
//!
//! 展示 AdaptiveRAG 的自适应检索决策:
//! NoRetrieval(闲聊) / SingleSearch(具体问题) / MultiQuery(复杂问题)。
//!
//! # 运行
//! ```bash
//! cargo run --example rag_adaptive_rag
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)
//! - `OPENAI_BASE_URL`:API 基址(可选)

use langchainrust::embeddings::{Embeddings, MockEmbeddings};
use langchainrust::retrieval::SimilarityRetriever;
use langchainrust::vector_stores::{InMemoryVectorStore, VectorStore};
use langchainrust::{AdaptiveRAG, Document, OpenAIChat, OpenAIConfig};
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

    // 2. 准备检索器
    let store = Arc::new(InMemoryVectorStore::new());
    let embeddings = Arc::new(MockEmbeddings::new(3));

    let docs = vec![
        Document::new("Rust 是一门系统编程语言,注重安全和性能。"),
        Document::new("Rust 的所有权系统避免了数据竞争和空指针。"),
        Document::new("Tokio 是 Rust 最流行的异步运行时。"),
        Document::new("async-std 是另一个 Rust 异步运行时,API 更接近标准库。"),
    ];
    let doc_texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
    let doc_embeddings = embeddings.embed_documents(&doc_texts).await?;
    store.add_documents(docs, doc_embeddings).await?;

    let retriever = SimilarityRetriever::new(store, embeddings);

    // 3. 创建 AdaptiveRAG
    let rag = AdaptiveRAG::new(llm, retriever)
        .with_retrieve_k(4)
        .with_multi_query_count(3);

    // 4. 三种查询场景
    // 场景 1: 闲聊 — LLM 判断不需要检索
    let result = rag.invoke("你好,今天天气怎么样?").await?;
    print_result("[NoRetrieval] 闲聊", &result);

    // 场景 2: 具体问题 — 单次检索
    let result = rag.invoke("Rust 的所有权系统是什么?").await?;
    print_result("[SingleSearch] 具体问题", &result);

    // 场景 3: 复杂问题 — 多角度检索
    let result = rag.invoke("对比 Tokio 和 async-std 的调度模型").await?;
    print_result("[MultiQuery] 复杂问题", &result);

    Ok(())
}

fn print_result(label: &str, result: &langchainrust::AdaptiveRAGResult) {
    println!("\n{}", label);
    println!("  决策: {}", result.decision);
    println!("  回答: {}", result.answer);
    println!("  来源文档数: {}", result.sources.len());
}
