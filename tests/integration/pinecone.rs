//! Pinecone 集成测试 - 需要 API Key
//!
//! 手动运行:
//! ```bash
//! PINECONE_API_KEY=... PINECONE_HOST=https://... cargo test --test integration_pinecone -- --ignored
//! ```

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::embeddings::Embeddings;
use langchainrust::vector_stores::{Document, PineconeStore};
use std::collections::HashMap;

#[tokio::test]
#[ignore = "需要 PINECONE_API_KEY + PINECONE_HOST + OPENAI_API_KEY"]
async fn test_pinecone_upsert_and_query() {
    let api_key = std::env::var("PINECONE_API_KEY").expect("设置 PINECONE_API_KEY");
    let host = std::env::var("PINECONE_HOST").expect("设置 PINECONE_HOST");
    let store = PineconeStore::new(api_key, host);
    let embeddings = TestConfig::get().embeddings();

    let docs = vec![
        Document {
            content: "Rust 是系统编程语言".to_string(),
            metadata: HashMap::new(),
            id: Some("doc1".to_string()),
        },
        Document {
            content: "Python 是脚本语言".to_string(),
            metadata: HashMap::new(),
            id: Some("doc2".to_string()),
        },
    ];
    store.upsert(&docs, &embeddings).await.expect("upsert 失败");

    let qvec = embeddings
        .embed_query("什么是 Rust")
        .await
        .expect("embed 失败");
    let results = store.query(qvec, 2).await.expect("query 失败");
    println!("查询结果: {:?}", results);
    assert!(!results.is_empty());
}
