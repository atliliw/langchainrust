//! Embeddings 测试
//!
//! 测试 OpenAIEmbeddings 的模型维度信息(纯本地,不触网)。

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::Embeddings;

/// 测试嵌入模型维度
///
/// 测试内容：
/// - 获取模型输出的向量维度
/// - 验证维度值有效（如 ada-002 为 1536）
#[tokio::test]
async fn test_embedding_dimension() {
    let embeddings = TestConfig::get().embeddings();

    let dim = embeddings.dimension();
    println!("Model dimension: {}", dim);
    assert!(dim > 0);
}
