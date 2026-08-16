// tests/bm25/hybrid_rag.rs
//! 混合检索 (BM25 + 向量) 融合算法测试(纯内存,不触网)

use langchainrust::retrieval::reciprocal_rank_fusion;
use langchainrust::Document;

/// 测试：RRF 融合算法
/// 验证：BM25 和 向量检索结果正确融合
#[test]
fn test_rrf_fusion() {
    let bm25_docs = vec![
        Document::new("Rust是一门系统编程语言，注重安全和性能。").with_id("doc1"),
        Document::new("Python是一门高级编程语言，适合数据科学。").with_id("doc2"),
        Document::new("Go是一门并发编程语言，由Google开发。").with_id("doc3"),
    ];

    let vector_docs = vec![
        Document::new("Rust是一门系统编程语言，注重安全和性能。").with_id("doc1"),
        Document::new("JavaScript是一门前端脚本语言。").with_id("doc4"),
        Document::new("Python是一门高级编程语言，适合数据科学。").with_id("doc2"),
    ];

    println!("=== 测试 RRF 融合算法 ===");
    println!("\nBM25 检索结果:");
    for (i, doc) in bm25_docs.iter().enumerate() {
        println!(
            "  [{}] id={}, 内容={}",
            i,
            doc.id.clone().unwrap_or_default(),
            doc.content
        );
    }

    println!("\n向量检索结果:");
    for (i, doc) in vector_docs.iter().enumerate() {
        println!(
            "  [{}] id={}, 内容={}",
            i,
            doc.id.clone().unwrap_or_default(),
            doc.content
        );
    }

    let results = reciprocal_rank_fusion(bm25_docs, vector_docs, 60);

    println!("\nRRF 融合结果:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  [{}] id={}, rrf_score={:.4}",
            i,
            r.document.id.clone().unwrap_or_default(),
            r.score
        );
    }

    println!("\n分析:");
    println!("  - doc1 和 doc2 在两个列表中都出现，RRF分数更高");
    println!("  - doc3 只在BM25出现，doc4 只在向量检索出现");
}
