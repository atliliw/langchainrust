//! Semantic Splitter 示例
//!
//! 展示如何使用 SemanticSplitter 进行语义分块。
//!
//! # 运行
//! ```bash
//! cargo run --example semantic_splitter
//! ```

use langchainrust::{SemanticSplitter, MockEmbeddings, RecursiveCharacterSplitter};
use langchainrust::retrieval::TextSplitter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Semantic Splitter 示例 ===\n");

    let text = r#"Rust 是一种系统编程语言。它注重内存安全和并发性能。
Rust 的所有权系统是其最独特的特性。编译器在编译时检查所有权规则,避免运行时错误。
Python 是一种解释型语言。它以简洁易读著称,广泛用于数据科学和 AI。
LangChain 是一个框架。它帮助开发者构建基于大语言模型的应用程序。"#;

    // 创建语义分块器(需要嵌入模型,相似度阈值 0.5,单块最大 200 字符)
    let embeddings = MockEmbeddings::new(4);
    let splitter = SemanticSplitter::new(embeddings, 0.5, 200);

    let chunks = splitter.split_text(text).await;
    println!("原文分为 {} 个语义块:\n", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("--- 块 {} ---", i + 1);
        println!("{}\n", chunk.trim());
    }

    // 对比:传统递归分块器
    let recursive = RecursiveCharacterSplitter::new(100, 20);
    let rec_chunks = recursive.split_text(text);
    println!("\n对比:递归分块器分为 {} 个块(按固定长度切分,可能切断语义)", rec_chunks.len());

    Ok(())
}
