//! 文档加载器示例
//!
//! 展示 TextLoader 加载文本文件并解析 metadata(无需 API Key)。
//!
//! # 运行
//! ```bash
//! cargo run --example rag_document_loaders
//! # 或指定文件
//! cargo run --example rag_document_loaders -- README.md
//! ```

use langchainrust::{DocumentLoader, TextLoader};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "README.md".to_string());
    let loader = TextLoader::new(PathBuf::from(path));
    let docs = loader.load().await?;

    println!("加载 {} 个文档", docs.len());
    for (i, doc) in docs.iter().enumerate() {
        let preview: String = doc.content.chars().take(200).collect();
        println!("--- 文档 {} ---", i + 1);
        println!("内容预览: {}", preview);
        println!("metadata: {:?}", doc.metadata);
    }
    Ok(())
}
