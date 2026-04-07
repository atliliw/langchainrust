// examples/rag_with_loaders.rs
//! RAG 系统集成示例
//! 
//! 演示 PDF/CSV 加载器与不同向量存储选项的完整 RAG 流程

use langchainrust::{
    Document, VectorStoreBuilder, 
    PDFLoader, CSVLoader, DocumentLoader, 
    SimilarityRetriever, RetrieverTrait, RecursiveCharacterSplitter, TextSplitter,
    MockEmbeddings
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LangChainRust - RAG 与文档加载器集成示例 ===\n");

    // 1. 选择向量存储 (使用内存版作为演示)
    //    在生产中可以改为: VectorStoreBuilder::qdrant(url, collection)
    println!("✓ 1. 初始化向量存储...");
    let store = VectorStoreBuilder::in_memory().build().await?;
    
    // 2. 初始化嵌入模型 (实际应用中可能是 OpenAIEmbeddings)
    let embeddings = Arc::new(MockEmbeddings::new(1536)); // 1536-dim for text-embedding-3-large equivalent
    println!("✓ 2. 初始化 Mock 嵌入模型...");

    // 3. 创建检索器
    let retriever = SimilarityRetriever::new(store.clone(), embeddings);
    println!("✓ 3. 创建相拟检索器...\n");

    // 4. 从 PDF 加载文档
    println!("✓ 4. 加载 PDF 文档...");
    if std::path::Path::new("sample.pdf").exists() {
        let pdf_loader = PDFLoader::new("sample.pdf");
        let pdf_docs = pdf_loader.load().await?;
        println!("   PDF 加载了 {} 个文档", pdf_docs.len());

        // 分割文档
        let splitter = RecursiveCharacterSplitter::new(500, 100);
        let mut chunks = Vec::new();
        for doc in pdf_docs {
            let doc_chunks = splitter.split_document(&doc);
            chunks.extend(doc_chunks);
        }
        println!("   分割为 {} 个文本块", chunks.len());

        // 添加到检索器
        if !chunks.is_empty() {
            retriever.add_documents(chunks).await?;
            println!("   已索引到向量存储");
        }
    } else {
        println!("   (sample.pdf 不存在，跳过 PDF 测试)");
    }

    // 5. 从 CSV 加载文档
    println!("✓ 5. 加载 CSV 数据...");
    if std::path::Path::new("sample_data.csv").exists() {
        let csv_loader = CSVLoader::new("sample_data.csv", "description"); // 假设 'description' 是内容列
        let csv_docs = csv_loader.load().await?;
        println!("   CSV 加载了 {} 个文档", csv_docs.len());

        // 添加到检索器
        if !csv_docs.is_empty() {
            retriever.add_documents(csv_docs).await?;
            println!("   已索引到向量存储");
        }
    } else {
        println!("   (sample_data.csv 不存在，跳过 CSV 测试)");
        println!("   创建了一个演示文档集合");
        
        // 创建一些示例文档用于演示
        let demo_docs = vec![
            Document::new("Rust is a systems programming language focusing on safety, speed, and concurrency.")
                .with_metadata("type", "fact")
                .with_metadata("topic", "programming"),
            Document::new("Vector databases specialize in fast similarity search for embedding vectors.")
                .with_metadata("type", "fact")
                .with_metadata("topic", "databases"),
            Document::new("Retrieval Augmented Generation (RAG) combines information retrieval with generative models.")
                .with_metadata("type", "fact") 
                .with_metadata("topic", "ai"),
            Document::new("PDF documents often contain structured information suitable for extraction.")
                .with_metadata("type", "fact")
                .with_metadata("topic", "documents"),
        ];

        retriever.add_documents(demo_docs).await?;
        println!("   添加了 4 个示例文档");
    }

    // 6. 检索演示
    println!("\n✓ 6. 执行检索查询...");
    let queries = vec![
        "Tell me about Rust programming language",
        "How do vector databases work?",
        "What is RAG in AI?"
    ];

    for query in queries {
        println!("\n   查询: \"{}\"", query);
        
        let results = retriever.retrieve(query, 2).await?;
        println!("   找到 {} 个相关文档:", results.len());
        
        for (i, doc) in results.iter().enumerate() {
            let content_preview = truncate(&doc.page_content(), 100);
            // 修复借用问题
            let topic_opt = doc.metadata.get("topic");
            let topic_str = match topic_opt {
                Some(topic) => topic.as_str(),
                None => "unknown"
            };
            println!("     [{}] '{}...' (topic: {})", i + 1, content_preview, topic_str);
        }
    }
    
    // 7. 总结
    let total_docs = store.count().await;
    println!("\n✓ 完成! 向量存储中共有 {} 个文档", total_docs);
    println!();
    println!("💡 框架优势:");
    println!("   • 统一的 DocumentLoader trait - 添加新文件格式很简单");
    println!("   • 统一的 VectorStore trait - 切换后端存储很容易");
    println!("   • 完整的 RAG 流程 - 从文件加载到相似搜索");
    println!("   • 可选的后端配置 - 无需外部服务也能用");
    
    Ok(())
}

fn truncate(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        None => s.to_string(),
        Some((idx, _)) => format!("{}...", &s[..idx]),
    }
}