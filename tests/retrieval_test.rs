use langchainrust::retrieval::{
    Document, DocumentChunk, EmbeddingModel, FixedSizeSplitter, InMemoryVectorStore,
    MockEmbeddingModel, RecursiveCharacterSplitter, Retriever, SimilarityRetriever,
    TextSplitter, VectorStore,
};
use std::sync::Arc;

#[test]
fn test_document_creation() {
    let doc = Document::new("这是一段测试文本".to_string())
        .with_metadata("source".to_string(), "test".to_string())
        .with_metadata("author".to_string(), "tester".to_string());

    assert_eq!(doc.content, "这是一段测试文本");
    assert_eq!(doc.metadata.get("source"), Some(&"test".to_string()));
    assert_eq!(doc.metadata.get("author"), Some(&"tester".to_string()));
}

#[test]
fn test_document_chunk_creation() {
    let chunk = DocumentChunk::new("分块内容".to_string(), 0)
        .with_metadata("page".to_string(), "1".to_string())
        .with_document_id("doc_001".to_string());

    assert_eq!(chunk.content, "分块内容");
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.document_id, Some("doc_001".to_string()));
}

#[test]
fn test_fixed_size_splitter() {
    let splitter = FixedSizeSplitter::new(10, 2);
    let doc = Document::new("这是一段比较长的测试文本，用于测试分块功能".to_string());

    let chunks = splitter.split_document(&doc).unwrap();

    println!("固定大小分割器结果:");
    for (i, chunk) in chunks.iter().enumerate() {
        println!("  Chunk {}: {} (长度: {})", i, chunk.content, chunk.content.len());
    }

    assert!(!chunks.is_empty());
    // 检查每个chunk的大小
    for chunk in &chunks {
        assert!(chunk.content.chars().count() <= 12); // 10 + 2 overlap tolerance
    }
}

#[test]
fn test_recursive_character_splitter() {
    let splitter = RecursiveCharacterSplitter::new(50, 10);
    let doc = Document::new(
        "这是第一段。\n\n这是第二段，比较长一些。\n\n这是第三段，用于测试递归分割功能。".to_string()
    );

    let chunks = splitter.split_document(&doc).unwrap();

    println!("递归字符分割器结果:");
    for (i, chunk) in chunks.iter().enumerate() {
        println!("  Chunk {}: {}", i, chunk.content);
    }

    assert!(!chunks.is_empty());
}

#[test]
fn test_text_splitter_split_text() {
    let splitter = FixedSizeSplitter::new(20, 5);

    let text = "这是一段用于测试纯文本分割的文本内容";
    let chunks = splitter.split_text(text).unwrap();

    println!("纯文本分割结果:");
    for (i, chunk) in chunks.iter().enumerate() {
        println!("  Chunk {}: {}", i, chunk);
    }

    assert!(!chunks.is_empty());
}

#[tokio::test]
async fn test_mock_embedding_model() {
    let model = MockEmbeddingModel::new(128);

    // 测试单个文本嵌入
    let embedding = model.embed("测试文本").await.unwrap();

    println!("嵌入维度: {}", embedding.len());
    println!("嵌入向量前5个值: {:?}", &embedding[..5]);

    assert_eq!(embedding.len(), 128);

    // 相同文本应该产生相同嵌入
    let embedding2 = model.embed("测试文本").await.unwrap();
    for (a, b) in embedding.iter().zip(embedding2.iter()) {
        assert!((a - b).abs() < 1e-6);
    }

    // 不同文本应该产生不同嵌入
    let embedding3 = model.embed("不同的文本").await.unwrap();
    let diff_count = embedding.iter().zip(embedding3.iter()).filter(|(a, b)| (**a - **b).abs() > 1e-6).count();
    assert!(diff_count > 0);
}

#[tokio::test]
async fn test_embedding_batch() {
    let model = MockEmbeddingModel::new(64);

    let texts = vec!["文本1", "文本2", "文本3"];
    let embeddings = model.embed_batch(texts).await.unwrap();

    println!("批量嵌入数量: {}", embeddings.len());
    println!("每个嵌入的维度: {}", embeddings[0].len());

    assert_eq!(embeddings.len(), 3);
    assert_eq!(embeddings[0].len(), 64);
}

#[tokio::test]
async fn test_in_memory_vector_store() {
    let mut store = InMemoryVectorStore::new();

    // 创建测试文档块
    let chunks = vec![
        DocumentChunk::new("Rust是一种系统编程语言".to_string(), 0),
        DocumentChunk::new("Python是一种脚本语言".to_string(), 1),
        DocumentChunk::new("JavaScript用于网页开发".to_string(), 2),
    ];

    // 创建对应的嵌入向量（模拟）
    let embeddings = vec![
        vec![0.1; 64],
        vec![0.2; 64],
        vec![0.3; 64],
    ];

    // 添加文档
    let docs_with_embeddings: Vec<(DocumentChunk, Vec<f32>)> = chunks.into_iter().zip(embeddings).collect();
    store.add_documents(docs_with_embeddings).await.unwrap();

    // 搜索
    let query = vec![0.15; 64];
    let results = store.similarity_search(query, 2).await.unwrap();

    println!("搜索结果数量: {}", results.len());
    for (chunk, score) in &results {
        println!("  - {} (相似度: {:.4})", chunk.content, score);
    }

    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_similarity_retriever() {
    // 创建嵌入模型
    let embedding_model = Arc::new(MockEmbeddingModel::new(64));

    // 创建向量存储
    let vector_store = Box::new(InMemoryVectorStore::new());

    // 创建检索器
    let retriever = SimilarityRetriever::new(vector_store, embedding_model.clone());

    // 创建并添加文档
    let chunks = vec![
        DocumentChunk::new("Rust是一种系统编程语言，注重内存安全".to_string(), 0)
            .with_metadata("category".to_string(), "programming".to_string()),
        DocumentChunk::new("Python是一种高级编程语言，易于学习".to_string(), 1)
            .with_metadata("category".to_string(), "programming".to_string()),
        DocumentChunk::new("苹果是一种水果，富含维生素".to_string(), 2)
            .with_metadata("category".to_string(), "fruit".to_string()),
    ];

    retriever.add_documents(chunks).await.unwrap();

    // 测试检索
    let results = retriever.retrieve("编程语言", 2).await.unwrap();

    println!("\n检索结果:");
    for result in &results {
        println!("  - {} (分数: {:.4})", result.chunk.content, result.score);
    }

    assert_eq!(results.len(), 2);
    // Mock模型的嵌入是基于哈希的，所以只检查返回了结果
    // 真实场景下会使用真实的嵌入模型
    for result in &results {
        println!("  - {} (分数: {:.4})", result.chunk.content, result.score);
    }
}

#[tokio::test]
async fn test_similarity_retriever_with_filter() {
    let embedding_model = Arc::new(MockEmbeddingModel::new(64));
    let vector_store = Box::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(vector_store, embedding_model);

    let chunks = vec![
        DocumentChunk::new("Rust编程".to_string(), 0)
            .with_metadata("category".to_string(), "programming".to_string()),
        DocumentChunk::new("苹果手机".to_string(), 1)
            .with_metadata("category".to_string(), "phone".to_string()),
        DocumentChunk::new("香蕉水果".to_string(), 2)
            .with_metadata("category".to_string(), "fruit".to_string()),
    ];

    retriever.add_documents(chunks).await.unwrap();

    // 使用过滤器检索
    let mut filter = std::collections::HashMap::new();
    filter.insert("category".to_string(), "programming".to_string());

    let results = retriever.retrieve_with_filter("测试", 5, filter).await.unwrap();

    println!("\n过滤检索结果:");
    for result in &results {
        println!("  - {} (分类: {:?})", result.chunk.content, result.chunk.metadata.get("category"));
    }

    // 所有结果应该都是programming分类
    for result in &results {
        assert_eq!(result.chunk.metadata.get("category"), Some(&"programming".to_string()));
    }
}

#[tokio::test]
async fn test_full_retrieval_pipeline() {
    println!("\n=== 完整检索流程测试 ===\n");

    // 1. 创建文档
    let doc = Document::new(
        "Rust是一种系统编程语言，由Mozilla开发。它注重内存安全、并发性能和执行效率。\
         Rust使用所有权系统来管理内存，不需要垃圾回收器。\
         Python是一种解释型、高级编程语言。它的设计哲学强调代码可读性。\
         JavaScript是一种脚本语言，主要用于网页开发。\
         苹果是一种常见的水果，富含维生素C和纤维。".to_string()
    );

    // 2. 分割文档
    let splitter = RecursiveCharacterSplitter::new(50, 10);
    let chunks = splitter.split_document(&doc).unwrap();
    println!("文档被分割为 {} 个块", chunks.len());

    // 3. 创建检索器
    let embedding_model = Arc::new(MockEmbeddingModel::new(128));
    let vector_store = Box::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(vector_store, embedding_model);

    // 4. 添加文档块
    retriever.add_documents(chunks).await.unwrap();
    println!("文档块已添加到向量存储");

    // 5. 检索相关内容
    let queries = vec!["编程语言", "水果", "内存安全"];
    
    for query in queries {
        println!("\n查询: \"{}\"", query);
        let results = retriever.retrieve(query, 2).await.unwrap();
        for (i, result) in results.iter().enumerate() {
            println!("  {}. {} (分数: {:.4})", i + 1, result.chunk.content, result.score);
        }
    }
}

#[test]
fn test_cosine_similarity() {
    // 测试余弦相似度计算
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let similarity = InMemoryVectorStore::cosine_similarity(&a, &b);
    println!("相同向量的相似度: {}", similarity);
    assert!((similarity - 1.0).abs() < 1e-6);

    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let similarity = InMemoryVectorStore::cosine_similarity(&a, &b);
    println!("正交向量的相似度: {}", similarity);
    assert!((similarity - 0.0).abs() < 1e-6);

    let a = vec![1.0, 1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let similarity = InMemoryVectorStore::cosine_similarity(&a, &b);
    println!("45度角向量的相似度: {}", similarity);
    assert!(similarity > 0.7 && similarity < 0.8); // cos(45°) ≈ 0.707
}
