//! 单元测试 - Retrieval (使用 MockEmbeddings)

use langchainrust::{
    Document, InMemoryVectorStore, MockEmbeddings, VectorStore,
    RecursiveCharacterSplitter, TextSplitter, cosine_similarity, Embeddings,
};

#[test]
fn test_document_creation() {
    let doc = Document::new("Rust is a systems programming language.");
    assert_eq!(doc.page_content(), "Rust is a systems programming language.");
}

#[test]
fn test_document_with_metadata() {
    let doc = Document::new("Test content").with_metadata("source", "test");
    assert_eq!(doc.page_content(), "Test content");
}

#[test]
fn test_text_splitter() {
    let text = "This is a long text that needs to be split into smaller chunks for processing.";
    let splitter = RecursiveCharacterSplitter::new(20, 5);
    
    let chunks = splitter.split_text(text);
    
    assert!(!chunks.is_empty());
}

#[tokio::test]
async fn test_in_memory_vector_store() {
    let store = InMemoryVectorStore::new();
    let embeddings = MockEmbeddings::new(128);
    
    let v1 = embeddings.embed_query("Rust is fast").await.unwrap();
    let v2 = embeddings.embed_query("Python is easy").await.unwrap();
    let vectors = vec![v1, v2];
    
    let docs = vec![
        Document::new("Rust is fast"),
        Document::new("Python is easy"),
    ];
    
    store.add_documents(docs, vectors).await.unwrap();
    
    let query_vec = embeddings.embed_query("fast language").await.unwrap();
    let results = store.similarity_search(&query_vec, 2).await.unwrap();
    
    assert_eq!(results.len(), 2);
}

#[test]
fn test_cosine_similarity_same() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 0.0001);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 0.0).abs() < 0.0001);
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![-1.0, 0.0, 0.0];
    
    let sim = cosine_similarity(&a, &b);
    assert!((sim - (-1.0)).abs() < 0.0001);
}