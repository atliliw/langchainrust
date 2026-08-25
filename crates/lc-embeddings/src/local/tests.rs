use super::*;
use crate::cosine_similarity;

// ---- BagOfWordsEmbeddings tests ----

#[tokio::test]
async fn test_bow_dimension() {
    let e = BagOfWordsEmbeddings::new(128);
    let v = e.embed_query("hello world").await.unwrap();
    assert_eq!(v.len(), 128);
    assert_eq!(e.dimension(), 128);
}

#[tokio::test]
async fn test_bow_same_text_same_vector() {
    let e = BagOfWordsEmbeddings::new(64);
    let a = e.embed_query("rust programming").await.unwrap();
    let b = e.embed_query("rust programming").await.unwrap();
    assert_eq!(a, b);
}

#[tokio::test]
async fn test_bow_different_text_different_vector() {
    let e = BagOfWordsEmbeddings::new(64);
    let a = e.embed_query("rust programming").await.unwrap();
    let b = e.embed_query("cooking recipe pasta").await.unwrap();
    assert_ne!(a, b);
}

#[tokio::test]
async fn test_bow_shared_words_more_similar() {
    let e = BagOfWordsEmbeddings::new(256);
    let base = e.embed_query("rust programming language").await.unwrap();
    let similar = e.embed_query("rust programming tutorial").await.unwrap();
    let different = e.embed_query("cooking pasta recipe").await.unwrap();

    let sim_similar = cosine_similarity(&base, &similar).unwrap_or(0.0);
    let sim_different = cosine_similarity(&base, &different).unwrap_or(0.0);
    assert!(
        sim_similar > sim_different,
        "Shared words should be more similar: {} vs {}",
        sim_similar,
        sim_different
    );
}

#[tokio::test]
async fn test_bow_normalized() {
    let e = BagOfWordsEmbeddings::new(64);
    let v = e.embed_query("some text here").await.unwrap();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
}

#[tokio::test]
async fn test_bow_empty_text_returns_error() {
    let e = BagOfWordsEmbeddings::new(64);
    let result = e.embed_query("").await;
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), EmbeddingError::EmptyInput));
}

#[tokio::test]
async fn test_bow_chinese_tokenize() {
    let e = BagOfWordsEmbeddings::new(128);
    let a = e.embed_query("机器学习").await.unwrap();
    let b = e.embed_query("机器学习").await.unwrap();
    assert_eq!(a, b);
    let c = e.embed_query("深度学习").await.unwrap();
    let sim = cosine_similarity(&a, &c).unwrap_or(0.0);
    assert!(
        sim > 0.0,
        "Shared '学习' should have positive similarity: {}",
        sim
    );
}

#[test]
fn test_bow_tokenize_english() {
    let t = BagOfWordsEmbeddings::tokenize("Hello, World! 123");
    assert!(t.contains(&"hello".to_string()));
    assert!(t.contains(&"world".to_string()));
    assert!(t.contains(&"123".to_string()));
}

#[test]
fn test_bow_tokenize_chinese() {
    let t = BagOfWordsEmbeddings::tokenize("机器学习");
    assert!(t.contains(&"机".to_string()));
    assert!(t.contains(&"学".to_string()));
    assert_eq!(t.len(), 4);
}

#[test]
fn test_bow_model_name() {
    let e = BagOfWordsEmbeddings::default_dim();
    assert_eq!(e.model_name(), "local-bow");
}

// ---- LocalEmbeddings backward compatibility test (without feature, is BagOfWordsEmbeddings alias) ----

/// P2-1: 该测试正是验证"无 feature 时 LocalEmbeddings = BagOfWordsEmbeddings",
/// 是有意使用已弃用别名,`#[allow(deprecated)]` 豁免降级警告。
#[allow(deprecated)]
#[tokio::test]
async fn test_local_embeddings_backward_compat() {
    // Without feature, LocalEmbeddings = BagOfWordsEmbeddings
    let e = LocalEmbeddings::new(64);
    let v = e.embed_query("test backward compat").await.unwrap();
    assert_eq!(v.len(), 64);
    assert_eq!(e.model_name(), "local-bow");
}
