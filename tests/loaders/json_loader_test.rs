// tests/loaders/json_loader_test.rs
//! JSONLoader 测试用例

use langchainrust::{JSONLoader, DocumentLoader, LoaderError};
use std::path::PathBuf;

fn get_test_data_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/loaders/data")
        .join(filename)
}

#[tokio::test]
async fn test_json_loader_sample_array() {
    let path = get_test_data_path("sample.json");
    let loader = JSONLoader::new(&path);
    
    let result = loader.load().await;
    
    assert!(result.is_ok(), "加载 sample.json 应该成功");
    let docs = result.unwrap();
    
    assert_eq!(docs.len(), 4, "数组应返回 4 个文档");
    
    assert!(docs[0].content.contains("Rust"));
    assert!(docs[1].content.contains("LangChain"));
}

#[tokio::test]
async fn test_json_loader_with_content_key() {
    let path = get_test_data_path("sample.json");
    let loader = JSONLoader::new_with_content_key(&path, "content");
    
    let result = loader.load().await;
    
    assert!(result.is_ok());
    let docs = result.unwrap();
    
    assert_eq!(docs.len(), 4);
    
    assert!(docs[0].content.contains("系统编程语言"));
    assert!(docs[1].content.contains("语言模型"));
}

#[tokio::test]
async fn test_json_loader_preserve_metadata() {
    let path = get_test_data_path("sample.json");
    let loader = JSONLoader::new_with_content_key(&path, "content");
    
    let docs = loader.load().await.unwrap();
    
    assert!(docs[0].metadata.contains_key("title"));
    assert!(docs[0].metadata.contains_key("author"));
    assert!(docs[0].metadata.contains_key("category"));
    
    assert_eq!(
        docs[0].metadata.get("title"),
        Some(&"Rust 入门指南".to_string())
    );
    assert_eq!(
        docs[0].metadata.get("author"),
        Some(&"Mozilla".to_string())
    );
}

#[tokio::test]
async fn test_json_loader_preserve_raw() {
    let path = get_test_data_path("sample.json");
    let loader = JSONLoader::new(&path).with_preserve_raw(true);
    
    let docs = loader.load().await.unwrap();
    
    assert!(docs[0].metadata.contains_key("raw_json"));
}

#[tokio::test]
async fn test_json_loader_nonexistent_file() {
    let loader = JSONLoader::new("./nonexistent.json");
    let result = loader.load().await;
    
    assert!(result.is_err());
}

#[tokio::test]
async fn test_json_loader_invalid_json() {
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    let mut temp_file = NamedTempFile::new().unwrap();
    write!(temp_file, "{{ invalid }}").unwrap();
    
    let loader = JSONLoader::new(temp_file.path());
    let result = loader.load().await;
    
    assert!(result.is_err());
    match result.unwrap_err() {
        LoaderError::JsonError(_) => {}
        _ => panic!("应返回 JsonError"),
    }
}