// tests/loaders/text_loader_test.rs
//! TextLoader 测试用例

use langchainrust::{TextLoader, DocumentLoader, LoaderError};
use std::path::PathBuf;

fn get_test_data_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/loaders/data")
        .join(filename)
}

#[tokio::test]
async fn test_text_loader_sample_file() {
    let path = get_test_data_path("sample.txt");
    let loader = TextLoader::new(&path);
    
    let result = loader.load().await;
    
    assert!(result.is_ok(), "加载 sample.txt 应该成功");
    let docs = result.unwrap();
    
    assert_eq!(docs.len(), 1, "默认加载应返回一个文档");
    
    let doc = &docs[0];
    assert!(doc.content.contains("LangChainRust"), "内容应包含 LangChainRust");
    assert!(doc.content.contains("BM25"), "内容应包含 BM25");
    assert_eq!(doc.metadata.get("format"), Some(&"text".to_string()));
    assert!(doc.metadata.contains_key("source"));
}

#[tokio::test]
async fn test_text_loader_split_by_line() {
    let path = get_test_data_path("sample.txt");
    let loader = TextLoader::new_with_line_split(&path);
    
    let result = loader.load().await;
    
    assert!(result.is_ok());
    let docs = result.unwrap();
    
    assert!(docs.len() > 1, "按行分割应返回多个文档");
    
    for (idx, doc) in docs.iter().enumerate() {
        assert!(doc.metadata.contains_key("line_number"));
        assert_eq!(
            doc.metadata.get("line_number"),
            Some(&(idx + 1).to_string())
        );
    }
}

#[tokio::test]
async fn test_text_loader_nonexistent_file() {
    let loader = TextLoader::new("./nonexistent_file.txt");
    let result = loader.load().await;
    
    assert!(result.is_err());
    
    match result.unwrap_err() {
        LoaderError::Other(msg) => assert!(msg.contains("不存在")),
        _ => panic!("应返回 Other 错误"),
    }
}

#[tokio::test]
async fn test_text_loader_metadata() {
    let path = get_test_data_path("sample.txt");
    let loader = TextLoader::new(&path);
    
    let docs = loader.load().await.unwrap();
    let doc = &docs[0];
    
    assert!(doc.metadata.contains_key("source"));
    assert!(doc.metadata.contains_key("format"));
    assert_eq!(doc.metadata.get("format"), Some(&"text".to_string()));
}