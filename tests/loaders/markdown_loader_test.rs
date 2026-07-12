// tests/loaders/markdown_loader_test.rs
//! MarkdownLoader 测试用例

use langchainrust::{MarkdownLoader, DocumentLoader, LoaderError};
use std::path::PathBuf;

fn get_test_data_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/loaders/data")
        .join(filename)
}

#[tokio::test]
async fn test_markdown_loader_sample_file() {
    let path = get_test_data_path("sample.md");
    let loader = MarkdownLoader::new(&path);
    
    let result = loader.load().await;
    
    assert!(result.is_ok(), "加载 sample.md 应该成功");
    let docs = result.unwrap();
    
    assert_eq!(docs.len(), 1, "不分割应返回一个文档");
    
    let doc = &docs[0];
    assert!(doc.content.contains("LangChainRust"));
    assert!(doc.content.contains("LLM"));
    assert_eq!(doc.metadata.get("format"), Some(&"markdown".to_string()));
}

#[tokio::test]
async fn test_markdown_loader_split_by_heading_level_1() {
    let path = get_test_data_path("sample.md");
    let loader = MarkdownLoader::new_with_heading_split(&path, 1);
    
    let result = loader.load().await;
    
    assert!(result.is_ok());
    let docs = result.unwrap();
    
    // sample.md 只有一个一级标题，所以只有 1 个文档
    assert!(!docs.is_empty(), "按一级标题分割应返回至少 1 个文档");
    
    let headings: Vec<&str> = docs.iter()
        .filter_map(|d| d.metadata.get("heading").map(|s| s.as_str()))
        .collect();
    
    // 验证一级标题被正确识别
    assert!(headings.iter().any(|h| h.contains("LangChainRust")));
}

#[tokio::test]
async fn test_markdown_loader_split_by_heading_level_2() {
    let path = get_test_data_path("sample.md");
    let loader = MarkdownLoader::new_with_heading_split(&path, 2);
    
    let result = loader.load().await;
    
    assert!(result.is_ok());
    let docs = result.unwrap();
    
    // sample.md 有多个二级标题：核心模块、RAG 功能、安装使用
    assert!(docs.len() >= 3, "按二级标题分割应有至少 3 个文档");
    
    let headings: Vec<&str> = docs.iter()
        .filter_map(|d| d.metadata.get("heading").map(|s| s.as_str()))
        .collect();
    
    // 验证二级标题被正确识别
    assert!(headings.iter().any(|h| h.contains("核心模块")));
    assert!(headings.iter().any(|h| h.contains("RAG")));
}

#[tokio::test]
async fn test_markdown_loader_heading_metadata() {
    let path = get_test_data_path("sample.md");
    let loader = MarkdownLoader::new_with_heading_split(&path, 1);
    
    let docs = loader.load().await.unwrap();
    
    for doc in &docs {
        assert!(doc.metadata.contains_key("heading"));
        assert!(doc.metadata.contains_key("heading_level"));
        assert_eq!(
            doc.metadata.get("heading_level"),
            Some(&"1".to_string())
        );
    }
}

#[tokio::test]
async fn test_markdown_loader_content_preserved() {
    let path = get_test_data_path("sample.md");
    let loader = MarkdownLoader::new_with_heading_split(&path, 2);
    
    let docs = loader.load().await.unwrap();
    
    // 找包含 LLM 相关内容的文档
    let llm_doc = docs.iter()
        .find(|d| d.content.contains("OpenAI") || d.content.contains("LLM"));
    
    assert!(llm_doc.is_some(), "应找到包含 LLM 内容的文档");
    let doc = llm_doc.unwrap();
    
    assert!(doc.content.contains("OpenAI") || doc.content.contains("LLM"));
}

#[tokio::test]
async fn test_markdown_loader_nonexistent_file() {
    let loader = MarkdownLoader::new("./nonexistent.md");
    let result = loader.load().await;
    
    assert!(result.is_err());
    match result.unwrap_err() {
        LoaderError::Other(msg) => assert!(msg.contains("不存在")),
        _ => panic!("应返回 Other 错误"),
    }
}