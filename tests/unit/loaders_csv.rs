#[cfg(test)]
mod test_csv_loader {
    use crate::retrieval::{CSVLoader, DocumentLoader};

    #[tokio::test]
    async fn test_csv_loader_nonexistent() {
        let loader = CSVLoader::new("./nonexistent.csv", "content");
        let result = loader.load().await;
        
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_csv_loader_content_column_not_found() {
        let tempfile = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(&tempfile, "col1,col2\nval1,val2\n").unwrap();
        
        let loader = CSVLoader::new(tempfile.path(), "content");
        let result = loader.load().await;
        
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::retrieval::LoaderError::CsvError(msg) => assert!(msg.contains("不存在")),
            _ => panic!("Expected CsvError"),
        }
    }
    
    #[tokio::test]
    async fn test_csv_loader_valid_data() {
        let tempfile = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(&tempfile, "title,content,author\nExample Title,\"This is the content\",John Doe\nAnother Title,\"More content\",Jane Smith\n").unwrap();
        
        let loader = CSVLoader::new(tempfile.path(), "content");
        let result = loader.load().await;
        
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 2);
        
        // 检查第一个文档的内容和元数据
        if !docs.is_empty() {
            let doc = &docs[0];
            assert!(doc.content.contains("This is the content"));
            assert_eq!(doc.metadata.get("title"), Some(&"Example Title".to_string()));
            assert_eq!(doc.metadata.get("author"), Some(&"John Doe".to_string()));
            assert_eq!(doc.metadata.get("content"), Some(&"This is the content".to_string()));
            assert_eq!(doc.metadata.get("format"), Some(&"csv".to_string()));
            assert_eq!(doc.metadata.get("content_column"), Some(&"content".to_string()));
        }
    }
}