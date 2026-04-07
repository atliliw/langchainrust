// 简单的测试验证我们的加载器能正常工作
use langchainrust::retrieval::{PDFLoader, CSVLoader, DocumentLoader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 这里只是一个概念验证，实际测试需要真实文件
    
    println!("PDFLoader and CSVLoader are available in the public API.");
    
    // 示例用法
    // let pdf_loader = PDFLoader::new("path/to/doc.pdf");
    // let csv_loader = CSVLoader::new("path/to/data.csv", "content_column");
    //
    // let pdf_documents = pdf_loader.load().await?;
    // let csv_documents = csv_loader.load().await?;
    
    println!("Successfully imported loaders from langchainrust!");
    Ok(())
}