use crate::retrieval::Retriever;
use std::sync::Arc;

/// 从向量数据库检索相关文档
pub async fn retrieve_context(
    retriever: &Option<Arc<dyn Retriever>>,
    query: &str,
    top_k: usize,
) -> Option<String> {
    let retriever = retriever.as_ref()?;

    println!("[检索] 正在从向量数据库检索相关文档...");

    match retriever.retrieve(query, top_k).await {
        Ok(results) => {
            if results.is_empty() {
                println!("[检索] 未找到相关文档");
                return None;
            }

            println!("[检索] 找到 {} 个相关文档:", results.len());
            for (i, result) in results.iter().enumerate() {
                println!("  [{}] 相似度: {:.4}", i + 1, result.score);
            }

            let context = results
                .iter()
                .enumerate()
                .map(|(i, r)| format!("[文档{}]\n{}", i + 1, r.chunk.content))
                .collect::<Vec<_>>()
                .join("\n\n");

            Some(context)
        }
        Err(e) => {
            println!("[检索] 检索失败: {}", e);
            None
        }
    }
}
