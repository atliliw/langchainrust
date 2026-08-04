//! GraphRAG 示例
//!
//! 展示 GraphRAG 的知识图谱构建 + Local/Global/Hybrid 查询。
//!
//! # 运行
//! ```bash
//! cargo run --example rag_graph_rag
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)
//! - `OPENAI_BASE_URL`:API 基址(可选)

use langchainrust::retrieval::graph_rag::{GraphRAG, GraphRAGConfig, QueryMode};
use langchainrust::{Document, OpenAIChat, OpenAIConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 配置 LLM
    let api_key = std::env::var("OPENAI_API_KEY").expect("请设置 OPENAI_API_KEY 环境变量");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let llm = OpenAIChat::new(OpenAIConfig {
        api_key,
        base_url,
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    });

    // 2. 创建 GraphRAG
    let graph_rag = GraphRAG::new(llm).with_config(
        GraphRAGConfig::new()
            .with_max_entities_per_doc(10)
            .with_max_relations_per_doc(10),
    );

    // 3. 添加文档（LLM 自动抽取实体和关系）
    let docs = vec![
        Document::new("张三是清华大学的教授,研究方向是人工智能。"),
        Document::new("李四是张三的学生,正在研究大语言模型。"),
        Document::new("王五也是张三的学生,研究方向是计算机视觉。"),
    ];
    graph_rag.add_documents(&docs).await?;
    println!("文档已添加,实体和关系已抽取");

    // 4. 构建社区（自动检测紧密关联的实体群）
    graph_rag.build_communities().await?;
    println!("社区检测完成");

    // 5. 三种查询模式
    // Local: 搜索实体邻居,适合具体问题
    let local_result = graph_rag
        .query("张三的学生有哪些?", QueryMode::Local)
        .await?;
    println!("\n[Local 查询] 张三的学生有哪些?");
    println!("回答: {}", local_result.answer);

    // Global: 搜索社区摘要,适合宏观问题
    let global_result = graph_rag
        .query("这个知识库涉及哪些研究领域?", QueryMode::Global)
        .await?;
    println!("\n[Global 查询] 涉及哪些研究领域?");
    println!("回答: {}", global_result.answer);

    // Hybrid: Local + Global 结合
    let hybrid_result = graph_rag
        .query("张三课题组的研究方向是什么?", QueryMode::Hybrid)
        .await?;
    println!("\n[Hybrid 查询] 张三课题组的研究方向?");
    println!("回答: {}", hybrid_result.answer);

    Ok(())
}
