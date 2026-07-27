//! Deep Research Agent 示例
//!
//! 展示 DeepResearchAgent 的多轮深度研究:
//! 拆子课题 → 搜索 → 综合 → 发现缺口 → 再搜 → 带引用报告。
//!
//! # 运行
//! ```bash
//! cargo run --example agent_deep_research
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)
//! - `OPENAI_BASE_URL`:API 基址(可选)

use langchainrust::tools::DuckDuckGoSearchTool;
use langchainrust::{DeepResearchAgent, OpenAIChat, OpenAIConfig};

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

    // 2. 创建 Deep Research Agent
    let agent = DeepResearchAgent::new(llm)
        .with_searcher(Box::new(DuckDuckGoSearchTool::new()))
        .with_max_rounds(3) // 最多 3 轮搜索
        .with_max_subtopics(5); // 最多拆 5 个子课题

    // 3. 执行深度研究
    let report = agent
        .research("Rust 异步运行时对比: tokio vs async-std vs smol")
        .await?;

    // 4. 输出报告
    println!("=== 深度研究报告 ===\n");
    println!("{}", report.markdown);

    println!("\n--- 引用 ---");
    for citation in &report.citations {
        println!(
            "[{}] {} {}",
            citation.index,
            citation.source,
            citation
                .url
                .as_ref()
                .map(|u| format!("({})", u))
                .unwrap_or_default()
        );
        println!("  {}", citation.snippet);
    }

    println!("\n--- 统计 ---");
    println!("子课题: {:?}", report.subtopics);
    println!("研究轮数: {}", report.rounds_completed);

    Ok(())
}
