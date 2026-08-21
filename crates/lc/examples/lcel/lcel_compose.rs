//! LCEL 组合体验示例 —— 提示词 + 记忆 + LLM + 解析器 + RAG 一条链
//!
//! v0.15.0 目标:全框架所有功能都能 `pipe` 成一条链。本示例把五个能力
//! 放进一个可运行程序,展示统一组合体验:
//!
//! 1. **提示词** `ChatPromptTemplate` —— Runnable 化后直接进链
//! 2. **记忆** `RunnableWithMessageHistory` —— "LLM + 记忆"整体作为一个 Runnable
//! 3. **LLM** 原生 `OpenAIChat` —— 不再套 `LLMClient`,直接 `pipe`
//! 4. **解析器** `StrOutputParser` —— 接住 `LLMResult`,自动取 `content`
//! 5. **RAG** `RagRunnable` —— 检索增强生成作为链的一段
//!
//! # 运行
//!
//! ```bash
//! cargo run --example lcel_compose
//! ```
//!
//! # 环境变量
//!
//! | 变量 | 说明 |
//! |---|---|
//! | `OPENAI_API_KEY` | API 密钥(必需) |
//! | `OPENAI_BASE_URL` | API 基址(可选,默认 OpenAI 官方) |
//! | `TEST_CHAT_MODEL` | 模型名(可选,默认 gpt-4o-mini) |
//!
//! RAG 段用 BM25 做本地关键词检索(不依赖向量库/网络),只有最后的
//! 回答生成走 LLM,方便在没有向量服务的环境里跑通整条链。

use langchainrust::{
    BM25Retriever, ChatPromptTemplate, ConversationBufferMemory, Document, Message, OpenAIChat,
    OpenAIConfig, RAGPipelineBuilder, RagRunnable, Runnable, RunnableExt, RunnableWithMessageHistory,
    StrOutputParser,
};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // === 0. 真实 LLM(原生 OpenAIChat,可直接 pipe) ===
    let api_key = std::env::var("OPENAI_API_KEY").expect("请设置 OPENAI_API_KEY 环境变量");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let model = std::env::var("TEST_CHAT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    let llm = OpenAIChat::new(OpenAIConfig {
        api_key,
        base_url,
        model,
        ..Default::default()
    });

    // === 1. 提示词 + LLM + 解析器(P0 核心链) ===
    // 链条类型: Runnable<HashMap<String, String>, String>
    let prompt = ChatPromptTemplate::from_messages([
        Message::system("你是一个简洁的 Rust 助手,只输出结论,不要多余文字。"),
        Message::human("{question}"),
    ]);
    let qa_chain = prompt.pipe(llm.clone()).pipe(StrOutputParser::new());

    let mut vars = HashMap::new();
    vars.insert(
        "question".to_string(),
        "一句话说明什么是 Rust 语言".to_string(),
    );
    let answer = qa_chain.invoke(vars, None).await?;
    println!("[1] 提示词+LLM+解析器\n     {answer}\n");

    // === 2. 记忆 + LLM + 解析器(多轮对话链) ===
    // 链条类型: Runnable<String, String>
    // 读记忆 → 拼用户输入 → LLM → 写回,全部封装在 RunnableWithMessageHistory 里。
    let memory = ConversationBufferMemory::new().with_return_messages(true);
    let chat_chain = RunnableWithMessageHistory::new(llm.clone(), memory)
        .pipe(StrOutputParser::new());

    let r1 = chat_chain.invoke("我叫小明,请记住我。".to_string(), None).await?;
    let r2 = chat_chain.invoke("我叫什么名字?".to_string(), None).await?;
    println!("[2] 记忆+LLM+解析器(多轮)\n     第一轮: {r1}\n     第二轮: {r2}\n");

    // === 3. RAG 链(BM25 本地检索 + LLM 生成) ===
    // 链条类型: Runnable<String, String>
    // BM25 检索不依赖向量库,只有回答生成走 LLM。
    let retriever = BM25Retriever::new();
    retriever.add_documents_sync(vec![
        Document::new("Rust 是一门系统编程语言,由 Mozilla 开发,注重安全和性能。")
            .with_id("rust_intro"),
        Document::new("Rust 的核心特性包括所有权系统、借用检查和零成本抽象。")
            .with_id("rust_features"),
        Document::new("LCEL(表达式语言)用 .pipe() 把提示词、模型、解析器串成一条链。")
            .with_id("lcel_intro"),
    ]);

    let pipeline = RAGPipelineBuilder::new()
        .llm(llm)
        .retriever(retriever)
        .retrieve_k(2)
        .build()?;
    let rag_chain = RagRunnable::new(Arc::new(pipeline));

    let answer = rag_chain
        .invoke("Rust 有哪些核心特性?".to_string(), None)
        .await?;
    println!("[3] RAG 链\n     {answer}\n");

    println!("=== lcel_compose 全链完成 ✅ ===");
    Ok(())
}
