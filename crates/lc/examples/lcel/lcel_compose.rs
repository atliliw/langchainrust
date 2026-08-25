//! LCEL composition example — prompt + memory + LLM + parser + RAG in one chain
//!
//! v0.15.0 goal: every framework feature can be `pipe`d into a single chain. This example
//! puts five capabilities into one runnable program to show the unified composition experience:
//!
//! 1. **Prompt** `ChatPromptTemplate` — a Runnable, goes straight into the chain
//! 2. **Memory** `RunnableWithMessageHistory` — "LLM + memory" as one Runnable
//! 3. **LLM** native `OpenAIChat` — no `LLMClient` wrapper, `pipe` it directly
//! 4. **Parser** `StrOutputParser` — catches the `LLMResult`, automatically takes `content`
//! 5. **RAG** `RagRunnable` — retrieval-augmented generation as one segment of the chain
//!
//! # Run
//!
//! ```bash
//! cargo run --example lcel_compose
//! ```
//!
//! # Environment variables
//!
//! | Variable | Description |
//! |---|---|
//! | `OPENAI_API_KEY` | API key (required) |
//! | `OPENAI_BASE_URL` | API base URL (optional, defaults to the official OpenAI endpoint) |
//! | `TEST_CHAT_MODEL` | Model name (optional, defaults to gpt-4o-mini) |
//!
//! The RAG segment uses BM25 local keyword search (no vector store / network), so only the
//! final answer generation calls the LLM — the whole chain runs even without a vector service.

use langchainrust::{
    BM25Retriever, ChatPromptTemplate, ConversationBufferMemory, Document, Message, OpenAIChat,
    OpenAIConfig, RAGPipelineBuilder, RagRunnable, Runnable, RunnableExt,
    RunnableWithMessageHistory, StrOutputParser,
};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // === 0. A real LLM (native OpenAIChat, can be piped directly) ===
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("please set the OPENAI_API_KEY environment variable");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let model = std::env::var("TEST_CHAT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    let llm = OpenAIChat::new(OpenAIConfig {
        api_key,
        base_url,
        model,
        ..Default::default()
    });

    // === 1. Prompt + LLM + parser (P0 core chain) ===
    // Chain type: Runnable<HashMap<String, String>, String>
    let prompt = ChatPromptTemplate::from_messages([
        Message::system("You are a concise Rust assistant. Output only the conclusion, no extra text."),
        Message::human("{question}"),
    ]);
    let qa_chain = prompt.pipe(llm.clone()).pipe(StrOutputParser::new());

    let mut vars = HashMap::new();
    vars.insert(
        "question".to_string(),
        "Explain in one sentence what the Rust language is".to_string(),
    );
    let answer = qa_chain.invoke(vars, None).await?;
    println!("[1] Prompt+LLM+Parser\n     {answer}\n");

    // === 2. Memory + LLM + parser (multi-turn chat chain) ===
    // Chain type: Runnable<String, String>
    // Read memory → append user input → LLM → write back, all wrapped inside
    // RunnableWithMessageHistory.
    let memory = ConversationBufferMemory::new().with_return_messages(true);
    let chat_chain =
        RunnableWithMessageHistory::new(llm.clone(), memory).pipe(StrOutputParser::new());

    let r1 = chat_chain
        .invoke("My name is Xiao Ming, please remember me.".to_string(), None)
        .await?;
    let r2 = chat_chain.invoke("What is my name?".to_string(), None).await?;
    println!("[2] Memory+LLM+Parser (multi-turn)\n     turn 1: {r1}\n     turn 2: {r2}\n");

    // === 3. RAG chain (BM25 local retrieval + LLM generation) ===
    // Chain type: Runnable<String, String>
    // BM25 retrieval needs no vector store; only answer generation calls the LLM.
    let retriever = BM25Retriever::new();
    retriever.add_documents_sync(vec![
        Document::new("Rust is a systems programming language developed by Mozilla, focused on safety and performance.")
            .with_id("rust_intro"),
        Document::new("Rust's core features include the ownership system, borrow checking, and zero-cost abstractions.")
            .with_id("rust_features"),
        Document::new("LCEL (expression language) uses .pipe() to string prompts, models, and parsers into one chain.")
            .with_id("lcel_intro"),
    ]);

    let pipeline = RAGPipelineBuilder::new()
        .llm(llm)
        .retriever(retriever)
        .retrieve_k(2)
        .build()?;
    let rag_chain = RagRunnable::new(Arc::new(pipeline));

    let answer = rag_chain
        .invoke("What are Rust's core features?".to_string(), None)
        .await?;
    println!("[3] RAG chain\n     {answer}\n");

    println!("=== lcel_compose full chain complete ✅ ===");
    Ok(())
}
