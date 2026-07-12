# langchainrust

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/langchainrust.svg)](https://crates.io/crates/langchainrust)
[![Documentation](https://docs.rs/langchainrust/badge.svg)](https://docs.rs/langchainrust)

A LangChain-inspired Rust framework for building LLM applications.

**What it solves**: Build Agents, RAG, BM25 keyword search, Hybrid retrieval, LangGraph workflows - all in pure Rust.

---

## Core Features

| Component | Description |
|-----------|-------------|
| **LLM** | OpenAI / Ollama / DeepSeek / Moonshot / Zhipu / Qwen / Anthropic Claude / Gemini |
| **Embeddings** | OpenAI / DeepSeek / Qwen embeddings |
| **Agents** | ReActAgent + FunctionCallingAgent |
| **Memory** | Buffer / Window / Summary / SummaryBuffer / Persistent |
| **Chains** | LLMChain / SequentialChain / ConversationChain / RouterChain / RetrievalQA / ConversationRetrieval / Stuff / Refine / MapReduce |
| **RAG** | Document splitting, vector store, semantic retrieval, MultiQuery, HyDE, Reranking |
| **BM25** | Keyword search, Chinese/English tokenization, AutoMerging, Chunked |
| **Hybrid** | BM25 + Vector hybrid retrieval, RRF fusion, Unified index |
| **LangGraph** | Graph workflows, Human-in-the-loop, Subgraph, Parallel, Checkpointer |
| **Output Parsers** | StrOutputParser, JsonOutputParser, CommaSeparatedList, Structured, Typed |
| **Tools** | Calculator / DateTime / Math / URLFetch / Wikipedia / WebSearch / PythonREPL |
| **Vector DB** | InMemory / Qdrant / MongoDB / ChromaDB / Redis / SQLite |
| **Cache** | LLMCache with TTL support |
| **Prompts** | PromptTemplate / ChatPromptTemplate / FewShotPromptTemplate |
| **Callbacks** | StdOut / LangSmith / FileHandler |

Full documentation: [中文文档](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE.md) | [English](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE_EN.md)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      langchainrust                           │
├─────────────────────────────────────────────────────────────┤
│  LLM Layer                                                   │
│  ├── OpenAIChat / OllamaChat                                 │
│  ├── DeepSeek / Moonshot / Zhipu / Qwen (OpenAI compatible) │
│  ├── AnthropicChat (Claude API)                              │
│  ├── GeminiChat                                              │
│  ├── Function Calling (bind_tools)                          │
│  └── Streaming (stream_chat)                                │
├─────────────────────────────────────────────────────────────┤
│  Embeddings Layer                                            │
│  ├── OpenAIEmbeddings / DeepSeekEmbeddings                   │
│  └── QwenEmbeddings / MockEmbeddings                         │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer                                                 │
│  ├── ReActAgent / FunctionCallingAgent                      │
│  ├── AgentExecutor                                          │
│  └── LangGraph (StateGraph, Subgraph, Parallel)             │
├─────────────────────────────────────────────────────────────┤
│  Retrieval Layer                                             │
│  ├── RAG (TextSplitter, VectorStore)                        │
│  ├── BM25 (Keyword Search, AutoMerging)                     │
│  ├── Hybrid (BM25 + Vector, RRF Fusion)                     │
│  └── HyDE / MultiQuery / Reranking                          │
│  └── Storage (InMemory, Qdrant, MongoDB)                    │
├─────────────────────────────────────────────────────────────┤
│  Utility Layer                                               │
│  ├── Memory (Buffer, Window, Summary)                       │
│  ├── Chains (LLMChain, SequentialChain)                     │
│  ├── Prompts (PromptTemplate, ChatPromptTemplate)           │
│  ├── Tools (Calculator, DateTime, URLFetch)                 │
│  ├── Output Parsers                                         │
│  ├── LLM Cache                                              │
│  └── Callbacks (LangSmith, StdOut, multipart batch)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

```toml
[dependencies]
langchainrust = "0.2.20"
tokio = { version = "1.0", features = ["full"] }

# Optional features
langchainrust = { version = "0.2.20", features = ["mongodb-persistence"] }  # MongoDB storage
langchainrust = { version = "0.2.20", features = ["qdrant-integration"] }    # Qdrant vector DB
langchainrust = { version = "0.2.20", features = ["redis-storage"] }         # Redis storage
langchainrust = { version = "0.2.20", features = ["sqlite-storage"] }        # SQLite storage
```

---

## Quick Start

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")?,
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        ..Default::default()
    };
    
    let llm = OpenAIChat::new(config);
    
    let response = llm.chat(vec![
        Message::system("You are a helpful assistant."),
        Message::human("What is Rust?"),
    ], None).await?;
    
    println!("{}", response.content);
    Ok(())
}
```

### Multi-Provider Support

```rust
use langchainrust::{
    DeepSeekChat, MoonshotChat, ZhipuChat, QwenChat,
    AnthropicChat, OllamaChat,
};

let deepseek = DeepSeekChat::from_env();
let moonshot = MoonshotChat::with_model("moonshot-v1-128k");
let claude = AnthropicChat::from_env();
let ollama = OllamaChat::new("llama3.2");
```

### BM25 Keyword Search

```rust
use langchainrust::{BM25Retriever, Document};

let mut retriever = BM25Retriever::new();

retriever.add_documents_sync(vec![
    Document::new("Rust is a systems programming language"),
    Document::new("Python is a scripting language"),
]);

let results = retriever.search("systems programming", 3);

for result in results {
    println!("Document: {}", result.document.content);
    println!("Score: {}", result.score);
}
```

More examples in [Usage Guide (中文)](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE.md).

---

## Examples

`examples/` 目录提供 12 个可运行示例,覆盖核心功能:

| 分类 | 示例 | 运行命令 | 需 API Key |
|------|------|---------|-----------|
| basic | chat / streaming / multi_provider | `cargo run --example basic_chat` | 是 |
| agent | function_calling / multi_tool | `cargo run --example agent_function_calling` | 是 |
| rag | bm25_search / document_loaders | `cargo run --example rag_bm25_search` | 否 |
| langgraph | basic_graph / conditional_edge | `cargo run --example langgraph_basic_graph` | 否 |
| memory | buffer_memory | `cargo run --example memory_buffer_memory` | 否 |
| chains | llm_chain / sequential_chain | `cargo run --example chains_llm_chain` | 是 |

需要 API Key 的示例从环境变量读取:

```bash
export OPENAI_API_KEY="your-key"
cargo run --example basic_chat
```

无需 API Key 的示例(BM25 / LangGraph / Memory / Loader)可直接运行,适合快速体验。

---

## Documentation

| Docs | Content |
|------|---------|
| [Usage Guide (中文)](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE.md) | LLM、Agent、Memory、RAG、BM25、Hybrid、LangGraph 详细用法 |
| [Usage Guide (English)](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE_EN.md) | Detailed usage for all components |
| [API Docs](https://docs.rs/langchainrust) | Rust API documentation |

---

## Testing

```bash
cargo test
```

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT or Apache-2.0, at your option.
