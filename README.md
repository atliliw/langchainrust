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
| **LLM** | OpenAI / Ollama / DeepSeek / Moonshot / Zhipu / Qwen / Anthropic Claude |
| **Embeddings** | OpenAI / DeepSeek / Qwen embeddings |
| **Agents** | ReActAgent + FunctionCallingAgent |
| **Memory** | Buffer / Window / Summary / SummaryBuffer |
| **Chains** | LLMChain / SequentialChain / RetrievalQA |
| **RAG** | Document splitting, vector store, semantic retrieval |
| **BM25** | Keyword search, Chinese/English tokenization, AutoMerging |
| **Hybrid** | BM25 + Vector hybrid retrieval, RRF fusion |
| **LangGraph** | Graph workflows, Human-in-the-loop, Subgraph |
| **Tools** | Calculator / DateTime / Math / URLFetch |
| **Vector DB** | InMemory / Qdrant / MongoDB |

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
│  └── Callbacks (LangSmith, StdOut, multipart batch)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

```toml
[dependencies]
langchainrust = "0.2.6"
tokio = { version = "1.0", features = ["full"] }

# Optional features
langchainrust = { version = "0.2.6", features = ["mongodb-persistence"] }  # MongoDB storage
langchainrust = { version = "0.2.6", features = ["qdrant-integration"] }    # Qdrant vector DB
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