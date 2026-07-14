# langchainrust

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/langchainrust.svg)](https://crates.io/crates/langchainrust)
[![Documentation](https://docs.rs/langchainrust/badge.svg)](https://docs.rs/langchainrust)

A LangChain-inspired Rust framework for building LLM applications.

**What it solves**: Build Agents, RAG, BM25 keyword search, Hybrid retrieval, LangGraph workflows, MCP tools, Guardrails, multi-agent Handoffs - all in pure Rust.

---

## Core Features

| Component | Description |
|-----------|-------------|
| **LLM** | OpenAI / Ollama / DeepSeek / Moonshot / Zhipu / Qwen / Anthropic Claude / Gemini + 多模态 Vision |
| **Embeddings** | OpenAI / DeepSeek / Qwen embeddings |
| **Agents** | ReActAgent / FunctionCallingAgent / Plan-Execute / Handoffs 多 Agent 交接 / Streaming Function Calling |
| **MCP** | Model Context Protocol Client(Stdio + SSE),MCP 工具适配为 BaseTool |
| **Memory** | Buffer / Window / Summary / SummaryBuffer / Persistent |
| **Sessions** | 多轮会话生命周期管理,可插拔存储(SessionManager + SessionStore) |
| **Chains** | LLMChain / SequentialChain / ConversationChain / RouterChain / RetrievalQA / ConversationRetrieval / Stuff / Refine / MapReduce |
| **RAG** | Document splitting, vector store, semantic retrieval, MultiQuery, HyDE, Reranking |
| **BM25** | Keyword search, Chinese/English tokenization, AutoMerging, Chunked |
| **Hybrid** | BM25 + Vector hybrid retrieval, RRF fusion, Unified index |
| **LangGraph** | Graph workflows, Human-in-the-loop, Subgraph, Parallel, Checkpointer |
| **Guardrails** | 输入/输出安全护栏,SensitiveInfo / ForbiddenWords / MaxLength,GuardedAgent |
| **Token Counter** | Tiktoken 计数 + TokenTrackingLLM 用量统计 + ModelPricing 成本估算 |
| **Output Parsers** | StrOutputParser, JsonOutputParser, CommaSeparatedList, Structured, Typed |
| **Tools** | Calculator / DateTime / Math / URLFetch / Wikipedia / WebSearch / PythonREPL / HTTPTool / FileTool(沙箱) / SQLTool(只读) |
| **Vector DB** | InMemory / Qdrant / MongoDB / ChromaDB / Redis / SQLite / PGVector / Pinecone |
| **Document Loaders** | Text / JSON / Markdown / PDF / CSV / HTML |
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
│  ├── AnthropicChat (Claude API) / GeminiChat                 │
│  ├── Function Calling (bind_tools) / Streaming (stream_chat)│
│  └── 多模态 Vision (ImageContent + human_with_image)        │
├─────────────────────────────────────────────────────────────┤
│  Embeddings Layer                                            │
│  ├── OpenAIEmbeddings / DeepSeekEmbeddings                   │
│  └── QwenEmbeddings / MockEmbeddings                         │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer                                                 │
│  ├── ReActAgent / FunctionCallingAgent                      │
│  ├── Plan-Execute Agent (规划-执行-重规划)                   │
│  ├── Handoffs (多 Agent 交接) / Streaming Function Calling  │
│  ├── GuardedAgent (Guardrails 安全护栏)                     │
│  ├── AgentExecutor                                          │
│  └── LangGraph (StateGraph, Subgraph, Parallel)             │
├─────────────────────────────────────────────────────────────┤
│  MCP Layer                                                   │
│  └── MCPClient (Stdio + SSE) -> MCPToolAdapter -> BaseTool   │
├─────────────────────────────────────────────────────────────┤
│  Retrieval Layer                                             │
│  ├── RAG (TextSplitter, VectorStore)                        │
│  ├── BM25 (Keyword Search, AutoMerging)                     │
│  ├── Hybrid (BM25 + Vector, RRF Fusion)                     │
│  ├── HyDE / MultiQuery / Reranking                          │
│  └── Loaders (Text/JSON/MD/PDF/CSV/HTML)                    │
├─────────────────────────────────────────────────────────────┤
│  Storage Layer                                               │
│  ├── Vector DB (InMemory, Qdrant, MongoDB, ChromaDB,        │
│  │              Redis, SQLite, PGVector, Pinecone)          │
│  └── Sessions (SessionManager + SessionStore)               │
├─────────────────────────────────────────────────────────────┤
│  Utility Layer                                               │
│  ├── Memory (Buffer, Window, Summary, SummaryBuffer)        │
│  ├── Chains (LLMChain, SequentialChain, RetrievalQA, ...)   │
│  ├── Prompts (PromptTemplate, ChatPromptTemplate, FewShot)  │
│  ├── Tools (Calculator, DateTime, URLFetch, HTTP/File/SQL)  │
│  ├── Output Parsers                                         │
│  ├── Token Counter (Tiktoken + Cost Tracking)               │
│  ├── LLM Cache                                              │
│  └── Callbacks (LangSmith, StdOut, FileHandler)             │
└─────────────────────────────────────────────────────────────┘
```

---

## What's New in 0.3.0

- **MCP 协议**: 连接任意 MCP Server(stdio/SSE),工具自动适配为 `BaseTool` 供 Agent 调用
- **多模态 Vision**: `ImageContent` + `Message::human_with_image`,OpenAI / Ollama 均支持
- **Sessions 会话管理**: `SessionManager` + 可插拔 `SessionStore`,多轮对话生命周期
- **Token 计数器**: `TiktokenCounter` + `TokenTrackingLLM` 用量统计 + `ModelPricing` 成本估算
- **Guardrails 安全护栏**: 输入/输出验证,SensitiveInfo / ForbiddenWords / MaxLength,`GuardedAgent`
- **Plan-Execute Agent**: 规划 → 执行 → 失败重规划(`PlanExecuteAgent`)
- **Handoffs 多 Agent 交接**: `HandoffManager` + `HandoffTool`,主 Agent 委托专业 Agent
- **Streaming Tool Calls**: `StreamingFunctionCallingAgent` 流式输出 + 工具调用事件
- **扩展工具**: `HTTPTool` / `FileTool`(沙箱)/ `SQLTool`(只读)
- **PGVector / Pinecone**: 新增两个向量库后端
- **HTML Loader**: 去标签/脚本/样式,提取纯文本

详见 [Usage Guide(中文)](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE.md)。

---

## Installation

```toml
[dependencies]
langchainrust = "0.3.0"
tokio = { version = "1.0", features = ["full"] }

# Optional features
langchainrust = { version = "0.3.0", features = ["mongodb-persistence"] }  # MongoDB storage
langchainrust = { version = "0.3.0", features = ["qdrant-integration"] }    # Qdrant vector DB
langchainrust = { version = "0.3.0", features = ["redis-storage"] }         # Redis storage
langchainrust = { version = "0.3.0", features = ["sqlite-storage"] }        # SQLite storage (+ SQLTool)
langchainrust = { version = "0.3.0", features = ["pgvector-storage"] }      # PGVector (需自配 sqlx/pgvector 依赖)
# PineconeStore 无需 feature,默认可用(reqwest HTTP API)
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
| [Usage Guide (中文)](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE.md) | LLM、Agent、Memory、RAG、BM25、Hybrid、LangGraph、MCP、Sessions、Guardrails、Token Counter、Plan-Execute、Handoffs、Streaming 详细用法 |
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
