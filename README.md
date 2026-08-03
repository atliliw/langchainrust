# langchainrust

[![Rust](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/langchainrust.svg)](https://crates.io/crates/langchainrust)
[![Documentation](https://docs.rs/langchainrust/badge.svg)](https://docs.rs/langchainrust)
[![CI](https://github.com/atliliw/langchainrust/actions/workflows/ci.yml/badge.svg)](https://github.com/atliliw/langchainrust/actions/workflows/ci.yml)
[![Crates.io Downloads](https://img.shields.io/crates/d/langchainrust.svg)](https://crates.io/crates/langchainrust)

A LangChain-inspired Rust framework for building LLM applications.

**What it solves**: Build Agents, RAG, BM25 keyword search, Hybrid retrieval, LangGraph workflows, MCP tools, Guardrails, multi-agent Handoffs — all in pure Rust.

---

## Core Features

| Component | Description |
|-----------|-------------|
| **LLM** | OpenAI / Ollama / DeepSeek / Moonshot / Zhipu / Qwen / Anthropic Claude / Gemini + Multimodal Vision + Assistants API (with requires_action tool dispatch) |
| **Embeddings** | OpenAI / DeepSeek / Qwen / Local (ort ONNX Runtime, feature gate) / Mock |
| **Agents** | ReActAgent / FunctionCallingAgent / Plan-Execute / Handoffs (multi-agent handoff) / Streaming Function Calling |
| **A2A** | Agent-to-Agent protocol, AgentCard/Task/Message + Server (with task persistence) + Client |
| **MCP** | Model Context Protocol Client + Server (Stdio + SSE), full 6 primitives, MCP tool adapter to BaseTool |
| **Memory** | Buffer / Window / Summary / SummaryBuffer / Persistent / VectorStore (semantic retrieval) / ContextWindow (Truncate + Summarize) |
| **Sessions** | Multi-turn conversation lifecycle management, pluggable storage (SessionManager + SessionStore) |
| **Chains** | LLMChain / SequentialChain / ConversationChain / RouterChain / RetrievalQA / ConversationRetrieval / Stuff / Refine / MapReduce + Chain streaming |
| **RAG** | Document splitting (including SemanticSplitter), vector store, semantic retrieval, MultiQuery, HyDE, Reranking, query_with_sources (citation tracing) |
| **Structured Output** | with_structured_output, StructuredOutputExt trait + JsonOutputParser fallback, Streaming Structured Output |
| **BM25** | Keyword search, Chinese/English tokenization, AutoMerging, Chunked |
| **Hybrid** | BM25 + Vector hybrid retrieval, RRF fusion, Unified index |
| **LangGraph** | Graph workflows, Human-in-the-loop, Subgraph, Parallel, Checkpointer |
| **Guardrails** | Input/output safety guardrails, SensitiveInfo / ForbiddenWords / MaxLength, GuardedAgent |
| **Token Counter** | Tiktoken counting + TokenTrackingLLM usage statistics + ModelPricing cost estimation |
| **Output Parsers** | StrOutputParser, JsonOutputParser, CommaSeparatedList, Structured, Typed |
| **Tools** | Calculator / DateTime / Math / URLFetch / Wikipedia / WebSearch / PythonREPL / HTTPTool / FileTool (sandbox) / SQLTool (read-only) / ComputerUseTool |
| **Vector DB** | InMemory / Qdrant / MongoDB / ChromaDB / Redis / SQLite / PGVector / Pinecone / FileVectorStore |
| **Document Loaders** | Text / JSON / Markdown / PDF / CSV / HTML + WebScraper / Sitemap / Docx |
| **Cache** | LLMCache with TTL support |
| **Prompts** | PromptTemplate / ChatPromptTemplate / FewShotPromptTemplate |
| **Callbacks** | StdOut / LangSmith / FileHandler / OpenTelemetry |
| **Evaluation** | ExactMatch / StringDistance / EmbeddingSimilarity / LLMAsJudge / PairwiseJudge / ContainsKeyword / RegexMatch / LengthCheck / Bleu / Faithfulness |
| **Advanced RAG** | CorrectiveRAG (self-correcting) / AdaptiveRAG (adaptive retrieval) / GraphRAG (knowledge graph) |
| **Model Routing** | RouterLLM with 5 strategies (Fallback / RoundRobin / LeastLatency / LowestCost / InputDirected) |
| **Deep Research** | Multi-round deep research agent with sub-topic decomposition, parallel search, deduplication, and citation reporting |
| **Code Interpreter** | LocalSandbox (subprocess + timeout) + E2B cloud sandbox + WASM sandbox (feature gate) |
| **Batch API** | BatchClient for OpenAI/Anthropic batch inference, 50% cost reduction |
| **Tracing** | Tracer + SpanGuard (RAII), InMemory / Console / OTel backends, parent-child span tree |

Full documentation: [Usage Guide](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE_EN.md) | [API Docs](https://docs.rs/langchainrust)

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
│  ├── Multimodal Vision (ImageContent + human_with_image)    │
│  ├── OpenAI Assistants API (with requires_action dispatch)   │
│  ├── OpenAI Responses API (web_search/file_search/code/...)  │
│  ├── Anthropic Extended Thinking (with_thinking)             │
│  ├── RouterLLM (5 strategies + Fallback)                     │
│  ├── BatchClient (OpenAI/Anthropic batch inference)          │
│  └── with_structured_output (StructuredOutputExt trait)      │
├─────────────────────────────────────────────────────────────┤
│  Embeddings Layer                                            │
│  ├── OpenAIEmbeddings / DeepSeekEmbeddings                   │
│  ├── QwenEmbeddings / MockEmbeddings                         │
│  └── LocalEmbeddings (ort ONNX Runtime, feature gate)       │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer                                                 │
│  ├── ReActAgent / FunctionCallingAgent                      │
│  ├── Plan-Execute Agent (plan -> execute -> replan)         │
│  ├── Handoffs (multi-agent handoff) / Streaming FC          │
│  ├── GuardedAgent (Guardrails safety)                       │
│  ├── DeepResearchAgent (multi-round research + citations)   │
│  ├── AgentExecutor                                          │
│  ├── A2A Server/Client (Agent-to-Agent protocol)            │
│  └── LangGraph (StateGraph, Subgraph, Parallel)             │
├─────────────────────────────────────────────────────────────┤
│  MCP Layer                                                   │
│  ├── MCPClient (Stdio + SSE) -> MCPToolAdapter -> BaseTool   │
│  ├── MCPServer (expose BaseTool to host)                     │
│  └── Full 6 primitives (resources/prompts/completion/...)    │
├─────────────────────────────────────────────────────────────┤
│  Retrieval Layer                                             │
│  ├── RAG (TextSplitter, SemanticSplitter, VectorStore)      │
│  ├── BM25 (Keyword Search, AutoMerging)                     │
│  ├── Hybrid (BM25 + Vector, RRF Fusion)                     │
│  ├── HyDE / MultiQuery / Reranking                          │
│  ├── CorrectiveRAG (grade + rewrite + hallucination detect) │
│  ├── AdaptiveRAG (LLM-routed retrieval strategy)            │
│  ├── GraphRAG (knowledge graph + community detection)       │
│  └── Loaders (Text/JSON/MD/PDF/CSV/HTML/Docx/Web/Sitemap)  │
├─────────────────────────────────────────────────────────────┤
│  Storage Layer                                               │
│  ├── Vector DB (InMemory, Qdrant, MongoDB, ChromaDB,        │
│  │              Redis, SQLite, PGVector, Pinecone, File)    │
│  └── Sessions (SessionManager + SessionStore)               │
├─────────────────────────────────────────────────────────────┤
│  Utility Layer                                               │
│  ├── Memory (Buffer, Window, Summary, SummaryBuffer, Vector,│
│  │           ContextWindow[Truncate+Summarize])             │
│  ├── Chains (LLMChain, SequentialChain, RetrievalQA, ...)   │
│  │         + Chain streaming (per-token output)              │
│  ├── Prompts (PromptTemplate, ChatPromptTemplate, FewShot)  │
│  ├── Tools (Calculator, DateTime, URLFetch, HTTP/File/SQL,  │
│  │          ComputerUseTool, CodeSandbox)                    │
│  ├── Output Parsers                                         │
│  ├── Token Counter (Tiktoken + Cost Tracking)               │
│  ├── LLM Cache                                              │
│  ├── Evaluation (10 evaluators, including Faithfulness)     │
│  ├── Tracing (Tracer + SpanGuard, InMemory/Console/OTel)   │
│  └── Callbacks (LangSmith, StdOut, FileHandler, Otel)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

```toml
[dependencies]
langchainrust = "0.7.1"
tokio = { version = "1.0", features = ["full"] }

# Optional features
langchainrust = { version = "0.7.1", features = ["mongodb-persistence"] }  # MongoDB storage
langchainrust = { version = "0.7.1", features = ["qdrant-integration"] }    # Qdrant vector DB
langchainrust = { version = "0.7.1", features = ["redis-storage"] }         # Redis storage
langchainrust = { version = "0.7.1", features = ["sqlite-storage"] }        # SQLite storage (+ SQLTool)
langchainrust = { version = "0.7.1", features = ["pgvector-storage"] }      # PGVector (requires user-configured sqlx/pgvector deps)
langchainrust = { version = "0.7.1", features = ["local-embeddings"] }      # Local ONNX embeddings (requires ort)
langchainrust = { version = "0.7.1", features = ["sandbox-e2b"] }           # E2B cloud sandbox
langchainrust = { version = "0.7.1", features = ["sandbox-wasm"] }          # WASM sandbox
langchainrust = { version = "0.7.1", features = ["opentelemetry"] }         # OpenTelemetry tracing
# PineconeStore / FileVectorStore require no feature flag, available by default
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

More examples in [Usage Guide](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE_EN.md).

---

## Examples

The `examples/` directory provides 25+ runnable examples covering core functionality:

| Category | Examples | Requires API Key |
|----------|----------|-----------------|
| basic | chat / streaming / multi_provider / token_counter | Yes |
| agent | function_calling / multi_tool / assistants / handoffs / plan_execute | Yes |
| rag | bm25_search / document_loaders / file_vectorstore / semantic_splitter | No |
| langgraph | basic_graph / conditional_edge | No |
| memory | buffer_memory / context_window / sessions / vectorstore_memory | No |
| chains | llm_chain / sequential_chain | Yes |
| evaluation | evaluation | No |
| guardrails | guardrails | No |
| mcp_server | mcp_server | No |
| otel | otel_tracing | No |

Examples requiring API keys read from environment variables:

```bash
export OPENAI_API_KEY="your-key"
cargo run --example basic_chat
```

Examples without API keys (BM25 / LangGraph / Memory / Loader) can run directly — great for quick exploration.

---

## Documentation

| Docs | Content |
|------|---------|
| [Usage Guide](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE_EN.md) | Detailed usage for all components |
| [API Docs](https://docs.rs/langchainrust) | Rust API documentation |
| [Changelog](https://github.com/atliliw/langchainrust/blob/main/CHANGELOG.md) | Release history and breaking changes |

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
