# langchainrust

[![Rust](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/langchainrust.svg)](https://crates.io/crates/langchainrust)
[![Documentation](https://docs.rs/langchainrust/badge.svg)](https://docs.rs/langchainrust)

A LangChain-inspired Rust framework for building LLM applications.

**What it solves**: Build Agents, RAG, BM25 keyword search, Hybrid retrieval, LangGraph workflows, MCP tools, Guardrails, multi-agent Handoffs - all in pure Rust.

---

## Core Features

| Component | Description |
|-----------|-------------|
| **LLM** | OpenAI / Ollama / DeepSeek / Moonshot / Zhipu / Qwen / Anthropic Claude / Gemini + 多模态 Vision + Assistants API（含 requires_action 工具调度） |
| **Embeddings** | OpenAI / DeepSeek / Qwen / Local(ort ONNX Runtime,feature gate) / Mock |
| **Agents** | ReActAgent / FunctionCallingAgent / Plan-Execute / Handoffs 多 Agent 交接 / Streaming Function Calling |
| **A2A** | Agent-to-Agent 协议,AgentCard/Task/Message + Server(含 task persistence) + Client |
| **MCP** | Model Context Protocol Client + Server(Stdio + SSE),MCP 工具适配为 BaseTool |
| **Memory** | Buffer / Window / Summary / SummaryBuffer / Persistent / VectorStore(语义检索) / **ContextWindow(v0.4.1,Truncate+Summarize)** |
| **Sessions** | 多轮会话生命周期管理,可插拔存储(SessionManager + SessionStore) |
| **Chains** | LLMChain / SequentialChain / ConversationChain / RouterChain / RetrievalQA / ConversationRetrieval / Stuff / Refine / MapReduce + **Chain 流式(v0.4.1)** |
| **RAG** | Document splitting(含 SemanticSplitter), vector store, semantic retrieval, MultiQuery, HyDE, Reranking, **query_with_sources 引用溯源** |
| **Structured Output** | **with_structured_output(v0.4.1)**,StructuredOutputExt trait + JsonOutputParser 降级 |
| **BM25** | Keyword search, Chinese/English tokenization, AutoMerging, Chunked |
| **Hybrid** | BM25 + Vector hybrid retrieval, RRF fusion, Unified index |
| **LangGraph** | Graph workflows, Human-in-the-loop, Subgraph, Parallel, Checkpointer |
| **Guardrails** | 输入/输出安全护栏,SensitiveInfo / ForbiddenWords / MaxLength,GuardedAgent |
| **Token Counter** | Tiktoken 计数 + TokenTrackingLLM 用量统计 + ModelPricing 成本估算 |
| **Output Parsers** | StrOutputParser, JsonOutputParser, CommaSeparatedList, Structured, Typed |
| **Tools** | Calculator / DateTime / Math / URLFetch / Wikipedia / WebSearch / PythonREPL / HTTPTool / FileTool(沙箱) / SQLTool(只读) / **ComputerUseTool(v0.4.1)** |
| **Vector DB** | InMemory / Qdrant / MongoDB / ChromaDB / Redis / SQLite / PGVector / Pinecone / **FileVectorStore(v0.4.1)** |
| **Document Loaders** | Text / JSON / Markdown / PDF / CSV / HTML + **WebScraper / Sitemap / Docx(v0.4.1)** |
| **Cache** | LLMCache with TTL support |
| **Prompts** | PromptTemplate / ChatPromptTemplate / FewShotPromptTemplate |
| **Callbacks** | StdOut / LangSmith / FileHandler / OpenTelemetry |
| **Evaluation** | ExactMatch / StringDistance / EmbeddingSimilarity / LLMAsJudge / PairwiseJudge / ContainsKeyword / RegexMatch / LengthCheck / Bleu / Faithfulness |

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
│  ├── 多模态 Vision (ImageContent + human_with_image)        │
│  ├── OpenAI Assistants API (含 requires_action 工具调度)    │
│  └── with_structured_output (StructuredOutputExt trait)      │
├─────────────────────────────────────────────────────────────┤
│  Embeddings Layer                                            │
│  ├── OpenAIEmbeddings / DeepSeekEmbeddings                   │
│  ├── QwenEmbeddings / MockEmbeddings                         │
│  └── LocalEmbeddings (ort ONNX Runtime, feature gate)       │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer                                                 │
│  ├── ReActAgent / FunctionCallingAgent                      │
│  ├── Plan-Execute Agent (规划-执行-重规划)                   │
│  ├── Handoffs (多 Agent 交接) / Streaming Function Calling  │
│  ├── GuardedAgent (Guardrails 安全护栏)                     │
│  ├── AgentExecutor                                          │
│  ├── A2A Server/Client (Agent-to-Agent 协议)                │
│  └── LangGraph (StateGraph, Subgraph, Parallel)             │
├─────────────────────────────────────────────────────────────┤
│  MCP Layer                                                   │
│  ├── MCPClient (Stdio + SSE) -> MCPToolAdapter -> BaseTool   │
│  └── MCPServer (暴露 BaseTool 给 host 调用)                  │
├─────────────────────────────────────────────────────────────┤
│  Retrieval Layer                                             │
│  ├── RAG (TextSplitter, SemanticSplitter, VectorStore)      │
│  ├── BM25 (Keyword Search, AutoMerging)                     │
│  ├── Hybrid (BM25 + Vector, RRF Fusion)                     │
│  ├── HyDE / MultiQuery / Reranking                          │
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
│  │         + Chain streaming (逐 token 输出)                │
│  ├── Prompts (PromptTemplate, ChatPromptTemplate, FewShot)  │
│  ├── Tools (Calculator, DateTime, URLFetch, HTTP/File/SQL,  │
│  │          ComputerUseTool)                                 │
│  ├── Output Parsers                                         │
│  ├── Token Counter (Tiktoken + Cost Tracking)               │
│  ├── LLM Cache                                              │
│  ├── Evaluation (10 种评测器, 含 Faithfulness)             │
│  └── Callbacks (LangSmith, StdOut, FileHandler, Otel)       │
└─────────────────────────────────────────────────────────────┘
```

---

## What's New in 0.4.1

- **Assistants requires_action 工具调度**: `OpenAIAssistant` 遇 `requires_action` 自动解析 tool_calls → ToolRegistry 执行 → submit_tool_outputs → 继续轮询至 completed
- **A2A Agent 协议**: `A2AServer` 暴露 agent + `A2AClient` 调远程 agent,JSON-RPC 风格(tasks/send/get/cancel),内存 task persistence
- **with_structured_output**: `StructuredOutputExt` trait,一行拿强类型结构,按 provider 走 function calling 或 JsonOutputParser 降级
- **Chain 流式**: `BaseChain::stream()` + LLMChain/ConversationChain 覆写,逐 token 回调 `on_llm_new_token`
- **ContextWindow 长上下文管理**: Truncate(按 token 数截断) + Summarize(LLM 摘要压缩) 策略,TokenCounter 集成
- **FileVectorStore**: JSON 持久化向量存储,原子写入(tmp+rename),跨实例持久化,维度校验
- **ComputerUseTool**: Anthropic computer use API 接入 + Native 截图/输入(feature gate `computer-use-native`)
- **更多 Document Loader**: DocxLoader(ZIP+XML)、WebScraperLoader(递归爬取+同域过滤)、SitemapLoader(sitemap.xml 解析)
- **LocalEmbeddings ort**: ONNX Runtime 神经网络嵌入(feature gate `local-embeddings`),替代 bag-of-words 占位
- **wiremock 测试基础设施**: dev-dependency + mock 辅助函数,默认测试不打真实网络
- **MSRV 声明**: `rust-version = "1.82"`,CI 含 1.82 矩阵
- **criterion benchmark**: benches/ 下 retrieval(6)/splitter(4)/embedding(4) 组基准
- **12+ 新 examples**: 覆盖 evaluation/mcp_server/guardrails/sessions/context_window/vectorstore_memory/semantic_splitter/file_vectorstore/otel/assistants/handoffs/plan_execute/token_counter

## What's New in 0.4.0

- **Evaluation 评估模块**: 10 种评测器(字面 / 语义 / 规则 / 经典 NLP / RAG),`EvalRunner` 跑评测集出报告,`Faithfulness` 检测 RAG 幻觉
- **MCP Server**: `MCPServer` 把本地工具暴露为 MCP Server,供 Claude Desktop / Cursor 调用
- **向量检索记忆**: `VectorStoreRetrieverMemory` 按当前输入语义召回历史
- **OpenAI Assistants API**: `OpenAIAssistant` 服务端会话状态(Assistants / Threads / Run)
- **语义分块**: `SemanticSplitter` 相邻句相似度骤降处断块
- **本地嵌入**: `LocalEmbeddings` 离线,纯 Rust 无外部依赖
- **OpenTelemetry 追踪**: `OtelHandler` 执行事件转 OTel span(feature `opentelemetry`)

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
langchainrust = "0.4.1"
tokio = { version = "1.0", features = ["full"] }

# Optional features
langchainrust = { version = "0.4.1", features = ["mongodb-persistence"] }  # MongoDB storage
langchainrust = { version = "0.4.1", features = ["qdrant-integration"] }    # Qdrant vector DB
langchainrust = { version = "0.4.1", features = ["redis-storage"] }         # Redis storage
langchainrust = { version = "0.4.1", features = ["sqlite-storage"] }        # SQLite storage (+ SQLTool)
langchainrust = { version = "0.4.1", features = ["pgvector-storage"] }      # PGVector (需自配 sqlx/pgvector 依赖)
langchainrust = { version = "0.4.1", features = ["local-embeddings"] }      # Local ONNX embeddings (需 ort)
# PineconeStore / FileVectorStore 无需 feature,默认可用
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

`examples/` 目录提供 25 个可运行示例,覆盖核心功能:

| 分类 | 示例 | 需 API Key |
|------|------|-----------|
| basic | chat / streaming / multi_provider / token_counter | 是 |
| agent | function_calling / multi_tool / assistants / handoffs / plan_execute | 是 |
| rag | bm25_search / document_loaders / file_vectorstore / semantic_splitter | 否 |
| langgraph | basic_graph / conditional_edge | 否 |
| memory | buffer_memory / context_window / sessions / vectorstore_memory | 否 |
| chains | llm_chain / sequential_chain | 是 |
| evaluation | evaluation | 否 |
| guardrails | guardrails | 否 |
| mcp_server | mcp_server | 否 |
| otel | otel_tracing | 否 |

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
