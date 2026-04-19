# langchainrust

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/langchainrust.svg)](https://crates.io/crates/langchainrust)
[![Documentation](https://docs.rs/langchainrust/badge.svg)](https://docs.rs/langchainrust)

A LangChain-inspired Rust framework for building LLM applications.

一个受 LangChain 启发的 Rust 框架，用于构建 LLM 应用。

**解决问题**：让 Rust 开发者能够快速构建 Agent、RAG、BM25 关键词检索、Hybrid 混合检索、LangGraph 工作流等 LLM 应用。

---


## 核心特性

| 组件 | 功能 |
|------|------|
| **LLM** | OpenAI / Ollama 兼容接口，流式输出，Function Calling |
| **Agents** | ReActAgent + FunctionCallingAgent |
| **Memory** | Buffer / Window / Summary / SummaryBuffer |
| **Chains** | LLMChain / SequentialChain / RetrievalQA |
| **RAG** | 文档分割、向量存储、语义检索 |
| **BM25** | 关键词检索、中英文分词、AutoMerging |
| **Hybrid** | BM25 + 向量混合检索、RRF 融合 |
| **LangGraph** | 图状工作流、Human-in-the-loop、Subgraph |
| **Tools** | Calculator / DateTime / Math / URLFetch |
| **MongoDB** | 持久化存储后端（feature: mongodb-persistence） |

完整功能文档: [中文文档](https://atliliw.github.io/langchainrust/docs/features.html) | [英文文档](https://atliliw.github.io/langchainrust/docs/features_en.html)

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      langchainrust                           │
├─────────────────────────────────────────────────────────────┤
│  LLM Layer                                                   │
│  ├── OpenAIChat / OllamaChat                                 │
│  ├── Function Calling (bind_tools)                          │
│  └── Streaming (stream_chat)                                │
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
│  └── Storage (InMemory, MongoDB)                            │
├─────────────────────────────────────────────────────────────┤
│  Utility Layer                                               │
│  ├── Memory (Buffer, Window, Summary)                       │
│  ├── Chains (LLMChain, SequentialChain)                     │
│  ├── Prompts (PromptTemplate, ChatPromptTemplate)           │
│  ├── Tools (Calculator, DateTime, URLFetch)                 │
│  └── Callbacks (LangSmith, StdOut)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 安装

```toml
[dependencies]
langchainrust = "0.2.6"
tokio = { version = "1.0", features = ["full"] }

# 可选功能
langchainrust = { version = "0.2.6", features = ["mongodb-persistence"] }  # MongoDB 存储
langchainrust = { version = "0.2.6", features = ["qdrant-integration"] }    # Qdrant 向量库
```

---

## 快速开始

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
        Message::system("你是一个友好的助手。"),
        Message::human("什么是 Rust？"),
    ], None).await?;
    
    println!("{}", response.content);
    Ok(())
}
```

### BM25 关键词检索

```rust
use langchainrust::{BM25Retriever, Document};

let mut retriever = BM25Retriever::new();

retriever.add_documents_sync(vec![
    Document::new("Rust 是一门系统编程语言"),
    Document::new("Python 是脚本语言"),
]);

let results = retriever.search("系统编程", 3);

for result in results {
    println!("文档: {}", result.document.content);
    println!("评分: {}", result.score);
}
```

更多示例见 [使用指南](docs/USAGE.md)。

---

## 文档

| 文档 | 内容 |
|------|------|
| [使用指南](docs/USAGE.md) | LLM、Agent、Memory、RAG、BM25、Hybrid、LangGraph 详细用法 |
| [中文功能文档](https://atliliw.github.io/langchainrust/docs/features.html) | 功能总览 + 使用示例（在线） |
| [英文功能文档](https://atliliw.github.io/langchainrust/docs/features_en.html) | 功能总览 + 使用示例（在线） |
| [API 文档](https://docs.rs/langchainrust) | Rust API 文档 |

---

## 示例

```bash
# 无需 API Key
cargo run --example prompt_template
cargo run --example tools

# 需要 API Key
export OPENAI_API_KEY="your-key"
cargo run --example hello_llm
cargo run --example agent_with_tools
```

详细用法见 [使用指南](docs/USAGE.md)。

---

## 测试

```bash
cargo test
```

---

## Roadmap

| 状态 | 功能 |
|------|------|
| ✅ 完成 | LangGraph、BM25、Hybrid、MongoDB 存储 |
| ⏳ 开发中 | LCEL 组合操作符 |
| 📋 规划中 | DeepSeek LLM、MultiQueryRetriever、Redis 存储 |

详见 [ROADMAP.md](ROADMAP.md)。

---

## 贡献

欢迎贡献！见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 许可证

MIT 或 Apache-2.0，任选其一。