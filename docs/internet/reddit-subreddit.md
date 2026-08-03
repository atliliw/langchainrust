# Reddit Subreddit: r/langchainrust

## 基本信息

- **Name**: langchainrust
- **Title**: langchainrust — LLM framework in pure Rust
- **Type**: Public

---

## Description (简短描述，500字符限制)

```
A LangChain-inspired framework for building LLM applications in pure Rust. 8 LLM providers (OpenAI, Anthropic, Gemini, Ollama, DeepSeek, Qwen, Moonshot, Zhipu), LangGraph workflows, MCP client/server, GraphRAG, CorrectiveRAG, AdaptiveRAG, BM25+Hybrid retrieval, 7 memory types, guardrails, and more. No Python needed.
```

---

## Sidebar / About 文本 (长描述，放在社区介绍里)

```
## langchainrust

A full-featured LLM framework in pure Rust. No Python dependency, no GC pauses, just safe and fast.

### Quick Start

```toml
[dependencies]
langchainrust = "0.7.1"
tokio = { version = "1.0", features = ["full"] }
```

### What's Included

| Component | Features |
|-----------|----------|
| LLMs | OpenAI, Anthropic, Gemini, Ollama, DeepSeek, Qwen, Moonshot, Zhipu |
| Agents | ReAct, FunctionCalling, Plan-Execute, Handoffs, DeepResearch, GuardedAgent |
| RAG | Basic → CorrectiveRAG → AdaptiveRAG → GraphRAG |
| Search | BM25 (Chinese/English), Hybrid (BM25+Vector RRF fusion) |
| Graph | LangGraph (Human-in-the-loop, Subgraph, Parallel, Checkpoint) |
| MCP | Full 6-primitive Client + Server (Stdio + SSE) |
| Memory | 7 types (Buffer → ContextWindow) |
| Vector DB | 9 backends (InMemory, SQLite, Qdrant, ChromaDB, Redis, PGVector, MongoDB, Pinecone, File) |
| Tools | 12+ built-in (Calculator, WebSearch, SQL, PythonREPL, ComputerUse, CodeSandbox) |
| Safety | Guardrails, SSRF prevention, SQL injection guards |
| Routing | RouterLLM with 5 strategies + Batch API (50% cost reduction) |

### Links

- GitHub: https://github.com/atliliw/langchainrust
- Docs: https://docs.rs/langchainrust
- crates.io: https://crates.io/crates/langchainrust

### Community Rules

1. Be respectful and constructive
2. Posts should be related to langchainrust or Rust LLM development
3. Bug reports and feature requests welcome
4. Share your projects built with langchainrust!
5. No spam or off-topic content
```

---

## Welcome Message (新成员欢迎消息)

```
Welcome to r/langchainrust! 🦀

This is the community for langchainrust, a full-featured LLM framework in pure Rust.

Getting started:
- GitHub: https://github.com/atliliw/langchainrust
- Docs: https://docs.rs/langchainrust
- Quick start: `cargo add langchainrust`

Feel free to ask questions, share your projects, or contribute!
```

---

## 推荐的帖子类型

创建后自己先发几个帖子，让社区看起来不空：

1. **[Announcement] langchainrust v0.7.1 released** — 功能列表 + 链接
2. **[Tutorial] Getting started with Ollama + langchainrust** — 简单教程
3. **[Discussion] What features do you want next?** — 引发讨论
4. **[Show] LangGraph workflow visualization** — 展示 Mermaid 图
