# Twitter/X 推广内容

---

## 第 1 条：主推文（发这个）

🦀 langchainrust v0.7.1 is out!

A full-featured LLM framework in pure Rust:
✅ 8 providers (OpenAI, Anthropic, Gemini, Ollama, DeepSeek, Qwen, Moonshot, Zhipu)
✅ LangGraph workflows + MCP client/server
✅ GraphRAG / CorrectiveRAG / AdaptiveRAG
✅ BM25 + Hybrid retrieval (RRF fusion)
✅ 7 memory types + Guardrails
✅ RouterLLM (5 strategies + failover)

No Python. No GC pauses. Single binary.

🔗 https://github.com/atliliw/langchainrust
📖 https://docs.rs/langchainrust

#rustlang #LLM #AI #OpenSource #RAG

---

## 第 2 条：功能亮点 — RAG（隔 1-2 天发）

RAG in pure Rust? Yes. 🦀

langchainrust supports the full RAG spectrum:

1️⃣ Basic RAG — vector store + retrieval
2️⃣ CorrectiveRAG — self-correcting with hallucination detection
3️⃣ AdaptiveRAG — LLM-routed strategy selection
4️⃣ GraphRAG — knowledge graph + community detection

Plus: BM25 keyword search, Hybrid retrieval (BM25+Vector RRF fusion), HyDE, MultiQuery, Reranking

All in Rust. No Python needed.

🔗 https://github.com/atliliw/langchainrust

#rustlang #RAG #AI #LLM #GraphRAG

---

## 第 3 条：功能亮点 — LangGraph（隔 1-2 天发）

Build agent workflows with LangGraph in Rust 🕸️

✅ StateGraph with conditional edges
✅ Human-in-the-loop
✅ Subgraph composition
✅ Parallel execution
✅ Checkpointing (save & resume)
✅ Mermaid visualization

From simple chatbots to complex multi-step agents — all in pure Rust.

```rust
let mut graph = StateGraph::new();
graph.add_node("retrieve", retrieve_step);
graph.add_node("generate", generate_step);
graph.add_edge("retrieve", "generate");
```

🔗 https://github.com/atliliw/langchainrust

#rustlang #AI #Agent #LangGraph

---

## 第 4 条：功能亮点 — MCP（隔 1-2 天发）

MCP in Rust — both client AND server 🔌

Most frameworks only implement MCP client. langchainrust implements all 6 MCP primitives on both sides:

Client side:
- Connect to any MCP server (Stdio + SSE)
- MCPToolAdapter → use MCP tools as BaseTool

Server side:
- Expose your Rust tools to any MCP host
- Full resources/prompts/completion/elicitation/roots/sampling

Build MCP-native AI apps in Rust. 🦀

🔗 https://github.com/atliliw/langchainrust

#rustlang #MCP #AI #LLM #OpenSource

---

## 第 5 条：Ollama + 本地模型（隔 1-2 天发）

Run LLMs locally with Ollama + Rust. No Python. No API keys. 🖥️

langchainrust first-class Ollama support:
✅ Tool calling
✅ Vision (image input)
✅ Streaming responses
✅ LocalEmbeddings (no API call)
✅ BM25 keyword search (Chinese + English)
✅ 9 vector store backends (InMemory, SQLite, Qdrant...)
✅ Code sandbox (Local + E2B + WASM)

Single binary. Zero cloud dependency. Pure Rust.

```rust
let llm = OllamaChat::new("llama3.2");
```

🔗 https://github.com/atliliw/langchainrust

#rustlang #Ollama #LocalLLM #AI #Privacy

---

## 第 6 条：架构图展示（隔 2-3 天发）

The full stack of langchainrust 🏗️

```
┌─────────────────────────────────────┐
│           LLM Layer (8 providers)    │
├─────────────────────────────────────┤
│    Agent Layer (6 agent types)       │
├─────────────────────────────────────┤
│    MCP Layer (Client + Server)       │
├─────────────────────────────────────┤
│    Retrieval Layer (RAG + BM25)      │
├─────────────────────────────────────┤
│    Storage Layer (9 Vector DBs)      │
├─────────────────────────────────────┤
│    Utility Layer (Memory/Chains/...)  │
└─────────────────────────────────────┘
```

Everything you need to build LLM apps in Rust. 🦀

🔗 https://github.com/atliliw/langchainrust

#rustlang #AI #LLM #OpenSource

---

## 发推技巧

1. **配图** — 每条推配一张截图（架构图、代码截图、运行效果），带图的推文互动率高 3x
2. **时间** — 美国时间 9-11 AM EST 发，对应北京时间 10 PM-12 AM
3. **间隔** — 不要一天发 6 条，隔 1-2 天发一条
4. **互动** — 别人回复一定要回，Twitter 算法喜欢活跃互动
5. **@ 人** — 可以 @rustlang，但不要 @ 太多
6. **Thread** — 如果内容长，用 Thread 形式（回复自己的推），不要全塞一条里
