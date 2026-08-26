# langchainrust

[![Rust](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/langchainrust.svg)](https://crates.io/crates/langchainrust)
[![Documentation](https://docs.rs/langchainrust/badge.svg)](https://docs.rs/langchainrust)
[![CI](https://github.com/atliliw/langchainrust/actions/workflows/ci.yml/badge.svg)](https://github.com/atliliw/langchainrust/actions/workflows/ci.yml)
[![Crates.io Downloads](https://img.shields.io/crates/d/langchainrust.svg)](https://crates.io/crates/langchainrust)

A LangChain-inspired Rust framework for building LLM applications.

**What it solves**: Build Agents, RAG, BM25 keyword search, Hybrid retrieval, LangGraph workflows, MCP tools, A2A agent-to-agent protocols, Guardrails, multi-agent Handoffs — all in pure Rust.

---

## Design Principles

The framework is engineered around a few hard rules that come out of its own design reviews. These are what make it feel different from hand-rolled LLM glue code:

| Principle | What it means in practice |
|-----------|---------------------------|
| **Explicit over silent** | No silent degradation. If an API promises X, it either delivers X or fails loudly. Batch-embedding alignment returns explicit errors instead of empty/shifted vectors; the keyword fallback for `EmbeddingMatcher` was removed; routing failures surface as errors, not swallowed `Err(_) => {}`. |
| **Consumed or deleted abstractions** | Every trait either has real implementors or is removed. `BaseChatMemory` is now implemented by all four memories, all retrievers implement `RetrieverTrait`, and `PairwiseJudge` plugs into the unified `Evaluator` pipeline — so users write against one abstraction, not five ad-hoc names. |
| **Structured output over text parsing** | Models that support `tool_calls` go through structured output (JSON schema / function calling). Regex-parsing model output is the last resort, not the default — it is the most common source of silent fragility. |
| **Production hardening first** | Tool execution has timeouts, LLM calls retry with exponential backoff, agent loops are capped (`max_iterations` clamped to `[1, 100]`), parallel actions are concurrency-limited, and `CancellationToken` propagates through every runnable. Eliminate the deterministic failure paths before adding features. |
| **Type system enforces safety** | Security properties live in types, not comments. Guardrails split `InputGuardrailResult` / `OutputGuardrailResult` so "Modify only applies to output" is enforced at compile time. |
| **Composition first** | Everything can be `pipe`d. Since v0.16.0, prompts, memory, native providers, parsers and RAG are all `Runnable` — `prompt.pipe(llm).pipe(parser)` compiles and runs. |
| **Honest implementations** | No fake backends. Empty-shell sandboxes (Wasm/E2B) were deleted; unsupported operations (e.g. Pinecone fetch-by-ID) return explicit `StorageError`s instead of pretending to work. |

---

## Core Features

### LLM & Providers

| Component | Description |
|-----------|-------------|
| **Unified LLM access** | 11 providers behind one `BaseChatModel` trait: OpenAI, Ollama, Anthropic Claude, Gemini, Azure, Cohere, DeepSeek, Qwen, Moonshot, Zhipu, Mistral. `LLMClient::from_env()` auto-detects any of the 11 from environment variables. |
| **OpenAI-compatible thin wrappers** | DeepSeek / Qwen / Moonshot / Zhipu / Mistral reuse the OpenAI request path; each keeps its own error variant (`ProviderError::DeepSeek`, etc.) so you can tell which vendor failed. New vendors are cheap to add. |
| **Chat & Streaming** | `chat()` (one full reply) and `stream_chat()` (first token in ~1s). Streaming chunks carry token usage (`StreamChunk`), so budget gates get real usage on the streaming path (v0.18). `config.streaming = true` makes `chat()` stream internally then aggregate. |
| **Function Calling** | `bind_tools()` + `result.tool_calls`, the native path for tool-capable models. |
| **Multimodal Vision** | `Message::human_with_image` / `human_with_audio` / `human_with_file` via schema `ImageContent` / `AudioContent` / `FileContent`. |
| **Thinking models** | Reasoning is kept in `LLMResult.thinking_content` and never leaked into `content` (DeepSeek-R1, GLM-5.2, Claude Extended Thinking). |
| **OpenAI Assistants API** | Stateful assistants with `requires_action` tool dispatch. |
| **OpenAI Responses API** | `web_search` / `file_search` / code tooling. |
| **Anthropic Extended Thinking** | `with_thinking` for Claude reasoning. |
| **Model Routing** | `RouterLLM` with 5 strategies — Fallback / RoundRobin / LeastLatency / LowestCost / InputDirected — remaining models always act as fallback. |
| **Batch API** | `BatchClient` for OpenAI / Anthropic batch inference (~50% cost reduction). |
| **LLM Cache** | `LLMCache` with TTL + true LRU eviction (hits refresh recency). |
| **Structured Output** | `with_structured_output` + `StructuredOutputExt` trait, `JsonOutputParser` fallback, and streaming structured output via `PartialJsonParser`. |
| **Token Counter** | `TiktokenCounter` (precise) / `CharRatioCounter` (Chinese-friendly estimate) + `TokenTrackingLLM` usage stats + `ModelPricing` cost estimation. |

### Embeddings

| Component | Description |
|-----------|-------------|
| **Unified `Embeddings` trait** | `embed_query` / `embed_documents` / `dimension` / `model_name`, with empty-input and batch-alignment checks enforced in the trait default path. |
| **Providers** | OpenAI (ada-002 / 3-small / 3-large), DeepSeek, Qwen, Cohere (embed-v3.0, 4 input types), FastEmbed (local ONNX), Mock (deterministic, for tests), BagOfWords (local, always available). |
| **Reliability** | Exponential-backoff retries (429/5xx, max 3, 4xx not retried), concurrent batching (OpenAI: 2048 docs/batch, concurrency 8), and unified normalization so downstream similarity is provider-independent. |
| **Local ONNX** | `LocalEmbeddings` via the `local-embeddings` feature (ort). |

### Composition: Chains & LCEL

| Component | Description |
|-----------|-------------|
| **LCEL** | `Runnable` with four base actions — `invoke` / `batch` / `stream` / `transform`. Operators: `pipe`, `RunnableLambda`, `RunnablePassthrough`, `RunnableParallel`, `RunnableBranch`, `RunnableBinding`, `RunnableWithFallbacks`, `RunnableAssign`, `with_retry`, `RunnableSequence`. Type-erased with `PhantomData` — dynamic composition with compiler-checked type matches. |
| **Unified composition (v0.15)** | Prompts, memory, native providers, parsers and RAG are all `Runnable`: `prompt.pipe(llm).pipe(StrOutputParser)` — no glue code. `RunnableWithMessageHistory` wraps "LLM + memory" as one runnable (auto read history → invoke → write back). `RagRunnable` makes retrieval-augmented generation one link of a chain. Native `OpenAIChat`/`QwenChat`/`DeepSeekChat` errors are unified into `LcelError`. |
| **Chains** | `BaseChain` with 9 implementations: `LLMChain`, `ConversationChain`, `SequentialChain`, `RouterChain`, `LLMRouterChain`, `RetrievalQA`, `ConversationRetrievalChain`, plus the 4 document chains — `Stuff` / `MapReduce` / `Refine` / `MapRerank`. Chain streaming per token, `ChainRunnable` bridges chains into LCEL. |
| **Prompts** | `PromptTemplate` (parsed once, cached segments), `ChatPromptTemplate` (Runnable, outputs `Vec<Message>`), `FewShotPromptTemplate` + `ExampleSelector`s (`LengthBasedExampleSelector`). `{{`/`}}` escapes, Chinese variable names, missing variables error loudly. |
| **Output Parsers** | `StrOutputParser`, `JsonOutputParser`, `CommaSeparatedListOutputParser`, `StructuredOutputParser`, `TypedOutputParser<T>` — all tolerant of dirty model output (markdown fences, trailing commas, trailing junk). |
| **Retrieval & Sessions in LCEL (v0.17)** | `RetrieverRunnable` wraps any retriever as `Runnable<String, Vec<Document>>`; `SessionManagerRunnable` wraps persistent sessions as `Runnable<(session_id, message), reply>` — both compose with `pipe` into a chain. |
| **Cancellation** | `CancellationToken` threads through `RunnableConfig` into every execution. |

### Agents & Multi-Agent

| Component | Description |
|-----------|-------------|
| **BaseAgent / AgentExecutor** | The "translator / butler" split: `BaseAgent` turns model output into a decision (Action / Actions / Finish), `AgentExecutor` is the one real loop — with tool timeouts, LLM retries, concurrency semaphore, and `max_iterations` clamped to `[1, 100]`. |
| **FunctionCallingAgent** | Recommended path — reads native `tool_calls` (requires model support). |
| **ReActAgent** | Text-regex thought/action loop, fallback for models without tool-calling. |
| **Plan-Execute** | Planner → per-step executor → replan on failure; the executor factory is configurable (no longer hardcoded to function calling). |
| **DeepResearch** | Multi-round research agent with sub-topic decomposition, parallel search, dedup, citation reporting. |
| **RAG Agents** | `CorrectiveRAGAgent` (self-correcting grade/rewrite/detect), `AdaptiveRAG` (LLM-routed retrieval), as standalone graphs. |
| **Handoffs** | Multi-agent handoff with `max_handoff_depth` (default 10) to stop A↔B ping-pong. |
| **Orchestrators** | `FanOutFanIn` (parallel fan-out + aggregate), `SequentialPipeline` (serial), `OrchestratorRunnable` for LCEL integration. |
| **Agent Hooks** | Approval (`on_before_tool_call` allow/reject/skip), `PromptInjectionHook`, `TokenBudgetHook`, `ContentFilterHook`, logging. |
| **Agent Gates (v0.16)** | Async human-approval gate — `.with_approval()` (Allow / Deny / Modify; Deny feeds the reason back as an observation, Modify rewrites the arguments). Budget gate — `.with_budget()` with hard caps on tool calls / tokens / wall-clock duration / iterations, exceeding returns `AgentError::BudgetExceeded`. Both default off. |
| **Cross-process resume (v0.18)** | `FileResumeStore` persists the pending human-approval / budget-gate state to disk (atomic write); a restarted executor loads the pending point and re-enters approval instead of restarting the agent loop. |
| **Streaming** | Token-level streaming via `StreamingFunctionCallingAgent` + `AgentStreamEvent`; tool-level events via `AgentExecutor::stream`. |
| **Tool Policies** | `ToolPolicy` / `ToolRisk` risk classification for tool access control. |

### Retrieval & RAG

| Component | Description |
|-----------|-------------|
| **Unified `RetrieverTrait`** | All retrievers implement it: `SimilarityRetriever`, `BM25Retriever` / `ChunkedBM25Retriever`, `UnifiedHybridIndex`, `ParentDocumentRetriever` — so any retrieval strategy plugs into the RAG pipeline. |
| **RAGPipeline** | `RAGPipelineBuilder` (llm + embeddings + vector store + retriever) → `index_documents` / `query` / `query_with_sources` (citation tracing). |
| **Document Loaders** | Text / JSON / Markdown / PDF / CSV / HTML + WebScraper / Sitemap / Docx. |
| **Splitting** | `RecursiveCharacterSplitter` (paragraph → line → sentence → char), `SemanticSplitter` (async semantic chunking). |
| **BM25** | Keyword search with Chinese/English tokenization, `ChunkedBM25Retriever` parent-child structure, AutoMerging. |
| **Hybrid** | `UnifiedHybridIndex` — BM25 + vector with RRF fusion, configurable `min_score`. |
| **Query Transformations** | `MultiQueryRetriever` (decompose into multiple queries), `HyDERetriever` (hypothetical document), `RerankingExecutor` + `KeywordReranker` / `BM25Reranker`. |
| **SelfQueryRetriever (v0.18)** | LLM splits a natural-language query into `{query, filter}` via structured call, with an `allowed_attributes` whitelist; retrieves through `similarity_search_with_filter`. Composes in LCEL as a `RetrieverRunnable`. |
| **GraphRAG** | Knowledge-graph RAG with Global / Local / Hybrid modes, entity extraction, community detection. |
| **Advanced RAG** | `CorrectiveRAG` (self-correcting), `AdaptiveRAG` (adaptive retrieval + structured routing decisions). |

### Memory & Sessions

| Component | Description |
|-----------|-------------|
| **Four memories** | `ConversationBufferMemory` (full), `ConversationBufferWindowMemory` (last k turns), `ConversationSummaryMemory` (LLM summary), `ConversationSummaryBufferMemory` (summary + recent raw). All implement `BaseChatMemory`. |
| **Semantic memory** | `VectorStoreRetrieverMemory` — retrieval by similarity, not recency. |
| **Context window** | `ContextWindow` with `Truncate` / `Summarize` strategies and pluggable `TokenCounter`; System messages are always preserved. |
| **Persistence** | `MongoPersistentMemory` (feature-gated) — generic over `BaseChatModel`, optimistic-lock concurrent writes, session-resume summary re-injection. |
| **Sessions** | `SessionManager` / `SessionStore` — multi-turn lifecycle (Active → Archived → Deleted), pluggable storage (`MemorySessionStore` default), `with_memory` bridge to memories. |

### Protocols: MCP & A2A

| Component | Description |
|-----------|-------------|
| **MCP** | Model Context Protocol client + server over **Stdio and SSE**. All 6 MCP primitives — **Resources / Prompts / Completion / Elicitation / Roots / Sampling** — implemented on both client and server sides. |
| **MCP resilience** | Auto-reconnect (subprocess crash → exponential backoff respawn + re-handshake + tool refresh), SSE heartbeat read-timeout (30s), tool-list cache with TTL + `list_changed` invalidation, `watch`-channel POST-URL re-discovery. |
| **MCP interop** | SSE client accepts both direct-response servers (POST returns JSON) and **202 + SSE-push** servers (POST returns `202 Accepted`, the JSON-RPC response arrives later over SSE and is correlated by request `id`). |
| **MCP tool adapter** | `MCPToolAdapter` implements `BaseTool` — MCP tools mix seamlessly with local tools in any agent; structured errors keep `{code, data}`; multi-type content (image/resource) preserved. |
| **MCP at scale** | Connection management, tool namespaces + conflict policy, static+dynamic tool discovery, per-tool timeout + progress, health checks + circuit breaker, per-server sandbox, sampling recursion guard, **MCP Gateway** (registry / pool / rate-limit / audit), multi-tenant isolation, protocol version negotiation. |
| **A2A** | Agent-to-Agent protocol: `AgentCard` discovery (`/.well-known/agent-card.json`), `A2ATask` / `TaskStatus` lifecycle (submitted → working → completed/failed), `send` / `get` / `cancel`, JSON-RPC over HTTP. `A2AServer` (task persistence) exposes a `BaseChain` as a network agent; `A2AClient` with bearer-token auth. |
| **A2A vs MCP** | A2A orchestrates **agent ↔ agent**; MCP lets an **agent call tools**. They compose: A2A between agents, MCP below them. |

### Quality: Guardrails, Evaluation, Observability

| Component | Description |
|-----------|-------------|
| **Guardrails** | Input/output safety rails around any agent or chain. `InputGuardrailResult` (Pass/Block) and `OutputGuardrailResult` (Pass/Block/Modify) are type-separated — Modify is compile-time impossible on input. `Guardable` trait lets you wrap any `BaseChain`. |
| **Built-in guardrails** | `SensitiveInfoGuardrail` (keywords + OpenAI-key regex + email + credit-card with Luhn check), `ForbiddenWordsGuardrail`, `MaxLengthGuardrail`. |
| **Streaming guardrails** | Two-phase: incremental keyword check (24-char sliding window) + full-output re-check. |
| **Audit** | `AuditSink` trait + `FileAuditSink` (JSON Lines) for violation persistence; LLM-sensitive judge for context-aware decisions. |
| **GuardedAgent** | Wrap an executor/chain → validate input → run → validate output; a blocked input never touches the network. |
| **Evaluation** | 10+ evaluators: `ExactMatch`, `ContainsKeyword`, `RegexMatch`, `LengthCheck`, `Bleu`, `StringDistance`, `EmbeddingSimilarity`, `LLMAsJudge`, `PairwiseJudge`, `Faithfulness`. `EvalRunner` batches all examples × all evaluators into a `Report`. |
| **LLM judge** | `StructuredJudge` shared with guardrails — prefers structured output, tolerant score parsing. |
| **Callbacks** | `CallbackHandler` (3 lifecycle methods minimum) + `CallbackManager` dispatcher. Built-in: `StdOutHandler`, `FileCallbackHandler`, `LangSmithHandler`, `OtelHandler`. |
| **Tracing** | `Tracer` + `SpanGuard` (RAII), `InMemory` / `Console` / `OTel` backends, parent-child span tree, GenAI Semantic Conventions. |

### Tools

| Component | Description |
|-----------|-------------|
| **Built-in tools** | `Calculator`, `SimpleMathTool`, `DateTimeTool`, `URLFetchTool`, `WikipediaTool`, `DuckDuckGoSearchTool`, `PythonREPLTool`. |
| **`#[tool]` macro** | Define a tool from a plain function — auto-converted to `BaseTool`; `StructuredTool` gives typed in/out with automatic JSON. |
| **Sandbox** | `SandboxTool` + `LocalSandbox` (subprocess + timeout). Tool-code execution is isolated — the Python blacklist is documented as *noise filtering, not a security boundary*. |
| **Extended tools** | `HTTPTool`, `FileTool` (sandboxed), `SQLTool` (read-only, `sqlite-storage` feature), `ComputerUseTool` (screen interaction). |
| **Security** | SSRF protection (`is_private_ip`) on URL/HTTP tools, path sandboxing, risk classification via `ToolPolicy`. |

### Vector Stores

| Component | Description |
|-----------|-------------|
| **Unified `VectorStore` trait** | `add_documents`, `similarity_search`, `similarity_search_with_min_score`, `similarity_search_with_filter` (`MetadataFilter` with Eq/Ne/Gt/Gte/Lt/Lte/In/Nin + And/Or, v0.18), `similarity_search_text` (auto-embeds if the store owns an embedder, else explicit error), `embed_query`, `get_document`, `delete_document`, `count`, `clear`. |
| **Backends** | InMemory, FileVectorStore (atomic write, fixed dim), ChunkedVectorStore (parent-child source retrieval), Qdrant, ChromaDB, LanceDB, Neo4j, Pinecone, Redis, MongoDB, SQLite, PGVector (typed `PGVectorStore` via the `pgvector-storage` feature). |
| **Honest errors** | `VectorStoreError` distinguishes `DocumentNotFound` / `EmbeddingError` / `StorageError` / `ConnectionError`; missing features fail loudly instead of silently degrading (e.g. Qdrant without the feature → `ConnectionError`, not in-memory fallback). |

---

## Architecture

langchainrust is a **22-crate workspace** with a single facade crate `langchainrust` (in `crates/lc`) that re-exports the public API. Layers depend downward — `lc-shared` / `lc-schema` sit at the bottom and are depended on by everyone, which is exactly how the circular-dependency problem is solved.

```
                      ┌─────────────────────────────────────┐
                      │   langchainrust  (facade, crates/lc) │
                      └───────────────┬─────────────────────┘
                                      │
        ┌─────────────┬───────────────┼───────────────┬─────────────┐
   Protocol Layer   Quality Layer   Intelligence    Composition   Providers
   ┌──────────┐   ┌────────────┐  ┌──────────────┐ ┌────────────┐ ┌──────────┐
   │ lc-mcp   │   │ lc-guardrails │  │ lc-agents   │ │ lc-chains  │ │ lc-providers │
   │ lc-a2a   │   │ lc-evaluation  │  │ lc-rag      │ │ lc-langgraph │ │ lc-embeddings │
   └────┬─────┘   │ lc-callbacks   │  │ lc-vector-stores │ └────┬───────┘ │ lc-prompts │
        │         └───────┬────────┘  └───────┬────────┘      │         │ lc-tools   │
        │                 │                    │               │         └─────┬──────┘
        └─────────────────┴────────────────────┴───────────────┴───────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                    Core & Foundation                      │
        │  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐   │
        │  │ lc-core  │  │ lc-schema│  │ lc-shared              │   │
        │  │ Runnable │  │ Message  │  │ Document / ToolCall /  │   │
        │  │ LCEL     │  │ types    │  │ TextSplitter           │   │
        │  └──────────┘  └──────────┘  └────────────────────────┘   │
        └──────────────────────────────────────────────────────────┘
```

| Crate | Role |
|-------|------|
| **lc-core** | Execution layer: `Runnable` / LCEL operators, `BaseChatModel`/`BaseLanguageModel`, `BaseTool`/`ToolRegistry`, output parsers, structured output, token counter, `LLMCache`, `RouterLLM`, `CancellationToken`, `StructuredJudge`, `BatchClient`, `cosine_similarity`. |
| **lc-schema** | `Message` (content + role + multimodal attachments), `MessageType`, `ImageContent`/`AudioContent`/`FileContent`. |
| **lc-shared** | Cross-crate foundation types: `Document`, `VectorDocument`, `SearchResult`, `ChunkDocument`, `ToolCall`/`FunctionCall`, `TextSplitter` — breaks the dependency cycle. |
| **lc-providers** | 11 LLM vendors behind `BaseChatModel`; `LLMClient` (auto-detect), `ProviderError` (per-vendor variants), `ChatModelWrapper` (error normalization for mixed routing). |
| **lc-prompts** | `PromptTemplate`, `ChatPromptTemplate` (Runnable), `FewShotPromptTemplate`, `ExampleSelector`s. |
| **lc-tools** | Built-in tool library + `#[tool]` proc macro (`lc-tools-derive`), sandbox. |
| **lc-embeddings** | `Embeddings` trait + 7 providers, retries, concurrency, normalization. |
| **lc-chains** | `BaseChain` + 9 chains, `ChainRunnable` bridge into LCEL. |
| **lc-langgraph** | `StateGraph`, conditional/FanOut/FanIn edges, `Reducer`s, `Checkpointer` (memory/file), `GraphPersistence`, `Subgraph`, dynamic injection. |
| **lc-agents** | ReAct / FunctionCalling / PlanExecute / CRAG / AdaptiveRAG / DeepResearch / Handoffs / Orchestrators / Hooks + human-approval gate (`ApprovalHandler`) / budget gate (`BudgetConfig`). |
| **lc-memory** | Buffer/Window/Summary/SummaryBuffer memories, `ContextWindow`, `MongoPersistentMemory`. |
| **lc-sessions** | `SessionManager` + `SessionStore` multi-turn lifecycle. |
| **lc-rag** | `RetrieverTrait` (Similarity/BM25/UnifiedHybrid), `RAGPipeline`, MultiQuery/HyDE/Reranking, GraphRAG. |
| **lc-vector-stores** | `VectorStore` trait + InMemory/File/Chunked/Qdrant/ChromaDB/LanceDB/Neo4j/Pinecone/Redis/Mongo/SQLite/PGVector backends. |
| **lc-mcp** | MCP client/server (Stdio+SSE), tool adapter, Gateway. |
| **lc-a2a** | A2A protocol server/client. |
| **lc-evaluation** | Rule evaluators + LLM judges, `EvalRunner` + `Report`. |
| **lc-guardrails** | Input/output guardrails, `Guardable`, streaming guardrails, audit sinks. |
| **lc-callbacks** | `CallbackHandler`/`CallbackManager` + StdOut/File/LangSmith/OTel + `Tracer`/`SpanGuard`. |
| **lc-testkit** | Record/replay test harness: `RecordingProvider` records real LLM exchanges to JSONL, `ReplayProvider` replays them offline with zero network — framework tests run without API keys. Phase 2 (v0.17): tool definition recording (`bind_tools`), out-of-order replay (`ReplayStrategy::{Fifo, ByToolName}`), agent-level offline replay, and chain scenarios transcribed from online tests. Phase 3 (v0.18): strict message-signature replay (`ReplayStrategy::Exact`). |

---

## Installation

```toml
[dependencies]
langchainrust = "0.18.0"
tokio = { version = "1.0", features = ["full"] }

# Optional features
langchainrust = { version = "0.18.0", features = ["mongodb-persistence"] }  # MongoDB storage
langchainrust = { version = "0.18.0", features = ["qdrant-integration"] }    # Qdrant vector DB
langchainrust = { version = "0.18.0", features = ["redis-storage"] }         # Redis storage
langchainrust = { version = "0.18.0", features = ["sqlite-storage"] }        # SQLite storage (+ SQLTool)
langchainrust = { version = "0.18.0", features = ["pgvector-storage"] }      # PGVector (requires user-configured sqlx/pgvector deps)
langchainrust = { version = "0.18.0", features = ["local-embeddings"] }      # Local ONNX embeddings (requires ort)
langchainrust = { version = "0.18.0", features = ["opentelemetry"] }         # OpenTelemetry tracing
langchainrust = { version = "0.18.0", features = ["fastembed"] }            # FastEmbed embeddings
langchainrust = { version = "0.18.0", features = ["vectorstore-memory"] }   # VectorStoreRetrieverMemory (semantic memory)
langchainrust = { version = "0.18.0", features = ["experimental"] }         # Experimental features
# PineconeStore / FileVectorStore require no feature flag, available by default
```

> **Note on MSRV**: Rust **1.82+** required.

---

## Quick Start

### 1. Basic chat

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")?,
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-4o-mini".to_string(),
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

### 2. Multi-provider — same interface, swap config

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

### 3. LCEL — compose everything into one chain

```rust
use langchainrust::{
    ChatPromptTemplate, Message, OpenAIChat, OpenAIConfig, RunnableExt, StrOutputParser,
};
use std::collections::HashMap;

let llm = OpenAIChat::new(OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY")?,
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-4o-mini".to_string(),
    ..Default::default()
});

// prompt.pipe(llm).pipe(parser) — everything is a Runnable (v0.15)
let prompt = ChatPromptTemplate::from_messages([
    Message::system("你是一个简洁的 Rust 助手。"),
    Message::human("{question}"),
]);
let chain = prompt.pipe(llm).pipe(StrOutputParser::new());

let mut vars = HashMap::new();
vars.insert("question".to_string(), "一句话说明什么是 Rust".to_string());
let answer = chain.invoke(vars, None).await?;
println!("{answer}");
```

For the full five-way composition — prompt + memory + LLM + parser + RAG in one program — see [`lcel_compose`](crates/lc/examples/lcel/lcel_compose.rs).

### 4. BM25 keyword search (no API key)

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

More examples in [中文使用指南](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE.md) or [Usage Guide](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE_EN.md).

---

## Examples

The `crates/lc/examples/` directory provides 39 runnable examples covering core functionality:

| Category | Examples | Requires API Key |
|----------|----------|-----------------|
| basic | chat / streaming / multi_provider / token_counter / quick_start / responses_api / batch_api / sandbox | Yes |
| agent | function_calling / multi_tool / assistants / handoffs / plan_execute / deep_research / extended_thinking | Yes |
| rag | bm25_search / document_loaders / file_vectorstore / semantic_splitter / adaptive_rag / corrective_rag / graph_rag | No |
| langgraph | basic_graph / conditional_edge | No |
| memory | buffer_memory / context_window / sessions / vectorstore_memory | No |
| chains | llm_chain / sequential_chain | Yes |
| lcel | lcel_pipe / lcel_compose | pipe: No / compose: Yes |
| evaluation | evaluation | No |
| guardrails | guardrails | No |
| mcp | mcp_server / mcp_sse_server / mcp_stdio_server | No |
| a2a | a2a_http_server | Yes |
| otel | otel_tracing | No |

Examples requiring API keys read from environment variables:

```bash
export OPENAI_API_KEY="your-key"
cargo run --example basic_chat
```

Examples without API keys (BM25 / LangGraph / Memory / Loader) can run directly — great for quick exploration.

---

## Production Considerations

Some hard-won guidance from the framework's design reviews:

- **Don't rely on silent keyword fallbacks.** `LocalEmbeddings` without the `local-embeddings` feature is a Bag-of-Words fallback — don't use it for real semantic retrieval. Enable the feature or use an API provider.
- **Embedding batch alignment errors are explicit.** Missing or shifted vectors now return `EmptyVectorInBatch` / `BatchMismatch` instead of empty slots — treat them as failures, don't skip them.
- **Chinese text: prefer Tiktoken-based token counting.** The `len/4` heuristic over-counts Chinese; `ContextWindow` uses `TiktokenCounter` when available.
- **Summary memory: old summaries survive LLM failure.** A failed summary leaves the previous summary and raw messages intact (`last_summary_error()` reports the failure); it never silently wipes history.
- **MCP / A2A auth is at the JSON-RPC layer.** Bearer-token auth returns HTTP 200 with `{"error":{"code":401,...}}` in the body — check the body, not just the status code. Agent-card discovery is intentionally public.
- **MCP connect waits on auto-reconnect.** Connecting to an unreachable MCP server can take ~30s (heartbeat + reconnect backoff) before returning a clean `connection_lost()` error — that's framework design, not a hang.
- **Pinecone intentionally lacks fetch-by-ID.** `get_document` / `get_embedding` / `clear` return explicit `StorageError`s; `count` uses `describe_index_stats`. Don't design around those ops with Pinecone.
- **SQLTool is read-only by default, use parameterized queries.** The Python blacklist is noise-filtering, not a security boundary — isolate real code execution with `LocalSandbox`.
- **Summary memory and sessions need no special wiring.** `SessionManager` accepts a `BaseMemory` via `with_memory` for persistent multi-turn apps.

---

## Documentation

| Docs | Content |
|------|---------|
| [中文使用指南](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE.md) | 所有组件的详细用法（中文） |
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
