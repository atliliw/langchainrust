# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] - 2026-08-05

### Added
- **LCEL 补全 — RunnableWithFallbacks**: Fallback composition for LCEL pipelines. If the primary Runnable fails, each fallback is tried in order. `RunnableExt::with_fallbacks()` method for fluent API
- **LCEL 补全 — RunnableAssign**: Inject new key-value pairs into `HashMap<String, Value>` mid-pipeline. Critical for RAG pipelines where context is injected alongside the question
- **LCEL 补全 — RunnableParallel.assign()**: Convenience method that pipes parallel output through RunnableAssign
- **`#[tool]` procedural macro**: New `lc-tools-derive` crate that auto-generates `BaseTool` + `Tool` impl from a simple function annotated with `#[tool(description = "...")]` and `#[param(desc = "...")]`. Reduces ~50 lines of boilerplate to 3 lines
- **Document Chain streaming — StuffDocumentsChain**: Override `stream()` to emit LLM tokens as `StreamToken` stream
- **Document Chain streaming — RefineDocumentsChain**: Initial + intermediate refine steps use invoke, final refine step streams via `stream_chat`
- **Document Chain streaming — MapReduceDocumentsChain**: Map phase runs parallel invoke, reduce phase streams via `stream_chat`
- **Document Chain streaming — SequentialChain**: All preceding chains use invoke, last chain's `stream()` is forwarded
- **Document Chain streaming — RouterChain**: After routing, delegates to selected chain's `stream()` method
- **Document Chain streaming — LLMRouterChain**: LLM routing call completes first, then delegates to selected chain's `stream()`

### Changed
- `lc-tools` now depends on `lc-tools-derive` and re-exports the `#[tool]` macro
- Workspace root `Cargo.toml` now includes `lc-tools-derive` as a member
- `CRATE_DEPENDENCIES.md` updated with `lc-tools-derive` in the dependency graph and publish order

### Deprecated
- Hand-written `BaseTool` + `Tool` implementations are still supported but `#[tool]` macro is preferred for new tools

## [0.9.0] - 2026-08-05

### Added
- **LCEL (LangChain Expression Language)**: Pipe composition for `Runnable` components, inspired by Python LangChain's `prompt | llm | parser` syntax
  - `RunnableExt::pipe()`: Chain any `Runnable<I, O>` into a pipeline via `.pipe(other)`
  - `RunnableSequence<I, O>`: Ordered pipeline of type-erased steps, supports `invoke` / `batch` / `stream` / `transform`
  - `RunnableLambda<I, O>`: Wrap sync/async closures as `Runnable` steps (`new_sync` / `new_sync_fallible` / `new_async`)
  - `RunnablePassthrough<I>`: Identity pass-through with true streaming (passes input stream through without buffering)
  - `RunnableParallel<I>`: Fan-out/fan-in — run multiple steps concurrently, collect results into `HashMap<String, Value>`
  - `RunnableBranch<I, O>`: Conditional routing — evaluate conditions in order, first match wins
  - `RunnableBinding<I, O>`: Bind config/kwargs to a `Runnable` for pre-configured invocation
  - `LcelError`: Unified error type for LCEL pipelines (9 variants, all String-based to avoid circular deps)
  - `LcelStreamEvent`: Fine-grained pipeline events (OnLlmStart/Stream/End, OnToolEnd, OnChainEnd)
  - `RunnableAny` trait + `RunnableAnyWrapper`: Type erasure for heterogeneous pipeline composition
  - `into_runnable_any()`: Helper to wrap any `Runnable` into `Box<dyn RunnableAny>`
- **`Runnable::transform()` method**: Core LCEL streaming primitive — takes input `Stream`, returns output `Stream`. Default implementation buffers input and invokes on last item; `RunnablePassthrough` overrides for true pass-through streaming
- **True streaming for `CompiledGraph`**: `stream()` now returns `Pin<Box<dyn Stream>>` (real async stream via tokio::sync::mpsc) instead of collecting all events into `Vec`. Added `stream_collected()` for backward compatibility
- **`CompiledGraph` Clone**: Added `#[derive(Clone)]` (all fields use Arc, so Clone is cheap) — required for true streaming (spawned tokio task needs owned graph)
- **`AgentExecutor::stream()`**: New method returning `Pin<Box<dyn Stream<Item = Result<AgentStreamEvent, AgentError>>>>` with ToolStart/ToolEnd events
- **`AgentStreamEvent` new variants**: `ToolStart { name, input }` and `ToolEnd { name, output }` for fine-grained tool execution observability
- **Adapter pattern**: `ChainRunnable`, `AgentRunnable`, `RagRunnable` — bridge existing types to `Runnable` trait for LCEL pipeline participation
- **Error conversions**: `From<CoreError> for LcelError`, `From<ProviderError> for LcelError` — seamless `?` operator across module boundaries
- **`lcel_pipe` example**: Demonstrates pipe, three-step pipeline, passthrough, async lambda, branch, parallel, batch, config binding

### Changed
- **Workspace version**: All 20 crates bumped from 0.8.0 to 0.9.0
- **`lc-core` dependencies**: Added `futures-util`, `tokio`, `tokio-stream`, `serde_json`, `uuid`
- **`lc-langgraph` dependencies**: Added `tokio-stream`
- **Facade crate re-exports**: All LCEL types (`RunnableSequence`, `RunnableLambda`, `RunnablePassthrough`, `RunnableParallel`, `RunnableBranch`, `RunnableBinding`, `LcelError`, `LcelStreamEvent`, `into_runnable_any`, etc.) re-exported from `langchainrust`

## [0.8.0] - 2026-08-04

### Changed
- **Workspace restructure**: Migrated from single-crate to multi-crate workspace architecture
  - `langchainrust` → facade crate (`lc`) re-exporting all sub-crates
  - `lc-core`: Runnable, BaseTool, BaseLanguageModel, output parsers, structured output, token counter, batch, cache
  - `lc-schema`: Message types (Human, AI, System, Tool)
  - `lc-shared`: Shared utilities
  - `lc-providers`: OpenAI / Ollama / DeepSeek / Moonshot / Zhipu / Qwen / Anthropic / Gemini
  - `lc-agents`: ReActAgent / FunctionCallingAgent / PlanExecute / Handoffs / CRAG / AdaptiveRAG / DeepResearch
  - `lc-chains`: LLMChain / SequentialChain / ConversationChain / RouterChain / RetrievalQA / Document chains
  - `lc-memory`: Buffer / Window / Summary / SummaryBuffer / Persistent / ContextWindow
  - `lc-embeddings`: OpenAI / DeepSeek / Qwen / Local / Mock
  - `lc-vector-stores`: InMemory / Qdrant / MongoDB / ChromaDB / Redis / SQLite / PGVector / Pinecone / File
  - `lc-retrieval`: RAG / BM25 / Hybrid / HyDE / MultiQuery / Reranking / GraphRAG / Loaders
  - `lc-prompts`: PromptTemplate / ChatPromptTemplate / FewShot
  - `lc-callbacks`: StdOut / LangSmith / File / OTel + Tracing
  - `lc-tools`: Calculator / DateTime / Math / URLFetch / Wikipedia / PythonREPL / HTTP / File / SQL / ComputerUse / Sandbox
  - `lc-evaluation`: 10 evaluators
  - `lc-guardrails`: Input/Output guardrails
  - `lc-sessions`: Session management
  - `lc-mcp`: MCP Client + Server
  - `lc-a2a`: A2A protocol
  - `lc-langgraph`: StateGraph / CompiledGraph / Checkpointer / Subgraph / Parallel

## [0.7.2] - 2026-08-03

### Fixed
- **docs.rs build**: pgvector.rs compilation with `--all-features`
- **LocalEmbeddings thread safety**: RefCell → RwLock for Send + Sync
- **Qdrant feature gate**: Parameter naming fix

### Fixed
- **docs.rs build failure**: pgvector.rs `use pgvector`/`use sqlx` imports caused compilation errors when building with `--all-features` without actual pgvector/sqlx crates. Refactored pgvector.rs to only expose pure helper functions (`validate_table_name`, `build_table_sql`); the full `PGVectorStore` implementation requires user-configured dependencies
- **LocalEmbeddings thread safety**: Replaced `RefCell<Session>` with `RwLock<Session>` in `LocalEmbeddings` to satisfy `Send + Sync` trait bounds required by the `Embeddings` trait
- **Qdrant provider feature gate**: Fixed `provider.rs` parameter naming (`_url`/`_collection` → `url`/`collection`) that caused E0423/E0425 errors when `qdrant-integration` feature was enabled

## [0.7.0] - 2026-08-01

### Changed
- **Version upgrade**: 0.6.0 → 0.7.0
- **Test fixes**: Updated MultiQuery, Anthropic Thinking, and VectorStore tests to support the new version and fix assertion logic

### Added
- **Documentation navigation**: Added page navigation menu to features.html / features_en.html

## [0.6.0] - 2026-07-30

### Added
- **CorrectiveRAG (Self-Correcting RAG)**: `CorrectiveRAGAgent` LangGraph state machine — retrieve → grade → [rewrite+web | keep] → generate, with hallucination detection
- **AdaptiveRAG (Adaptive Retrieval)**: LLM routing to `NoRetrieval` / `SingleSearch` / `MultiQuery`, reusing existing Retrievers
- **GraphRAG (Knowledge Graph RAG)**: LLM extracts entities + relationships → build graph → Label Propagation community detection + summarization → Global/Local/Hybrid queries
- **Deep Research Agent**: Multi-round search (query generation → parallel retrieval → deduplication → sub-topic aggregation → comprehensive report + citations)
- **RouterLLM**: 5 routing strategies (Fallback / RoundRobin / LeastLatency / LowestCost / InputDirected) + failover
- **MCP Full Protocol**: Added resources / prompts / completion / elicitation / roots / sampling primitives, with corresponding Client/Server handlers
- **Code Interpreter Sandbox**: `CodeSandbox` trait + `LocalSandbox` (subprocess) + E2B/WASM backends (feature gate)
- **OpenAI Responses API**: `ResponsesModel` implementing `BaseChatModel`, using `/v1/responses`, with built-in web_search/file_search/code_interpreter/computer_use
- **Anthropic Extended Thinking**: `ThinkingConfig` + `with_thinking(budget_tokens)`, thinking block parsing, streaming thinking
- **Streaming Structured Output**: `PartialJsonParser` incremental JSON parsing + `stream_structured_output<T>` streaming partial structures
- **Batch API**: `BatchClient` submit/poll/results/cancel, OpenAI + Anthropic batch endpoints
- **Agent Observability / Tracing**: `Tracer` + `SpanGuard` (RAII) + `TracingBackend` trait + InMemory/Console/OTel backends

### Changed
- **Large-scale code refactoring**: Split multiple large files into modular subdirectories
  - `tracing.rs` → `tracing/` (backend, span, tracer, tests)
  - `document_chains.rs` → `document_chains/` (stuff, refine, map_reduce, map_rerank, tests)
  - `batch.rs` → `batch/` (client, types, tests)
  - `structured_output.rs` → `structured_output/` (extract, parser, streaming, tests)
  - `compiled.rs` → `compiled/` (graph, invoke, parallel, stream, types, validate, visualize, tests)
  - `anthropic.rs` → `anthropic/` (chat, config, error, impls, types, tests)
  - `responses.rs` → `responses/` (model, types, tests)
  - `context_window.rs` → `context_window/` (manager, trimmer, tests)
  - `document_store.rs` → `document_store/` (store, chunked, types, tests)
  - `computer.rs` → `computer/` (actions, screen)
- **thiserror 1.0 → 2.0** (breaking): Full migration of error types
- **Unified error module**: Added `src/error.rs`, centralizing all error type definitions
- **Document deduplication optimization**: AdaptiveRAG document deduplication changed from first-80-characters to full-content hash
- **Community summarization refactoring**: GraphRAG community summaries now use a unified prompt template system
- **Provider environment variable error handling**: Added error handling to `from_env()` for Ollama and OpenAI
- **Streaming error propagation**: Fixed network error propagation mechanism in streaming
- **Gemini enhancements**: Extended Gemini provider functionality
- **LLMResult**: Added `thinking_content: Option<String>` field (`#[serde(default)]` for backward compatibility)
- **CallbackHandler**: Added `on_llm_thinking` default method
- **AnthropicChat**: Added `with_thinking(budget_tokens)` builder
- **Cargo.toml**: Added feature gates `sandbox-e2b`, `sandbox-wasm`

### Fixed
- **Security**: PythonREPLTool added dangerous import checks; HTTPTool SSRF switched to async DNS; URLFetchTool added private IP filtering; SQLTool blocks semicolon/comment/subquery bypasses; Gemini API key moved from URL to header
- **Panic fixes**: `choices[0]` changed to `.first().ok_or()`; `from_env()` returns Result instead of expect panic; Regex changed to LazyLock; Mutex poison changed to `into_inner()`
- **SSE streaming**: Ollama/Anthropic/Gemini added cross-chunk buffer, no longer losing tokens
- **Multi-round Function Calling**: Anthropic system messages placed in top-level `system` field; Ollama AI messages include tool_calls; Gemini tool_result uses functionResponse
- **Concurrency safety**: langgraph/compiled switched to tokio::sync::RwLock; sessions/memory_store switched to tokio::sync::Mutex; HandoffManager merged into single Mutex; MCP Transport added request-level mutex
- **Data correctness**: parent_id separator changed from `_` to `::`; error propagation replaces `.ok()` silent swallowing; UTF-8 splitting at character boundaries; negative-score document filtering; RRF document IDs use content hash
- **Runnable::stream() pseudo-streaming**: OpenAI/Anthropic/Ollama changed to emit LLMResult per token
- **Batch API**: Anthropic message mapping corrected — system messages extracted to top-level `system` field
- **RouterLLM**: Mutex poison changed to into_inner; RoundRobin index zero-check; stream_chat updates latency statistics
- **JSON fixes**: repair_partial_json correctly tracks braces inside strings; handles escaped quotes; UTF-8 character boundary checks
- **Other**: cosine_similarity uses epsilon floating-point zero check; different-length vectors return error; cache expired entry cleanup; structured_output parse supports markdown wrapping; score range validation 0-1

## [0.5.2] - 2026-07-28

### Fixed
- **GraphRAG community summaries using entity IDs instead of names** (`retrieval::graph_rag::community`): `summarize_community` directly concatenated `r.source`/`r.target` (entity IDs in `e_xxx` format), causing LLM to receive meaningless IDs and severely degrading community summary quality, which dragged down Global/Hybrid queries. Changed to look up entity names via `store.get_entity()`, consistent with `format_relation` logic in `query.rs`
- **Deep Research comprehensive report embedded as JSON string causing escape failures** (`agents::deep_research::synthesizer`): Requiring LLM to output the full markdown report as a JSON string field `"report"` meant markdown containing `\n`, `"`, `\` needed JSON escaping, and LLM escape error rates were high, causing `serde_json` parse failures that broke the entire synthesis step. Changed to delimiter format `<<<REPORT>>>...<<<END_REPORT>>><<<GAPS>>>[...]<<<END_GAPS>>>`, where the report portion takes raw text without escaping; old JSON format retained as fallback for compatibility
- **document_store panic inside tokio runtime** (`vector_stores::document_store`): `InMemoryDocumentStore` and `InMemoryChunkedDocumentStore` internally used `tokio::sync::RwLock`, but `_blocking` methods called `blocking_read()`/`blocking_write()`, which triggered `Cannot block the current thread from within a runtime` panic in `#[tokio::test]` async contexts. Changed to `std::sync::RwLock`, with all lock operations using `.read().unwrap()`/`.write().unwrap()`, working in both sync and async contexts; lock hold times are short (no cross-await locking), so no deadlock risk
- **CRAG scoring threshold stuck in subjective range** (`agents::crag`): Default `grade_threshold = 0.5` was exactly in the range where LLM scoring is most unstable, and `parse_grade_response` ambiguous responses also defaulted to 0.5, which exactly equaled the threshold, making corrective action nearly random. Default threshold changed to 0.6, ambiguous response score changed to 0.4, so they no longer overlap; `GradeResult` gained `is_ambiguous` field to mark whether the score came from ambiguous parsing
- **CRAG hallucination detection self-check bias** (`agents::crag`): Generation and hallucination detection used the same LLM, and the model tends to agree with its own output. Added `with_grader_llm()` builder to inject an independent LLM for hallucination detection (falls back to main LLM when not set, backward compatible); hallucination detection prompt adds adversarial perspective ("Be skeptical"); hallucination detection LLM call failure degrades to returning `grounded: false` instead of aborting the entire flow

### Changed
- **Sandbox feature declarations added** (`Cargo.toml`): Code referencing `#[cfg(feature = "sandbox-e2b")]` / `#[cfg(feature = "sandbox-wasm")]` had features not previously declared in `[features]`, causing clippy `unexpected cfg condition value` warnings. Added `sandbox-e2b` / `sandbox-wasm` feature declarations
- **clippy zero warnings**: Fixed `or_insert_with(Vec::new)` -> `or_default()`, `#[allow(deprecated)]` on internal `OpenAIConfig::from_env()` calls, and other warnings

## [0.5.0] - 2026-07-23

### Added
- **Model Routing + Fallback + Load Balancing (`core::router_llm`)** (#2): `RouterLLM` implements `BaseChatModel`, selecting models by strategy across a heterogeneous provider pool and falling back on failure
  - `RoutingStrategy`: `Fallback` (primary-first) / `RoundRobin` / `LeastLatency` (EMA latency statistics) / `LowestCost` / `InputDirected` (closure selects model by input)
  - `RouterError` unified error, via internal `ModelAdapter` converging heterogeneous provider errors into a single type
  - `with_fallbacks(primary, fallbacks)` / `with_model` / `with_named_model` / `with_cost` builder
- **Agentic / Corrective RAG (`agents::crag`)** (#1): `CorrectiveRAGAgent` LangGraph state machine: retrieve -> grade -> [rewrite+web|keep] -> generate, with hallucination detection
- **MCP Full Protocol (`mcp`)** (#3): Added resources / prompts / completion / elicitation / roots / sampling primitives, with corresponding Client/Server handlers, 39 unit tests
- **Code Interpreter Sandbox (`tools::sandbox`)** (#4): `CodeSandbox` trait + `SandboxTool<BaseTool>` + `LocalSandbox` (subprocess) + E2B/WASM backends (feature gate)
- **OpenAI Responses API (`language_models::openai::responses`)** (#5): `ResponsesModel` implements `BaseChatModel`, using `/v1/responses`, with built-in web_search/file_search/code_interpreter/computer_use, streaming event parsing
- **GraphRAG Knowledge Graph RAG (`retrieval::graph_rag`)** (#6): LLM extracts entities + relationships -> build graph -> Label Propagation community detection + summarization -> Global/Local/Hybrid queries, no petgraph dependency
- **Anthropic Extended Thinking (`language_models::providers::anthropic`)** (#7): `ThinkingConfig` + `with_thinking(budget_tokens)`, thinking block parsing, `on_llm_thinking` callback, `LLMResult.thinking_content`, streaming thinking
- **Deep Research Agent (`agents::deep_research`)** (#8): Multi-round search (query generation -> parallel retrieval -> deduplication -> sub-topic aggregation -> comprehensive report + citations), `ResearchReport` + `Citation`
- **Streaming Structured Output (`core::structured_output`)** (#9): `PartialJsonParser` incremental JSON parsing + `stream_structured_output<T>` streaming partial structures + `StreamingStructuredOutputExt` trait
- **Adaptive RAG (`agents::adaptive_rag`)** (#10): LLM routing to `NoRetrieval`/`SingleSearch`/`MultiQuery`, reusing existing Retrievers
- **Batch API (`core::batch`)** (#11): `BatchClient` submit/poll/results/cancel, OpenAI + Anthropic batch endpoints, `submit_and_wait` convenience method
- **Agent Observability / Deep Tracing (`callbacks::tracing`)** (#12): `Tracer` + `SpanGuard` (RAII) + `TracingBackend` trait + `InMemoryTracingBackend`/`ConsoleTracingBackend`/`OtelTracingBackend`, parent-child span tree

### Changed
- `LLMResult` added `thinking_content: Option<String>` field (#[serde(default)] for backward compatibility)
- `CallbackHandler` trait added `on_llm_thinking` default method
- `AnthropicChat` added `with_thinking(budget_tokens)` builder
- Cargo.toml added feature gates: `sandbox-e2b`, `sandbox-wasm`

### Fixed
- **Security**: PythonREPLTool added dangerous import checks (os/subprocess/socket and 17 other modules); HTTPTool SSRF switched to async DNS to prevent rebinding; URLFetchTool added private IP filtering; SQLTool blocks semicolon/comment/subquery bypasses; Gemini API key moved from URL to header (C1-C5)
- **Panic fixes**: choices[0] changed to `.first().ok_or()` (OpenAI+Ollama); from_env() returns Result instead of expect panic (ResponsesConfig); Regex changed to LazyLock for one-time compilation; Mutex poison changed to `into_inner()` recovery (C7-C11)
- **SSE streaming**: Ollama/Anthropic/Gemini providers added cross-chunk buffer, no longer losing tokens; callbacks changed from `drop()` to `.then()` async execution; Gemini stream_chat added callbacks; activated dead module in responses.rs (H1-H7)
- **Multi-round Function Calling**: Anthropic system messages placed in top-level `system` field; Ollama AI messages include tool_calls; Gemini tool_result uses functionResponse; Anthropic tool_result uses content block format (H42-H45)
- **Concurrency safety**: langgraph/compiled switched to tokio::sync::RwLock; sessions/memory_store switched to tokio::sync::Mutex; mongo_memory avoids blocking_write deadlock; HandoffManager merged into single Mutex; MCP Transport added request-level mutex (C17-C19,C23,H9-H13)
- **Data correctness**: parent_id separator changed from `_` to `::`; error propagation replaces `.ok()` silent swallowing; stream finalizer rewritten as reliable mechanism; UTF-8 splitting at character boundaries (not bytes); negative-score document filtering; RRF document IDs use content hash (C12-C16,H23-H27,H46)
- **Runnable::stream() pseudo-streaming**: OpenAI/Anthropic/Ollama changed to emit LLMResult per token (H4)
- **Batch API**: Anthropic message mapping corrected — system messages extracted to top-level `system` field, tool messages use tool_result format (H40)
- **RouterLLM**: Mutex poison changed to into_inner; RoundRobin index zero-check; stream_chat updates latency statistics; Arc shared messages reduce clone (H33-H38)
- **JSON fixes**: repair_partial_json correctly tracks braces inside strings; handles escaped quotes; UTF-8 character boundary checks (C20-C21,M37)
- **Other**: cosine_similarity uses epsilon floating-point zero check; different-length vectors return error; cache expired structured_output parse supports markdown wrapping; score range validation 0-1; A2A error response corrected; thiserror replaces manual Error implementations; multiple Regex changed to LazyLock; vector store operations optimized (C22,M21-M23,M30-M34,M7-M8)

## [0.4.2] - 2026-07-22

### Added
- **Shared math utilities `core::math`**: Added `src/core/math.rs`, extracting `cosine_similarity` shared implementation (with doctest + unit tests), reused by vector_stores / retrieval / embeddings / evaluation and 12 other locations, removing inline duplicate implementations across modules
- **Calculator safe expression evaluation**: `Calculator` tool integrated `meval` crate (`meval::eval_str`), supporting arithmetic / powers / functions (sin/cos/tan/sqrt/log/exp/abs) / constants (pi/e), replacing hand-written parsing
- **HTTP tool URL parsing**: `HTTPTool` integrated `url` crate, using `url::Url::parse` for URL normalization

### Changed
- **reqwest 0.11 -> 0.12** (breaking): Full migration of HTTP client code, affecting providers / embeddings / tools / mcp / a2a / vector_stores and other modules
- **Internal refactoring and deduplication**: Code cleanup in chains (document_chains / conversation_chain / retrieval_qa / llm_chain / router_chain), tools (calculator / http / url_fetch / python_repl), embeddings (deepseek / qwen), vector_stores (memory / file_store / chunked), mcp/transport, a2a/server, pinecone and other modules, unified reuse of `core::math::cosine_similarity`

### Fixed
- `MapRerankDocumentsChain::extract_score` integration test generic `M` type inference failure (`tests/unit/conversation_retrieval_chains.rs`), aligned with source code style by explicitly specifying types

## [0.4.1] - 2026-07-20

### Added
- **Assistants requires_action tool dispatch**: `OpenAIAssistant` polling encounters `requires_action` → parse `tool_calls` → execute via `ToolRegistry` → `submit_tool_outputs` → continue polling until `completed`/`failed`/`cancelled`
- **A2A Agent Protocol**: Added `src/a2a/` module
  - `AgentCard` / `A2ATask` / `A2AMessage` / `TaskStatus` / `A2ARequest` / `A2AResponse` / `A2AErrorData` protocol types
  - `A2AServer`: handler functions (`tasks/send`/`tasks/get`/`tasks/cancel`), pluggable into any HTTP framework (axum/actix/warp), includes `RwLock<HashMap>` in-memory task persistence
  - `A2AClient`: reqwest HTTP client, `get_agent_card()`/`send_task()`/`get_task()`/`cancel_task()`
- **with_structured_output**: `StructuredOutputExt` trait + standalone function, routing by provider to function calling or `JsonOutputParser` fallback, 11 tests
- **Chain streaming**: `BaseChain::stream()` default implementation + `LLMChain`/`ConversationChain` overrides, per-token callback `on_llm_new_token`
- **ContextWindow long context management**: `ContextWindow<M: BaseChatModel>`, two strategies:
  - `Strategy::Truncate`: Truncate old messages by token count
  - `Strategy::Summarize`: Compress old conversations via LLM summary when exceeding limit
  - `TiktokenCounter` integration, 18 tests
- **FileVectorStore**: JSON-persisted vector store, atomic writes (tmp+rename), cross-instance persistence, dimension validation, full `VectorStore` trait implementation
- **ComputerUseTool**: Anthropic computer use API integration + Native screenshot/keyboard/mouse (feature gate `computer-use-native`)
- **DocxLoader**: ZIP extraction + XML parsing of `<w:t>` text nodes
- **WebScraperLoader**: Web scraping, recursive link tracking, same-domain filtering, configurable max depth/page count
- **SitemapLoader**: Parse sitemap.xml, batch crawl pages
- **LocalEmbeddings ort**: ONNX Runtime neural network embeddings (feature gate `local-embeddings`, depends on `ort` + `ndarray`), replacing original bag-of-words placeholder implementation
- **wiremock test infrastructure**: `wiremock` as dev-dependency, mock helper functions, example tests, default tests do not hit real network
- **MSRV declaration**: `rust-version = "1.82"`, CI matrix includes 1.82
- **criterion benchmark**: `benches/` with retrieval(6)/splitter(4)/embedding(4) benchmark groups
- **12+ new examples**: evaluation / mcp_server / guardrails / sessions / context_window / vectorstore_memory / semantic_splitter / file_vectorstore / otel / assistants / handoffs / plan_execute / token_counter

### Changed
- **Unused import fix**: `async_trait` in `evaluation/pairwise.rs` moved into `#[cfg(test)]`
- **LocalEmbeddings**: Original bag-of-words implementation retained as default, ort implementation under `local-embeddings` feature
- **VectorStoreProvider**: `provider.rs` added `FileVectorStore` factory method
- **lib.rs**: Exported A2A module, `ContextWindow`, `FileVectorStore`, `StructuredOutputExt`, new loaders and other public APIs

### Fixed
- **Examples compilation fix**: All 25 examples compile successfully (fixed API name mismatches / type inference / unused imports / missing async etc.)
- **A2A server task persistence**: `tasks/get` and `tasks/cancel` previously always returned "not found", now implemented with in-memory storage and state query/transitions

## [0.4.0] - 2026-07-14

### Added
- **Evaluation module**: 10 evaluators (5 categories), framework includes `Score` / `Example` / `Dataset` / `Evaluator` / `Predictor` / `EvalRunner` / `Report`
  - Literal: `ExactMatch` / `StringDistance` (Levenshtein edit distance normalized)
  - Semantic: `EmbeddingSimilarity` / `LLMAsJudge` / `PairwiseJudge` (swaps A/B positions to reduce position bias)
  - Rule-based: `ContainsKeyword` / `RegexMatch` / `LengthCheck`
  - Classic NLP: `Bleu` (character-level tokenization + smoothing)
  - RAG: `Faithfulness` (splits claims for per-item verification to catch hallucinations, `join_all` concurrency, configurable `llm_split` / `empty_score`)
- **MCP Server**: `MCPServer` exposes local `BaseTool` as MCP Server (`initialize` / `tools/list` / `tools/call`), callable by Claude Desktop / Cursor and other hosts, symmetric with `MCPClient`
- **VectorStoreRetrieverMemory**: Vector retrieval memory, embeds each conversation turn into vector store, semantically recalls top-k history by current input
- **OpenAIAssistant**: Wraps OpenAI Assistants API (`Assistants` / `Threads` / `Run`, server-side session state)
- **SemanticSplitter**: Semantic chunker, splits at points where adjacent sentence similarity drops sharply, supports Chinese and English sentence splitting
- **LocalEmbeddings**: Lightweight local embeddings (word frequency hash + L2 normalization, pure Rust with no external dependencies)
- **OtelHandler**: OpenTelemetry callback handler, execution events converted to OTel spans (feature `opentelemetry`)

### Changed
- **Dependencies**: Added optional dependency `opentelemetry` + feature flag `opentelemetry` (disabled by default, does not affect default compilation)
- **Exports**: `lib.rs` exports evaluation module, `MCPServer`, `OpenAIAssistant`, `VectorStoreRetrieverMemory`, `LocalEmbeddings`, `SemanticSplitter`, `OtelHandler`

## [0.3.0] - 2026-07-12

### Added
- **examples directory**: 12 runnable examples (basic / agent / rag / langgraph / memory / chains)
- **MCP protocol support**: `MCPClient` (Stdio + SSE transport, `tools/list` + `tools/call`, MCPTool -> BaseTool adapter)
- **Multimodal vision**: `ImageContent` + `Message::human_with_image` (OpenAI / Ollama Vision serialization)
- **Sessions session management**: `SessionManager` + `SessionStore` (Memory) + multi-turn conversation memory
- **Token counter**: `TiktokenCounter` + `TokenTrackingLLM` + `ModelPricing` (cost estimation)
- **Guardrails safety guardrails**: `InputGuardrail` / `OutputGuardrail` + `SensitiveInfoGuardrail` + `GuardedAgent`
- **Plan-Execute Agent**: `Planner` + `PlanExecuteAgent` (plan - execute - replan)
- **Handoffs multi-agent handoff**: `HandoffManager` + `HandoffTool`
- **Streaming Tool Calls**: `StreamingFunctionCallingAgent` (`invoke_stream`)
- **Tool extensions**: `SQLTool` (read-only + table whitelist) + `HTTPTool` + `FileTool` (sandbox + extension whitelist)
- **PGVector vector store**: `PGVectorStore` (feature `pgvector-storage`, requires user to configure sqlx/pgvector dependencies)
- **Pinecone vector store**: `PineconeStore` (reqwest HTTP API)
- **HTML Loader**: `HTMLLoader` (regex text extraction, removes script/style)

### Changed
- **OpenAIChat**: Added `Clone` derive (supports PlanExecuteAgent / Streaming reuse)
- **Message**: Added `images` field (multimodal) + `additional_kwargs` with `serde(default)` for backward compatibility
- **Cleanup**: `compiled.rs` clippy fixes (type_complexity / collapsible_match / unnecessary_lazy_evaluations)

## [0.2.20] - 2026-05-05

### Fixed
- **create_resume_execution**: Fixed stripping `after_` prefix issue

### Changed
- **Documentation**: Updated HTML interrupt/checkpointer API

## [0.2.19] - 2026-05-05

### Added
- **Interrupt/Resume support**: LangGraph interrupt/resume execution
  - `last_checkpoint_state` state saving
  - `create_resume_execution` resume execution from interrupt point

### Changed
- **Documentation**: Updated interrupt/resume API documentation (Chinese and English)

## [0.2.18] - 2026-04-30

### Added
- **Output Parsers**: StrOutputParser + CommaSeparatedListOutputParser + JsonOutputParser + StructuredOutputParser + TypedOutputParser
- **Document Chains**: StuffDocumentsChain + RefineDocumentsChain + MapReduceDocumentsChain + MapRerankDocumentsChain
- **ConversationRetrievalChain**: Retrieval-augmented conversation with memory
- **Google Gemini**: GeminiChat (native API)
- **ChromaDB**: Lightweight vector database HTTP API
- **LLM Cache**: In-memory cache + TTL
- **Redis/SQLite storage**: RedisDocumentStore + SQLiteDocumentStore
- **Tools extensions**: Wikipedia + DuckDuckGo + PythonREPL
- **FewShotPrompt + ExampleSelectors**: Few-shot prompt templates + example selectors
- **LCEL composition operators**: RunnableSequence + RunnableParallel + RunnablePassthrough + RunnableLambda + BitOr trait
- **Qdrant**: `delete_by_metadata` method
- **MongoPersistentMemory**: Conditional compilation (only available when `mongodb-persistence` feature is enabled)

## [0.2.17] - 2025-04-24

### Added
- **Memory persistence**: Added PersistentMemory trait and MongoPersistentMemory implementation
  - `PersistentMemory` trait: Framework-level persistence interface, supports load_from_store/save_to_store/delete_session
  - `MongoPersistentMemory`: MongoDB storage, composes ConversationSummaryBufferMemory compression logic
  - `PersistenceConfig`: Configures auto_save/auto_load/token_limit
  - `MemoryData`: Memory data serialization structure
  - Framework handles compression algorithm, business layer handles storage implementation
- **ConversationSummaryBufferMemory**: Added `chat_memory_mut()` method for mutable access

## [0.2.16] - 2025-04-24

### Fixed
- **BM25 splitting algorithm**: Fixed Parent-Child splitting using simple character slicing that broke semantic boundaries
  - `InMemoryChunkedDocumentStore`: Uses `RecursiveCharacterSplitter` instead of `chars[start..end]`
  - `MongoChunkedDocumentStore`: Same change, MongoDB storage also uses semantic splitting
  - Separator priority: paragraph > line > period > space > character
  - Added chunk_overlap (default chunk_size / 10) to prevent boundary information loss

### Added
- **Documentation**: `docs/bm25_split_fix.md` with detailed explanation of splitting algorithm fix

## [0.2.15] - 2025-04-23

### Fixed
- **MongoChunkedDocumentStore**: Fixed compatibility issue of blocking methods inside tokio runtime
  - Uses `tokio::task::block_in_place` + `Handle::current().block_on` instead of creating a new runtime
  - Resolves "Cannot block the current thread from within a runtime" error

## [0.2.14] - 2025-04-23

### Changed
- **ChunkedDocumentStoreTrait**: Added blocking method support
  - `add_parent_document_blocking`: Synchronous add parent document
  - `get_parent_document_blocking`: Synchronous get parent document
  - `get_chunk_blocking`: Synchronous get chunk
  - `blocking_get_chunks_for_parent`: Synchronous get all chunks for a parent document
- **MongoChunkedDocumentStore**: Implemented blocking methods (using tokio runtime bridge)
- **ChunkedBM25Retriever/ChunkedBM25Index**: Changed to generic structs, supporting multiple DocumentStore backends
  - Default type parameter: `ChunkedBM25Retriever<S: ChunkedDocumentStoreTrait = ChunkedDocumentStore>`
  - Backward compatible: existing code continues to work without modification

### Fixed
- BM25 MongoDB persistence support: `MongoChunkedDocumentStore` can now be used as BM25 retriever storage backend

## [0.2.13] - 2025-04-22

### Added
- **LLM Providers**: Unified third-party LLM provider support
  - `DeepSeekChat`: DeepSeek API support
  - `MoonshotChat`: Moonshot (Kimi) API support
  - `QwenChat`: Alibaba Cloud Tongyi Qwen API support
  - `ZhipuChat`: Zhipu ChatGLM API support
  - `AnthropicChat`: Anthropic Claude API support
  - All providers use OpenAI-compatible interface or native API
- **Embeddings extensions**: New embedding generation services
  - `DeepSeekEmbeddings`: DeepSeek embedding generation
  - `QwenEmbeddings`: Tongyi Qwen embedding generation
- **Ollama enhancements**: Multimodal and tool calling improvements
  - Vision parameter support: `with_image()`, `with_images()`
  - Tool calling improvements: Better function calling support
  - Configuration enhancements: New `OllamaConfig` configuration options

### Changed
- **LangSmith Client**: `request_id` tracing enhancements
  - Optimized request tracing chain
  - Supports multi-level run tracing
- **Qdrant Vector Store**: Refactoring and optimization
  - Better error handling
  - Improved connection management
- **LangGraph Compiled**: State management improvements
- **MultiQuery Retriever**: Error handling optimization

### Configuration
- **Cargo.toml**: demo directory excluded (not uploaded to crates.io)

## [0.2.12] - 2025-04-19

### Documentation
- **Callbacks documentation**: Complete LangSmith tracing description
- **README**: Updated multi-Provider support list

## [0.2.11] - 2025-04-17

### Added
- **Document Loaders**: Document loader series
  - `TextLoader`: Plain text loading
  - `JSONLoader`: JSON document loading
  - `MarkdownLoader`: Markdown document loading
  - `PDFLoader`: PDF document extraction
  - `CSVLoader`: CSV data loading
- **MultiQuery Retriever**: Multi-query expanded retrieval
  - Automatically generates multiple related queries
  - Merges multi-query results
  - Improves retrieval recall rate
- **HyDE (Hypothetical Document Embeddings)**: Hypothetical document embeddings
  - Generates hypothetical answers based on questions
  - Uses hypothetical answers to retrieve relevant documents
- **Reranking**: Re-ranking module
  - Supports multiple re-ranking strategies
  - Improves retrieval precision

## [0.2.6] - 2025-04-18

### Added
- **LangGraph**: Graph workflow framework
  - `StateGraph`: State graph builder
  - `CompiledGraph`: Compiled executable graph
  - `GraphNode` trait + `SyncNode` + `AsyncNode`: Node abstractions
  - `GraphEdge` + `ConditionalEdge`: Edges and conditional routing
  - `StateSchema` trait + `AgentState`: State management
  - `Reducer` trait + `AppendReducer`: State update strategies
- **LangGraph Checkpointer**: Execution state persistence
  - `MemoryCheckpointer`: In-memory persistence
  - `ThreadSafeMemoryCheckpointer`: Thread-safe version
  - `FileCheckpointer`: File persistence
- **LangGraph Visualization**: Graph structure visualization output
  - `visualize_ascii()`: ASCII graphics
  - `visualize_mermaid()`: Mermaid diagram format
  - `visualize_json()`: JSON structure output
- **LangGraph Human-in-the-loop**: Human intervention mechanism
  - `interrupt_before`: Interrupt before execution
  - `interrupt_after`: Interrupt after execution
  - `resume()`: Resume execution from interrupt point
- **LangGraph Graph Validation**: Graph integrity validation
  - `validate_cycles()`: Infinite loop detection
  - `validate_unreachable_nodes()`: Orphan node detection
  - `validate_duplicate_edges()`: Duplicate edge detection
- **LangGraph Subgraph**: Subgraph nesting support
  - `SubgraphNode`: Subgraph node wrapper
  - State mappers: Parent-child graph state conversion
- **LangGraph Parallel**: Parallel node execution
  - `invoke_parallel()`: Execute multiple nodes in parallel
  - FanOut/FanIn pattern support
- **LangGraph Persistence**: Graph definition persistence
  - `GraphDefinition`: Graph definition structure
  - `NodeRegistry`: Node registry
  - `save_to_file()` / `load_from_file()`: Serialization/deserialization

### Tests
- Added `tests/langgraph/` directory (10+ test files)
- LangGraph basic tests, conditional edges, state management
- Async nodes, Checkpointer, visualization tests
- Human-in-the-loop, Subgraph, Parallel tests

### Documentation
- README.md updated core features list
- ROADMAP.md added LangGraph module details

## [0.2.5] - 2025-04-15

### Added
- **RetrievalQA**: One-stop retrieval QA Chain
  - Automatically retrieves relevant documents (RAG core)
  - Assembles Prompt (context + question)
  - LLM generates answer based on context
  - `query()` interface, one-line QA
  - `with_return_source_documents(true)` returns source documents
  - `with_prompt_template()` custom Prompt
  - `with_k()` configure retrieval count
- **RouterChain**: Conditional routing Chain
  - Automatically routes to different Chains based on input keywords
  - `LLMRouterChain` uses LLM for intelligent routing decisions
  - Supports default Chain (used when no match)
- **ConversationChain**: Conversation Chain with memory
  - Automatically saves and loads conversation history
  - Supports multi-turn conversation memory
  - `predict()` simplified interface
- **Memory system completion**: Complete conversation memory management
  - `ConversationBufferMemory`: No compression, saves all conversation history
  - `ConversationBufferWindowMemory`: Window truncation, keeps only the most recent k turns
  - `ConversationSummaryMemory`: LLM intelligent summarization, compresses old conversations
  - `ConversationSummaryBufferMemory`: Hybrid strategy, summary + recent conversations
  - `ChatMessageHistory`: Underlying message storage container
- **Streaming output enhancement**: Complete LLM stream_chat implementation
  - `stream_chat()`: Real-time per-token output
  - Typewriter effect, lower perceived latency for users
  - Supports streaming partial collection, mid-stream stop

### Tests
- Added `tests/unit/memory.rs` (Memory basic tests)
- Added `tests/unit/summary_buffer_memory.rs` (Compression trigger tests)
- Added `tests/unit/llm_stream.rs` (Streaming output tests)
- Added `tests/unit/retrieval_qa.rs` (RetrievalQA tests)
- Added `tests/unit/router_chain.rs` (RouterChain tests)

### Documentation
- USAGE.md added Memory detailed description
- USAGE.md added streaming output usage examples
- README.md updated Memory features list

## [0.2.4] - 2025-04-13

### Added
- **FunctionCallingAgent**: Agent using native Function Calling
  - Does not depend on text parsing, directly handles `tool_calls`
  - Type-safe: Tool parameters defined via JSON Schema
  - More reliable: Leverages model's native support, no dependency on Prompt Engineering
  - More efficient: Lower token consumption
- **to_tool_definition()**: Conversion function from BaseTool to ToolDefinition
  - Automatically generates JSON Schema from `args_schema()`
  - Simplifies tool binding workflow
- **Test directory**: Added `tests/function_calling/` dedicated to Function Calling tests
  - 5 test cases covering single-tool, multi-tool, system prompt and other scenarios
  - Comparison tests: ReActAgent vs FunctionCallingAgent

### Changed
- **OpenAI response parsing**: Fixed parsing error when `content` is null during Function Calling
  - `OpenAIMessage.content` changed to `Option<String>`
  - `OpenAIMessage.finish_reason` changed to `Option<String>`
- **Project structure**: Added `function_calling/` submodule under `agents/` directory
- **Exports**: Added `FunctionCallingAgent` and `to_tool_definition` public exports

### Documentation
- README added FunctionCallingAgent usage examples
- Added internal documentation explaining the differences and applicable scenarios of the two Agent types

## [0.2.3] - 2025-04-11

### Changed
- Removed Python reference comments from source code to keep codebase clean

## [0.2.2] - 2025-04-11

### Added
- **Callback System**: Complete execution tracing and monitoring framework
  - `CallbackHandler` trait: Defines LLM/Chain/Tool/Retriever callback interfaces
  - `CallbackManager`: Multi-handler management and dispatch
  - `StdOutHandler`: Console log output
  - `LangSmithHandler`: LangSmith platform tracing integration
  - `RunTree`: Run hierarchy and trace ID management
  - `RunType`: LLM/Chain/Tool/Retriever type enum
- **Tool Callbacks**: Full lifecycle tracing of tool execution
  - `on_tool_start`: Log input when tool starts
  - `on_tool_end`: Log output when tool completes
  - `on_tool_error`: Log error when tool fails
- **Tool Calling enhancements**: Complete OpenAI function calling support
  - `bind_tools()`: LLM binds tool definitions
  - `ToolDefinition`: Tool definition structure (name, description, parameters)
  - `ToolCall` / `ToolCallResult`: Tool call parsing
  - `with_structured_output<T>()`: Structured output method
  - `StructuredOutput<T>`: Generic structured output wrapper
  - `StructuredTool<T>`: Generic structured tool wrapper
- **Runnable interface**: LCEL base trait
  - `Runnable<Input, Output>`: Unified execution interface
  - `RunnableConfig`: Configuration supporting callbacks, tags, metadata
  - `invoke()` / `batch()` methods

### Changed
- `OpenAIChat` implements `Runnable<Vec<Message>, String>` trait
- `RunnableConfig` supports callback system integration (`with_callbacks()`)
- AgentExecutor automatically triggers tool callbacks

### Documentation
- Added `docs/internal/ROADMAP.md`: Feature development roadmap
- Added `docs/internal/FEATURE_PLAN.md`: Detailed implementation plan
- README updated with callback system usage examples

## [0.2.1] - 2025-04-09

### Changed
- **Project Structure Cleanup**: Reorganized documentation and tests
  - Moved internal docs to `docs/internal/` (not published)
  - Moved test files from root to `tests/` directory
  - Removed `examples/` directory (examples now in tests)
- **Git Configuration**: Updated `.gitignore` to exclude AI tool directories
  - Added `.sisyphus/` to gitignore
  - Added `docs/internal/` to gitignore
- **Documentation**: Updated README with complete RAG + LLM examples

### Removed
- Removed `examples/` directory and Cargo.toml example configurations
- Removed internal documentation from git tracking

## [0.2.0] - 2025-04-07

### Added
- **Complete RAG + LLM Integration**: Full retrieval-augmented generation pipeline
  - `OpenAIEmbeddings`: Real AI-powered vector generation
  - Automatic vector generation in `add_documents()`
  - Batch embedding API calls for efficiency
- **Qdrant Vector Database Support**: Production-ready vector storage
  - `QdrantVectorStore`: Full integration with Qdrant
  - `QdrantConfig`: Configurable vector size, distance metrics
  - Feature-gated: `qdrant-integration` feature
- **Comprehensive RAG Test Suite**: 6 complete tests with real API calls
  - `test_inmemory_embeddings_real`
  - `test_rag_inmemory_full_pipeline`
  - `test_rag_with_document_splitting`
  - `test_rag_qdrant_full_pipeline`
  - `test_compare_memory_vs_qdrant`
  - `test_rag_multi_turn_conversation`

### Changed
- **Vector Generation**: Now automatic via `retriever.add_documents(docs)`
- **README**: Added complete RAG + LLM examples with real embeddings

## [0.1.2] - 2024-04-07

### Added
- **Prompts Module**: New `PromptTemplate` and `ChatPromptTemplate` for flexible prompt engineering
  - `PromptTemplate`: String template with `{variable}` placeholders
  - `ChatPromptTemplate`: Multi-message template for chat scenarios
- **OpenAIError Export**: `OpenAIError` is now publicly accessible from `langchainrust::language_models::openai`
- **Example Configuration**: All examples configured in `Cargo.toml` for easy running
- **LICENSE File**: MIT License for open source distribution

### Changed
- **Refactored Examples**: All examples updated to match current API
  - Fixed `chain_pipeline.rs` to use proper LLMChain API
  - Fixed `memory_conversation.rs` to use `ChatMessageHistory`
  - Fixed `full_pipeline.rs` to work with current components
  - Removed unused imports in `multi_tool_agent.rs` and `rag_demo.rs`
- **Removed Reference Comments**: Cleaned up "reference Python version" comments from all source files
- **Improved Documentation**: 
  - Rewritten README with bilingual support (English/Chinese)
  - Updated examples/README with clearer structure
  - Added comprehensive API usage examples

### Fixed
- All examples now compile and run successfully
- Proper trait imports (`BaseChain`, `BaseMemory`) in examples
- Type annotation issues resolved in chain examples

### Documentation
- Bilingual README (English + Chinese)
- Improved code examples with error handling
- Added configuration tables and feature descriptions
- Cleaner project structure documentation

## [0.1.1] - 2024-03-XX

### Added
- Initial release with core features
- LLM support (OpenAI compatible)
- ReActAgent with tool calling
- Memory management
- Chains (LLMChain, SequentialChain)
- RAG components
- Built-in tools (Calculator, DateTime, Math, URLFetch)

[0.9.0]: https://github.com/atliliw/langchainrust/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/atliliw/langchainrust/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/atliliw/langchainrust/compare/v0.7.1...v0.7.2
[0.7.0]: https://github.com/atliliw/langchainrust/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/atliliw/langchainrust/compare/v0.5.2...v0.6.0
[0.5.0]: https://github.com/atliliw/langchainrust/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/atliliw/langchainrust/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/atliliw/langchainrust/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/atliliw/langchainrust/compare/v0.3.0...v0.4.0
[0.2.14]: https://github.com/atliliw/langchainrust/compare/v0.2.13...v0.2.14
[0.2.13]: https://github.com/atliliw/langchainrust/compare/v0.2.12...v0.2.13
[0.2.12]: https://github.com/atliliw/langchainrust/compare/v0.2.11...v0.2.12
[0.2.11]: https://github.com/atliliw/langchainrust/compare/v0.2.6...v0.2.11
[0.2.6]: https://github.com/atliliw/langchainrust/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/atliliw/langchainrust/compare/v0.2.4...v0.2.5
[0.2.3]: https://github.com/atliliw/langchainrust/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/atliliw/langchainrust/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/atliliw/langchainrust/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/atliliw/langchainrust/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/atliliw/langchainrust/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/atliliw/langchainrust/releases/tag/v0.1.1
