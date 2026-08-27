# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.18.1] - 2026-08-27

Patch release fixing the 7 review findings from the 0.18.0 gate.

### Fixed
- **Sandbox path traversal in `A2AServer`** (`lc-a2a`): file reads/writes are now checked against a normalized (canonicalized) sandbox root and any path containing `..` is rejected outright. Previously a request could escape the sandbox via lexical `..` components or a symlinked ancestor and read/write files outside the root (S1)
- **`SelfQueryRetriever` refuses to degrade silently** (`lc-rag`): a generated filter referencing a field outside `allowed_attributes` now returns `RetrieverError::InvalidFilter` instead of dropping the filter and running an unfiltered search; an empty whitelist (filtering disabled) still logs a warning and ignores the filter (S2)
- **`FileCheckpointer` writes atomically** (`lc-langgraph`): `save` writes to a `.tmp` sibling then renames, so a crash mid-write can no longer leave a truncated checkpoint visible to `list` (S3)
- **Docs corrected** (`langchainrust`): USAGE.md / USAGE_EN.md now state that `run_once` resolves `requires_action` inside its own polling loop (it does not abandon the run); the CHANGELOG wording about `crates/lc/tests/` now correctly says the directory stays gitignored by design (S4, S5)
- **Breaking — `corpus_bleu` returns `Result`** (`lc-evaluation`): a predictions/references length mismatch returns `Err(EvalError::LengthMismatch { predictions, references })` instead of silently returning a meaningless score; every exit path is now `Ok` (S6)
- **ReAct streaming drops empty chunks** (`lc-agents`): the final-answer stream no longer emits empty-string tokens, matching the FunctionCalling agent (S7)

### Migration
- **`corpus_bleu` callers** (`lc-evaluation`): the return type is now `Result<f64, EvalError>`; callers must `?` / `.unwrap()` / match on `EvalError::LengthMismatch`

## [0.18.0] - 2026-08-26

### Added
- **`StreamChunk` with `TokenUsage`** (`lc-core`, breaking): `stream_chat` now yields `Result<StreamChunk, _>` where `StreamChunk { text, token_usage: Option<TokenUsage> }` — providers that report usage fill `token_usage` on the final chunk, so the streaming path no longer loses token accounting (S1)
- **Function-calling per-token streaming** (`lc-agents`): `FunctionCallingAgent` overrides `plan_stream` to stream final-answer tokens (reusing the ReAct pattern), and the budget gate receives real cumulative usage on the streaming path (S2)
- **`MetadataFilter`** (`lc-vector-stores`, breaking): `VectorStore` gains `similarity_search_with_filter`; `MetadataFilter::{Field, And, Or}` over `FilterOp::{Eq, Ne, Gt, Gte, Lt, Lte, In, Nin}`; backends that cannot filter return an explicit `UnsupportedFilter` instead of silently ignoring (S3)
- **`SelfQueryRetriever`** (`lc-rag`): an LLM splits a natural-language query into `{query, filter}` via structured call, with an `allowed_attributes` whitelist guarding filter fields; composes into LCEL as a `RetrieverRunnable` (S4)
- **`PGVectorStore`** (`lc-vector-stores`, `pgvector-storage` feature): typed `VectorStore` implementation on sqlx + pgvector, feature-gated to keep the libsqlite3-sys linkage conflict out of default builds; table-name whitelist + parameterized SQL preserved (S5)
- **Cross-process resume** (`lc-agents`): `FileResumeStore` persists the pending human-approval / budget-gate state to disk (atomic write + recovery), so a restarted executor loads the pending point and re-enters approval instead of restarting the agent loop (S6)
- **`ReplayStrategy::Exact`** (`lc-testkit`): strict signature-matched replay — each request is matched to its recorded exchange by the full messages signature; no match returns an explicit `TestkitError` (Fifo / ByToolName unchanged, non-breaking) (S7)
- **MCP server primitives wired** (`lc-mcp`): `resources/list`, `resources/read`, `prompts/list`, `prompts/get`, `completion/complete` are handled via provider registration (unregistered → `method_not_found`); `sampling/createMessage` and `elicitation/create` ship as server→host initiation methods behind injected callbacks (no callback → explicit error); `initialize` advertises only the registered capabilities (S10)

### Changed
- **Breaking — `stream_chat` chunk type** (workspace): the chunk is `StreamChunk` instead of `String`; consumers destructure `text`, and streaming token usage is now available directly instead of relying on non-streaming `invoke` (S1)
- **Breaking — `VectorStore` trait** (`lc-vector-stores`): third-party implementations must add `similarity_search_with_filter` (the default delegates to `similarity_search` when `filter: None` and errors on unsupported filters) (S3)
- **11 known-failing online tests marked `#[ignore]` with reasons** (`langchainrust`): `crates/lc/tests/` remains gitignored by design (functional-test directory stays out of git/CI, run locally only); the failing cases (chains 7× model-permission `AccessDenied.Unpurchased`, core f04 / evaluation f06 output drift, mcp f10/f11 remote unreachable) are explicit `#[ignore]` with the cause noted (S8)

### Migration
- **`docs/internal/v0.18.0/MIGRATION_0.17_to_0.18.md`**: covers the `StreamChunk` destructure and the `VectorStore` trait addition (S9)

## [0.17.0] - 2026-08-26

### Added
- **lc-testkit phase 2 — tool recording & out-of-order replay**: `RecordedExchange` gains an optional `tools` field (`#[serde(default)]`, old fixtures deserialize unchanged); `RecordingProvider::bind_tools` records bound tool definitions into exchanges, `ReplayProvider::bind_tools` returns itself so tool-calling agent loops run fully offline; new `ReplayStrategy::{Fifo, ByToolName}` covers concurrent/out-of-order replay (`ByToolName` routes each request to the exchange whose tools / `tool_calls` match). Ships agent-level offline replay (`tests/agent_offline.rs`) and six chain scenarios transcribed from online tests (`tests/chains_offline.rs`)
- **`predict_tools`** (`lc-core`): one-shot tool call — `bind_tools` + `chat` in a single entry point; returns an explicit `PredictToolsError::ToolsUnsupported` instead of silently degrading when the model cannot bind tools
- **`RetrieverRunnable`** (`lc-rag`): wraps any `RetrieverTrait` as `Runnable<String, Vec<Document>>`, so retrieval becomes an LCEL chain step
- **`SessionManagerRunnable`** (`lc-sessions`): wraps persistent sessions as `Runnable<(session_id, message), reply>` — repeated invokes with the same session id accumulate history automatically
- **`ParentDocumentRetriever`** (`lc-rag`): parent-child retrieval — small chunks are indexed, a hit on any chunk returns the full parent document; `ParentDocument → prompt → LLM` composes as one LCEL chain

### Changed
- **Breaking — `ToolCall::new` removed**: the deprecated 3-positional-arg constructor is gone; use `ToolCall::builder(id).name(..).arguments(..)`
- **Breaking — `LocalEmbeddings` fallback alias removed** (no `local-embeddings` feature): the name is no longer available without the feature; use `BagOfWordsEmbeddings` explicitly or enable the feature

## [0.16.0] - 2026-08-25

### Added
- **Configurable runtime selection** (`lc-core`): `RunnableConfigurable` routes a chain between a default runnable and named alternatives at invoke time, driven by `config.configurable["key"]` (the Rust counterpart of Python LCEL's `configurable_alternatives`); `RunnableConfigurableFields` applies per-field overrides (`configurable_fields`). `RunnableConfig` gains builder methods `with_configurable` / `with_tag` / `with_metadata` / `with_max_concurrency` / `with_run_id` / `with_run_name` / `with_callbacks`
- **`RunnablePick` / `pluck`** (`lc-core`): keep selected keys — or pull one value — out of any runnable whose output is `HashMap<String, Value>` (Python `pick` / `pluck` counterpart). The composition is checked at compile time: a chain that does not produce a map simply won't compile
- **LangGraph checkpoint improvements** (`lc-langgraph`): the checkpoint lifecycle is reworked and its documentation unified across `USAGE.md` / `USAGE_EN.md`
- **3 new runnable examples + deployment guides** (`langchainrust`): `a2a_http_server`, `mcp_sse_server`, `mcp_stdio_server`, each with a matching `*-deployment.md` walkthrough
- **`lc-testkit` — record/replay test harness** (new crate, #22): `RecordingProvider` wraps any `BaseChatModel`, records real request/response exchanges to JSONL; `ReplayProvider` replays them offline with zero network, so framework tests run without API keys. Ships a round-trip test and a real `LLMChain` replay test (`fixtures/llm_chain_f01.jsonl`)
- **Agent human-approval gate** (`lc-agents`): `AgentExecutor::with_approval(Arc<dyn ApprovalHandler>)` pauses before tool execution; `ApprovalDecision::{Allow, Deny, Modify}` — `Deny` feeds the reason back as an observation (mirrors the `Skip` hook path), `Modify` rewrites the arguments before execution. Async (`approve(&self, ctx).await`), default off
- **Agent budget gate** (`lc-agents`): `AgentExecutor::with_budget(BudgetConfig)` enforces hard limits on cumulative tool calls, LLM tokens, wall-clock duration and iterations; exceeding returns `AgentError::BudgetExceeded` with the precise `limit`/`actual`. Default off — zero behavior change when unset

### Changed
- **Breaking — typed errors across the API surface**: ~78 public signatures switch from `Result<_, String>` to crate-specific `thiserror` errors (lc-providers / lc-embeddings / lc-callbacks / lc-core / lc-vector-stores / lc-agents / lc-prompts / lc-tools / lc-mcp / lc-rag), with `From` conversions bridging them into `LcelError`
- **Breaking — `Document` metadata carries structured values** (`lc-shared`): `metadata` is now `HashMap<String, serde_json::Value>`; `with_metadata` accepts `impl Into<Value>`, so non-string metadata (numbers, nested objects) survives serialization
- **Breaking — chains accept `Arc<dyn BaseChatModel>`** (`lc-chains`): the nine chain constructors drop the `<M: BaseChatModel>` type parameter for `Arc<dyn BaseChatModel<Error = ProviderError>>`, letting one chain instance swap model implementations without re-monomorphizing
- **Production error/log strings unified to English** (workspace): log / panic / `expect` text in non-test code is English; LLM judge prompts, tool-result strings, and test code intentionally keep Chinese
- **Public error enums marked `#[non_exhaustive]`** (workspace): 60+ `*Error` enums gain `#[non_exhaustive]`, so future minor versions can add variants without breaking consumers
- **Constructors take `impl Into<String>`** (`lc-langgraph` / `lc-memory` / `lc-agents` / `lc-a2a` / `lc-rag` / `lc-shared`): string parameters accept `&str` / `String` directly, no `String::from` ceremony at call sites
- **`ChunkedDocumentStoreTrait` no longer ships fake `save` / `load`** (`lc-vector-stores`): the trait-level defaults that always errored are removed; each backend exposes its own inherent `save` / `load`
- **Core error-handling & logging cleanup** (`lc-core`): runnable error propagation, fallback behavior, and provider error mapping hardened
- **CI**: a semver gate (`cargo-semver-checks`) and a wasm32 check (`lc-schema` / `lc-shared`) are added; the docs job enforces `RUSTDOCFLAGS: -D warnings`

### Deprecated
- **`ToolCall::new(id, name, args)`** (`lc-tools`): replaced by `ToolCallBuilder` — build with `.with_id()` / `.with_name()` / `.with_arguments()`; the positional constructor stays until 1.0

### Fixed
- **Production `unwrap` hardened** (workspace): a2a server, runnable binding / configurable paths, `json_parser`, and the docx loader no longer panic on unexpected data
- **428 undocumented public items documented** (workspace): every crate builds clean under `missing_docs`
- **LCEL default `transform` streams element-wise** (`lc-core`): the fallback `transform` no longer buffers the whole upstream stream before emitting — each input item runs its `stream` and is pushed downstream as it arrives, so `llm.pipe(parser)` emits progressively instead of waiting for the full answer (F1)
- **MCP requests are timeout-bounded** (`lc-mcp`): every POST send and response-body read is wrapped in a 30s timeout; a server that accepts the connection but never replies now returns a clear error (then reconnect-and-retry) instead of hanging forever (F2)
- **`AgentExecutor::stream` doc is honest about text granularity** (`lc-agents`): documents that `Text` events arrive per token only when the agent streams internally; otherwise the whole final answer arrives as a single `Text` event (F3)
- **MCP SSE client accepts 202 + SSE-push servers** (`lc-mcp`): when a server answers POST with `202 Accepted` and pushes the JSON-RPC response over SSE, the client correlates it by request `id` and returns it; direct-response servers still work unchanged (F4)
- **`#[tool]` no longer swallows serialization failures** (`lc-tools-derive`): `run()` returns `ExecutionFailed` instead of silently falling back to `Debug` text; and `invoke()` passes through `ToolError` unchanged instead of flattening it into `ExecutionFailed` (breaking) (F5)
- **ReAct parser prefers Action, Final Answer takes last occurrence** (`lc-agents`): an Action wins even when "Final Answer:" appears in the thought text; the final answer is read from the last occurrence (F6)
- **Errored rounds are still saved to agent memory** (`lc-agents`): when `invoke` fails, the user input + an error marker are written to memory, so the next round keeps context instead of losing the previous input (F7)

## [0.15.0] - 2026-08-20

### Added
- **Parsers accept `LLMResult`** (`lc-core`): `StrOutputParser` / `JsonOutputParser` / `CommaSeparatedListOutputParser` / `StructuredOutputParser` / `TypedOutputParser` are now `Runnable<LLMResult, _>` — `invoke` reads `input.content` and delegates to the existing `parse(&str)` (unchanged), so any parser chains directly after any LLM with no `.content` glue
- **`From<OutputParserError> for LcelError`** (`lc-core`): output-parser errors flow into the unified pipeline error, letting parsers sit at any non-first position in a `pipe()` chain
- **`ChatPromptTemplate` is `Runnable<HashMap<String, String>, Vec<Message>>`** (`lc-prompts`): prompt templates enter LCEL chains as the first step (`prompt.pipe(llm)`); lc-prompts gains an lc-core dependency (cycle-checked)
- **`RunnableWithMessageHistory`** (`lc-memory`): wraps "LLM + memory" as a single `Runnable<String, LLMResult>` — reads history → appends the user message → `llm.chat` → writes the exchange back, so multi-turn memory composes in one pipe
- **`From<OpenAIError> for LcelError`** (`lc-providers`): native `OpenAIChat` pipes directly without wrapping in `LLMClient` (Qwen / DeepSeek already bridged via `ProviderError`)
- **`lcel_compose` example** (`langchainrust`): one runnable program composing prompt + memory + LLM + parser + RAG in a single chain (`cargo run --example lcel_compose`)

## [0.14.0] - 2026-08-14

### Added
- **True streaming in `RunnableSequence`** (`lc-core`): `stream` / `transform` now send a single input through the first step via `stream_any` and each later step via a real `transform` forward-path — LLM-backed chains emit tokens incrementally instead of degrading to `invoke`; LLM steps implement streaming `transform` (prompt-in → token-stream-out)
- **True concurrency in `Runnable::batch`** (`lc-core`): default `batch` maps inputs with `buffered(limit)`, honoring `config.max_concurrency` (order-preserving, bounded concurrency) instead of serial `for` loops
- **Shared SSRF guard** (`lc-tools`): a single `ssrf` module (private-IP detection + URL check) is now used by both `url_fetch` and `extended::http` — one implementation, no duplication
- **`URLFetchInput::include_headers` implemented**: response headers are merged into the fetch output when enabled (previously a dead field)
- **`select_examples_by_length` on a trait** (`lc-prompts`): FewShot example-length selection exposed as a trait method with a default implementation
- **Multi-provider `from_env`** (`lc-providers`): Azure / DeepSeek / Qwen / Moonshot / Zhipu / Cohere / Gemini environment-variable detection
- **`VectorStore::embed_query`** (`lc-vector-stores`): default returns `None` (no auto-embedding); similarity search falls back to embedding the query when a value is provided
- **`json_repair` module** (`lc-shared`): tolerant-JSON repair moved down from `lc-core` and reused by `ToolCall::parse_arguments` and `parse_llm_json`, adding an unescaped-quote repair step
- **`RunnableConfig` `temperature` / `max_tokens` fields**: per-client overrides flow into provider request bodies

### Changed
- **Workspace version**: All 21 crates bumped from 0.13.0 to 0.14.0
- **`MessageType::type_str` returns `String`** and includes the tool_call_id (`tool:{id}`) for tool messages
- **`count_tokens` returns `Result<usize, _>`** instead of panicking (`expect`) when the global encoder is missing
- **LLM cache is a true LRU**: hits refresh `cached_at`, eviction keeps `min_by_key(cached_at)` (no more FIFO-impersonating-LRU)
- **`RunnableConfig::merge` semantics**: `tags` deduplicated preserving insertion order; `callbacks` merged (handlers appended) rather than wholesale-overwritten
- **`bind_tools` honored** through `ChatModelWrapper` / `LLMClient` (no longer silently swallowed); `with_temperature` / `with_max_tokens` take effect via per-client overrides
- **Callback dispatch unified**: providers route through `CallbackManager::dispatch_*` instead of touching handlers directly; a generic combo handler keeps default delegation
- **OpenTelemetry span parenting**: child spans derive from the parent span context; backend `end_span` removes by run_id
- **Qdrant without the `qdrant-integration` feature errors** instead of silently degrading
- **FewShot format validates variables**: undeclared `{var}` placeholders in the suffix error out instead of staying as literal text
- **`extract_unique_links` truly deduplicates** (insertion order preserved); content length now reports the actual body length
- **SQLTool executes parameterized statements** (positional `?N` bindings) instead of raw string interpolation
- **PythonREPL blacklist hardened**: detects `__import__` / `eval` / `exec` / `execfile` / `compile` builtin calls; error text clarifies the blacklist is a noise filter, not a security boundary
- **Shared `MediaContent` struct** (`lc-schema`): Image / Audio / File multimodal types reuse a common url+mime_type structure
- **PromptTemplate placeholder caching**: placeholders parsed once at construction into `(Range, name)`; `format` no longer re-scans the template
- **`CharacterTextSplitter` dead code removed**; `chunk_size` is now a hard cap (overlap counts toward the quota); the auto `chunk` metadata key no longer overwrites an existing user key
- **`SearchResult` is serializable** (`Serialize` / `Deserialize`)
- **StdOutHandler respects `verbose`** on `on_run_error`; LangSmith dead state and unused batch-ingest methods removed

### Fixed
- **Recursion limit off-by-one** (`lc-langgraph`): a graph that uses exactly `limit` steps no longer misreports `RecursionLimitReached`; `FanOut` branches share the main-path step budget
- **Session concurrency** (`lc-sessions`): per-session striped locks around `chat` / `clear` / `archive` (no more get→modify→llm→update races); `Deleted` dead status removed; LLM errors mapped to a dedicated `SessionError::Llm`
- **`has_tool_calls`** no longer panics on a `None` tool_calls list
- **Reasoning content** (`lc-providers`): empty `content` no longer silently falls back to `reasoning_content` (thinking stays in `thinking_content`)
- **`stream_chat` request** now actually sends `stream=true` instead of carrying a dead config field
- **Vector-store `delete_by_metadata` / `count`** return real values instead of hard-coded placeholders; `std::sync::RwLock` + `unwrap` replaced with `tokio::sync::RwLock` + `?`
- **`parse_with_retry` actually retries** instead of discarding the retry parameters
- **Vector-store score filtering** uses a configurable threshold instead of a hard `score > 0`
- **Wasm / E2B sandbox shells removed** (`lc-tools`): the feature-gated backends always returned `NotImplemented` and are deleted (with their feature flags) rather than promising unsupported backends
- **`fetch_url` header test** accounts for reqwest lowercase-normalizing response header names

## [0.13.0] - 2026-08-13

### Added
- **LCEL adapters**: `AgentEventRunnable` (`Runnable<String, AgentStreamEvent>`) exposes the full agent event stream (`Text` / `ToolCall` / `ToolStart` / `ToolEnd` / `PipelineStep` / `FinalAnswer` / `Error`) instead of filtering to the final answer; `OrchestratorRunnable<O: Orchestrator>` wraps high-level orchestrators (PlanExecute / AdaptiveRAG / CorrectiveRAG / DeepResearch / FanOutFanIn / SequentialPipeline / TaskAdapter / ReviewOrchestrator) as `Runnable` — `config.metadata["trace_id"]` propagates into `RunContext`
- **Unified `Orchestrator` trait**: `PlanExecuteAgent` / `DeepResearchAgent` / `CorrectiveRAGAgent` / `AdaptiveRAG` previously had incompatible `run()` signatures; they now converge on a single trait with `Input` / `Output` associated types and a shared `RunContext`
- **ConversationChain pluggable memory**: `ConversationChain::from_memory(llm, Arc<Mutex<dyn BaseMemory>>)` accepts any `BaseMemory` implementation (window / summary / vector-store / persistent); new `ConversationChainBuilder` adds system-prompt and input/output-key customization
- **AdaptiveRAG structured routing**: tool-call routing decisions now use structured output
- **Per-call token tracking**: `last_token_usage()` on `FunctionCallingAgent` and `ReActAgent`
- **PlanExecuteAgent execution factory**: custom execution-agent factory with `FunctionCallingAgent` fallback
- **Agent hardening**: per-execution metrics (LLM/tool call counts, token usage, duration); LLM result cache keyed on input + intermediate steps + executor namespace; `PromptInjectionHook` (detect / sanitize prompt injection in tool-returned content); `TokenBudgetHook` (token-budget / call-quota limiting); `ToolPolicy` (risk-graded tool permissions + sandbox gating); exponential-backoff retry for LLM provider calls; structured tool-call output helper for planner / router / grader; `AgentTask` type (objective / expected_output / allowed_tools) for multi-agent dispatch
- **A2A enterprise scaling**: `FederationGateway` (cross-org federation), `AgentRegistry` (skill-aware discovery), `SkillRouter` (skill-based dispatch), `ResilientA2AClient` (layered fault tolerance), `RateLimiter` (concurrency + rate limiting), security module (anti-impersonation / tamper / hostile-agent defense), ~1000-agent scale building blocks, pluggable `TaskStore` persistence, axum HTTP serving (`A2AServer::serve`, feature-gated), and an `AgentExecutor` ↔ `BaseChain` adapter for stateful multi-turn tasks
- **MCP at 100+ Server scale**: `ConnectionManager` (lazy start + idle reclamation + pooling), `MCPGateway` (single entry, `server:tool` routing), `TenantGateway` (per-tenant isolation), health-check circuit breaker, static + dynamic tool discovery, `server:tool` namespace with conflict policy, per-tool timeout with progress reset, streaming tool output (`notifications/tool_partial`), `ToolOrchestrator` (DAG tool execution), per-Server process sandbox isolation
- **Embeddings**: Cohere provider; FastEmbed embeddings (feature-gated `fastembed`); shared OpenAI-compatible base for DeepSeek / Qwen; exponential-backoff HTTP retry
- **Guardrails**: audit sink (persist violations); LLM judge to reduce sensitive-info false positives
- **Shared LLM judge** (`lc-core`): structured `bind_tools` judge reused by evaluation + guardrails, with text-parsing fallback
- **RAG structured output**: structured helper for GraphRAG entity extraction and MultiQuery query generation

### Changed
- **Workspace version**: All 21 crates bumped from 0.12.0 to 0.13.0
- **Orchestrators unified**: PlanExecute / DeepResearch / CorrectiveRAG / AdaptiveRAG migrated onto the shared `Orchestrator` trait
- **Dependency docs corrected**: `docs/internal/CRATE_DEPENDENCIES.md` dependency lists and publish order updated for `lc-sessions` (adds `lc-memory`), `lc-chains` (adds `lc-callbacks`), `lc-guardrails` (adds `lc-chains` / `lc-schema` / `lc-providers`), and `lc-a2a` (adds `lc-agents`)

### Fixed
- **lc-vector-stores doc IDs**: `lancedb.rs` / `neo4j.rs` now generate a UUID fallback when a document has no ID
- **lc-vector-stores Neo4j auth**: basic auth header now base64-encodes `username:password`
- **Adapters polish**: error handling and code formatting cleanup across lc-agents / lc-chains / lc-rag / lc-core adapters

## [0.12.0] - 2026-08-06

### Added
- **Agent streaming — CRAG**: Step-by-step streaming with granular `PipelineStep` events (retrieving → retrieved → grading → graded → correcting → corrected → generating → hallucination_check → FinalAnswer)
- **Agent streaming — AdaptiveRAG**: Route-first-then-branch streaming (routing → routed → retrieving/generating → FinalAnswer) with `RagDecision` visibility
- **Agent streaming — DeepResearch**: Multi-round research streaming (planning → searching → synthesizing → gaps_found → completed → FinalAnswer) with gap detection between rounds
- **Chain tests**: 13 new unit tests for lc-chains (5 in base.rs, 8 in sequential_chain.rs)
- **Agent stream tests**: 4 new streaming tests for CRAG and AdaptiveRAG

### Changed
- **CRAG internal visibility**: `CRAGState`, `retrieve()`, `grade_documents()`, `correct()`, `generate()`, `hallucination_check()`, `format_reasoning()` made `pub(crate)` for streaming access
- **DeepResearch internal visibility**: `build_citations()` made `pub(crate)` for streaming access
- **Clippy compliance**: Fixed `map_or(false, ...)` → `is_some_and()`, removed needless borrows/returns, collapsed identical branches, added `#[allow(dead_code)]` for serde-only structs

### Fixed
- **6 doctest failures**: Changed `use langchainrust::` to `use lc_core::` with `no_run`/`ignore` markers — doctests now pass cleanly
- **16 test code warnings**: Removed unused imports/variables, added `#[allow(dead_code)]` for test-only structs — `cargo test --workspace --lib` now produces 0 warnings
- **lc-embeddings Cohere dimension**: Collapsed identical if/else branches to `let dimension = 1024`
- **lc-vector-stores naming**: `HashMap_is_empty` → `hash_map_is_empty` (snake_case), removed needless `.into_iter()`
- **lc-chains formatting**: `&format!(...)` → `format!(...)` (3 places)

## [0.11.0] - 2026-08-06

### Added
- **Callback 贯穿**: Callbacks now propagate through the full execution pipeline — LLMChain, AgentExecutor, and all chain/agent layers dispatch `on_llm_start/end`, `on_tool_start/end`, `on_chain_start/end` events. `CallbackPropagatable` trait for uniform callback propagation
- **RunnableRetry**: Retry mechanism for LCEL pipelines with configurable max retries, exponential backoff, jitter, and retryable error filtering. `RunnableExt::with_retry()` for fluent API
- **CancellationToken**: Cooperative cancellation for LCEL pipelines. `RunnableConfig` now carries an optional `CancellationToken`; long-running operations (retry loops, streaming) check `is_cancelled()` before each step
- **Agent Hook 系统**: Composable lifecycle interception for agents — `AgentHook` trait with `on_before_completion`, `on_after_completion`, `on_before_tool_call`, `on_after_tool_call`, `on_stream_chunk`, `on_agent_start/end`, `on_error` callbacks. Built-in hooks: `ApprovalHook` (human-in-the-loop), `ContentFilterHook` (stream filtering), `LoggingHook` (structured logging)
- **Anthropic Extended Thinking**: `with_thinking()` on AnthropicChat enables extended thinking with configurable budget tokens. Thinking content returned in `LLMResult.thinking_content`
- **OTel GenAI SemConv**: TraceSpan now includes 8 GenAI semantic convention fields (gen_ai_system, gen_ai_request_model, gen_ai_response_model, gen_ai_finish_reason, gen_ai_request_max_tokens, gen_ai_request_temperature, gen_ai_operation_name, gen_ai_tool_name) for OpenTelemetry observability
- **Message extensions**: `Message::human_with_images()`, `Message::merge_tool_results()`, and `Message::has_tool_calls()` helper methods on lc-schema

### Changed
- **Workspace version**: All 21 crates bumped from 0.9.0 to 0.11.0
- **Tool input serialization**: Fixed double-encoding bug where `ToolInput::String` containing JSON was wrapped in `Value::String()` then double-encoded by `serde_json::to_string()`. Now attempts `serde_json::from_str()` first to parse as proper `Value`
- **LangGraph stream API**: `stream()` returns `Pin<Box<dyn Stream>>` (not awaitable); use `stream_collected()` for `await`-able `Vec<StreamEvent>` result
- **Clippy compliance**: All workspace crates now pass `cargo clippy --workspace -- -D warnings` with 0 warnings

### Fixed
- Tool input double-serialization causing 8 integration test failures (calculator, search, etc.)
- TraceSpan missing gen_ai fields causing 6 unit test compilation errors
- LangGraph `stream()` compilation error (dyn Stream is not Future)
- Cargo cache corruption (E0786) requiring `cargo clean`

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

[0.16.0]: https://github.com/atliliw/langchainrust/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/atliliw/langchainrust/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/atliliw/langchainrust/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/atliliw/langchainrust/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/atliliw/langchainrust/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/atliliw/langchainrust/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/atliliw/langchainrust/compare/v0.9.0...v0.10.0
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
