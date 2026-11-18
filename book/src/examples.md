# Examples

Every example is a standalone binary:

```bash
cargo run -p langchainrust --example <name>
```

Some need a feature flag or an API key; both are noted below.
Examples that call a model read `OPENAI_API_KEY` (and optionally
`OPENAI_MODEL`).

## Basics

| Example | Shows | Env |
|---|---|---|
| `basic_chat` | Minimal chat completion | key |
| `basic_streaming` | Token streaming | key |
| `basic_multi_provider` | Swap providers behind one interface | key |
| `basic_quick_start` | The 3-line happy path | key |
| `basic_responses_api` | OpenAI Responses API | key |
| `basic_batch_api` | Batch inference | key |
| `basic_sandbox` | Local code sandbox | — |
| `basic_token_counter` | Token counting and cost | — |

## Agents

| Example | Shows | Env |
|---|---|---|
| `quickstart_agent` | Agent + calculator in one loop | key |
| `agent_function_calling` | Native tool calls | key |
| `agent_multi_tool` | Several tools in one turn | key |
| `agent_plan_execute` | Plan → execute → replan | key |
| `agent_handoffs` | Multi-agent handoff | key |
| `agent_subagent` | Fresh-context child agent, summary only | key |
| `agent_assistants` | OpenAI Assistants API | key |
| `agent_deep_research` | Multi-round research with citations | key |
| `agent_extended_thinking` | Provider reasoning/thinking channel | key |

## RAG

| Example | Shows | Env |
|---|---|---|
| `rag_bm25_search` | BM25 keyword search, no embeddings | — |
| `rag_document_loaders` | Load text/JSON/Markdown/PDF/CSV/HTML | — |
| `semantic_splitter` | Semantic chunking | — |
| `file_vectorstore` | File-backed vector store | — |
| `rag_corrective_rag` | Grade → rewrite → re-retrieve | key |
| `rag_adaptive_rag` | LLM decides whether to retrieve | key |
| `rag_graph_rag` | Knowledge-graph retrieval | key |

## Memory & Sessions

| Example | Shows | Env |
|---|---|---|
| `memory_buffer_memory` | Conversation buffer | — |
| `memory_context_window` | Fit history into a token budget | — |
| `memory_vectorstore_memory` | Semantic memory retrieval | — |
| `memory_sessions` | Multi-turn session lifecycle | — |
| `sessions_durable` | Event log + checkpoint + recover + fork | key |

## Chains & LCEL

> `chains_*` require `--features chains` (v0.31+); the LCEL examples need no feature.

| Example | Shows | Env |
|---|---|---|
| `chains_llm_chain` | The basic LLM chain | key |
| `chains_sequential_chain` | Multi-step chain | key |
| `lcel_pipe` | `prompt.pipe(llm).pipe(parser)` | — |
| `lcel_compose` | Parallel / branching composition | key |

## LangGraph

| Example | Shows | Env |
|---|---|---|
| `langgraph_basic_graph` | Nodes, edges, state | — |
| `langgraph_conditional_edge` | Conditional routing | — |

## Evaluation

| Example | Shows | Env |
|---|---|---|
| `evaluation` | Batch evaluation and report | key |
| `ragas_eval` | RAGAS context/answer metrics | key |
| `evaluation_verifier` | logprob Verifier + best-of-N | key |
| `evaluation_judge_reliability` | Order swap, κ, self-preference | — |

## Guardrails

> Requires `--features guardrails` (v0.31+).

| Example | Shows | Env |
|---|---|---|
| `guardrails` | Input/output rails, PII, schema | — |

## MCP & A2A

> `mcp_*` require `--features mcp`; `a2a_http_server` requires `--features a2a` (v0.31+).
> `mcp_server` (the Calculator demo) needs no feature.

| Example | Shows | Env |
|---|---|---|
| `mcp_server` | Expose tools over MCP | — |
| `mcp_http_server` | MCP over stateless HTTP | — |
| `mcp_stdio_server` | MCP over stdio | — |
| `a2a_http_server` | Expose an agent over A2A | — |

## Observability

| Example | Shows | Feature |
|---|---|---|
| `otel_tracing` | OTel spans | `--features opentelemetry` |
| `otlp_tracing` | OTLP export | `--features otlp` |

## Servers

| Example | Shows | Feature |
|---|---|---|
| `agent_sse_server` | Agent events over SSE | — |
| `openai_server` | OpenAI-compatible endpoint | `--features openai-server` |
| `openai_server_tools` | OpenAI endpoint returning `tool_calls` | `--features openai-server` |

---

## Notes

- Examples that call a model fail fast when `OPENAI_API_KEY` is missing.
- Feature-gated examples are skipped by `cargo check --examples` unless the
  feature is enabled; CI checks both the default and `openai-server` sets.
- The source lives in `crates/lc/examples/`; each file starts with a `//!`
  header describing what it shows, how to run it, and which env vars it needs.
