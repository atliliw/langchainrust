# Tracing & Callbacks

LangChainRust provides a callback system for observability, tracing, and monitoring with handlers for console, file, LangSmith, and OpenTelemetry.

## Callback System

| Component | Description |
|-----------|-------------|
| `CallbackHandler` | Trait with lifecycle hooks (LLM, chain, tool, retriever) |
| `CallbackManager` | Manages multiple handlers, dispatches events |
| `RunTree` | Structured run tracking with parent-child relationships |
| `RunType` | `Llm`, `Chain`, `Tool`, `Retriever`, `Embedding`, `Prompt`, `Parser` |

## Callback Handlers

| Handler | Struct | Output |
|---------|--------|--------|
| Console | `StdOutHandler` | Prints to stdout |
| File | `FileCallbackHandler` | Writes to file (Plain/JSON/JSONLines) |
| LangSmith | `LangSmithHandler` | Sends to LangSmith platform |
| OpenTelemetry | `OtelHandler` | Emits OTel spans with GenAI SemConv |

## Setup

```rust
use langchainrust::{
    CallbackManager, StdOutHandler, FileCallbackHandler,
    LangSmithHandler, LogFormat,
};
use std::sync::Arc;

let manager = CallbackManager::new()
    .add_handler(Arc::new(StdOutHandler::new()))
    .add_handler(Arc::new(FileCallbackHandler::new("./traces.jsonl")?
        .with_format(LogFormat::JsonLines)))
    .add_handler(Arc::new(LangSmithHandler::from_env()?));
```

## LangSmith Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGSMITH_API_KEY` | API key (starts with `ls_`) | Required |
| `LANGSMITH_TRACING` | Enable tracing | `true` |
| `LANGSMITH_PROJECT` | Project name | `default` |
| `LANGSMITH_ENDPOINT` | API endpoint | LangSmith official |
| `LANGSMITH_WORKSPACE_ID` | Workspace ID | For org accounts |

## OpenTelemetry (GenAI SemConv)

```rust
// Feature: opentelemetry
use langchainrust::OtelHandler;

let handler = OtelHandler::from_global("langchainrust");
// Sets GenAI Semantic Convention attributes:
// gen_ai.system, gen_ai.request.model, gen_ai.response.model,
// gen_ai.response.finish_reason, gen_ai.client.token.usage.*,
// gen_ai.request.max_tokens, gen_ai.request.temperature,
// gen_ai.operation.name, gen_ai.tool.name
```

## Tracing Subsystem

```rust
use langchainrust::{Tracer, SpanKind, SpanGuard, InMemoryTracingBackend};

let backend = Arc::new(InMemoryTracingBackend::new());
let tracer = Tracer::new(backend.clone());

let mut span = tracer.start("llm_call", SpanKind::Llm);
span.with_tokens(SpanTokenUsage { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 });
span.end(); // RAII: ends on drop if not called

// Inspect traces
let spans = backend.spans();
```

## RunTree

```rust
use langchainrust::{RunTree, RunType};

let run = RunTree::new("my_chain", RunType::Chain, serde_json::json!({"input": "hello"}))
    .with_tag("production")
    .with_project("my-project");

let child = run.create_child("llm_step", RunType::Llm, serde_json::json!({"prompt": "..."}));
```
