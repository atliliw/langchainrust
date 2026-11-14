# OpenAI-Compatible Server

MCP and A2A cover **agent ↔ agent** and **agent ↔ tool**. The `openai-server`
feature adds the entry point most existing applications speak: OpenAI's
`/v1/chat/completions`.

```toml
langchainrust = { version = "0.30.0", features = ["openai-server"] }
```

## Serving a chat model

```rust
use std::sync::Arc;
use langchainrust::agents::{openai_compat_router, ModelBackend};

let backend = Arc::new(ModelBackend::new(Arc::new(llm)));
let app = openai_compat_router(backend, "my-agent");

// axum::serve(listener, app).await?;
```

## Endpoints

| Method | Path | Behavior |
|---|---|---|
| GET | `/v1/models` | Returns the configured model id |
| POST | `/v1/chat/completions` | `stream: false` → JSON; `stream: true` → SSE followed by `[DONE]` |

## Custom backend

`ChatBackend` is the seam between the wire protocol and your agent/chain:

```rust
#[async_trait::async_trait]
impl ChatBackend for MyAgentBackend {
    async fn complete(&self, messages: Vec<Message>) -> Result<LLMResult, String> { ... }
    async fn stream_chat(&self, messages: Vec<Message>)
        -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, String>> + Send>>, String> { ... }
}
```

## Scope

- The `tool` message role and `tool_calls` response fields are not mapped yet.
- `temperature` / `max_tokens` are accepted for compatibility but do not override
  the backend per request; sampling belongs to the backend configuration.
