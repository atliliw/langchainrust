# OpenAI-Compatible Server

MCP and A2A cover **agent ↔ agent** and **agent ↔ tool**. The `openai-server`
feature adds the entry point most existing applications speak: OpenAI's
`/v1/chat/completions`.

```toml
# `lc-agents` exposes the feature today:
lc-agents = { version = "0.31", features = ["openai-server"] }
```

> The `langchainrust` facade forwards it as `openai-server`. If you build from
> `main`, enable `features = ["openai-server"]` on the facade as well.

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

## Tool calling

The server maps the OpenAI tool protocol end to end:

| Direction | Shape |
|---|---|
| Request `role: "tool"` | requires `tool_call_id`; mapped to a tool message |
| Request `tool_calls` on an assistant message | preserved for the next turn |
| Response `message.tool_calls[]` | `{id, type: "function", function: {name, arguments}}` |
| Response `finish_reason` | `"tool_calls"` when the model requested tools |
| Streaming `delta.tool_calls[]` | each entry carries `index` |

```json
{
  "model": "my-agent",
  "messages": [
    { "role": "user", "content": "weather in beijing?" },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": { "name": "get_weather", "arguments": "{}" }
        }
      ]
    },
    { "role": "tool", "tool_call_id": "call_1", "content": "sunny" }
  ]
}
```

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

- `tool` role, `tool_calls` responses and streaming tool deltas are mapped.
- `temperature` / `max_tokens` are accepted for compatibility but do not override
  the backend per request; sampling belongs to the backend configuration.
- Client SDKs: [Language Clients (Python / TypeScript)](sdks.md).
