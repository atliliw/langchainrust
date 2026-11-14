# Language Clients (Python / TypeScript)

The Rust core is the runtime. Other languages talk to it over the
OpenAI-compatible HTTP endpoint, so they never need to write Rust.

Start the server once:

```bash
cargo run -p langchainrust --example openai_server --features openai-server
# OpenAI-compatible server on http://127.0.0.1:8080
```

## Python

```bash
pip install -e sdks/python
```

```python
from langchainrust import Client

c = Client("http://127.0.0.1:8080", model="gpt-4o-mini")

print(c.chat("What is Rust?").content)

for chunk in c.stream("hello"):
    print(chunk, end="")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
result = c.chat_with_tools("weather in beijing?", tools)
for call in result.tool_calls:
    print(call.name, call.arguments)
```

### Methods

| Method | Returns |
|---|---|
| `chat(input, tools=None)` | `ChatResult(content, finish_reason, tool_calls, usage)` |
| `stream(input)` | iterator of text deltas |
| `chat_with_tools(input, tools)` | same as `chat(..., tools=...)` |
| `models()` | the server's model list |

Errors raise `LangChainRustError` with an optional `status`.

## TypeScript

```bash
npm install langchainrust
# or, from this repository: npm install ./sdks/typescript
```

```ts
import { Client } from "langchainrust";

const c = new Client("http://127.0.0.1:8080", "gpt-4o-mini");

console.log((await c.chat("What is Rust?")).content);

for await (const chunk of c.stream("hello")) {
  process.stdout.write(chunk);
}

const result = await c.chatWithTools("weather in beijing?", tools);
for (const call of result.toolCalls) {
  console.log(call.name, call.arguments);
}
```

`Client` exposes `chat`, `chatWithTools`, `stream` and `models`; errors are
thrown as `LangChainRustError` with an optional `status`.

## Any OpenAI SDK

Because the endpoint is OpenAI-compatible, existing SDKs work unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
print(client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
))
```

## Scope

- These clients are thin: they only speak HTTP and do not re-implement agent
  logic.
- Sampling parameters belong to the backend configuration; the request-level
  `temperature` / `max_tokens` are accepted for compatibility but do not
  override it.
