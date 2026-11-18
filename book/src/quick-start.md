# Quick Start

From zero to a working agent in 5 minutes.

## Prerequisites

- Rust 1.85+
- An API key for OpenAI or any OpenAI-compatible endpoint

## 1. Rust: an agent with a tool

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # optional
cargo run -p langchainrust --example quickstart_agent
```

The example (`crates/lc/examples/quickstart_agent.rs`) builds an
`AgentExecutor` through `AgentBuilder::build_executor()` with `Calculator`,
runs one tool-calling loop, and prints the answer:

```rust
use langchainrust::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let executor = AgentBuilder::new()
        .llm(OpenAIChat::new(OpenAIConfig::from_env_result()?))
        .system("You are a concise assistant. Use tools when needed.")
        .tool(Calculator::new())
        .max_iterations(5)
        .build_executor()?;

    let answer = executor.invoke("What is 25 * 37?".to_string()).await?;
    println!("{answer}");
    Ok(())
}
```

## 2. Python: call the runtime over HTTP

**Terminal 1 — start the OpenAI-compatible server:**

```bash
cargo run -p langchainrust --example openai_server --features openai-server
# OpenAI-compatible server on http://127.0.0.1:8080
```

**Terminal 2 — install the thin client and call it:**

```bash
pip install -e sdks/python

python - <<'PY'
from langchainrust import Client

c = Client("http://127.0.0.1:8080", model="gpt-4o-mini")
print(c.chat("What is Rust?").content)
PY
```

Any OpenAI SDK works against the same endpoint:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
print(client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
))
```

## 3. TypeScript: the same endpoint

```bash
npm install langchainrust
# or, from this repository: npm install ./sdks/typescript
```

```ts
import { Client } from "langchainrust";

const c = new Client("http://127.0.0.1:8080", "gpt-4o-mini");
console.log((await c.chat("What is Rust?")).content);
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY` error | Export the key; the config fails fast when it is missing |
| Port already in use | `LC_ADDR=127.0.0.1:9090` before starting the server |
| `unknown feature openai-server` | Add `--features openai-server` |
| Python `ModuleNotFoundError` | Run `pip install -e sdks/python` from the repo root |

## Next

- [Agents](agents.md) — hooks, budgets, approvals, subagents
- [MCP](mcp.md) — expose and consume tools
- [A2A](a2a.md) — agent-to-agent
- [Evaluation](evaluation.md) — verifiers and regression gates
