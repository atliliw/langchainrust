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

The example (`crates/lc/examples/quickstart_agent.rs`) builds a
`FunctionCallingAgent` with `Calculator`, runs one tool-calling loop, and
prints the answer:

```rust
use std::sync::Arc;

use langchainrust::agents::{AgentExecutor, FunctionCallingAgent};
use langchainrust::core::tools::BaseTool;
use langchainrust::tools::Calculator;
use langchainrust::{OpenAIChat, OpenAIConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = OpenAIChat::new(OpenAIConfig::from_env_result()?);
    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];

    let agent = FunctionCallingAgent::new(
        llm,
        tools.clone(),
        Some("You are a concise assistant. Use tools when needed.".to_string()),
    );

    let executor = AgentExecutor::new(Arc::new(agent), tools).with_max_iterations(5);
    println!("{}", executor.invoke("What is 25 * 37?".to_string()).await?);
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
