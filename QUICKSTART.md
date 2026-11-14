# 5-Minute Quickstart

Two paths. Pick one.

---

## Rust: an agent with a tool

**Prerequisites:** Rust 1.85+ and an API key.

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # optional
```

Run the checked-in example:

```bash
cargo run -p langchainrust --example quickstart_agent
```

It builds a `FunctionCallingAgent` with `Calculator`, runs one tool-calling
loop, and prints the answer.

The whole program (`crates/lc/examples/quickstart_agent.rs`):

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

---

## Python: call the runtime over HTTP

The Rust core exposes an OpenAI-compatible endpoint, so Python does **not**
need to write Rust.

**Terminal 1 — start the server:**

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

The same endpoint works with any OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
print(client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
))
```

Tool calling works too — send a `tools` array and read `tool_calls` back.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY` error | Export the key; `OpenAIConfig::from_env_result()` fails fast when it is missing |
| Port already in use | `LC_ADDR=127.0.0.1:9090` before starting the server |
| `unknown feature openai-server` | Run with `--features openai-server` |
| Python `ModuleNotFoundError: langchainrust` | Run `pip install -e sdks/python` from the repo root |

---

## Next

- [Agents](book/src/agents.md) — hooks, budgets, approvals, subagents
- [MCP](book/src/mcp.md) — expose and consume tools
- [A2A](book/src/a2a.md) — agent-to-agent
- [Evaluation](book/src/evaluation.md) — verifiers and regression gates
- [Full usage guide](docs/USAGE_EN.md)
