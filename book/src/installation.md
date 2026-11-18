# Installation

## Cargo

```toml
[dependencies]
langchainrust = "0.31"
```

The default build is the five main paths. For a one-import start:

```rust
use langchainrust::prelude::*;
```

## Feature Flags

LangChainRust uses feature flags to keep the default build lean:

```toml
[dependencies.langchainrust]
version = "0.31"
features = ["qdrant-integration"]  # Optional features
```

### Available Features

| Feature | Description |
|---------|-------------|
| `mcp` | MCP client/server (opt-in since v0.31) |
| `a2a` | A2A protocol (implies `chains`) |
| `guardrails` | Input/output guardrails |
| `chains` | `LLMChain` / `SequentialChain` / `RetrievalQA` |
| `full` | All extended surfaces above + `observability` |
| `qdrant-integration` | Qdrant vector store |
| `mongodb-persistence` | MongoDB document store |
| `redis-storage` | Redis document store |
| `sqlite-storage` | SQLite document store |
| `pgvector-storage` | PGVector vector store |
| `local-embeddings` | ONNX Runtime local embeddings |
| `fastembed` | FastEmbed local embeddings (ONNX) |
| `openai-server` | OpenAI-compatible HTTP server (`/v1/models`, `/v1/chat/completions`) |

## Individual Crates

You can also use individual crates directly:

```toml
[dependencies]
lc-core = "0.31"
lc-providers = "0.31"
lc-agents = "0.31"
```

## Environment Variables

Most providers require API keys set as environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export MISTRAL_API_KEY=...
export COHERE_API_KEY=...
export DEEPSEEK_API_KEY=...
export QWEN_API_KEY=...
```

## See also

- [Full usage guide](https://github.com/atliliw/langchainrust/blob/main/docs/USAGE_EN.md)
- [Language Clients (Python / TypeScript)](sdks.md)
