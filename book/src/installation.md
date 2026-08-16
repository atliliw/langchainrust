# Installation

## Cargo

```toml
[dependencies]
langchainrust = "0.14.0"
```

## Feature Flags

LangChainRust uses feature flags to keep the default build lean:

```toml
[dependencies.langchainrust]
version = "0.14.0"
features = ["qdrant-integration"]  # Optional features
```

### Available Features

| Feature | Description |
|---------|-------------|
| `qdrant-integration` | Qdrant vector store |
| `mongodb-persistence` | MongoDB document store |
| `redis-storage` | Redis document store |
| `sqlite-storage` | SQLite document store |
| `pgvector-storage` | PGVector vector store |
| `local-embeddings` | ONNX Runtime local embeddings |
| `fastembed` | FastEmbed local embeddings (ONNX) |

## Individual Crates

You can also use individual crates directly:

```toml
[dependencies]
lc-core = "0.14.0"
lc-providers = "0.14.0"
lc-agents = "0.14.0"
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
