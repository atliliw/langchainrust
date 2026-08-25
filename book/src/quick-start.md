# Quick Start

Get started with LangChainRust in 5 minutes.

## Prerequisites

- Rust 1.82+ (edition 2021)
- An OpenAI API key (or any supported LLM provider)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
langchainrust = "0.16.0"
tokio = { version = "1", features = ["full"] }
```

## Your First Chat

```rust
use langchainrust::openai::OpenAIChat;
use langchainrust::language_models::{BaseChatModel, LLMResult};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() {
    let llm = OpenAIChat::from_env_result().expect("Set OPENAI_API_KEY");

    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::human("What is Rust?"),
    ];

    let result = llm.chat(messages, None).await.expect("Chat failed");
    println!("{}", result.content);
}
```

## Agent with Tools

```rust
use langchainrust::agents::{AgentExecutor, BaseAgent};
use langchainrust::tools::CalculatorTool;

let agent = ReActAgent::new(llm, tools);
let executor = AgentExecutor::new(Arc::new(agent), tools)
    .with_max_iterations(10);

let result = executor.invoke("What is 25 * 37?".to_string()).await?;
```

## RAG Pipeline

```rust
use langchainrust::rag::RAGPipeline;

let pipeline = RAGPipeline::new(llm, retriever);
let answer = pipeline.run("What is LangChain?").await?;
```

## Next Steps

- [LLM Providers](./llm-providers.md) — Configure different LLM backends
- [Agents](./agents.md) — Build autonomous agents
- [LCEL Pipelines](./lcel.md) — Compose pipelines with `pipe()`
- [Vector Stores](./vector-stores.md) — Store and retrieve embeddings
