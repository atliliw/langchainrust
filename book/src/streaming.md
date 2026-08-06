# Streaming

LangChainRust supports streaming at multiple levels: LLM token streaming, agent event streaming, chain streaming, and LCEL pipeline streaming.

## Streaming Levels

| Level | Method | Output |
|-------|--------|--------|
| LLM | `stream_chat()` | Token-by-token text |
| Agent | `AgentExecutor::stream()` | `AgentStreamEvent` stream |
| Chain | `BaseChain::stream()` | `StreamToken` stream |
| LCEL | `Runnable::stream()` / `transform()` | `LcelStreamEvent` stream |
| Structured Output | `stream_structured_output()` | Partial `T` values |

## LLM Streaming

```rust
use langchainrust::openai::OpenAIChat;

let llm = OpenAIChat::new(OpenAIConfig { streaming: true, ..Default::default() });
let messages = vec![
    Message::system("You are a helpful assistant."),
    Message::human("Explain Rust ownership."),
];

let mut stream = llm.stream_chat(messages, None).await?;
while let Some(chunk) = stream.next().await {
    let token = chunk?;
    print!("{}", token.content);
}
```

## Agent Streaming

```rust
use langchainrust::AgentExecutor;

let stream = executor.stream(input);
while let Some(event) = stream.next().await {
    match event? {
        AgentStreamEvent::Text { content } => print!("{}", content),
        AgentStreamEvent::ToolStart { name, input } => {
            println!("[Tool: {}] Input: {}", name, input);
        }
        AgentStreamEvent::ToolEnd { name, output } => {
            println!("[Tool: {}] Output: {}", name, output);
        }
        AgentStreamEvent::FinalAnswer { content } => {
            println!("Answer: {}", content);
        }
        AgentStreamEvent::PipelineStep { step, detail } => { /* ... */ }
        AgentStreamEvent::Error { message } => eprintln!("Error: {}", message),
    }
}
```

## Chain Streaming

```rust
use langchainrust::{LLMChain, StreamToken};

let chain = LLMChain::new(llm, "Explain: {topic}");
let mut stream = chain.stream(inputs).await?;
while let Some(token) = stream.next().await {
    match token? {
        StreamToken { token, is_final: false } => print!("{}", token),
        StreamToken { token, is_final: true } => println!("\n[Done]"),
    }
}
```

## LCEL Streaming

```rust
use langchainrust::LcelStreamEvent;

let stream = pipeline.stream(input, None).await?;
while let Some(event) = stream.next().await {
    match event? {
        LcelStreamEvent::OnLlmStream { token } => print!("{}", token),
        LcelStreamEvent::OnToolEnd { output } => println!("Tool: {}", output),
        LcelStreamEvent::OnChainEnd { output } => println!("Chain: {:?}", output),
        _ => {}
    }
}
```

## Structured Output Streaming

```rust
use langchainrust::StreamingStructuredOutputExt;

let stream = llm.stream_structured_output::<MyStruct>(schema, "prompt").await?;
while let Some(partial) = stream.next().await {
    let partial: MyStruct = partial?;
    // Fields populate incrementally as tokens arrive
}
```
