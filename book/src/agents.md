# Agents

Agents are autonomous systems that use LLMs to decide which actions to take.

## Agent Types

| Agent | Description | Use Case |
|-------|-------------|----------|
| ReActAgent | Reason + Act loop | General-purpose tasks |
| FunctionCallingAgent | Native tool calling | OpenAI/Anthropic tools |
| PlanExecuteAgent | Plan → Execute → Replan | Complex multi-step tasks |
| CRAG | Self-correcting RAG | Retrieval with quality control |
| AdaptiveRAG | Adaptive retrieval routing | Different retrieval strategies |
| DeepResearch | Multi-round research | Comprehensive reports |

## Basic Usage

```rust
use langchainrust::agents::{ReActAgent, AgentExecutor, BaseAgent};

let agent = ReActAgent::new(llm, prompt_template);
let executor = AgentExecutor::new(Arc::new(agent), tools)
    .with_max_iterations(10);

let result = executor.invoke("What is 2+2?".to_string()).await?;
```

## Hooks

Add lifecycle hooks for approval, content filtering, and logging:

```rust
use langchainrust::hooks::{ApprovalHook, ContentFilterHook, LoggingHook};

let executor = AgentExecutor::new(agent, tools)
    .hook(ApprovalHook::new())                              // Require approval
    .hook(ContentFilterHook::new(vec!["secret".into()]))    // Filter words
    .hook(LoggingHook::new());                               // Log events
```

## Streaming

```rust
let stream = executor.stream(input);
while let Some(event) = stream.next().await {
    match event? {
        AgentStreamEvent::ToolStart { name, input } => { /* ... */ }
        AgentStreamEvent::ToolEnd { name, output } => { /* ... */ }
        AgentStreamEvent::FinalAnswer { content } => { /* ... */ }
        AgentStreamEvent::PipelineStep { step, detail } => { /* ... */ }
        AgentStreamEvent::Error { message } => { /* ... */ }
    }
}
```

## Cancellation

```rust
use langchainrust::runnables::CancellationToken;

let token = CancellationToken::new();
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(30)).await;
    token.cancel();
});

let result = executor.invoke_with_config(input, Some(
    RunnableConfig::new().with_cancellation_token(token)
)).await;
```
