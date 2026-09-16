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
| `Supervisor` (orchestrator) | Runtime routing to named sub-agents | One objective, several specialist agents |

## Basic Usage

```rust
use langchainrust::agents::{ReActAgent, AgentExecutor, BaseAgent};

let agent = ReActAgent::new(llm, prompt_template);
let executor = AgentExecutor::new(Arc::new(agent), tools)
    .with_max_iterations(10);

let result = executor.invoke("What is 2+2?".to_string()).await?;
```

## Supervisor: Dynamic Sub-Agent Routing

Where `SequentialPipeline` runs a fixed stage list and `FanOutFanIn` broadcasts to
everyone, the `Supervisor` asks a router model **once per round** which named worker
should handle the next subtask — or whether the job is done. Each worker is a complete
independent agent behind the `Orchestrator` trait (its own executor, budget, hooks);
the worker's answer is fed back into the router's scratchpad, so the next routing
decision is based on what earlier workers actually produced.

```rust
use langchainrust::{
    Supervisor, TaskAdapter, Orchestrator, RunContext, AgentTask,
};
use std::sync::Arc;

// Any String -> String Orchestrator (an AgentExecutor, another pipeline, ...)
// becomes a worker through TaskAdapter (it receives an AgentTask, returns String).
let researcher: Arc<dyn Orchestrator<Input = AgentTask, Output = String>> =
    Arc::new(TaskAdapter::new(Arc::new(research_executor)));
let writer = Arc::new(TaskAdapter::new(Arc::new(writer_executor)));

let supervisor = Supervisor::new(
    router_llm,                                   // String -> String; prompt it to return
    vec![                                         // {"next":"<worker>","task":"..."} or
        ("researcher".to_string(), researcher),   // {"next":"FINISH","answer":"..."}
        ("writer".to_string(), writer),
    ],
    8,                                            // max delegation rounds
);

let answer = supervisor
    .run_with_context(
        AgentTask::new("Write a one-page brief on reranking in RAG"),
        &RunContext::new_random(),
    )
    .await?;
```

This is exactly **one level** of sub-agent recursion: workers are leaf orchestrators
and cannot themselves delegate. Routing is bounded by `max_rounds` (`.with_max_rounds`
adjusts it later; values below 1 clamp to 1) — if the model never emits
`SUPERVISOR_FINISH`, the run errors instead of looping forever. Decisions parse as JSON
first, with a `<<<NEXT>>>` delimiter fallback (`parse_supervisor_decision`), so weaker
models still work with an envelope prompt (`supervisor_envelope`).

## Parallel Tool Calls

When a model emits several tool calls in one turn, the executor runs them **concurrently**
— bounded by a semaphore (default cap 8) rather than spawned unbounded:

```rust
let executor = AgentExecutor::new(agent, tools)
    .with_max_concurrency(4);   // at most 4 in-flight tool calls per turn
```

Observations are zipped back to their actions in the model's call order (not completion
order), so multi-tool turns are deterministic; the same guarantee holds under streaming.

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
