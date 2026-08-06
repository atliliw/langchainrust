# LCEL Pipelines

The LangChain Expression Language (LCEL) provides pipe-based composition for `Runnable` components, inspired by Python LangChain's `prompt | llm | parser` syntax.

## Core Types

| Type | Description |
|------|-------------|
| `Runnable<I, O>` | Base execution trait (invoke, batch, stream, transform) |
| `RunnableExt` | Extension providing `pipe()` composition |
| `RunnableSequence<I, O>` | Ordered pipeline of chained steps |
| `RunnableLambda<I, O>` | Closure wrapper (sync/async) |
| `RunnablePassthrough<I>` | Identity pass-through with true streaming |
| `RunnableParallel<I>` | Fan-out/fan-in, results in `HashMap<String, Value>` |
| `RunnableBranch<I, O>` | Conditional routing (first match wins) |
| `RunnableBinding<I, O>` | Pre-bind config/kwargs to a Runnable |
| `RunnableWithFallbacks<I, O>` | Primary + fallback chain |
| `RunnableAssign` | Inject key-value pairs into HashMap mid-pipeline |
| `RunnableRetry<I, O>` | Retry with exponential backoff |

## Pipe Composition

```rust
use langchainrust::{RunnableExt, RunnableLambda, RunnablePassthrough};

// Chain runnables with pipe()
let chain = prompt.pipe(llm).pipe(parser);

// Lambda steps
let step1 = RunnableLambda::new_sync(|input: String| input.to_uppercase());
let step2 = RunnableLambda::new_sync(|input: String| format!("Result: {}", input));
let pipeline = step1.pipe(step2);

let result = pipeline.invoke("hello".to_string(), None).await?;
// "Result: HELLO"
```

## Parallel & Branch

```rust
use langchainrust::{RunnableParallel, RunnableBranch, RunnablePassthrough};

// Parallel fan-out
let parallel = RunnableParallel::<String>::new()
    .with("upper", RunnableLambda::new_sync(|s: String| s.to_uppercase()))
    .with("len", RunnableLambda::new_sync(|s: String| s.len().to_string()));

let result = parallel.invoke("hello".to_string(), None).await?;
// HashMap {"upper": "HELLO", "len": "5"}

// Branch (conditional routing)
let branch = RunnableBranch::new(default_step)
    .when(condition_runnable, branch_a)
    .when(condition_runnable, branch_b);
```

## Fallbacks & Retry

```rust
use langchainrust::{RetryConfig, RetryOn};

// Fallbacks
let chain = primary_llm.with_fallbacks(vec![fallback_llm1, fallback_llm2]);

// Retry with exponential backoff
let retry_config = RetryConfig::new(3)
    .with_initial_delay(Duration::from_millis(500))
    .with_max_delay(Duration::from_secs(10))
    .with_backoff_multiplier(2.0)
    .with_retry_on(RetryOn::TransientErrors); // 429, 500, 502, 503, 504

let chain = llm.with_retry(retry_config);
```

## Assign & Binding

```rust
use langchainrust::RunnableAssign;

// Inject context into HashMap pipeline
let assign = RunnableAssign::new()
    .with("context", retriever_step);

// Bind config
let bound = llm.pipe(RunnableConfig::new().with_tag("production"));
```

## Stream Events

```rust
use langchainrust::LcelStreamEvent;

// LCEL streaming via transform()
let stream = chain.transform(input_stream, config).await?;
while let Some(event) = stream.next().await {
    match event? {
        LcelStreamEvent::OnLlmStart { .. } => { /* ... */ }
        LcelStreamEvent::OnLlmStream { token } => print!("{}", token),
        LcelStreamEvent::OnLlmEnd { .. } => { /* ... */ }
        LcelStreamEvent::OnToolEnd { output } => { /* ... */ }
        LcelStreamEvent::OnChainEnd { output } => { /* ... */ }
    }
}
```
