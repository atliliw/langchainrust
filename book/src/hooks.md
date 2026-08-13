# Hooks

Hooks provide a composable middleware mechanism for intercepting agent lifecycle events: completions, tool calls, streaming, and errors.

## AgentHook Trait

```rust
#[async_trait]
pub trait AgentHook: Send + Sync {
    fn on_before_completion(&self, ctx: &mut CompletionContext) -> CompletionAction;
    fn on_after_completion(&self, ctx: &mut CompletionResult) -> Result<(), HookError>;
    fn on_before_tool_call(&self, ctx: &mut ToolCallContext) -> ToolCallAction;
    fn on_after_tool_call(&self, ctx: &mut ToolResultContext) -> Result<(), HookError>;
    fn on_stream_chunk(&self, chunk: &str) -> StreamAction;
    fn on_agent_start(&self, input: &str) -> Result<(), HookError>;
    fn on_agent_end(&self, output: &str) -> Result<(), HookError>;
    fn on_error(&self, error: &HookError) -> ErrorAction;
}
```

All methods have default no-op implementations -- implement only what you need.

## Built-in Hooks

| Hook | Struct | Purpose |
|------|--------|---------|
| Approval | `ApprovalHook` | Require manual approval before tool calls |
| Content Filter | `ContentFilterHook` | Filter sensitive words from stream output |
| Logging | `LoggingHook` | Log all lifecycle events |
| Prompt Injection | `PromptInjectionHook` | Detect & sanitize prompt injection in tool outputs |
| Token Budget | `TokenBudgetHook` | Enforce token budget / LLM call quota (rate limiting) |

## ApprovalHook

```rust
use langchainrust::hooks::ApprovalHook;

// Manual approval (default) -- rejects tool calls
let hook = ApprovalHook::new();

// Auto-approve all tool calls
let hook = ApprovalHook::auto_approve();

// Configurable
let hook = ApprovalHook::new().with_auto_approve(false);
```

## ContentFilterHook

```rust
use langchainrust::hooks::ContentFilterHook;

let hook = ContentFilterHook::new(vec![
    "password".into(), "secret".into(), "internal".into(),
])
.with_placeholder("[REDACTED]")  // Replace sensitive words
.with_drop_token(false);         // If true, filter entire chunk instead of replacing
```

## PromptInjectionHook

Detects and sanitizes prompt-injection text in tool outputs. A malicious page /
retrieved document that says "ignore previous instructions" is replaced with a
safe marker before it can reach the next `plan()` prompt, breaking cross-round
pollution.

```rust
use langchainrust::hooks::PromptInjectionHook;

let hook = PromptInjectionHook::new();   // Default injection patterns

// Custom patterns and marker
let hook = PromptInjectionHook::new()
    .with_patterns(vec!["evil-text".into()])
    .with_marker("[BLOCKED:{}]");        // `{}` is replaced with the matched pattern

let detected = hook.detected_count();    // Total injections sanitized so far
```

## TokenBudgetHook

Enforces a token budget and/or an LLM call quota before each completion.
`on_before_completion` rejects (aborting the run) when the estimated usage would
exceed the budget; `on_after_completion` accumulates real token usage.

```rust
use langchainrust::hooks::TokenBudgetHook;

let hook = TokenBudgetHook::new(10_000) // Total token budget for the run
    .with_max_calls(20);                // Optional: max LLM calls
```

## LoggingHook

```rust
use langchainrust::hooks::LoggingHook;

let hook = LoggingHook::new();           // Basic logging
let hook = LoggingHook::with_tokens();   // Include token usage logging
```

## Adding Hooks to Agent

```rust
use langchainrust::{AgentExecutor, BaseAgent};
use langchainrust::hooks::{ApprovalHook, ContentFilterHook, LoggingHook};

let executor = AgentExecutor::new(agent, tools)
    .hook(ApprovalHook::new())                                          // Require approval
    .hook(ContentFilterHook::new(vec!["secret".into()]))                // Filter words
    .hook(LoggingHook::with_tokens());                                   // Log events
```

## Tool Permission Policy

Beyond hooks, `AgentExecutor` can enforce a tool permission policy before every
tool execution — two gates:

1. **Permission tier** (`ToolRisk`): `Safe` < `Standard` < `Dangerous`. A tool
   whose risk exceeds `max_permitted` is rejected.
2. **Sandbox gate**: a `Dangerous` tool must be declared as having been moved
   into a sandboxed environment (`ToolPolicy::sandboxed`) or it is rejected,
   preventing dangerous tools from running un-sandboxed. (Actual sandbox
   backends live in `lc-tools` — `SandboxTool` / `CodeSandbox` — which the
   operator wires onto the tool.)

```rust
use langchainrust::{AgentExecutor, ToolPolicy, ToolRisk};

let policy = ToolPolicy::new()
    .risk("code_interpreter", ToolRisk::Dangerous) // Declare risk tier
    .sandboxed("code_interpreter")                // Declare it runs in a sandbox
    .with_max_permitted(ToolRisk::Standard);      // Gate: reject riskier tools

let executor = AgentExecutor::new(agent, tools).with_tool_policy(policy);
```

A rejected tool call aborts the run with an `AgentError`. When no policy is
configured, tool execution is unrestricted.

## Action Types

| Action | Hook | Variants |
|--------|------|----------|
| `CompletionAction` | `on_before_completion` | `Continue`, `Modify { messages }`, `Reject { reason }` |
| `ToolCallAction` | `on_before_tool_call` | `Continue`, `Modify { name, arguments }`, `Reject { reason }`, `Skip` |
| `StreamAction` | `on_stream_chunk` | `Forward(String)`, `Filter`, `Replace(String)` |
| `ErrorAction` | `on_error` | `Propagate`, `Retry`, `Ignore` |

## Custom Hook

```rust
use langchainrust::hooks::{AgentHook, ToolCallAction, ToolCallContext, StreamAction};

struct RateLimitHook { max_calls: usize, calls: AtomicUsize }

#[async_trait]
impl AgentHook for RateLimitHook {
    fn on_before_tool_call(&self, _ctx: &mut ToolCallContext) -> ToolCallAction {
        if self.calls.fetch_add(1, Ordering::SeqCst) >= self.max_calls {
            ToolCallAction::Reject { reason: "Rate limit exceeded".into() }
        } else {
            ToolCallAction::Continue
        }
    }
}
```
