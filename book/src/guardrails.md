# Guardrails

Guardrails provide input/output validation to protect production agents from malicious input and sensitive information leakage.

## Core Concepts

| Concept | Type | Description |
|---------|------|-------------|
| `InputGuardrail` | Trait | Validates agent input before execution |
| `OutputGuardrail` | Trait | Validates agent output after execution |
| `GuardrailsConfig` | Struct | Configures input + output guardrails |
| `GuardrailRunner` | Struct | Executes guardrail validation pipeline |
| `GuardedAgent` | Struct | Agent wrapper with automatic guardrail enforcement |

## Guardrail Result

Input and output guardrails return **different** result types, so the type system
enforces that `Modify` can only be produced on the output side.

```rust
// Input side: no Modify variant — input guardrails can only Pass or Block
pub enum InputGuardrailResult {
    Pass,
    Block { reason: String },
}

// Output side: Modify is the only place rewriting is allowed
pub enum OutputGuardrailResult {
    Pass,
    Block { reason: String },
    Modify { new_value: String },
}
```

## Built-in Validators

| Validator | Side | Description |
|-----------|------|-------------|
| `MaxLengthGuardrail` | Input | Blocks if input exceeds character limit |
| `ForbiddenWordsGuardrail` | Input | Blocks on forbidden word match (case-insensitive) |
| `SensitiveInfoGuardrail` | Output | Detects API keys, emails, credit cards, sensitive keywords |

## Usage

```rust
use langchainrust::{
    GuardrailsConfig, GuardedAgent, GuardrailRunner,
    MaxLengthGuardrail, ForbiddenWordsGuardrail, SensitiveInfoGuardrail,
};
use std::sync::Arc;

// Configure guardrails
let config = GuardrailsConfig::new()
    .with_input(Arc::new(MaxLengthGuardrail::new(1000)))
    .with_input(Arc::new(ForbiddenWordsGuardrail::new(vec![
        "hack".into(), "exploit".into(),
    ])))
    .with_output(Arc::new(SensitiveInfoGuardrail::new()));

// Wrap agent with guardrails
let mut guarded = GuardedAgent::new(
    Arc::new(executor),
    config,
);

let result = guarded.invoke("What is my password?".to_string()).await?;
// Returns Err(GuardrailError::Blocked { .. }) if guardrail triggers

// Check violations
let violations = guarded.violations();
```

## SensitiveInfoGuardrail Details

Detects:
- Sensitive keywords: "password", "token", "secret", "api_key", "credential" (plus Chinese equivalents)
- OpenAI-style API keys (regex `sk-...`)
- Email addresses
- Credit card numbers (with Luhn validation)

```rust
let guardrail = SensitiveInfoGuardrail::new()
    .with_keywords(vec!["internal_key".into()]);
```

## Custom Guardrails

```rust
use langchainrust::{InputGuardrail, InputGuardrailResult};

struct NoSQLInjection;

#[async_trait]
impl InputGuardrail for NoSQLInjection {
    fn name(&self) -> &str { "no_sql_injection" }
    async fn validate(&self, input: &str) -> InputGuardrailResult {
        if input.contains("{$") || input.contains("{$gt:") {
            InputGuardrailResult::Block { reason: "Potential NoSQL injection".into() }
        } else {
            InputGuardrailResult::Pass
        }
    }
}
```

> 输入护栏返回 [`InputGuardrailResult`]（没有 `Modify` 变体），输出护栏返回
> [`OutputGuardrailResult`]（含 `Modify`）。类型系统强制"修改结果仅输出侧可产生"。
