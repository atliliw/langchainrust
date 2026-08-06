# Tool Macro

The `#[tool]` procedural macro auto-generates `BaseTool` + `Tool` implementations from a simple function, reducing ~50 lines of boilerplate to 3 lines.

## Basic Usage

```rust
use langchainrust::{tool, BaseTool, Tool, ToolError};

#[tool(description = "Greets a person by name")]
fn greet(
    #[param(desc = "The name of the person to greet")]
    name: String,
) -> Result<String, ToolError> {
    Ok(format!("Hello, {}!", name))
}
```

This generates:
- `GreetTool` struct (with `new()` and `Default`)
- `GreetInput` struct (with `Deserialize` + `JsonSchema`)
- `impl Tool for GreetTool` (type-safe `invoke`)
- `impl BaseTool for GreetTool` (string-based `run`, `name`, `description`, `args_schema`)
- The original `greet` function is preserved

## Parameters

```rust
#[tool(description = "Adds two numbers")]
fn add_numbers(
    #[param(desc = "The first number")]
    a: i64,
    #[param(desc = "The second number")]
    b: i64,
) -> Result<i64, ToolError> {
    Ok(a + b)
}
// Generates: AddNumbersTool, AddNumbersInput
```

Parameter type rules:

| Type | Schema | Required |
|------|--------|----------|
| `String`, `i64`, `f64`, `bool` | Primitive | Yes |
| `Option<T>` | Nullable | No |
| `Vec<T>` | Array | Yes |

## Optional Parameters

```rust
#[tool(description = "Says hello with optional title")]
fn hello_optional(
    #[param(desc = "The person's name")]
    name: String,
    #[param(desc = "Optional title (e.g., Dr., Mr.)")]
    title: Option<String>,
) -> Result<String, ToolError> {
    match title {
        Some(t) => Ok(format!("Hello, {} {}!", t, name)),
        None => Ok(format!("Hello, {}!", name)),
    }
}
// Generates: HelloOptionalTool, HelloOptionalInput
```

## Using Generated Tools

```rust
// Type-safe invocation via Tool trait
let tool = GreetTool::new();
let input = GreetInput { name: "Alice".into() };
let result = tool.invoke(input).await?;
assert_eq!(result, "Hello, Alice!");

// String-based invocation via BaseTool trait (for agents)
let tool = GreetTool::new();
assert_eq!(tool.name(), "greet");
assert_eq!(tool.description(), "Greets a person by name");
let result = tool.run(r#"{"name":"Bob"}"#.to_string()).await?;
assert_eq!(result, "\"Hello, Bob!\"");

// Get JSON schema for LLM function calling
let schema = tool.args_schema(); // Auto-generated from JsonSchema
```

## Attributes Reference

| Attribute | Target | Required | Description |
|-----------|--------|----------|-------------|
| `#[tool(description = "...")]` | Function | Yes | Tool description shown to the LLM |
| `#[param(desc = "...")]` | Parameter | No | Parameter description in JSON schema |

The struct and input types are named by converting the function name to PascalCase and appending `Tool` or `Input` (e.g., `add_numbers` becomes `AddNumbersTool` / `AddNumbersInput`).
