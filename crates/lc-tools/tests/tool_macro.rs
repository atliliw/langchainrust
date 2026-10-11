// Integration test for the #[tool] procedural macro
use lc_tools::{tool, BaseTool, Tool, ToolError};

/// A simple tool defined with the #[tool] macro.
#[tool(description = "Greets a person by name")]
fn greet(
    #[param(desc = "The name of the person to greet")] name: String,
) -> Result<String, ToolError> {
    Ok(format!("Hello, {}!", name))
}

#[tokio::test]
async fn tool_macro_generates_base_tool() {
    let tool = GreetTool::new();
    assert_eq!(tool.name(), "greet");
    assert_eq!(tool.description(), "Greets a person by name");
}

#[tokio::test]
async fn tool_macro_run_works() {
    let tool = GreetTool::new();
    let result = tool.run(r#"{"name": "Alice"}"#.to_string()).await.unwrap();
    assert!(result.contains("Hello, Alice!"));
}

#[tokio::test]
async fn tool_macro_invoke_works() {
    let tool = GreetTool::new();
    let input = GreetInput {
        name: "Bob".to_string(),
    };
    let result = Tool::invoke(&tool, input).await.unwrap();
    assert_eq!(result, "Hello, Bob!");
}

#[tokio::test]
async fn tool_macro_args_schema() {
    let tool = GreetTool::new();
    let schema = BaseTool::args_schema(&tool).unwrap();
    assert!(schema.is_object());
}

/// A tool with multiple parameters.
#[tool(description = "Adds two numbers")]
fn add_numbers(
    #[param(desc = "The first number")] a: i64,
    #[param(desc = "The second number")] b: i64,
) -> Result<i64, ToolError> {
    Ok(a + b)
}

#[tokio::test]
async fn tool_macro_multiple_params() {
    let tool = AddNumbersTool::new();
    let result = tool.run(r#"{"a": 3, "b": 4}"#.to_string()).await.unwrap();
    assert!(result.contains("7"));
}

/// A tool with an optional parameter.
#[tool(description = "Says hello with optional title")]
fn hello_optional(
    #[param(desc = "The person's name")] name: String,
    #[param(desc = "Optional title like Mr/Ms")] title: Option<String>,
) -> Result<String, ToolError> {
    match title {
        Some(t) => Ok(format!("Hello, {} {}!", t, name)),
        None => Ok(format!("Hello, {}!", name)),
    }
}

#[tokio::test]
async fn tool_macro_optional_param() {
    let tool = HelloOptionalTool::new();
    // Without title
    let result = tool.run(r#"{"name": "Alice"}"#.to_string()).await.unwrap();
    assert!(result.contains("Hello, Alice!"));
    // With title
    let result = tool
        .run(r#"{"name": "Alice", "title": "Dr"}"#.to_string())
        .await
        .unwrap();
    assert!(result.contains("Hello, Dr Alice!"));
}

/// A type that forces serialization to fail (for the F5 test): the `Serialize` impl always
/// returns an error, so at runtime `serde_json::to_string` is guaranteed to fail, without a
/// `Debug` fallback. The generated tool struct exposes `Output = FailingSerialize`, hence `pub`.
pub struct FailingSerialize;

impl serde::Serialize for FailingSerialize {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Err(serde::ser::Error::custom(
            "intentional serialization failure",
        ))
    }
}

/// F5 test: the output type's serialization must always fail. `run()` must return
/// `ExecutionFailed`, not silently fall back to Debug text fed to the LLM.
#[tool(description = "Outputs a type whose Serialize impl always fails")]
fn failing_output() -> Result<FailingSerialize, ToolError> {
    Ok(FailingSerialize)
}

#[tokio::test]
async fn tool_macro_run_errors_on_unserializable_output() {
    let tool = FailingOutputTool::new();
    let err = tool
        .run(r#"{}"#.to_string())
        .await
        .expect_err("run should fail on unserializable output");
    assert!(
        matches!(err, ToolError::ExecutionFailed(_)),
        "expected ExecutionFailed, got: {}",
        err
    );
}

/// F5 test: when the function returns `Result<_, ToolError>`, `invoke()` passes the original
/// error through directly, keeping the error type as `InvalidInput` instead of flattening it
/// into `ExecutionFailed`.
#[tool(description = "Guards against 'bad' input")]
fn guarded(#[param(desc = "The value to check")] value: String) -> Result<String, ToolError> {
    if value == "bad" {
        return Err(ToolError::InvalidInput("bad value".to_string()));
    }
    Ok(value)
}

#[tokio::test]
async fn tool_macro_invoke_passes_through_tool_error() {
    let tool = GuardedTool::new();
    let input = GuardedInput {
        value: "bad".to_string(),
    };
    let err = Tool::invoke(&tool, input)
        .await
        .expect_err("invoke should return the tool's error");
    assert!(
        matches!(err, ToolError::InvalidInput(_)),
        "expected InvalidInput passed through, got: {}",
        err
    );
}

/// An `async fn` tool — the macro must `.await` the generated call.
#[tool(description = "Computes a doubled value asynchronously")]
async fn async_double(#[param(desc = "The value to double")] value: i64) -> Result<i64, ToolError> {
    Ok(value * 2)
}

/// T2: the error type written through the `lc-tools` **re-export** path. The old
/// "last segment is `ToolError`, previous one is `tools`" heuristic did not recognise
/// `lc_tools::ToolError`, so a real `ToolError` was silently flattened into
/// `ExecutionFailed`; the path allowlist recognises it, so the variant survives.
#[tool(description = "Guards against 'bad' input, error written via the lc-tools re-export")]
fn reexport_guard(
    #[param(desc = "The value to check")] value: String,
) -> Result<String, lc_tools::ToolError> {
    if value == "bad" {
        return Err(lc_tools::ToolError::InvalidInput("bad value".to_string()));
    }
    Ok(value)
}

#[tokio::test]
async fn tool_macro_reexport_path_passes_through() {
    let tool = ReexportGuardTool::new();
    let err = tool
        .run(r#"{"value":"bad"}"#.to_string())
        .await
        .expect_err("run should return the tool's error");
    assert!(
        matches!(err, ToolError::InvalidInput(_)),
        "expected the re-exported ToolError to pass through, got: {err}"
    );
}

/// T2: an error type reachable only through an alias is outside the name allowlist, so
/// `auto` cannot know it. `#[tool(error = passthrough)]` states it explicitly instead of
/// relying on path guessing.
type AliasedError = ToolError;

#[tool(
    description = "Guards against 'bad' input via an aliased error path",
    error = passthrough
)]
fn aliased_guard(
    #[param(desc = "The value to check")] value: String,
) -> Result<String, AliasedError> {
    if value == "bad" {
        return Err(ToolError::InvalidInput("bad value".to_string()));
    }
    Ok(value)
}

#[tokio::test]
async fn tool_macro_error_passthrough_opt_in() {
    let tool = AliasedGuardTool::new();
    let err = tool
        .run(r#"{"value":"bad"}"#.to_string())
        .await
        .expect_err("run should return the tool's error");
    assert!(
        matches!(err, ToolError::InvalidInput(_)),
        "expected passthrough via the aliased path, got: {err}"
    );
}

/// T2: the reverse escape hatch — `error = wrap` forces the flattening branch even though
/// the written path is the library `ToolError`.
#[tool(description = "Forces the wrapping error mode", error = wrap)]
fn wrapped_guard(#[param(desc = "The value to check")] value: String) -> Result<String, ToolError> {
    if value == "bad" {
        return Err(ToolError::InvalidInput("bad value".to_string()));
    }
    Ok(value)
}

#[tokio::test]
async fn tool_macro_error_wrap_opt_in() {
    let tool = WrappedGuardTool::new();
    let err = tool
        .run(r#"{"value":"bad"}"#.to_string())
        .await
        .expect_err("run should return the tool's error");
    assert!(
        matches!(err, ToolError::ExecutionFailed(_)),
        "expected the variant to be flattened, got: {err}"
    );
}

#[tokio::test]
async fn tool_macro_async_fn_supported() {
    let tool = AsyncDoubleTool::new();
    // `invoke` path
    let input = AsyncDoubleInput { value: 21 };
    let out = Tool::invoke(&tool, input).await.unwrap();
    assert_eq!(out, 42);
    // `run` (string) path
    let out = tool.run(r#"{"value": 5}"#.to_string()).await.unwrap();
    assert_eq!(out, "10");
}

/// `parameters()` must expose the OpenAI-style object schema.
#[tokio::test]
async fn tool_macro_parameters_json_schema() {
    let tool = GreetTool::new();
    let params = tool.parameters().expect("structures produce a schema");
    assert_eq!(params["type"], "object");
    assert!(params.get("properties").is_some());
    assert!(params.get("required").is_some());
}

/// `describe()` returns the (name, description, args_schema) triple.
#[tokio::test]
async fn tool_macro_describe_triple() {
    let tool = GreetTool::new();
    let (name, desc, schema) = tool.describe();
    assert_eq!(name, "greet");
    assert_eq!(desc, "Greets a person by name");
    assert!(schema.is_some());
    // The triple's schema is the same source as `parameters()` and `BaseTool::args_schema`
    // (`schemars::schema_for!(Input)`), so a registry can register a tool from any of them
    // without the three drifting apart.
    assert_eq!(schema, tool.parameters());
    assert_eq!(schema, BaseTool::args_schema(&tool));
}
