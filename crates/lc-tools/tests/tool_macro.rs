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

/// 强制序列化失败的类型(F5 测试用):`Serialize` 实现恒返回错误,
/// 运行期必然触发 `serde_json::to_string` 失败,且不要求 `Debug` 兜底。
/// 生成工具结构体暴露 `Output = FailingSerialize`,故需 `pub`。
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

/// F5 测试:输出类型序列化必失败。`run()` 必须返回 `ExecutionFailed`,
/// 而不是静默回退成 Debug 文本喂给 LLM。
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

/// F5 测试:函数返回 `Result<_, ToolError>` 时,`invoke()` 直接透传原错误,
/// 错误类型保持 `InvalidInput`,而不是被压平成 `ExecutionFailed`。
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
