// Integration test for the #[tool] procedural macro
use lc_tools::{tool, BaseTool, Tool, ToolError};

/// A simple tool defined with the #[tool] macro.
#[tool(description = "Greets a person by name")]
fn greet(#[param(desc = "The name of the person to greet")] name: String) -> Result<String, ToolError> {
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
    let result = tool.run(r#"{"name": "Alice", "title": "Dr"}"#.to_string()).await.unwrap();
    assert!(result.contains("Hello, Dr Alice!"));
}
