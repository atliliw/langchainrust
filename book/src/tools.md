# Tools

LangChainRust ships with 12+ built-in tools and a `#[tool]` macro for custom tools.

## Built-in Tools

| Tool | Struct | Name | Key Input |
|------|--------|------|-----------|
| Calculator | `Calculator` | `calculator` | `expression: String` |
| DateTime | `DateTimeTool` | `datetime` | `operation`, `datetime`, `unit` |
| Math | `SimpleMathTool` | `math` | `operation`, `value`, `value2` |
| Python REPL | `PythonREPLTool` | `python_repl` | `code: String` |
| DuckDuckGo Search | `DuckDuckGoSearchTool` | `web_search` | `query: String` |
| URL Fetch | `URLFetchTool` | `url_fetch` | `url`, `operation` |
| Wikipedia | `WikipediaTool` | `wikipedia` | `query`, `lang` |
| File | `FileTool` | `file_operation` | `op`, `path`, `content` |
| HTTP | `HTTPTool` | `http_request` | `url`, `method`, `body` |
| SQL | `SQLTool` | `sql_query` | SQL string |
| Computer Use | `ComputerUseTool` | `computer_use` | `action`, `coordinate` |
| Sandbox | `SandboxTool<S>` | `code_interpreter` | `code: String` |

## Using Tools with Agents

```rust
use langchainrust::tools::{Calculator, DateTimeTool, SimpleMathTool};
use langchainrust::{FunctionCallingAgent, AgentExecutor, BaseAgent};
use std::sync::Arc;

let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(DateTimeTool::new()),
    Arc::new(SimpleMathTool::new()),
];

let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
    .with_max_iterations(5);

let result = executor.invoke("Calculate 15 * 4 and tell me the time".to_string()).await?;
```

## Security Features

**HTTPTool**: SSRF protection by default -- blocks private IPs (127.0.0.1, 10.x, 192.168.x, cloud metadata).

**FileTool**: Path traversal prevention, extension whitelist, size limits.

**SQLTool**: SELECT-only, blocks SQL comments and dangerous patterns (sleep, benchmark, exec).

**PythonREPLTool**: Disabled by default; blocks dangerous imports (os, subprocess, socket); configurable timeout.

## Sandbox

```rust
use langchainrust::tools::sandbox::{LocalSandbox, CodeSandbox, Language, SandboxTool};

let sandbox = LocalSandbox::new();
let result = sandbox.run("print(2 + 2)", Language::Python, 10_000).await?;
// result.stdout, result.stderr, result.exit_code

let tool = SandboxTool::new(LocalSandbox::new(), Language::Python).with_timeout(10_000);
```

## Computer Use

```rust
use langchainrust::tools::ComputerUseTool;

let tool = ComputerUseTool::new_anthropic("sk-...", 1920, 1080);
// Actions: screenshot, click, type, scroll, key_press, wait
```
