# MCP Protocol

The Model Context Protocol (MCP) is Anthropic's open standard for connecting LLM applications to external tools and data sources. LangChainRust provides both a client and server implementation.

## Feature Overview

| Feature | Type | Description |
|---------|------|-------------|
| `MCPClient` | Client | Connect to any MCP server, list/call tools |
| `MCPServer` | Server | Expose `BaseTool` implementations via MCP |
| `MCPToolAdapter` | Adapter | Wrap MCP tools as `BaseTool` for agent use |
| `StdioTransport` | Transport | Child process stdin/stdout JSON-RPC |
| `SseTransport` | Transport | HTTP SSE + POST JSON-RPC |
| `MCPConfig` | Config | `Stdio` or `Sse` connection configuration |

## MCP Client

```rust
use langchainrust::{MCPClient, MCPConfig};

// Connect via stdio (spawn a child process)
let config = MCPConfig::stdio(
    "npx",
    vec!["@anthropic/mcp-server-filesystem".to_string(), "/tmp".to_string()],
);
let client = MCPClient::connect(config).await?;

// List available tools
let tools = client.list_tools().await?;
for tool in &tools {
    println!("{}: {}", tool.name, tool.description);
}

// Call a tool
let result = client.call_tool("read_file", serde_json::json!({"path": "/tmp/hello.txt"})).await?;
println!("{}", result.text());

// Convert all MCP tools to BaseTool for agent use
let base_tools: Vec<Arc<dyn BaseTool>> = client.as_tools().await;
```

## MCP Server

```rust
use langchainrust::{MCPServer, BaseTool, Calculator};
use std::sync::Arc;

let server = MCPServer::new()
    .with_tool(Arc::new(Calculator::new()) as Arc<dyn BaseTool>)
    .with_tool(Arc::new(DateTimeTool::new()) as Arc<dyn BaseTool>)
    .with_server_info("my-mcp-server", "1.0.0");

// Serve via stdio (for use by MCP clients like Claude Desktop)
server.serve_stdio().await?;
```

## SSE Transport

```rust
use langchainrust::MCPConfig;

// Connect to a remote MCP server via SSE
let config = MCPConfig::sse("http://localhost:3000/sse");
let client = MCPClient::connect(config).await?;
let tools = client.list_tools().await?;
```

## Protocol Details

- **Version**: `2024-11-05`
- **Format**: JSON-RPC 2.0 over stdio or SSE
- **Handshake**: Client sends `initialize`, server responds with capabilities
- **Methods**: `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`, `completion/complete`, `sampling/createMessage`

## Sub-Protocol Support

| Sub-protocol | Client | Server |
|-------------|--------|--------|
| Tools | `list_tools`, `call_tool` | `serve_stdio` |
| Resources | `list_resources`, `read_resource` | -- |
| Prompts | `list_prompts`, `get_prompt` | -- |
| Sampling | `create_message` | -- |
| Completion | `complete` | -- |
| Roots | `list_roots` | -- |
| Elicitation | `elicit` | -- |
