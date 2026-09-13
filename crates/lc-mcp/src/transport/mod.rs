//! MCP transports (0.22.4 B1: multi-track again).
//!
//! - [`stateless`]: the 2026-07-28 self-contained HTTP POST track (no
//!   handshake; every request carries `_meta` + `Mcp-Method`/`Mcp-Name`
//!   headers). This remains the framework's extension track.
//! - [`stdio`]: the official MCP stdio transport — newline-delimited JSON-RPC
//!   over a spawned local subprocess, with the standard
//!   `initialize` → `notifications/initialized` handshake.
//! - [`streamable_http`]: the official MCP Streamable HTTP **client**
//!   transport — JSON-RPC POSTs answered by a direct JSON body or an SSE
//!   stream, with an optional server-assigned `Mcp-Session-Id` and OAuth 2.1
//!   bearer auth.
//! - `streamable_http_server` (crate-internal module): the official MCP
//!   Streamable HTTP **server** transport, backing
//!   `MCPServer::serve_streamable_http`.

pub mod stateless;
pub mod stdio;
pub mod streamable_http;
pub(crate) mod streamable_http_server;

pub use stateless::{default_meta, StatelessTransport};
pub use stdio::{StdioCommand, StdioTransport};
pub use streamable_http::StreamableHttpTransport;
