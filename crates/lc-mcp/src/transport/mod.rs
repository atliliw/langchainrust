//! MCP transport (0.22.0 single-track: stateless HTTP POST only).
//!
//! The 2026-07-28 spec removed the handshake / session model: every request
//! is a self-contained JSON-RPC POST tagged with `Mcp-Method` / `Mcp-Name`
//! headers. The legacy SSE / stdio transports and the handshake-based
//! `MCPClient` were removed in this version (see
//! `docs/internal/v0.22.0/MIGRATION.md`).

pub mod stateless;

pub use stateless::{default_meta, StatelessTransport};
