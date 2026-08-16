// lc-tools/src/lib.rs
//! Built-in tools for langchainrust.
//!
//! Provides a collection of ready-to-use tools for agents:
//! - `Calculator`: Math expression evaluation
//! - `DateTimeTool`: Date/time queries and calculations
//! - `SimpleMathTool`: Advanced math operations
//! - `PythonREPLTool`: Python code execution (disabled by default)
//! - `DuckDuckGoSearchTool`: Web search via DuckDuckGo
//! - `URLFetchTool`: Web page fetching and parsing
//! - `WikipediaTool`: Wikipedia search
//! - `FileTool`: Sandboxed file operations
//! - `HTTPTool`: HTTP requests with SSRF protection
//! - `SQLTool`: Read-only SQL queries (feature-gated)
//! - `ComputerUseTool`: Screen interaction via Anthropic API
//! - `SandboxTool` / `LocalSandbox`: Code execution sandbox

mod calculator;
mod datetime;
mod expr_eval;
pub mod extended;
mod math;
mod python_repl;
pub mod sandbox;
mod search;
mod ssrf;
mod url_fetch;
mod wikipedia;

pub use calculator::{Calculator, CalculatorInput, CalculatorOutput};
pub use datetime::{DateTimeInput, DateTimeOutput, DateTimeTool};
pub use extended::{
    ComputerMode, ComputerUseInput, ComputerUseOutput, ComputerUseTool, FileTool, HTTPTool,
};
pub use math::{MathInput, MathOutput, SimpleMathTool};
pub use python_repl::{PythonREPLInput, PythonREPLOutput, PythonREPLTool};
pub use sandbox::{CodeSandbox, Language, LocalSandbox, RunResult, SandboxError, SandboxTool};
pub use search::{DuckDuckGoSearchTool, SearchInput, SearchOutput};
pub use url_fetch::{URLFetchInput, URLFetchOutput, URLFetchTool};
pub use wikipedia::{WikipediaInput, WikipediaOutput, WikipediaTool};

// Re-export core tool types for convenience
pub use lc_core::tools::{BaseTool, Tool, ToolError};

// Re-export #[tool] procedural macro
pub use lc_tools_derive::tool;
