//! Streaming Tool Calls
//!
//! Provides a streaming agent that emits LLM text token by token and exposes
//! tool-call state.

pub mod state;
pub mod tool_call_stream;

pub use state::{AgentStreamEvent, ToolCallState};
pub use tool_call_stream::StreamingFunctionCallingAgent;
