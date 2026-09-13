//! Streaming Tool Calls
//!
//! Provides a streaming agent that emits LLM text token by token and exposes
//! tool-call state.
//!
//! B8 (v0.22.4): [`sse`] maps [`AgentStreamEvent`] to Server-Sent Events
//! frames; the `sse-server` feature serves them at `text/event-stream` over
//! axum 0.7 (`POST|GET /agent/stream`).

pub mod sse;
pub mod state;
pub mod tool_call_stream;

pub use state::{AgentStreamEvent, ToolCallState};
pub use tool_call_stream::StreamingFunctionCallingAgent;

// Framework-neutral SSE surface (always available); axum serving types are
// re-exported only under the `sse-server` feature.
pub use sse::{
    agent_sse_frames, encode_sse_frame, sse_event_name, sse_event_payload, AgentEventStream,
    AgentSseRequest, SseFrame, SseOptions,
};

#[cfg(feature = "sse-server")]
pub use sse::{
    agent_sse_get_handler, agent_sse_handler, agent_sse_router, agent_sse_router_with,
    serve_agent_sse, serve_agent_sse_on, AgentSseQuery, AgentSseServerConfig, AgentSseState,
    AgentStreamFactory, AgentStreamFuture, DEFAULT_SSE_HEARTBEAT,
};
