//! Client-side prompts/resources/completion entry points (T2 §4/§5).
//!
//! The framework serves `prompts/*`, `resources/*` and `completion/*` on the
//! server track, but the tool-oriented `McpToolClient` surface only exposes
//! `tools/list` / `tools/call` — a client could not list prompts, read a
//! resource, or request completions from a remote server. This module closes
//! that gap with a thin capability trait each transport client implements over
//! its existing `send(method, params)` primitive, so a peer speaks the full
//! data-plane protocol, not just tool execution.
//!
//! Every method here mirrors the server-side provider methods by name, so a
//! client can round-trip against a server that also uses this framework (B2/T5).

use async_trait::async_trait;
use serde_json::Value;

use crate::completion::{CompletionRequest, CompletionResult};
use crate::prompts::{
    GetPromptParams, GetPromptResult, ListPromptsParams, ListPromptsResult, Prompt,
};
use crate::protocol::MCPError;
use crate::resources::{
    ListResourcesParams, ListResourcesResult, ReadResourceParams, ReadResourceResult,
};
use crate::roots::ListRootsResult;
use crate::tasks::{CancelTaskParams, GetTaskParams, TaskResponse};

/// Capabilities a client exposes beyond tool execution (T2 §5).
///
/// The primitive [`send_capability`](Self::send_capability) is the only method
/// a transport must implement; every typed entry point below is a default that
/// marshals params and unmarshals the typed result. Implemented for the
/// stateless, stdio, and streamable clients.
#[async_trait]
pub trait McpCapabilityClient: Send + Sync {
    /// Sends one JSON-RPC request and returns the result payload (per-transport).
    ///
    /// Implemented by each client as a thin wrapper over its own `send`.
    async fn send_capability(&self, method: &str, params: Option<Value>)
        -> Result<Value, MCPError>;

    /// `prompts/list`: returns the server's prompt templates.
    async fn list_prompts(&self) -> Result<Vec<Prompt>, MCPError> {
        let page = self.list_prompts_page(None).await?;
        Ok(page.prompts)
    }

    /// `prompts/list` with cursor pagination (#79).
    ///
    /// Sends the opaque `cursor` param and returns the full envelope including
    /// `nextCursor`, so a caller can page through a large prompt set. `None` is
    /// the first page.
    async fn list_prompts_page(
        &self,
        cursor: Option<String>,
    ) -> Result<ListPromptsResult, MCPError> {
        let params = ListPromptsParams { cursor };
        let params = serde_json::to_value(&params).map_err(|e| {
            MCPError::new(-1, format!("failed to encode prompts/list params: {e}"))
        })?;
        let result = self.send_capability("prompts/list", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse prompt list: {e}")))
    }

    /// `prompts/get`: fetches one prompt with optional arguments.
    async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<GetPromptResult, MCPError> {
        let params = GetPromptParams {
            name: name.to_string(),
            arguments,
        };
        let params = serde_json::to_value(&params)
            .map_err(|e| MCPError::new(-1, format!("failed to encode prompts/get params: {e}")))?;
        let result = self.send_capability("prompts/get", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse prompt: {e}")))
    }

    /// `resources/list`: returns the server's resources.
    async fn list_resources(&self) -> Result<ListResourcesResult, MCPError> {
        self.list_resources_page(None).await
    }

    /// `resources/list` with cursor pagination (#79).
    ///
    /// Sends the opaque `cursor` param and returns the full envelope including
    /// `nextCursor`, so a caller can page through a large resource set. `None`
    /// is the first page.
    async fn list_resources_page(
        &self,
        cursor: Option<String>,
    ) -> Result<ListResourcesResult, MCPError> {
        let params = ListResourcesParams { cursor };
        let params = serde_json::to_value(&params).map_err(|e| {
            MCPError::new(-1, format!("failed to encode resources/list params: {e}"))
        })?;
        let result = self.send_capability("resources/list", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse resource list: {e}")))
    }

    /// `resources/read`: reads one resource by URI.
    async fn read_resource(&self, uri: &str) -> Result<ReadResourceResult, MCPError> {
        let params = ReadResourceParams {
            uri: uri.to_string(),
        };
        let params = serde_json::to_value(&params).map_err(|e| {
            MCPError::new(-1, format!("failed to encode resources/read params: {e}"))
        })?;
        let result = self.send_capability("resources/read", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse resource: {e}")))
    }

    /// `completion/complete`: requests autocomplete suggestions.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResult, MCPError> {
        let params = serde_json::to_value(&request).map_err(|e| {
            MCPError::new(
                -1,
                format!("failed to encode completion/complete params: {e}"),
            )
        })?;
        let result = self
            .send_capability("completion/complete", Some(params))
            .await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse completion: {e}")))
    }

    /// `roots/list`: returns the server's declared root directories.
    async fn list_roots(&self) -> Result<ListRootsResult, MCPError> {
        let result = self.send_capability("roots/list", None).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse roots: {e}")))
    }

    /// `tasks/get` (T3 Tasks extension): polls a durable task handle.
    async fn get_task(&self, id: &str) -> Result<TaskResponse, MCPError> {
        let params = GetTaskParams {
            id: id.to_string(),
        };
        let params = serde_json::to_value(&params)
            .map_err(|e| MCPError::new(-1, format!("failed to encode tasks/get params: {e}")))?;
        let result = self.send_capability("tasks/get", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse task: {e}")))
    }

    /// `tasks/cancel` (T3 Tasks extension): cancels a running/queued task,
    /// returning whether it was actually cancelled.
    async fn cancel_task(&self, id: &str) -> Result<bool, MCPError> {
        let params = CancelTaskParams {
            id: id.to_string(),
        };
        let params = serde_json::to_value(&params)
            .map_err(|e| MCPError::new(-1, format!("failed to encode tasks/cancel params: {e}")))?;
        let result = self.send_capability("tasks/cancel", Some(params)).await?;
        Ok(result
            .get("cancelled")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    /// `logging/setLevel` (0.27.0 #79 item ④): asks the server to emit only
    /// `notifications/message` frames at or above `level`. Answers `{}` on
    /// success. A server without a log sink answers `method_not_found`.
    async fn set_log_level(&self, level: crate::logging::LogLevel) -> Result<(), MCPError> {
        let params = serde_json::json!({ "level": level });
        self.send_capability(crate::logging::LOGGING_SETLEVEL, Some(params))
            .await?;
        Ok(())
    }
}

#[async_trait]
impl McpCapabilityClient for crate::client_stateless::StatelessMcpClient {
    async fn send_capability(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, MCPError> {
        self.send(method, params).await
    }
}

#[async_trait]
impl McpCapabilityClient for crate::client_stdio::StdioMcpClient {
    async fn send_capability(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, MCPError> {
        self.send(method, params).await
    }
}

#[async_trait]
impl McpCapabilityClient for crate::client_streamable::StreamableMcpClient {
    async fn send_capability(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, MCPError> {
        self.send(method, params).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client_stateless::StatelessMcpClient;
    use crate::completion::{CompletionArgument, CompletionRef};
    use crate::test_support::{start_fake_stateless_server, StatelessMode};

    /// Round-trip the full data-plane surface against a fake stateless server
    /// that mirrors the framework's provider-backed handlers (T2 §5).
    #[tokio::test]
    async fn client_fetches_prompts_resources_and_completions() {
        let server = start_fake_stateless_server(StatelessMode::Full).await;
        let client = StatelessMcpClient::connect(&server.url);

        let prompts = client.list_prompts().await.expect("prompts/list");
        assert!(!prompts.is_empty(), "server exposes at least one prompt");

        let got = client
            .get_prompt(&prompts[0].name, None)
            .await
            .expect("prompts/get");
        assert!(!got.messages.is_empty(), "prompt has messages");

        let res = client.list_resources().await.expect("resources/list");
        assert!(
            !res.resources.is_empty(),
            "server exposes at least one resource"
        );

        let ro = client
            .read_resource(&res.resources[0].uri)
            .await
            .expect("resources/read");
        assert!(!ro.contents.is_empty(), "resource has content");

        let comp = client
            .complete(CompletionRequest {
                reference: CompletionRef {
                    ref_type: "ref/prompt".into(),
                    uri: prompts[0].name.clone(),
                },
                argument: CompletionArgument {
                    name: "language".into(),
                    value: "ru".into(),
                },
            })
            .await
            .expect("completion/complete");
        assert!(!comp.values.is_empty(), "completion returns suggestions");

        let roots = client.list_roots().await.expect("roots/list");
        assert!(!roots.roots.is_empty(), "server exposes at least one root");
    }
}
