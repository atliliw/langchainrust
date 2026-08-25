//! 流式工具输出(P2-9):长任务工具"边跑边推"增量片段。
//!
//! 服务器把长任务的部分结果拆成多个片段,经 `notifications/tool_partial`
//! 推送;客户端 [`subscribe_tool_stream`](crate::MCPClient::subscribe_tool_stream)
//! 订阅后按序接收增量,直至收到 `final: true` 的片段。
//!
//! 配合 P1-7 多类型内容:每个片段携带独立的 [`MCPContent`](crate::MCPContent)
//! (文本 / 图片 / 资源),`render_text` 统一渲染,非文本内容以占位描述代表。
//!
//! ## 推送格式(`notifications/tool_partial` 的 params)
//!
//! ```json
//! {
//!   "tool": "read_file",
//!   "seq": 0,
//!   "progress": 0.33,
//!   "content": { "type": "text", "text": "第一段" },
//!   "final": false
//! }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use tokio::sync::broadcast;
use tokio::time::{timeout, Duration};

use super::types::MCPContent;

/// 流式工具输出的一个增量片段(P2-9)。
///
/// 字段对应 `notifications/tool_partial` 推送的 params;`seq` 单调递增,
/// 供排序 / 去重 / 断点续传;`is_final` 标记最后一个片段,`collect` 据此终止。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialContent {
    /// 所属工具名。
    pub tool: String,
    /// 片段序号(单调递增,0 起)。
    pub seq: u64,
    /// 片段内容(P1-7 多类型:文本 / 图片 / 资源)。
    pub content: MCPContent,
    /// 可选进度(0.0~1.0)。
    pub progress: Option<f32>,
    /// 是否最后一个片段(服务器推送 `final: true`)。
    #[serde(rename = "final", default)]
    pub is_final: bool,
}

impl PartialContent {
    /// 渲染本片段为文本(配合 P1-7):文本原样;图片 / 资源用占位描述代表。
    pub fn render_text(&self) -> String {
        self.content.render_text()
    }
}

/// 流式订阅的错误类别(P2-9)。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolStreamError {
    /// 广播缓冲区积压导致丢帧(推送太快,消费不及)。
    Lagged,
    /// `collect` 在限定时间内未收到 `final` 片段。
    Timeout,
}

impl fmt::Display for ToolStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolStreamError::Lagged => write!(
                f,
                "tool stream lagged: chunks were pushed too fast and buffered increments were dropped"
            ),
            ToolStreamError::Timeout => write!(f, "tool stream timed out: no final chunk received within the deadline"),
        }
    }
}

impl std::error::Error for ToolStreamError {}

/// 某个工具的流式增量订阅(P2-9)。
///
/// 由 [`MCPClient::subscribe_tool_stream`](crate::MCPClient::subscribe_tool_stream)
/// 创建;只投递属于本工具名的增量,其他工具的推送被过滤。
///
/// 广播通道只投递给"订阅时刻之后"的推送,因此应**先订阅再调用工具**。
pub struct ToolStream {
    rx: broadcast::Receiver<PartialContent>,
    tool: String,
}

impl ToolStream {
    pub(crate) fn new(rx: broadcast::Receiver<PartialContent>, tool: String) -> Self {
        Self { rx, tool }
    }

    /// 等待下一个属于本工具的增量片段。
    ///
    /// - `Ok(Some(chunk))` — 收到一个增量;
    /// - `Ok(None)` — 通道已关闭(连接关闭),流结束;
    /// - `Err(Lagged)` — 推送过快导致丢帧,增量不连续。
    pub async fn next(&mut self) -> Result<Option<PartialContent>, ToolStreamError> {
        loop {
            match self.rx.recv().await {
                Ok(c) if c.tool == self.tool => return Ok(Some(c)),
                Ok(_) => continue, // 其他工具的推送,过滤
                Err(broadcast::error::RecvError::Lagged(_)) => return Err(ToolStreamError::Lagged),
                Err(broadcast::error::RecvError::Closed) => return Ok(None),
            }
        }
    }

    /// 收集到 `final` 片段为止的全部增量;返回的最后一个片段 `is_final == true`。
    ///
    /// 若流在收到 final 前关闭,返回已收集的片段(不报错);若 `deadline`
    /// 期限内未收到 final,返回 [`ToolStreamError::Timeout`]。
    pub async fn collect(
        &mut self,
        deadline: Duration,
    ) -> Result<Vec<PartialContent>, ToolStreamError> {
        timeout(deadline, async {
            let mut out = Vec::new();
            while let Some(c) = self.next().await? {
                let is_final = c.is_final;
                out.push(c);
                if is_final {
                    break;
                }
            }
            Ok(out)
        })
        .await
        .map_err(|_| ToolStreamError::Timeout)?
    }
}

/// 把 `notifications/tool_partial` 的 params 解析为 [`PartialContent`]。
///
/// 客户端事件监听器复用;字段缺失 / 内容不可解析返回 `None`(静默丢弃,
/// 不因一个畸形片段打断流)。
pub(crate) fn parse_partial_notification(params: Option<Value>) -> Option<PartialContent> {
    let p = params?;
    let tool = p.get("tool")?.as_str()?.to_string();
    let seq = p.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let progress = p.get("progress").and_then(|v| v.as_f64()).map(|f| f as f32);
    let content: MCPContent = serde_json::from_value(p.get("content")?.clone()).ok()?;
    let is_final = p.get("final").and_then(|v| v.as_bool()).unwrap_or(false);
    Some(PartialContent {
        tool,
        seq,
        content,
        progress,
        is_final,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::PostMode;
    use crate::{MCPClient, MCPConfig, MCPServer};
    use lc_core::tools::ToolError;
    use lc_core::BaseTool;
    use std::sync::Arc;

    /// 测试用工具:回显输入(供 in-memory 端到端流式测试)。
    struct EchoTool;
    #[async_trait::async_trait]
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "回显输入"
        }
        async fn run(&self, input: String) -> Result<String, ToolError> {
            Ok(input)
        }
    }

    fn chunk(tool: &str, seq: u64, text: &str, is_final: bool) -> PartialContent {
        PartialContent {
            tool: tool.to_string(),
            seq,
            content: MCPContent::Text {
                text: text.to_string(),
            },
            progress: Some(seq as f32 + 1.0),
            is_final,
        }
    }

    /// params 按 wire 格式序列化后解析回原值(field ↔ JSON `final` 互转)。
    #[test]
    fn test_parse_partial_notification_roundtrip() {
        let c = PartialContent {
            tool: "read_file".to_string(),
            seq: 2,
            content: MCPContent::Text {
                text: "第二段".to_string(),
            },
            progress: Some(0.66),
            is_final: true,
        };
        let json = serde_json::to_value(&c).unwrap();
        let parsed = parse_partial_notification(Some(json))
            .expect("should parse back to the original value");
        assert_eq!(parsed.tool, "read_file");
        assert_eq!(parsed.seq, 2);
        assert_eq!(parsed.render_text(), "第二段");
        assert_eq!(parsed.progress, Some(0.66));
        assert!(parsed.is_final);
    }

    /// 畸形 params(缺字段 / content 不可解析)→ None,不 panic。
    #[test]
    fn test_parse_partial_notification_malformed_is_none() {
        assert!(parse_partial_notification(None).is_none());
        assert!(parse_partial_notification(Some(serde_json::json!({}))).is_none());
        assert!(parse_partial_notification(Some(serde_json::json!({
            "tool": "x", "seq": 0, "content": "not-an-object"
        })))
        .is_none());
    }

    /// 过滤:只投递本工具名的增量,其他工具的推送被跳过。
    #[tokio::test]
    async fn test_stream_filters_by_tool() {
        let server = Arc::new(MCPServer::new().with_tool(Arc::new(EchoTool)));
        let client =
            MCPClient::with_transport(Box::new(crate::InMemoryTransport::new(server.clone())))
                .await
                .expect("in-memory connection should succeed");
        let mut stream = client.subscribe_tool_stream("echo");

        // 先推一个"其他工具"的片段:应被过滤,collect 不受影响。
        server.publish_partial(chunk("other", 0, "无关片段", false));
        server.publish_partial(chunk("echo", 0, "第一段", false));
        server.publish_partial(chunk("echo", 1, "第二段", true));

        let chunks = stream
            .collect(Duration::from_secs(2))
            .await
            .expect("should receive increments");
        assert_eq!(
            chunks.len(),
            2,
            "chunks from other tools should be filtered"
        );
        assert_eq!(chunks[0].render_text(), "第一段");
        assert_eq!(chunks[0].progress, Some(1.0));
        assert!(!chunks[0].is_final);
        assert!(chunks[1].is_final, "collect should end with a final chunk");
    }

    /// 多类型内容(P1-7):片段携带图片内容时,render_text 以占位描述代表。
    #[tokio::test]
    async fn test_partial_multi_type_content_renders_placeholder() {
        let server = Arc::new(MCPServer::new().with_tool(Arc::new(EchoTool)));
        let client =
            MCPClient::with_transport(Box::new(crate::InMemoryTransport::new(server.clone())))
                .await
                .expect("in-memory connection should succeed");
        let mut stream = client.subscribe_tool_stream("echo");

        server.publish_partial(PartialContent {
            tool: "echo".to_string(),
            seq: 0,
            content: MCPContent::Image {
                data: "base64...".to_string(),
                mime_type: "image/png".to_string(),
            },
            progress: None,
            is_final: true,
        });
        let chunks = stream
            .collect(Duration::from_secs(2))
            .await
            .expect("should receive increments");
        assert_eq!(chunks.len(), 1);
        assert!(
            chunks[0].render_text().contains("[image: image/png"),
            "{}",
            chunks[0].render_text()
        );
    }

    /// collect 超时:未在期限内收到 final 片段 → ToolStreamError::Timeout。
    #[tokio::test]
    async fn test_collect_times_out_without_final() {
        let server = Arc::new(MCPServer::new().with_tool(Arc::new(EchoTool)));
        let client =
            MCPClient::with_transport(Box::new(crate::InMemoryTransport::new(server.clone())))
                .await
                .expect("in-memory connection should succeed");
        let mut stream = client.subscribe_tool_stream("echo");

        // 只推非 final 片段,永远等不到收尾 → 超时。
        server.publish_partial(chunk("echo", 0, "卡住", false));
        let err = stream
            .collect(Duration::from_millis(100))
            .await
            .expect_err("should time out");
        assert_eq!(err, ToolStreamError::Timeout);
    }

    /// SSE 路径端到端:StreamingCall 假服务器在首次 tools/call 后沿 SSE
    /// 长连接推送 3 个增量片段,客户端 subscribe 全部收到。
    #[tokio::test]
    async fn test_subscribe_collects_partials_via_sse() {
        let fake = crate::test_support::start_fake_sse_server(PostMode::StreamingCall).await;
        let client = MCPClient::connect(MCPConfig::sse(&fake.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");
        let mut stream = client.subscribe_tool_stream("echo");

        // call_seen 门控:触发服务器开始沿 SSE 推流(见 test_support)。
        let out = client
            .call_tool("echo", serde_json::json!({"msg": "hi"}))
            .await;
        assert!(out.is_ok(), "normal call should still succeed");

        let chunks = stream
            .collect(Duration::from_secs(5))
            .await
            .expect("should receive streaming increments");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].render_text(), "chunk0");
        assert_eq!(chunks[0].progress, Some(1.0 / 3.0));
        assert_eq!(chunks[1].render_text(), "chunk1");
        assert_eq!(chunks[2].render_text(), "chunk2");
        assert!(chunks[2].is_final, "last chunk should be marked final");
    }
}
