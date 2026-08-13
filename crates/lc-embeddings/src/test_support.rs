// lc-embeddings/src/test_support.rs
//! 测试辅助：极简 HTTP stub 服务，返回 OpenAI/DeepSeek/Qwen/Cohere 兼容的
//! embeddings JSON 响应，用于不依赖真实网络的批量对齐测试（P0-1）。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// 极简 HTTP stub：前 `failures_before_success` 次请求返回 `transient_status`，
/// 之后返回 `success_status`（body 为 `success_body`）。返回 `(base_url, 已接收请求数)`。
///
/// 用于 P2-5 重试语义测试（429/5xx 重试、4xx 不重试、重试耗尽）以及
/// provider 层的重试接线测试。
pub async fn spawn_status_stub(
    transient_status: u16,
    failures_before_success: usize,
    success_status: u16,
    success_body: &str,
) -> (String, Arc<AtomicUsize>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = Arc::new(AtomicUsize::new(0));
    let r = requests.clone();
    // 转为 owned,才能随任务逃出本函数（success_body 的生命周期不满足 'static）。
    let success_body = success_body.to_string();

    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let r = r.clone();
            // 每个连接都可能用到成功 body,需 clone 进各自的 handler 任务。
            let success_body = success_body.clone();
            tokio::spawn(async move {
                // 读 header 区到 \r\n\r\n
                let mut header = Vec::new();
                let mut byte = [0u8; 1];
                while header.len() < 64 * 1024 {
                    if socket.read_exact(&mut byte).await.is_err() {
                        return;
                    }
                    header.push(byte[0]);
                    if header.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                // 读 body（丢弃,只数请求）
                let header_str = String::from_utf8_lossy(&header).to_lowercase();
                let content_length: usize = header_str
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                let mut body = vec![0u8; content_length];
                if content_length > 0 && socket.read_exact(&mut body).await.is_err() {
                    return;
                }

                let n = r.fetch_add(1, Ordering::SeqCst);
                let code = if n < failures_before_success {
                    transient_status
                } else {
                    success_status
                };
                let reason = match code {
                    429 => "Too Many Requests",
                    503 => "Service Unavailable",
                    400 => "Bad Request",
                    _ => "OK",
                };
                let resp_body = if code == success_status {
                    &success_body
                } else {
                    "{\"error\":\"transient\"}"
                };
                let response = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    code,
                    reason,
                    resp_body.len(),
                    resp_body
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });

    (format!("http://{}", addr), requests)
}

/// 启动一个返回 OpenAI 风格 embeddings 响应的 HTTP stub。
///
/// `n_vectors(input_count)` 决定每个请求返回的向量数量，便于模拟
/// 正常返回（`|n| n`）、少返回（`|n| n.saturating_sub(1)`）、
/// 超量返回（`|_| 100`）等批量对齐场景。
pub async fn spawn_embeddings_stub(n_vectors: Arc<dyn Fn(usize) -> usize + Send + Sync>) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let n_vectors = n_vectors.clone();
            tokio::spawn(async move {
                // 读 header 区（到 \r\n\r\n 结束）
                let mut header = Vec::new();
                let mut byte = [0u8; 1];
                while header.len() < 64 * 1024 {
                    if socket.read_exact(&mut byte).await.is_err() {
                        return;
                    }
                    header.push(byte[0]);
                    if header.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }

                let header_str = String::from_utf8_lossy(&header).to_lowercase();
                let content_length: usize = header_str
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);

                // 读 body
                let mut body = vec![0u8; content_length];
                if content_length > 0 && socket.read_exact(&mut body).await.is_err() {
                    return;
                }
                let body_str = String::from_utf8_lossy(&body);

                // 解析请求文本：OpenAI/DeepSeek/Qwen 用 "input"，Cohere 用 "texts"
                let inputs: Vec<String> = serde_json::from_str::<serde_json::Value>(&body_str)
                    .ok()
                    .and_then(|v| v.get("input").cloned().or_else(|| v.get("texts").cloned()))
                    .map(|input| match input {
                        serde_json::Value::String(s) => vec![s],
                        serde_json::Value::Array(a) => a
                            .iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect(),
                        _ => vec![],
                    })
                    .unwrap_or_default();
                let input_count = inputs.len();

                // 每个请求返回前 n 个输入的向量;每个向量由文本字节和编码,
                // 便于测试验证文本↔向量对齐。超出输入数量的部分用合成向量填充。
                let n = n_vectors(input_count);
                let data: Vec<serde_json::Value> = (0..n)
                    .map(|i| {
                        // P2-8: 向量 [sum, 1.0] 归一化后仍随文本不同而不同,
                        // 让对齐测试在归一化语义下仍能区分每条文本。
                        let embedding = match inputs.get(i) {
                            Some(s) => vec![s.bytes().map(|b| b as f32).sum::<f32>(), 1.0_f32],
                            None => vec![i as f32 + 1000.0, 0.0_f32],
                        };
                        serde_json::json!({
                            "object": "embedding",
                            "index": i,
                            "embedding": embedding,
                        })
                    })
                    .collect();
                let json = serde_json::json!({
                    "data": data,
                    "model": "stub",
                    "usage": { "prompt_tokens": 0, "total_tokens": 0 },
                })
                .to_string();

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    json.len(),
                    json
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });

    format!("http://{}", addr)
}
