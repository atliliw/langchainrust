// lc-embeddings/src/test_support.rs
//! Test helpers: a minimal HTTP stub server returning OpenAI/DeepSeek/Qwen/Cohere-compatible
//! embeddings JSON responses, for batch-alignment tests that do not rely on a real network (P0-1).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Minimal HTTP stub: the first `failures_before_success` requests return `transient_status`,
/// then `success_status` (with `success_body`). Returns `(base_url, received request count)`.
///
/// Used for P2-5 retry-semantics tests (429/5xx retried, 4xx not retried, retries exhausted)
/// and provider-level retry wiring tests.
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
    // Convert to owned so it can escape this function with the task (success_body's lifetime is not 'static).
    let success_body = success_body.to_string();

    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let r = r.clone();
            // Every connection may use the success body, so clone it into each handler task.
            let success_body = success_body.clone();
            tokio::spawn(async move {
                // Read the header section up to \r\n\r\n
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
                // Read the body (discarded; only counting requests)
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

/// Starts an HTTP stub returning OpenAI-style embeddings responses.
///
/// `n_vectors(input_count)` decides how many vectors each request returns, simulating
/// normal returns (`|n| n`), short returns (`|n| n.saturating_sub(1)`), excess returns
/// (`|_| 100`), and other batch-alignment scenarios.
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
                // Read the header section (up to \r\n\r\n)
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

                // Read the body
                let mut body = vec![0u8; content_length];
                if content_length > 0 && socket.read_exact(&mut body).await.is_err() {
                    return;
                }
                let body_str = String::from_utf8_lossy(&body);

                // Parse the request text: OpenAI/DeepSeek/Qwen use "input", Cohere uses "texts"
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

                // Each request returns vectors for the first n inputs; each vector is encoded
                // from the text bytes, letting tests verify text↔vector alignment. Entries
                // beyond the input count are filled with synthetic vectors.
                let n = n_vectors(input_count);
                let data: Vec<serde_json::Value> = (0..n)
                    .map(|i| {
                        // P2-8: vector [sum, 1.0] still differs per text after normalization,
                        // so alignment tests can distinguish each text under normalization semantics.
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
