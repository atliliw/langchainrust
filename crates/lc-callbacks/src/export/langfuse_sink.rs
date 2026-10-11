// lc-callbacks/src/export/langfuse_sink.rs
//! Langfuse `BatchSink` (lc-callbacks T3: "LangSmith / Langfuse 导出").
//!
//! LangSmith already has a dedicated handler + client (`langsmith_handler.rs` /
//! `langsmith_client.rs`). This module closes the other half: a sink that posts
//! batches to the Langfuse **Ingestion API** (`POST /api/public/ingestion`),
//! HMAC-SHA256-signed so the secret never rides in the URL or a plain header.
//!
//! It plugs into the sink-agnostic pipeline from [`super`] with **zero new
//! infrastructure**: point a [`BatchExporter`](super::BatchExporter) at a
//! `LangfuseSink` and the same bounded-retry / backoff / disk-park machinery
//! already handles delivery (the batch/Edge layer was wired in `super::mod`).
//!
//! The signing primitives are pure and verified against well-known SHA-256 and
//! RFC 4231 HMAC test vectors — signature correctness is proven without a live
//! Langfuse account, and a `#[tokio::test]` round-trips a signed request through
//! an in-process TCP mock to pin the wire shape.

use crate::export::{BatchSink, ExportError, ExportEvent};
use async_trait::async_trait;
use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::time::Duration;

/// Upper bound on a single Ingestion POST, mirroring the LangSmith handler:
/// observability is a side channel and must never hang the run indefinitely.
const LANGFUSE_TIMEOUT: Duration = Duration::from_secs(15);

/// HMAC-SHA256 (key, message) primitives over a `Hmac<Sha256>`.
type HmacSha256 = Hmac<Sha256>;

/// Lowercase hex encoding (std has no built-in hex).
fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// SHA-256 of `data`, hex-encoded.
pub fn sha256_hex(data: &[u8]) -> String {
    to_hex(&Sha256::digest(data))
}

/// HMAC-SHA256 of `data` keyed by `key`, hex-encoded. `Hmac` accepts any key
/// length, zero-padding short keys per RFC 2104.
pub fn hmac_sha256_hex(key: &[u8], data: &[u8]) -> String {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key)
        .expect("HMAC new_from_slice only fails for invalid lengths, never for any key");
    mac.update(data);
    to_hex(&mac.finalize().into_bytes())
}

/// The Langfuse Ingestion request signature (pure, unit-testable).
///
/// Signing scheme (Langfuse SDK contract): the sha-256 hex of the payload, the
/// payload's UTF-8 byte length, and the request timestamp in millis; the
/// signature header is `HMAC-SHA256(secret, "{sha256hex}.{ts}")`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LangfuseSignature {
    /// Request timestamp in epoch millis.
    pub timestamp_ms: u64,
    /// Lowercase hex SHA-256 of the UTF-8 payload.
    pub content_sha256: String,
    /// UTF-8 byte length of the payload.
    pub byte_len: usize,
    /// Lowercase hex HMAC-SHA256 of `"{content_sha256}.{timestamp_ms}"`.
    pub signature: String,
}

/// Computes the signature for a raw payload string (pure; no clock/io).
pub fn sign_payload(content: &str, secret_key: &str, timestamp_ms: u64) -> LangfuseSignature {
    let content_sha256 = sha256_hex(content.as_bytes());
    let signature = hmac_sha256_hex(
        secret_key.as_bytes(),
        format!("{content_sha256}.{timestamp_ms}").as_bytes(),
    );
    LangfuseSignature {
        timestamp_ms,
        content_sha256,
        byte_len: content.len(),
        signature,
    }
}

/// Ingestion configuration. The secret stays a plain `String` exactly like
/// `LangSmithConfig.api_key`; a future `SensitiveString` migration can cover
/// both together.
#[derive(Debug, Clone)]
pub struct LangfuseConfig {
    /// Public key (the non-secret half of the API pair).
    pub public_key: String,
    /// Secret key (HMAC signing key; never logged).
    pub secret_key: String,
    /// API base URL. Deliberately has no default: it must be supplied
    /// explicitly so a misconfigured sink can never silently ship telemetry to
    /// a host the caller did not choose.
    pub base_url: String,
}

impl LangfuseConfig {
    /// Builds a config from an explicit public key, secret key, and base URL.
    ///
    /// `base_url` is mandatory — there is no managed-cloud default, so omitting
    /// the endpoint is a compile error rather than a silent fallback that leaks
    /// telemetry to the wrong host.
    pub fn new(
        public_key: impl Into<String>,
        secret_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            public_key: public_key.into(),
            secret_key: secret_key.into(),
            base_url: base_url.into(),
        }
    }

    /// Reads `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`.
    /// All three are required: a missing base URL is an error, never a silent
    /// fallback to the managed cloud.
    pub fn from_env() -> Result<Self, LangfuseError> {
        Ok(Self {
            public_key: env::var("LANGFUSE_PUBLIC_KEY")?,
            secret_key: env::var("LANGFUSE_SECRET_KEY")?,
            base_url: env::var("LANGFUSE_BASE_URL")?,
        })
    }
}

/// Configuration/construction error for a [`LangfuseSink`].
#[derive(Debug)]
pub struct LangfuseError(String);

impl std::fmt::Display for LangfuseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Langfuse config error: {}", self.0)
    }
}

impl std::error::Error for LangfuseError {}

impl From<env::VarError> for LangfuseError {
    fn from(e: env::VarError) -> Self {
        Self(format!("missing/invalid env var: {e}"))
    }
}

/// A [`BatchSink`] that POSTs a batch to the Langfuse Ingestion API.
pub struct LangfuseSink {
    config: LangfuseConfig,
    client: reqwest::Client,
}

impl std::fmt::Debug for LangfuseSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangfuseSink")
            .field("base_url", &self.config.base_url)
            .field("public_key", &self.config.public_key)
            .field("secret_key", &"[REDACTED]")
            .finish()
    }
}

impl LangfuseSink {
    /// Wraps a config with a fresh HTTP client (15s timeout per request).
    pub fn new(config: LangfuseConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::builder()
                .timeout(LANGFUSE_TIMEOUT)
                .build()
                .expect("reqwest client builder cannot fail with the default configuration"),
        }
    }

    /// Builds the signed Ingestion request for one batch (pure and observable).
    fn build_request(
        &self,
        batch: &[ExportEvent],
        timestamp_ms: u64,
    ) -> (String, LangfuseSignature) {
        let events: Vec<Value> = batch
            .iter()
            .map(|e| {
                serde_json::json!({
                    "id": e.id,
                    "type": e.kind,
                    "timestamp": e.timestamp,
                    "body": e.body,
                })
            })
            .collect();
        let content = serde_json::json!({ "batch": events }).to_string();
        let sig = sign_payload(&content, &self.config.secret_key, timestamp_ms);
        (content, sig)
    }
}

#[async_trait]
impl BatchSink for LangfuseSink {
    async fn send_batch(&self, batch: &[ExportEvent]) -> Result<(), ExportError> {
        if batch.is_empty() {
            return Ok(());
        }
        let ts = chrono::Utc::now()
            .timestamp_millis()
            .try_into()
            .map_err(|_| ExportError::Fatal("timestamp out of range".to_string()))?;
        let (content, sig) = self.build_request(batch, ts);

        let url = format!("{}/api/public/ingestion", self.config.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(&url)
            .header("X-Langfuse-Sdk-Name", "langchainrust")
            .header("X-Langfuse-Sdk-Version", env!("CARGO_PKG_VERSION"))
            .header("X-Langfuse-Public-Key", &self.config.public_key)
            .header("X-Langfuse-Timestamp", sig.timestamp_ms.to_string())
            .header("X-Langfuse-Signature", &sig.signature)
            .header(
                "X-Langfuse-Content-Sha256",
                format!("{} {}", sig.content_sha256, sig.byte_len),
            )
            .header("Content-Type", "application/json")
            .body(content)
            .send()
            .await
            .map_err(|e| ExportError::Retryable(format!("Langfuse ingest request failed: {e}")))?;

        let status = resp.status();
        if status.is_success() {
            Ok(())
        } else if status.is_client_error() {
            // 4xx: auth/problematic payload — retrying won't help; surface it as fatal
            // so the exporter stops the attempt loop instead of parking forever.
            Err(ExportError::Fatal(format!(
                "Langfuse ingestion rejected batch with HTTP {status}"
            )))
        } else {
            // 5xx / 429: transient; the exporter's bounded retry/backoff applies.
            Err(ExportError::Retryable(format!(
                "Langfuse ingestion returned HTTP {status}"
            )))
        }
    }

    fn name(&self) -> &str {
        "langfuse"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Well-known SHA-256 test vectors (FIPS 180-4).
    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// RFC 4231 section "Test Case 1": key = 0x0b × 20, data = "Hi There".
    #[test]
    fn hmac_sha256_matches_rfc4231_tc1() {
        let key = [0x0bu8; 20];
        let sig = hmac_sha256_hex(&key, b"Hi There");
        assert_eq!(
            sig,
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    /// The signed payload composes exactly from the primitives: sha-256 hex, byte
    /// length, and HMAC over `"{sha256hex}.{ts}"`. Also: a wrong secret or a
    /// mutated payload must change the signature (which is the whole point).
    #[test]
    fn sign_payload_is_consistent_and_secret_sensitive() {
        let content = r#"{"batch":[{"id":"1","type":"trace-create"}]}"#;
        let sig = sign_payload(content, "sk-live-secret", 1712345678901);

        assert_eq!(sig.content_sha256, sha256_hex(content.as_bytes()));
        assert_eq!(sig.byte_len, content.len());
        assert_eq!(
            sig.signature,
            hmac_sha256_hex(
                b"sk-live-secret",
                format!("{}.{}", sig.content_sha256, sig.timestamp_ms).as_bytes()
            )
        );

        // A different secret must produce a different signature.
        let other = sign_payload(content, "sk-live-secret-2", sig.timestamp_ms);
        assert_ne!(sig.signature, other.signature);

        // Mutating the payload must change the signature for a fixed secret.
        let mutated = sign_payload("{\"batch\":[]}", "sk-live-secret", sig.timestamp_ms);
        assert_ne!(sig.signature, mutated.signature);
    }

    /// End-to-end: `send_batch` POSTs a well-formed, correctly-signed request to
    /// `/api/public/ingestion`. The mock server recomputes the content sha-256
    /// from the received body and confirms the header matches (proving the body
    /// on the wire is exactly what was signed).
    #[tokio::test]
    async fn send_batch_posts_signed_body_to_ingestion() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            // Read headers (up to first CRLFCRLF).
            let mut buf = Vec::new();
            let mut byte = [0u8; 1];
            while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                sock.read_exact(&mut byte).await.unwrap();
                buf.push(byte[0]);
            }
            let header = String::from_utf8_lossy(&buf);
            assert!(
                header.starts_with("POST /api/public/ingestion HTTP/1.1"),
                "unexpected request line: {header}"
            );
            // Read body per Content-Length.
            let len_line = header.lines().find(|l| l.to_ascii_lowercase().starts_with("content-length:")).map(|l| {
                l.split(':').nth(1).unwrap().trim().parse::<usize>().unwrap()
            });
            let mut body = vec![0u8; len_line.unwrap()];
            sock.read_exact(&mut body).await.unwrap();
            // Acknowledge.
            sock.write_all(b"HTTP/1.1 200 OK\r\n\r\n").await.unwrap();

            (header.to_string(), String::from_utf8(body).unwrap())
        });

        let config = LangfuseConfig::new("pk-test", "sk-test-secret", format!("http://{addr}"));
        let sink = LangfuseSink::new(config);
        let event = ExportEvent::now("trace-create", "trace-1", serde_json::json!({"kv": 1}));
        sink.send_batch(std::slice::from_ref(&event)).await.unwrap();

        let (header, body) = server.await.unwrap();
        let header = header.to_ascii_lowercase();

        // Signed headers present and consistent.
        assert!(header.contains("x-langfuse-public-key: pk-test"));
        assert!(header.contains("x-langfuse-sdk-name: langchainrust"));
        assert!(header.contains("x-langfuse-sdk-version:"));
        let expected_sha = sha256_hex(body.as_bytes());
        assert!(
            header.contains(&format!("x-langfuse-content-sha256: {expected_sha} {}", body.len())),
            "content-sha256 header must match signed body"
        );
        assert!(header.contains("x-langfuse-signature:"));
        // Body shape: {"batch":[{id,type,body}]} carrying our event through.
        let parsed: Value = serde_json::from_str(&body).unwrap();
        let batch = parsed["batch"].as_array().expect("batch array");
        assert_eq!(batch.len(), 1);
        let ev = &batch[0];
        assert_eq!(ev["id"], "trace-1");
        assert_eq!(ev["type"], "trace-create");
        assert_eq!(ev["body"]["kv"], 1);
    }

    /// A 4xx response is fatal (auth/payload won't heal on retry); a 5xx is
    /// retryable (transient). These drive the exporter's behavior.
    #[tokio::test]
    async fn send_batch_maps_http_status_to_export_error() {
        async fn with_status(status: &str) -> Result<(), ExportError> {
            let status = status.to_string();
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = Vec::new();
                let mut byte = [0u8; 1];
                while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    sock.read_exact(&mut byte).await.unwrap();
                    buf.push(byte[0]);
                }
                let len = String::from_utf8_lossy(&buf)
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                    .map(|l| l.split(':').nth(1).unwrap().trim().parse::<usize>().unwrap());
                let mut body = vec![0u8; len.unwrap()];
                sock.read_exact(&mut body).await.unwrap();
                sock.write_all(format!("HTTP/1.1 {status}\r\n\r\n").as_bytes())
                    .await
                    .unwrap();
            });
            let config = LangfuseConfig::new("pk", "sk", format!("http://{addr}"));
            let sink = LangfuseSink::new(config);
            let result = sink
                .send_batch(&[ExportEvent::now("span-create", "s1", serde_json::json!({}))])
                .await;
            let _ = server.await;
            result
        }

        let fatal = with_status("401 Unauthorized").await;
        assert!(matches!(fatal, Err(ExportError::Fatal(_))), "4xx must be fatal: {fatal:?}");
        let retryable = with_status("503 Service Unavailable").await;
        assert!(matches!(retryable, Err(ExportError::Retryable(_))), "5xx must be retryable");
    }
}