// src/retrieval/loaders/mod.rs
//! Document loader implementations
//!
//! Provides document loading from files in various formats, including PDF, CSV, Text, JSON,
//! Markdown, HTML, etc. v0.4.1 added the WebScraper, Sitemap, and Docx loaders.

mod csv;
mod docx;
mod html;
mod json;
mod markdown;
mod pdf;
mod sitemap;
mod text;
mod web_scraper;

pub use csv::CSVLoader;
pub use docx::DocxLoader;
pub use html::HTMLLoader;
pub use json::JSONLoader;
pub use markdown::MarkdownLoader;
pub use pdf::PDFLoader;
pub use sitemap::SitemapLoader;
pub use text::TextLoader;
pub use web_scraper::WebScraperLoader;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lc_vector_stores::Document;
use std::time::Duration;

/// A5: cap the response body a URL-based loader will accept, so a hostile or
/// misconfigured target cannot exhaust memory by streaming an unbounded body.
const MAX_HTTP_BODY_BYTES: usize = 1024 * 1024; // 1 MiB

/// A5: shared SSRF-hardened HTTP fetch used by the URL-based loaders
/// (`HTMLLoader`, `WebScraperLoader`, `SitemapLoader`).
///
/// Redirects are *disabled* at the transport layer — manual redirect handling lives
/// inside `lc_core::ssrf::guarded_get`, which resolves each hop once, validates every
/// address, pins the validated IPs for the actual connection (DNS-rebinding closed),
/// and applies the per-request timeout. The body is then read with a 1 MiB hard cap.
/// Returns the post-redirect final URL and the body text so callers can record the
/// true `source`.
pub(crate) async fn guarded_fetch(
    url: &str,
    timeout: Duration,
) -> Result<(String, String), LoaderError> {
    let resp = lc_core::ssrf::guarded_get(url, true, Some(timeout))
        .await
        .map_err(|e| LoaderError::Other(format!("HTTP request failed {}: {}", url, e)))?;

    let final_url = resp.url().as_str().to_string();

    let status = resp.status();
    if !status.is_success() {
        return Err(LoaderError::Other(format!(
            "HTTP error {}: {}",
            url, status
        )));
    }

    // Stream-capped read: enforce the 1 MiB limit incrementally instead of reading an
    // unbounded body into memory first and only checking afterwards.
    let body = read_capped_body(resp.bytes_stream(), url, MAX_HTTP_BODY_BYTES).await?;

    let text = String::from_utf8(body)
        .map_err(|_| LoaderError::Other(format!("response for {} is not valid UTF-8", url)))?;
    Ok((final_url, text))
}

/// Drain a byte stream into a `Vec<u8>`, aborting as soon as the accumulated
/// size exceeds `max_bytes`. The cap is checked *after every chunk*, so an
/// oversized body is rejected before the remainder of the stream is polled —
/// the caller (and the network peer) never have to buffer the whole response.
/// (A5)
pub(crate) async fn read_capped_body<S, T, E>(
    mut stream: S,
    url: &str,
    max_bytes: usize,
) -> Result<Vec<u8>, LoaderError>
where
    S: Stream<Item = Result<T, E>> + Unpin,
    T: AsRef<[u8]>,
    E: std::fmt::Display,
{
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .map_err(|e| LoaderError::Other(format!("failed to read response {}: {}", url, e)))?;
        body.extend_from_slice(chunk.as_ref());
        if body.len() > max_bytes {
            return Err(LoaderError::Other(format!(
                "response for {} exceeds the {:.0} KiB size limit",
                url,
                max_bytes / 1024
            )));
        }
    }
    Ok(body)
}

/// Document loader error type
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LoaderError {
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// CSV parse error
    #[error("CSV parse error: {0}")]
    CsvError(String),

    /// PDF parse error
    #[error("PDF parse error: {0}")]
    PdfError(String),

    /// JSON parse error
    #[error("JSON parse error: {0}")]
    JsonError(String),

    /// Unknown error
    #[error("unknown error: {0}")]
    Other(String),
}

impl From<pdf_extract::Error> for LoaderError {
    fn from(err: pdf_extract::Error) -> Self {
        LoaderError::PdfError(err.to_string())
    }
}

/// Document loader trait
///
/// Defines the common interface for loading documents from a source.
#[async_trait]
pub trait DocumentLoader: Send + Sync {
    /// Loads documents from the source
    ///
    /// # Returns
    /// The loaded documents
    async fn load(&self) -> Result<Vec<Document>, LoaderError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn guarded_fetch_blocks_private_loopback() {
        // A5: a loopback URL must be rejected by the SSRF guard before any request
        // is sent — no network I/O happens, so this test needs no mock server.
        let err = guarded_fetch("http://127.0.0.1/sitemap.xml", Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("SSRF"),
            "expected an SSRF rejection, got: {err}"
        );
    }

    #[tokio::test]
    async fn guarded_fetch_blocks_link_local() {
        // A5: the cloud metadata address must be blocked even via IPv4-mapped IPv6.
        let err = guarded_fetch(
            "http://[::ffff:169.254.169.254]/latest",
            Duration::from_secs(5),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("SSRF"),
            "expected an SSRF rejection, got: {err}"
        );
    }

    #[tokio::test]
    async fn read_capped_body_accepts_body_exactly_at_the_limit() {
        use futures_util::stream;

        let half = vec![b'a'; MAX_HTTP_BODY_BYTES / 2];
        let chunks: Vec<Result<Vec<u8>, std::convert::Infallible>> =
            vec![Ok(half.clone()), Ok(half)];

        let body = read_capped_body(
            stream::iter(chunks),
            "http://example.com/x",
            MAX_HTTP_BODY_BYTES,
        )
        .await
        .expect("exactly 1 MiB must be accepted");
        assert_eq!(body.len(), MAX_HTTP_BODY_BYTES);
    }

    #[tokio::test]
    async fn read_capped_body_rejects_oversize_without_polling_the_rest() {
        use futures_util::stream;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // First chunk is already 600 KiB, the second pushes the total to 1.2 MiB.
        // A third "poison" chunk records whether it was ever pulled from the
        // stream — with incremental capping the reader must abort before that.
        let poison_pulled = Arc::new(AtomicUsize::new(0));
        let counter = poison_pulled.clone();
        let chunks: Vec<Result<Vec<u8>, std::convert::Infallible>> = vec![
            Ok(vec![b'a'; 600 * 1024]),
            Ok(vec![b'b'; 600 * 1024]),
            Ok(vec![b'c'; 1024]),
        ];
        let s = stream::unfold(0usize, move |idx| {
            let chunks = chunks.clone();
            let counter = counter.clone();
            async move {
                if idx >= chunks.len() {
                    return None;
                }
                if idx == 2 {
                    counter.fetch_add(1, Ordering::SeqCst);
                }
                let item = chunks[idx].clone();
                Some((item, idx + 1))
            }
        });
        // unfold streams are not `Unpin` (they pin the in-flight future); reqwest's
        // byte stream is Unpin, and boxing matches the production bound.
        let s = Box::pin(s);

        let err = read_capped_body(s, "http://example.com/huge", MAX_HTTP_BODY_BYTES)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("1024 KiB size limit"),
            "expected a size-limit error, got: {err}"
        );
        assert_eq!(
            poison_pulled.load(Ordering::SeqCst),
            0,
            "stream must not be polled for chunks past the cap (unbounded buffering)"
        );
    }

    #[tokio::test]
    async fn read_capped_body_rejects_non_utf8_inside_the_limit() {
        use futures_util::stream;

        let chunks: Vec<Result<Vec<u8>, std::convert::Infallible>> =
            vec![Ok(b"abc".to_vec()), Ok(vec![0xff, 0xfe])];

        // The capped reader itself returns raw bytes; mirror guarded_fetch's
        // UTF-8 conversion to lock the surfaced error.
        let body = read_capped_body(stream::iter(chunks), "http://example.com/bin", 64)
            .await
            .expect("small body accepted");
        let err = String::from_utf8(body).unwrap_err();
        assert!(err.to_string().contains("invalid utf-8"));
    }
}
