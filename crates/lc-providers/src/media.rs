// lc-providers/src/media.rs
//! Shared multimodal media resolution for chat request builders (B7, v0.22.4).
//!
//! Every chat backend speaks a different wire dialect for attachments:
//! OpenAI Chat Completions uses `image_url` / `input_audio` / `input_video` /
//! `file` blocks, Anthropic only accepts base64 `image`/`document` blocks, and
//! Gemini uses `inline_data` / `file_data` parts. This module is the single
//! place that:
//!
//! 1. parses `data:` URIs into (MIME, raw base64),
//! 2. fetches plain-URL attachments through the SSRF-guarded [`lc_core::ssrf`]
//!    path (same rules as Whisper — caller-supplied URLs must never reach the
//!    intranet), enforcing a size cap, and rewrites them to data URIs,
//! 3. applies per-provider capability policies (Anthropic rejects
//!    audio/video; Ollama only speaks images; only PDFs are accepted as
//!    `file`/`document` attachments) so unsupported media is an explicit
//!    request-build error instead of being silently dropped,
//! 4. builds the OpenAI-family user content blocks from one
//!    [`lc_schema::Message`].
//!
//! Preprocessing happens once at each provider's `chat_internal` /
//! `stream_chat_internal` entry point; the sync request-body mappers then only
//! see normalized values and stay side-effect free (and unit-testable).

use base64::Engine;
use lc_schema::{MediaPart, Message};
use serde_json::{json, Value};

/// Maximum size of a single fetched attachment (20 MiB).
///
/// Gemini's `inline_data` request cap is ~20 MiB total; keeping the same
/// ceiling everywhere bounds memory use and blocks giant-URL DoS.
pub(crate) const MAX_MEDIA_FETCH_BYTES: usize = 20 * 1024 * 1024;

/// Errors raised while normalizing multimodal attachments.
#[derive(Debug, thiserror::Error)]
pub(crate) enum MediaError {
    /// The `data:` URI is malformed (bad prefix / missing payload).
    #[error("multimodal input: malformed data URI ({0})")]
    InvalidDataUri(String),

    /// Fetching the attachment over HTTP(S) failed (network, non-2xx, SSRF).
    #[error("multimodal input: failed to fetch media URL {url}: {reason}")]
    Fetch {
        /// URL that failed.
        url: String,
        /// Underlying reason.
        reason: String,
    },

    /// Downloaded (or advertised) body exceeds [`MAX_MEDIA_FETCH_BYTES`].
    #[error("multimodal input: media at {url} exceeds size limit ({size} bytes > {limit} bytes)")]
    TooLarge {
        /// Media URL.
        url: String,
        /// Observed size.
        size: usize,
        /// Configured limit.
        limit: usize,
    },

    /// The provider cannot accept this attachment (wrong modality/MIME/URI scheme).
    #[error("multimodal input: {0}")]
    Unsupported(String),
}

/// Which provider dialect the attachments must be normalized for.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MediaPolicy {
    /// OpenAI Chat Completions (also Azure): images pass through; audio/files
    /// are inlined; video passes through as URL.
    OpenAi,
    /// Anthropic Messages: images/PDFs must be base64 (URLs are fetched);
    /// audio/video are unsupported.
    Anthropic,
    /// Gemini: everything inlines (or `gs://` File API URIs pass through).
    Gemini,
    /// Ollama OpenAI-compatible shim: image input only.
    Ollama,
}

impl MediaPolicy {
    fn provider_name(&self) -> &'static str {
        match self {
            MediaPolicy::OpenAi => "OpenAI",
            MediaPolicy::Anthropic => "Anthropic",
            MediaPolicy::Gemini => "Gemini",
            MediaPolicy::Ollama => "Ollama",
        }
    }
}

/// Splits a data URI into (media type, raw base64 payload).
///
/// Accepts `data:<mime>;base64,<data>` (the form every schema content type
/// produces). Plain (URL-encoded) data URIs without the `base64` marker are
/// rejected — providers want base64 payloads.
pub(crate) fn data_uri_parts(uri: &str) -> Option<(&str, &str)> {
    let rest = uri.strip_prefix("data:")?;
    let comma = rest.find(',')?;
    let meta = &rest[..comma];
    let data = &rest[comma + 1..];
    if !meta.contains("base64") {
        return None;
    }
    let mime = meta
        .split(';')
        .next()
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .unwrap_or("application/octet-stream");
    Some((mime, data))
}

/// Infers a MIME type from a URL/path extension.
pub(crate) fn mime_from_extension(url: &str) -> Option<&'static str> {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    Some(match ext.as_str() {
        // images
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        // audio
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "ogg" | "oga" => "audio/ogg",
        "opus" => "audio/opus",
        "webm" => "audio/webm",
        "m4a" | "aac" => "audio/mp4",
        // video
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "avi" => "video/x-msvideo",
        // documents
        "pdf" => "application/pdf",
        _ => return None,
    })
}

/// Extracts a filename hint from the tail of a URL/data URI.
pub(crate) fn filename_from_url(url: &str, fallback: &str) -> String {
    // Data URIs must be checked whole: their MIME segment (e.g.
    // "application/pdf") contains a '/' that confuses tail extraction.
    if url.starts_with("data:") {
        return fallback.to_string();
    }
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let tail = path.rsplit('/').next().unwrap_or("");
    if tail.is_empty() {
        fallback.to_string()
    } else {
        tail.to_string()
    }
}

/// Maps an audio MIME type to the OpenAI `input_audio.format` wire value.
fn openai_audio_format(mime: &str) -> Option<&'static str> {
    Some(match mime {
        "audio/wav" | "audio/x-wav" | "audio/wave" => "wav",
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/flac" | "audio/x-flac" => "flac",
        "audio/ogg" => "ogg",
        "audio/opus" => "opus",
        "audio/webm" => "webm",
        "audio/mp4" | "audio/x-m4a" | "audio/m4a" => "m4a",
        "audio/aac" => "aac",
        "audio/l16" | "audio/pcm" => "pcm",
        _ => return None,
    })
}

/// Fetches an http(s) URL through the SSRF guard and encodes it as a data URI.
async fn fetch_to_data_uri(url: &str) -> Result<String, MediaError> {
    // Only http(s) is fetchable; guarded_get also re-validates and re-pins every
    // redirect hop against the validated address set.
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(MediaError::Unsupported(format!(
            "unsupported media reference scheme: {url}"
        )));
    }

    let response = lc_core::ssrf::guarded_get(url, true, None)
        .await
        .map_err(|e| MediaError::Fetch {
            url: url.to_string(),
            reason: e.to_string(),
        })?;

    if !response.status().is_success() {
        return Err(MediaError::Fetch {
            url: url.to_string(),
            reason: format!("HTTP {}", response.status()),
        });
    }

    // Reject oversized bodies up front when Content-Length is present.
    if let Some(len) = response.content_length() {
        if len as usize > MAX_MEDIA_FETCH_BYTES {
            return Err(MediaError::TooLarge {
                url: url.to_string(),
                size: len as usize,
                limit: MAX_MEDIA_FETCH_BYTES,
            });
        }
    }

    // Prefer the server-declared content type, fall back to URL extension.
    // Extracted before `response.bytes()` consumes the response.
    let mime = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .map(str::to_ascii_lowercase)
        .or_else(|| mime_from_extension(url).map(str::to_string))
        .ok_or_else(|| {
            MediaError::Unsupported(format!(
                "cannot determine MIME type for fetched media: {url}"
            ))
        })?;

    let bytes = response.bytes().await.map_err(|e| MediaError::Fetch {
        url: url.to_string(),
        reason: e.to_string(),
    })?;
    if bytes.len() > MAX_MEDIA_FETCH_BYTES {
        return Err(MediaError::TooLarge {
            url: url.to_string(),
            size: bytes.len(),
            limit: MAX_MEDIA_FETCH_BYTES,
        });
    }

    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{mime};base64,{encoded}"))
}

/// Returns true when a file reference is a PDF (explicit MIME, data URI, or extension).
fn is_pdf_reference(url: &str, explicit_mime: Option<&str>) -> bool {
    if explicit_mime
        .map(|m| m.eq_ignore_ascii_case("application/pdf"))
        .unwrap_or(false)
    {
        return true;
    }
    if let Some((mime, _)) = data_uri_parts(url) {
        return mime.eq_ignore_ascii_case("application/pdf");
    }
    mime_from_extension(url) == Some("application/pdf")
}

/// Normalizes one media reference in place.
///
/// `require_inline` — http(s) URLs are fetched and converted to data URIs
/// (Anthropic/Gemini, OpenAI audio/PDF). When false, http(s) URLs and data URIs
/// pass through. `allow_gs` — Google Cloud Storage `gs://` URIs pass through
/// (Gemini File API only).
async fn normalize_reference(
    url: &mut String,
    require_inline: bool,
    allow_gs: bool,
) -> Result<(), MediaError> {
    if url.starts_with("data:") {
        if data_uri_parts(url).is_none() {
            return Err(MediaError::InvalidDataUri(url.clone()));
        }
        return Ok(());
    }
    if url.starts_with("gs://") {
        return if allow_gs {
            Ok(())
        } else {
            Err(MediaError::Unsupported(format!(
                "gs:// URIs are only supported by Gemini: {url}"
            )))
        };
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        if require_inline {
            *url = fetch_to_data_uri(url).await?;
        }
        return Ok(());
    }
    Err(MediaError::Unsupported(format!(
        "unsupported media reference (expected http(s) URL or data: URI): {url}"
    )))
}

/// Rewrites every attachment in `messages` according to the provider policy.
///
/// Call this at the top of a chat entry point, before building the request
/// body. It performs all network I/O and validation; the sync mappers only
/// handle the normalized result.
///
/// HTTP fetches go through `lc_core::ssrf::guarded_get`, which builds its own
/// per-request client with validated, pinned DNS answers (A3), so no caller
/// client is needed.
pub(crate) async fn resolve_message_media(
    messages: &mut [Message],
    policy: MediaPolicy,
) -> Result<(), MediaError> {
    for message in messages.iter_mut() {
        for image in &mut message.images {
            let fetch = matches!(policy, MediaPolicy::Anthropic | MediaPolicy::Gemini);
            normalize_reference(&mut image.url, fetch, matches!(policy, MediaPolicy::Gemini))
                .await?;
        }

        for audio in &mut message.audio {
            match policy {
                MediaPolicy::OpenAi => normalize_reference(&mut audio.url, true, false).await?,
                MediaPolicy::Gemini => normalize_reference(&mut audio.url, true, true).await?,
                MediaPolicy::Anthropic | MediaPolicy::Ollama => {
                    return Err(MediaError::Unsupported(format!(
                        "{} does not support audio input in chat messages",
                        policy.provider_name()
                    )));
                }
            }
        }

        for video in &mut message.videos {
            match policy {
                MediaPolicy::OpenAi => {
                    // input_video takes an https URL/file_id; data URIs pass through too.
                    normalize_reference(&mut video.url, false, false).await?
                }
                MediaPolicy::Gemini => normalize_reference(&mut video.url, true, true).await?,
                MediaPolicy::Anthropic | MediaPolicy::Ollama => {
                    return Err(MediaError::Unsupported(format!(
                        "{} does not support video input in chat messages",
                        policy.provider_name()
                    )));
                }
            }
        }

        for file in &mut message.files {
            if !is_pdf_reference(&file.url, file.mime_type.as_deref()) {
                return Err(MediaError::Unsupported(format!(
                    "{} only supports PDF file attachments (got {:?})",
                    policy.provider_name(),
                    file.mime_type
                )));
            }
            match policy {
                MediaPolicy::OpenAi | MediaPolicy::Anthropic => {
                    normalize_reference(&mut file.url, true, false).await?
                }
                MediaPolicy::Gemini => normalize_reference(&mut file.url, true, true).await?,
                MediaPolicy::Ollama => {
                    return Err(MediaError::Unsupported(
                        "Ollama does not support file/document attachments".to_string(),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Builds the OpenAI Chat Completions user `content` blocks for a multimodal
/// message.
///
/// Returns `None` for plain-text messages so callers keep the byte-identical
/// string-content form. [`resolve_message_media`] with [`MediaPolicy::OpenAi`]
/// must have run first: audio blocks need inlined base64 and `file` blocks need
/// PDF data URIs. Anything that still cannot be mapped is skipped with a
/// warning rather than silently changing request semantics.
pub(crate) fn openai_user_blocks(message: &Message) -> Option<Vec<Value>> {
    if !message.is_multimodal() {
        return None;
    }
    let mut blocks: Vec<Value> = Vec::new();
    if !message.content.is_empty() {
        blocks.push(json!({"type": "text", "text": message.content}));
    }

    for part in message.media_parts() {
        match part {
            MediaPart::Image(img) => {
                blocks.push(json!({"type": "image_url", "image_url": {"url": img.url}}));
            }
            MediaPart::Audio(audio) => {
                let Some((mime, data)) = data_uri_parts(&audio.url) else {
                    log::warn!("skipping audio attachment with unresolved reference");
                    continue;
                };
                let format = openai_audio_format(mime).unwrap_or("wav");
                blocks.push(json!({
                    "type": "input_audio",
                    "input_audio": {"data": data, "format": format},
                }));
            }
            MediaPart::Video(video) => {
                blocks.push(json!({
                    "type": "input_video",
                    "input_video": {"url": video.url},
                }));
            }
            MediaPart::File(file) => {
                let filename = file
                    .name
                    .clone()
                    .unwrap_or_else(|| filename_from_url(&file.url, "document.pdf"));
                blocks.push(json!({
                    "type": "file",
                    "file": {"file_data": file.url, "filename": filename},
                }));
            }
        }
    }

    Some(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_schema::{AudioContent, FileContent, ImageContent, VideoContent};

    #[test]
    fn data_uri_parsing() {
        let (mime, data) = data_uri_parts("data:image/png;base64,QUJD").unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(data, "QUJD");

        let (mime, _) = data_uri_parts("data:application/pdf;base64,QUJD").unwrap();
        assert_eq!(mime, "application/pdf");

        assert!(data_uri_parts("data:text/plain,hello").is_none());
        assert!(data_uri_parts("https://example.com/a.png").is_none());
    }

    #[test]
    fn extension_mime_table() {
        assert_eq!(
            mime_from_extension("https://e.com/a.JPG?x=1"),
            Some("image/jpeg")
        );
        assert_eq!(
            mime_from_extension("clip.mov#frag"),
            Some("video/quicktime")
        );
        assert_eq!(mime_from_extension("doc.pdf"), Some("application/pdf"));
        assert_eq!(mime_from_extension("song.mp3"), Some("audio/mpeg"));
        assert_eq!(mime_from_extension("noext"), None);
    }

    #[test]
    fn filename_tail_extraction() {
        assert_eq!(
            filename_from_url("https://e.com/d/r.pdf?token=x", "f"),
            "r.pdf"
        );
        assert_eq!(filename_from_url("https://e.com/folder/", "f.pdf"), "f.pdf");
        assert_eq!(
            filename_from_url("data:application/pdf;base64,AAA", "f.pdf"),
            "f.pdf"
        );
    }

    #[test]
    fn pdf_detection() {
        assert!(is_pdf_reference("https://e.com/a.pdf", None));
        assert!(is_pdf_reference("data:application/pdf;base64,AAA", None));
        assert!(is_pdf_reference(
            "https://e.com/blob/123",
            Some("application/pdf")
        ));
        assert!(!is_pdf_reference("https://e.com/a.csv", None));
        assert!(!is_pdf_reference(
            "data:text/csv;base64,AAA",
            Some("text/csv")
        ));
    }

    #[tokio::test]
    async fn ollama_policy_rejects_non_image_attachments() {
        let mut messages = vec![Message::human_with_audio(
            "听",
            AudioContent::from_base64("AAA"),
        )];
        let err = resolve_message_media(&mut messages, MediaPolicy::Ollama)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Ollama does not support audio"));

        let mut messages = vec![Message::human_with_video(
            "看",
            VideoContent::from_base64("AAA"),
        )];
        assert!(resolve_message_media(&mut messages, MediaPolicy::Ollama)
            .await
            .is_err());

        let mut messages = vec![Message::human_with_file(
            "读",
            FileContent::from_base64("AAA", "application/pdf"),
        )];
        assert!(resolve_message_media(&mut messages, MediaPolicy::Ollama)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn anthropic_policy_rejects_audio_and_video() {
        let mut messages = vec![Message::human_with_audio(
            "听",
            AudioContent::from_base64("AAA"),
        )];
        let err = resolve_message_media(&mut messages, MediaPolicy::Anthropic)
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("Anthropic does not support audio input"));

        let mut messages = vec![Message::human_with_video(
            "看",
            VideoContent::from_base64("AAA"),
        )];
        assert!(resolve_message_media(&mut messages, MediaPolicy::Anthropic)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn non_pdf_files_are_rejected_everywhere() {
        for policy in [
            MediaPolicy::OpenAi,
            MediaPolicy::Anthropic,
            MediaPolicy::Gemini,
        ] {
            let mut messages = vec![Message::human_with_file(
                "数据",
                FileContent::from_url_with_mime("https://e.com/data.csv", "text/csv"),
            )];
            let err = resolve_message_media(&mut messages, policy)
                .await
                .unwrap_err();
            assert!(
                err.to_string().contains("only supports PDF"),
                "policy {policy:?} => {err}"
            );
        }
    }

    #[tokio::test]
    async fn gs_uris_only_pass_gemini_policy() {
        let mut messages =
            vec![Message::human("图").with_image(ImageContent::from_url("gs://bucket/a.png"))];
        resolve_message_media(&mut messages, MediaPolicy::Gemini)
            .await
            .unwrap();
        assert_eq!(messages[0].images[0].url, "gs://bucket/a.png");

        let err = resolve_message_media(&mut messages, MediaPolicy::OpenAi)
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("gs:// URIs are only supported by Gemini"));
    }

    #[test]
    fn openai_blocks_cover_all_modalities() {
        let msg = Message::human("all")
            .with_image(ImageContent::from_url("https://e.com/a.png"))
            .with_audio(AudioContent::from_base64_with_mime("AAA", "audio/mp3"))
            .with_video(VideoContent::from_url("https://e.com/v.mp4"))
            .with_file(FileContent::from_base64("UEYG", "application/pdf").with_name("r.pdf"));
        let blocks = openai_user_blocks(&msg).unwrap();
        assert_eq!(blocks.len(), 5); // text + 4 attachments
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image_url");
        assert_eq!(blocks[1]["image_url"]["url"], "https://e.com/a.png");
        assert_eq!(blocks[2]["type"], "input_audio");
        assert_eq!(blocks[2]["input_audio"]["format"], "mp3");
        assert_eq!(blocks[2]["input_audio"]["data"], "AAA");
        assert_eq!(blocks[3]["type"], "input_video");
        assert_eq!(blocks[3]["input_video"]["url"], "https://e.com/v.mp4");
        assert_eq!(blocks[4]["type"], "file");
        assert_eq!(
            blocks[4]["file"]["file_data"],
            "data:application/pdf;base64,UEYG"
        );
        assert_eq!(blocks[4]["file"]["filename"], "r.pdf");

        // Plain text stays a plain string (None → caller uses string content).
        assert!(openai_user_blocks(&Message::human("hi")).is_none());
    }

    #[test]
    fn audio_format_aliases() {
        assert_eq!(openai_audio_format("audio/x-wav"), Some("wav"));
        assert_eq!(openai_audio_format("audio/mpeg"), Some("mp3"));
        assert_eq!(openai_audio_format("audio/mp4"), Some("m4a"));
        assert_eq!(openai_audio_format("audio/ogg"), Some("ogg"));
    }

    #[tokio::test]
    async fn ssrf_blocks_loopback_fetch_during_resolution() {
        let mut messages = vec![Message::human_with_image(
            "内网图",
            "http://127.0.0.1:9/secret.png",
        )];
        let err = resolve_message_media(&mut messages, MediaPolicy::Anthropic)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("SSRF"), "{err}");
        // URL left untouched on failure
        assert_eq!(messages[0].images[0].url, "http://127.0.0.1:9/secret.png");
    }
}
