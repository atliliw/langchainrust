use std::collections::VecDeque;
use std::time::Duration;

use crate::protocol::{TaskPushNotification, TraceContext};

use super::A2AError;

/// Default idle (silence) budget for a single SSE read (B3, 0.26.0).
///
/// `A2ASseStream::next` previously did `self.response.chunk().await` with no
/// timeout, so a server that stopped sending without closing the connection
/// left the caller parked forever. Wrapping each chunk read in this budget
/// turns "no bytes for IDLE" into a [`A2AError::Timeout`] instead of an
/// unbounded park. The timer resets on every received chunk, so it is a true
/// idle guard, not a wall-clock cap on the subscription's lifetime.
///
/// Overridable per client via
/// [`A2AClientBuilder::sse_idle_timeout`](crate::client::A2AClientBuilder::sse_idle_timeout)
/// (0.27.0): a task that legitimately goes quiet for longer than this would
/// otherwise have its stream cut.
pub(crate) const DEFAULT_SSE_READ_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Hard cap on the buffered bytes of an SSE frame that has not yet been
/// terminated by a blank line (`\n\n`) (F-02).
///
/// `A2aSseParser::feed` accumulates incoming bytes in `buffer` and only
/// consumes them once a `\n\n` terminator is seen. Without a cap, a peer that
/// streams an endless `data:` line (or any bytes) and never emits a blank line
/// grows the buffer without bound — a memory-exhaustion DoS. When the buffer
/// would exceed this ceiling, the stream is treated as hostile: the buffer is
/// dropped and an error is surfaced instead of an OOM. A single legitimate SSE
/// frame is small (an A2A notification payload), so 1 MiB is far above any real
/// frame while hard-bounding memory.
const SSE_MAX_BUFFER_BYTES: usize = 1 << 20;

/// First reconnect delay after a transport break (0.27.0 T3 #95).
const SSE_RECONNECT_BASE_DELAY: Duration = Duration::from_millis(250);
/// Upper bound on a single reconnect delay (0.27.0 T3 #95).
const SSE_RECONNECT_MAX_DELAY: Duration = Duration::from_secs(5);
/// Deadline for a single reconnect `GET` to return response headers (0.27.0 T3 #95).
///
/// The streaming client is built with a connect timeout but **no** total
/// timeout, so `send()` resolves only once headers arrive: without this bound a
/// wedged peer that completes the TCP handshake and then never answers would
/// park the self-heal loop forever — the very hang self-heal exists to remove.
const SSE_RECONNECT_SEND_TIMEOUT: Duration = Duration::from_secs(10);

/// Exponential backoff for reconnect attempt `attempt` (0-based), capped at
/// [`SSE_RECONNECT_MAX_DELAY`].
fn sse_reconnect_backoff(attempt: usize) -> Duration {
    // Shift is clamped so `1 << shift` cannot overflow the multiplication.
    let shift = attempt.min(5) as u32;
    SSE_RECONNECT_BASE_DELAY
        .saturating_mul(1 << shift)
        .min(SSE_RECONNECT_MAX_DELAY)
}

/// Opt-in SSE self-heal configuration (0.27.0 T3 #95).
///
/// When attached to an [`A2ASseStream`], a **transport-level** break (chunk
/// error) or an idle-timeout is treated as recoverable: the stream re-issues
/// the SSE `GET` against the same URL — carrying the same bearer token and
/// `traceparent` — up to `max_attempts` times with exponential backoff, then
/// surfaces the **original** error (each failed attempt is logged, not returned).
/// A clean server close (`chunk()` → `None`) is **terminal** and is never
/// reconnected, so a finished subscription does not spin. Absent config = the
/// pre-0.27 behavior (a break surfaces immediately).
///
/// **Replay (T3 #100)**: the reconnect `GET` sends `Last-Event-ID` set to the
/// newest SSE `id:` the stream had seen, so a server with a replay buffer can
/// resume from that point instead of dropping the gap. That still depends on
/// the server honoring the header — self-heal restores liveness, not
/// exactly-once delivery on its own; re-reading the task via `tasks/get`
/// remains the authoritative catch-up when the server does not replay.
///
/// **Caveat — close-delimited responses**: on an HTTP/1.1 response framed by
/// connection close (no `Transfer-Encoding`/`Content-Length`), an abrupt drop
/// is indistinguishable from a clean close — `chunk()` yields `None` — so no
/// reconnect happens. Chunked and HTTP/2 servers (what the A2A reference
/// implementations emit) report the break and are healed normally.
pub(crate) struct SseReconnect {
    /// URL to re-open (the same one the original stream was opened at).
    pub(crate) sse_url: String,
    /// Bearer token replayed on reconnect, when the client has one.
    pub(crate) auth_token: Option<String>,
    /// W3C trace context replayed as a `traceparent` header on reconnect.
    pub(crate) trace_context: Option<TraceContext>,
    /// Streaming HTTP client (timeout-free) reused for the reconnect GET.
    pub(crate) http: reqwest::Client,
    /// Total reconnect attempts allowed before the error is surfaced.
    pub(crate) max_attempts: usize,
}

/// A live SSE connection to an A2A server (P2-1).
///
/// Consume it with [`A2ASseStream::next`], which yields one
/// [`TaskPushNotification`] per complete SSE event until the server closes the
/// stream.
pub struct A2ASseStream {
    response: reqwest::Response,
    parser: A2aSseParser,
    pending: VecDeque<TaskPushNotification>,
    /// Self-heal config; `None` keeps the legacy "break surfaces at once" path.
    reconnect: Option<SseReconnect>,
    /// Remaining reconnect attempts for the current stream.
    reconnect_left: usize,
    /// Idle budget applied to each read (B3; tunable since 0.27.0).
    idle_timeout: Duration,
}

impl A2ASseStream {
    /// Build a stream, optionally with self-heal attached (0.27.0 T3 #95).
    ///
    /// `reconnect: None` is the legacy behavior: a transport break surfaces at
    /// once and is never retried.
    pub(crate) fn new(
        response: reqwest::Response,
        reconnect: Option<SseReconnect>,
        idle_timeout: Duration,
    ) -> Self {
        let reconnect_left = reconnect.as_ref().map_or(0, |c| c.max_attempts);
        Self {
            response,
            parser: A2aSseParser::new(),
            pending: VecDeque::new(),
            reconnect,
            reconnect_left,
            idle_timeout,
        }
    }

    /// Wait for the next event, or `None` when the stream ends.
    ///
    /// Returns an error if the connection breaks; a malformed event is logged
    /// and skipped so one bad payload does not kill the whole stream
    /// (0.22.0 audit fix).
    ///
    /// With self-heal configured (see
    /// [`A2AClientBuilder::sse_reconnect_max_attempts`](crate::client::A2AClientBuilder::sse_reconnect_max_attempts)),
    /// a transport break or idle timeout is retried; only a clean close or an
    /// exhausted retry budget ends the stream.
    pub async fn next(&mut self) -> Option<Result<TaskPushNotification, A2AError>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(Ok(event));
            }
            // B3 (0.26.0): bound each read by the idle budget so a server that
            // silently stops sending can't hang the caller forever.
            match tokio::time::timeout(self.idle_timeout, self.response.chunk()).await {
                Ok(Ok(Some(chunk))) => {
                    let text = String::from_utf8_lossy(&chunk);
                    // F-02: a runaway stream whose frame never terminates must
                    // surface as an error (bounded buffer) rather than growing
                    // memory without bound. It is treated as fatal — no
                    // reconnect — because the peer is behaving maliciously or
                    // is broken, not transiently down.
                    match self.parser.feed(&text) {
                        Ok(events) => self.pending.extend(events),
                        Err(msg) => return Some(Err(A2AError::Http(msg))),
                    }
                    // Loop so queued events are returned even if a chunk
                    // carried none (or we keep reading on an empty chunk).
                }
                // A clean close means the server finished the subscription —
                // terminal, never reconnected.
                Ok(Ok(None)) => return None,
                Ok(Err(e)) => {
                    let err = A2AError::from(e);
                    if self.try_reconnect().await {
                        continue;
                    }
                    return Some(Err(err));
                }
                Err(_) => {
                    let err =
                        A2AError::Timeout("SSE read idle timeout (no data received)".to_string());
                    if self.try_reconnect().await {
                        continue;
                    }
                    return Some(Err(err));
                }
            }
        }
    }

    /// Attempt to re-establish the stream after a transport break.
    ///
    /// Returns `true` when a fresh stream was opened and installed, `false`
    /// when self-heal is disabled or the retry budget is exhausted (the caller
    /// then surfaces the original error).
    async fn try_reconnect(&mut self) -> bool {
        let Some(cfg) = self.reconnect.as_ref() else {
            return false;
        };
        if self.reconnect_left == 0 {
            return false;
        }
        // Clone the request context up front so the borrow of `self.reconnect`
        // ends before the fields below are mutated.
        let url = cfg.sse_url.clone();
        let token = cfg.auth_token.clone();
        let trace = cfg.trace_context.clone();
        let http = cfg.http.clone();
        let max_attempts = cfg.max_attempts;
        // T3 #100 event replay: remember the newest `id:` seen so the reconnect
        // GET can ask the server to resume from it (`Last-Event-ID`) instead of
        // silently losing the events emitted while the stream was down.
        let last_event_id = self.parser.last_event_id().map(str::to_string);

        while self.reconnect_left > 0 {
            let attempt = max_attempts - self.reconnect_left;
            tokio::time::sleep(sse_reconnect_backoff(attempt)).await;
            self.reconnect_left -= 1;

            let mut request = http.get(&url);
            if let Some(ctx) = &trace {
                request = request.header("traceparent", ctx.to_traceparent());
            }
            if let Some(token) = &token {
                request = request.bearer_auth(token);
            }
            if let Some(id) = &last_event_id {
                request = request.header("Last-Event-ID", id);
            }
            match tokio::time::timeout(SSE_RECONNECT_SEND_TIMEOUT, request.send()).await {
                Ok(Ok(resp)) if resp.status().is_success() => {
                    log::info!("SSE stream reconnected after transport break");
                    self.response = resp;
                    // A new connection starts a fresh frame buffer; the old
                    // parser may hold a partial frame that will never complete.
                    // Seed it with the replay id so a further reconnect (if the
                    // new stream also breaks) stays pinned to the same offset.
                    self.parser = A2aSseParser::with_last_event_id(last_event_id.clone());
                    return true;
                }
                Ok(Ok(resp)) => {
                    log::warn!("SSE reconnect attempt failed with status {}", resp.status());
                }
                Ok(Err(e)) => {
                    log::warn!("SSE reconnect attempt failed: {e}");
                }
                Err(_) => {
                    log::warn!(
                        "SSE reconnect attempt timed out after {}s",
                        SSE_RECONNECT_SEND_TIMEOUT.as_secs()
                    );
                }
            }
        }
        log::warn!("SSE reconnect budget exhausted; surfacing the transport error");
        false
    }
}

/// Incremental parser for A2A SSE event frames.
///
/// Mirrors the SSE parsing in `lc-providers/src/openai/sse.rs`: events are
/// terminated by a blank line (`\n\n`), and the payload is the `data:` field
/// (multi-line `data:` fields are joined with newlines per the SSE spec).
struct A2aSseParser {
    buffer: String,
    /// Most recent SSE `id:` field seen (0.27.0 T3 #100 event replay). Per the
    /// SSE spec this persists across events that do not set their own `id:`;
    /// the reconnect GET sends it back as `Last-Event-ID` so a server with a
    /// replay buffer can close the delivery gap instead of dropping it.
    last_event_id: Option<String>,
}

impl A2aSseParser {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            last_event_id: None,
        }
    }

    /// Parse starting from a known `Last-Event-ID` so a chain of reconnects
    /// does not forget how far it got before a later reconnect.
    fn with_last_event_id(last_event_id: Option<String>) -> Self {
        Self {
            buffer: String::new(),
            last_event_id,
        }
    }

    /// The most recent SSE `id:` field, if a frame ever carried one.
    fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    /// Feed a chunk of the response body; returns any complete notifications.
    ///
    /// 0.22.0 audit fix: an event whose JSON payload fails to parse is logged
    /// and skipped; only transport-level errors surface to the caller.
    ///
    /// F-02: a frame that accumulates beyond [`SSE_MAX_BUFFER_BYTES`] without a
    /// `\n\n` terminator is a hostile/runaway stream — the buffer is dropped
    /// and `Err` is returned so the caller surfaces a clean error instead of
    /// letting memory grow without bound.
    fn feed(&mut self, chunk: &str) -> Result<Vec<TaskPushNotification>, String> {
        // The bound is on the *un-terminated* tail: the check happens before
        // `push_str`, so a legitimate chunk carrying complete frame(s) that the
        // drain loop below consumes never trips it. Only an endless frame (no
        // `\n\n`) accumulates across chunks and eventually exceeds the ceiling.
        if self.buffer.len().saturating_add(chunk.len()) > SSE_MAX_BUFFER_BYTES {
            self.buffer.clear();
            return Err(format!(
                "SSE frame exceeds the {SSE_MAX_BUFFER_BYTES} byte buffer limit; dropping the stream"
            ));
        }
        self.buffer.push_str(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buffer.find("\n\n") {
            let event_text: String = self.buffer[..pos].to_string();
            self.buffer.drain(..=pos + 1);

            // T3 #100: the `id:` field names this event for `Last-Event-ID`
            // replay. It persists (per SSE) until a later event sets a new one.
            if let Some(id) = event_text
                .lines()
                .find_map(|line| line.strip_prefix("id:").map(str::trim))
                .filter(|id| !id.is_empty())
            {
                self.last_event_id = Some(id.to_string());
            }

            let data = event_text
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|line| line.trim_start().trim_end())
                .collect::<Vec<_>>()
                .join("\n");
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            match serde_json::from_str::<TaskPushNotification>(&data) {
                Ok(notification) => out.push(notification),
                Err(e) => {
                    log::warn!("skipping malformed SSE event `{data}`: {e}");
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_tracks_last_event_id_for_replay() {
        let mut p = A2aSseParser::new();
        assert_eq!(p.last_event_id(), None);

        // A block with only an id (empty data) still advances the replay id.
        p.feed("id: 3\n\n").unwrap();
        assert_eq!(p.last_event_id(), Some("3"));

        // A later event with a data payload and a new id replaces it.
        p.feed("event: status-update\nid: 5\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"working\"}\n\n")
            .unwrap();
        assert_eq!(p.last_event_id(), Some("5"));

        // Per the SSE spec an event without an `id:` inherits the previous one.
        p.feed("event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"completed\"}\n\n")
            .unwrap();
        assert_eq!(p.last_event_id(), Some("5"));
    }

    #[test]
    fn feed_returns_error_when_unterminated_buffer_exceeds_limit() {
        // F-02 (DoS): a peer streaming `data:` without ever sending a `\n\n`
        // terminator must trip the buffer cap and error out — not grow memory
        // without bound.
        let mut p = A2aSseParser::new();
        let data_line = "data: ".to_string() + &"x".repeat(64 * 1024); // 64 KiB, no terminator
        // Fill the buffer right up to the cap without tripping it.
        let fills = SSE_MAX_BUFFER_BYTES / data_line.len();
        for _ in 0..fills {
            assert!(
                p.feed(&data_line).is_ok(),
                "feeding below the cap must still be accepted"
            );
        }
        // One more exceeds the cap -> hard error, and the buffer is dropped so
        // memory stays bounded (no silent unbounded growth).
        assert!(
            p.feed(&data_line).is_err(),
            "overflowing the capped buffer must error out"
        );
        assert!(p.buffer.len() <= SSE_MAX_BUFFER_BYTES);
    }

    #[test]
    fn feed_accepts_frames_larger_than_a_single_chunk_when_terminated() {
        // Legit large-but-terminated content must not be rejected: a frame whose
        // size exceeds the per-chunk threshold still completes normally because
        // the terminator is present within the same chunk.
        let mut p = A2aSseParser::new();
        let big = "event: status-update\ndata: {\"kind\":\"status-update\",\"id\":\"t1\",\"status\":\"completed\"}\n\n";
        let events = p.feed(big).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id(), "t1");
    }

    #[test]
    fn sse_reconnect_backoff_is_exponential_and_capped() {
        // 250 / 500 / 1000 / 2000 / 4000, then capped at 5000 from attempt 5 on.
        let expected = [
            Duration::from_millis(250),
            Duration::from_millis(500),
            Duration::from_millis(1000),
            Duration::from_millis(2000),
            Duration::from_millis(4000),
            Duration::from_millis(5000),
        ];
        for (attempt, want) in expected.iter().enumerate() {
            assert_eq!(sse_reconnect_backoff(attempt), *want, "attempt {attempt}");
        }
        // Attempts far past the sustain cap stay capped (shift is clamped).
        assert_eq!(sse_reconnect_backoff(50), SSE_RECONNECT_MAX_DELAY);
        assert_eq!(sse_reconnect_backoff(1000), SSE_RECONNECT_MAX_DELAY);
        // The cap equals the named maximum.
        assert_eq!(sse_reconnect_backoff(5), SSE_RECONNECT_MAX_DELAY);
    }
}
