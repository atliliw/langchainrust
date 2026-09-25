//! PII detection and redaction rail.
//!
//! [`PiiRedactionGuardrail`] is an output-side rail that rewrites detected personally
//! identifiable information — email addresses, China mobile phone numbers, resident-identity
//! numbers, bank card numbers (Luhn-checked) and IPv4 addresses — to fixed tokens such as
//! `[REDACTED_EMAIL]`. Unlike [`crate::validators::SensitiveInfoGuardrail`], which *blocks*
//! output containing secrets/PII, this rail returns
//! [`Modify`](crate::guardrail::OutputGuardrailResult::Modify): the content flows on, the PII
//! does not.
//!
//! The same type implements [`StreamingOutputGuardrail`]. Streaming cannot rewrite text that
//! has already been emitted, so the rail uses **hold-back buffering**: the last
//! `HOLD_BACK_CHARS` (48) characters are withheld until enough following text proves an
//! identifier is not still being written (e.g. a bank card number arriving in two chunks).
//! The buffered tail is released by [`StreamingOutputGuardrail::flush`] at end of stream.
//! Because the buffer lives on the rail instance, use one rail instance per guarded stream
//! (a [`GuardrailsConfig`](crate::GuardrailsConfig) is consumed by `GuardedAgent::new`).

use std::borrow::Cow;
use std::sync::{Mutex, OnceLock};

use async_trait::async_trait;
use regex::{Captures, Regex};

use crate::guardrail::{
    ChunkAction, ChunkContext, FlushOutput, OutputGuardrail, OutputGuardrailResult,
    StreamingOutputGuardrail,
};
use crate::validators::SensitiveInfoGuardrail;

/// Default hold-back window for streaming redaction: long enough to cover the longest
/// fixed-length identifier (19-digit card plus separators, 18-character identity number).
const HOLD_BACK_CHARS: usize = 48;

/// Categories of personally identifiable information the rail can redact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PiiKind {
    /// Email addresses (`name@example.com`).
    Email,
    /// China mobile phone numbers (`1[3-9]xxxxxxxxx`, optional `+86` prefix).
    Phone,
    /// 18-digit China resident identity numbers (GB 11643 checksum verified).
    NationalId,
    /// 15–19 digit bank / credit card numbers (Luhn checksum verified).
    CreditCard,
    /// Dotted-quad IPv4 addresses with validated octets.
    IpV4,
}

impl PiiKind {
    /// Detection order: longest/most-specific identifiers first so shorter patterns cannot
    /// match inside them.
    fn detection_order() -> [PiiKind; 5] {
        [
            PiiKind::CreditCard,
            PiiKind::NationalId,
            PiiKind::Phone,
            PiiKind::Email,
            PiiKind::IpV4,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            PiiKind::Email => "[REDACTED_EMAIL]",
            PiiKind::Phone => "[REDACTED_PHONE]",
            PiiKind::NationalId => "[REDACTED_NATIONAL_ID]",
            PiiKind::CreditCard => "[REDACTED_CREDIT_CARD]",
            PiiKind::IpV4 => "[REDACTED_IP]",
        }
    }
}

fn card_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // M-1: 13–19 digits, aligned with the blocking-side `CREDIT_CARD_RE` and `luhn_check`
    // (the checksum filters candidates, so widening the scan to 13–19 is low-false-positive).
    RE.get_or_init(|| {
        Regex::new(r"\b\d(?:[ -]?\d){12,18}\b").expect("static regex literal must compile")
    })
}

fn national_id_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b\d{17}[\dXx]\b").expect("static regex literal must compile"))
}

fn phone_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `\b` (not look-around, which the `regex` crate lacks) keeps an 11-digit phone from
    // matching inside a longer digit run. Country-code-prefixed forms (`+86`, glued or
    // hyphenated) are handled by [`phone_prefixed_re`] in an earlier pass, because a glued
    // prefix leaves no word boundary before the `1`.
    RE.get_or_init(|| Regex::new(r"\b1[3-9]\d{9}\b").expect("static regex literal must compile"))
}

fn phone_prefixed_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Group 1 is a consumed left boundary (start of text or a non-word, non-`+` char) so a
    // prefixed number cannot match inside a longer digit run; the closure reinserts it.
    RE.get_or_init(|| {
        Regex::new(r"(^|[^\w+])(\+?86[-\s]?1[3-9]\d{9})\b")
            .expect("static regex literal must compile")
    })
}

fn email_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
            .expect("static regex literal must compile")
    })
}

fn ipv4_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // 左边缘刻意不加 `\b`(K5):紧贴的前置数字不得阻止发现真四段——`5`192.168.0.1` 里
    // 的 `192.168.0.1` 必须命中。右边缘保留 `\b`:一个四段不会继续接数字/点而不破坏八位组
    // 语法。左侧边界延续(前置 `.`)与右侧延伸(后置数字/点)都在替换闭包内拒绝——
    // `regex` crate 不支持 look-around。
    RE.get_or_init(|| {
        Regex::new(r"(?:\d{1,3}\.){3}\d{1,3}\b").expect("static regex literal must compile")
    })
}

/// Output guardrail that redacts PII instead of blocking.
///
/// Construct with [`new`](Self::new) (all categories) and narrow with
/// [`only`](Self::only)/[`disable`](Self::disable). Register the same instance as an output
/// rail (whole-output rewrite) and, when streaming, as a streaming rail (hold-back rewrite).
#[derive(Debug)]
pub struct PiiRedactionGuardrail {
    enabled: Vec<PiiKind>,
    hold_back: usize,
    // streaming state: raw text seen / chars of the redacted stream already released.
    stream_raw: Mutex<String>,
    stream_released: Mutex<usize>,
}

impl Default for PiiRedactionGuardrail {
    fn default() -> Self {
        Self::new()
    }
}

impl PiiRedactionGuardrail {
    /// Creates the rail with every [`PiiKind`] enabled.
    pub fn new() -> Self {
        Self {
            enabled: PiiKind::detection_order().to_vec(),
            hold_back: HOLD_BACK_CHARS,
            stream_raw: Mutex::new(String::new()),
            stream_released: Mutex::new(0),
        }
    }

    /// Restricts redaction to the given categories.
    pub fn only(mut self, kinds: impl IntoIterator<Item = PiiKind>) -> Self {
        let wanted: Vec<PiiKind> = kinds.into_iter().collect();
        self.enabled = PiiKind::detection_order()
            .into_iter()
            .filter(|k| wanted.contains(k))
            .collect();
        self
    }

    /// Disables one category.
    pub fn disable(mut self, kind: PiiKind) -> Self {
        self.enabled.retain(|k| *k != kind);
        self
    }

    /// Sets the streaming hold-back window (characters). The default (48,
    /// `HOLD_BACK_CHARS`) covers every fixed-length identifier; shrink it to reduce output
    /// latency at the cost of identifiers split unusually far across chunks.
    pub fn with_hold_back(mut self, chars: usize) -> Self {
        self.hold_back = chars;
        self
    }

    /// Resets streaming state so the rail can be reused across independent streams (K2).
    ///
    /// The hold-back buffer (plus the release cursor) belongs to exactly one stream; reusing a
    /// rail instance without resetting would let the previous stream's bytes leak into the next
    /// redaction window. The framework calls this once per `invoke_stream` via
    /// [`StreamingOutputGuardrail::reset`]; a caller driving the rail manually should call it
    /// before each stream.
    pub fn reset(&self) {
        if let Ok(mut raw) = self.stream_raw.lock() {
            raw.clear();
        }
        if let Ok(mut released) = self.stream_released.lock() {
            *released = 0;
        }
    }

    /// Redacts every enabled category in `text`, returning the rewritten string.
    pub fn redact(&self, text: &str) -> String {
        let mut out = text.to_string();
        for kind in &self.enabled {
            out = match kind {
                PiiKind::CreditCard => card_re()
                    .replace_all(&out, |caps: &Captures| {
                        let matched = caps
                            .get(0)
                            .expect("capture group 0 always present on a Regex match")
                            .as_str();
                        let digits: String =
                            matched.chars().filter(|c| c.is_ascii_digit()).collect();
                        if SensitiveInfoGuardrail::luhn_check(&digits) {
                            Cow::Borrowed(PiiKind::CreditCard.label())
                        } else {
                            Cow::Owned(matched.to_string())
                        }
                    })
                    .into_owned(),
                PiiKind::NationalId => national_id_re()
                    .replace_all(&out, |caps: &Captures| {
                        let matched = caps
                            .get(0)
                            .expect("capture group 0 always present on a Regex match")
                            .as_str();
                        if valid_china_id(matched) {
                            Cow::Borrowed(PiiKind::NationalId.label())
                        } else {
                            Cow::Owned(matched.to_string())
                        }
                    })
                    .into_owned(),
                PiiKind::Phone => {
                    let prefixed = phone_prefixed_re()
                        .replace_all(&out, |caps: &Captures| {
                            format!("{}{}", &caps[1], PiiKind::Phone.label())
                        })
                        .into_owned();
                    phone_re()
                        .replace_all(&prefixed, PiiKind::Phone.label())
                        .into_owned()
                }
                PiiKind::Email => email_re()
                    .replace_all(&out, PiiKind::Email.label())
                    .into_owned(),
                PiiKind::IpV4 => ipv4_re()
                    .replace_all(&out, |caps: &Captures| {
                        let m = caps
                            .get(0)
                            .expect("capture group 0 always present on a Regex match");
                        let bytes = out.as_bytes();
                        // 左边界只在前置 `.` 时拒绝(处在更长的点分序列中间);前置**数字**不拒绝
                        // (K5:`5`192.168.0.1` 仍要 redact 真四段 `192.168.0.1`)。右边界在后置
                        // `.`/数字时拒绝——四段不会继续接数字/点而不破坏八位组语法。
                        let left_is_dot = m
                            .start()
                            .checked_sub(1)
                            .map(|i| bytes[i] == b'.')
                            .unwrap_or(false);
                        let right_extends = Some(m.end())
                            .filter(|&e| e < bytes.len())
                            .map(|e| matches!(bytes[e], b'.' | b'0'..=b'9'))
                            .unwrap_or(false);
                        let bounded = !left_is_dot && !right_extends;
                        if bounded && valid_ipv4(m.as_str()) {
                            Cow::Borrowed(PiiKind::IpV4.label())
                        } else {
                            Cow::Owned(m.as_str().to_string())
                        }
                    })
                    .into_owned(),
            };
        }
        out
    }
}

#[async_trait]
impl OutputGuardrail for PiiRedactionGuardrail {
    fn name(&self) -> &str {
        "PiiRedactionGuardrail"
    }

    async fn validate(&self, output: &str) -> OutputGuardrailResult {
        let redacted = self.redact(output);
        if redacted == output {
            OutputGuardrailResult::Pass
        } else {
            OutputGuardrailResult::Modify {
                new_value: redacted,
            }
        }
    }
}

#[async_trait]
impl StreamingOutputGuardrail for PiiRedactionGuardrail {
    fn name(&self) -> &str {
        "PiiRedactionGuardrail"
    }

    async fn validate_chunk(&self, ctx: &ChunkContext<'_>) -> ChunkAction {
        self.stream_raw
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_str(ctx.token);
        let redacted = {
            let raw = self.stream_raw.lock().unwrap_or_else(|e| e.into_inner());
            self.redact(&raw)
        };

        let total = redacted.chars().count();
        // K1: pattern-aware hold-back — on top of the configured window, additionally withhold
        // the trailing run of chars that could still be an in-progress identifier (digits and
        // their separators for cards/ids/phones/ipv4; an unfinished email word run). Without
        // this, a fixed window smaller than the identifier lets its raw prefix slip out before
        // enough following text proves it whole. Computed on the already-redacted text: a
        // completed identifier is already shrunk to a `[REDACTED_*]` label and no longer holds.
        let pending = pending_hold(&redacted);
        let release = total.saturating_sub(self.hold_back.max(pending));
        let mut released = self
            .stream_released
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if release <= *released {
            // the new token is still inside the hold-back window: emit nothing yet.
            return ChunkAction::Replace(String::new());
        }
        let delta: String = redacted
            .chars()
            .skip(*released)
            .take(release - *released)
            .collect();
        *released = release;
        if delta == ctx.token {
            ChunkAction::Pass
        } else {
            ChunkAction::Replace(delta)
        }
    }

    async fn flush(&self) -> FlushOutput {
        let raw = std::mem::take(&mut *self.stream_raw.lock().unwrap_or_else(|e| e.into_inner()));
        let mut released_count = self
            .stream_released
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let redacted = self.redact(&raw);
        let tail: String = redacted.chars().skip(*released_count).collect();
        *released_count = 0;
        if tail.is_empty() {
            FlushOutput::Empty
        } else if redacted != raw {
            // the buffered tail contains a redaction label: this is an intervention.
            FlushOutput::Rewritten(tail)
        } else {
            FlushOutput::Release(tail)
        }
    }

    fn reset(&self) {
        PiiRedactionGuardrail::reset(self);
    }
}

/// K1: returns how many chars at `text`'s tail could still be part of an unfinished PII
/// identifier. Two backward scans cover every enabled category:
///
/// - **Numeric** (card / national id / phone / ipv4): swallows `[0-9 .\-+]` (digits plus the
///   identifier-internal separators — space, hyphen, dot, plus), and counts only when the run
///   contains at least one digit (a pure separator/space tail should not hold back output).
/// - **Email**: swallows the email charset (`A-Za-z0-9._%+\-@`) and counts only when an `@`
///   was seen (a digit/letter run without `@` cannot be an in-progress email).
///
/// Take the max of both. It runs on the **already-redacted** text: a finished identifier is
/// already a `[REDACTED_*]` label, so it never over-holds for a still-whole identifier.
fn pending_hold(text: &str) -> usize {
    let mut hold = 0usize;

    // numeric tail: digits and their separators, but only if at least one digit is present.
    let mut digit_run = 0;
    let mut has_digit = false;
    for c in text.chars().rev() {
        match c {
            '0'..='9' => {
                has_digit = true;
                digit_run += 1;
            }
            ' ' | '-' | '.' | '+' => digit_run += 1,
            _ => break,
        }
    }
    if has_digit {
        hold = hold.max(digit_run);
    }

    // email tail: email charset, but only if an '@' was seen (and at least one char left of it).
    let mut email_run = 0;
    let mut seen_at = false;
    for c in text.chars().rev() {
        if c == '@' {
            seen_at = true;
            email_run += 1;
        } else if matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9' | '.' | '_' | '%' | '+' | '-') {
            email_run += 1;
        } else {
            break;
        }
    }
    if seen_at && email_run >= 2 {
        hold = hold.max(email_run);
    }

    hold
}

/// GB 11643 checksum for 18-digit resident identity numbers (last char may be X).
fn valid_china_id(number: &str) -> bool {
    let bytes = number.as_bytes();
    if bytes.len() != 18 {
        return false;
    }
    const WEIGHTS: [u32; 17] = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2];
    const CHECK: [char; 11] = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2'];
    let mut sum = 0u32;
    for (i, w) in WEIGHTS.iter().enumerate() {
        match (bytes[i] as char).to_digit(10) {
            Some(d) => sum += d * w,
            None => return false,
        }
    }
    let last = bytes[17] as char;
    CHECK[(sum % 11) as usize].eq_ignore_ascii_case(&last)
}

/// Validates every octet of a dotted-quad IPv4 literal.
fn valid_ipv4(text: &str) -> bool {
    let mut octets = 0;
    for part in text.split('.') {
        match part.parse::<u16>() {
            Ok(n) if n <= 255 => octets += 1,
            _ => return false,
        }
    }
    octets == 4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn redact(text: &str) -> String {
        PiiRedactionGuardrail::new().redact(text)
    }

    #[test]
    fn redacts_email() {
        assert_eq!(
            redact("reach me at jane.doe@example.com please"),
            "reach me at [REDACTED_EMAIL] please"
        );
    }

    #[test]
    fn redacts_china_mobile() {
        assert_eq!(redact("call 13812345678 now"), "call [REDACTED_PHONE] now");
        assert_eq!(redact("tel +86-13912345678 x"), "tel [REDACTED_PHONE] x");
        assert_eq!(redact("tel +8613912345678 x"), "tel [REDACTED_PHONE] x");
        assert_eq!(redact("tel 86-13912345678 x"), "tel [REDACTED_PHONE] x");
        // an 11-digit substring of a longer digit run is not a phone number
        assert_eq!(redact("id=1231381234567890"), "id=1231381234567890");
    }

    #[test]
    fn redacts_national_id_only_with_checksum() {
        // structurally valid sample number (checksum-verified, not a real person)
        let valid = "11010519491231002X";
        assert!(valid_china_id(valid));
        assert!(redact(&format!("id {valid}")).contains("[REDACTED_NATIONAL_ID]"));
        // same shape, wrong checksum stays untouched
        assert!(!valid_china_id("110105194912310029"));
        assert_eq!(redact("n 110105194912310029"), "n 110105194912310029");
    }

    #[test]
    fn redacts_luhn_card() {
        // 4111 1111 1111 1111 is the standard Visa test PAN (passes Luhn).
        assert!(redact("card 4111 1111 1111 1111 ok").contains("[REDACTED_CREDIT_CARD]"));
        // same length, Luhn-invalid stays untouched
        assert_eq!(redact("x 4111 1111 1111 1112 y"), "x 4111 1111 1111 1112 y");
    }

    #[test]
    fn redacts_ipv4() {
        assert_eq!(redact("host 192.168.0.1!"), "host [REDACTED_IP]!");
        assert_eq!(redact("bad 999.1.1.1"), "bad 999.1.1.1");
        // K5: a glued preceding digit must not hide the true quad inside a longer digit run.
        assert_eq!(redact("n 5192.168.0.1 x"), "n 5[REDACTED_IP] x");
        // a quad that is a slice of a longer dotted run still stays untouched
        assert_eq!(redact("1.2.3.4.5"), "1.2.3.4.5");
    }

    #[test]
    fn category_selection_is_respected() {
        let rail = PiiRedactionGuardrail::new().disable(PiiKind::Email);
        let out = rail.redact("a@b.com 13812345678");
        assert!(out.contains("a@b.com"));
        assert!(out.contains("[REDACTED_PHONE]"));
    }

    #[tokio::test]
    async fn output_rail_modifies() {
        let rail = PiiRedactionGuardrail::new();
        match rail.validate("mail a@b.com").await {
            OutputGuardrailResult::Modify { new_value } => {
                assert_eq!(new_value, "mail [REDACTED_EMAIL]")
            }
            other => panic!("expected Modify, got {other:?}"),
        }
        assert!(rail.validate("nothing here").await.is_pass());
    }

    #[tokio::test]
    async fn streaming_redacts_split_identifier_and_flushes_holdback() {
        // split the phone number across chunks: hold-back guarantees it cannot leak early
        let rail = PiiRedactionGuardrail::new().with_hold_back(8);
        let mut emitted = String::new();
        for token in ["call ", "138", "1234", "5678", " thanks"] {
            let ctx = ChunkContext {
                token,
                window: token,
                full: token,
            };
            match rail.validate_chunk(&ctx).await {
                ChunkAction::Pass => emitted.push_str(token),
                ChunkAction::Replace(delta) => emitted.push_str(&delta),
                ChunkAction::Block => panic!("unexpected block"),
            }
        }
        let flushed = rail.flush().await;
        assert!(matches!(flushed, FlushOutput::Rewritten(_)));
        if let Some(tail) = flushed.into_text() {
            emitted.push_str(&tail);
        }
        assert!(
            !emitted.contains("13812345678"),
            "raw phone leaked: {emitted}"
        );
        assert!(emitted.contains("[REDACTED_PHONE]"), "got: {emitted}");
        // no other characters lost apart from the redacted number
        assert!(emitted.contains("call"));
        assert!(emitted.contains("thanks"));
    }

    #[tokio::test]
    async fn streaming_passthrough_when_no_pii() {
        let rail = PiiRedactionGuardrail::new().with_hold_back(4);
        let mut emitted = String::new();
        for token in ["ab", "cd", "ef"] {
            let ctx = ChunkContext {
                token,
                window: token,
                full: token,
            };
            match rail.validate_chunk(&ctx).await {
                ChunkAction::Pass => emitted.push_str(token),
                ChunkAction::Replace(delta) => emitted.push_str(&delta),
                ChunkAction::Block => panic!("unexpected block"),
            }
        }
        // a clean stream releases its buffered tail verbatim, flagged as no intervention
        let flushed = rail.flush().await;
        assert!(matches!(flushed, FlushOutput::Release(_)));
        if let Some(tail) = flushed.into_text() {
            emitted.push_str(&tail);
        }
        assert_eq!(emitted, "abcdef");
    }

    #[tokio::test]
    async fn streaming_pattern_aware_hold_never_leaks_raw_prefix() {
        // hold-back (4) smaller than the split gap: with only the fixed window, chunk 3
        // ("call 1381234") would release 4 chars from offset 4 and leak the raw digits "1381"
        // before the number is whole. The pattern-aware hold withholds the trailing digit run,
        // so no raw digit of the phone can ever be emitted — only the redaction label.
        let rail = PiiRedactionGuardrail::new().with_hold_back(4);
        let mut emitted = String::new();
        for token in ["call ", "138", "1234", "5678", " thanks"] {
            let ctx = ChunkContext {
                token,
                window: token,
                full: token,
            };
            match rail.validate_chunk(&ctx).await {
                ChunkAction::Pass => emitted.push_str(token),
                ChunkAction::Replace(delta) => emitted.push_str(&delta),
                ChunkAction::Block => panic!("unexpected block"),
            }
            assert!(
                !emitted.contains("138"),
                "raw phone prefix leaked on chunk {token:?}: {emitted:?}"
            );
        }
        let flushed = rail.flush().await;
        assert!(matches!(flushed, FlushOutput::Rewritten(_)));
        if let Some(tail) = flushed.into_text() {
            emitted.push_str(&tail);
        }
        assert_eq!(emitted, "call [REDACTED_PHONE] thanks");
    }

    #[tokio::test]
    async fn streaming_reset_isolates_streams() {
        // K2: a reused rail instance must not leak one stream's hold-back buffer into the next.
        let rail = PiiRedactionGuardrail::new().with_hold_back(4);
        // stream 1: a phone number arrives split across chunks, stream left unflushed so the
        // raw "13812345678" still sits in the buffer when we move on.
        let mut out1 = String::new();
        for token in ["138", "12345", "678"] {
            let ctx = ChunkContext {
                token,
                window: token,
                full: token,
            };
            match rail.validate_chunk(&ctx).await {
                ChunkAction::Pass => out1.push_str(token),
                ChunkAction::Replace(d) => out1.push_str(&d),
                ChunkAction::Block => panic!("unexpected block"),
            }
        }
        // without reset, stream 2 would inherit stream 1's raw bytes and release cursor.
        rail.reset();
        let mut out2 = String::new();
        for token in ["ab", "cd", "ef"] {
            let ctx = ChunkContext {
                token,
                window: token,
                full: token,
            };
            match rail.validate_chunk(&ctx).await {
                ChunkAction::Pass => out2.push_str(token),
                ChunkAction::Replace(d) => out2.push_str(&d),
                ChunkAction::Block => panic!("unexpected block"),
            }
        }
        let flushed = rail.flush().await;
        assert!(matches!(flushed, FlushOutput::Release(_)));
        if let Some(tail) = flushed.into_text() {
            out2.push_str(&tail);
        }
        assert_eq!(out2, "abcdef");
    }
}
