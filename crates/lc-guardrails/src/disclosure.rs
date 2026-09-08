// lc-guardrails/src/disclosure.rs
//! AI-interaction transparency disclosure (0.21.0 S6.3, EU AI Act Art. 50).
//!
//! EU AI Act Article 50 (effective 2026-08-02) requires that people interacting
//! with an AI system are informed they are interacting with AI. This module
//! provides the *capability* — generate the disclosure statement, route it to a
//! sink (audit log / callback surface) — without forcing any particular UX.
//!
//! The disclosure is fire-and-forget from the caller's perspective: it never
//! blocks or fails the surrounding flow (a disclosure failure is a warning,
//! not an agent error).

use crate::audit::{AuditSink, FileAuditSink};
use crate::runner::GuardrailViolation;
use std::sync::Arc;

/// Default disclosure statement (Art. 50: "interacting with AI").
pub const DEFAULT_DISCLOSURE: &str =
    "Notice: you are interacting with an AI system. Responses are AI-generated and may contain errors.";

/// Disclosure configuration.
#[derive(Debug, Clone)]
pub struct DisclosureConfig {
    /// The disclosure statement (template; `{system}` placeholder is replaced
    /// with the system name when provided).
    pub statement: String,
    /// Whether disclosure has been shown for the session already (Art. 50
    /// allows a one-time notice for a continuous interaction; the caller
    /// decides the policy — this flag just records it in the record).
    pub session_disclosed: bool,
}

impl Default for DisclosureConfig {
    fn default() -> Self {
        Self {
            statement: DEFAULT_DISCLOSURE.to_string(),
            session_disclosed: false,
        }
    }
}

impl DisclosureConfig {
    /// Creates a config with the default statement.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overrides the disclosure statement (`{system}` placeholder supported).
    pub fn with_statement(mut self, statement: impl Into<String>) -> Self {
        self.statement = statement.into();
        self
    }

    /// Marks this as a repeat interaction within an already-disclosed session.
    pub fn with_session_disclosed(mut self, session_disclosed: bool) -> Self {
        self.session_disclosed = session_disclosed;
        self
    }

    /// Renders the disclosure for a system name (`{system}` replaced).
    pub fn render(&self, system: &str) -> String {
        self.statement.replace("{system}", system)
    }
}

/// Emits the AI-interaction disclosure to the audit surface.
///
/// Fire-and-forget: a sink failure is logged as a warning and never propagates
/// (a disclosure problem must not break the user flow). Returns the rendered
/// statement so callers can attach it to their own UX surface (chat preamble,
/// UI banner, response metadata) as well.
pub async fn disclose(
    audit: &Arc<dyn AuditSink>,
    config: &DisclosureConfig,
    system: &str,
    trace_id: Option<&str>,
) -> String {
    let text = config.render(system);
    let violation = GuardrailViolation {
        guardrail_name: "ai_disclosure".to_string(),
        stage: "disclosure".to_string(),
        reason: format!(
            "disclosed system={} session_disclosed={} trace_id={}",
            system,
            config.session_disclosed,
            trace_id.unwrap_or("-")
        ),
    };
    audit.record(&violation).await;
    text
}

/// Convenience: disclose to a file audit sink.
pub async fn disclose_to_file(
    audit: &Arc<FileAuditSink>,
    config: &DisclosureConfig,
    system: &str,
    trace_id: Option<&str>,
) -> String {
    let text = config.render(system);
    let violation = GuardrailViolation {
        guardrail_name: "ai_disclosure".to_string(),
        stage: "disclosure".to_string(),
        reason: format!(
            "disclosed system={} session_disclosed={} trace_id={}",
            system,
            config.session_disclosed,
            trace_id.unwrap_or("-")
        ),
    };
    audit.record(&violation).await;
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MemoryAudit {
        records: Mutex<Vec<GuardrailViolation>>,
    }

    #[async_trait::async_trait]
    impl AuditSink for MemoryAudit {
        fn name(&self) -> &str {
            "memory"
        }
        async fn record(&self, violation: &GuardrailViolation) {
            self.records.lock().unwrap().push(violation.clone());
        }
    }
    #[test]
    fn default_statement_render() {
        let config = DisclosureConfig::new();
        let text = config.render("support-bot");
        assert!(text.contains("AI"), "default statement mentions AI");
        assert!(!text.contains("{system}"), "placeholder fully replaced");
    }

    /// Custom statement with `{system}` placeholder.
    #[test]
    fn custom_statement_with_placeholder() {
        let config = DisclosureConfig::new().with_statement("{system} is an AI assistant.");
        assert_eq!(config.render("helper"), "helper is an AI assistant.");
    }

    /// Disclose records to the sink and returns the rendered text.
    #[tokio::test]
    async fn disclose_records_and_returns_text() {
        let concrete = Arc::new(MemoryAudit::default());
        let audit: Arc<dyn AuditSink> = concrete.clone();
        let config = DisclosureConfig::new();
        let text = disclose(&audit, &config, "bot", Some("trace-1")).await;
        assert!(text.contains("AI"));
        let records = concrete.records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].guardrail_name, "ai_disclosure");
        assert_eq!(records[0].stage, "disclosure");
        assert!(records[0].reason.contains("trace-1"));
    }

    /// `session_disclosed` is carried in the record (policy hook for one-time notices).
    #[tokio::test]
    async fn session_disclosed_flag_recorded() {
        let concrete = Arc::new(MemoryAudit::default());
        let audit: Arc<dyn AuditSink> = concrete.clone();
        let config = DisclosureConfig::new().with_session_disclosed(true);
        let _ = disclose(&audit, &config, "bot", None).await;
        let records = concrete.records.lock().unwrap();
        assert!(records[0].reason.contains("session_disclosed=true"));
    }
}
