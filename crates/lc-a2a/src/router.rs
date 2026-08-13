//! Skill-based chain routing (P2-4).
//!
//! `A2AServer` can be handed a [`SkillRouter`] so that incoming `tasks/send`
//! requests carrying a `skillId` are dispatched to a different underlying
//! chain than the default one, based on the skills advertised on the agent
//! card. [`SkillMapRouter`] is the concrete static mapping shipped here.

use std::collections::HashMap;
use std::sync::Arc;

use lc_chains::base::BaseChain;

/// Resolves a chain for a requested skill id (P2-4).
///
/// Return `None` to fall back to the server's default chain. Implementations
/// must be cheap and infallible — they run on every `tasks/send`.
pub trait SkillRouter: Send + Sync {
    /// The chain that should handle `skill_id`, if any.
    fn chain_for(&self, skill_id: &str) -> Option<Arc<dyn BaseChain>>;
}

/// Static `skill_id -> chain` mapping.
///
/// The keys should match the skill ids advertised on the agent card so that
/// clients can discover which skills are routable.
#[derive(Default)]
pub struct SkillMapRouter {
    skills: HashMap<String, Arc<dyn BaseChain>>,
}

impl SkillMapRouter {
    /// Create an empty router (all requests fall through to the default chain).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a chain for a skill id.
    pub fn with_skill(mut self, skill_id: impl Into<String>, chain: Arc<dyn BaseChain>) -> Self {
        self.skills.insert(skill_id.into(), chain);
        self
    }
}

impl SkillRouter for SkillMapRouter {
    fn chain_for(&self, skill_id: &str) -> Option<Arc<dyn BaseChain>> {
        self.skills.get(skill_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_chains::base::{ChainError, ChainResult};
    use serde_json::Value;

    /// A trivial chain that names itself, so routing can be observed.
    struct NamedChain(String);

    #[async_trait::async_trait]
    impl BaseChain for NamedChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["input"]
        }

        fn output_keys(&self) -> Vec<&str> {
            vec!["output"]
        }

        async fn invoke(&self, _inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            let mut out = HashMap::new();
            out.insert("output".to_string(), Value::String(self.0.clone()));
            Ok(out)
        }

        fn name(&self) -> &str {
            &self.0
        }
    }

    fn arc_named(name: &str) -> Arc<dyn BaseChain> {
        Arc::new(NamedChain(name.to_string()))
    }

    #[test]
    fn empty_router_falls_through() {
        let router = SkillMapRouter::new();
        assert!(router.chain_for("anything").is_none());
    }

    #[test]
    fn routes_by_skill_id() {
        let router = SkillMapRouter::new()
            .with_skill("research", arc_named("research-chain"))
            .with_skill("summarize", arc_named("summary-chain"));

        assert!(router.chain_for("research").is_some());
        assert!(router.chain_for("summarize").is_some());
        // Unknown skill falls through to the default chain.
        assert!(router.chain_for("nope").is_none());
        // Distinct chains per skill.
        let a = router.chain_for("research").unwrap();
        let b = router.chain_for("summarize").unwrap();
        assert_ne!(a.name(), b.name());
    }
}
