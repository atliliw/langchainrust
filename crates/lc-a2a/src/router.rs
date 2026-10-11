//! Skill-based chain routing (P2-4).
//!
//! `A2AServer` can be handed a [`SkillRouter`] so that incoming `tasks/send`
//! requests carrying a `skillId` are dispatched to a different underlying
//! chain than the default one, based on the skills advertised on the agent
//! card. [`SkillMapRouter`] is a concrete static mapping;
//! [`SkillAffinityRouter`] (T3) matches the requested skill against a set of
//! declared [`AgentSkill`]s by affinity and serves it with the best-matching
//! chain — used for choosing the chain a resumed task returns to.

use std::collections::HashMap;
use std::sync::Arc;

use lc_chains::base::BaseChain;

use crate::protocol::{AgentCard, AgentSkill};
use crate::scale::skill_affinity;

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

/// Error returned by [`SkillAffinityRouter::select`] (T3).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SkillAffinityError {
    /// No declared skill matched `skill_id` at or above the configured
    /// minimum score.
    #[error("no skill affinity match for `{skill_id}` (best {best:.3} < required {min:.3})")]
    NoMatch {
        /// The requested skill id.
        skill_id: String,
        /// The best affinity score found.
        best: f64,
        /// The configured minimum acceptable score.
        min: f64,
        /// The names of the declared skills that were considered.
        considered: Vec<String>,
    },
}

/// One declared skill and the chain that serves it (T3).
#[derive(Clone)]
struct AffinityRoute {
    skill: AgentSkill,
    chain: Arc<dyn BaseChain>,
}

/// Resolves a chain by **skill affinity** (T3 skill-affinity routing).
///
/// Instead of a key->chain map, [`SkillAffinityRouter`] holds a set of declared
/// [`AgentSkill`]s (typically from an [`AgentCard`]) each paired with the chain
/// that serves it. [`SkillAffinityRouter::select`] scores the requested skill
/// id against every declared skill with [`skill_affinity`] and returns the
/// best-matching chain — exact skill-id matches always win — or an error /
/// configured fallback when nothing clears [`SkillAffinityRouter::with_min_score`].
///
/// This is what a resumed task should use to pick its chain: the card's stated
/// skills (not a random pick, not "the first one") determine who serves it.
pub struct SkillAffinityRouter {
    routes: Vec<AffinityRoute>,
    /// The default declarative threshold. Codes at 0.5, so only exact id /
    /// exact name / substring matches qualify by default; lower it for fuzzy
    /// description-overlap matching (the token branch tops out below 0.6).
    min_score: f64,
    fallback: Option<Arc<dyn BaseChain>>,
}

impl Default for SkillAffinityRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillAffinityRouter {
    /// An empty router.
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            min_score: 0.5,
            fallback: None,
        }
    }

    /// Lower/raise the skill-affinity acceptance threshold (default `0.5`).
    ///
    /// Exact id/name and substring matches score `>= 0.7`, so the default
    /// accepts only those. Set below `0.6` (e.g. `0.3`) to admit fuzzy
    /// bag-of-words overlaps on the skill description.
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }

    /// Serve a fixed `chain` when no declared skill clears the threshold,
    /// instead of returning an error.
    pub fn with_fallback(mut self, chain: Arc<dyn BaseChain>) -> Self {
        self.fallback = Some(chain);
        self
    }

    /// Register a declared [`AgentSkill`] served by `chain`.
    pub fn with_route(mut self, skill: AgentSkill, chain: Arc<dyn BaseChain>) -> Self {
        self.routes.push(AffinityRoute { skill, chain });
        self
    }

    /// Register every skill declared on an [`AgentCard`], resolving each to a
    /// chain via `resolve` after the server's skill map.
    ///
    /// Declared skills without a resolvable chain are skipped, so the router
    /// never matches against a skill it could not honour.
    pub fn with_routes_from_card<F>(mut self, card: &AgentCard, mut resolve: F) -> Self
    where
        F: FnMut(&AgentSkill) -> Option<Arc<dyn BaseChain>>,
    {
        for skill in &card.skills {
            if let Some(chain) = resolve(skill) {
                self.routes.push(AffinityRoute {
                    skill: skill.clone(),
                    chain,
                });
            }
        }
        self
    }

    /// The number of declared skill routes.
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// Whether no skill route is configured.
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    /// The configured minimum acceptance score.
    pub fn min_score(&self) -> f64 {
        self.min_score
    }

    /// Choose the chain whose declared skill best matches `skill_id`.
    ///
    /// Matching is deterministic: ties go to the first-registered route.
    /// On no match at or above `min_score`, returns the configured fallback
    /// (if any) or [`SkillAffinityError::NoMatch`]. The no-match path is never
    /// silent — it logs a warning so a mismatched resume is diagnosable.
    pub fn select(&self, skill_id: &str) -> Result<Arc<dyn BaseChain>, SkillAffinityError> {
        let mut best: Option<(f64, &AffinityRoute)> = None;
        for route in &self.routes {
            let score = skill_affinity(skill_id, &route.skill);
            match best {
                Some((b, _)) if b >= score => {}
                _ => best = Some((score, route)),
            }
        }
        let (score, route) = match best {
            Some(x) => x,
            None => return self.serve_none(skill_id),
        };
        if score < self.min_score {
            return self.serve_none(skill_id);
        }
        Ok(route.chain.clone())
    }

    /// Handle a no-match outcome: warn, then prefer the configured fallback
    /// over the error.
    fn serve_none(&self, skill_id: &str) -> Result<Arc<dyn BaseChain>, SkillAffinityError> {
        let considered: Vec<String> = self
            .routes
            .iter()
            .map(|r| {
                let max = skill_affinity(skill_id, &r.skill);
                format!("{}={max:.3}", r.skill.id)
            })
            .collect();
        log::warn!(
            "skill affinity: no route for `{skill_id}` clears thresholds {:.3}; considered [{}]{}",
            self.min_score,
            considered.join(", "),
            if self.fallback.is_some() { "; using fallback" } else { "" },
        );
        match &self.fallback {
            Some(chain) => Ok(chain.clone()),
            None => Err(SkillAffinityError::NoMatch {
                skill_id: skill_id.to_string(),
                best: 0.0,
                min: self.min_score,
                considered,
            }),
        }
    }
}

impl SkillRouter for SkillAffinityRouter {
    fn chain_for(&self, skill_id: &str) -> Option<Arc<dyn BaseChain>> {
        // `select` logs the warn; here a Return{(None)} lets the server fall
        // back to its default chain.
        self.select(skill_id).ok()
    }
}

impl std::fmt::Debug for SkillAffinityRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillAffinityRouter")
            .field("routes", &self.routes.len())
            .field("min_score", &self.min_score)
            .field("has_fallback", &self.fallback.is_some())
            .finish_non_exhaustive()
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
