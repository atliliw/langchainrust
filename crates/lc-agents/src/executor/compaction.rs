// lc-agents/src/executor/compaction.rs
//! Context compaction for long-running agent loops (0.21.0 S6.1).
//!
//! Long sessions accumulate `intermediate_steps` until the context window is
//! wasted on stale tool observations (context rot). Compaction drops the
//! oldest steps — **always at whole-step boundaries** so the model never sees
//! an orphaned action without its observation (each `AgentStep` bundles the
//! action + observation, which is the atomic unit here).
//!
//! Follows the [`super::budget::BudgetConfig`] discipline:
//! - configuration is all-off by default (`None` on the executor = no
//!   compaction, zero behavior change);
//! - the same semantics run in the invoke and stream paths;
//! - trigger and strategy are pure functions, unit-testable without an agent.
//!
//! `RecursiveSummarization` (LLM-summarize dropped turns into a synthetic
//! prefix) is deliberately NOT in this version — it needs an LLM call, cost
//! accounting and quality tuning. The strategy enum keeps room for it.

use crate::types::AgentStep;

/// When to compact. Checked before every `plan()` round.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CompactionTrigger {
    /// Compact when the number of accumulated steps (turns) exceeds `turns`.
    TurnCount(usize),
    /// Compact when the cumulative reported token usage exceeds `tokens`.
    /// Agents that do not report tokens never trigger this variant (use
    /// `TurnCount` or pair with `Any`).
    TokenCount(usize),
    /// Compact when either sub-trigger fires.
    Any(Box<CompactionTrigger>, Box<CompactionTrigger>),
    /// Compact when both sub-triggers fire.
    All(Box<CompactionTrigger>, Box<CompactionTrigger>),
}

impl CompactionTrigger {
    /// Whether the trigger fires at `(turns, tokens)`.
    pub fn should_compact(&self, turns: usize, tokens: usize) -> bool {
        match self {
            CompactionTrigger::TurnCount(limit) => turns > *limit,
            CompactionTrigger::TokenCount(limit) => tokens > *limit,
            CompactionTrigger::Any(a, b) => {
                a.should_compact(turns, tokens) || b.should_compact(turns, tokens)
            }
            CompactionTrigger::All(a, b) => {
                a.should_compact(turns, tokens) && b.should_compact(turns, tokens)
            }
        }
    }
}

/// How to compact. All strategies drop the oldest turns and keep at least
/// [`CompactionConfig::min_recent_turns`] recent ones.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CompactionStrategy {
    /// Keep only the most recent `keep_recent_turns` steps (sliding window).
    SlidingWindow {
        /// Number of recent steps to keep.
        keep_recent_turns: usize,
    },
    /// Drop the oldest steps until the estimated token footprint is within
    /// `max_tokens`, but never below `keep_recent_turns` steps.
    TokenBudget {
        /// Estimated token ceiling for the retained history.
        max_tokens: usize,
        /// Number of recent steps always kept, even over budget.
        keep_recent_turns: usize,
    },
    /// ClearToolUses (C2, v0.22.1 §S8): trim context without dropping turns.
    ///
    /// Replaces the tool observation text of steps older than the most recent
    /// `keep_recent_turns` with a short `placeholder`. Every step is retained —
    /// history length is preserved and no action is orphaned from its
    /// (placeholder) observation, so the model still sees the full k-ary
    /// sequence of tool calls, just without the bulky results. Idempotent:
    /// already-clear observations are left untouched on later compactions.
    ClearToolUses {
        /// Number of most recent steps whose observations stay intact.
        keep_recent_turns: usize,
        /// Placeholder text inserted in place of cleared observations.
        placeholder: String,
    },
}

/// Token estimate for one step: ~4 bytes per token over the serialized step
/// (tool name + input + observation). Providers that do not report per-step
/// usage leave no better signal — this mirrors the byte-length fallback used
/// elsewhere (`TokenTrackingLLM`, `get_num_tokens`).
pub fn estimate_step_tokens(step: &AgentStep) -> usize {
    let input_len = match &step.action.tool_input {
        crate::types::ToolInput::String { value } => value.len(),
        crate::types::ToolInput::Object { value } => value.to_string().len(),
    };
    (step.action.tool.len() + input_len + step.observation.len()) / 4
}

/// Compaction configuration: trigger + strategy + safety floor.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// When to compact.
    pub trigger: CompactionTrigger,
    /// How to compact.
    pub strategy: CompactionStrategy,
    /// Safety floor: never drop below this many recent steps, even if the
    /// trigger and strategy would remove more. Prevents pathological configs
    /// from wiping the whole history.
    pub min_recent_turns: usize,
}

impl CompactionConfig {
    /// Creates a config from trigger + strategy (default floor of 2).
    pub fn new(trigger: CompactionTrigger, strategy: CompactionStrategy) -> Self {
        Self {
            trigger,
            strategy,
            min_recent_turns: 2,
        }
    }

    /// Sets the safety floor (never keep fewer than this many steps).
    pub fn with_min_recent_turns(mut self, min_recent_turns: usize) -> Self {
        self.min_recent_turns = min_recent_turns;
        self
    }

    /// Returns the retained steps and how many were dropped.
    ///
    /// Pure: `(kept, dropped)` with `kept.len() + dropped == steps.len()` and
    /// `kept` a suffix of `steps` (order and pairing preserved — no orphaned
    /// actions). A no-op returns `(the same steps, 0)` when the trigger does
    /// not fire or the floor is already reached.
    pub fn compact(&self, steps: &[AgentStep], tokens: usize) -> (Vec<AgentStep>, usize) {
        if !self.trigger.should_compact(steps.len(), tokens) {
            return (steps.to_vec(), 0);
        }
        let floor = self.min_recent_turns.min(steps.len());

        // ClearToolUses doesn't drop — it rewrites observations in place and returns the full
        // history, so handle it before the drop-oriented strategies.
        if let CompactionStrategy::ClearToolUses {
            keep_recent_turns,
            placeholder,
        } = &self.strategy
        {
            return self.clear_tool_uses(steps, *keep_recent_turns, placeholder);
        }

        let keep = match &self.strategy {
            CompactionStrategy::SlidingWindow { keep_recent_turns } => {
                (*keep_recent_turns).max(floor)
            }
            CompactionStrategy::TokenBudget {
                max_tokens,
                keep_recent_turns,
            } => {
                // Walk from the newest step backwards, accumulating the token
                // estimate; stop at the budget (or at the keep/floor limits).
                let mut kept_tokens = 0usize;
                let mut kept = 0usize;
                for step in steps.iter().rev() {
                    if kept >= steps.len()
                        || kept >= (*keep_recent_turns).max(floor)
                            && kept_tokens + estimate_step_tokens(step) > *max_tokens
                    {
                        break;
                    }
                    kept_tokens += estimate_step_tokens(step);
                    kept += 1;
                }
                kept.max((*keep_recent_turns).max(floor)).min(steps.len())
            }
            CompactionStrategy::ClearToolUses { .. } => unreachable!("handled above"),
        };
        let keep = keep.min(steps.len());
        let dropped = steps.len() - keep;
        if dropped == 0 {
            return (steps.to_vec(), 0);
        }
        (steps[steps.len() - keep..].to_vec(), dropped)
    }

    /// C2: replace tool observations older than the most recent `keep_recent_turns` with a
    /// placeholder. Returns the full (unchanged-length) history and the count of observations
    /// actually rewritten (already-clear ones are skipped). No step is orphaned: every clear
    /// keeps its action + a (placeholder) observation.
    fn clear_tool_uses(
        &self,
        steps: &[AgentStep],
        keep_recent_turns: usize,
        placeholder: &str,
    ) -> (Vec<AgentStep>, usize) {
        if steps.is_empty() || keep_recent_turns >= steps.len() {
            return (steps.to_vec(), 0);
        }
        let mut out = steps.to_vec();
        let mut cleared = 0usize;
        let clear_count = steps.len() - keep_recent_turns;
        for step in out.iter_mut().take(clear_count) {
            if step.observation != placeholder {
                step.observation = placeholder.to_string();
                cleared += 1;
            }
        }
        (out, cleared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentAction, ToolInput};

    fn step(name: &str, observation_len: usize) -> AgentStep {
        AgentStep::new(
            AgentAction {
                tool: name.to_string(),
                tool_input: ToolInput::String {
                    value: "input".to_string(),
                },
                log: String::new(),
            },
            "x".repeat(observation_len),
        )
    }

    #[test]
    fn trigger_turn_count() {
        let t = CompactionTrigger::TurnCount(3);
        assert!(!t.should_compact(3, 0));
        assert!(t.should_compact(4, 0));
    }

    #[test]
    fn trigger_token_count() {
        let t = CompactionTrigger::TokenCount(100);
        assert!(!t.should_compact(0, 100));
        assert!(t.should_compact(0, 101));
    }

    /// Agents that do not report tokens (tokens=0) never fire a TokenCount trigger.
    #[test]
    fn trigger_token_count_never_fires_without_tokens() {
        let t = CompactionTrigger::TokenCount(0);
        assert!(!t.should_compact(10, 0));
    }

    #[test]
    fn trigger_any_and_all() {
        let turn = CompactionTrigger::TurnCount(2);
        let token = CompactionTrigger::TokenCount(10);
        let any = CompactionTrigger::Any(Box::new(turn.clone()), Box::new(token.clone()));
        let all = CompactionTrigger::All(Box::new(turn), Box::new(token));
        // turns fire, tokens do not.
        assert!(any.should_compact(5, 0));
        assert!(!all.should_compact(5, 0));
        assert!(all.should_compact(5, 100));
    }

    /// SlidingWindow keeps exactly the newest N steps, order preserved.
    #[test]
    fn sliding_window_keeps_recent_suffix() {
        let config = CompactionConfig::new(
            CompactionTrigger::TurnCount(2),
            CompactionStrategy::SlidingWindow {
                keep_recent_turns: 2,
            },
        )
        .with_min_recent_turns(1);
        let steps: Vec<AgentStep> = (0..5).map(|i| step(&format!("t{i}"), 10)).collect();
        let (kept, dropped) = config.compact(&steps, 0);
        assert_eq!(dropped, 3);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].action.tool, "t3", "suffix preserved");
        assert_eq!(kept[1].action.tool, "t4");
    }

    /// Under the trigger threshold: no-op (same steps, zero dropped).
    #[test]
    fn no_compaction_below_trigger() {
        let config = CompactionConfig::new(
            CompactionTrigger::TurnCount(10),
            CompactionStrategy::SlidingWindow {
                keep_recent_turns: 2,
            },
        );
        let steps: Vec<AgentStep> = (0..5).map(|i| step(&format!("t{i}"), 10)).collect();
        let (kept, dropped) = config.compact(&steps, 0);
        assert_eq!(dropped, 0);
        assert_eq!(kept.len(), 5);
    }

    /// The safety floor wins over an aggressive strategy.
    #[test]
    fn min_recent_turns_floor() {
        let config = CompactionConfig::new(
            CompactionTrigger::TurnCount(1),
            CompactionStrategy::SlidingWindow {
                keep_recent_turns: 0,
            },
        )
        .with_min_recent_turns(2);
        let steps: Vec<AgentStep> = (0..6).map(|i| step(&format!("t{i}"), 10)).collect();
        let (kept, dropped) = config.compact(&steps, 0);
        assert_eq!(kept.len(), 2);
        assert_eq!(dropped, 4);
        assert_eq!(kept[1].action.tool, "t5");
    }

    /// TokenBudget drops the oldest steps until the estimate fits, keeping the
    /// mandated minimum.
    #[test]
    fn token_budget_drops_oldest_until_fit() {
        // Each step: tool "t" + "input" + 400-byte observation → ~404/4 ≈ 101 tokens.
        let config = CompactionConfig::new(
            CompactionTrigger::TokenCount(150),
            CompactionStrategy::TokenBudget {
                max_tokens: 150,
                keep_recent_turns: 1,
            },
        )
        .with_min_recent_turns(1);
        let steps: Vec<AgentStep> = (0..5).map(|i| step(&format!("t{i}"), 400)).collect();
        let total: usize = steps.iter().map(estimate_step_tokens).sum();
        assert!(total > 150, "precondition: history over budget");

        let (kept, dropped) = config.compact(&steps, total);
        assert!(dropped >= 1, "over budget → drop");
        let kept_tokens: usize = kept.iter().map(estimate_step_tokens).sum();
        // Either within budget, or protected by the keep floor.
        assert!(
            kept_tokens <= 150 || kept.len() <= 1,
            "kept={} dropped={} tokens={}",
            kept.len(),
            dropped,
            kept_tokens
        );
    }

    /// TokenBudget keeps at least `keep_recent_turns` even when each step alone
    /// busts the budget.
    #[test]
    fn token_budget_respects_keep_floor() {
        let config = CompactionConfig::new(
            CompactionTrigger::TokenCount(10),
            CompactionStrategy::TokenBudget {
                max_tokens: 10,
                keep_recent_turns: 2,
            },
        )
        .with_min_recent_turns(1);
        let steps: Vec<AgentStep> = (0..4).map(|i| step(&format!("t{i}"), 400)).collect();
        let (kept, _) = config.compact(&steps, 500);
        assert_eq!(kept.len(), 2, "keep floor wins over the budget");
        assert_eq!(kept[0].action.tool, "t2");
    }

    /// Compaction never splits an action/observation pair — by construction
    /// (`AgentStep` bundles both), but assert the invariant anyway.
    #[test]
    fn compaction_never_orphans_tool_results() {
        let config = CompactionConfig::new(
            CompactionTrigger::TurnCount(0),
            CompactionStrategy::SlidingWindow {
                keep_recent_turns: 3,
            },
        );
        let steps: Vec<AgentStep> = (0..8).map(|i| step(&format!("t{i}"), 50)).collect();
        let (kept, dropped) = config.compact(&steps, 0);
        assert_eq!(kept.len() + dropped, steps.len());
        // Every kept step has a non-empty observation paired with its action.
        for s in &kept {
            assert!(!s.observation.is_empty());
        }
    }

    /// Estimate is proportional to content size.
    #[test]
    fn estimate_scales_with_content() {
        assert!(estimate_step_tokens(&step("tool", 400)) > estimate_step_tokens(&step("tool", 40)));
    }

    // ---------------------------------------------------------------------
    // C2: ClearToolUses
    // ---------------------------------------------------------------------

    #[test]
    fn clear_tool_uses_replaces_old_observations_keeps_recent() {
        let config = CompactionConfig::new(
            CompactionTrigger::TurnCount(2),
            CompactionStrategy::ClearToolUses {
                keep_recent_turns: 2,
                placeholder: "[cleared]".into(),
            },
        );
        let steps: Vec<AgentStep> = (0..5)
            .map(|i| step(&format!("t{i}"), i as usize * 100))
            .collect();

        let (kept, cleared) = config.compact(&steps, 0);
        // history length unchanged — nothing is dropped
        assert_eq!(kept.len(), 5, "ClearToolUses must not drop steps");
        assert_eq!(cleared, 3, "oldest 3 observations cleared");
        // recent two observations intact
        assert_eq!(kept[3].observation, "x".repeat(300));
        assert_eq!(kept[4].observation, "x".repeat(400));
        // older observations replaced by the placeholder
        for s in &kept[..3] {
            assert_eq!(s.observation, "[cleared]");
        }
    }

    #[test]
    fn clear_tool_uses_never_orphans_actions() {
        let config = CompactionConfig::new(
            CompactionTrigger::TurnCount(0),
            CompactionStrategy::ClearToolUses {
                keep_recent_turns: 1,
                placeholder: "[cleared]".into(),
            },
        );
        let steps: Vec<AgentStep> = (0..7).map(|i| step(&format!("t{i}"), 50)).collect();
        let (kept, _) = config.compact(&steps, 0);
        assert_eq!(kept.len(), steps.len());
        for (idx, s) in kept.iter().enumerate() {
            // action preserved with a non-empty observation (intact or placeholder)
            assert!(!s.observation.is_empty(), "step {idx} orphaned");
        }
        // actions themselves untouched (order + names preserved)
        for (idx, s) in kept.iter().enumerate() {
            assert_eq!(s.action.tool, format!("t{idx}"));
        }
    }

    #[test]
    fn clear_tool_uses_is_idempotent() {
        let config = CompactionConfig::new(
            CompactionTrigger::TurnCount(0),
            CompactionStrategy::ClearToolUses {
                keep_recent_turns: 2,
                placeholder: "[cleared]".into(),
            },
        );
        let steps: Vec<AgentStep> = (0..5).map(|i| step(&format!("t{i}"), 50)).collect();
        let (first, c1) = config.compact(&steps, 0);
        assert_eq!(c1, 3);
        // compacting the already-cleared history clears nothing new
        let (second, c2) = config.compact(&first, 0);
        assert_eq!(c2, 0);
        // history is byte-identical after the idempotent second pass
        fn obs(v: &[AgentStep]) -> Vec<&str> {
            v.iter().map(|s| s.observation.as_str()).collect()
        }
        assert_eq!(obs(&second), obs(&first));
    }

    #[test]
    fn clear_tool_uses_keeps_everything_when_under_keep() {
        let config = CompactionConfig::new(
            CompactionTrigger::TurnCount(10),
            CompactionStrategy::ClearToolUses {
                keep_recent_turns: 3,
                placeholder: "[cleared]".into(),
            },
        );
        // trigger (TurnCount 10) doesn't fire → no-op like other strategies
        let steps: Vec<AgentStep> = (0..5).map(|i| step(&format!("t{i}"), 50)).collect();
        let (kept, cleared) = config.compact(&steps, 0);
        assert_eq!(cleared, 0);
        // byte-identical pass-through, nothing rewritten
        let same = kept
            .iter()
            .zip(steps.iter())
            .all(|(a, b)| a.observation == b.observation && a.action.tool == b.action.tool);
        assert!(same);
    }
}
