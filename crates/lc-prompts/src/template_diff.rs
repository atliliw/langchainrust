// crates/lc-prompts/src/template_diff.rs
//! Template diff / blast-radius analysis (0.27.0 "模板 diff / 影响面分析").
//!
//! [`diff_variables`](crate::diff_variables) compares a *template's* variables
//! against the variables **one caller** supplied. That is the construct-time
//! concern. This module answers a different, publication-time question: when an
//! operator edits a template (old → new), **who is affected**?
//!
//! A template change can break existing callers that construct a prompt and
//! supply a specific variable set:
//!
//! - A variable is **added** → every caller that does not supply it now hits a
//!   `MissingVariable` at render. This is the breaking direction.
//! - A variable is **removed** → callers that kept passing it still render
//!   (the extra value is ignored), so this is *not* breaking by itself — unless
//!   the caller enables strict unused-variable validation.
//! - The variable slots are **reordered** → the rendered prompt changes but the
//!   required set does not; callers still work, output shape shifts.

use crate::template_parser::{parse_template, template_variables};
use crate::validation::dedup_in_order;
use std::collections::HashSet;

/// The blast radius of changing one prompt template from an old version to a
/// new version.
///
/// Lists are in first-appearance (template) order — never hash order — so
/// diffs are deterministic.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TemplateDiff {
    /// Variables the new template requires that the old one did not. **Breaking**
    /// for existing callers that do not supply them yet.
    pub added: Vec<String>,
    /// Variables the old template required that the new one dropped. Not breaking
    /// on its own (surplus caller values are ignored) unless the caller runs
    /// strict unused-variable validation.
    pub removed: Vec<String>,
    /// Whether the relative order of the shared variable slots changed.
    pub reordered: bool,
}

impl TemplateDiff {
    /// Returns `true` when the two templates differ in any observable way.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && !self.reordered
    }

    /// Returns `true` when applying the new template would make some existing
    /// caller's `format` fail: the new template requires a variable none of them
    /// had to provide before.
    pub fn is_breaking(&self) -> bool {
        !self.added.is_empty()
    }
}

/// Diffs the variable structure of two prompt-template strings.
///
/// `old` is the version currently in use; `new` is the proposed edit. Parsing
/// follows the same rules as [`parse_template`]: `{{`/`}}` escapes, malformed
/// placeholders stay literal. It does **not** require either template to be
/// strictly well-formed.
///
/// # Example
/// ```
/// use lc_prompts::diff_templates;
/// let diff = diff_templates("Hello, {name}!", "Hello, {name} and {day}!");
/// assert!(diff.is_breaking());          // `{day}` was added
/// assert_eq!(diff.added, ["day"]);
/// assert_eq!(diff.removed, Vec::<String>::new());
/// ```
pub fn diff_templates(old: &str, new: &str) -> TemplateDiff {
    let old_vars = variable_order(old);
    let new_vars = variable_order(new);
    let old_set: HashSet<&str> = old_vars.iter().map(String::as_str).collect();
    let new_set: HashSet<&str> = new_vars.iter().map(String::as_str).collect();

    let added = new_vars
        .iter()
        .filter(|v| !old_set.contains(v.as_str()))
        .cloned()
        .collect();
    let removed = old_vars
        .iter()
        .filter(|v| !new_set.contains(v.as_str()))
        .cloned()
        .collect();

    // Reordered only when the *shared* variables appear in a different relative
    // order; purely additive/removal changes in template order must not count.
    let shared_old: Vec<&str> = old_vars
        .iter()
        .filter(|v| new_set.contains(v.as_str()))
        .map(String::as_str)
        .collect();
    let shared_new: Vec<&str> = new_vars
        .iter()
        .filter(|v| old_set.contains(v.as_str()))
        .map(String::as_str)
        .collect();
    let reordered = shared_old != shared_new;

    TemplateDiff {
        added,
        removed,
        reordered,
    }
}

/// Template variables deduplicated in first-appearance order.
fn variable_order(template: &str) -> Vec<String> {
    let segments = parse_template(template);
    dedup_in_order(&template_variables(&segments))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_templates_are_an_empty_diff() {
        let diff = diff_templates("Hi {name}, today is {day}.", "Hi {name}, today is {day}.");
        assert!(diff.is_empty());
        // pure literal-text tweak with no variable change is also empty
        let moved = diff_templates("Hello, {name}!", " Hello, {name}!  ");
        assert!(moved.is_empty(), "text-only change has no variable impact");
    }

    #[test]
    fn adding_a_variable_is_breaking() {
        let diff = diff_templates("Hello, {name}!", "Hello, {name} and {day}!");
        assert!(diff.is_breaking());
        assert_eq!(diff.added, vec!["day".to_string()]);
        assert!(diff.removed.is_empty());
        assert!(!diff.reordered);
    }

    #[test]
    fn removing_a_variable_is_not_breaking_by_itself() {
        let diff = diff_templates("Hello, {name} and {day}!", "Hello, {name}!");
        assert!(!diff.is_breaking());
        assert_eq!(diff.removed, vec!["day".to_string()]);
        assert!(diff.added.is_empty());
    }

    #[test]
    fn reordering_variable_slots_is_detected_without_a_set_change() {
        let diff = diff_templates("{a} then {b}", "{b} then {a}");
        assert!(diff.reordered);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert!(!diff.is_empty());
        // not breaking: the required set is unchanged
        assert!(!diff.is_breaking());
    }

    #[test]
    fn duplicates_do_not_register_and_escapes_are_respected() {
        // `{name}` appears twice in old and new → no diff
        let same = diff_templates("{name} and {name}", "{name} and {name}");
        assert!(same.is_empty());
        // `{{literal}}` is not a variable on either side
        let esc = diff_templates("{{x}} {name}", "{{x}} {name}");
        assert!(esc.is_empty());
    }

    #[test]
    fn order_is_first_appearance_not_hash_order() {
        let diff = diff_templates("{c} {a} {b}", "{c} {x} {a}");
        assert_eq!(diff.added, vec!["x".to_string()]);
        assert_eq!(diff.removed, vec!["b".to_string()]);
        assert!(!diff.reordered, "shared {{c}},{{a}} keep their relative order");
    }
}