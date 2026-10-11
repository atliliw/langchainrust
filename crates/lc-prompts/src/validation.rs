// crates/lc-prompts/src/validation.rs
//! Variable validation and diffing (0.27.0).
//!
//! A template's variables and a caller's variables are two sets that have to
//! agree, and both directions of disagreement are silent failures:
//!
//! - **missing** — a variable the template uses but the caller did not supply.
//!   `format` does catch this one, because it cannot substitute what it does not
//!   have.
//! - **unused** — a variable the caller supplied that the template never uses.
//!   Nothing catches this: a typo in the caller's map is ignored, and the prompt
//!   reaches the model with the placeholder still in it.
//!
//! [`VariableDiff`] reports both directions at once and [`diff_variables`]
//! computes it, so a caller can decide whether extra variables are tolerable
//! (the lenient default) or a mistake (strict).
//!
//! The construct-time half of the same idea — a template's *declared* variables
//! versus the variables it actually uses — is what
//! [`FewShotPromptTemplate::validate`](crate::FewShotPromptTemplate::validate)
//! checks, since that is the only template type that declares them separately.
//!
//! Every list here is in first-appearance order (template order for the required
//! side, caller order for the supplied side) and never in hash order, so error
//! messages are deterministic.

use crate::error::PromptsError;
use std::collections::HashSet;

/// The difference between the variables a template needs and the variables a
/// caller supplied.
///
/// See the [module docs](self) for why both directions matter.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VariableDiff {
    /// Needed by the template but not supplied; `format` would fail on these.
    /// In template order.
    pub missing: Vec<String>,
    /// Supplied but appearing nowhere in the template; `format` ignores these
    /// silently. In caller order.
    pub unused: Vec<String>,
}

impl VariableDiff {
    /// Returns `true` when the two sets match exactly — nothing missing and
    /// nothing unused.
    pub fn is_empty(&self) -> bool {
        self.missing.is_empty() && self.unused.is_empty()
    }

    /// Returns `true` when the template has everything it needs. Extra supplied
    /// variables are tolerated by this check but still reported in
    /// [`unused`](Self::unused).
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }

    /// Turns the diff into an error, accepting only an exact match.
    ///
    /// A missing variable is reported as [`PromptsError::MissingVariable`] with
    /// the first one in template order — deliberately the same error, and the
    /// same variable, a lenient `format` would have produced, so strict mode is
    /// a drop-in replacement rather than a new failure vocabulary. Unused
    /// variables are reported as [`PromptsError::UnusedVariables`].
    ///
    /// `template` identifies the offending template in the error message.
    ///
    /// # Errors
    /// Returns the first missing variable, or the unused variables if nothing is
    /// missing.
    pub fn require_exact(&self, template: &str) -> Result<(), PromptsError> {
        if let Some(first) = self.missing.first() {
            return Err(PromptsError::MissingVariable(first.clone()));
        }
        if !self.unused.is_empty() {
            return Err(PromptsError::UnusedVariables {
                template: template.to_string(),
                variables: self.unused.clone(),
            });
        }
        Ok(())
    }
}

/// Computes the [`VariableDiff`] between the variables a template needs and the
/// variables a caller supplied.
///
/// Both sides are deduplicated before comparing, so a template that mentions
/// `{name}` twice and a caller that lists `name` once compare as equal.
///
/// # Example
/// ```ignore
/// let diff = diff_variables(&["name".to_string(), "day".to_string()], ["name", "typo"]);
/// assert_eq!(diff.missing, ["day"]);
/// assert_eq!(diff.unused, ["typo"]);
/// ```
pub fn diff_variables<'a, I>(required: &[String], supplied: I) -> VariableDiff
where
    I: IntoIterator<Item = &'a str>,
{
    let required = dedup_in_order(required);
    let supplied = dedup_owned(supplied.into_iter().map(str::to_string));

    VariableDiff {
        missing: required
            .iter()
            .filter(|var| !supplied.contains(var))
            .cloned()
            .collect(),
        unused: supplied
            .iter()
            .filter(|var| !required.contains(var))
            .cloned()
            .collect(),
    }
}

/// Deduplicates `values`, keeping the first occurrence of each and preserving
/// appearance order.
pub fn dedup_in_order(values: &[String]) -> Vec<String> {
    dedup_owned(values.iter().cloned())
}

/// Deduplicating collector shared by [`dedup_in_order`] and [`diff_variables`].
fn dedup_owned<I: IntoIterator<Item = String>>(values: I) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            out.push(value);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn exact_match_is_an_empty_diff() {
        let required = owned(&["name", "day"]);
        let diff = diff_variables(&required, ["day", "name"]);

        assert!(diff.is_empty());
        assert!(diff.is_complete());
        assert!(diff.require_exact("t").is_ok());
    }

    #[test]
    fn both_directions_are_reported_at_once() {
        let required = owned(&["name", "day"]);
        let diff = diff_variables(&required, ["name", "typo"]);

        assert_eq!(diff.missing, owned(&["day"]));
        assert_eq!(diff.unused, owned(&["typo"]));
        assert!(!diff.is_empty());
        assert!(!diff.is_complete());
    }

    #[test]
    fn missing_is_reported_before_unused() {
        let required = owned(&["day"]);
        let err = diff_variables(&required, ["typo"])
            .require_exact("t")
            .unwrap_err();

        // same variant and same variable a lenient format would have produced
        assert!(matches!(err, PromptsError::MissingVariable(ref v) if v == "day"));
    }

    #[test]
    fn unused_alone_is_its_own_error() {
        let required = owned(&["day"]);
        let err = diff_variables(&required, ["day", "typo"])
            .require_exact("tpl")
            .unwrap_err();

        match err {
            PromptsError::UnusedVariables {
                template,
                variables,
            } => {
                assert_eq!(template, "tpl");
                assert_eq!(variables, owned(&["typo"]));
            }
            other => panic!("expected UnusedVariables, got {other:?}"),
        }
    }

    #[test]
    fn duplicates_on_either_side_do_not_register_as_a_diff() {
        let required = owned(&["name", "name"]);
        let diff = diff_variables(&required, ["name", "name"]);

        assert!(diff.is_empty());
    }

    #[test]
    fn order_follows_appearance_not_hash_order() {
        // regression: a HashSet-backed implementation would shuffle these and
        // make both the message and any test asserting on it unstable
        let required = owned(&["c", "a", "b"]);
        let diff = diff_variables(&required, ["z", "y", "x"]);

        assert_eq!(diff.missing, owned(&["c", "a", "b"]));
        assert_eq!(diff.unused, owned(&["z", "y", "x"]));
    }

    #[test]
    fn dedup_keeps_the_first_occurrence() {
        assert_eq!(
            dedup_in_order(&owned(&["b", "a", "b", "a"])),
            owned(&["b", "a"])
        );
        assert!(dedup_in_order(&[]).is_empty());
    }
}
