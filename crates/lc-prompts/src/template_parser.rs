// crates/lc-prompts/src/template_parser.rs
//! Shared prompt template parser.
//!
//! Both [`crate::PromptTemplate`] and [`crate::ChatPromptTemplate`] format
//! `{variable}` placeholders. This module centralizes the parsing so the two
//! never drift apart:
//!
//! - Variables use `{name}` with a name of `[A-Za-z_][A-Za-z0-9_]*` or CJK
//!   characters (`is_alphabetic` covers Han, Latin, etc.).
//! - `{{` and `}}` escape literal braces.
//!
//! Templates are parsed once into [`TemplateSegment`]s; `format` then walks the
//! segments instead of re-scanning with a regex on every call.

use crate::error::PromptsError;
use std::collections::HashMap;

/// A parsed segment of a prompt template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateSegment {
    /// Literal text.
    Text(String),
    /// A `{name}` variable placeholder.
    Variable(String),
}

/// Parses a template into segments, handling `{{`/`}}` escaping and
/// validating variable names.
///
/// Lone `{`/`}` and malformed `{...}` are treated as literal text. Use
/// [`validate_template`] when templates should fail fast on a malformed
/// placeholder instead of silently rendering it literally.
pub fn parse_template(template: &str) -> Vec<TemplateSegment> {
    scan_template(template).segments()
}

/// Validates that a template's placeholders are all well-formed, returning an
/// error on the first malformed one (`{` without a matching `}`, or a name
/// inside `{...}` that is not a valid variable name).
///
/// This is the strict counterpart to [`parse_template`], which deliberately
/// falls back to literal text. Call it at construction time to catch template
/// typos early rather than emitting stray `{` into prompts.
pub fn validate_template(template: &str) -> Result<(), PromptsError> {
    scan_template(template).require_well_formed()
}

/// Result of a single scan of a template: the lenient segment list plus any
/// malformed places encountered. One scanner serves both [`parse_template`]
/// (which ignores the issues) and [`validate_template`] (which surfaces them),
/// so the lenient and strict behaviors can never drift apart.
struct TemplateScan {
    segments: Vec<TemplateSegment>,
    /// Human-readable descriptions of malformed placeholders, in order.
    issues: Vec<String>,
}

impl TemplateScan {
    /// The lenient segment list.
    fn segments(self) -> Vec<TemplateSegment> {
        self.segments
    }

    /// Returns an error describing the first malformed placeholder, if any.
    fn require_well_formed(self) -> Result<(), PromptsError> {
        match self.issues.into_iter().next() {
            Some(issue) => Err(PromptsError::MalformedTemplate(issue)),
            None => Ok(()),
        }
    }
}

fn scan_template(template: &str) -> TemplateScan {
    let chars: Vec<char> = template.chars().collect();
    let n = chars.len();
    let mut segments: Vec<TemplateSegment> = Vec::new();
    let mut issues: Vec<String> = Vec::new();
    let mut text = String::new();
    let mut i = 0;

    while i < n {
        let c = chars[i];

        if c == '{' {
            // Escaped `{{` → literal `{`
            if i + 1 < n && chars[i + 1] == '{' {
                text.push('{');
                i += 2;
                continue;
            }

            // Find the closing `}` for a potential `{name}`
            let mut j = i + 1;
            while j < n && chars[j] != '}' {
                j += 1;
            }
            if j < n {
                let name: String = chars[i + 1..j].iter().collect();
                if is_valid_var_name(&name) {
                    if !text.is_empty() {
                        segments.push(TemplateSegment::Text(std::mem::take(&mut text)));
                    }
                    segments.push(TemplateSegment::Variable(name));
                    i = j + 1;
                    continue;
                }
                // `{...}` present and closed, but the name is invalid
                issues.push(format!("invalid variable name inside `{{{name}}}`"));
            } else {
                // No closing `}` before end of input
                issues.push(format!("unclosed `{c}` placeholder"));
            }

            // Not a variable — literal `{`
            text.push('{');
            i += 1;
            continue;
        }

        if c == '}' {
            // Escaped `}}` → literal `}`
            if i + 1 < n && chars[i + 1] == '}' {
                text.push('}');
                i += 2;
                continue;
            }
            // Lone `}` is literal
            issues.push("stray `}` with no matching `{`".to_string());
            text.push('}');
            i += 1;
            continue;
        }

        text.push(c);
        i += 1;
    }

    if !text.is_empty() {
        segments.push(TemplateSegment::Text(text));
    }

    TemplateScan { segments, issues }
}

/// Provides variable lookup for the formatter, so both a borrowed `&str`-keyed
/// map and an owned `String`-keyed map can be substituted through one code path
/// without building an intermediate `HashMap<&str, &str>` on every invocation.
pub trait TemplateVars {
    /// Looks up a variable's value by name.
    fn get(&self, name: &str) -> Option<&str>;
}

impl TemplateVars for HashMap<String, String> {
    fn get(&self, name: &str) -> Option<&str> {
        self.get(name).map(String::as_str)
    }
}

impl TemplateVars for HashMap<&str, &str> {
    fn get(&self, name: &str) -> Option<&str> {
        self.get(name).copied()
    }
}

/// Formats pre-parsed segments, substituting variables from `variables`.
///
/// Returns an error listing the first missing variable.
pub fn format_template(
    template: &str,
    segments: &[TemplateSegment],
    variables: &HashMap<&str, &str>,
) -> Result<String, PromptsError> {
    format_template_vars(template, segments, variables)
}

/// Formats from any [`TemplateVars`] source (owned or borrowed), avoiding the
/// intermediate map allocation that the ref-based [`format_template`] implies.
pub fn format_template_vars<V: TemplateVars>(
    template: &str,
    segments: &[TemplateSegment],
    variables: &V,
) -> Result<String, PromptsError> {
    let mut result = String::with_capacity(template.len());
    for seg in segments {
        match seg {
            TemplateSegment::Text(t) => result.push_str(t),
            TemplateSegment::Variable(name) => match variables.get(name.as_str()) {
                Some(v) => result.push_str(v),
                None => return Err(PromptsError::MissingVariable(name.clone())),
            },
        }
    }
    Ok(result)
}

/// Collects variable names in order of appearance (duplicates preserved).
pub fn template_variables(segments: &[TemplateSegment]) -> Vec<String> {
    segments
        .iter()
        .filter_map(|seg| match seg {
            TemplateSegment::Variable(name) => Some(name.clone()),
            TemplateSegment::Text(_) => None,
        })
        .collect()
}

/// A variable name starts with a letter or `_`, followed by letters, digits,
/// `_`. `is_alphabetic`/`is_alphanumeric` also accept CJK characters.
fn is_valid_var_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic() {
        let segments = parse_template("你好，{name}！");
        assert_eq!(
            segments,
            vec![
                TemplateSegment::Text("你好，".to_string()),
                TemplateSegment::Variable("name".to_string()),
                TemplateSegment::Text("！".to_string()),
            ]
        );
    }

    #[test]
    fn test_parse_escaped_braces() {
        // {{ and }} become literal braces
        let segments = parse_template("{{literal}} {var} }}end{{");
        let vars = template_variables(&segments);
        assert_eq!(vars, vec!["var".to_string()]);
    }

    #[test]
    fn test_parse_cjk_and_underscore_variables() {
        let segments = parse_template("{中文名} and {_private} and {a1}");
        let vars = template_variables(&segments);
        assert_eq!(
            vars,
            vec![
                "中文名".to_string(),
                "_private".to_string(),
                "a1".to_string()
            ]
        );
    }

    #[test]
    fn test_parse_digit_start_is_literal() {
        // {123} is not a valid variable name → stays literal
        let segments = parse_template("id: {123} {name}");
        let vars = template_variables(&segments);
        assert_eq!(vars, vec!["name".to_string()]);
    }

    #[test]
    fn test_parse_unclosed_brace_is_literal() {
        let segments = parse_template("{unclosed and {name}");
        let vars = template_variables(&segments);
        assert_eq!(vars, vec!["name".to_string()]);
    }

    #[test]
    fn test_format_substitutes_and_escapes() {
        let segments = parse_template("{{a}} {x} }} b {y}");
        let mut vars = HashMap::new();
        vars.insert("x", "X");
        vars.insert("y", "Y");
        let out = format_template("{{a}} {x} }} b {y}", &segments, &vars).unwrap();
        assert_eq!(out, "{a} X } b Y");
    }

    #[test]
    fn test_format_missing_variable_errors() {
        let segments = parse_template("{x} {missing}");
        let mut vars = HashMap::new();
        vars.insert("x", "X");
        let err = format_template("{x} {missing}", &segments, &vars).unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn test_validate_accepts_well_formed() {
        // valid names, escaped braces, CJK names → no malformed placeholders
        assert!(validate_template("a {x} b }}}} c {中文}").is_ok());
    }

    #[test]
    fn test_validate_rejects_unclosed() {
        let err = validate_template("{name").unwrap_err();
        assert!(err.to_string().contains("unclosed"));
    }

    #[test]
    fn test_validate_accepts_escaped_only() {
        // `}}` is an escaped literal `}`, not a stray close → no issue
        assert!(validate_template("x }}").is_ok());
    }

    #[test]
    fn test_validate_rejects_invalid_name() {
        // digit-start name is valid-literal for lenient parse but strict flags it
        let err = validate_template("{123}").unwrap_err();
        assert!(err.to_string().contains("invalid variable name"));
    }

    #[test]
    fn test_validate_rejects_stray_close() {
        let err = validate_template("done }").unwrap_err();
        assert!(err.to_string().contains("stray"));
    }

    #[test]
    fn test_lenient_parse_still_literal() {
        // parse_template keeps the backwards-compatible lenient behavior even
        // though validate_template now flags the same input
        let segs = parse_template("{unclosed and {name}");
        assert_eq!(template_variables(&segs), vec!["name".to_string()]);
    }

    #[test]
    fn test_format_vars_accepts_owned_map() {
        // `invoke` (Runnable) path formats straight from an owned String map,
        // reusing a single code path without a per-call ref-map allocation
        let segments = parse_template("{x} {y}");
        let mut vars: HashMap<String, String> = HashMap::new();
        vars.insert("x".to_string(), "X".to_string());
        vars.insert("y".to_string(), "Y".to_string());
        let out = format_template_vars("{x} {y}", &segments, &vars).unwrap();
        assert_eq!(out, "X Y");
    }
}
