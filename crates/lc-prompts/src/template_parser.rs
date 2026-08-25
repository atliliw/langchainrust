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
/// Lone `{`/`}` and malformed `{...}` are treated as literal text.
pub fn parse_template(template: &str) -> Vec<TemplateSegment> {
    let chars: Vec<char> = template.chars().collect();
    let n = chars.len();
    let mut segments: Vec<TemplateSegment> = Vec::new();
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

    segments
}

/// Formats pre-parsed segments, substituting variables from `variables`.
///
/// Returns an error listing the first missing variable.
pub fn format_template(
    template: &str,
    segments: &[TemplateSegment],
    variables: &HashMap<&str, &str>,
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
}
