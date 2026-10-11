// crates/lc-prompts/src/prompt_template.rs
//! Simple string template

use crate::error::PromptsError;
use crate::template_parser::{
    format_template, format_template_vars, parse_template, template_variables, validate_template,
    TemplateSegment,
};
use crate::validation::{diff_variables, VariableDiff};
use async_trait::async_trait;
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Prompt template
///
/// A template using `{variable}` format with variable substitution; `{{` and `}}` escape
/// literal braces.
///
/// The template is **parsed once** at construction and the parsed segment list is
/// cached on the instance, so every subsequent [`format`](Self::format) /
/// [`Runnable::invoke`] render is an O(template-length) substitution with no re-parse.
/// A per-instance render counter lets you observe how often a template actually ran
/// (see [`render_count`](Self::render_count)).
#[derive(Debug)]
pub struct PromptTemplate {
    template: String,
    segments: Vec<TemplateSegment>,
    /// 0.27.0 L267: number of successful renders. Atomic so the counter bumps
    /// through `&self` methods *and* the instance stays `Send + Sync` (shared
    /// via the template cache and across an async runtime).
    renders: AtomicU64,
}

impl PromptTemplate {
    /// Creates a new prompt template
    ///
    /// # Arguments
    /// * `template` - Template string, using `{variable}` markers
    ///
    /// # Example
    /// ```ignore
    /// let template = PromptTemplate::new("你好，{name}！今天是{day}。");
    /// let mut vars = HashMap::new();
    /// vars.insert("name", "小明");
    /// vars.insert("day", "星期一");
    /// let result = template.format(&vars).unwrap();
    /// ```
    pub fn new(template: impl Into<String>) -> Self {
        let template = template.into();
        let segments = parse_template(&template);
        Self {
            template,
            segments,
            renders: AtomicU64::new(0),
        }
    }

    /// Formats the template, replacing all variables
    ///
    /// # Arguments
    /// * `variables` - Variable mapping
    ///
    /// # Returns
    /// The replaced string, or a missing-variable error
    ///
    /// # Errors
    /// Returns an error if the template has a variable with no matching value in `variables`
    pub fn format(&self, variables: &HashMap<&str, &str>) -> Result<String, PromptsError> {
        let out = format_template(&self.template, &self.segments, variables);
        if out.is_ok() {
            self.renders.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    /// Number of times this template has rendered successfully (hist of
    /// `format`, `format_strict` and `Runnable::invoke`). A gauge you can poll
    /// to see how often a hot template actually ran, independent of construction.
    pub fn render_count(&self) -> u64 {
        self.renders.load(Ordering::Relaxed)
    }

    /// Resets the render counter to zero.
    pub fn reset_render_count(&self) {
        self.renders.store(0, Ordering::Relaxed);
    }

    /// Returns all variable names the template needs
    ///
    /// # Returns
    /// The variable-name list
    pub fn variables(&self) -> Vec<String> {
        template_variables(&self.segments)
    }

    /// Compares the template's variables against the ones a caller would supply.
    ///
    /// Use this at construction time to check a caller's variable map before
    /// formatting, and to see variables the template will ignore (see
    /// [`VariableDiff`] for why that direction matters).
    pub fn variable_diff<'a, I>(&self, supplied: I) -> VariableDiff
    where
        I: IntoIterator<Item = &'a str>,
    {
        diff_variables(&self.variables(), supplied)
    }

    /// Formats the template, refusing malformed placeholders and unused
    /// variables as well as missing ones.
    ///
    /// [`format`](Self::format) is lenient in two ways that hide mistakes: a
    /// stray `{` is emitted as literal text, and a supplied variable the
    /// template never uses is ignored. This variant fails on both — use it where
    /// a template and its callers are both under your control and a typo should
    /// fail fast. A missing variable produces exactly the error `format` would.
    ///
    /// # Errors
    /// Returns [`PromptsError::MalformedTemplate`] for a malformed placeholder,
    /// [`PromptsError::MissingVariable`] for an unsupplied template variable, or
    /// [`PromptsError::UnusedVariables`] when the caller supplied extras.
    pub fn format_strict(&self, variables: &HashMap<&str, &str>) -> Result<String, PromptsError> {
        self.validate()?;
        self.variable_diff(variables.keys().copied())
            .require_exact(&self.template)?;
        self.format(variables)
    }

    /// Returns the raw template string
    pub fn template(&self) -> &str {
        &self.template
    }

    /// Strictly validates that every placeholder is well-formed, returning an
    /// error on the first malformed one (`{` with no matching `}`, or an invalid
    /// variable name inside braces).
    ///
    /// `new` parses leniently — stray `{`/`}` fall back to literal text, which
    /// is desirable for templates that legitimately contain braces (JSON, code).
    /// Call this at construction time when a template typo should fail fast
    /// instead of emitting a stray `{` into a prompt.
    ///
    /// # Errors
    /// Returns an error describing the first malformed placeholder, if any
    pub fn validate(&self) -> Result<(), PromptsError> {
        validate_template(&self.template)
    }
}

impl std::fmt::Display for PromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.template)
    }
}

// Runnable form: lets the prompt join an LCEL chain, so `prompt.pipe(...)` works.
// Receives an owned variable map (HashMap<String, String>) and formats straight
// from it via `format_template_vars`, avoiding a per-call ref-map allocation.
// Same as `ChatPromptTemplate`; errors go through `LcelError::Chain`.
#[async_trait]
impl Runnable<HashMap<String, String>, String> for PromptTemplate {
    type Error = LcelError;

    async fn invoke(
        &self,
        input: HashMap<String, String>,
        _config: Option<RunnableConfig>,
    ) -> Result<String, LcelError> {
        let out = format_template_vars(&self.template, &self.segments, &input)
            .map_err(|e| LcelError::Chain(e.to_string()));
        if out.is_ok() {
            self.renders.fetch_add(1, Ordering::Relaxed);
        }
        out
    }
}

/// A source-keyed cache of [`PromptTemplate`] instances.
///
/// Constructing the same template string twice normally re-parses it each time.
/// [`PromptTemplateCache::get_or_insert`] parses once per unique template and
/// hands back the shared [`Arc`], so a hot prompt that is continuously rebuilt
/// pays the parse cost once and keeps one shared render counter. The cache is
/// unbounded; for very large sets of distinct templates consider evicting whole
/// entries via [`clear`](Self::clear) rather than per-template accounting.
#[derive(Debug, Default)]
pub struct PromptTemplateCache {
    inner: RwLock<HashMap<String, Arc<PromptTemplate>>>,
}

impl PromptTemplateCache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the cached template for `template`, or parses-and-caches it.
    ///
    /// Two calls with the same source string yield the same [`Arc`]: subsequent
    /// renders share one parsed segment list and one render counter.
    pub fn get_or_insert(&self, template: impl Into<String>) -> Arc<PromptTemplate> {
        let template = template.into();
        if let Some(t) = self
            .inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&template)
        {
            return Arc::clone(t);
        }
        // Parse outside the write lock so concurrent first-renderers of
        // different templates don't serialize on template parsing.
        let instance = Arc::new(PromptTemplate::new(template.clone()));
        let mut guard = match self.inner.write() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        guard.entry(template).or_insert(instance).clone()
    }

    /// Number of distinct templates currently cached.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether the cache holds no templates.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evicts every cached template (their render counters are discarded too).
    pub fn clear(&self) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_template() {
        let template = PromptTemplate::new("你好，{name}！");
        let mut vars = HashMap::new();
        vars.insert("name", "小明");

        let result = template.format(&vars).unwrap();
        assert_eq!(result, "你好，小明！");
    }

    #[test]
    fn test_multiple_variables() {
        let template = PromptTemplate::new("{greeting}，{name}！今天是{day}。");
        let mut vars = HashMap::new();
        vars.insert("greeting", "早上好");
        vars.insert("name", "小红");
        vars.insert("day", "星期一");

        let result = template.format(&vars).unwrap();
        assert_eq!(result, "早上好，小红！今天是星期一。");
    }

    #[test]
    fn test_missing_variable() {
        let template = PromptTemplate::new("你好，{name}！今天是{day}。");
        let mut vars = HashMap::new();
        vars.insert("name", "小明");

        let result = template.format(&vars);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("day"));
    }

    #[test]
    fn test_get_variables() {
        let template = PromptTemplate::new("{a}, {b}, {c}");
        let vars = template.variables();
        assert_eq!(vars, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_escaped_braces() {
        let template = PromptTemplate::new("{{literal}} {name} }}");
        let mut vars = HashMap::new();
        vars.insert("name", "N");

        let result = template.format(&vars).unwrap();
        assert_eq!(result, "{literal} N }");
    }

    #[test]
    fn test_cjk_variable_name() {
        let template = PromptTemplate::new("你好，{姓名}！");
        let mut vars = HashMap::new();
        vars.insert("姓名", "小明");

        let result = template.format(&vars).unwrap();
        assert_eq!(result, "你好，小明！");
    }

    #[test]
    fn test_validate_well_formed_ok() {
        // valid names + escaped braces → no malformed placeholders
        let template = PromptTemplate::new("a {x} {{literal}} b");
        assert!(template.validate().is_ok());
    }

    #[test]
    fn test_validate_malformed_errors() {
        // unclosed placeholder is flagged by validate, even though format
        // would fall back to literal text
        let template = PromptTemplate::new("x {name");
        let err = template.validate().unwrap_err();
        assert!(err.to_string().contains("unclosed"));
    }

    // ---- T2: variable diff and strict formatting ----

    #[test]
    fn test_variable_diff_reports_both_directions() {
        let template = PromptTemplate::new("{a} {b}");
        let diff = template.variable_diff(["a", "typo"]);

        assert_eq!(diff.missing, vec!["b".to_string()]);
        assert_eq!(diff.unused, vec!["typo".to_string()]);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_variable_diff_is_empty_for_an_exact_match() {
        let template = PromptTemplate::new("{a} {b} {a}");
        assert!(template.variable_diff(["b", "a"]).is_empty());
    }

    #[test]
    fn test_format_strict_accepts_an_exact_match() {
        let template = PromptTemplate::new("你好，{name}！");
        let mut vars = HashMap::new();
        vars.insert("name", "小明");

        assert_eq!(template.format_strict(&vars).unwrap(), "你好，小明！");
    }

    #[test]
    fn test_format_strict_rejects_unused_variables() {
        // lenient `format` silently ignores the typo; strict mode names it
        let template = PromptTemplate::new("你好，{name}！");
        let mut vars = HashMap::new();
        vars.insert("name", "小明");
        vars.insert("nmae", "小明");

        assert!(template.format(&vars).is_ok());
        match template.format_strict(&vars).unwrap_err() {
            PromptsError::UnusedVariables { variables, .. } => {
                assert_eq!(variables, vec!["nmae".to_string()]);
            }
            other => panic!("expected UnusedVariables, got {other:?}"),
        }
    }

    #[test]
    fn test_format_strict_reports_a_missing_variable_like_format() {
        let template = PromptTemplate::new("你好，{name}！今天是{day}。");
        let mut vars = HashMap::new();
        vars.insert("name", "小明");

        let strict = template.format_strict(&vars).unwrap_err();
        let lenient = template.format(&vars).unwrap_err();

        // same variant, same variable — strict mode is a drop-in replacement
        assert!(matches!(strict, PromptsError::MissingVariable(ref v) if v == "day"));
        assert_eq!(strict.to_string(), lenient.to_string());
    }

    #[test]
    fn test_format_strict_rejects_malformed_placeholders() {
        let template = PromptTemplate::new("x {name");
        let mut vars = HashMap::new();
        vars.insert("name", "N");

        assert!(matches!(
            template.format_strict(&vars).unwrap_err(),
            PromptsError::MalformedTemplate(_)
        ));
    }

    #[tokio::test]
    async fn test_runnable_invoke() {
        let template = PromptTemplate::new("你好，{name}！");
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "小明".to_string());

        let result = template.invoke(vars, None).await.unwrap();
        assert_eq!(result, "你好，小明！");
    }

    #[tokio::test]
    async fn test_runnable_pipe() {
        use lc_core::runnables::{RunnableExt, RunnableLambda};

        let template = PromptTemplate::new("你好，{name}！");
        let chain = template.pipe(RunnableLambda::new_sync(|s: String| s.contains("小明")));

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "小明".to_string());
        let result = chain.invoke(vars, None).await.unwrap();
        assert!(result);
    }

    // ---- L267: render stats + template cache ----

    #[test]
    fn test_render_count_tracks_successful_renders() {
        let template = PromptTemplate::new("你好，{name}！");
        assert_eq!(template.render_count(), 0);

        let mut ok_vars = HashMap::new();
        ok_vars.insert("name", "小明");
        template.format(&ok_vars).unwrap();
        template.format(&ok_vars).unwrap();
        assert_eq!(template.render_count(), 2);

        // A failed render (missing variable) is not counted.
        let mut bad_vars = HashMap::new();
        bad_vars.insert("name", "小明");
        template.format(&bad_vars).unwrap(); // still ok
        let missing = HashMap::new();
        assert!(template.format(&missing).is_err());
        assert_eq!(template.render_count(), 3);

        template.reset_render_count();
        assert_eq!(template.render_count(), 0);
    }

    #[test]
    fn test_render_count_counts_strict_and_runnable_paths() {
        let template = PromptTemplate::new("{a} {b}");

        let mut vars = HashMap::new();
        vars.insert("a", "x");
        vars.insert("b", "y");
        template.format_strict(&vars).unwrap();
        assert_eq!(
            template.render_count(),
            1,
            "strict path must bump the counter"
        );

        let mut owned = HashMap::new();
        owned.insert("a".to_string(), "x".to_string());
        owned.insert("b".to_string(), "y".to_string());
        // Runnable::invoke via the trait method on PromptTemplate.
        use lc_core::runnables::Runnable;
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(template.invoke(owned, None)).unwrap();
        assert_eq!(
            template.render_count(),
            2,
            "runnable path must bump the counter"
        );
    }

    #[test]
    fn test_template_cache_parses_once_and_shared_counter() {
        let cache = PromptTemplateCache::new();
        assert!(cache.is_empty());

        let a = cache.get_or_insert("你好，{name}！");
        let b = cache.get_or_insert("你好，{name}！");
        let c = cache.get_or_insert("另一个 {t}");

        assert_eq!(cache.len(), 2, "two distinct templates cached");
        // Same source -> same Arc (shared parsed segments + shared counter).
        assert!(
            Arc::ptr_eq(&a, &b),
            "get_or_insert must return the identical Arc"
        );
        assert!(!Arc::ptr_eq(&a, &c));

        let mut vars = HashMap::new();
        vars.insert("name", "小明");
        a.format(&vars).unwrap();
        assert_eq!(b.render_count(), 1, "a and b share one render counter");

        cache.clear();
        assert!(cache.is_empty());
    }
}
