// crates/lc-prompts/src/prompt_template.rs
//! Simple string template

use crate::error::PromptsError;
use crate::template_parser::{
    format_template, parse_template, template_variables, TemplateSegment,
};
use async_trait::async_trait;
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use std::collections::HashMap;

/// Prompt template
///
/// A template using `{variable}` format with variable substitution; `{{` and `}}` escape
/// literal braces.
pub struct PromptTemplate {
    template: String,
    segments: Vec<TemplateSegment>,
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
        Self { template, segments }
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
        format_template(&self.template, &self.segments, variables)
    }

    /// Returns all variable names the template needs
    ///
    /// # Returns
    /// The variable-name list
    pub fn variables(&self) -> Vec<String> {
        template_variables(&self.segments)
    }

    /// Returns the raw template string
    pub fn template(&self) -> &str {
        &self.template
    }
}

impl std::fmt::Display for PromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.template)
    }
}

// Runnable form: lets the prompt join an LCEL chain, so `prompt.pipe(...)` works.
// Receives an owned variable map (HashMap<String, String>), converting to references and
// delegating to `format`. Same as `ChatPromptTemplate`; errors go through `LcelError::Chain`.
#[async_trait]
impl Runnable<HashMap<String, String>, String> for PromptTemplate {
    type Error = LcelError;

    async fn invoke(
        &self,
        input: HashMap<String, String>,
        _config: Option<RunnableConfig>,
    ) -> Result<String, LcelError> {
        // `format` takes &HashMap<&str, &str>; convert from the owned map to references here
        let vars: HashMap<&str, &str> = input
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        self.format(&vars)
            .map_err(|e| LcelError::Chain(e.to_string()))
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
}
