// crates/lc-prompts/src/few_shot.rs
//! Few-shot prompt template (FewShotPromptTemplate)
//!
//! Provides the ability to embed examples in a prompt, helping the LLM understand the expected output format.

use crate::error::PromptsError;
use crate::PromptTemplate;
use std::collections::HashMap;

/// Example-selector trait
///
/// Selects the most suitable examples from an example pool.
pub trait ExampleSelector: Send + Sync {
    /// Selects examples
    fn select_examples(&self, input: &HashMap<String, String>) -> Vec<&HashMap<String, String>>;

    /// Selects examples under a length constraint.
    ///
    /// The default implementation delegates to [`select_examples`](ExampleSelector::select_examples).
    /// Implementations that need precise length accounting (like [`LengthBasedExampleSelector`])
    /// override this method.
    fn select_examples_by_length(
        &self,
        input: &HashMap<String, String>,
        example_prompt: &PromptTemplate,
        prefix: &str,
        suffix: &str,
    ) -> Vec<&HashMap<String, String>> {
        let _ = (example_prompt, prefix, suffix);
        self.select_examples(input)
    }

    /// Returns all examples
    fn examples(&self) -> &[HashMap<String, String>];

    /// Adds an example
    fn add_example(&mut self, example: HashMap<String, String>);
}

/// Length-based example selector
///
/// Dynamically selects an appropriate number of examples based on input length, ensuring the
/// maximum length limit is not exceeded.
pub struct LengthBasedExampleSelector {
    examples: Vec<HashMap<String, String>>,
    /// Maximum text length (characters), default 2048
    max_length: usize,
}

impl LengthBasedExampleSelector {
    /// Creates a length-based example selector
    pub fn new(examples: Vec<HashMap<String, String>>) -> Self {
        Self {
            examples,
            max_length: 2048,
        }
    }

    /// Sets the maximum text length (characters)
    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = max;
        self
    }
}

impl ExampleSelector for LengthBasedExampleSelector {
    fn select_examples(&self, input: &HashMap<String, String>) -> Vec<&HashMap<String, String>> {
        // Calculate input length to determine how many examples fit
        let input_text: String = input.values().cloned().collect::<Vec<_>>().join("");
        let input_len = input_text.len();
        // Use a simple heuristic: longer inputs leave less room for examples
        let available = self.max_length.saturating_sub(input_len);

        let mut selected = Vec::new();
        let mut used = 0usize;

        for example in &self.examples {
            // Estimate example length from its values
            let ex_len: usize = example.values().map(|v| v.len()).sum();
            if used + ex_len <= available || selected.is_empty() {
                selected.push(example);
                used += ex_len;
            } else {
                break;
            }
        }

        selected
    }

    fn select_examples_by_length(
        &self,
        input: &HashMap<String, String>,
        example_prompt: &PromptTemplate,
        prefix: &str,
        suffix: &str,
    ) -> Vec<&HashMap<String, String>> {
        // compute the input length
        let input_text: String = input.values().cloned().collect::<Vec<_>>().join("");
        let input_len = prefix.len() + suffix.len() + input_text.len();
        let available = self.max_length.saturating_sub(input_len);

        let mut selected = Vec::new();
        let mut used = 0usize;

        for example in &self.examples {
            // estimate with the real formatted length (including separator slack), ensuring the cap is not exceeded
            let example_vars: HashMap<&str, &str> = example
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();

            if let Ok(formatted) = example_prompt.format(&example_vars) {
                let ex_len = formatted.len() + 10; // plus slack for the separator
                if used + ex_len <= available || selected.is_empty() {
                    selected.push(example);
                    used += ex_len;
                } else {
                    break;
                }
            }
        }

        selected
    }

    fn examples(&self) -> &[HashMap<String, String>] {
        &self.examples
    }

    fn add_example(&mut self, example: HashMap<String, String>) {
        self.examples.push(example);
    }
}

/// Few-shot prompt template
///
/// Embeds examples in the prompt, helping the LLM understand the expected output format.
/// The equivalent of Python LangChain's `FewShotPromptTemplate`.
///
/// # Example
/// ```ignore
/// use langchainrust::prompts::{FewShotPromptTemplate, PromptTemplate};
/// use std::collections::HashMap;
///
/// let examples = vec![
///     HashMap::from([("input".into(), "苹果".into()), ("output".into(), "水果".into())]),
///     HashMap::from([("input".into(), "玫瑰".into()), ("output".into(), "花".into())]),
/// ];
///
/// let example_prompt = PromptTemplate::new("输入: {input} -> 输出: {output}");
///
/// let few_shot = FewShotPromptTemplate::new(
///     examples,
///     example_prompt,
///     "请将以下词语分类：",
///     "输入: {input} ->",
///     vec!["input"],
/// );
/// ```
pub struct FewShotPromptTemplate {
    /// Example list
    examples: Vec<HashMap<String, String>>,
    /// Prompt template for formatting a single example
    example_prompt: PromptTemplate,
    /// Example prefix (placed before all examples)
    prefix: String,
    /// Example suffix (placed after all examples, usually the user input)
    suffix: String,
    /// Separator between examples
    example_separator: String,
    /// Input-variable name list
    input_variables: Vec<String>,
    /// Optional custom example selector
    example_selector: Option<Box<dyn ExampleSelector>>,
}

impl FewShotPromptTemplate {
    #[allow(clippy::too_many_arguments)]
    /// Creates a FewShotPromptTemplate
    pub fn new(
        examples: Vec<HashMap<String, String>>,
        example_prompt: PromptTemplate,
        prefix: impl Into<String>,
        suffix: impl Into<String>,
        input_variables: Vec<String>,
    ) -> Self {
        Self {
            examples,
            example_prompt,
            prefix: prefix.into(),
            suffix: suffix.into(),
            example_separator: "\n\n".to_string(),
            input_variables,
            example_selector: None,
        }
    }

    /// Sets the example separator
    pub fn with_example_separator(mut self, separator: impl Into<String>) -> Self {
        self.example_separator = separator.into();
        self
    }

    /// Sets a custom example selector
    pub fn with_example_selector(mut self, selector: Box<dyn ExampleSelector>) -> Self {
        self.example_selector = Some(selector);
        self
    }

    /// Sets the prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Sets the suffix
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }

    /// Formats the complete prompt
    pub fn format(&self, variables: &HashMap<&str, &str>) -> Result<String, PromptsError> {
        // verify all input variables have values
        for var in &self.input_variables {
            if !variables.contains_key(var.as_str()) {
                return Err(PromptsError::MissingVariable(var.clone()));
            }
        }

        // select examples
        let selected_examples: Vec<&HashMap<String, String>> =
            if let Some(ref selector) = self.example_selector {
                let input_map: HashMap<String, String> = variables
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
                selector.select_examples_by_length(
                    &input_map,
                    &self.example_prompt,
                    &self.prefix,
                    &self.suffix,
                )
            } else {
                self.examples.iter().collect()
            };

        // format each example
        let example_texts: Result<Vec<String>, PromptsError> = selected_examples
            .iter()
            .map(|example| {
                let example_vars: HashMap<&str, &str> = example
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect();
                self.example_prompt.format(&example_vars)
            })
            .collect();

        let example_texts = example_texts?;
        let examples_str = example_texts.join(&self.example_separator);

        // Format the suffix (user input).
        // Validated through PromptTemplate; a missing variable errors out instead of silently
        // keeping the literal `{var}` text.
        let suffix_formatted = if self.suffix.is_empty() {
            String::new()
        } else {
            PromptTemplate::new(self.suffix.clone()).format(variables)?
        };

        // compose the complete prompt
        let mut parts: Vec<String> = Vec::new();

        if !self.prefix.is_empty() {
            parts.push(self.prefix.clone());
        }

        if !examples_str.is_empty() {
            parts.push(examples_str);
        }

        if !suffix_formatted.is_empty() {
            parts.push(suffix_formatted);
        }

        Ok(parts.join("\n\n"))
    }

    /// Returns the input-variable list
    pub fn input_variables(&self) -> &[String] {
        &self.input_variables
    }

    /// Returns the example list
    pub fn examples(&self) -> &[HashMap<String, String>] {
        &self.examples
    }

    /// Adds an example
    pub fn add_example(&mut self, example: HashMap<String, String>) {
        self.examples.push(example);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_example(input: &str, output: &str) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("input".to_string(), input.to_string());
        map.insert("output".to_string(), output.to_string());
        map
    }

    #[test]
    fn test_few_shot_basic() {
        let examples = vec![make_example("苹果", "水果"), make_example("玫瑰", "花")];

        let example_prompt = PromptTemplate::new("输入: {input} -> 输出: {output}");
        let few_shot = FewShotPromptTemplate::new(
            examples,
            example_prompt,
            "请分类以下词语：",
            "输入: {input} ->",
            vec!["input".to_string()],
        );

        let mut vars = HashMap::new();
        vars.insert("input", "太阳");

        let result = few_shot.format(&vars).unwrap();
        assert!(result.contains("请分类以下词语"));
        assert!(result.contains("苹果"));
        assert!(result.contains("水果"));
        assert!(result.contains("太阳"));
    }

    #[test]
    fn test_few_shot_missing_variable() {
        let few_shot = FewShotPromptTemplate::new(
            vec![],
            PromptTemplate::new("示例: {input} -> {output}"),
            "",
            "输入: {input}",
            vec!["input".to_string(), "extra".to_string()],
        );

        let mut vars = HashMap::new();
        vars.insert("input", "test");

        let result = few_shot.format(&vars);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("extra"));
    }

    #[test]
    fn test_suffix_undeclared_variable_errors() {
        // Q3: an undeclared variable in the suffix must error, not silently keep the literal `{answer}` text
        let few_shot = FewShotPromptTemplate::new(
            vec![],
            PromptTemplate::new("{input} -> {output}"),
            "Prefix",
            "结果是: {answer}",
            vec!["input".to_string()],
        );

        let mut vars = HashMap::new();
        vars.insert("input", "x");

        let result = few_shot.format(&vars);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("answer"));
    }

    #[test]
    fn test_few_shot_empty_examples() {
        let few_shot = FewShotPromptTemplate::new(
            vec![],
            PromptTemplate::new("{input} -> {output}"),
            "Prefix",
            "Suffix: {input}",
            vec!["input".to_string()],
        );

        let mut vars = HashMap::new();
        vars.insert("input", "hello");

        let result = few_shot.format(&vars).unwrap();
        assert!(result.contains("Prefix"));
        assert!(result.contains("hello"));
        assert!(!result.contains("->")); // no examples, so no separator arrow expected
    }

    #[test]
    fn test_few_shot_custom_separator() {
        let examples = vec![make_example("a", "1"), make_example("b", "2")];

        let few_shot = FewShotPromptTemplate::new(
            examples,
            PromptTemplate::new("{input}={output}"),
            "",
            "",
            vec![],
        )
        .with_example_separator(" | ");

        let vars = HashMap::new();
        let result = few_shot.format(&vars).unwrap();
        assert_eq!(result, "a=1 | b=2");
    }

    #[test]
    fn test_few_shot_add_example() {
        let mut few_shot = FewShotPromptTemplate::new(
            vec![make_example("old", "value")],
            PromptTemplate::new("{input}={output}"),
            "",
            "",
            vec![],
        );

        assert_eq!(few_shot.examples().len(), 1);
        few_shot.add_example(make_example("new", "value2"));
        assert_eq!(few_shot.examples().len(), 2);
    }

    #[test]
    fn test_length_based_selector() {
        let examples = vec![make_example("long text here", "short")];

        let selector = LengthBasedExampleSelector::new(examples).with_max_length(100);

        let input_vars = HashMap::new();
        let selected = selector.select_examples(&input_vars);
        assert!(!selected.is_empty());
    }

    #[test]
    fn test_length_based_selector_by_length_is_object_safe() {
        let examples = vec![make_example("a", "1"), make_example("b", "2")];

        // called through a trait object: verifies the method is in the trait (Q2)
        let selector: Box<dyn ExampleSelector> =
            Box::new(LengthBasedExampleSelector::new(examples).with_max_length(100));

        let input_vars = HashMap::new();
        let example_prompt = PromptTemplate::new("{input}={output}");
        let selected = selector.select_examples_by_length(&input_vars, &example_prompt, "", "");
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_few_shot_with_selector() {
        let examples = vec![make_example("a", "1"), make_example("b", "2")];

        let selector = Box::new(LengthBasedExampleSelector::new(examples.clone()));
        let few_shot = FewShotPromptTemplate::new(
            examples,
            PromptTemplate::new("{input}={output}"),
            "Prefix",
            "{input}",
            vec!["input".to_string()],
        )
        .with_example_selector(selector);

        let mut vars = HashMap::new();
        vars.insert("input", "test");

        let result = few_shot.format(&vars).unwrap();
        assert!(result.contains("Prefix"));
    }
}
