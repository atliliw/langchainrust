// crates/lc-prompts/src/few_shot.rs
//! Few-shot prompt template (FewShotPromptTemplate)
//!
//! Provides the ability to embed examples in a prompt, helping the LLM understand the expected output format.

use crate::error::PromptsError;
use crate::template_parser::{parse_template, template_variables, validate_template};
use crate::validation::{dedup_in_order, diff_variables, VariableDiff};
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
        let input_len = input_text.chars().count();
        // Use a simple heuristic: longer inputs leave less room for examples
        let available = self.max_length.saturating_sub(input_len);

        let mut selected = Vec::new();
        let mut used = 0usize;

        for example in &self.examples {
            // Estimate example length from its values
            let ex_len: usize = example.values().map(|v| v.chars().count()).sum();
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
        let input_len =
            prefix.chars().count() + suffix.chars().count() + input_text.chars().count();
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
                let ex_len = formatted.chars().count() + 10; // plus slack for the separator
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
    /// Example prefix (placed before all examples, formatted with the caller's variables)
    prefix: String,
    /// Example suffix (placed after all examples, usually the user input,
    /// formatted with the caller's variables)
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

        // Format the prefix and the suffix — the two parts that take the caller's
        // variables — through PromptTemplate, so a missing variable errors out
        // instead of silently keeping the literal `{var}` text.
        //
        // The prefix used to be emitted verbatim, which left `{var}` in the
        // prompt with no error at all. Formatting it matches Python LangChain's
        // `FewShotPromptTemplate`, which this type mirrors.
        let format_part = |text: &str| -> Result<String, PromptsError> {
            if text.is_empty() {
                Ok(String::new())
            } else {
                PromptTemplate::new(text.to_string()).format(variables)
            }
        };
        let prefix_formatted = format_part(&self.prefix)?;
        let suffix_formatted = format_part(&self.suffix)?;

        // compose the complete prompt
        let mut parts: Vec<String> = Vec::new();

        if !prefix_formatted.is_empty() {
            parts.push(prefix_formatted);
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
    ///
    /// A template that passed [`validate`](Self::validate) can be invalidated by
    /// this, so call `validate` again after mutating.
    pub fn add_example(&mut self, example: HashMap<String, String>) {
        self.examples.push(example);
    }

    /// Validates the template's construction: well-formed placeholders, a
    /// declared-variable list that agrees with the prompt text, and examples that
    /// can actually be rendered.
    ///
    /// Nothing checked any of this before, and each omission failed quietly:
    ///
    /// 1. **Malformed placeholders** in the prefix, suffix or example prompt
    ///    (see [`PromptTemplate::validate`] for the lenient-vs-strict rationale).
    /// 2. **Declared vs used variables.** `input_variables` is the contract
    ///    `format` enforces, but nothing tied it to the prompt text: a suffix
    ///    variable that was never declared could only fail at call time, and a
    ///    declared variable the suffix never used made `format` demand a value it
    ///    would then ignore.
    /// 3. **Examples missing a variable the example prompt needs.** Without a
    ///    selector such an example fails the whole `format`; with one it was
    ///    silently *skipped* by
    ///    [`LengthBasedExampleSelector`], quietly shrinking the example pool.
    ///
    /// Note that mutating builders ([`add_example`](Self::add_example),
    /// [`with_prefix`](Self::with_prefix), [`with_suffix`](Self::with_suffix))
    /// can invalidate the result; call this after the last mutation.
    ///
    /// # Errors
    /// Returns [`PromptsError::MalformedTemplate`],
    /// [`PromptsError::FewShotVariableMismatch`] (reporting both directions at
    /// once), or [`PromptsError::ExampleMissingVariable`] for the first
    /// unrenderable example.
    pub fn validate(&self) -> Result<(), PromptsError> {
        // 1. placeholders must be well-formed
        validate_template(&self.prefix)?;
        validate_template(&self.suffix)?;
        self.example_prompt.validate()?;

        // 2. declared input variables must match what the prompt text uses.
        // Only the prefix and suffix take the caller's variables; the example
        // prompt is formatted from each example's own keys.
        let mut used = template_variables(&parse_template(&self.prefix));
        used.extend(template_variables(&parse_template(&self.suffix)));
        let used = dedup_in_order(&used);
        let declared = dedup_in_order(&self.input_variables);

        let undeclared: Vec<String> = used
            .iter()
            .filter(|var| !declared.contains(var))
            .cloned()
            .collect();
        let unused: Vec<String> = declared
            .iter()
            .filter(|var| !used.contains(var))
            .cloned()
            .collect();
        if !undeclared.is_empty() || !unused.is_empty() {
            return Err(PromptsError::FewShotVariableMismatch { undeclared, unused });
        }

        // 3. every example must supply what the example prompt needs
        let required = dedup_in_order(&self.example_prompt.variables());
        for (index, example) in self.examples.iter().enumerate() {
            if let Some(variable) = required.iter().find(|var| !example.contains_key(*var)) {
                return Err(PromptsError::ExampleMissingVariable {
                    index,
                    variable: variable.clone(),
                });
            }
        }

        Ok(())
    }

    /// Validates the template and returns it, so construction can fail fast:
    ///
    /// ```ignore
    /// let template = FewShotPromptTemplate::new(examples, example_prompt, prefix, suffix, vars)
    ///     .validated()?;
    /// ```
    ///
    /// # Errors
    /// Returns whatever [`validate`](Self::validate) returns.
    pub fn validated(self) -> Result<Self, PromptsError> {
        self.validate()?;
        Ok(self)
    }

    /// Compares the declared input variables against the ones a caller would
    /// supply — the call-time half of what [`validate`](Self::validate) checks at
    /// construction time.
    pub fn variable_diff<'a, I>(&self, supplied: I) -> VariableDiff
    where
        I: IntoIterator<Item = &'a str>,
    {
        diff_variables(&self.input_variables, supplied)
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
    fn test_length_based_selector_counts_chars_not_bytes() {
        // CJK characters occupy 3 bytes each in UTF-8. `max_length` is documented as
        // characters, so the selector must count characters (not bytes): under a 7-char
        // budget (empty input) both examples fit when counted by char, but a byte count
        // would fit only one.
        let examples = vec![make_example("苹果", "水果"), make_example("玫瑰", "花")];
        let selector = LengthBasedExampleSelector::new(examples).with_max_length(7);

        let input_vars = HashMap::new();
        let selected = selector.select_examples(&input_vars);
        assert_eq!(selected.len(), 2);
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

    // ---- T2: construct-time variable validation ----

    /// A consistent template, used as the positive case below.
    fn valid_template() -> FewShotPromptTemplate {
        FewShotPromptTemplate::new(
            vec![make_example("苹果", "水果")],
            PromptTemplate::new("{input} -> {output}"),
            "请用{lang}回答：",
            "输入: {input} ->",
            vec!["input".to_string(), "lang".to_string()],
        )
    }

    #[test]
    fn prefix_is_formatted_with_the_callers_variables() {
        // the prefix used to be emitted verbatim, leaving `{lang}` in the prompt
        // with no error — Python LangChain's FewShotPromptTemplate formats it
        let mut vars = HashMap::new();
        vars.insert("input", "太阳");
        vars.insert("lang", "中文");

        let result = valid_template().format(&vars).unwrap();
        assert!(result.contains("请用中文回答："), "got: {result}");
        assert!(!result.contains("{lang}"));
    }

    #[test]
    fn a_prefix_placeholder_with_no_value_errors() {
        let mut vars = HashMap::new();
        vars.insert("input", "太阳");
        // `lang` deliberately omitted: format must fail rather than emit `{lang}`

        let err = valid_template().format(&vars).unwrap_err();
        assert!(err.to_string().contains("lang"), "got: {err}");
    }

    #[test]
    fn validate_accepts_a_consistent_template() {
        assert!(valid_template().validate().is_ok());
    }

    #[test]
    fn validate_ignores_example_prompt_variables_in_the_declared_diff() {
        // `output` is used by the example prompt but filled from each example's
        // own keys, so it must not be reported as undeclared...
        let template = FewShotPromptTemplate::new(
            vec![make_example("a", "1")],
            PromptTemplate::new("{input} -> {output}"),
            "",
            "{input}",
            vec!["input".to_string()],
        );

        assert!(template.validate().is_ok());
        // ...and it is not an input variable the caller has to supply either
        assert_eq!(template.input_variables(), ["input"]);
    }

    #[test]
    fn validate_reports_undeclared_and_unused_variables_together() {
        // declared ["input", "spare"], but the prompt text only uses {topic}
        let template = FewShotPromptTemplate::new(
            vec![],
            PromptTemplate::new("{input} -> {output}"),
            "",
            "{topic}",
            vec!["input".to_string(), "spare".to_string()],
        );

        match template.validate().unwrap_err() {
            PromptsError::FewShotVariableMismatch { undeclared, unused } => {
                assert_eq!(undeclared, vec!["topic".to_string()]);
                assert_eq!(unused, vec!["input".to_string(), "spare".to_string()]);
            }
            other => panic!("expected FewShotVariableMismatch, got {other:?}"),
        }
    }

    #[test]
    fn validate_counts_prefix_and_suffix_variables_as_used() {
        // `lang` is used only by the prefix; it must not be reported as unused
        let template = valid_template();

        assert!(template.validate().is_ok());
        assert!(template.variable_diff(["input", "lang"]).is_empty());
    }

    #[test]
    fn validate_reports_malformed_placeholders() {
        let template = FewShotPromptTemplate::new(
            vec![],
            PromptTemplate::new("{input} -> {output}"),
            "前缀 {lang",
            "",
            vec!["lang".to_string()],
        );

        let err = template.validate().unwrap_err();
        assert!(
            matches!(err, PromptsError::MalformedTemplate(_)),
            "got: {err:?}"
        );
    }

    #[test]
    fn validate_reports_an_example_that_cannot_be_rendered() {
        // the second example lacks `output`, so it cannot be formatted; with a
        // selector it used to be skipped silently instead
        let template = FewShotPromptTemplate::new(
            vec![
                make_example("a", "1"),
                HashMap::from([("input".to_string(), "b".to_string())]),
            ],
            PromptTemplate::new("{input} -> {output}"),
            "",
            "{input}",
            vec!["input".to_string()],
        );

        match template.validate().unwrap_err() {
            PromptsError::ExampleMissingVariable { index, variable } => {
                assert_eq!(index, 1);
                assert_eq!(variable, "output");
            }
            other => panic!("expected ExampleMissingVariable, got {other:?}"),
        }
    }

    #[test]
    fn validate_catches_an_example_added_after_construction() {
        let mut template = valid_template();
        assert!(template.validate().is_ok());

        template.add_example(HashMap::from([("input".to_string(), "b".to_string())]));

        assert!(matches!(
            template.validate().unwrap_err(),
            PromptsError::ExampleMissingVariable { index: 1, .. }
        ));
    }

    #[test]
    fn validated_returns_the_template_when_consistent() {
        let template = valid_template().validated().unwrap();
        assert_eq!(template.input_variables(), ["input", "lang"]);
    }

    #[test]
    fn validated_propagates_the_validation_error() {
        let result = FewShotPromptTemplate::new(
            vec![],
            PromptTemplate::new("{input}"),
            "",
            "{input}",
            vec![],
        )
        .validated();

        // `.err()` rather than `.unwrap_err()`: the template holds a boxed
        // selector and so is not `Debug`, which `unwrap_err` requires of the Ok
        // side. Callers using `?` are unaffected.
        assert!(matches!(
            result.err().expect("validation should have failed"),
            PromptsError::FewShotVariableMismatch { .. }
        ));
    }

    #[test]
    fn variable_diff_reports_both_directions_at_call_time() {
        let diff = valid_template().variable_diff(["input", "typo"]);

        assert_eq!(diff.missing, vec!["lang".to_string()]);
        assert_eq!(diff.unused, vec!["typo".to_string()]);
    }
}
