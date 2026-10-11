// crates/lc-prompts/src/chat_prompt_template.rs
//! Chat message template

use crate::error::PromptsError;
use crate::template_parser::{
    format_template_vars, parse_template, template_variables, validate_template, TemplateSegment,
    TemplateVars,
};
use crate::validation::{dedup_in_order, diff_variables, VariableDiff};
use async_trait::async_trait;
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use lc_schema::{Message, MessageType};
use std::collections::HashMap;

/// Role name used in error messages and `Display`, so the two cannot drift.
fn role_name(message_type: &MessageType) -> &'static str {
    match message_type {
        MessageType::System => "System",
        MessageType::Human => "Human",
        MessageType::AI => "AI",
        MessageType::Tool { .. } => "Tool",
    }
}

/// Chat prompt template
///
/// Supports a multi-message template; each message's content can use `{variable}` format.
/// Message content is parsed once at construction; `format` replaces placeholders directly
/// instead of re-scanning the regex every time.
pub struct ChatPromptTemplate {
    messages: Vec<Message>,
    segments: Vec<Vec<TemplateSegment>>,
}

impl ChatPromptTemplate {
    /// Creates a new chat template
    ///
    /// # Arguments
    /// * `messages` - Message list; message content can use `{variable}` markers
    ///
    /// # Example
    /// ```ignore
    /// let template = ChatPromptTemplate::new(vec![
    ///     Message::system("你是一个{role}助手。"),
    ///     Message::human("你好，我是{name}。"),
    /// ]);
    /// let mut vars = HashMap::new();
    /// vars.insert("role", "编程");
    /// vars.insert("name", "小明");
    /// let messages = template.format(&vars).unwrap();
    /// ```
    pub fn new(messages: Vec<Message>) -> Self {
        let segments = messages
            .iter()
            .map(|msg| parse_template(&msg.content))
            .collect();
        Self { messages, segments }
    }

    /// Formats the template, replacing variables in all messages
    ///
    /// # Arguments
    /// * `variables` - Variable mapping
    ///
    /// # Returns
    /// The replaced message list, or a missing-variable error
    ///
    /// # Errors
    /// Returns an error if any message has a variable with no matching value in `variables`
    pub fn format(&self, variables: &HashMap<&str, &str>) -> Result<Vec<Message>, PromptsError> {
        self.format_with(variables)
    }

    /// Formats from any [`TemplateVars`] source (borrowed or owned), so the
    /// `Runnable::invoke` path — which holds an owned `HashMap<String, String>` —
    /// need not build an intermediate `HashMap<&str, &str>` per call.
    fn format_with<V: TemplateVars>(&self, variables: &V) -> Result<Vec<Message>, PromptsError> {
        self.messages
            .iter()
            .zip(&self.segments)
            .enumerate()
            .map(|(index, (msg, segments))| {
                // The cause is preserved as the error source, and the message's
                // position as fields, instead of being flattened into a
                // `MissingVariable` whose payload was a prose sentence rather
                // than a variable name.
                let content =
                    format_template_vars(&msg.content, segments, variables).map_err(|source| {
                        PromptsError::MessageFormat {
                            index,
                            role: role_name(&msg.message_type),
                            source: Box::new(source),
                        }
                    })?;
                let mut formatted = msg.clone();
                formatted.content = content;
                Ok(formatted)
            })
            .collect()
    }

    /// Returns all variable names the template needs
    ///
    /// # Returns
    /// The variable-name list (deduplicated, in message-then-appearance order)
    pub fn variables(&self) -> Vec<String> {
        let mut all = Vec::new();
        for segments in &self.segments {
            all.extend(template_variables(segments));
        }
        dedup_in_order(&all)
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
    /// stray `{` is emitted as literal text, and a supplied variable no message
    /// uses is ignored. This variant fails on both — use it where the template
    /// and its callers are both under your control and a typo should fail fast.
    /// A missing variable produces exactly the error `format` would, wrapped in
    /// [`PromptsError::MessageFormat`] with the offending message's index.
    ///
    /// # Errors
    /// Returns [`PromptsError::MalformedTemplate`] for a malformed placeholder,
    /// [`PromptsError::MissingVariable`] for an unsupplied template variable, or
    /// [`PromptsError::UnusedVariables`] when the caller supplied extras.
    pub fn format_strict(
        &self,
        variables: &HashMap<&str, &str>,
    ) -> Result<Vec<Message>, PromptsError> {
        self.validate()?;
        self.variable_diff(variables.keys().copied())
            .require_exact(&self.to_string())?;
        self.format(variables)
    }

    /// Returns the raw message templates
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Strictly validates that every message's placeholders are well-formed,
    /// returning an error on the first malformed one. See
    /// [`PromptTemplate::validate`](crate::PromptTemplate::validate) for the
    /// lenient-vs-strict rationale.
    ///
    /// # Errors
    /// Returns an error describing the first malformed placeholder, if any
    pub fn validate(&self) -> Result<(), PromptsError> {
        for msg in &self.messages {
            validate_template(&msg.content)?;
        }
        Ok(())
    }

    /// Quickly creates a template from a message array
    ///
    /// # Example
    /// ```ignore
    /// let template = ChatPromptTemplate::from_messages([
    ///     Message::system("你是一个{role}助手。"),
    ///     Message::human("{question}"),
    /// ]);
    /// ```
    pub fn from_messages(messages: impl Into<Vec<Message>>) -> Self {
        Self::new(messages.into())
    }
}

// Runnable form: lets the prompt join an LCEL chain, so `prompt.pipe(llm)` works.
// Receives an owned variable map (HashMap<String, String>) and formats straight from it
// via `format_with`, avoiding a per-call ref-map allocation.
#[async_trait]
impl Runnable<HashMap<String, String>, Vec<Message>> for ChatPromptTemplate {
    type Error = LcelError;

    async fn invoke(
        &self,
        input: HashMap<String, String>,
        _config: Option<RunnableConfig>,
    ) -> Result<Vec<Message>, LcelError> {
        self.format_with(&input)
            .map_err(|e| LcelError::Chain(e.to_string()))
    }
}

impl std::fmt::Display for ChatPromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for msg in &self.messages {
            writeln!(f, "{}: {}", role_name(&msg.message_type), msg.content)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chat_template() {
        let template = ChatPromptTemplate::new(vec![
            Message::system("你是一个{role}助手。"),
            Message::human("你好，我是{name}。"),
        ]);

        let mut vars = HashMap::new();
        vars.insert("role", "编程");
        vars.insert("name", "小明");

        let messages = template.format(&vars).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "你是一个编程助手。");
        assert_eq!(messages[1].content, "你好，我是小明。");
        // Non-content fields should be preserved (Q4: no more manual per-field cloning)
        assert!(matches!(
            messages[0].message_type,
            lc_schema::MessageType::System
        ));
        assert!(matches!(
            messages[1].message_type,
            lc_schema::MessageType::Human
        ));
    }

    #[test]
    fn test_from_messages() {
        let template = ChatPromptTemplate::from_messages([
            Message::system("系统消息"),
            Message::human("用户消息"),
        ]);

        assert_eq!(template.messages().len(), 2);
    }

    #[test]
    fn test_get_variables() {
        let template = ChatPromptTemplate::new(vec![
            Message::system("你是一个{role}，专精于{domain}。"),
            Message::human("我是{name}，请问{question}"),
        ]);

        let vars = template.variables();
        assert!(vars.contains(&"role".to_string()));
        assert!(vars.contains(&"domain".to_string()));
        assert!(vars.contains(&"name".to_string()));
        assert!(vars.contains(&"question".to_string()));
    }

    #[test]
    fn test_missing_variable() {
        let template = ChatPromptTemplate::new(vec![Message::human("你好，{name}！今天是{day}。")]);

        let mut vars = HashMap::new();
        vars.insert("name", "小明");

        let result = template.format(&vars);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("day"));
    }

    #[test]
    fn test_escaped_braces_in_message() {
        let template = ChatPromptTemplate::new(vec![Message::system(
            "请输出 JSON: {{\"key\": \"{value}\"}}",
        )]);

        let mut vars = HashMap::new();
        vars.insert("value", "v");

        let messages = template.format(&vars).unwrap();
        assert_eq!(messages[0].content, "请输出 JSON: {\"key\": \"v\"}");
    }

    #[test]
    fn test_multimodal_fields_preserved_on_format() {
        let msg = Message::human_with_image("{question}", "https://example.com/i.png");
        let template = ChatPromptTemplate::new(vec![msg]);

        let mut vars = HashMap::new();
        vars.insert("question", "看这张图");

        let messages = template.format(&vars).unwrap();
        assert_eq!(messages[0].content, "看这张图");
        assert_eq!(messages[0].images.len(), 1);
        assert_eq!(messages[0].images[0].url, "https://example.com/i.png");
    }

    // ---- T2: error provenance, deterministic order, variable diff ----

    #[test]
    fn test_error_keeps_the_cause_and_locates_the_message() {
        // `day` is missing from message 1. The error must say which message, and
        // the underlying error must still be reachable as the source — before,
        // both were flattened into a `MissingVariable` payload that was a prose
        // sentence instead of a variable name.
        let template = ChatPromptTemplate::new(vec![
            Message::system("系统"),
            Message::human("你好，{name}！今天是{day}。"),
        ]);

        let mut vars = HashMap::new();
        vars.insert("name", "小明");

        match template.format(&vars).unwrap_err() {
            PromptsError::MessageFormat {
                index,
                role,
                source,
            } => {
                assert_eq!(index, 1);
                assert_eq!(role, "Human");
                assert!(matches!(*source, PromptsError::MissingVariable(ref v) if v == "day"));
            }
            other => panic!("expected MessageFormat, got {other:?}"),
        }
    }

    #[test]
    fn test_error_source_is_exposed_as_the_std_error_source() {
        use std::error::Error;

        let template = ChatPromptTemplate::new(vec![Message::human("{day}")]);
        let err = template.format(&HashMap::new()).unwrap_err();

        let source = err.source().expect("cause should be reachable");
        assert!(source.to_string().contains("day"), "got: {source}");
    }

    #[test]
    fn test_variables_are_in_message_then_appearance_order() {
        // regression: a HashSet-backed implementation returned these in
        // arbitrary order, which made the list unusable for a stable diff
        let template = ChatPromptTemplate::new(vec![
            Message::system("{role} {domain}"),
            Message::human("{name} {role}"),
        ]);

        assert_eq!(template.variables(), ["role", "domain", "name"]);
    }

    #[test]
    fn test_variable_diff_reports_both_directions() {
        let template = ChatPromptTemplate::new(vec![Message::human("{name} {day}")]);
        let diff = template.variable_diff(["name", "typo"]);

        assert_eq!(diff.missing, vec!["day".to_string()]);
        assert_eq!(diff.unused, vec!["typo".to_string()]);
    }

    #[test]
    fn test_format_strict_accepts_an_exact_match() {
        let template = ChatPromptTemplate::new(vec![Message::human("你好，{name}！")]);
        let mut vars = HashMap::new();
        vars.insert("name", "小明");

        let messages = template.format_strict(&vars).unwrap();
        assert_eq!(messages[0].content, "你好，小明！");
    }

    #[test]
    fn test_format_strict_rejects_unused_variables() {
        let template = ChatPromptTemplate::new(vec![Message::human("你好，{name}！")]);
        let mut vars = HashMap::new();
        vars.insert("name", "小明");
        vars.insert("nmae", "小明");

        // lenient format ignores the typo; strict mode names it
        assert!(template.format(&vars).is_ok());
        assert!(matches!(
            template.format_strict(&vars).unwrap_err(),
            PromptsError::UnusedVariables { .. }
        ));
    }

    #[test]
    fn test_format_strict_rejects_malformed_placeholders() {
        // lenient format emits the stray `{`; strict mode refuses it
        let template = ChatPromptTemplate::new(vec![Message::human("x {name")]);
        let mut vars = HashMap::new();
        vars.insert("name", "N");

        assert!(template.format(&vars).is_ok());
        assert!(matches!(
            template.format_strict(&vars).unwrap_err(),
            PromptsError::MalformedTemplate(_)
        ));
    }
}
