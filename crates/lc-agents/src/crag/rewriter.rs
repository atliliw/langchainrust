// src/agents/crag/rewriter.rs
//! Query rewriting for CRAG.
//!
//! When retrieved documents score below the threshold, the query is
//! rewritten to improve retrieval quality.

use lc_core::language_models::BaseChatModel;
use lc_schema::Message;

/// Query rewriting error types.
#[derive(Debug, thiserror::Error)]
pub enum RewriterError {
    /// LLM invocation failed.
    #[error("LLM error during query rewriting: {0}")]
    LLMError(String),

    /// Failed to parse the rewritten query.
    #[error("Failed to extract rewritten query from LLM response: {0}")]
    ParseError(String),
}

/// Rewrites queries to improve retrieval quality.
pub struct QueryRewriter<'a, M: BaseChatModel> {
    llm: &'a M,
}

impl<'a, M: BaseChatModel> QueryRewriter<'a, M> {
    /// Creates a new query rewriter.
    pub fn new(llm: &'a M) -> Self {
        Self { llm }
    }

    /// Rewrites the query to be more effective for document retrieval.
    ///
    /// The rewriter generates alternative phrasings that may match
    /// documents the original query missed.
    pub async fn rewrite(&self, query: &str) -> Result<String, RewriterError> {
        let prompt = build_rewrite_prompt(query);

        let messages = vec![Message::human(&prompt)];
        let result = self
            .llm
            .chat(messages, None)
            .await
            .map_err(|e| RewriterError::LLMError(e.to_string()))?;

        let rewritten = extract_rewritten_query(&result.content);
        if rewritten.is_empty() {
            return Err(RewriterError::ParseError(result.content));
        }

        Ok(rewritten)
    }

    /// Generates multiple alternative queries for broader retrieval.
    ///
    /// Returns a list of rewritten queries including the original.
    pub async fn generate_alternatives(
        &self,
        query: &str,
        count: usize,
    ) -> Result<Vec<String>, RewriterError> {
        let prompt = build_alternatives_prompt(query, count);

        let messages = vec![Message::human(&prompt)];
        let result = self
            .llm
            .chat(messages, None)
            .await
            .map_err(|e| RewriterError::LLMError(e.to_string()))?;

        let alternatives = parse_alternatives(&result.content);
        if alternatives.is_empty() {
            // Fallback: return the original query
            return Ok(vec![query.to_string()]);
        }

        Ok(alternatives)
    }
}

/// Builds the single query rewrite prompt.
fn build_rewrite_prompt(query: &str) -> String {
    use lc_prompts::PromptTemplate;
    use std::collections::HashMap;

    let template = PromptTemplate::new(REWRITE_PROMPT);
    let mut vars = HashMap::new();
    vars.insert("query", query);
    template
        .format(&vars)
        .unwrap_or_else(|_| REWRITE_PROMPT.to_string())
}

/// Builds the alternatives generation prompt.
fn build_alternatives_prompt(query: &str, count: usize) -> String {
    use lc_prompts::PromptTemplate;
    use std::collections::HashMap;

    let template = PromptTemplate::new(ALTERNATIVES_PROMPT);
    let mut vars = HashMap::new();
    vars.insert("query", query);
    let count_str = count.to_string();
    vars.insert("count", &count_str);
    template
        .format(&vars)
        .unwrap_or_else(|_| ALTERNATIVES_PROMPT.to_string())
}

/// Extracts the rewritten query from the LLM response.
///
/// Looks for "Rewritten query:" prefix, or takes the first non-empty line
/// if no prefix is found.
fn extract_rewritten_query(response: &str) -> String {
    let trimmed = response.trim();

    // Try to find "Rewritten query:" prefix
    if let Some(pos) = trimmed.find("Rewritten query:") {
        let after = trimmed[pos + "Rewritten query:".len()..].trim();
        if let Some(line) = after.lines().next() {
            let cleaned = line.trim().trim_start_matches('-').trim();
            if !cleaned.is_empty() {
                return cleaned.to_string();
            }
        }
    }

    // Fallback: take the first non-empty line
    for line in trimmed.lines() {
        let cleaned = line.trim();
        if !cleaned.is_empty() && !cleaned.starts_with('#') {
            return cleaned.to_string();
        }
    }

    String::new()
}

/// Parses multiple alternative queries from the LLM response.
///
/// Expects numbered or bulleted list format.
fn parse_alternatives(response: &str) -> Vec<String> {
    let mut alternatives = Vec::new();

    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Strip numbered prefix like "1. " or "1) "
        let cleaned = trimmed
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .trim_start_matches(['.', ')', '-'])
            .trim();

        if !cleaned.is_empty() {
            alternatives.push(cleaned.to_string());
        }
    }

    alternatives
}

/// Prompt template for single query rewriting.
const REWRITE_PROMPT: &str = r#"You are a query rewriter. Your task is to rewrite the given query to improve document retrieval results.

Original query: {query}

Instructions:
1. Analyze the original query and identify the core information need.
2. Rewrite the query using different phrasing, synonyms, or more specific terms.
3. The rewritten query should be optimized for semantic search retrieval.
4. Keep the rewritten query concise and focused.

Respond with ONLY the rewritten query, no explanation needed.

Rewritten query:"#;

/// Prompt template for generating multiple alternative queries.
const ALTERNATIVES_PROMPT: &str = r#"You are a query rewriter. Generate {count} alternative versions of the following query to improve document retrieval coverage.

Original query: {query}

Instructions:
1. Each alternative should approach the information need from a different angle.
2. Use synonyms, related terms, or more specific phrasing.
3. Keep each alternative concise and focused.
4. Number each alternative.

Alternative queries:"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_rewritten_query_with_prefix() {
        let response = "Rewritten query: What are the key features of Rust programming language?";
        let result = extract_rewritten_query(response);
        assert_eq!(
            result,
            "What are the key features of Rust programming language?"
        );
    }

    #[test]
    fn test_extract_rewritten_query_without_prefix() {
        let response = "What are the main characteristics of the Rust language?";
        let result = extract_rewritten_query(response);
        assert_eq!(
            result,
            "What are the main characteristics of the Rust language?"
        );
    }

    #[test]
    fn test_extract_rewritten_query_multiline() {
        let response = "Here is the rewritten query:\nRewritten query: How does Rust ensure memory safety?\n\nThis focuses on the safety aspect.";
        let result = extract_rewritten_query(response);
        assert_eq!(result, "How does Rust ensure memory safety?");
    }

    #[test]
    fn test_extract_rewritten_query_empty() {
        let result = extract_rewritten_query("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_alternatives_numbered() {
        let response = "1. What is Rust?\n2. How does Rust work?\n3. Rust language overview";
        let result = parse_alternatives(response);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "What is Rust?");
        assert_eq!(result[1], "How does Rust work?");
        assert_eq!(result[2], "Rust language overview");
    }

    #[test]
    fn test_parse_alternatives_bulleted() {
        let response = "- First alternative\n- Second alternative";
        let result = parse_alternatives(response);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_alternatives_empty() {
        let result = parse_alternatives("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_rewrite_prompt() {
        let prompt = build_rewrite_prompt("What is Rust?");
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("Rewritten query:"));
    }

    #[test]
    fn test_build_alternatives_prompt() {
        let prompt = build_alternatives_prompt("What is Rust?", 3);
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("3"));
    }
}
