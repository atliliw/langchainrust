// src/agents/crag/grader.rs
//! Document grading for CRAG.
//!
//! Uses an LLM to score document relevance against a query.

use crate::core::language_models::BaseChatModel;
use crate::schema::Message;
use crate::vector_stores::Document;

/// Grade result for a single document.
#[derive(Debug, Clone)]
pub struct GradeResult {
    /// Relevance score in [0.0, 1.0].
    pub score: f64,
    /// Optional reasoning from the LLM.
    pub reasoning: Option<String>,
}

/// Grades documents for relevance to a query using an LLM.
pub struct DocumentGrader<'a, M: BaseChatModel> {
    llm: &'a M,
}

impl<'a, M: BaseChatModel> DocumentGrader<'a, M> {
    /// Creates a new document grader.
    pub fn new(llm: &'a M) -> Self {
        Self { llm }
    }

    /// Grades a single document for relevance to the query.
    ///
    /// The LLM is asked to classify the document as "relevant" or "irrelevant"
    /// and provide a confidence score. The score is parsed from the response.
    pub async fn grade(
        &self,
        query: &str,
        document: &Document,
    ) -> Result<GradeResult, GraderError> {
        let prompt = build_grade_prompt(query, &document.content);

        let messages = vec![Message::human(&prompt)];
        let result = self
            .llm
            .chat(messages, None)
            .await
            .map_err(|e| GraderError::LLMError(e.to_string()))?;

        parse_grade_response(&result.content)
    }

    /// Grades multiple documents in parallel.
    ///
    /// Returns a vector of grade results in the same order as the input documents.
    pub async fn grade_all(
        &self,
        query: &str,
        documents: &[Document],
    ) -> Result<Vec<GradeResult>, GraderError> {
        use futures_util::future::join_all;

        let futures: Vec<_> = documents.iter().map(|doc| self.grade(query, doc)).collect();

        let results = join_all(futures).await;
        // Collect results, returning the first error if any
        results.into_iter().collect()
    }
}

/// Builds the grading prompt by replacing placeholders.
fn build_grade_prompt(query: &str, document_content: &str) -> String {
    GRADE_PROMPT
        .replace("{query}", query)
        .replace("{document_content}", document_content)
}

/// Parses the LLM grading response into a structured result.
///
/// Expected format: the LLM should respond with "relevant" or "irrelevant"
/// and optionally a numeric score. We parse flexibly:
/// - If "relevant" appears: base score 0.8, boosted by any explicit score
/// - If "irrelevant" appears: base score 0.2, adjusted by any explicit score
/// - If a numeric score (0-1) is found, use it directly
fn parse_grade_response(response: &str) -> Result<GradeResult, GraderError> {
    let lower = response.to_lowercase();

    // Try to extract an explicit numeric score first.
    let explicit_score = extract_numeric_score(&lower);

    let (score, reasoning) = if let Some(s) = explicit_score {
        (s.clamp(0.0, 1.0), Some(response.to_string()))
    } else if lower.contains("relevant") && !lower.contains("irrelevant") {
        (0.8, Some(response.to_string()))
    } else if lower.contains("irrelevant") {
        (0.2, Some(response.to_string()))
    } else {
        // Ambiguous response: default to moderate score
        (0.5, Some(response.to_string()))
    };

    Ok(GradeResult { score, reasoning })
}

/// Extracts a numeric score from the response text.
///
/// Looks for patterns like "score: 0.7", "0.7", "7/10", etc.
/// Avoids misinterpreting version numbers (e.g., "v1.0", "2.0.3") by
/// requiring scores to be in [0.0, 1.0] or [1.0, 10.0] range and
/// skipping tokens that look like version identifiers.
fn extract_numeric_score(text: &str) -> Option<f64> {
    // First try X/Y fraction patterns (e.g., "7/10", "8/10")
    for part in text.split(|c: char| c.is_whitespace() || c == ',' || c == ';') {
        let trimmed = part.trim();
        if let Some(slash_pos) = trimmed.find('/') {
            let numerator_str = trimmed[..slash_pos].trim();
            let denominator_str = trimmed[slash_pos + 1..].trim();
            if let (Ok(num), Ok(den)) =
                (numerator_str.parse::<f64>(), denominator_str.parse::<f64>())
            {
                if den > 0.0 {
                    let ratio = num / den;
                    if (0.0..=1.0).contains(&ratio) {
                        return Some(ratio);
                    }
                }
            }
        }
    }

    // Then try "score: X" or "Score: X" patterns first (most reliable)
    for part in text.split(|c: char| c.is_whitespace() || c == ',' || c == ';') {
        let trimmed = part.trim().to_lowercase();
        if let Some(rest) = trimmed.strip_prefix("score") {
            let candidate = rest.trim_start_matches([':', '=', ' ']);
            if let Ok(val) = candidate.parse::<f64>() {
                if (0.0..=1.0).contains(&val) {
                    return Some(val);
                }
                if (1.0..=10.0).contains(&val) {
                    return Some(val / 10.0);
                }
            }
        }
    }

    // Finally try plain numeric values, but skip version-like patterns
    for part in text.split(|c: char| !c.is_ascii_digit() && c != '.') {
        if part.is_empty() {
            continue;
        }
        // Skip version-like patterns: "1.0" preceded by "v" or containing multiple dots
        // e.g., "2.0.3" is a version, not a score
        if part.matches('.').count() > 1 {
            continue;
        }
        if let Ok(val) = part.parse::<f64>() {
            if (0.0..=1.0).contains(&val) {
                return Some(val);
            }
            // Handle scores like "7" meaning 0.7 (on a 0-10 scale)
            if (1.0..=10.0).contains(&val) {
                return Some(val / 10.0);
            }
        }
    }
    None
}

/// Grading error types.
#[derive(Debug, thiserror::Error)]
pub enum GraderError {
    /// LLM invocation failed.
    #[error("LLM error during grading: {0}")]
    LLMError(String),

    /// Failed to parse the grading response.
    #[error("Failed to parse grading response: {0}")]
    ParseError(String),
}

/// Prompt template for document grading.
const GRADE_PROMPT: &str = r#"You are a document relevance grader. Given a user query and a document, determine if the document is relevant to answering the query.

Query: {query}

Document: {document_content}

Instructions:
1. Read the query and document carefully.
2. Determine if the document contains information that directly helps answer the query.
3. Respond with your assessment in this exact format:

Relevance: [relevant/irrelevant]
Score: [0.0 to 1.0]
Reasoning: [brief explanation]

A score of 1.0 means the document is perfectly relevant, 0.0 means completely irrelevant.
A document is "relevant" if it contains information that directly addresses the query.
A document is "irrelevant" if it does not contain useful information for answering the query."#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_relevant_response() {
        let result = parse_grade_response(
            "Relevance: relevant\nScore: 0.9\nReasoning: Document directly addresses the query.",
        )
        .unwrap();
        assert!(result.score >= 0.8);
    }

    #[test]
    fn test_parse_irrelevant_response() {
        let result = parse_grade_response(
            "Relevance: irrelevant\nScore: 0.1\nReasoning: Document is about a different topic.",
        )
        .unwrap();
        assert!(result.score <= 0.3);
    }

    #[test]
    fn test_parse_explicit_score() {
        let result =
            parse_grade_response("Score: 0.75, the document is somewhat relevant.").unwrap();
        assert!((result.score - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_parse_ambiguous_response() {
        let result = parse_grade_response("The document mentions the topic briefly.").unwrap();
        assert!((result.score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_extract_numeric_score_decimal() {
        assert_eq!(extract_numeric_score("score: 0.85"), Some(0.85));
    }

    #[test]
    fn test_extract_numeric_score_out_of_ten() {
        assert_eq!(extract_numeric_score("7/10"), Some(0.7));
    }

    #[test]
    fn test_extract_numeric_score_none() {
        assert_eq!(extract_numeric_score("no score here"), None);
    }

    #[test]
    fn test_build_grade_prompt() {
        let prompt = build_grade_prompt("What is Rust?", "Rust is a systems language.");
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("Rust is a systems language."));
        assert!(prompt.contains("Relevance:"));
    }
}
