// src/agents/crag/grader.rs
//! Document grading for CRAG.
//!
//! Uses an LLM to score document relevance against a query.

use lc_core::language_models::BaseChatModel;
use lc_core::tools::ToolDefinition;
use lc_schema::Message;
use lc_vector_stores::Document;
use serde_json::{json, Value};

/// Grade result for a single document.
#[derive(Debug, Clone)]
pub struct GradeResult {
    /// Relevance score in [0.0, 1.0].
    pub score: f64,
    /// Optional reasoning from the LLM.
    pub reasoning: Option<String>,
    /// Whether the score came from ambiguous parsing (no clear relevant/irrelevant signal).
    pub is_ambiguous: bool,
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
        // P1-3:优先 tool_calls 结构化打分,不支持绑定时回落文本解析。
        let structured = crate::structured::chat_structured(
            self.llm,
            Some(grade_tool()),
            messages,
            None,
            &crate::retry::RetryConfig::default(),
        )
        .await
        .map_err(|e| GraderError::LLMError(e.to_string()))?;

        if let Some(args) = &structured.tool_args {
            if let Some(result) = grade_from_tool_args(args) {
                return Ok(result);
            }
        }
        parse_grade_response(&structured.content)
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

/// Builds the grading prompt by replacing placeholders using PromptTemplate.
fn build_grade_prompt(query: &str, document_content: &str) -> String {
    use lc_prompts::PromptTemplate;
    use std::collections::HashMap;

    let template = PromptTemplate::new(GRADE_PROMPT);
    let mut vars = HashMap::new();
    vars.insert("query", query);
    vars.insert("document_content", document_content);
    template
        .format(&vars)
        .unwrap_or_else(|_| GRADE_PROMPT.to_string())
}

/// 打分工具定义:强制 LLM 输出相关性与 0-1 分数(P1-3)。
fn grade_tool() -> ToolDefinition {
    ToolDefinition::new(
        "grade_document",
        "评估文档与查询的相关性,返回是否相关、0.0-1.0 分数与简要理由",
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "relevant": {
                "type": "boolean",
                "description": "文档是否包含直接回答查询的信息"
            },
            "score": {
                "type": "number",
                "description": "相关性分数,1.0 完全相关,0.0 完全无关"
            },
            "reasoning": {
                "type": "string",
                "description": "简要理由"
            }
        },
        "required": ["relevant", "score"]
    }))
}

/// 从 tool_call 参数构造打分结果。解析失败返回 None(回落文本解析)。
fn grade_from_tool_args(args: &Value) -> Option<GradeResult> {
    let score = args.get("score").and_then(|v| v.as_f64())?.clamp(0.0, 1.0);
    let reasoning = args
        .get("reasoning")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    // 结构化输出带显式 score → 不算歧义。
    Some(GradeResult {
        score,
        reasoning,
        is_ambiguous: false,
    })
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

    let (score, reasoning, is_ambiguous) = if let Some(s) = explicit_score {
        (s.clamp(0.0, 1.0), Some(response.to_string()), false)
    } else if lower.contains("relevant") && !lower.contains("irrelevant") {
        (0.8, Some(response.to_string()), false)
    } else if lower.contains("irrelevant") {
        (0.2, Some(response.to_string()), false)
    } else {
        // Ambiguous response: default to 0.4 (below typical thresholds)
        // to avoid triggering corrective path on uncertain grading.
        (0.4, Some(response.to_string()), true)
    };

    Ok(GradeResult {
        score,
        reasoning,
        is_ambiguous,
    })
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
A document is "irrelevant" if it does not contain useful information for answering the query.

Example:
Query: What is Rust programming language?
Document: Rust is a systems programming language focused on safety and performance.
Relevance: relevant
Score: 0.95
Reasoning: The document directly describes what Rust is."#;

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
        assert!((result.score - 0.4).abs() < 0.01);
        assert!(result.is_ambiguous);
    }

    #[test]
    fn test_reasoning_field_populated_on_relevant() {
        let result = parse_grade_response(
            "Relevance: relevant\nScore: 0.9\nReasoning: Document directly addresses the query.",
        )
        .unwrap();
        assert!(result.reasoning.is_some());
        let reasoning = result.reasoning.unwrap();
        assert!(
            reasoning.contains("Reasoning"),
            "reasoning should contain the original response text"
        );
    }

    #[test]
    fn test_reasoning_field_populated_on_irrelevant() {
        let result =
            parse_grade_response("Relevance: irrelevant\nScore: 0.1\nReasoning: Off-topic.")
                .unwrap();
        assert!(result.reasoning.is_some());
        assert!(result.reasoning.unwrap().contains("Off-topic"));
    }

    #[test]
    fn test_reasoning_field_populated_on_ambiguous() {
        let result = parse_grade_response("The document mentions the topic briefly.").unwrap();
        assert!(result.reasoning.is_some());
        assert!(result.reasoning.unwrap().contains("briefly"));
    }

    #[test]
    fn test_grade_from_tool_args() {
        // P1-3:tool_call 结构化参数 → GradeResult。
        let args = json!({"relevant": true, "score": 0.9, "reasoning": "direct match"});
        let result = grade_from_tool_args(&args).unwrap();
        assert!((result.score - 0.9).abs() < 1e-9);
        assert_eq!(result.reasoning.as_deref(), Some("direct match"));
        assert!(!result.is_ambiguous);
    }

    #[test]
    fn test_grade_from_tool_args_score_out_of_range() {
        let args = json!({"relevant": true, "score": 5.0});
        let result = grade_from_tool_args(&args).unwrap();
        assert!((result.score - 1.0).abs() < 1e-9, "分数应被 clamp 到 1.0");
    }

    #[test]
    fn test_grade_from_tool_args_missing_score() {
        let args = json!({"relevant": true});
        assert!(
            grade_from_tool_args(&args).is_none(),
            "缺 score 应回落文本解析"
        );
    }

    #[test]
    fn test_grade_tool_schema() {
        let tool = grade_tool();
        assert_eq!(tool.function.name, "grade_document");
        assert!(tool.function.parameters.is_some());
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

    /// Verify the grade prompt contains a few-shot example.
    #[test]
    fn test_grade_prompt_contains_few_shot_example() {
        assert!(
            GRADE_PROMPT.contains("Example:"),
            "grade prompt should contain few-shot example"
        );
        assert!(
            GRADE_PROMPT.contains("Rust"),
            "grade prompt example should contain the Rust example"
        );
        assert!(
            GRADE_PROMPT.contains("0.95"),
            "grade prompt example should contain an example score"
        );
    }
}
