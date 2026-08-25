// lc-agents/src/deep_research/types.rs
//! Public data types (citation / report) and the gap→query parser.

use super::planner;

/// A citation referencing a source used in the research report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Citation {
    /// 1-based citation index used in the report body (e.g. \[1\]).
    pub index: usize,
    /// Human-readable source title or description.
    pub source: String,
    /// Optional URL for the source.
    pub url: Option<String>,
    /// Short snippet quoted or paraphrased from the source.
    pub snippet: String,
}

/// The final output of a deep research run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResearchReport {
    /// The full report in markdown format with inline citation markers.
    pub markdown: String,
    /// Ordered list of citations referenced in the report.
    pub citations: Vec<Citation>,
    /// Sub-topics that were investigated.
    pub subtopics: Vec<String>,
    /// Number of research rounds completed.
    pub rounds_completed: usize,
}

/// Parses the LLM output for gap→query mapping into a flat list of queries.
///
/// Expected format: `[{"gap": "...", "queries": ["q1", "q2"]}, ...]`
///
/// Validates that every gap in the input list has at least one corresponding query.
/// If a gap has no queries in the parsed output, a warning is logged and a
/// fallback query is generated from the gap text itself.
pub(crate) fn parse_gap_queries(
    content: &str,
    original_gaps: &[String],
) -> Result<Vec<String>, super::ResearchError> {
    #[derive(serde::Deserialize)]
    struct GapMapping {
        gap: String,
        queries: Vec<String>,
    }

    let json_str = planner::extract_json(content);
    let mappings: Vec<GapMapping> = serde_json::from_str(&json_str).map_err(|e| {
        let preview: String = content.chars().take(200).collect();
        super::ResearchError::Llm(format!(
            "failed to parse gap→query mapping: {} | raw: {}",
            e, preview
        ))
    })?;

    // Collect all queries, ensuring each original gap is covered.
    let mut all_queries: Vec<String> = Vec::new();
    let mut covered_gaps: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for mapping in &mappings {
        // Check if this mapping corresponds to any original gap (fuzzy match by substring)
        for (i, gap) in original_gaps.iter().enumerate() {
            if !covered_gaps.contains(&i)
                && (mapping.gap.contains(gap.as_str()) || gap.contains(mapping.gap.as_str()))
            {
                covered_gaps.insert(i);
            }
        }
        all_queries.extend(mapping.queries.iter().filter(|q| !q.is_empty()).cloned());
    }

    // For any uncovered gap, generate a fallback query from the gap text itself.
    for (i, gap) in original_gaps.iter().enumerate() {
        if !covered_gaps.contains(&i) {
            log::warn!(
                "Deep Research: gap '{}' has no follow-up queries, using gap as query",
                gap
            );
            all_queries.push(gap.clone());
        }
    }

    Ok(all_queries)
}
