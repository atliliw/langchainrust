// src/agents/deep_research/synthesizer.rs
//! Synthesizer - aggregates search results and uses the LLM to write a
//! comprehensive markdown report with inline citations, then identifies
//! any remaining information gaps for follow-up rounds.

use crate::core::language_models::BaseChatModel;
use crate::schema::Message;

use super::planner::ResearchPlan;
use super::searcher::SearchResult;
use super::ResearchError;

/// Output of the synthesis step: a markdown report and a list of
/// information gaps that need follow-up queries.
pub struct SynthesisOutput {
    /// The markdown report with inline citation markers like [1], [2].
    pub report: String,
    /// Information gaps that remain after this synthesis pass.
    pub gaps: Vec<String>,
}

/// Synthesizes a research report from the plan and collected search results.
///
/// Returns the markdown report and a list of information gaps.
pub async fn synthesize<M: BaseChatModel>(
    llm: &M,
    topic: &str,
    plan: &ResearchPlan,
    results: &[SearchResult],
) -> Result<(String, Vec<String>), ResearchError> {
    let subtopics_text = plan
        .subtopics
        .iter()
        .enumerate()
        .map(|(i, st)| {
            format!(
                "{}. {} (queries: {})",
                i + 1,
                st.name,
                st.queries.join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let sources_text = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let url_part = if r.url.is_empty() {
                String::new()
            } else {
                format!(" ({})", r.url)
            };
            format!("[{}] {}{}: {}", i + 1, r.title, url_part, r.snippet)
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "Research topic: {}\n\n\
         Sub-topics investigated:\n{}\n\n\
         Sources collected:\n{}\n\n\
         Write a comprehensive research report in markdown format. \
         Use inline citation markers [1], [2], etc. to reference the sources above. \
         The report should cover all sub-topics and synthesize findings across sources.\n\n\
         After the report, identify any information gaps that need further research. \
         Output a JSON object with two fields:\n\
         - \"report\": the full markdown report string\n\
         - \"gaps\": an array of strings describing information gaps (empty array if none)\n\n\
         Only output the JSON object, nothing else.",
        topic, subtopics_text, sources_text,
    );

    let messages = vec![
        Message::system(
            "You are a research synthesis assistant. Write comprehensive, \
             well-structured reports with proper citations. Output only valid JSON.",
        ),
        Message::human(prompt),
    ];

    let response = llm
        .chat(messages, None)
        .await
        .map_err(|e| ResearchError::Llm(format!("{:?}", e)))?;

    parse_synthesis(&response.content)
}

/// Parses the LLM synthesis output into a report and gaps list.
fn parse_synthesis(content: &str) -> Result<(String, Vec<String>), ResearchError> {
    let json_str = extract_json_object(content);

    #[derive(serde::Deserialize)]
    struct SynthesisResponse {
        report: String,
        #[serde(default)]
        gaps: Vec<String>,
    }

    let parsed: SynthesisResponse = serde_json::from_str(&json_str).map_err(|e| {
        let preview: String = content.chars().take(200).collect();
        ResearchError::Llm(format!(
            "failed to parse synthesis response: {} | raw: {}",
            e, preview
        ))
    })?;

    Ok((parsed.report, parsed.gaps))
}

/// Extracts a JSON object from LLM output, tolerating markdown fences
/// and surrounding text.
fn extract_json_object(content: &str) -> String {
    let trimmed = content.trim();

    // Strip markdown code fences
    let stripped = if trimmed.starts_with("```") {
        trimmed
            .strip_prefix("```json")
            .or_else(|| trimmed.strip_prefix("```"))
            .unwrap_or(trimmed)
            .strip_suffix("```")
            .unwrap_or(trimmed)
            .trim()
    } else {
        trimmed
    };

    // Find the outermost { and its matching }
    if let Some(start) = stripped.find('{') {
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escape_next = false;
        let bytes = stripped.as_bytes();

        for i in start..bytes.len() {
            let ch = bytes[i];
            if escape_next {
                escape_next = false;
                continue;
            }
            if ch == b'\\' {
                if in_string {
                    escape_next = true;
                }
                continue;
            }
            if ch == b'"' {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }
            if ch == b'{' {
                depth += 1;
            } else if ch == b'}' {
                depth -= 1;
                if depth == 0 {
                    return stripped[start..=i].to_string();
                }
            }
        }
    }

    stripped.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_object_plain() {
        let input = r#"{"report": "text", "gaps": []}"#;
        assert_eq!(extract_json_object(input), input);
    }

    #[test]
    fn test_extract_json_object_markdown() {
        let input = "```json\n{\"report\": \"text\", \"gaps\": []}\n```";
        let expected = r#"{"report": "text", "gaps": []}"#;
        assert_eq!(extract_json_object(input), expected);
    }

    #[test]
    fn test_extract_json_object_with_surrounding_text() {
        let input = r#"Here is the result: {"report": "text", "gaps": ["gap1"]} done."#;
        let expected = r#"{"report": "text", "gaps": ["gap1"]}"#;
        assert_eq!(extract_json_object(input), expected);
    }

    #[test]
    fn test_extract_json_object_nested() {
        let input = "{\"report\": \"# Title\\n\\nContent [1].\", \"gaps\": []}";
        assert_eq!(extract_json_object(input), input);
    }

    #[test]
    fn test_parse_synthesis_valid() {
        let content =
            "{\"report\": \"# Report\\n\\nContent [1].\", \"gaps\": [\"need more data\"]}";
        let (report, gaps) = parse_synthesis(content).unwrap();
        assert!(report.contains("# Report"));
        assert_eq!(gaps, vec!["need more data"]);
    }

    #[test]
    fn test_parse_synthesis_no_gaps() {
        let content = "{\"report\": \"# Report\\n\\nDone.\", \"gaps\": []}";
        let (report, gaps) = parse_synthesis(content).unwrap();
        assert!(report.contains("Done"));
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_parse_synthesis_missing_gaps_field() {
        // gaps defaults to empty vec via #[serde(default)]
        let content = "{\"report\": \"# Report\"}";
        let (_, gaps) = parse_synthesis(content).unwrap();
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_parse_synthesis_invalid() {
        let content = "not json";
        let result = parse_synthesis(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_json_object_with_braces_in_strings() {
        let input = r#"{"report": "Use {curly} braces", "gaps": []}"#;
        assert_eq!(extract_json_object(input), input);
    }
}
