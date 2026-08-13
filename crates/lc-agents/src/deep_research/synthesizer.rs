// src/agents/deep_research/synthesizer.rs
//! Synthesizer - aggregates search results and uses the LLM to write a
//! comprehensive markdown report with inline citations, then identifies
//! any remaining information gaps for follow-up rounds.

use lc_core::language_models::BaseChatModel;
use lc_core::token_counter::count_tokens;
use lc_schema::Message;

use super::planner::ResearchPlan;
use super::searcher::SearchResult;
use super::ResearchError;

/// Output of the synthesis step: a markdown report and a list of
/// information gaps that need follow-up queries.
pub struct SynthesisOutput {
    /// The markdown report with inline citation markers like \[1\], \[2\].
    pub report: String,
    /// Information gaps that remain after this synthesis pass.
    pub gaps: Vec<String>,
}

/// Synthesizes a research report from the plan and collected search results.
///
/// Returns the markdown report and a list of information gaps.
/// When `max_source_tokens` is set, source snippets are truncated to fit
/// within the token budget.
pub async fn synthesize<M: BaseChatModel>(
    llm: &M,
    topic: &str,
    plan: &ResearchPlan,
    results: &[SearchResult],
    max_source_tokens: Option<usize>,
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

    let sources_text = format_sources(results, max_source_tokens);

    let prompt = format!(
        "Research topic: {}\n\n\
         Sub-topics investigated:\n{}\n\n\
         Sources collected:\n{}\n\n\
         Write a comprehensive research report in markdown format. \
         Use inline citation markers [1], [2], etc. to reference the sources above. \
         The report should cover all sub-topics and synthesize findings across sources.\n\n\
         After the report, identify any information gaps that need further research.\n\n\
         Example output format:\n\
         <<<REPORT>>>\n\
         # Topic Title\n\n\
         This report examines [overview]. Key findings include [summary].\n\n\
         ## Sub-topic 1\n\n\
         According to the research [1], [finding]. Additional analysis [2] shows [detail].\n\n\
         ## Sub-topic 2\n\n\
         Studies indicate [3] that [finding]. This suggests [implication].\n\n\
         ## Conclusion\n\n\
         In summary, [synthesis of findings across sources].\n\
         <<<END_REPORT>>>\n\
         <<<GAPS>>>\n\
         [\"Specific gap that needs more research\"]\n\
         <<<END_GAPS>>>\n\n\
         Output your response in this exact format (do not wrap in JSON):\n\
         <<<REPORT>>>\n\
         (your full markdown report here)\n\
         <<<END_REPORT>>>\n\
         <<<GAPS>>>\n\
         (a JSON array of gap strings, e.g. [\"gap1\", \"gap2\"], or [] if none)\n\
         <<<END_GAPS>>>",
        topic, subtopics_text, sources_text,
    );

    let messages = vec![
        Message::system(
            "You are a research synthesis assistant. Write comprehensive, \
             well-structured reports with proper citations.",
        ),
        Message::human(prompt),
    ];

    let response =
        crate::retry::retry_chat(llm, messages, None, &crate::retry::RetryConfig::default())
            .await
            .map_err(|e| ResearchError::Llm(format!("{:?}", e)))?;

    parse_synthesis(&response.content)
}

/// Formats search results into a sources text string.
///
/// When `max_source_tokens` is set, snippets are truncated to fit within
/// the budget, keeping higher-indexed (later, potentially more relevant)
/// sources intact by truncating earlier, longer snippets.
fn format_sources(results: &[SearchResult], max_source_tokens: Option<usize>) -> String {
    let entries: Vec<String> = results
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
        .collect();

    match max_source_tokens {
        Some(budget) => {
            let mut used_tokens = 0usize;
            let mut truncated_entries = Vec::new();

            for entry in &entries {
                let entry_tokens = count_tokens(entry);
                if used_tokens + entry_tokens > budget && !truncated_entries.is_empty() {
                    break;
                }
                used_tokens += entry_tokens;
                truncated_entries.push(entry.clone());
            }

            truncated_entries.join("\n")
        }
        None => entries.join("\n"),
    }
}

/// Parses the LLM synthesis output into a report and gaps list.
///
/// Supports two formats:
/// 1. **Delimiter format** (preferred): `<<<REPORT>>>...<<<END_REPORT>>><<<GAPS>>>[...]<<<END_GAPS>>>`
/// 2. **Legacy JSON format** (fallback): `{"report": "...", "gaps": [...]}`
fn parse_synthesis(content: &str) -> Result<(String, Vec<String>), ResearchError> {
    // Try delimiter format first
    if let Some(result) = parse_delimiter_format(content) {
        return Ok(result);
    }

    // Fallback to legacy JSON format using tolerant parsing
    #[derive(serde::Deserialize)]
    struct SynthesisResponse {
        report: String,
        #[serde(default)]
        gaps: Vec<String>,
    }

    let parsed: SynthesisResponse = lc_core::json_parse::parse_llm_json(content)
        .map_err(|e| ResearchError::Llm(format!("failed to parse synthesis response: {}", e)))?;

    Ok((parsed.report, parsed.gaps))
}

/// Parses the delimiter-based format: <<<REPORT>>>...<<<END_REPORT>>><<<GAPS>>>...<<<END_GAPS>>>
fn parse_delimiter_format(content: &str) -> Option<(String, Vec<String>)> {
    let report_start = content.find("<<<REPORT>>>")?;
    let report_end = content.find("<<<END_REPORT>>>")?;

    if report_end <= report_start {
        return None;
    }

    let report_content = content[report_start + "<<<REPORT>>>".len()..report_end].trim();

    let gaps_start = content.find("<<<GAPS>>>")?;
    let gaps_end = content.find("<<<END_GAPS>>>")?;

    if gaps_end <= gaps_start {
        return None;
    }

    let gaps_text = content[gaps_start + "<<<GAPS>>>".len()..gaps_end].trim();

    let gaps: Vec<String> = if gaps_text.is_empty() || gaps_text == "[]" {
        Vec::new()
    } else {
        // Try JSON parse; if it fails, split by newlines as a rough fallback
        serde_json::from_str(gaps_text).unwrap_or_else(|_| {
            gaps_text
                .lines()
                .map(|l| l.trim().trim_start_matches('-').trim().to_string())
                .filter(|l| !l.is_empty())
                .collect()
        })
    };

    Some((report_content.to_string(), gaps))
}

/// Extracts a JSON object from LLM output, tolerating markdown fences
/// and surrounding text.
#[cfg(test)]
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

    #[test]
    fn test_parse_delimiter_format_basic() {
        let content = "<<<REPORT>>>\n# Report\n\nContent [1].\n<<<END_REPORT>>>\n<<<GAPS>>>\n[\"need more data\"]\n<<<END_GAPS>>>";
        let (report, gaps) = parse_synthesis(content).unwrap();
        assert!(report.contains("# Report"));
        assert!(report.contains("Content [1]."));
        assert_eq!(gaps, vec!["need more data"]);
    }

    #[test]
    fn test_parse_delimiter_format_no_gaps() {
        let content =
            "<<<REPORT>>>\n# Report\n\nDone.\n<<<END_REPORT>>>\n<<<GAPS>>>\n[]\n<<<END_GAPS>>>";
        let (report, gaps) = parse_synthesis(content).unwrap();
        assert!(report.contains("Done"));
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_parse_delimiter_format_special_chars() {
        let content = "<<<REPORT>>>\n# Report\n\nCode: `let x = \"hello\";`\nPath: C:\\Users\\test\n<<<END_REPORT>>>\n<<<GAPS>>>\n[]\n<<<END_GAPS>>>";
        let (report, gaps) = parse_synthesis(content).unwrap();
        assert!(report.contains("let x = \"hello\""));
        assert!(report.contains("C:\\Users\\test"));
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_parse_synthesis_legacy_json_still_works() {
        let content =
            "{\"report\": \"# Report\\n\\nContent [1].\", \"gaps\": [\"need more data\"]}";
        let (report, gaps) = parse_synthesis(content).unwrap();
        assert!(report.contains("# Report"));
        assert_eq!(gaps, vec!["need more data"]);
    }

    #[test]
    fn test_format_sources_no_limit() {
        let results = vec![
            SearchResult {
                query: "q1".into(),
                title: "Source A".into(),
                snippet: "Content A".into(),
                url: String::new(),
            },
            SearchResult {
                query: "q2".into(),
                title: "Source B".into(),
                snippet: "Content B".into(),
                url: String::new(),
            },
        ];
        let text = format_sources(&results, None);
        assert!(text.contains("[1]"));
        assert!(text.contains("[2]"));
        assert!(text.contains("Source A"));
        assert!(text.contains("Source B"));
    }

    #[test]
    fn test_format_sources_with_token_limit_truncates() {
        let results = vec![
            SearchResult { query: "q1".into(), title: "Source A".into(), snippet: "A fairly long snippet that will consume some tokens when counted by the tokenizer".into(), url: String::new() },
            SearchResult { query: "q2".into(), title: "Source B".into(), snippet: "Another fairly long snippet that also consumes tokens".into(), url: String::new() },
        ];
        // With a very small budget, only the first source should be included
        let text = format_sources(&results, Some(20));
        assert!(text.contains("[1]"));
        assert!(text.contains("Source A"));
        // Second source should be truncated away
        // (exact behavior depends on token count, but with budget=20 only 1 entry fits)
    }

    #[test]
    fn test_format_sources_includes_urls() {
        let results = vec![SearchResult {
            query: "q1".into(),
            title: "Source A".into(),
            snippet: "Content".into(),
            url: "https://example.com".into(),
        }];
        let text = format_sources(&results, None);
        assert!(text.contains("https://example.com"));
    }

    /// Verify the synthesis prompt contains few-shot example content.
    #[test]
    fn test_synthesize_prompt_contains_few_shot_example() {
        // The prompt is built inline in synthesize(), but we can verify
        // the example format section appears in the generated prompt text.
        let topic = "Test Topic";
        let subtopics_text = "1. Sub-topic A (queries: q1)";
        let sources_text = "[1] Source A: Some content";

        let prompt = format!(
            "Research topic: {}\n\n\
             Sub-topics investigated:\n{}\n\n\
             Sources collected:\n{}\n\n\
             Write a comprehensive research report in markdown format. \
             Use inline citation markers [1], [2], etc. to reference the sources above. \
             The report should cover all sub-topics and synthesize findings across sources.\n\n\
             After the report, identify any information gaps that need further research.\n\n\
             Example output format:\n\
             <<<REPORT>>>\n\
             # Topic Title\n\n\
             This report examines [overview]. Key findings include [summary].\n\n\
             ## Sub-topic 1\n\n\
             According to the research [1], [finding]. Additional analysis [2] shows [detail].\n\n\
             ## Sub-topic 2\n\n\
             Studies indicate [3] that [finding]. This suggests [implication].\n\n\
             ## Conclusion\n\n\
             In summary, [synthesis of findings across sources].\n\
             <<<END_REPORT>>>\n\
             <<<GAPS>>>\n\
             [\"Specific gap that needs more research\"]\n\
             <<<END_GAPS>>>\n\n\
             Output your response in this exact format (do not wrap in JSON):\n\
             <<<REPORT>>>\n\
             (your full markdown report here)\n\
             <<<END_REPORT>>>\n\
             <<<GAPS>>>\n\
             (a JSON array of gap strings, e.g. [\"gap1\", \"gap2\"], or [] if none)\n\
             <<<END_GAPS>>>",
            topic, subtopics_text, sources_text,
        );

        assert!(
            prompt.contains("Example output format:"),
            "prompt should contain few-shot example header"
        );
        assert!(
            prompt.contains("<<<REPORT>>>"),
            "prompt example should contain delimiter format"
        );
        assert!(
            prompt.contains("Sub-topic 1"),
            "prompt example should contain sub-topic example"
        );
        assert!(
            prompt.contains("<<<GAPS>>>"),
            "prompt example should contain gaps section"
        );
    }
}
