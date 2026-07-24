// src/agents/deep_research/planner.rs
//! Planner - decomposes a research topic into sub-topics and generates
//! search queries for each sub-topic using the LLM.

use crate::core::language_models::BaseChatModel;
use crate::schema::Message;

use super::ResearchError;

/// A sub-topic with its associated search queries.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SubTopic {
    /// Short name for the sub-topic.
    pub name: String,
    /// Search queries to investigate this sub-topic.
    pub queries: Vec<String>,
}

/// A research plan containing decomposed sub-topics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResearchPlan {
    /// The original research topic.
    pub topic: String,
    /// Decomposed sub-topics with their queries.
    pub subtopics: Vec<SubTopic>,
}

impl ResearchPlan {
    /// Collects all search queries across all sub-topics.
    pub fn all_queries(&self) -> Vec<String> {
        self.subtopics
            .iter()
            .flat_map(|st| st.queries.clone())
            .collect()
    }
}

/// Uses the LLM to decompose a topic into sub-topics with search queries.
pub async fn plan<M: BaseChatModel>(
    llm: &M,
    topic: &str,
    max_subtopics: usize,
) -> Result<ResearchPlan, ResearchError> {
    let prompt = format!(
        "Decompose the following research topic into at most {} sub-topics. \
         For each sub-topic, generate 1-3 specific search queries.\n\n\
         Topic: {}\n\n\
         Output a JSON array of objects with \"name\" and \"queries\" fields. \
         Example: [{{\"name\": \"Sub-topic 1\", \"queries\": [\"query1\", \"query2\"]}}]\n\
         Only output the JSON array, nothing else.",
        max_subtopics, topic,
    );
    let messages = vec![
        Message::system("You are a research planning assistant. Output only valid JSON."),
        Message::human(prompt),
    ];

    let response = llm
        .chat(messages, None)
        .await
        .map_err(|e| ResearchError::Llm(format!("{:?}", e)))?;

    let subtopics = parse_subtopics(&response.content)?;
    Ok(ResearchPlan {
        topic: topic.to_string(),
        subtopics,
    })
}

/// Parses the LLM output into a list of `SubTopic` structs.
///
/// Tolerates markdown code fences and surrounding text.
fn parse_subtopics(content: &str) -> Result<Vec<SubTopic>, ResearchError> {
    let json_str = extract_json(content);
    let subtopics: Vec<SubTopic> = serde_json::from_str(&json_str).map_err(|e| {
        let preview: String = content.chars().take(200).collect();
        ResearchError::Llm(format!(
            "failed to parse sub-topics: {} | raw: {}",
            e, preview
        ))
    })?;
    Ok(subtopics)
}

/// Parses a JSON array of strings from LLM output.
///
/// Used for follow-up query generation.
pub fn parse_json_array(content: &str) -> Result<Vec<String>, ResearchError> {
    let json_str = extract_json(content);
    serde_json::from_str(&json_str).map_err(|e| {
        let preview: String = content.chars().take(200).collect();
        ResearchError::Llm(format!(
            "failed to parse JSON array: {} | raw: {}",
            e, preview
        ))
    })
}

/// Extracts JSON from LLM output, tolerating markdown code fences
/// and surrounding text.
fn extract_json(content: &str) -> String {
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

    // Find the first [ or { bracket, whichever comes first
    let start_bracket = stripped.find(['[', '{']);

    if let Some(start) = start_bracket {
        let open_char = stripped.as_bytes()[start];
        let close_char = if open_char == b'[' { b']' } else { b'}' };

        // Find the matching closing bracket
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
            if ch == open_char {
                depth += 1;
            } else if ch == close_char {
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
    fn test_extract_json_plain_array() {
        let input = r#"[{"name": "A", "queries": ["q1"]}]"#;
        assert_eq!(extract_json(input), input);
    }

    #[test]
    fn test_extract_json_markdown_fenced() {
        let input = "```json\n[{\"name\": \"A\", \"queries\": [\"q1\"]}]\n```";
        let expected = r#"[{"name": "A", "queries": ["q1"]}]"#;
        assert_eq!(extract_json(input), expected);
    }

    #[test]
    fn test_extract_json_with_surrounding_text() {
        let input = r#"Here is the plan: [{"name": "A", "queries": ["q1"]}] Done."#;
        let expected = r#"[{"name": "A", "queries": ["q1"]}]"#;
        assert_eq!(extract_json(input), expected);
    }

    #[test]
    fn test_extract_json_nested_brackets() {
        let input = r#"[{"name": "A", "queries": ["q1", "q2"]}, {"name": "B", "queries": ["q3"]}]"#;
        assert_eq!(extract_json(input), input);
    }

    #[test]
    fn test_extract_json_object() {
        let input = r#"Result: {"report": "text", "gaps": []} end"#;
        let expected = r#"{"report": "text", "gaps": []}"#;
        assert_eq!(extract_json(input), expected);
    }

    #[test]
    fn test_parse_subtopics_valid() {
        let content = r#"[{"name": "AI Ethics", "queries": ["AI ethics policy"]}]"#;
        let subtopics = parse_subtopics(content).unwrap();
        assert_eq!(subtopics.len(), 1);
        assert_eq!(subtopics[0].name, "AI Ethics");
        assert_eq!(subtopics[0].queries, vec!["AI ethics policy"]);
    }

    #[test]
    fn test_parse_subtopics_invalid() {
        let content = "not json at all";
        let result = parse_subtopics(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_array_valid() {
        let content = r#"["query1", "query2"]"#;
        let queries = parse_json_array(content).unwrap();
        assert_eq!(queries, vec!["query1", "query2"]);
    }

    #[test]
    fn test_parse_json_array_markdown() {
        let content = "```json\n[\"q1\", \"q2\"]\n```";
        let queries = parse_json_array(content).unwrap();
        assert_eq!(queries, vec!["q1", "q2"]);
    }

    #[test]
    fn test_research_plan_all_queries() {
        let plan = ResearchPlan {
            topic: "test".to_string(),
            subtopics: vec![
                SubTopic {
                    name: "A".to_string(),
                    queries: vec!["q1".to_string(), "q2".to_string()],
                },
                SubTopic {
                    name: "B".to_string(),
                    queries: vec!["q3".to_string()],
                },
            ],
        };
        assert_eq!(plan.all_queries(), vec!["q1", "q2", "q3"]);
    }

    #[test]
    fn test_extract_json_with_brackets_in_strings() {
        let input = r#"[{"name": "A [1]", "queries": ["q1"]}]"#;
        assert_eq!(extract_json(input), input);
    }
}
