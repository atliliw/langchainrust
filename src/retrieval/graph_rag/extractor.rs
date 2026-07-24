// src/retrieval/graph_rag/extractor.rs
//! LLM-based entity and relation extraction from document text.
//!
//! Sends a structured prompt to the LLM, parses the JSON response into
//! [`ExtractedEntity`] and [`ExtractedRelation`] lists.

use crate::core::language_models::{BaseChatModel, LLMResult};
use crate::schema::Message;
use serde::Deserialize;

/// An entity extracted from text by the LLM.
#[derive(Debug, Clone, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    #[serde(rename = "type")]
    pub entity_type: String,
    #[serde(default)]
    pub description: String,
}

/// A relation extracted from text by the LLM.
#[derive(Debug, Clone, Deserialize)]
pub struct ExtractedRelation {
    pub source: String,
    pub target: String,
    #[serde(rename = "type")]
    pub relation_type: String,
    #[serde(default)]
    pub description: String,
}

/// The full LLM extraction response.
#[derive(Debug, Deserialize)]
pub struct ExtractionResult {
    #[serde(default)]
    pub entities: Vec<ExtractedEntity>,
    #[serde(default)]
    pub relations: Vec<ExtractedRelation>,
}

const EXTRACTION_PROMPT: &str = r#"You are a knowledge graph extraction assistant. Given the following text, extract entities and their relations.

Return a JSON object with exactly two keys:
- "entities": an array of objects, each with keys "name", "type", "description"
- "relations": an array of objects, each with keys "source", "target", "type", "description"

Rules:
- "source" and "target" in relations must match entity "name" values exactly.
- Keep entity types simple: Person, Organization, Location, Technology, Concept, Event, etc.
- Keep relation types simple: works_at, located_in, uses, created, part_of, related_to, etc.
- Extract at most {max_entities} entities and {max_relations} relations.
- Return ONLY the JSON object, no other text.

Text:
{text}"#;

/// Extracts entities and relations from text using the LLM.
pub async fn extract<M: BaseChatModel>(
    llm: &M,
    text: &str,
    max_entities: usize,
    max_relations: usize,
) -> Result<ExtractionResult, super::GraphRAGError> {
    let prompt = EXTRACTION_PROMPT
        .replace("{max_entities}", &max_entities.to_string())
        .replace("{max_relations}", &max_relations.to_string())
        .replace("{text}", text);

    let messages = vec![Message::human(prompt)];

    let response: LLMResult = llm
        .chat(messages, None)
        .await
        .map_err(|e| super::GraphRAGError::LLMError(e.to_string()))?;

    parse_extraction(&response.content)
}

/// Parses the LLM JSON response into an `ExtractionResult`.
pub fn parse_extraction(raw: &str) -> Result<ExtractionResult, super::GraphRAGError> {
    // Try to extract JSON from the response (may be wrapped in markdown code block)
    let json_str = extract_json(raw);

    serde_json::from_str::<ExtractionResult>(&json_str).map_err(|e| {
        super::GraphRAGError::ExtractionError(format!(
            "Failed to parse extraction JSON: {}. Raw: {}",
            e,
            &raw[..raw.len().min(200)]
        ))
    })
}

/// Attempts to extract a JSON object from text that may contain markdown fences.
fn extract_json(text: &str) -> String {
    let trimmed = text.trim();

    // Case 1: wrapped in ```json ... ```
    if let Some(rest) = trimmed.strip_prefix("```json") {
        if let Some(end) = rest.find("```") {
            return rest[..end].trim().to_string();
        }
    }

    // Case 2: wrapped in ``` ... ```
    if let Some(rest) = trimmed.strip_prefix("```") {
        if let Some(end) = rest.find("```") {
            return rest[..end].trim().to_string();
        }
    }

    // Case 3: find first { ... last }
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            if end > start {
                return trimmed[start..=end].to_string();
            }
        }
    }

    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_extraction_valid() {
        let raw = r#"{"entities":[{"name":"Alice","type":"Person","description":"A developer"}],"relations":[{"source":"Alice","target":"Rust","type":"uses","description":"Alice uses Rust"}]}"#;
        let result = parse_extraction(raw).unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].name, "Alice");
        assert_eq!(result.relations.len(), 1);
        assert_eq!(result.relations[0].source, "Alice");
    }

    #[test]
    fn test_parse_extraction_markdown_wrapped() {
        let raw = r#"```json
{"entities":[{"name":"Bob","type":"Person","description":"A manager"}],"relations":[]}
```"#;
        let result = parse_extraction(raw).unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].name, "Bob");
    }

    #[test]
    fn test_parse_extraction_empty_arrays() {
        let raw = r#"{"entities":[],"relations":[]}"#;
        let result = parse_extraction(raw).unwrap();
        assert!(result.entities.is_empty());
        assert!(result.relations.is_empty());
    }

    #[test]
    fn test_parse_extraction_invalid() {
        let raw = "not json at all";
        assert!(parse_extraction(raw).is_err());
    }

    #[test]
    fn test_extract_json_plain() {
        let input = r#"{"key": "value"}"#;
        assert_eq!(extract_json(input), input);
    }

    #[test]
    fn test_extract_json_code_fence() {
        let input = "```json\n{\"key\": \"value\"}\n```";
        assert_eq!(extract_json(input), "{\"key\": \"value\"}");
    }

    #[test]
    fn test_extract_json_with_surrounding_text() {
        let input = "Here is the result:\n{\"key\": \"value\"}\nDone.";
        assert_eq!(extract_json(input), "{\"key\": \"value\"}");
    }
}
