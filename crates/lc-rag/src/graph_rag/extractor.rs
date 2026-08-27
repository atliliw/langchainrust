// src/retrieval/graph_rag/extractor.rs
//! LLM-based entity and relation extraction from document text.
//!
//! Sends a structured prompt to the LLM, parses the JSON response into
//! [`ExtractedEntity`] and [`ExtractedRelation`] lists.

use lc_core::language_models::BaseChatModel;
use lc_core::tools::ToolDefinition;
use lc_schema::Message;
use serde::Deserialize;
use serde_json::json;

use crate::structured::{chat_structured, StructuredChatResult};

/// An entity extracted from text by the LLM.
#[derive(Debug, Clone, Deserialize)]
pub struct ExtractedEntity {
    /// Entity name.
    pub name: String,
    /// Entity type (e.g. Person, Organization, Technology).
    #[serde(rename = "type")]
    pub entity_type: String,
    /// Optional description of the entity.
    #[serde(default)]
    pub description: String,
}

/// A relation extracted from text by the LLM.
#[derive(Debug, Clone, Deserialize)]
pub struct ExtractedRelation {
    /// Source entity name.
    pub source: String,
    /// Target entity name.
    pub target: String,
    /// Relation type (e.g. works_at, uses, part_of).
    #[serde(rename = "type")]
    pub relation_type: String,
    /// Optional description of the relation.
    #[serde(default)]
    pub description: String,
}

/// The full LLM extraction response.
#[derive(Debug, Deserialize)]
pub struct ExtractionResult {
    /// Extracted entities.
    #[serde(default)]
    pub entities: Vec<ExtractedEntity>,
    /// Extracted relations.
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

Example:
Text: "Alice works at Google as a software engineer. She uses Python and TensorFlow."
Output:
{
  "entities": [
    {"name": "Alice", "type": "Person", "description": "A software engineer at Google"},
    {"name": "Google", "type": "Organization", "description": "A technology company"},
    {"name": "Python", "type": "Technology", "description": "A programming language"},
    {"name": "TensorFlow", "type": "Technology", "description": "A machine learning framework"}
  ],
  "relations": [
    {"source": "Alice", "target": "Google", "type": "works_at", "description": "Alice is employed at Google"},
    {"source": "Alice", "target": "Python", "type": "uses", "description": "Alice uses Python"},
    {"source": "Alice", "target": "TensorFlow", "type": "uses", "description": "Alice uses TensorFlow"}
  ]
}

Text:
{text}"#;

/// Entity/relation extraction tool definition (P2-1): forces the LLM to output structured
/// JSON arguments.
fn extraction_tool() -> ToolDefinition {
    ToolDefinition::new(
        "extract_entities_relations",
        "从文本中提取知识图谱实体与关系,返回 entities 与 relations 两个数组",
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "type": { "type": "string" },
                        "description": { "type": "string" }
                    },
                    "required": ["name", "type", "description"]
                }
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": { "type": "string" },
                        "target": { "type": "string" },
                        "type": { "type": "string" },
                        "description": { "type": "string" }
                    },
                    "required": ["source", "target", "type", "description"]
                }
            }
        },
        "required": ["entities", "relations"]
    }))
}

/// Parses one structured call result: prefers tool_calls arguments, then text JSON.
fn parse_structured(result: &StructuredChatResult) -> Option<ExtractionResult> {
    if let Some(args) = &result.tool_args {
        if let Ok(parsed) = serde_json::from_value::<ExtractionResult>(args.clone()) {
            return Some(parsed);
        }
    }
    parse_extraction(&result.content).ok()
}

/// Extracts entities and relations from text using the LLM.
pub async fn extract<M: BaseChatModel>(
    llm: &M,
    text: &str,
    max_entities: usize,
    max_relations: usize,
) -> Result<ExtractionResult, super::GraphRAGError> {
    let prompt = {
        use lc_prompts::PromptTemplate;
        let template = PromptTemplate::new(EXTRACTION_PROMPT);
        let max_entities_str = max_entities.to_string();
        let max_relations_str = max_relations.to_string();
        let mut vars: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        vars.insert("max_entities", &max_entities_str);
        vars.insert("max_relations", &max_relations_str);
        vars.insert("text", text);
        template
            .format(&vars)
            .unwrap_or_else(|_| EXTRACTION_PROMPT.to_string())
    };

    extract_with_retry(llm, text, prompt).await
}

/// P2-1/P2-2: prefers native tool_calls structured output; when text JSON parsing fails,
/// retries 1-2 times with a "return JSON only" hint, and only then returns an error.
async fn extract_with_retry<M: BaseChatModel>(
    llm: &M,
    original_text: &str,
    prompt: String,
) -> Result<ExtractionResult, super::GraphRAGError> {
    const MAX_RETRIES: usize = 2;
    let mut current_prompt = prompt;

    for attempt in 0..=MAX_RETRIES {
        let result = chat_structured(
            llm,
            Some(extraction_tool()),
            vec![Message::human(&current_prompt)],
        )
        .await
        .map_err(|e| super::GraphRAGError::LLMError(e.to_string()))?;

        if let Some(parsed) = parse_structured(&result) {
            return Ok(parsed);
        }

        if attempt < MAX_RETRIES {
            current_prompt = format!(
                "上次的输出不是合法 JSON,无法解析。请重新从下面文本提取实体与关系,\
                 只返回一个 JSON 对象(键为 entities 与 relations),不要包含任何解释、\
                 编号、引号或代码块。\n\n文本:\n{}\n\n上次输出(无效):\n{}\n\n只输出 JSON 对象:",
                original_text, result.content
            );
        }
    }

    Err(super::GraphRAGError::ExtractionError(
        "LLM repeatedly returned invalid JSON; entity/relation extraction failed".to_string(),
    ))
}

/// Parses the LLM JSON response into an `ExtractionResult`.
pub fn parse_extraction(raw: &str) -> Result<ExtractionResult, super::GraphRAGError> {
    lc_core::json_parse::parse_llm_json::<ExtractionResult>(raw).map_err(|e| {
        super::GraphRAGError::ExtractionError(format!("Failed to parse extraction JSON: {}", e))
    })
}

/// Attempts to extract a JSON object from text that may contain markdown fences.
#[cfg(test)]
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

    /// Verify the extraction prompt contains a few-shot example.
    #[test]
    fn test_extraction_prompt_contains_few_shot_example() {
        assert!(
            EXTRACTION_PROMPT.contains("Example:"),
            "extraction prompt should contain few-shot example"
        );
        assert!(
            EXTRACTION_PROMPT.contains("Alice"),
            "extraction prompt example should contain entity 'Alice'"
        );
        assert!(
            EXTRACTION_PROMPT.contains("works_at"),
            "extraction prompt example should contain relation type 'works_at'"
        );
    }

    /// P2-1: the extraction tool definition carries a full JSON Schema (entities + relations).
    #[test]
    fn test_extraction_tool_schema() {
        let tool = extraction_tool();
        assert_eq!(tool.function.name, "extract_entities_relations");
        let params = tool.function.parameters.expect("parameters should exist");
        assert!(params["properties"]["entities"].is_object());
        assert!(params["properties"]["relations"].is_object());
    }

    /// P2-1: tool_calls arguments parse into an ExtractionResult.
    #[test]
    fn test_parse_structured_tool_args() {
        let result = StructuredChatResult {
            content: "".to_string(),
            tool_args: Some(json!({
                "entities": [{"name": "Alice", "type": "Person", "description": "dev"}],
                "relations": []
            })),
        };
        let parsed = parse_structured(&result).expect("tool_args should parse successfully");
        assert_eq!(parsed.entities.len(), 1);
        assert_eq!(parsed.entities[0].name, "Alice");
        assert!(parsed.relations.is_empty());
    }

    /// P2-1: falls back to text JSON parsing when there are no tool_calls.
    #[test]
    fn test_parse_structured_text_fallback() {
        let result = StructuredChatResult {
            content: r#"{"entities": [{"name": "Bob", "type": "Person", "description": "mgr"}], "relations": []}"#.to_string(),
            tool_args: None,
        };
        let parsed = parse_structured(&result).expect("text JSON should parse successfully");
        assert_eq!(parsed.entities[0].name, "Bob");
    }

    /// P2-1: when tool_args deserialization fails and the text is not JSON -> None
    /// (triggering a retry).
    #[test]
    fn test_parse_structured_none() {
        let result = StructuredChatResult {
            content: "not json".to_string(),
            tool_args: Some(json!("not an object")),
        };
        assert!(parse_structured(&result).is_none());
    }
}
