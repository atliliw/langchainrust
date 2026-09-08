// lc-providers/src/openai/response_format.rs
//! OpenAI `response_format` types (0.21.0 S3.1).
//!
//! 2026 structured-output alignment: engine-side constraint via
//! `response_format: { type: "json_schema", json_schema: { name, schema, strict } }`
//! guarantees schema-valid JSON without client-side parsing retries. The local
//! `PartialJsonParser` remains the fallback path for providers that do not
//! support it (or for streaming partial output).

use serde::{Deserialize, Serialize};

/// OpenAI `response_format` request field.
///
/// Serialized with an internal `type` tag (`text` / `json_object` / `json_schema`),
/// matching the OpenAI chat-completions request schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Plain text (default).
    Text,
    /// JSON mode: any valid JSON, no schema constraint.
    JsonObject,
    /// Schema-constrained JSON: the engine enforces the schema (strict mode).
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// The schema specification.
        json_schema: JsonSchemaSpec,
    },
}

/// The `json_schema` payload of [`ResponseFormat::JsonSchema`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonSchemaSpec {
    /// Schema name (1-64 chars, `^[a-zA-Z0-9_-]+$`).
    pub name: String,
    /// The JSON Schema the output must conform to.
    pub schema: serde_json::Value,
    /// OpenAI strict mode: guarantees the output conforms to the schema.
    /// Strict mode requires `additionalProperties: false` and all properties
    /// listed in `required` — use [`make_strict_schema`] to normalize.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl ResponseFormat {
    /// Convenience constructor: strict json_schema response format.
    pub fn json_schema(name: impl Into<String>, schema: serde_json::Value) -> Self {
        ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: name.into(),
                schema,
                strict: Some(true),
            },
        }
    }
}

/// Normalizes a JSON Schema in place for OpenAI strict mode.
///
/// For every object level that has `properties`:
/// - sets `additionalProperties: false` (OpenAI strict requires it);
/// - sets `required` to ALL property names (strict mode has no optional
///   fields; model-side "optional" is emulated by allowing `null` via union).
///
/// Idempotent: re-normalizing an already-strict schema is a no-op. Recurses
/// into `properties`, `items`, and `anyOf`/`oneOf`/`allOf` branches.
pub fn make_strict_schema(schema: &mut serde_json::Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };

    if obj.contains_key("properties") {
        obj.insert(
            "additionalProperties".to_string(),
            serde_json::Value::Bool(false),
        );
        let mut required: Vec<serde_json::Value> = Vec::new();
        if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
            for name in props.keys() {
                required.push(serde_json::Value::String(name.clone()));
            }
            for value in props.values_mut() {
                make_strict_schema(value);
            }
        }
        obj.insert("required".to_string(), serde_json::Value::Array(required));
    }

    for key in ["items", "additionalProperties", "not"] {
        if let Some(value) = obj.get_mut(key) {
            if value.is_object() {
                make_strict_schema(value);
            }
        }
    }
    for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
        if let Some(list) = obj.get_mut(key).and_then(|l| l.as_array_mut()) {
            for value in list.iter_mut() {
                make_strict_schema(value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_response_format_serialization_text() {
        let value = serde_json::to_value(ResponseFormat::Text).unwrap();
        assert_eq!(value, json!({"type": "text"}));
    }

    #[test]
    fn test_response_format_serialization_json_object() {
        let value = serde_json::to_value(ResponseFormat::JsonObject).unwrap();
        assert_eq!(value, json!({"type": "json_object"}));
    }

    /// 0.21.0 S3.1: the request body shape matches the OpenAI chat-completions
    /// `response_format` schema (`type` tag + nested `json_schema` object).
    #[test]
    fn test_response_format_serialization_json_schema() {
        let format = ResponseFormat::json_schema("person", json!({"type": "object"}));
        let value = serde_json::to_value(&format).unwrap();
        assert_eq!(
            value,
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {"type": "object"},
                    "strict": true
                }
            })
        );
    }

    /// Strict normalization: `additionalProperties: false` + all properties required.
    #[test]
    fn test_make_strict_schema_top_level() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"],
            "additionalProperties": true
        });
        make_strict_schema(&mut schema);
        assert_eq!(schema["additionalProperties"], json!(false));
        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(required, vec!["age", "name"]);
    }

    /// Strict normalization recurses into nested objects and arrays.
    #[test]
    fn test_make_strict_schema_recursive() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}}
                    }
                }
            }
        });
        make_strict_schema(&mut schema);
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(schema["required"], json!(["tags"]));
        let items = &schema["properties"]["tags"]["items"];
        assert_eq!(items["additionalProperties"], json!(false));
        assert_eq!(items["required"], json!(["label"]));
    }

    /// Strict normalization recurses into anyOf/oneOf branches.
    #[test]
    fn test_make_strict_schema_variants() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "object", "properties": {"a": {"type": "string"}}},
                        {"type": "object", "properties": {"b": {"type": "integer"}}}
                    ]
                }
            }
        });
        make_strict_schema(&mut schema);
        let branches = &schema["properties"]["value"]["anyOf"];
        assert_eq!(branches[0]["additionalProperties"], json!(false));
        assert_eq!(branches[1]["additionalProperties"], json!(false));
    }

    /// Non-object schemas (string, enum) pass through untouched.
    #[test]
    fn test_make_strict_schema_non_object() {
        let mut schema = json!({"type": "string", "enum": ["a", "b"]});
        make_strict_schema(&mut schema);
        assert_eq!(schema, json!({"type": "string", "enum": ["a", "b"]}));

        let mut scalar = json!("just a string");
        make_strict_schema(&mut scalar);
        assert_eq!(scalar, json!("just a string"));
    }

    /// Idempotency: normalizing an already-strict schema changes nothing.
    #[test]
    fn test_make_strict_schema_idempotent() {
        let mut schema = json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false
        });
        make_strict_schema(&mut schema);
        let once = schema.clone();
        make_strict_schema(&mut schema);
        assert_eq!(schema, once);
    }
}
