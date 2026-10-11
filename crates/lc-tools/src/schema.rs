// lc-tools/src/schema.rs
//! OpenAI/Anthropic `tools[].function.parameters` direct-export helpers.
//!
//! [`BaseTool::args_schema`] returns a schemars **draft-07** schema that carries
//! bookkeeping (`$schema`, `title`) and may be wrapped in a top-level `$ref`. Chat
//! APIs (OpenAI function calling, Anthropic `tools[]`) expect a lean
//! `{ "type": "object", "properties": …, "required": […] }` payload and some
//! reject unknown sibling keys or bare `$ref` wrappers. These helpers normalize a
//! tool's schema into exactly that shape — complementing the `#[tool]` macro's
//! inherent `parameters()` (raw `schema_for!`) with a provider-ready export that
//! also works for hand-written [`BaseTool`] implementations.

use lc_core::tools::BaseTool;

/// Converts a schemars draft-07 schema into a minimal `function.parameters` object.
///
/// - Drops the `$schema` / `title` bookkeeping keys.
/// - Resolves a lone top-level `{"$ref": "#/definitions/<Name>"}` into the pointed
///   definition, so a wrapped argument type still yields a usable `properties` map.
/// - Drops the `definitions` map once nothing in the body references it, keeping
///   the payload lean.
pub fn parameters_from_schema(schema: &serde_json::Value) -> serde_json::Value {
    let mut body = resolve_top_level_ref(schema);
    // Compute before borrowing `body` mutably: dropping `$schema`/`title` can never add a
    // `$ref`, so whether inline definitions are still referenced is a stable fact here.
    let has_refs = references_definitions(&body);

    if let serde_json::Value::Object(obj) = &mut body {
        obj.remove("$schema");
        obj.remove("title");
        // Safe to drop inline definitions only when no `$ref` still references them.
        if !has_refs {
            obj.remove("definitions");
        }
    }

    body
}

/// Resolves a lone top-level `{"$ref": "#/definitions/<Name>"}` wrapper (the usual
/// shape `schemars` emits for a renamed / newtype / enum argument) into the
/// referenced definition, so the caller receives a real `{type, properties|…}` body.
fn resolve_top_level_ref(schema: &serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(obj) = schema else {
        return schema.clone();
    };
    // A `{"$ref": "#/definitions/<Name>", "definitions": {…}}` wrapper (what schemars
    // emits for a renamed / newtype / enum argument) always pairs the pointer with its
    // definitions. Match on the pointer alone — single-entry and multi-entry both unwrap.
    let Some(serde_json::Value::String(pointer)) = obj.get("$ref") else {
        return schema.clone();
    };
    let Some(name) = pointer.strip_prefix("#/definitions/") else {
        return schema.clone();
    };
    schema
        .get("definitions")
        .and_then(|d| d.get(name))
        .cloned()
        .unwrap_or_else(|| schema.clone())
}

/// Whether the (possibly-resolved) schema body still contains a `$ref`; if so the
/// `definitions` block must be preserved for the payload to stay self-contained.
fn references_definitions(schema: &serde_json::Value) -> bool {
    schema.to_string().contains("\"$ref\"")
}

/// Direct-export interface: returns the tool's `parameters` in OpenAI/Anthropic
/// `tools[].function.parameters` form, or `None` if the tool exposes no argument
/// schema.
///
/// ```
/// use lc_tools::schema::tool_parameters;
/// use lc_tools::DateTimeTool;
///
/// let tool = DateTimeTool::new();
/// let parameters = tool_parameters(&tool).expect("datetime has an args schema");
/// assert_eq!(parameters["type"], "object");
/// ```
pub fn tool_parameters(tool: &dyn BaseTool) -> Option<serde_json::Value> {
    tool.args_schema()
        .map(|schema| parameters_from_schema(&schema))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use lc_core::tools::ToolError;
    use schemars::{schema_for, JsonSchema};
    use serde::Deserialize;
    use serde_json::json;

    // Fields are only exercised indirectly via the generated JSON schema, so the struct is
    // never (de)constructed in tests; silence the dead_code lint that would trip `-D warnings`.
    #[allow(dead_code)]
    #[derive(Debug, Deserialize, JsonSchema)]
    struct Args {
        /// Number of items.
        n: u32,
        optional_tag: Option<String>,
    }

    struct StubTool;
    #[async_trait]
    impl BaseTool for StubTool {
        fn name(&self) -> &str {
            "stub"
        }
        fn description(&self) -> &str {
            "stub tool"
        }
        async fn run(&self, input: String) -> Result<String, ToolError> {
            Ok(input)
        }
        fn args_schema(&self) -> Option<serde_json::Value> {
            serde_json::to_value(schema_for!(Args)).ok()
        }
    }

    #[test]
    fn normalizes_object_schema() {
        let schema = serde_json::to_value(schema_for!(Args)).unwrap();
        let params = parameters_from_schema(&schema);
        assert_eq!(params["type"], "object");
        assert!(params.get("properties").is_some(), "{params}");
        assert!(params.get("$schema").is_none(), "{params}");
        assert!(params.get("title").is_none(), "{params}");
        // Non-referenced definitions block is dropped for a lean payload.
        assert!(params.get("definitions").is_none(), "{params}");
    }

    #[test]
    fn resolves_top_level_ref_wrapper() {
        // A lone {"$ref": "#/definitions/Args"} is what a renamed type yields; the
        // normalizer must unwrap it into a usable object body, not return a bare $ref.
        let schema = json!({
            "$ref": "#/definitions/Args",
            "definitions": {
                "Args": {
                    "type": "object",
                    "properties": { "n": { "type": "integer" } }
                }
            }
        });
        let params = parameters_from_schema(&schema);
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["n"]["type"].is_string(), "{params}");
    }

    #[test]
    fn tool_parameters_uses_args_schema() {
        let params = tool_parameters(&StubTool).unwrap();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["n"].is_object(), "{params}");
    }
}
