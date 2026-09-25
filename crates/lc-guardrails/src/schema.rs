//! Schema-bound output guardrail.
//!
//! [`SchemaOutputGuardrail`] binds an LLM output to a JSON Schema (the same schema
//! `schemars::schema_for!` / `StructuredOutput<T>` uses): non-JSON output, type mismatches,
//! missing required keys and (in strict mode) undeclared properties are blocked fail-fast
//! instead of being handed to downstream parsing.
//!
//! The same type also implements [`StreamingOutputGuardrail`]: on each streaming token it
//! parses the *accumulated* raw output with the lenient, repair-capable
//! [`lc_core::structured_output::PartialJsonParser`], and only blocks on contradictions that
//! are already definitive (wrong type, closed enum... handled strictly at the terminal check,
//! unknown property). Missing required keys are tolerated while the document is still being
//! written; the terminal [`OutputGuardrail`] re-check performs the strict validation.

use async_trait::async_trait;
use lc_core::structured_output::PartialJsonParser;
use schemars::JsonSchema;
use serde_json::Value;

use crate::guardrail::{
    ChunkAction, ChunkContext, OutputGuardrail, OutputGuardrailResult, StreamingOutputGuardrail,
};

/// Output guardrail that validates model output against a JSON Schema.
///
/// Stateless and freely shareable: one instance can be registered as both an output rail
/// ([`OutputGuardrail`], strict terminal validation) and a streaming rail
/// ([`StreamingOutputGuardrail`], lenient incremental validation).
///
/// # Example
/// ```
/// use lc_guardrails::{GuardrailsConfig, SchemaOutputGuardrail};
/// use schemars::JsonSchema;
/// use serde::Deserialize;
/// use std::sync::Arc;
///
/// #[derive(Deserialize, JsonSchema)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// // terminal validation bound to the structured-output schema
/// let rail = SchemaOutputGuardrail::for_type::<Person>();
/// let config = GuardrailsConfig::new()
///     .with_output(Arc::new(rail.clone()))
///     .with_streaming(Arc::new(rail));
/// ```
#[derive(Debug, Clone)]
pub struct SchemaOutputGuardrail {
    /// Root JSON Schema.
    schema: Value,
    /// When true, properties not declared by the schema are rejected even if the schema does
    /// not carry `additionalProperties: false` (the schemars default omits the keyword).
    strict_properties: bool,
}

impl SchemaOutputGuardrail {
    /// Builds a rail from an already-built JSON Schema (as `serde_json::Value`, e.g.
    /// the value returned by `StructuredOutput::<T>::schema()`).
    ///
    /// Strict property checking is on by default; use
    /// [`allow_additional_properties`](Self::allow_additional_properties) to opt out.
    pub fn new(schema: Value) -> Self {
        Self {
            schema,
            strict_properties: true,
        }
    }

    /// Builds a rail from the schema that `schemars` derives for `T`.
    ///
    /// Panics if `schemars` produces a schema this version of `serde_json` cannot serialize.
    /// A failing serialization must not silently degrade into a `Null` schema — a null schema
    /// validates *every* value, silently turning the output rail into a no-op (fail-open).
    /// The schema is derived locally and deterministically, so serialization failure is a
    /// framework-invariant violation, not a runtime condition worth degrading on.
    pub fn for_type<T: JsonSchema>() -> Self {
        let schema = serde_json::to_value(schemars::schema_for!(T)).unwrap_or_else(|e| {
            panic!(
                "SchemaOutputGuardrail::for_type: schemars schema for `{}` could not be \
                 serialized ({}); refusing to degrade into a permissive Null schema",
                std::any::type_name::<T>(),
                e
            )
        });
        Self::new(schema)
    }

    /// Opt out of rejecting undeclared properties when the schema itself does not set
    /// `additionalProperties`.
    pub fn allow_additional_properties(mut self) -> Self {
        self.strict_properties = false;
        self
    }

    /// Strict terminal validation of a complete output (fenced JSON and leading/trailing
    /// prose tolerated). Returns `Err(reason)` on the first schema violation.
    pub fn validate_complete(&self, output: &str) -> Result<(), String> {
        let json = extract_json_slice(output);
        let value: Value =
            serde_json::from_str(json).map_err(|e| format!("output is not valid JSON: {e}"))?;
        validate(
            &self.schema,
            &self.schema,
            &value,
            CheckMode::terminal(self.strict_properties),
            "$",
        )
    }

    /// Lenient incremental validation of a partially-streamed output.
    ///
    /// Returns `Ok` while the document is still forming; only definitive contradictions
    /// (wrong value type, undeclared property) block. Unparseable / still-incomplete output
    /// always passes — the terminal check is the strict gate.
    fn validate_partial(&self, full: &str) -> Result<(), String> {
        let mut parser = PartialJsonParser::new();
        match parser.push_and_parse(full) {
            Ok(value) => validate(
                &self.schema,
                &self.schema,
                &value,
                CheckMode::partial(self.strict_properties),
                "$",
            ),
            // Incomplete (preamble, unclosed structure) is expected mid-stream.
            Err(_) => Ok(()),
        }
    }
}

#[async_trait]
impl OutputGuardrail for SchemaOutputGuardrail {
    fn name(&self) -> &str {
        "SchemaOutputGuardrail"
    }

    async fn validate(&self, output: &str) -> OutputGuardrailResult {
        match self.validate_complete(output) {
            Ok(()) => OutputGuardrailResult::Pass,
            Err(reason) => OutputGuardrailResult::Block { reason },
        }
    }
}

#[async_trait]
impl StreamingOutputGuardrail for SchemaOutputGuardrail {
    fn name(&self) -> &str {
        "SchemaOutputGuardrail"
    }

    async fn validate_chunk(&self, ctx: &ChunkContext<'_>) -> ChunkAction {
        match self.validate_partial(ctx.full) {
            Ok(()) => ChunkAction::Pass,
            // the precise reason is surfaced by the terminal check if the model keeps going;
            // ChunkAction::Block intentionally carries no per-rail payload.
            Err(_) => ChunkAction::Block,
        }
    }
}

/// Strict/lenient knobs for one validation pass.
#[derive(Debug, Clone, Copy)]
struct CheckMode {
    /// Terminal (full-output) validation: enforce `required`, `enum`, ranges.
    terminal: bool,
    /// Reject undeclared properties when the schema omits `additionalProperties`.
    strict_properties: bool,
}

impl CheckMode {
    fn terminal(strict_properties: bool) -> Self {
        Self {
            terminal: true,
            strict_properties,
        }
    }

    fn partial(strict_properties: bool) -> Self {
        Self {
            terminal: false,
            strict_properties,
        }
    }
}

/// Validates `value` against `schema`, resolving `$ref`s against `root`.
///
/// Covers the JSON-Schema subset that `schemars` 1.x emits for ordinary Rust types:
/// type (string/array forms, integer), properties/required/additionalProperties, items,
/// enum/const, oneOf/anyOf/allOf, `$ref` + `$defs`, and the common length/range bounds.
fn validate(
    root: &Value,
    schema: &Value,
    value: &Value,
    mode: CheckMode,
    path: &str,
) -> Result<(), String> {
    // resolve $ref (schemars may inline, but enum/newtype variants still emit refs).
    let schema = resolve_ref(root, schema);

    if let Some(expected) = schema.get("const") {
        if mode.terminal && expected != value {
            return Err(format!("{path}: expected const {expected}, got {value}"));
        }
    }

    if let Some(cases) = schema.get("enum").and_then(Value::as_array) {
        // a half-written value mid-stream must not be rejected against a closed set;
        // the terminal check enforces it.
        if mode.terminal && !cases.iter().any(|c| c == value) {
            return Err(format!("{path}: value {value} is not one of {cases:?}"));
        }
    }

    if let Some(subs) = schema.get("allOf").and_then(Value::as_array) {
        for sub in subs {
            validate(root, sub, value, mode, path)?;
        }
    }

    if let Some(subs) = schema.get("anyOf").and_then(Value::as_array) {
        let any = subs
            .iter()
            .any(|sub| validate(root, sub, value, mode, path).is_ok());
        if mode.terminal && !any {
            return Err(format!(
                "{path}: value matches none of anyOf({} subschemas)",
                subs.len()
            ));
        }
    }

    if let Some(subs) = schema.get("oneOf").and_then(Value::as_array) {
        let matches = subs
            .iter()
            .filter(|sub| validate(root, sub, value, mode, path).is_ok())
            .count();
        if mode.terminal && matches != 1 {
            return Err(format!(
                "{path}: value matches {matches} of oneOf({} subschemas), expected exactly 1",
                subs.len()
            ));
        }
        // Partial mode is deliberately lenient here: a still-forming document that has not yet
        // satisfied any oneOf branch may later converge (a floating point literal, a partial
        // string, an omitted-but-imminent optional wrapper). This matches how `enum`/`const`/
        // `required` are left for the terminal check — blocking on `matches == 0` mid-stream
        // would drop a valid in-progress document. Terminal validation remains the strict gate.
    }

    if let Some(ty) = schema.get("type") {
        check_type(root, schema, ty, value, mode, path)?;
    }

    Ok(())
}

/// Type-keyword dispatch; object/array carry their own nested validation.
fn check_type(
    root: &Value,
    schema: &Value,
    ty: &Value,
    value: &Value,
    mode: CheckMode,
    path: &str,
) -> Result<(), String> {
    let type_allows = |name: &str| match ty {
        Value::String(s) => s == name,
        Value::Array(arr) => arr.iter().any(|t| t.as_str() == Some(name)),
        _ => false,
    };

    let actual = json_type(value);
    let allowed = match actual {
        "null" => type_allows("null"),
        "boolean" => type_allows("boolean"),
        "string" => type_allows("string"),
        "object" => type_allows("object"),
        "array" => type_allows("array"),
        // JSON Schema: an integer value also satisfies "number".
        "integer" => type_allows("integer") || type_allows("number"),
        "number" => type_allows("number"),
        other => unreachable!("json_type returned {other}"),
    };
    if !allowed {
        return Err(format!("{path}: expected type {ty}, got {actual}"));
    }

    match value {
        Value::Object(map) => validate_object(root, schema, map, mode, path),
        Value::Array(items) => validate_array(root, schema, items, mode, path),
        Value::String(s) => validate_string(schema, s, mode, path),
        Value::Number(n) => validate_number(schema, n, mode, path),
        _ => Ok(()),
    }
}

fn validate_object(
    root: &Value,
    schema: &Value,
    map: &serde_json::Map<String, Value>,
    mode: CheckMode,
    path: &str,
) -> Result<(), String> {
    if mode.terminal {
        if let Some(min) = schema.get("minProperties").and_then(Value::as_u64) {
            if (map.len() as u64) < min {
                return Err(format!(
                    "{path}: expected at least {min} properties, got {}",
                    map.len()
                ));
            }
        }
        if let Some(max) = schema.get("maxProperties").and_then(Value::as_u64) {
            if map.len() as u64 > max {
                return Err(format!(
                    "{path}: expected at most {max} properties, got {}",
                    map.len()
                ));
            }
        }
    }

    let properties = schema.get("properties").and_then(Value::as_object);
    let additional = schema.get("additionalProperties");
    let deny_extra = match additional {
        Some(Value::Bool(false)) => true,
        Some(_) => false,
        None => mode.strict_properties,
    };

    for (key, sub_value) in map {
        let child_path = format!("{path}.{key}");
        match properties.and_then(|p| p.get(key)) {
            Some(sub) => validate(root, sub, sub_value, mode, &child_path)?,
            None => {
                if deny_extra {
                    return Err(format!("{child_path}: property not declared in schema"));
                }
                // `additionalProperties: {schema}` applies to every undeclared property.
                if let Some(sub @ Value::Object(_)) = additional {
                    validate(root, sub, sub_value, mode, &child_path)?;
                }
            }
        }
    }

    if mode.terminal {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for key in required {
                if let Some(name) = key.as_str() {
                    if !map.contains_key(name) {
                        return Err(format!("{path}: missing required property `{name}`"));
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_array(
    root: &Value,
    schema: &Value,
    items: &[Value],
    mode: CheckMode,
    path: &str,
) -> Result<(), String> {
    if mode.terminal {
        if let Some(min) = schema.get("minItems").and_then(Value::as_u64) {
            if (items.len() as u64) < min {
                return Err(format!(
                    "{path}: expected at least {min} items, got {}",
                    items.len()
                ));
            }
        }
        if let Some(max) = schema.get("maxItems").and_then(Value::as_u64) {
            if items.len() as u64 > max {
                return Err(format!(
                    "{path}: expected at most {max} items, got {}",
                    items.len()
                ));
            }
        }
    }
    if let Some(item_schema) = schema.get("items") {
        for (i, item) in items.iter().enumerate() {
            validate(root, item_schema, item, mode, &format!("{path}[{i}]"))?;
        }
    }
    Ok(())
}

fn validate_string(schema: &Value, s: &str, mode: CheckMode, path: &str) -> Result<(), String> {
    if !mode.terminal {
        // length constraints are meaningless while the string is still being streamed.
        return Ok(());
    }
    if let Some(min) = schema.get("minLength").and_then(Value::as_u64) {
        if (s.chars().count() as u64) < min {
            return Err(format!("{path}: string shorter than minLength {min}"));
        }
    }
    if let Some(max) = schema.get("maxLength").and_then(Value::as_u64) {
        if s.chars().count() as u64 > max {
            return Err(format!("{path}: string longer than maxLength {max}"));
        }
    }
    Ok(())
}

fn validate_number(
    schema: &Value,
    n: &serde_json::Number,
    mode: CheckMode,
    path: &str,
) -> Result<(), String> {
    if !mode.terminal {
        // "3" can still become "30"; ranges are enforced only at the terminal check.
        return Ok(());
    }
    let as_f64 = || n.as_f64().unwrap_or(f64::NAN);
    if let Some(min) = schema.get("minimum").and_then(Value::as_f64) {
        if as_f64() < min {
            return Err(format!("{path}: value below minimum {min}"));
        }
    }
    if let Some(max) = schema.get("maximum").and_then(Value::as_f64) {
        if as_f64() > max {
            return Err(format!("{path}: value above maximum {max}"));
        }
    }
    Ok(())
}

/// JSON-Schema type name of a value (numbers split into integer/number).
fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n)
            if n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0) =>
        {
            "integer"
        }
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Resolves a local `$ref` ("#/$defs/Name" / "#/definitions/Name") against the root schema.
/// Schemas without a ref are returned unchanged.
fn resolve_ref<'a>(root: &'a Value, schema: &'a Value) -> &'a Value {
    let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
        return schema;
    };
    let Some(rest) = reference.strip_prefix("#/") else {
        return schema;
    };
    let mut target = root;
    for segment in rest.split('/') {
        let segment = segment.replace("~1", "/").replace("~0", "~");
        match target.get(&segment) {
            Some(next) => target = next,
            None => return schema,
        }
    }
    target
}

/// Extracts the JSON value slice from model output, tolerating a ```` ```json ```` fence and
/// leading/trailing prose. Mirrors the streaming fence stripper: locate the first `{`/`[`,
/// walk to the matching close tracking string state.
fn extract_json_slice(text: &str) -> &str {
    let trimmed = text.trim();

    // explicit fenced block: take everything between the opening fence and the closing fence.
    if let Some(rest) = trimmed.strip_prefix("```") {
        let after_lang = match rest.find('\n') {
            Some(pos) => &rest[pos + 1..],
            None => rest,
        };
        let body = match after_lang.rfind("```") {
            Some(pos) => &after_lang[..pos],
            None => after_lang,
        };
        return body.trim();
    }

    let bytes = trimmed.as_bytes();
    let Some(start) = bytes.iter().position(|b| *b == b'{' || *b == b'[') else {
        // primitive (top-level string/number) or garbage: let serde produce the error.
        return trimmed;
    };

    let mut depth: i64 = 0;
    let mut in_string = false;
    let mut escape_next = false;
    let mut end = bytes.len();
    let mut idx = start;
    while idx < bytes.len() {
        let b = bytes[idx];
        if escape_next {
            escape_next = false;
        } else if b == b'\\' && in_string {
            escape_next = true;
        } else if b == b'"' {
            in_string = !in_string;
        } else if !in_string {
            match b {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = idx + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        idx += 1;
    }
    trimmed[start..end].trim()
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Deserialize, JsonSchema)]
    // fields exist only to drive schema generation, never read
    #[allow(dead_code)]
    struct Person {
        name: String,
        age: u32,
        nickname: Option<String>,
    }

    fn rail() -> SchemaOutputGuardrail {
        SchemaOutputGuardrail::for_type::<Person>()
    }

    fn ctx<'a>(token: &'a str, full: &'a str) -> ChunkContext<'a> {
        ChunkContext {
            token,
            window: full,
            full,
        }
    }

    #[test]
    fn accepts_valid_object() {
        rail()
            .validate_complete(r#"{"name":"Alice","age":30}"#)
            .unwrap();
    }

    #[test]
    fn accepts_fenced_json_with_prose() {
        let out = "结果是:\n```json\n{\"name\":\"Bob\",\"age\":12}\n```\n";
        rail().validate_complete(out).unwrap();
    }

    #[test]
    fn rejects_non_json() {
        let err = rail().validate_complete("I do not know").unwrap_err();
        assert!(err.contains("not valid JSON"), "got: {err}");
    }

    #[test]
    fn rejects_wrong_type() {
        let err = rail()
            .validate_complete(r#"{"name":"Alice","age":"thirty"}"#)
            .unwrap_err();
        assert!(
            err.contains("$.age") && err.contains("integer"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_missing_required() {
        let err = rail().validate_complete(r#"{"name":"Alice"}"#).unwrap_err();
        assert!(err.contains("age"), "got: {err}");
    }

    #[test]
    fn rejects_unknown_property_in_strict_mode() {
        let err = rail()
            .validate_complete(r#"{"name":"A","age":1,"mood":"ok"}"#)
            .unwrap_err();
        assert!(err.contains("mood"), "got: {err}");
        // opting out restores permissive behavior
        rail()
            .allow_additional_properties()
            .validate_complete(r#"{"name":"A","age":1,"mood":"ok"}"#)
            .unwrap();
    }

    #[test]
    fn accepts_null_for_optional_field() {
        // schemars models Option<T> as a nullable / anyOf schema; both encodings must pass.
        rail()
            .validate_complete(r#"{"name":"A","age":1,"nickname":null}"#)
            .unwrap();
    }

    #[test]
    fn validates_ref_and_enum_in_handwritten_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["role"],
            "properties": {
                "role": { "$ref": "#/$defs/Role" }
            },
            "$defs": {
                "Role": { "type": "string", "enum": ["admin", "user"] }
            }
        });
        let rail = SchemaOutputGuardrail::new(schema);
        rail.validate_complete(r#"{"role":"admin"}"#).unwrap();
        let err = rail.validate_complete(r#"{"role":"root"}"#).unwrap_err();
        assert!(err.contains("$.role"), "got: {err}");
    }

    #[tokio::test]
    async fn streaming_progressive_states() {
        let rail = rail();
        // preamble
        assert!(rail.validate_chunk(&ctx("Here", "Here")).await == ChunkAction::Pass);
        // partial object, required key still missing — tolerated
        assert!(
            rail.validate_chunk(&ctx("", r#"{"name":"Alice","ag"#))
                .await
                == ChunkAction::Pass
        );
        // definitive type contradiction on a completed value — blocked early
        assert!(
            rail.validate_chunk(&ctx("", r#"{"name":"Alice","age":"x"}"#))
                .await
                == ChunkAction::Block
        );
        // valid complete document — streaming still passes, terminal gate also passes
        assert!(
            rail.validate_chunk(&ctx("", r#"{"name":"Alice","age":9}"#))
                .await
                == ChunkAction::Pass
        );
        rail.validate_complete(r#"{"name":"Alice","age":9}"#)
            .unwrap();
        // terminal strictness: missing required blocks
        assert!(rail.validate_complete(r#"{"name":"Alice"}"#).is_err());
    }
}
