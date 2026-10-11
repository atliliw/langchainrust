//! Schema-bound output guardrail.
//!
//! [`SchemaOutputGuardrail`] binds an LLM output to a JSON Schema (the same schema
//! `schemars::schema_for!` / `StructuredOutput<T>` uses): non-JSON output, type mismatches,
//! missing required keys and (in strict mode) undeclared properties are blocked fail-fast
//! instead of being handed to downstream parsing.
//!
//! Strict mode synthesizes `additionalProperties: false` when the schema omits the keyword, and
//! a synthesized denial is judged against the whole composition (`allOf`/`anyOf`/`oneOf`
//! branches and `$ref` targets) so a `#[serde(flatten)]` field declared by a sibling branch is
//! not mistaken for an extra one. Every block reason ends with the JSON shape the schema
//! expects, so the feedback says what to produce rather than only what was wrong.
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
use std::collections::BTreeSet;

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
    ///
    /// The reason names the offending path and ends with the JSON shape the schema expects, so
    /// a caller (or a retry loop feeding the message back to the model) is told what to produce
    /// rather than only what was wrong.
    pub fn validate_complete(&self, output: &str) -> Result<(), String> {
        let json = extract_json_slice(output);
        let value: Value = serde_json::from_str(json)
            .map_err(|e| self.with_expected_shape(format!("output is not valid JSON: {e}")))?;
        validate(
            &self.schema,
            &self.schema,
            &value,
            CheckMode::terminal(self.strict_properties),
            "$",
        )
        .map_err(|reason| self.with_expected_shape(reason))
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
            )
            .map_err(|reason| self.with_expected_shape(reason)),
            // Incomplete (preamble, unclosed structure) is expected mid-stream.
            Err(_) => Ok(()),
        }
    }

    /// Appends the expected JSON shape to a block reason.
    ///
    /// A reason alone ("property not declared in schema") leaves the caller guessing; the
    /// skeleton says what a conforming value looks like, which is the actionable half.
    fn with_expected_shape(&self, reason: String) -> String {
        format!("{reason}; expected shape: {}", prototype_json(&self.schema))
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
            // The reason travels with the block: it is what the runner audits and what a
            // phase-two rewrite rail (if any) is handed. The terminal check still produces the
            // authoritative reason if the model keeps going.
            Err(reason) => ChunkAction::Block { reason },
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
    validate_within(root, schema, value, mode, path, None)
}

/// [`validate`] with the enclosing composition's declared property names in hand.
///
/// `declared_for_value` is the set of property names declared by the schema object that owns
/// `value`, together with its `allOf`/`anyOf`/`oneOf` branches and their `$ref` targets.
/// `None` means "this is a value nested in a property, work the set out from `schema`"; `Some`
/// is passed down when a composition branch validates the *same* value as its parent, so the
/// branch does not mistake a sibling's property for an extra one (the shape `schemars` emits for
/// `#[serde(flatten)]`).
fn validate_within(
    root: &Value,
    schema: &Value,
    value: &Value,
    mode: CheckMode,
    path: &str,
    declared_for_value: Option<&BTreeSet<String>>,
) -> Result<(), String> {
    // resolve $ref (schemars may inline, but enum/newtype variants still emit refs).
    let schema = resolve_ref(root, schema);

    let composition_declared = if declared_for_value.is_none()
        && mode.strict_properties
        && matches!(value, Value::Object(_))
    {
        let mut declared = BTreeSet::new();
        collect_declared_properties(root, schema, &mut declared, 0);
        Some(declared)
    } else {
        None
    };
    let declared = declared_for_value.or(composition_declared.as_ref());

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
            validate_within(root, sub, value, mode, path, declared)?;
        }
    }

    if let Some(subs) = schema.get("anyOf").and_then(Value::as_array) {
        let any = subs
            .iter()
            .any(|sub| validate_within(root, sub, value, mode, path, declared).is_ok());
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
            .filter(|sub| validate_within(root, sub, value, mode, path, declared).is_ok())
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
        check_type(root, schema, ty, value, mode, path, declared)?;
    } else {
        // JSON Schema allows the object/array keywords without a `type` keyword, and `allOf`
        // branches are commonly written that way. Skipping them because no `type` was declared
        // fails open: a value that violates `properties` or `items` would pass because nothing
        // ever looked at it. The checks run only when the schema actually carries such a
        // keyword, so a `{"anyOf": [...]}` / `{"enum": [...]}` wrapper is not mistaken for an
        // object schema and asked to deny every key.
        match value {
            Value::Object(map) if has_object_keywords(schema) => {
                validate_object(root, schema, map, mode, path, declared)?
            }
            Value::Array(items) if has_array_keywords(schema) => {
                validate_array(root, schema, items, mode, path)?
            }
            _ => {}
        }
    }

    Ok(())
}

/// Whether `schema` carries any keyword that constrains an object, i.e. whether running
/// [`validate_object`] against it can mean anything.
fn has_object_keywords(schema: &Value) -> bool {
    [
        "properties",
        "required",
        "additionalProperties",
        "patternProperties",
        "minProperties",
        "maxProperties",
    ]
    .iter()
    .any(|keyword| schema.get(keyword).is_some())
}

/// Whether `schema` carries any keyword that constrains an array.
fn has_array_keywords(schema: &Value) -> bool {
    ["items", "minItems", "maxItems"]
        .iter()
        .any(|keyword| schema.get(keyword).is_some())
}

/// Type-keyword dispatch; object/array carry their own nested validation.
fn check_type(
    root: &Value,
    schema: &Value,
    ty: &Value,
    value: &Value,
    mode: CheckMode,
    path: &str,
    declared: Option<&BTreeSet<String>>,
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
        Value::Object(map) => validate_object(root, schema, map, mode, path, declared),
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
    declared: Option<&BTreeSet<String>>,
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
    // `additionalProperties: false` is an explicit statement about this schema object and is
    // honoured exactly as written. When the keyword is *omitted*, the denial is synthesized by
    // `strict_properties`, and a synthesized denial is judged against `declared` — the property
    // names of the whole composition, not this branch alone. `patternProperties` holds regexes
    // this validator does not evaluate, so its presence means extra keys cannot be judged at
    // all: staying silent beats rejecting a key that a pattern would have allowed.
    let synthesize_denial = additional.is_none() && schema.get("patternProperties").is_none();

    for (key, sub_value) in map {
        let child_path = format!("{path}.{key}");
        match properties.and_then(|p| p.get(key)) {
            Some(sub) => validate(root, sub, sub_value, mode, &child_path)?,
            None => {
                let deny_extra = match additional {
                    Some(Value::Bool(false)) => true,
                    Some(_) => false,
                    None => {
                        synthesize_denial
                            && declared.is_some_and(|declared| !declared.contains(key))
                    }
                };
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

/// Collects every property name declared by `schema` or by anything it composes with —
/// `allOf`/`anyOf`/`oneOf` branches and `$ref` targets, transitively.
///
/// Only consulted for a *synthesized* `additionalProperties: false`, where the question is
/// whether a key is genuinely undeclared or merely declared somewhere else in the composition.
/// A recursive type (`$ref` pointing back at an ancestor) would otherwise recurse forever, so
/// `depth` caps the walk; beyond it the set is incomplete, which can only make the check more
/// permissive, never wrongly strict.
fn collect_declared_properties(
    root: &Value,
    schema: &Value,
    out: &mut BTreeSet<String>,
    depth: usize,
) {
    const MAX_DEPTH: usize = 16;
    if depth > MAX_DEPTH {
        return;
    }
    let schema = resolve_ref(root, schema);
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        out.extend(properties.keys().cloned());
    }
    for keyword in ["allOf", "anyOf", "oneOf"] {
        if let Some(subs) = schema.get(keyword).and_then(Value::as_array) {
            for sub in subs {
                collect_declared_properties(root, sub, out, depth + 1);
            }
        }
    }
}

/// Depth cap for [`prototype`]: a recursive schema must not expand without bound.
const PROTOTYPE_MAX_DEPTH: usize = 8;

/// Derives a fillable JSON skeleton from `schema` — what a conforming value looks like.
///
/// Used as feedback in block reasons: a caller (or a retry loop feeding the message back to the
/// model) learns the expected shape, not just that something was wrong. Every choice is the
/// first legal one: `const`/`enum` give their first case, a union gives its first branch, an
/// object gives one entry per declared property, an array one element. The result is a valid
/// JSON value, not necessarily a *valid instance* (a `minLength` string is still `""`).
fn prototype(root: &Value, schema: &Value, depth: usize) -> Value {
    if depth > PROTOTYPE_MAX_DEPTH {
        return Value::Null;
    }
    let schema = resolve_ref(root, schema);

    if let Some(constant) = schema.get("const") {
        return constant.clone();
    }
    if let Some(first) = schema
        .get("enum")
        .and_then(Value::as_array)
        .and_then(|cases| cases.first())
    {
        return first.clone();
    }
    if let Some(subs) = schema.get("allOf").and_then(Value::as_array) {
        // `allOf` narrows a value, so the skeleton is the merge of every branch's object shape —
        // which is exactly how schemars renders `#[serde(flatten)]`.
        let mut merged = serde_json::Map::new();
        for sub in subs {
            if let Value::Object(fields) = prototype(root, sub, depth + 1) {
                merged.extend(fields);
            }
        }
        if !merged.is_empty() {
            return Value::Object(merged);
        }
    }
    if let Some(first) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(Value::as_array)
        .and_then(|subs| subs.first())
    {
        return prototype(root, first, depth + 1);
    }

    match schema_type_name(schema) {
        Some("object") => {
            let mut fields = serde_json::Map::new();
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                for (name, sub) in properties {
                    fields.insert(name.clone(), prototype(root, sub, depth + 1));
                }
            }
            Value::Object(fields)
        }
        Some("array") => {
            let item = schema
                .get("items")
                .map(|sub| prototype(root, sub, depth + 1))
                .unwrap_or(Value::Null);
            Value::Array(vec![item])
        }
        Some("string") => Value::String(String::new()),
        Some("integer") | Some("number") => Value::from(0),
        Some("boolean") => Value::Bool(false),
        _ => Value::Null,
    }
}

/// The schema's declared type name: a plain `type` string, or the first entry of a `type` array
/// that is not `"null"` (so `{"type": ["string", "null"]}` reads as a string).
fn schema_type_name(schema: &Value) -> Option<&str> {
    match schema.get("type")? {
        Value::String(name) => Some(name.as_str()),
        Value::Array(names) => names
            .iter()
            .filter_map(Value::as_str)
            .find(|name| *name != "null"),
        _ => None,
    }
}

/// Renders [`prototype`] of the root schema as JSON for a block reason.
fn prototype_json(root: &Value) -> String {
    serde_json::to_string(&prototype(root, root, 0)).unwrap_or_else(|_| "{}".to_string())
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
        // definitive type contradiction on a completed value — blocked early, with the reason
        match rail
            .validate_chunk(&ctx("", r#"{"name":"Alice","age":"x"}"#))
            .await
        {
            ChunkAction::Block { reason } => {
                assert!(reason.contains("$.age"), "got: {reason}");
                assert!(reason.contains("expected shape"), "got: {reason}");
            }
            other => panic!("应为 Block, 实际: {other:?}"),
        }
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

    // ---- T2: composition-aware strictness + prototype feedback ----

    /// The schema `schemars` emits for a struct with `#[serde(flatten)]`: the outer `properties`
    /// sit beside an `allOf` branch that holds the flattened fields.
    fn flattened_schema() -> Value {
        serde_json::json!({
            "type": "object",
            "properties": { "own": { "type": "string" } },
            "required": ["own"],
            "allOf": [{ "$ref": "#/$defs/Inner" }],
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": { "flattened": { "type": "integer" } },
                    "required": ["flattened"]
                }
            }
        })
    }

    #[test]
    fn accepts_property_declared_by_a_sibling_allof_branch() {
        // regression: strict mode synthesized `additionalProperties: false` per branch, so the
        // outer branch rejected the flattened field even though the `allOf` branch declares it —
        // a legal output blocked.
        SchemaOutputGuardrail::new(flattened_schema())
            .validate_complete(r#"{"own":"a","flattened":1}"#)
            .unwrap();
    }

    #[test]
    fn still_rejects_a_property_declared_nowhere() {
        let err = SchemaOutputGuardrail::new(flattened_schema())
            .validate_complete(r#"{"own":"a","flattened":1,"extra":true}"#)
            .unwrap_err();
        assert!(err.contains("$.extra"), "got: {err}");
    }

    #[test]
    fn explicit_additional_properties_false_stays_strict() {
        // The keyword is a statement about its own schema object (per the JSON Schema spec it
        // does not see `allOf` siblings), so an explicit `false` is honoured as written even
        // though the synthesized denial would have allowed the key.
        let mut schema = flattened_schema();
        schema["additionalProperties"] = Value::Bool(false);
        let err = SchemaOutputGuardrail::new(schema)
            .validate_complete(r#"{"own":"a","flattened":1}"#)
            .unwrap_err();
        assert!(err.contains("$.flattened"), "got: {err}");
    }

    #[test]
    fn pattern_properties_suspends_the_synthesized_denial() {
        // The validator does not evaluate regexes, so it cannot judge an extra key here: staying
        // silent beats rejecting a key a pattern would have allowed.
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "id": { "type": "string" } },
            "patternProperties": { "^x-": { "type": "string" } }
        });
        SchemaOutputGuardrail::new(schema)
            .validate_complete(r#"{"id":"a","x-trace":"t"}"#)
            .unwrap();
    }

    #[test]
    fn checks_object_keywords_without_a_type_keyword() {
        // fail-open regression: the object keywords were only reachable through a `type`
        // keyword, so a schema written without one validated anything.
        let schema = serde_json::json!({ "properties": { "a": { "type": "string" } } });
        let err = SchemaOutputGuardrail::new(schema)
            .validate_complete(r#"{"a":1}"#)
            .unwrap_err();
        assert!(err.contains("$.a"), "got: {err}");
    }

    #[test]
    fn does_not_treat_a_bare_union_as_an_object_schema() {
        // a wrapper with no object keywords must not be asked to deny every key
        let schema = serde_json::json!({
            "anyOf": [
                { "type": "object", "properties": { "a": { "type": "string" } } },
                { "type": "object", "properties": { "b": { "type": "string" } } }
            ]
        });
        let rail = SchemaOutputGuardrail::new(schema);
        rail.validate_complete(r#"{"a":"x"}"#).unwrap();
        rail.validate_complete(r#"{"b":"x"}"#).unwrap();
    }

    #[test]
    fn prototype_gives_a_fillable_skeleton() {
        let rail = rail();
        // a nullable field shows the shape of its non-null branch, which is the informative half
        assert_eq!(
            prototype(&rail.schema, &rail.schema, 0),
            serde_json::json!({ "name": "", "age": 0, "nickname": "" })
        );
    }

    #[test]
    fn prototype_follows_const_enum_and_first_union_branch() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "role": { "type": "string", "enum": ["admin", "user"] },
                "flag": { "const": true },
                "opt": { "type": ["string", "null"] },
                "tags": { "type": "array", "items": { "type": "string" } }
            }
        });
        assert_eq!(
            prototype(&schema, &schema, 0),
            serde_json::json!({
                "role": "admin",
                "flag": true,
                "opt": "",
                "tags": [""]
            })
        );
    }

    #[test]
    fn block_reason_carries_the_expected_shape() {
        let err = rail().validate_complete(r#"{"name":"A"}"#).unwrap_err();
        assert!(err.contains("expected shape"), "got: {err}");
        assert!(err.contains("\"age\""), "got: {err}");
        // unparseable output gets the same feedback: the shape is what the caller must produce
        let err = rail().validate_complete("I do not know").unwrap_err();
        assert!(err.contains("not valid JSON"), "got: {err}");
        assert!(err.contains("expected shape"), "got: {err}");
    }
}
