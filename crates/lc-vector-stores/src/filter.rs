// lc-vector-stores/src/filter.rs
//! Backend-agnostic metadata filter types.
//!
//! Design goal (S3): the `VectorStore` retrieval methods uniformly accept [`MetadataFilter`],
//! and each backend translates it into its native query syntax (Qdrant payload filter /
//! Pinecone filter / Chroma where / LanceDB SQL clause / Cypher WHERE / in-memory
//! evaluator …). The type shape mirrors LangChain's `FilterType`: single-field conditions
//! plus AND/OR combinations, covering the common comparison and set operators.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::HashMap;

/// Comparison operator for a single-field metadata condition.
///
/// `Serialize` uses the derived impl (outputs `Eq`/`Gt`…); `Deserialize` is
/// hand-implemented (see below), accepting lowercase and symbolic forms
/// (`"eq"`/`"="`/`">="`) in addition to the derived shape, so SelfQuery (S4)
/// can deserialize LLM structured output more robustly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FilterOp {
    /// Equal to.
    Eq,
    /// Not equal to.
    Ne,
    /// Greater than.
    Gt,
    /// Greater than or equal to.
    Gte,
    /// Less than.
    Lt,
    /// Less than or equal to.
    Lte,
    /// In the given value set.
    In,
    /// Not in the given value set.
    Nin,
}

/// Metadata filter expression: a single-field condition, or an AND/OR combination.
///
/// Evaluation semantics live in [`MetadataFilter::matches`]; each backend's syntax
/// translation (Qdrant / Pinecone / Chroma / LanceDB / Cypher) lives in its own file.
/// This module only carries the types and in-process evaluation (for in-memory/file
/// and other local backends).
///
/// `Serialize` uses the derived impl; `Deserialize` is hand-implemented
/// ([`MetadataFilter::from_json`]), supporting both the derived shape and the lenient
/// single-condition shape used by SelfQuery (S4).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum MetadataFilter {
    /// A single `key op value` condition.
    Field {
        /// Metadata field name.
        key: String,
        /// Comparison operator.
        op: FilterOp,
        /// Comparison value (an array for `In`/`Nin`).
        value: Value,
    },
    /// All sub-filters must match.
    And(Vec<MetadataFilter>),
    /// At least one sub-filter must match.
    Or(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// Builds a single-field condition.
    pub fn field(key: impl Into<String>, op: FilterOp, value: impl Into<Value>) -> Self {
        Self::Field {
            key: key.into(),
            op,
            value: value.into(),
        }
    }

    /// Builds an AND combination (an empty list is always true).
    pub fn and(filters: Vec<MetadataFilter>) -> Self {
        Self::And(filters)
    }

    /// Builds an OR combination (an empty list is always false).
    pub fn or(filters: Vec<MetadataFilter>) -> Self {
        Self::Or(filters)
    }

    /// Evaluates a document's metadata against this filter.
    ///
    /// Missing fields follow SQL NULL semantics:
    /// - `Eq` / `In` / ordering operators (`Gt/Gte/Lt/Lte`) **do not match** a missing field (absence is not a value).
    /// - `Ne` / `Nin` **match** a missing field (NULL ≠ value).
    pub fn matches(&self, metadata: &HashMap<String, Value>) -> bool {
        match self {
            Self::Field { key, op, value } => {
                let Some(actual) = metadata.get(key) else {
                    return matches!(op, FilterOp::Ne | FilterOp::Nin);
                };
                Self::value_matches(op, actual, value)
            }
            Self::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Self::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
        }
    }

    /// Single-value evaluation: numbers compare numerically (1 == 1.0), strings lexicographically, other types by JSON equality.
    fn value_matches(op: &FilterOp, actual: &Value, expected: &Value) -> bool {
        match op {
            FilterOp::Eq => values_eq(actual, expected),
            FilterOp::Ne => !values_eq(actual, expected),
            FilterOp::Gt => values_cmp(actual, expected).is_some_and(|o| o == Ordering::Greater),
            FilterOp::Gte => values_cmp(actual, expected).is_some_and(|o| o != Ordering::Less),
            FilterOp::Lt => values_cmp(actual, expected).is_some_and(|o| o == Ordering::Less),
            FilterOp::Lte => values_cmp(actual, expected).is_some_and(|o| o != Ordering::Greater),
            FilterOp::In => expected
                .as_array()
                .is_some_and(|set| set.iter().any(|v| values_eq(actual, v))),
            FilterOp::Nin => expected
                .as_array()
                .is_some_and(|set| !set.iter().any(|v| values_eq(actual, v))),
        }
    }
}

impl<'de> Deserialize<'de> for FilterOp {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let op = raw.trim().to_ascii_lowercase();
        let parsed = match op.as_str() {
            "eq" | "=" | "==" => FilterOp::Eq,
            "ne" | "!=" | "<>" => FilterOp::Ne,
            "gt" | ">" => FilterOp::Gt,
            "gte" | ">=" => FilterOp::Gte,
            "lt" | "<" => FilterOp::Lt,
            "lte" | "<=" => FilterOp::Lte,
            "in" => FilterOp::In,
            "nin" | "not in" => FilterOp::Nin,
            other => {
                return Err(serde::de::Error::custom(format!(
                    "unknown metadata filter op: {other}"
                )))
            }
        };
        Ok(parsed)
    }
}

impl<'de> Deserialize<'de> for MetadataFilter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::from_json(value).map_err(serde::de::Error::custom)
    }
}

impl MetadataFilter {
    /// Builds a filter from JSON, accepting two shapes:
    /// - The derived serialized shape: `{"Field": {...}}` / `{"And": [...]}` / `{"Or": [...]}`.
    /// - The lenient single-condition shape: `{"key": ..., "op": ..., "value": ...}`
    ///   (op accepts case-insensitive and symbolic forms).
    ///
    /// Used by SelfQuery (S4) to deserialize LLM structured output directly, tolerating
    /// real-model output variations.
    pub fn from_json(value: Value) -> Result<Self, String> {
        let obj = value
            .as_object()
            .ok_or_else(|| format!("MetadataFilter must be an object, got {value}"))?;
        if let Some(field) = obj.get("Field") {
            return Self::field_from_json(field);
        }
        if let Some(items) = obj.get("And") {
            let arr = items
                .as_array()
                .ok_or_else(|| "MetadataFilter And must be an array".to_string())?;
            let filters: Result<Vec<_>, _> = arr.iter().cloned().map(Self::from_json).collect();
            return Ok(Self::And(filters?));
        }
        if let Some(items) = obj.get("Or") {
            let arr = items
                .as_array()
                .ok_or_else(|| "MetadataFilter Or must be an array".to_string())?;
            let filters: Result<Vec<_>, _> = arr.iter().cloned().map(Self::from_json).collect();
            return Ok(Self::Or(filters?));
        }
        if obj.contains_key("key") || obj.contains_key("op") {
            return Self::field_from_json(&value);
        }
        Err(format!("unrecognized MetadataFilter JSON: {value}"))
    }

    fn field_from_json(value: &Value) -> Result<Self, String> {
        let obj = value
            .as_object()
            .ok_or_else(|| format!("MetadataFilter field must be an object, got {value}"))?;
        let key = obj
            .get("key")
            .and_then(|k| k.as_str())
            .ok_or_else(|| "MetadataFilter field is missing string key".to_string())?
            .to_string();
        let op = serde_json::from_value(
            obj.get("op")
                .cloned()
                .ok_or_else(|| "MetadataFilter field is missing op".to_string())?,
        )
        .map_err(|e| format!("invalid field op: {e}"))?;
        let value = obj
            .get("value")
            .cloned()
            .ok_or_else(|| "MetadataFilter field is missing value".to_string())?;
        Ok(Self::Field { key, op, value })
    }
}

/// Numeric-aware JSON equality: `1` and `1.0` are equal; everything else uses `Value`'s PartialEq.
fn values_eq(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

/// Numeric/string comparison; returns `None` when the types are incomparable (the condition is treated as not matching).
fn values_cmp(a: &Value, b: &Value) -> Option<Ordering> {
    if let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) {
        return x.partial_cmp(&y);
    }
    if let (Some(x), Some(y)) = (a.as_str(), b.as_str()) {
        return Some(x.cmp(y));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta() -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("source".to_string(), json!("docs"));
        m.insert("year".to_string(), json!(2024));
        m.insert("tags".to_string(), json!(["rust", "ml"]));
        m
    }

    #[test]
    fn test_eq_ne() {
        let m = meta();
        assert!(MetadataFilter::field("source", FilterOp::Eq, "docs").matches(&m));
        assert!(!MetadataFilter::field("source", FilterOp::Eq, "blog").matches(&m));
        assert!(MetadataFilter::field("source", FilterOp::Ne, "blog").matches(&m));
        assert!(!MetadataFilter::field("source", FilterOp::Ne, "docs").matches(&m));
        // numbers compare numerically (1 == 1.0)
        assert!(MetadataFilter::field("year", FilterOp::Eq, 2024.0).matches(&m));
    }

    #[test]
    fn test_ordering_ops() {
        let m = meta();
        assert!(MetadataFilter::field("year", FilterOp::Gt, 2020).matches(&m));
        assert!(MetadataFilter::field("year", FilterOp::Gte, 2024).matches(&m));
        assert!(!MetadataFilter::field("year", FilterOp::Gt, 2024).matches(&m));
        assert!(MetadataFilter::field("year", FilterOp::Lt, 2030).matches(&m));
        assert!(MetadataFilter::field("year", FilterOp::Lte, 2024).matches(&m));
        assert!(!MetadataFilter::field("year", FilterOp::Gt, "abc").matches(&m));
    }

    #[test]
    fn test_in_nin() {
        let m = meta();
        assert!(MetadataFilter::field("source", FilterOp::In, vec!["docs", "web"]).matches(&m));
        assert!(!MetadataFilter::field("source", FilterOp::In, vec!["blog", "web"]).matches(&m));
        assert!(MetadataFilter::field("source", FilterOp::Nin, vec!["blog", "web"]).matches(&m));
        assert!(!MetadataFilter::field("source", FilterOp::Nin, vec!["docs"]).matches(&m));
        // a non-array value for In/Nin is treated as not matching (conservative).
        assert!(!MetadataFilter::field("source", FilterOp::In, "docs").matches(&m));
    }

    #[test]
    fn test_and_or_composition() {
        let m = meta();
        let both = MetadataFilter::and(vec![
            MetadataFilter::field("source", FilterOp::Eq, "docs"),
            MetadataFilter::field("year", FilterOp::Gte, 2024),
        ]);
        assert!(both.matches(&m));

        let either = MetadataFilter::or(vec![
            MetadataFilter::field("source", FilterOp::Eq, "blog"),
            MetadataFilter::field("year", FilterOp::Gt, 2020),
        ]);
        assert!(either.matches(&m));

        let neither = MetadataFilter::or(vec![
            MetadataFilter::field("source", FilterOp::Eq, "blog"),
            MetadataFilter::field("year", FilterOp::Lt, 2000),
        ]);
        assert!(!neither.matches(&m));
    }

    #[test]
    fn test_missing_key_semantics() {
        let m = meta();
        // missing fields: Eq/In/ordering do not match, Ne/Nin match (SQL NULL semantics).
        assert!(!MetadataFilter::field("missing", FilterOp::Eq, "x").matches(&m));
        assert!(!MetadataFilter::field("missing", FilterOp::In, vec!["x"]).matches(&m));
        assert!(!MetadataFilter::field("missing", FilterOp::Gt, 1).matches(&m));
        assert!(MetadataFilter::field("missing", FilterOp::Ne, "x").matches(&m));
        assert!(MetadataFilter::field("missing", FilterOp::Nin, vec!["x"]).matches(&m));
    }

    #[test]
    fn test_serialize_roundtrip() {
        let f = MetadataFilter::and(vec![MetadataFilter::field("source", FilterOp::Eq, "docs")]);
        let json = serde_json::to_string(&f).unwrap();
        let back: MetadataFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(f, back);
    }

    /// S4: lenient deserialization — case-insensitive/symbolic op + direct single-condition shape (LLM-friendly).
    #[test]
    fn test_deserialize_lenient_shapes() {
        // lenient single-condition shape: lowercase op.
        let f: MetadataFilter =
            serde_json::from_value(json!({"key": "source", "op": "eq", "value": "docs"})).unwrap();
        assert_eq!(f, MetadataFilter::field("source", FilterOp::Eq, "docs"));

        // symbolic op.
        let f: MetadataFilter =
            serde_json::from_value(json!({"key": "year", "op": ">=", "value": 2020})).unwrap();
        assert_eq!(f, MetadataFilter::field("year", FilterOp::Gte, 2020));

        // the derived shape still parses (backward compatible).
        let f: MetadataFilter = serde_json::from_value(json!({
            "And": [
                {"Field": {"key": "source", "op": "Eq", "value": "docs"}},
                {"Field": {"key": "year", "op": "Lt", "value": 2030}}
            ]
        }))
        .unwrap();
        assert_eq!(
            f,
            MetadataFilter::and(vec![
                MetadataFilter::field("source", FilterOp::Eq, "docs"),
                MetadataFilter::field("year", FilterOp::Lt, 2030),
            ])
        );

        // unknown op errors explicitly.
        let err = serde_json::from_value::<MetadataFilter>(
            json!({"key": "year", "op": "like", "value": 2020}),
        );
        assert!(err.is_err());
    }
}
