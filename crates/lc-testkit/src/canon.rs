// lc-testkit/src/canon.rs
//! Deterministic JSON comparison (T1, v0.25.0).
//!
//! `serde_json`'s object ordering is not guaranteed across builds (and
//! `preserve_order` + `--all-features` in this workspace shifts map key order),
//! so asserting `desired == got` for fixtures is order-dependent. This helper
//! canonicalizes both sides — objects as sorted key→value maps, arrays as
//! order-preserving vecs — then compares, giving a deterministic equality that
//! is immune to key-order drift while still honoring array order.

use serde_json::Value;

/// Canonicalizable JSON value: objects map to sorted key/value pairs, arrays
/// keep order, scalars as-is.
#[derive(Debug, PartialEq, Eq)]
pub enum CanJson {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<CanJson>),
    /// Sorted by key so `{"b":1,"a":2}` === `{"a":2,"b":1}`.
    Object(Vec<(String, CanJson)>),
}

impl From<&Value> for CanJson {
    fn from(v: &Value) -> Self {
        match v {
            Value::Null => CanJson::Null,
            Value::Bool(b) => CanJson::Bool(*b),
            Value::Number(n) => CanJson::Number(n.to_string()),
            Value::String(s) => CanJson::String(s.clone()),
            Value::Array(a) => CanJson::Array(a.iter().map(CanJson::from).collect()),
            Value::Object(m) => {
                let mut pairs = m
                    .iter()
                    .map(|(k, v)| (k.clone(), CanJson::from(v)))
                    .collect::<Vec<_>>();
                // Sort by key — note: numbers like "2" < "10" lexicographically;
                // that's fine for *equality* (keys are unique within one object, so
                // sort only needs determinism, not numeric ordering).
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
                CanJson::Object(pairs)
            }
        }
    }
}

/// Asserts two JSON values are equal ignoring object key order but honoring
/// array order. Panics with a delta-ish message listing the first divergence.
///
/// ```
/// use lc_testkit::assert_json_canonical;
/// use serde_json::json;
///
/// let a = json!({"model":"gpt","n":1,"msgs":[1,2]});
/// let b = json!({"msgs":[1,2],"n":1,"model":"gpt"}); // different key order
/// assert_json_canonical(&a, &b, "key order must not matter");
/// ```
pub fn assert_json_canonical(actual: &Value, desired: &Value, context: &str) {
    let a = CanJson::from(actual);
    let d = CanJson::from(desired);
    assert_eq!(
        a, d,
        "JSON mismatch (canonical, key-order-agnostic): {context}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn equal_ignores_object_key_order() {
        let a = json!({"model":"gpt-4o-mini","n":1,"roles":{"user":"u"}});
        let b = json!({"roles":{"user":"u"},"n":1,"model":"gpt-4o-mini"});
        assert_json_canonical(&a, &b, "reordered keys");
    }

    #[test]
    fn nested_object_compare() {
        let a = json!({"embeddings":{"float":[[1.0, 2.0],[3.0, 4.0]]},"meta":{"n":2}});
        let b = json!({"meta":{"n":2},"embeddings":{"float":[[1.0, 2.0],[3.0, 4.0]]}});
        assert_json_canonical(&a, &b, "nested arrays order-preserving");
    }

    #[test]
    #[should_panic]
    fn array_order_matters() {
        let a = json!([1, 2, 3]);
        let b = json!([3, 2, 1]);
        assert_json_canonical(&a, &b, "array order must be honored");
    }

    #[test]
    #[should_panic]
    fn mismatch_detected() {
        let a = json!({"model":"gpt"});
        let b = json!({"model":"claude"});
        assert_json_canonical(&a, &b, "different model");
    }
}
