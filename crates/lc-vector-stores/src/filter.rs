// lc-vector-stores/src/filter.rs
//! 跨后端一致的元数据过滤类型。
//!
//! 设计目标(S3):`VectorStore` 的检索方法统一接收 [`MetadataFilter`],由每个后端
//! 自行翻译成原生查询语法(Qdrant payload filter / Pinecone filter / Chroma where /
//! LanceDB SQL 子句 / Cypher WHERE / 内存求值器 …)。类型形状对齐 LangChain 的
//! `FilterType`:单字段条件 + AND/OR 组合,覆盖常用比较与集合操作符。

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::HashMap;

/// 单字段元数据条件的比较操作符。
///
/// `Serialize` 用派生(输出 `Eq`/`Gt`…);`Deserialize` 手工实现(见下),
/// 除派生形状外还接受小写、符号形式(`"eq"`/`"="`/`">="`),供 SelfQuery(S4)
/// 从 LLM 结构化输出反序列化时更鲁棒。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FilterOp {
    /// 等于。
    Eq,
    /// 不等于。
    Ne,
    /// 大于。
    Gt,
    /// 大于等于。
    Gte,
    /// 小于。
    Lt,
    /// 小于等于。
    Lte,
    /// 命中给定值集合之一。
    In,
    /// 命中给定值集合之外。
    Nin,
}

/// 元数据过滤表达式:单个字段条件,或 AND/OR 组合。
///
/// 求值语义见 [`MetadataFilter::matches`];后端各自的语法翻译(如 Qdrant /
/// Pinecone / Chroma / LanceDB / Cypher)放在对应后端文件里,这里只承载类型与
/// 进程内求值(供内存/文件等本地后端使用)。
///
/// `Serialize` 用派生;`Deserialize` 手工实现([`MetadataFilter::from_json`]),
/// 兼容派生形状与 SelfQuery(S4) 用到的宽松单条件形状。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum MetadataFilter {
    /// 单个 `key op value` 条件。
    Field {
        /// 元数据字段名。
        key: String,
        /// 比较操作符。
        op: FilterOp,
        /// 比较值(`In`/`Nin` 时为数组)。
        value: Value,
    },
    /// 所有子过滤都必须命中。
    And(Vec<MetadataFilter>),
    /// 至少一个子过滤命中。
    Or(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// 构造单字段条件。
    pub fn field(key: impl Into<String>, op: FilterOp, value: impl Into<Value>) -> Self {
        Self::Field {
            key: key.into(),
            op,
            value: value.into(),
        }
    }

    /// 构造 AND 组合(空列表恒为真)。
    pub fn and(filters: Vec<MetadataFilter>) -> Self {
        Self::And(filters)
    }

    /// 构造 OR 组合(空列表恒为假)。
    pub fn or(filters: Vec<MetadataFilter>) -> Self {
        Self::Or(filters)
    }

    /// 对一份文档的元数据求值。
    ///
    /// 缺失字段的语义对齐 SQL NULL:
    /// - `Eq` / `In` / 排序操作符(`Gt/Gte/Lt/Lte`)对缺失字段**不匹配**(缺失不是某个值)。
    /// - `Ne` / `Nin` 对缺失字段**匹配**(NULL ≠ value)。
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

    /// 单值求值:数字按数值比较(1 == 1.0),字符串按字典序,其余类型走 JSON 相等。
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
    /// 从 JSON 构造过滤条件,接受两种形状:
    /// - 派生序列化形状:`{"Field": {...}}` / `{"And": [...]}` / `{"Or": [...]}`。
    /// - 宽松单条件形状:`{"key": ..., "op": ..., "value": ...}`(op 大小写/符号宽松)。
    ///
    /// 供 SelfQuery(S4) 从 LLM 结构化输出直接反序列化,兼容真实模型的输出差异。
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

/// 数值感知的 JSON 相等:`1` 与 `1.0` 视为相等,其余走 `Value` 的 PartialEq。
fn values_eq(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

/// 数值/字符串比较;类型不可比时返回 `None`(条件视为不命中)。
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
        // 数字按数值比较(1 == 1.0)
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
        // In/Nin 的 value 非数组时视为不命中(保守)。
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
        // 缺失字段:Eq/In/排序不命中,Ne/Nin 命中(SQL NULL 语义)。
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

    /// S4: 宽松反序列化 —— 大小写/符号 op + 直接单条件形状(LLM 友好)。
    #[test]
    fn test_deserialize_lenient_shapes() {
        // 宽松单条件形状:op 小写。
        let f: MetadataFilter =
            serde_json::from_value(json!({"key": "source", "op": "eq", "value": "docs"})).unwrap();
        assert_eq!(f, MetadataFilter::field("source", FilterOp::Eq, "docs"));

        // 符号 op。
        let f: MetadataFilter =
            serde_json::from_value(json!({"key": "year", "op": ">=", "value": 2020})).unwrap();
        assert_eq!(f, MetadataFilter::field("year", FilterOp::Gte, 2020));

        // 派生形状仍可解析(回退兼容)。
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

        // 未知 op 显式报错。
        let err = serde_json::from_value::<MetadataFilter>(
            json!({"key": "year", "op": "like", "value": 2020}),
        );
        assert!(err.is_err());
    }
}
