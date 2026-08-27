//! SQL tool (read-only, SQLite, supports bind parameters)

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use regex::Regex;

use lc_core::tools::ToolError;
use lc_core::BaseTool;

/// Lazy-compiled regex for extracting table names from SQL.
static TABLE_NAME_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)")
        .unwrap()
});

/// SQL query tool (read-only SELECT, table whitelist)
pub struct SQLTool {
    conn: Mutex<rusqlite::Connection>,
    allowed_tables: Vec<String>,
}

impl SQLTool {
    /// Creates a new SQL tool over the SQLite database at `path`.
    pub fn new(path: &str) -> Result<Self, ToolError> {
        let conn = rusqlite::Connection::open(path)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        Ok(Self {
            conn: Mutex::new(conn),
            allowed_tables: Vec::new(),
        })
    }

    /// Restricts queries to the given whitelist of table names.
    pub fn with_allowed_tables(mut self, tables: Vec<String>) -> Self {
        self.allowed_tables = tables;
        self
    }

    /// Extract table names from a SELECT SQL statement.
    fn extract_table_names(sql: &str) -> Vec<String> {
        let mut tables = Vec::new();
        for cap in TABLE_NAME_RE.captures_iter(sql) {
            for name in cap[1].split(',') {
                let trimmed = name.trim().to_lowercase();
                if !trimmed.is_empty() {
                    tables.push(trimmed);
                }
            }
        }
        tables
    }

    /// Execute a SELECT query (read-only, without bind parameters).
    pub fn execute(&self, sql: &str) -> Result<Vec<HashMap<String, String>>, ToolError> {
        self.execute_parameterized(sql, &[])
    }

    /// Execute a SELECT query with bind parameters (read-only, single statement).
    ///
    /// Placeholders `?1` / `?2` ... in the SQL text are bound by position from `params` (Q5) —
    /// values are no longer spliced into the SQL as literals, so injected content only matches
    /// as a literal value and cannot become a new statement.
    /// Validation rules match `execute`: only a single SELECT is allowed; multi-statements,
    /// comments, and dangerous functions are rejected; an optional table whitelist applies.
    pub fn execute_parameterized(
        &self,
        sql: &str,
        params: &[rusqlite::types::Value],
    ) -> Result<Vec<HashMap<String, String>>, ToolError> {
        let trimmed = sql.trim();

        if !trimmed.to_lowercase().starts_with("select") {
            return Err(ToolError::InvalidInput(
                "Only SELECT queries are allowed (read-only)".to_string(),
            ));
        }

        if trimmed.contains(';') {
            return Err(ToolError::InvalidInput(
                "Semicolons are not allowed in queries (single SELECT only)".to_string(),
            ));
        }

        if trimmed.contains("--") || trimmed.contains("/*") || trimmed.contains("*/") {
            return Err(ToolError::InvalidInput(
                "SQL comments are not allowed in queries".to_string(),
            ));
        }

        let lower = trimmed.to_lowercase();
        let dangerous_patterns = [
            "into outfile",
            "into dumpfile",
            "load_file",
            "benchmark",
            "sleep",
            "waitfor",
            "exec",
            "execute",
            "xp_",
            "sp_",
        ];
        for pattern in dangerous_patterns {
            if lower.contains(pattern) {
                return Err(ToolError::InvalidInput(format!(
                    "Potentially dangerous SQL pattern detected: '{}'",
                    pattern
                )));
            }
        }

        if !self.allowed_tables.is_empty() {
            let tables_in_sql = Self::extract_table_names(sql);
            let allowed_lower: Vec<String> = self
                .allowed_tables
                .iter()
                .map(|t| t.to_lowercase())
                .collect();
            for table in &tables_in_sql {
                if !allowed_lower.contains(table) {
                    return Err(ToolError::InvalidInput(format!(
                        "Table '{}' is not in the allowed list. Allowed: {:?}, found: {:?}",
                        table, self.allowed_tables, tables_in_sql
                    )));
                }
            }
            if tables_in_sql.is_empty() {
                return Err(ToolError::InvalidInput(
                    "SQL does not reference any table. At least one table must be specified."
                        .to_string(),
                ));
            }
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        let col_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                let mut m = HashMap::new();
                for (i, col) in col_names.iter().enumerate() {
                    let val: String = row
                        .get::<_, Option<String>>(i)
                        .unwrap_or(None)
                        .unwrap_or_default();
                    m.insert(col.clone(), val);
                }
                Ok(m)
            })
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        let mut result = Vec::new();
        for r in rows {
            result.push(r.map_err(|e| ToolError::ExecutionFailed(e.to_string()))?);
        }
        Ok(result)
    }
}

#[async_trait]
impl BaseTool for SQLTool {
    fn name(&self) -> &str {
        "sql_query"
    }

    fn description(&self) -> &str {
        "Execute SQL SELECT queries (read-only). Input: SQL string."
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let (sql, params) = parse_sql_input(&input)?;
        let rows = self.execute_parameterized(&sql, &params)?;
        serde_json::to_string(&rows).map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    }
}

/// Parses the tool input: prefers `{"sql": "...", "params": [...]}` (parameterized, Q5),
/// otherwise treats the whole input as plain SQL text (compatible with the old interface).
fn parse_sql_input(input: &str) -> Result<(String, Vec<rusqlite::types::Value>), ToolError> {
    if input.trim_start().starts_with('{') {
        let json: serde_json::Value = serde_json::from_str(input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON parse error: {}", e)))?;
        let sql = json
            .get("sql")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidInput("JSON input must have a 'sql' string field".to_string())
            })?
            .to_string();
        let params = json
            .get("params")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(json_to_sql_value).collect())
            .unwrap_or_default();
        Ok((sql, params))
    } else {
        Ok((input.to_string(), Vec::new()))
    }
}

/// Converts a JSON value into an SQL bind parameter: null → NULL, bool → 0/1,
/// number → Integer/Real, string → Text, any other compound value → NULL.
fn json_to_sql_value(v: &serde_json::Value) -> rusqlite::types::Value {
    use rusqlite::types::Value as SqlValue;
    match v {
        serde_json::Value::Null => SqlValue::Null,
        serde_json::Value::Bool(b) => SqlValue::Integer(*b as i64),
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(SqlValue::Integer)
            .or_else(|| n.as_f64().map(SqlValue::Real))
            .unwrap_or(SqlValue::Null),
        serde_json::Value::String(s) => SqlValue::Text(s.clone()),
        _ => SqlValue::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_with_data() -> SQLTool {
        let tool = SQLTool::new(":memory:").unwrap();
        {
            let conn = tool.conn.lock().unwrap_or_else(|e| e.into_inner());
            conn.execute("CREATE TABLE users (id INTEGER, name TEXT)", [])
                .unwrap();
            conn.execute("INSERT INTO users VALUES (1, 'Alice')", [])
                .unwrap();
            conn.execute("INSERT INTO users VALUES (2, 'Bob')", [])
                .unwrap();
            // The orders table is used by the multi-table JOIN whitelist tests (its earlier
            // absence made prepare fail with "no such table", but the SQL tests were gated
            // behind a feature so they never ran by default and stayed dormant).
            conn.execute("CREATE TABLE orders (id INTEGER, user_id INTEGER)", [])
                .unwrap();
            conn.execute("INSERT INTO orders VALUES (1, 1)", [])
                .unwrap();
        }
        tool
    }

    #[test]
    fn test_select() {
        let tool = tool_with_data();
        let rows = tool.execute("SELECT * FROM users").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("name"), Some(&"Alice".to_string()));
    }

    #[test]
    fn test_non_select_rejected() {
        let tool = tool_with_data();
        assert!(tool.execute("DROP TABLE users").is_err());
        assert!(tool.execute("INSERT INTO users VALUES (3, 'Eve')").is_err());
    }

    #[test]
    fn test_allowed_tables_exact_match() {
        let tool = tool_with_data().with_allowed_tables(vec!["users".to_string()]);
        assert!(tool.execute("SELECT * FROM users").is_ok());
        assert!(tool.execute("SELECT * FROM users2").is_err());
    }

    #[test]
    fn test_allowed_tables_blocks_unknown() {
        let tool = tool_with_data().with_allowed_tables(vec!["orders".to_string()]);
        assert!(tool.execute("SELECT * FROM users").is_err());
    }

    #[test]
    fn test_extract_table_names() {
        let tables = SQLTool::extract_table_names("SELECT * FROM users WHERE id = 1");
        assert_eq!(tables, vec!["users"]);

        let tables = SQLTool::extract_table_names(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id",
        );
        assert_eq!(tables, vec!["users", "orders"]);

        let tables = SQLTool::extract_table_names("SELECT * FROM users, orders");
        assert_eq!(tables, vec!["users", "orders"]);
    }

    #[test]
    fn test_allowed_tables_with_join() {
        let tool =
            tool_with_data().with_allowed_tables(vec!["users".to_string(), "orders".to_string()]);
        assert!(tool
            .execute("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
            .is_ok());
    }

    #[test]
    fn test_allowed_tables_blocks_partial_join() {
        let tool = tool_with_data().with_allowed_tables(vec!["users".to_string()]);
        assert!(tool
            .execute("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
            .is_err());
    }

    #[test]
    fn test_semicolon_rejected() {
        let tool = tool_with_data();
        assert!(tool
            .execute("SELECT * FROM users; DROP TABLE users")
            .is_err());
    }

    #[test]
    fn test_sql_comments_rejected() {
        let tool = tool_with_data();
        assert!(tool.execute("SELECT * FROM users -- comment").is_err());
        assert!(tool
            .execute("SELECT * FROM users /* block comment */")
            .is_err());
    }

    #[test]
    fn test_dangerous_patterns_rejected() {
        let tool = tool_with_data();
        assert!(tool.execute("SELECT sleep(1) FROM users").is_err());
        assert!(tool.execute("SELECT benchmark(1, 1) FROM users").is_err());
    }

    /// Q5: bind parameters take effect by position.
    #[test]
    fn test_parameterized_query() {
        let tool = tool_with_data();
        let rows = tool
            .execute_parameterized(
                "SELECT * FROM users WHERE name = ?1",
                &[rusqlite::types::Value::Text("Alice".to_string())],
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some(&"Alice".to_string()));
    }

    /// Q5: even if a parameter value contains an injection snippet, it only matches
    /// as a literal value and cannot form a new statement.
    #[test]
    fn test_parameterized_prevents_injection() {
        let tool = tool_with_data();
        let rows = tool
            .execute_parameterized(
                "SELECT * FROM users WHERE name = ?1",
                &[rusqlite::types::Value::Text(
                    "Alice'; DROP TABLE users;--".to_string(),
                )],
            )
            .unwrap();
        assert!(rows.is_empty(), "注入片段只应作为字面值匹配不到任何行");

        // The table still exists; subsequent queries are unaffected
        let rows = tool.execute("SELECT COUNT(*) FROM users").unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[tokio::test]
    async fn test_run_accepts_parameterized_json() {
        let tool = tool_with_data();
        let result = tool
            .run(r#"{"sql": "SELECT * FROM users WHERE id = ?1", "params": [1]}"#.to_string())
            .await;
        assert!(result.is_ok(), "got error: {:?}", result.err());
        let rows: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(rows.as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_run_accepts_raw_sql() {
        let tool = tool_with_data();
        let result = tool.run("SELECT * FROM users".to_string()).await;
        assert!(result.is_ok());
        let rows: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(rows.as_array().unwrap().len(), 2);
    }
}
