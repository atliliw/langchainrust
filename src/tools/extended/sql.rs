//! SQL tool (read-only, SQLite)

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use regex::Regex;

use crate::core::tools::ToolError;
use crate::BaseTool;

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
    pub fn new(path: &str) -> Result<Self, ToolError> {
        let conn = rusqlite::Connection::open(path)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        Ok(Self {
            conn: Mutex::new(conn),
            allowed_tables: Vec::new(),
        })
    }

    pub fn with_allowed_tables(mut self, tables: Vec<String>) -> Self {
        self.allowed_tables = tables;
        self
    }

    /// Extract table names from a SELECT SQL statement.
    ///
    /// Matches patterns like `FROM table`, `JOIN table`, `FROM t1, t2`.
    /// Returns lowercase table names for comparison.
    fn extract_table_names(sql: &str) -> Vec<String> {
        let mut tables = Vec::new();
        for cap in TABLE_NAME_RE.captures_iter(sql) {
            // Split comma-separated table names
            for name in cap[1].split(',') {
                let trimmed = name.trim().to_lowercase();
                if !trimmed.is_empty() {
                    tables.push(trimmed);
                }
            }
        }
        tables
    }

    /// Execute a SELECT query (read-only)
    pub fn execute(&self, sql: &str) -> Result<Vec<HashMap<String, String>>, ToolError> {
        let trimmed = sql.trim();

        // Must start with SELECT (case-insensitive)
        if !trimmed.to_lowercase().starts_with("select") {
            return Err(ToolError::InvalidInput(
                "Only SELECT queries are allowed (read-only)".to_string(),
            ));
        }

        // Block semicolons to prevent multi-statement injection
        if trimmed.contains(';') {
            return Err(ToolError::InvalidInput(
                "Semicolons are not allowed in queries (single SELECT only)".to_string(),
            ));
        }

        // Block SQL comments that could be used for injection
        if trimmed.contains("--") || trimmed.contains("/*") || trimmed.contains("*/") {
            return Err(ToolError::InvalidInput(
                "SQL comments are not allowed in queries".to_string(),
            ));
        }

        // Block common dangerous patterns that could bypass read-only restriction
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

        // Table whitelist check: exact match on extracted table names
        if !self.allowed_tables.is_empty() {
            let tables_in_sql = Self::extract_table_names(sql);
            let allowed_lower: Vec<String> = self
                .allowed_tables
                .iter()
                .map(|t| t.to_lowercase())
                .collect();
            // ALL extracted tables must be in the allowed list
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
            .query_map([], |row| {
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
        let rows = self.execute(&input)?;
        serde_json::to_string(&rows).map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_with_data() -> SQLTool {
        let tool = SQLTool::new(":memory:").unwrap();
        {
            let conn = tool.conn.lock().unwrap();
            conn.execute("CREATE TABLE users (id INTEGER, name TEXT)", [])
                .unwrap();
            conn.execute("INSERT INTO users VALUES (1, 'Alice')", [])
                .unwrap();
            conn.execute("INSERT INTO users VALUES (2, 'Bob')", [])
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
        // "users2" should NOT match "users" — exact table name matching
        assert!(tool.execute("SELECT * FROM users2").is_err());
    }

    #[test]
    fn test_allowed_tables_blocks_unknown() {
        let tool = tool_with_data().with_allowed_tables(vec!["orders".to_string()]);
        // SQL does not reference any allowed table
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
        // JOIN with all tables in allowed list should pass
        assert!(tool
            .execute("SELECT * FROM users JOIN orders ON users.id = orders.user_id")
            .is_ok());
    }

    #[test]
    fn test_allowed_tables_blocks_partial_join() {
        let tool = tool_with_data().with_allowed_tables(vec!["users".to_string()]);
        // JOIN with a table NOT in allowed list should be blocked
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
}
