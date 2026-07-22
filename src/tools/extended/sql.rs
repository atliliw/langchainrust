//! SQL tool (read-only, SQLite)

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use regex::Regex;
use rusqlite::Connection;

use crate::core::tools::ToolError;
use crate::BaseTool;

/// SQL query tool (read-only SELECT, table whitelist)
pub struct SQLTool {
    conn: Mutex<Connection>,
    allowed_tables: Vec<String>,
}

impl SQLTool {
    pub fn new(path: &str) -> Result<Self, ToolError> {
        let conn =
            Connection::open(path).map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
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
        // Match FROM/JOIN followed by table names (possibly comma-separated)
        let re = Regex::new(r"(?i)\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)").unwrap();
        let mut tables = Vec::new();
        for cap in re.captures_iter(sql) {
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
        let lower = sql.trim().to_lowercase();
        if !lower.starts_with("select") {
            return Err(ToolError::InvalidInput(
                "Only SELECT queries are allowed (read-only)".to_string(),
            ));
        }

        // Table whitelist check: exact match on extracted table names
        if !self.allowed_tables.is_empty() {
            let tables_in_sql = Self::extract_table_names(sql);
            let allowed_lower: Vec<String> =
                self.allowed_tables.iter().map(|t| t.to_lowercase()).collect();
            let any_allowed = tables_in_sql
                .iter()
                .any(|t| allowed_lower.contains(t));
            if !any_allowed {
                return Err(ToolError::InvalidInput(format!(
                    "SQL does not reference any allowed table. Allowed: {:?}, found: {:?}",
                    self.allowed_tables, tables_in_sql
                )));
            }
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        let col_names: Vec<String> = stmt
            .column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
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
            conn.execute("INSERT INTO users VALUES (2, 'Bob')", []).unwrap();
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

        let tables = SQLTool::extract_table_names("SELECT * FROM users JOIN orders ON users.id = orders.user_id");
        assert_eq!(tables, vec!["users", "orders"]);

        let tables = SQLTool::extract_table_names("SELECT * FROM users, orders");
        assert_eq!(tables, vec!["users", "orders"]);
    }

    #[test]
    fn test_allowed_tables_with_join() {
        let tool = tool_with_data().with_allowed_tables(vec!["users".to_string()]);
        // JOIN with allowed table should pass
        assert!(tool.execute("SELECT * FROM users JOIN orders ON users.id = orders.user_id").is_ok());
    }
}
