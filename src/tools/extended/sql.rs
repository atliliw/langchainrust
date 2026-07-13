//! SQL 工具(只读,SQLite)

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use rusqlite::Connection;

use crate::core::tools::ToolError;
use crate::BaseTool;

/// SQL 查询工具(只读 SELECT,表白名单)
pub struct SQLTool {
    conn: Mutex<Connection>,
    allowed_tables: Vec<String>,
}

impl SQLTool {
    pub fn new(path: &str) -> Result<Self, ToolError> {
        let conn = Connection::open(path).map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        Ok(Self {
            conn: Mutex::new(conn),
            allowed_tables: Vec::new(),
        })
    }

    pub fn with_allowed_tables(mut self, tables: Vec<String>) -> Self {
        self.allowed_tables = tables;
        self
    }

    /// 执行 SELECT 查询(只读)
    pub fn execute(&self, sql: &str) -> Result<Vec<HashMap<String, String>>, ToolError> {
        let lower = sql.trim().to_lowercase();
        if !lower.starts_with("select") {
            return Err(ToolError::InvalidInput("只允许 SELECT(只读)".to_string()));
        }

        // 表白名单检查(简单 contains)
        if !self.allowed_tables.is_empty() {
            let any_allowed = self
                .allowed_tables
                .iter()
                .any(|t| lower.contains(&t.to_lowercase()));
            if !any_allowed {
                return Err(ToolError::InvalidInput(format!(
                    "SQL 未涉及允许的表: {:?}",
                    self.allowed_tables
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
                    let val: String = row.get::<_, Option<String>>(i).unwrap_or(None).unwrap_or_default();
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
        "执行 SQL SELECT 查询(只读)。输入 SQL 字符串。"
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
    fn test_allowed_tables() {
        let tool = tool_with_data().with_allowed_tables(vec!["users".to_string()]);
        assert!(tool.execute("SELECT * FROM users").is_ok());
        // 查询不存在的表(涉及允许表名但表不存在)-> SQL 错误
        assert!(tool.execute("SELECT * FROM users2").is_err());
    }

    #[test]
    fn test_allowed_tables_blocks_unknown() {
        let tool = tool_with_data().with_allowed_tables(vec!["orders".to_string()]);
        // SQL 不含允许的表名
        assert!(tool.execute("SELECT * FROM users").is_err());
    }
}
