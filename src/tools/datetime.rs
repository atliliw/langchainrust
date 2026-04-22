// src/tools/datetime.rs
//! Date and time tool for agents.
//!
//! Provides date/time query and calculation functionality.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Local, NaiveDate, NaiveDateTime, Duration, Datelike, Weekday, TimeZone};

use crate::core::tools::{BaseTool, Tool, ToolError};

/// DateTime tool input parameters.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DateTimeInput {
    /// Operation type: "now", "format", "add", "subtract", "weekday", "diff".
    pub operation: String,
    
    /// Date/time string (optional, format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).
    pub datetime: Option<String>,
    
    /// Time unit: "days", "hours", "minutes", "weeks", "months", "years".
    pub unit: Option<String>,
    
    /// Value for add/subtract operations.
    pub value: Option<i64>,
    
    /// Target date/time for diff operation.
    pub target: Option<String>,
}

/// DateTime tool output result.
#[derive(Debug, Serialize)]
pub struct DateTimeOutput {
    /// Operation result.
    pub result: String,
    
    /// Operation type.
    pub operation: String,
    
    /// Additional details.
    pub details: Option<String>,
}

/// DateTime tool for querying and manipulating dates and times.
pub struct DateTimeTool;

impl DateTimeTool {
    /// Creates a new DateTimeTool instance.
    pub fn new() -> Self {
        Self
    }
    
    /// 解析日期时间字符串
    fn parse_datetime(&self, dt_str: &str) -> Result<DateTime<Local>, ToolError> {
        // 尝试解析完整日期时间
        if let Ok(dt) = NaiveDateTime::parse_from_str(dt_str, "%Y-%m-%d %H:%M:%S") {
            return Local.from_local_datetime(&dt).single()
                .ok_or_else(|| ToolError::ExecutionFailed("无效的日期时间".to_string()));
        }
        
        // 尝试解析日期
        if let Ok(date) = NaiveDate::parse_from_str(dt_str, "%Y-%m-%d") {
            let dt = date.and_hms_opt(0, 0, 0)
                .ok_or_else(|| ToolError::ExecutionFailed("无效的日期".to_string()))?;
            return Local.from_local_datetime(&dt).single()
                .ok_or_else(|| ToolError::ExecutionFailed("无效的日期时间".to_string()));
        }
        
        Err(ToolError::ExecutionFailed(
            format!("无法解析日期时间: {}，请使用格式 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS", dt_str)
        ))
    }
    
    /// 获取当前时间
    fn get_now(&self) -> DateTimeOutput {
        let now = Local::now();
        DateTimeOutput {
            result: now.format("%Y-%m-%d %H:%M:%S").to_string(),
            operation: "now".to_string(),
            details: Some(format!(
                "星期{}，第{}周",
                match now.weekday() {
                    Weekday::Mon => "一",
                    Weekday::Tue => "二",
                    Weekday::Wed => "三",
                    Weekday::Thu => "四",
                    Weekday::Fri => "五",
                    Weekday::Sat => "六",
                    Weekday::Sun => "日",
                },
                now.iso_week().week()
            )),
        }
    }
    
    /// 格式化日期时间
    fn format_datetime(&self, dt_str: &str) -> Result<DateTimeOutput, ToolError> {
        let dt = self.parse_datetime(dt_str)?;
        
        Ok(DateTimeOutput {
            result: dt.format("%Y年%m月%d日 %H时%M分%S秒").to_string(),
            operation: "format".to_string(),
            details: Some(format!(
                "星期{}，{}年{}周",
                match dt.weekday() {
                    Weekday::Mon => "一",
                    Weekday::Tue => "二",
                    Weekday::Wed => "三",
                    Weekday::Thu => "四",
                    Weekday::Fri => "五",
                    Weekday::Sat => "六",
                    Weekday::Sun => "日",
                },
                dt.year(),
                dt.iso_week().week()
            )),
        })
    }
    
    /// 添加时间
    fn add_time(
        &self,
        dt_str: &str,
        value: i64,
        unit: &str,
    ) -> Result<DateTimeOutput, ToolError> {
        let dt = self.parse_datetime(dt_str)?;
        
        let new_dt = match unit {
            "seconds" => dt + Duration::seconds(value),
            "minutes" => dt + Duration::minutes(value),
            "hours" => dt + Duration::hours(value),
            "days" => dt + Duration::days(value),
            "weeks" => dt + Duration::weeks(value),
            "months" => {
                // 简化的月份计算
                let months = value as i32;
                let new_month = dt.month() as i32 + months;
                let year = dt.year() + (new_month - 1) / 12;
                let month = ((new_month - 1) % 12 + 1) as u32;
                
                dt.with_year(year)
                    .and_then(|d| d.with_month(month))
                    .ok_or_else(|| ToolError::ExecutionFailed("月份计算失败".to_string()))?
            }
            "years" => {
                dt.with_year(dt.year() + value as i32)
                    .ok_or_else(|| ToolError::ExecutionFailed("年份计算失败".to_string()))?
            }
            _ => return Err(ToolError::ExecutionFailed(
                format!("不支持的时间单位: {}，请使用: seconds, minutes, hours, days, weeks, months, years", unit)
            )),
        };
        
        Ok(DateTimeOutput {
            result: new_dt.format("%Y-%m-%d %H:%M:%S").to_string(),
            operation: "add".to_string(),
            details: Some(format!("{} {} 后", value, self.unit_to_chinese(unit))),
        })
    }
    
    /// 减去时间
    fn subtract_time(
        &self,
        dt_str: &str,
        value: i64,
        unit: &str,
    ) -> Result<DateTimeOutput, ToolError> {
        self.add_time(dt_str, -value, unit)
    }
    
    /// 获取星期几
    fn get_weekday(&self, dt_str: &str) -> Result<DateTimeOutput, ToolError> {
        let dt = self.parse_datetime(dt_str)?;
        
        let weekday = match dt.weekday() {
            Weekday::Mon => "星期一",
            Weekday::Tue => "星期二",
            Weekday::Wed => "星期三",
            Weekday::Thu => "星期四",
            Weekday::Fri => "星期五",
            Weekday::Sat => "星期六",
            Weekday::Sun => "星期日",
        };
        
        Ok(DateTimeOutput {
            result: weekday.to_string(),
            operation: "weekday".to_string(),
            details: Some(format!("{} 是 {}", dt.format("%Y-%m-%d"), weekday)),
        })
    }
    
    /// 计算时间差
    fn diff_time(
        &self,
        dt1_str: &str,
        dt2_str: &str,
    ) -> Result<DateTimeOutput, ToolError> {
        let dt1 = self.parse_datetime(dt1_str)?;
        let dt2 = self.parse_datetime(dt2_str)?;
        
        let diff = dt2.signed_duration_since(dt1);
        
        let days = diff.num_days();
        let hours = diff.num_hours() % 24;
        let minutes = diff.num_minutes() % 60;
        
        Ok(DateTimeOutput {
            result: format!("{}天 {}小时 {}分钟", days.abs(), hours.abs(), minutes.abs()),
            operation: "diff".to_string(),
            details: Some(if diff.num_seconds() >= 0 {
                format!("从 {} 到 {} 相隔 {}天 {}小时 {}分钟",
                    dt1.format("%Y-%m-%d"), dt2.format("%Y-%m-%d"),
                    days, hours, minutes)
            } else {
                format!("从 {} 到 {} 相隔 {}天 {}小时 {}分钟",
                    dt2.format("%Y-%m-%d"), dt1.format("%Y-%m-%d"),
                    days.abs(), hours.abs(), minutes.abs())
            }),
        })
    }
    
    /// 单位转中文
    fn unit_to_chinese(&self, unit: &str) -> String {
        match unit {
            "seconds" => "秒",
            "minutes" => "分钟",
            "hours" => "小时",
            "days" => "天",
            "weeks" => "周",
            "months" => "月",
            "years" => "年",
            _ => unit,
        }.to_string()
    }
}

impl Default for DateTimeTool {
    fn default() -> Self {
        Self::new()
    }
}

/// 实现 Tool trait
#[async_trait]
impl Tool for DateTimeTool {
    type Input = DateTimeInput;
    type Output = DateTimeOutput;
    
    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        match input.operation.as_str() {
            "now" => Ok(self.get_now()),
            "format" => {
                let dt = input.datetime.ok_or_else(|| 
                    ToolError::InvalidInput("format 操作需要 datetime 参数".to_string()))?;
                self.format_datetime(&dt)
            }
            "add" => {
                let dt = input.datetime.ok_or_else(|| 
                    ToolError::InvalidInput("add 操作需要 datetime 参数".to_string()))?;
                let value = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("add 操作需要 value 参数".to_string()))?;
                let unit = input.unit.ok_or_else(|| 
                    ToolError::InvalidInput("add 操作需要 unit 参数".to_string()))?;
                self.add_time(&dt, value, &unit)
            }
            "subtract" => {
                let dt = input.datetime.ok_or_else(|| 
                    ToolError::InvalidInput("subtract 操作需要 datetime 参数".to_string()))?;
                let value = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("subtract 操作需要 value 参数".to_string()))?;
                let unit = input.unit.ok_or_else(|| 
                    ToolError::InvalidInput("subtract 操作需要 unit 参数".to_string()))?;
                self.subtract_time(&dt, value, &unit)
            }
            "weekday" => {
                let dt = input.datetime.ok_or_else(|| 
                    ToolError::InvalidInput("weekday 操作需要 datetime 参数".to_string()))?;
                self.get_weekday(&dt)
            }
            "diff" => {
                let dt1 = input.datetime.ok_or_else(|| 
                    ToolError::InvalidInput("diff 操作需要 datetime 参数".to_string()))?;
                let dt2 = input.target.ok_or_else(|| 
                    ToolError::InvalidInput("diff 操作需要 target 参数".to_string()))?;
                self.diff_time(&dt1, &dt2)
            }
            _ => Err(ToolError::InvalidInput(
                format!("不支持的操作: {}，请使用: now, format, add, subtract, weekday, diff", input.operation)
            )),
        }
    }
}

/// 实现 BaseTool trait
#[async_trait]
impl BaseTool for DateTimeTool {
    fn name(&self) -> &str {
        "datetime"
    }
    
    fn description(&self) -> &str {
        "日期时间工具。支持多种操作：
        
操作类型:
- now: 获取当前时间
- format: 格式化日期时间
- add: 添加时间
- subtract: 减去时间
- weekday: 获取星期几
- diff: 计算时间差

示例:
- 获取当前时间: {\"operation\": \"now\"}
- 格式化日期: {\"operation\": \"format\", \"datetime\": \"2024-01-15\"}
- 添加3天: {\"operation\": \"add\", \"datetime\": \"2024-01-15\", \"value\": 3, \"unit\": \"days\"}
- 计算差值: {\"operation\": \"diff\", \"datetime\": \"2024-01-01\", \"target\": \"2024-01-15\"}"
    }
    
    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: DateTimeInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON 解析失败: {}", e)))?;
        
        let output = self.invoke(parsed).await?;
        
        Ok(format!(
            "{}\n详细信息: {}",
            output.result,
            output.details.unwrap_or_default()
        ))
    }
    
    fn args_schema(&self) -> Option<serde_json::Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(DateTimeInput)).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_datetime_now() {
        let tool = DateTimeTool::new();
        
        let input = DateTimeInput {
            operation: "now".to_string(),
            datetime: None,
            unit: None,
            value: None,
            target: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert!(!result.result.is_empty());
        assert!(result.details.is_some());
    }
    
    #[tokio::test]
    async fn test_datetime_format() {
        let tool = DateTimeTool::new();
        
        let input = DateTimeInput {
            operation: "format".to_string(),
            datetime: Some("2024-01-15".to_string()),
            unit: None,
            value: None,
            target: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert!(result.result.contains("2024年"));
        assert!(result.result.contains("01月"));
        assert!(result.result.contains("15日"));
    }
    
    #[tokio::test]
    async fn test_datetime_add_days() {
        let tool = DateTimeTool::new();
        
        let input = DateTimeInput {
            operation: "add".to_string(),
            datetime: Some("2024-01-15".to_string()),
            unit: Some("days".to_string()),
            value: Some(3),
            target: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert!(result.result.contains("2024-01-18"));
    }
    
    #[tokio::test]
    async fn test_datetime_weekday() {
        let tool = DateTimeTool::new();
        
        // 2024-01-15 是星期一
        let input = DateTimeInput {
            operation: "weekday".to_string(),
            datetime: Some("2024-01-15".to_string()),
            unit: None,
            value: None,
            target: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert_eq!(result.result, "星期一");
    }
    
    #[tokio::test]
    async fn test_datetime_diff() {
        let tool = DateTimeTool::new();
        
        let input = DateTimeInput {
            operation: "diff".to_string(),
            datetime: Some("2024-01-01".to_string()),
            unit: None,
            value: None,
            target: Some("2024-01-15".to_string()),
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert!(result.result.contains("14天"));
    }
}