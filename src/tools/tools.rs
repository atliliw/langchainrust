use crate::tools::{Tool, ToolInput, ToolOutput};
use async_trait::async_trait;

pub struct Calculator;

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "执行基本数学运算，支持加减乘除"
    }

    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, Box<dyn std::error::Error>> {
        let expression = input
            .parameters
            .get("expression")
            .ok_or("缺少 expression 参数")?;

        let result = self.evaluate_expression(expression)?;

        Ok(ToolOutput {
            success: true,
            result,
        })
    }

    fn parameters(&self) -> Vec<(&str, &str)> {
        vec![("expression", "要计算的数学表达式，例如: 3+5 或 10/2")]
    }
    fn return_direct(&self) -> bool {
        false
    }
}

impl Calculator {
    fn evaluate_expression(&self, expression: &str) -> Result<String, Box<dyn std::error::Error>> {
        let expression = expression.trim().replace(" ", "");

        let op_pos = expression.find(&['+', '-', '*', '/'][..]);

        if let Some(pos) = op_pos {
            let operator = expression.chars().nth(pos).unwrap();
            let left_part = &expression[..pos];
            let right_part = &expression[pos + 1..];

            let left = left_part.parse::<f64>()?;
            let right = right_part.parse::<f64>()?;

            let result = match operator {
                '+' => left + right,
                '-' => left - right,
                '*' => left * right,
                '/' => {
                    if right == 0.0 {
                        return Err("除数不能为零".into());
                    }
                    left / right
                }
                _ => return Err("不支持的运算符".into()),
            };

            Ok(result.to_string())
        } else {
            Ok(expression.to_string())
        }
    }
}

pub struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str {
        "weather"
    }

    fn description(&self) -> &str {
        "获取指定城市的天气信息（模拟）"
    }

    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, Box<dyn std::error::Error>> {
        let city = input.parameters.get("city").ok_or("缺少 city 参数")?;

        let weather_info = format!("{}市当前天气：晴天，气温25℃，空气湿度60%", city);

        Ok(ToolOutput {
            success: true,
            result: weather_info,
        })
    }

    fn parameters(&self) -> Vec<(&str, &str)> {
        vec![("city", "城市名称，如：北京、上海")]
    }

    fn return_direct(&self) -> bool {
        false
    }
}

/// 日期时间工具
pub struct DateTimeTool;

#[async_trait]
impl Tool for DateTimeTool {
    fn name(&self) -> &str {
        "datetime"
    }

    fn description(&self) -> &str {
        "获取当前日期和时间"
    }

    async fn invoke(&self, _input: ToolInput) -> Result<ToolOutput, Box<dyn std::error::Error>> {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let duration = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let timestamp = duration.as_secs();
        
        let days = timestamp / 86400;
        let years = 1970 + days / 365;
        let remaining_days = days % 365;
        let months = remaining_days / 30 + 1;
        let day = remaining_days % 30 + 1;
        
        let hours = (timestamp % 86400) / 3600;
        let minutes = (timestamp % 3600) / 60;
        let seconds = timestamp % 60;
        
        let result = format!(
            "当前时间：{}年{}月{}日 {:02}:{:02}:{:02}（北京时间约为8点）",
            years, months, day, hours + 8, minutes, seconds
        );

        Ok(ToolOutput {
            success: true,
            result,
        })
    }

    fn parameters(&self) -> Vec<(&str, &str)> {
        vec![]
    }

    fn return_direct(&self) -> bool {
        false
    }
}

/// 文本处理工具
pub struct TextTool;

#[async_trait]
impl Tool for TextTool {
    fn name(&self) -> &str {
        "text"
    }

    fn description(&self) -> &str {
        "文本处理工具，支持：word_count（字数统计）、upper（转大写）、lower（转小写）、reverse（反转）"
    }

    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, Box<dyn std::error::Error>> {
        let operation = input.parameters.get("operation").map(|s| s.as_str()).unwrap_or("word_count");
        let text = input.parameters.get("text").ok_or("缺少 text 参数")?;

        let result = match operation {
            "word_count" | "count" => {
                let char_count = text.chars().count();
                let word_count = text.split_whitespace().count();
                format!("字符数：{}，词数：{}", char_count, word_count)
            }
            "upper" | "uppercase" => text.to_uppercase(),
            "lower" | "lowercase" => text.to_lowercase(),
            "reverse" => text.chars().rev().collect::<String>(),
            _ => format!("未知操作：{}，支持的操作：word_count, upper, lower, reverse", operation),
        };

        Ok(ToolOutput {
            success: true,
            result,
        })
    }

    fn parameters(&self) -> Vec<(&str, &str)> {
        vec![
            ("operation", "操作类型：word_count（字数统计）、upper（转大写）、lower（转小写）、reverse（反转）"),
            ("text", "要处理的文本内容"),
        ]
    }

    fn return_direct(&self) -> bool {
        false
    }
}

/// 网络搜索工具（模拟）
pub struct WebSearchTool;

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "模拟网络搜索，返回相关搜索结果"
    }

    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, Box<dyn std::error::Error>> {
        let query = input.parameters.get("query").ok_or("缺少 query 参数")?;

        let results = format!(
            "搜索结果（模拟）：\n\
            1. {} - 相关百科条目\n\
            2. {} - 最新新闻动态\n\
            3. {} - 相关技术文章\n\
            \n提示：这是一个模拟搜索结果，实际使用时请接入真实搜索API",
            query, query, query
        );

        Ok(ToolOutput {
            success: true,
            result: results,
        })
    }

    fn parameters(&self) -> Vec<(&str, &str)> {
        vec![("query", "搜索关键词")]
    }

    fn return_direct(&self) -> bool {
        false
    }
}

/// JSON 解析工具
pub struct JsonTool;

#[async_trait]
impl Tool for JsonTool {
    fn name(&self) -> &str {
        "json"
    }

    fn description(&self) -> &str {
        "JSON 工具，支持：parse（解析）、format（格式化）、extract（提取字段）"
    }

    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, Box<dyn std::error::Error>> {
        let operation = input.parameters.get("operation").map(|s| s.as_str()).unwrap_or("parse");
        let data = input.parameters.get("data").ok_or("缺少 data 参数")?;

        let result = match operation {
            "parse" | "validate" => {
                match serde_json::from_str::<serde_json::Value>(data) {
                    Ok(_) => "JSON 格式有效".to_string(),
                    Err(e) => format!("JSON 解析失败：{}", e),
                }
            }
            "format" | "prettify" => {
                match serde_json::from_str::<serde_json::Value>(data) {
                    Ok(v) => serde_json::to_string_pretty(&v)?,
                    Err(e) => format!("JSON 解析失败：{}", e),
                }
            }
            "extract" => {
                let key = input.parameters.get("key").ok_or("extract 操作需要 key 参数")?;
                match serde_json::from_str::<serde_json::Value>(data) {
                    Ok(v) => {
                        if let Some(value) = v.get(key) {
                            format!("{}: {}", key, value)
                        } else {
                            format!("未找到字段：{}", key)
                        }
                    }
                    Err(e) => format!("JSON 解析失败：{}", e),
                }
            }
            _ => format!("未知操作：{}，支持的操作：parse, format, extract", operation),
        };

        Ok(ToolOutput {
            success: true,
            result,
        })
    }

    fn parameters(&self) -> Vec<(&str, &str)> {
        vec![
            ("operation", "操作类型：parse（验证）、format（格式化）、extract（提取字段）"),
            ("data", "JSON 字符串"),
            ("key", "要提取的字段名（仅 extract 操作需要）"),
        ]
    }

    fn return_direct(&self) -> bool {
        false
    }
}
