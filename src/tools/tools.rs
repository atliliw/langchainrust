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
        let expression = input.parameters.get("expression")
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
}

impl Calculator {
    fn evaluate_expression(&self, expression: &str) -> Result<String, Box<dyn std::error::Error>> {
        let expression = expression.trim().replace(" ", "");
        
        // 找到第一个运算符
        let op_pos = expression.find(&['+', '-', '*', '/'][..]);
        
        if let Some(pos) = op_pos {
            let operator = expression.chars().nth(pos).unwrap();
            
            // 安全地分割字符串
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
            // 如果没有运算符，直接返回数字
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
        let city = input.parameters.get("city")
            .ok_or("缺少 city 参数")?;
        
        let weather_info = format!("{}市当前天气：晴天，气温25℃，空气湿度60%", city);
        
        Ok(ToolOutput {
            success: true,
            result: weather_info,
        })
    }
    
    fn parameters(&self) -> Vec<(&str, &str)> {
        vec![("city", "城市名称，如：北京、上海")]
    }
}