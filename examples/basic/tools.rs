// examples/basic/tools.rs
//! 基础示例 4: 使用工具
//!
//! 运行: cargo run --example tools
//!
//! 功能: 演示如何直接调用内置工具

use langchainrust::{
    Calculator, DateTimeTool, SimpleMathTool, URLFetchTool,
    BaseTool,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 基础示例 4: 使用工具 ===\n");
    
    // ========== 1. Calculator 计算器 ==========
    println!("--- 1. Calculator 计算器 ---\n");
    
    let calc = Calculator::new();
    
    let result = calc.run(r#"{"expression": "123 + 456"}"#.to_string()).await?;
    println!("计算 123 + 456: {}", result);
    
    let result = calc.run(r#"{"expression": "100 - 37"}"#.to_string()).await?;
    println!("计算 100 - 37: {}", result);
    
    // ========== 2. DateTimeTool 日期时间 ==========
    println!("\n--- 2. DateTimeTool 日期时间 ---\n");
    
    let datetime = DateTimeTool::new();
    
    // 获取当前时间
    let result = datetime.run(r#"{"operation": "now"}"#.to_string()).await?;
    println!("当前时间: {}", result);
    
    // 获取星期几
    let result = datetime.run(r#"{"operation": "weekday", "datetime": "2024-01-01"}"#.to_string()).await?;
    println!("2024-01-01 是: {}", result);
    
    // 日期加减
    let result = datetime.run(r#"{"operation": "add", "datetime": "2024-01-01", "value": 7, "unit": "days"}"#.to_string()).await?;
    println!("2024-01-01 加 7 天: {}", result);
    
    // ========== 3. SimpleMathTool 高级数学 ==========
    println!("\n--- 3. SimpleMathTool 高级数学 ---\n");
    
    let math = SimpleMathTool::new();
    
    // 幂运算
    let result = math.run(r#"{"operation": "power", "value": 2, "value2": 10}"#.to_string()).await?;
    println!("2 的 10 次方: {}", result);
    
    // 平方根
    let result = math.run(r#"{"operation": "sqrt", "value": 144}"#.to_string()).await?;
    println!("144 的平方根: {}", result);
    
    // 阶乘
    let result = math.run(r#"{"operation": "factorial", "value": 5}"#.to_string()).await?;
    println!("5 的阶乘: {}", result);
    
    // GCD
    let result = math.run(r#"{"operation": "gcd", "value": 48, "value2": 18}"#.to_string()).await?;
    println!("48 和 18 的最大公约数: {}", result);
    
    // 三角函数
    let result = math.run(r#"{"operation": "sin", "value": 1.5708}"#.to_string()).await?;
    println!("sin(π/2): {}", result);
    
    // ========== 4. URLFetchTool 网页抓取 (需要网络) ==========
    println!("\n--- 4. URLFetchTool 网页抓取 ---\n");
    
    let url_fetch = URLFetchTool::new();
    
    // 提取元数据
    match url_fetch.run(r#"{"operation": "metadata", "url": "https://example.com"}"#.to_string()).await {
        Ok(result) => println!("example.com 元数据:\n{}", result),
        Err(e) => println!("跳过网络请求: {}", e),
    }
    
    println!("\n=== 示例完成 ===");
    Ok(())
}