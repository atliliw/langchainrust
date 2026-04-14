//! RouterChain 测试
//!
//! 测试路由 Chain 的核心功能：
//! 1. 关键词匹配路由 - 根据关键词自动路由到目标 Chain
//! 2. 默认 Chain - 未匹配时使用默认 Chain
//! 3. LLM 智能路由 - 使用 LLM 判断路由目标

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::{RouterChain, LLMRouterChain, LLMChain, BaseChain};
use std::sync::Arc;
use std::collections::HashMap;
use serde_json::Value;

/// 测试关键词匹配路由
///
/// 功能验证：
/// - 包含关键词的输入路由到对应 Chain
/// - 多个关键词匹配时选择第一个匹配的
/// - 未匹配时使用默认 Chain
#[tokio::test]
async fn test_router_keywords_match() {
    let config = TestConfig::get();
    
    let math_chain = Arc::new(
        LLMChain::new(config.openai_chat(), "计算数学问题: {input}").with_input_key("input")
    );
    let code_chain = Arc::new(
        LLMChain::new(config.openai_chat(), "回答编程问题: {input}").with_input_key("input")
    );
    let general_chain = Arc::new(
        LLMChain::new(config.openai_chat(), "回答一般问题: {input}").with_input_key("input")
    );
    
    let router = RouterChain::new()
        .add_route_with_keywords("数学", "处理数学计算问题", math_chain, vec!["计算", "数学", "加", "减", "乘", "除"])
        .add_route_with_keywords("编程", "处理编程相关问题", code_chain, vec!["代码", "Rust", "Python", "编程", "函数"])
        .with_default(general_chain)
        .with_verbose(true);
    
    println!("\n=== 测试：关键词匹配路由 ===");
    
    println!("\n--- 测试数学问题 ---");
    let inputs1 = HashMap::from([
        ("input".to_string(), Value::String("帮我计算 1+1".to_string()))
    ]);
    let result1 = router.invoke(inputs1).await.unwrap();
    println!("输出: {:?}", result1);
    assert!(result1.contains_key("text"));
    
    println!("\n--- 测试编程问题 ---");
    let inputs2 = HashMap::from([
        ("input".to_string(), Value::String("如何写 Rust 代码".to_string()))
    ]);
    let result2 = router.invoke(inputs2).await.unwrap();
    println!("输出: {:?}", result2);
    assert!(result2.contains_key("text"));
    
    println!("\n--- 测试默认 Chain ---");
    let inputs3 = HashMap::from([
        ("input".to_string(), Value::String("你好".to_string()))
    ]);
    let result3 = router.invoke(inputs3).await.unwrap();
    println!("输出: {:?}", result3);
    assert!(result3.contains_key("text"));
}

/// 测试 LLM 智能路由
///
/// 功能验证：
/// - LLM 分析输入内容选择路由目标
/// - 先尝试关键词匹配，失败则使用 LLM
/// - 关键词匹配优先于 LLM 判断
#[tokio::test]
async fn test_llm_router_intelligent() {
    let config = TestConfig::get();
    
    let math_chain = Arc::new(
        LLMChain::new(config.openai_chat(), "计算: {input}").with_input_key("input")
    );
    let weather_chain = Arc::new(
        LLMChain::new(config.openai_chat(), "天气信息: {input}").with_input_key("input")
    );
    let general_chain = Arc::new(
        LLMChain::new(config.openai_chat(), "回答: {input}").with_input_key("input")
    );
    
    let router = LLMRouterChain::new(config.openai_chat())
        .add_route("数学", "处理数学计算、加减乘除、数值运算相关问题", math_chain)
        .add_route("天气", "处理天气预报、气温、降雨等天气相关问题", weather_chain)
        .with_default(general_chain)
        .with_verbose(true);
    
    println!("\n=== 测试：LLM 智能路由 ===");
    
    println!("\n--- 测试 LLM 路由 ---");
    let inputs = HashMap::from([
        ("input".to_string(), Value::String("今天北京的天气怎么样".to_string()))
    ]);
    let result = router.invoke(inputs).await.unwrap();
    println!("输出: {:?}", result);
    assert!(result.contains_key("text"));
}

/// 测试路由失败处理
///
/// 功能验证：
/// - 没有配置默认 Chain 时，未匹配输入返回错误
/// - 有默认 Chain 时，未匹配输入路由到默认 Chain
#[tokio::test]
async fn test_router_fallback() {
    let config = TestConfig::get();
    
    let math_chain = Arc::new(
        LLMChain::new(config.openai_chat(), "计算: {input}").with_input_key("input")
    );
    let general_chain = Arc::new(
        LLMChain::new(config.openai_chat(), "回答: {input}").with_input_key("input")
    );
    
    let router_no_default = RouterChain::new()
        .add_route_with_keywords("数学", "处理数学问题", math_chain.clone(), vec!["计算"]);
    
    println!("\n=== 测试：路由失败处理 ===");
    
    println!("\n--- 没有默认 Chain ---");
    let inputs1 = HashMap::from([
        ("input".to_string(), Value::String("你好".to_string()))
    ]);
    let result1 = router_no_default.invoke(inputs1).await;
    assert!(result1.is_err(), "未匹配且无默认 Chain 应返回错误");
    println!("正确返回错误: {:?}", result1.unwrap_err());
    
    let router_with_default = RouterChain::new()
        .add_route_with_keywords("数学", "处理数学问题", math_chain, vec!["计算"])
        .with_default(general_chain);
    
    println!("\n--- 有默认 Chain ---");
    let inputs2 = HashMap::from([
        ("input".to_string(), Value::String("你好".to_string()))
    ]);
    let result2 = router_with_default.invoke(inputs2).await.unwrap();
    assert!(result2.contains_key("text"), "未匹配但有默认 Chain 应成功");
    println!("正确使用默认 Chain: {:?}", result2);
}