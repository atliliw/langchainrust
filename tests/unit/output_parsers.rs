//! 输出解析器单元测试
//!
//! 测试将 LLM 原始输出文本解析为结构化数据的功能。
//! 包含以下解析器的测试：
//! - StrOutputParser: 字符串透传
//! - CommaSeparatedListOutputParser: 逗号分隔列表
//! - JsonOutputParser: JSON 解析
//! - StructuredOutputParser: 键值对解析
//! - TypedOutputParser: 类型化 JSON 解析
//! - Runnable 接口兼容性

use langchainrust::{
    BaseOutputParser, OutputParserError,
    StrOutputParser, CommaSeparatedListOutputParser,
    JsonOutputParser, StructuredOutputParser, TypedOutputParser,
};
use langchainrust::core::runnables::Runnable;
use serde::Deserialize;
use std::collections::HashMap;

// ============================================================
// StrOutputParser 测试
// ============================================================

/// 测试 StrOutputParser 基础功能
///
/// 验证：解析器直接将输入文本原样返回，不做任何转换。
#[tokio::test]
async fn test_str_parser_basic() {
    let parser = StrOutputParser::new();
    let result = parser.parse("Hello, world!").await.unwrap();
    assert_eq!(result, "Hello, world!");
}

/// 测试 StrOutputParser 处理空字符串
///
/// 验证：空字符串也能正常透传返回。
#[tokio::test]
async fn test_str_parser_empty() {
    let parser = StrOutputParser::new();
    let result = parser.parse("").await.unwrap();
    assert_eq!(result, "");
}

/// 测试 StrOutputParser 处理多行文本
///
/// 验证：多行文本保持原样返回。
#[tokio::test]
async fn test_str_parser_multiline() {
    let parser = StrOutputParser::new();
    let text = "line1\nline2\nline3";
    let result = parser.parse(text).await.unwrap();
    assert_eq!(result, text);
}

/// 测试 StrOutputParser 的 Runnable 接口
///
/// 验证：通过 invoke 方法调用也能得到相同结果。
#[tokio::test]
async fn test_str_parser_as_runnable() {
    let parser = StrOutputParser::new();
    let result = parser.invoke("test".to_string(), None).await.unwrap();
    assert_eq!(result, "test");
}

/// 测试 StrOutputParser 的 Default 实现
#[tokio::test]
async fn test_str_parser_default() {
    let parser: StrOutputParser = Default::default();
    let result = parser.parse("default").await.unwrap();
    assert_eq!(result, "default");
}

// ============================================================
// CommaSeparatedListOutputParser 测试
// ============================================================

/// 测试 CommaSeparatedListOutputParser 基础功能
///
/// 验证：英文逗号分隔的文本被正确解析为字符串列表。
#[tokio::test]
async fn test_list_parser_basic() {
    let parser = CommaSeparatedListOutputParser::new();
    let result = parser.parse("apple, banana, cherry").await.unwrap();
    assert_eq!(result, vec!["apple", "banana", "cherry"]);
}

/// 测试 CommaSeparatedListOutputParser 支持中文逗号
///
/// 验证：中文逗号（，）也被识别为分隔符。
#[tokio::test]
async fn test_list_parser_chinese_comma() {
    let parser = CommaSeparatedListOutputParser::new();
    let result = parser.parse("苹果，香蕉，樱桃").await.unwrap();
    assert_eq!(result, vec!["苹果", "香蕉", "樱桃"]);
}

/// 测试 CommaSeparatedListOutputParser 处理空输入
///
/// 验证：空字符串返回空列表而非错误。
#[tokio::test]
async fn test_list_parser_empty() {
    let parser = CommaSeparatedListOutputParser::new();
    let result = parser.parse("").await.unwrap();
    assert!(result.is_empty());
}

/// 测试 CommaSeparatedListOutputParser 去除空白
///
/// 验证：每个项目的前后空白被自动去除。
#[tokio::test]
async fn test_list_parser_trim_whitespace() {
    let parser = CommaSeparatedListOutputParser::new();
    let result = parser.parse("  a  ,  b  ,  c  ").await.unwrap();
    assert_eq!(result, vec!["a", "b", "c"]);
}

/// 测试 CommaSeparatedListOutputParser 的 Runnable 接口
#[tokio::test]
async fn test_list_parser_as_runnable() {
    let parser = CommaSeparatedListOutputParser::new();
    let result = parser.invoke("x, y, z".to_string(), None).await.unwrap();
    assert_eq!(result, vec!["x", "y", "z"]);
}

/// 测试 CommaSeparatedListOutputParser 的格式指令
///
/// 验证：格式指令不为空且包含有用信息。
#[tokio::test]
async fn test_list_parser_format_instructions() {
    let parser = CommaSeparatedListOutputParser::new();
    let instructions = parser.get_format_instructions();
    assert!(!instructions.is_empty());
    assert!(instructions.contains("逗号"));
}

// ============================================================
// JsonOutputParser 测试
// ============================================================

/// 测试 JsonOutputParser 基础 JSON 对象解析
///
/// 验证：标准 JSON 对象能正确解析为 serde_json::Value。
#[tokio::test]
async fn test_json_parser_basic_object() {
    let parser = JsonOutputParser::new();
    let result = parser.parse(r#"{"name": "Rust", "year": 2015}"#).await.unwrap();
    assert_eq!(result["name"], "Rust");
    assert_eq!(result["year"], 2015);
}

/// 测试 JsonOutputParser 解析 JSON 数组
#[tokio::test]
async fn test_json_parser_array() {
    let parser = JsonOutputParser::new();
    let result = parser.parse("[1, 2, 3]").await.unwrap();
    assert_eq!(result[0], 1);
    assert_eq!(result[2], 3);
}

/// 测试 JsonOutputParser 从 Markdown 代码块提取 JSON
///
/// 验证：能正确提取 ```json ... ``` 代码块中的 JSON。
#[tokio::test]
async fn test_json_parser_from_markdown() {
    let parser = JsonOutputParser::new();
    let input = "以下是结果：\n```json\n{\"status\": \"ok\", \"code\": 200}\n```\n";
    let result = parser.parse(input).await.unwrap();
    assert_eq!(result["status"], "ok");
    assert_eq!(result["code"], 200);
}

/// 测试 JsonOutputParser 处理非法 JSON
///
/// 验证：非法 JSON 输入返回 ParseError。
#[tokio::test]
async fn test_json_parser_invalid_input() {
    let parser = JsonOutputParser::new();
    let result = parser.parse("{invalid json}").await;
    assert!(result.is_err());
    match result {
        Err(OutputParserError::JsonError(_)) => {} // 期望 JSON 错误
        _ => panic!("应当返回 JsonError"),
    }
}

/// 测试 JsonOutputParser 处理普通文本（无 JSON）
///
/// 验证：不含 JSON 的文本也返回错误。
#[tokio::test]
async fn test_json_parser_plain_text() {
    let parser = JsonOutputParser::new();
    let result = parser.parse("Hello, this is plain text").await;
    assert!(result.is_err());
}

/// 测试 JsonOutputParser 的 Runnable 接口
#[tokio::test]
async fn test_json_parser_as_runnable() {
    let parser = JsonOutputParser::new();
    let result = parser.invoke(r#"{"key": "value"}"#.to_string(), None).await.unwrap();
    assert_eq!(result["key"], "value");
}

/// 测试 JsonOutputParser 的部分解析模式
///
/// 验证：partial 模式下也能解析完整 JSON。
#[tokio::test]
async fn test_json_parser_partial_complete_json() {
    let parser = JsonOutputParser::new_partial();
    let result = parser.parse(r#"{"a": 1, "b": 2}"#).await.unwrap();
    assert_eq!(result["a"], 1);
    assert_eq!(result["b"], 2);
}

/// 测试 JsonOutputParser 的格式指令
#[tokio::test]
async fn test_json_parser_format_instructions() {
    let parser = JsonOutputParser::new();
    let instructions = parser.get_format_instructions();
    assert!(!instructions.is_empty());
    assert!(instructions.contains("JSON"));
}

// ============================================================
// StructuredOutputParser 测试
// ============================================================

/// 测试 StructuredOutputParser 基础功能
///
/// 验证：标准 key: value 格式能被正确解析为 HashMap。
#[tokio::test]
async fn test_structured_parser_basic() {
    let parser = StructuredOutputParser::new();
    let input = "name: Rust\n year: 2015\n type: systems";
    let result = parser.parse(input).await.unwrap();
    assert_eq!(result.get("name").unwrap(), "Rust");
    assert_eq!(result.get("year").unwrap(), "2015");
    assert_eq!(result.get("type").unwrap(), "systems");
}

/// 测试 StructuredOutputParser 处理空行
///
/// 验证：空行被自动跳过，不影响结果。
#[tokio::test]
async fn test_structured_parser_skip_empty_lines() {
    let parser = StructuredOutputParser::new();
    let input = "a: 1\n\nb: 2\n\n\nc: 3";
    let result = parser.parse(input).await.unwrap();
    assert_eq!(result.len(), 3);
}

/// 测试 StructuredOutputParser 使用自定义分隔符
///
/// 验证：可以通过 with_separator 指定不同的分隔符。
#[tokio::test]
async fn test_structured_parser_custom_separator() {
    let parser = StructuredOutputParser::with_separator('=');
    let input = "name= Rust\n version= 2024";
    let result = parser.parse(input).await.unwrap();
    assert_eq!(result.get("name").unwrap(), "Rust");
    assert_eq!(result.get("version").unwrap(), "2024");
}

/// 测试 StructuredOutputParser 处理空输入
#[tokio::test]
async fn test_structured_parser_empty_input() {
    let parser = StructuredOutputParser::new();
    let result = parser.parse("").await.unwrap();
    assert!(result.is_empty());
}

/// 测试 StructuredOutputParser 处理无分隔符的行
///
/// 验证：没有分隔符的行被忽略。
#[tokio::test]
async fn test_structured_parser_skip_lines_without_separator() {
    let parser = StructuredOutputParser::new();
    let input = "a: 1\nthis line has no separator\nb: 2";
    let result = parser.parse(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result.get("a").unwrap(), "1");
    assert_eq!(result.get("b").unwrap(), "2");
}

/// 测试 StructuredOutputParser 的 Runnable 接口
#[tokio::test]
async fn test_structured_parser_as_runnable() {
    let parser = StructuredOutputParser::new();
    let result = parser.invoke("x: 1\ny: 2".to_string(), None).await.unwrap();
    let map: HashMap<String, String> = result;
    assert_eq!(map.get("x").unwrap(), "1");
}

/// 测试 StructuredOutputParser 的格式指令
#[tokio::test]
async fn test_structured_parser_format_instructions() {
    let parser = StructuredOutputParser::new();
    let instructions = parser.get_format_instructions();
    assert!(!instructions.is_empty());
    assert!(instructions.contains(":"));
}

// ============================================================
// TypedOutputParser 测试
// ============================================================

/// 测试 TypedOutputParser 基础反序列化
///
/// 验证：JSON 字符串能被正确反序列化为目标 Rust 结构体。
#[tokio::test]
async fn test_typed_parser_basic() {
    #[derive(Deserialize, Debug, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    let parser = TypedOutputParser::<Person>::new();
    let person = parser.parse(r#"{"name": "Alice", "age": 30}"#).await.unwrap();
    assert_eq!(person.name, "Alice");
    assert_eq!(person.age, 30);
}

/// 测试 TypedOutputParser 从 Markdown 代码块提取
///
/// 验证：能从 Markdown 代码块中提取 JSON 并反序列化。
#[tokio::test]
async fn test_typed_parser_from_markdown() {
    #[derive(Deserialize, Debug, PartialEq)]
    struct Response {
        result: String,
        score: f64,
    }

    let parser = TypedOutputParser::<Response>::new();
    let input = "```json\n{\"result\": \"success\", \"score\": 0.95}\n```";
    let response = parser.parse(input).await.unwrap();
    assert_eq!(response.result, "success");
    assert_eq!(response.score, 0.95);
}

/// 测试 TypedOutputParser 类型不匹配错误
///
/// 验证：当 JSON 字段与结构体不匹配时返回 TypeError。
#[tokio::test]
async fn test_typed_parser_type_mismatch() {
    #[derive(Deserialize, Debug)]
    struct Point {
        x: i32,
        y: i32,
    }

    let parser = TypedOutputParser::<Point>::new();
    // 缺少 y 字段
    let result = parser.parse(r#"{"x": 10}"#).await;
    assert!(result.is_err());
    match result {
        Err(OutputParserError::TypeError(_)) => {} // 期望类型错误
        _ => panic!("应当返回 TypeError"),
    }
}

/// 测试 TypedOutputParser 处理非法 JSON
#[tokio::test]
async fn test_typed_parser_invalid_json() {
    #[derive(Deserialize, Debug)]
    struct Data {
        value: String,
    }

    let parser = TypedOutputParser::<Data>::new();
    let result = parser.parse("not json at all").await;
    assert!(result.is_err());
    match result {
        Err(OutputParserError::JsonError(_)) => {} // 期望 JSON 错误
        _ => panic!("应当返回 JsonError"),
    }
}

/// 测试 TypedOutputParser 的 Runnable 接口
#[tokio::test]
async fn test_typed_parser_as_runnable() {
    #[derive(Deserialize, Debug, PartialEq)]
    struct Config {
        host: String,
        port: u16,
    }

    let parser = TypedOutputParser::<Config>::new();
    let config = parser.invoke(r#"{"host": "localhost", "port": 8080}"#.to_string(), None).await.unwrap();
    assert_eq!(config.host, "localhost");
    assert_eq!(config.port, 8080);
}
