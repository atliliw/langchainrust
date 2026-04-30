//! 少样本提示模板 (FewShotPromptTemplate) 单元测试
//!
//! 测试以下功能：
//! - FewShotPromptTemplate 基本格式化
//! - 示例分隔符
//! - 示例选择器
//! - 缺少变量错误处理
//! - 添加示例
//! - ExampleSelector trait

use langchainrust::{
    PromptTemplate,
    FewShotPromptTemplate, ExampleSelector, LengthBasedExampleSelector,
};
use std::collections::HashMap;

/// 辅助函数：创建示例
fn make_example(input: &str, output: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    map.insert("input".to_string(), input.to_string());
    map.insert("output".to_string(), output.to_string());
    map
}

/// 测试 FewShotPromptTemplate 基本格式化
///
/// 验证：前缀、示例、用户输入被正确组合。
#[test]
fn test_few_shot_basic_format() {
    let examples = vec![
        make_example("苹果", "水果"),
        make_example("玫瑰", "花"),
    ];

    let example_prompt = PromptTemplate::new("输入: {input} -> 类别: {output}");
    let few_shot = FewShotPromptTemplate::new(
        examples,
        example_prompt,
        "请将以下词语分类：",
        "输入: {input} ->",
        vec!["input".to_string()],
    );

    let mut vars = HashMap::new();
    vars.insert("input", "太阳");

    let result = few_shot.format(&vars).unwrap();

    assert!(result.contains("请将以下词语分类"), "应包含前缀");
    assert!(result.contains("苹果"), "应包含示例 1 的输入");
    assert!(result.contains("水果"), "应包含示例 1 的输出");
    assert!(result.contains("玫瑰"), "应包含示例 2 的输入");
    assert!(result.contains("花"), "应包含示例 2 的输出");
    assert!(result.contains("太阳"), "应包含用户输入");
}

/// 测试缺少输入变量时返回错误
///
/// 验证：未提供的变量导致 format 返回错误信息。
#[test]
fn test_few_shot_missing_variable() {
    let few_shot = FewShotPromptTemplate::new(
        vec![],
        PromptTemplate::new("{input} -> {output}"),
        "",
        "输入: {input}",
        vec!["input".to_string(), "missing".to_string()],
    );

    let mut vars = HashMap::new();
    vars.insert("input", "test");

    let result = few_shot.format(&vars);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("missing"),
        "应提示缺少的变量名");
}

/// 测试空示例列表
///
/// 验证：没有示例时，只输出前缀和后缀。
#[test]
fn test_few_shot_empty_examples() {
    let few_shot = FewShotPromptTemplate::new(
        vec![],
        PromptTemplate::new("{input} -> {output}"),
        "Prefix",
        "User: {input}",
        vec!["input".to_string()],
    );

    let mut vars = HashMap::new();
    vars.insert("input", "hello");

    let result = few_shot.format(&vars).unwrap();
    assert_eq!(result, "Prefix\n\nUser: hello");
}

/// 测试自定义示例分隔符
///
/// 验证：使用 with_example_separator 设置的分隔符生效。
#[test]
fn test_few_shot_custom_separator() {
    let examples = vec![
        make_example("a", "1"),
        make_example("b", "2"),
        make_example("c", "3"),
    ];

    let few_shot = FewShotPromptTemplate::new(
        examples,
        PromptTemplate::new("{input}={output}"),
        "",
        "",
        vec![],
    ).with_example_separator(" | ");

    let vars = HashMap::new();
    let result = few_shot.format(&vars).unwrap();
    assert_eq!(result, "a=1 | b=2 | c=3");
}

/// 测试动态添加示例
///
/// 验证：add_example 方法正确增加示例数量。
#[test]
fn test_few_shot_add_example() {
    let mut few_shot = FewShotPromptTemplate::new(
        vec![make_example("old", "value")],
        PromptTemplate::new("{input}={output}"),
        "",
        "",
        vec![],
    );

    assert_eq!(few_shot.examples().len(), 1);

    few_shot.add_example(make_example("new", "value2"));
    assert_eq!(few_shot.examples().len(), 2);

    // 验证新示例被格式化
    let result = few_shot.format(&HashMap::new()).unwrap();
    assert!(result.contains("old=value"));
    assert!(result.contains("new=value2"));
}

/// 测试 LengthBasedExampleSelector
///
/// 验证：基于长度的选择器能正确初始化。
#[test]
fn test_length_based_selector_new() {
    let examples = vec![
        make_example("a", "1"),
        make_example("b", "2"),
    ];

    let selector = LengthBasedExampleSelector::new(examples);
    assert_eq!(selector.examples().len(), 2);
}

/// 测试 LengthBasedExampleSelector 添加示例
///
/// 验证：add_example 方法正确工作。
#[test]
fn test_length_based_selector_add() {
    let mut selector = LengthBasedExampleSelector::new(vec![]);
    assert!(selector.select_examples(&HashMap::new()).is_empty());

    selector.add_example(make_example("x", "y"));
    assert_eq!(selector.examples().len(), 1);
}

/// 测试 FewShotPromptTemplate 使用示例选择器
///
/// 验证：with_example_selector 可以设置自定义选择器。
#[test]
fn test_few_shot_with_selector() {
    let examples = vec![
        make_example("q1", "a1"),
        make_example("q2", "a2"),
    ];

    let selector = Box::new(LengthBasedExampleSelector::new(examples.clone()));
    let few_shot = FewShotPromptTemplate::new(
        examples,
        PromptTemplate::new("Q: {input}\nA: {output}"),
        "Examples:",
        "Q: {input}\nA:",
        vec!["input".to_string()],
    ).with_example_selector(selector);

    let mut vars = HashMap::new();
    vars.insert("input", "my question");

    let result = few_shot.format(&vars).unwrap();
    assert!(result.contains("Examples:"));
    assert!(result.contains("my question"));
}

/// 测试不包含输入变量的场景
///
/// 验证：input_variables 为空时不需要提供变量。
#[test]
fn test_few_shot_no_input_vars() {
    let few_shot = FewShotPromptTemplate::new(
        vec![make_example("only", "example")],
        PromptTemplate::new("{input}={output}"),
        "Static prefix",
        "",
        vec![],
    );

    let vars = HashMap::new();
    let result = few_shot.format(&vars).unwrap();
    assert_eq!(result, "Static prefix\n\nonly=example");
}
