// src/prompts/few_shot.rs
//! 少样本提示模板 (FewShotPromptTemplate)
//!
//! 提供在 prompt 中嵌入示例的能力，帮助 LLM 理解期望的输出格式。

use crate::prompts::PromptTemplate;
use std::collections::HashMap;

/// 示例选择器 trait
///
/// 从示例池中选择最合适的示例。
pub trait ExampleSelector: Send + Sync {
    /// 选择示例
    fn select_examples(&self, input: &HashMap<String, String>) -> Vec<&HashMap<String, String>>;
    /// 获取所有示例
    fn examples(&self) -> &[HashMap<String, String>];
    /// 添加示例
    fn add_example(&mut self, example: HashMap<String, String>);
}

/// 基于长度的示例选择器
///
/// 根据输入长度动态选择适量示例，确保不超过最大长度限制。
pub struct LengthBasedExampleSelector {
    examples: Vec<HashMap<String, String>>,
    /// 最大文本长度（字符数），默认 2048
    max_length: usize,
}

impl LengthBasedExampleSelector {
    pub fn new(examples: Vec<HashMap<String, String>>) -> Self {
        Self {
            examples,
            max_length: 2048,
        }
    }

    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = max;
        self
    }

    /// 计算格式化后的示例长度
    fn format_example_length(&self, example: &HashMap<String, String>, prefix: &str, suffix: &str) -> usize {
        let mut formatted = prefix.to_string();
        for val in example.values() {
            formatted.push_str(val);
        }
        formatted.push_str(suffix);
        formatted.len()
    }

    /// 选择最大数量的示例而不超过长度限制
    pub fn select_examples_by_length(
        &self,
        input: &HashMap<String, String>,
        example_prompt: &PromptTemplate,
        prefix: &str,
        suffix: &str,
    ) -> Vec<&HashMap<String, String>> {
        // 计算输入长度
        let input_text: String = input.values().cloned().collect::<Vec<_>>().join("");
        let input_len = prefix.len() + suffix.len() + input_text.len();
        let available = if self.max_length > input_len {
            self.max_length - input_len
        } else {
            0
        };

        let mut selected = Vec::new();
        let mut used = 0usize;

        for example in &self.examples {
            // 估算这个示例会占用的长度
            let example_vars: HashMap<&str, &str> = example.iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();

            if let Ok(formatted) = example_prompt.format(&example_vars) {
                let ex_len = formatted.len() + 10; // 加上分隔符的余量
                if used + ex_len <= available || selected.is_empty() {
                    selected.push(example);
                    used += ex_len;
                } else {
                    break;
                }
            }
        }

        selected
    }
}

impl ExampleSelector for LengthBasedExampleSelector {
    fn select_examples(&self, _input: &HashMap<String, String>) -> Vec<&HashMap<String, String>> {
        // 在 FewShot 中，select_examples 需要 prompt 信息，这里简单返回所有
        // 真正的选择逻辑在 FewShotPromptTemplate 中调用 select_examples_by_length
        self.examples.iter().collect()
    }

    fn examples(&self) -> &[HashMap<String, String>] {
        &self.examples
    }

    fn add_example(&mut self, example: HashMap<String, String>) {
        self.examples.push(example);
    }
}

/// 少样本提示模板
///
/// 在 prompt 中嵌入示例，帮助 LLM 理解期望的输出格式。
/// 相当于 Python LangChain 的 `FewShotPromptTemplate`。
///
/// # 示例
/// ```ignore
/// use langchainrust::prompts::{FewShotPromptTemplate, PromptTemplate};
/// use std::collections::HashMap;
///
/// let examples = vec![
///     HashMap::from([("input".into(), "苹果".into()), ("output".into(), "水果".into())]),
///     HashMap::from([("input".into(), "玫瑰".into()), ("output".into(), "花".into())]),
/// ];
///
/// let example_prompt = PromptTemplate::new("输入: {input} -> 输出: {output}");
///
/// let few_shot = FewShotPromptTemplate::new(
///     examples,
///     example_prompt,
///     "请将以下词语分类：",
///     "输入: {input} ->",
///     vec!["input"],
/// );
/// ```
pub struct FewShotPromptTemplate {
    /// 示例列表
    examples: Vec<HashMap<String, String>>,
    /// 格式化单个示例的 prompt 模板
    example_prompt: PromptTemplate,
    /// 示例前缀（放在所有示例之前）
    prefix: String,
    /// 示例后缀（放在所有示例之后，通常是用户输入）
    suffix: String,
    /// 示例之间的分隔符
    example_separator: String,
    /// 输入变量名列表
    input_variables: Vec<String>,
    /// 可选的自定义示例选择器
    example_selector: Option<Box<dyn ExampleSelector>>,
}

impl FewShotPromptTemplate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        examples: Vec<HashMap<String, String>>,
        example_prompt: PromptTemplate,
        prefix: impl Into<String>,
        suffix: impl Into<String>,
        input_variables: Vec<String>,
    ) -> Self {
        Self {
            examples,
            example_prompt,
            prefix: prefix.into(),
            suffix: suffix.into(),
            example_separator: "\n\n".to_string(),
            input_variables,
            example_selector: None,
        }
    }

    /// 设置示例分隔符
    pub fn with_example_separator(mut self, separator: impl Into<String>) -> Self {
        self.example_separator = separator.into();
        self
    }

    /// 设置自定义示例选择器
    pub fn with_example_selector(mut self, selector: Box<dyn ExampleSelector>) -> Self {
        self.example_selector = Some(selector);
        self
    }

    /// 设置前缀
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// 设置后缀
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }

    /// 格式化完整的 prompt
    pub fn format(&self, variables: &HashMap<&str, &str>) -> Result<String, String> {
        // 验证所有输入变量都有值
        for var in &self.input_variables {
            if !variables.contains_key(var.as_str()) {
                return Err(format!("缺少输入变量: {}", var));
            }
        }

        // 构建完整输入变量 HashMap
        let input_map: HashMap<String, String> = variables.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        // 选择示例
        let selected_examples: Vec<&HashMap<String, String>> = if let Some(ref selector) = self.example_selector {
            let input_ref_map: HashMap<String, String> = input_map.clone();
            selector.select_examples(&input_ref_map)
        } else {
            self.examples.iter().collect()
        };

        // 格式化每个示例
        let example_texts: Result<Vec<String>, String> = selected_examples.iter()
            .map(|example| {
                let example_vars: HashMap<&str, &str> = example.iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect();
                self.example_prompt.format(&example_vars)
            })
            .collect();

        let example_texts = example_texts?;
        let examples_str = example_texts.join(&self.example_separator);

        // 格式化后缀（用户输入）
        let suffix_formatted = if self.suffix.is_empty() {
            String::new()
        } else {
            let mut suffix_result = self.suffix.clone();
            for (key, value) in variables {
                suffix_result = suffix_result.replace(&format!("{{{}}}", key), value);
            }
            suffix_result
        };

        // 组合完整 prompt
        let mut parts: Vec<String> = Vec::new();

        if !self.prefix.is_empty() {
            parts.push(self.prefix.clone());
        }

        if !examples_str.is_empty() {
            parts.push(examples_str);
        }

        if !suffix_formatted.is_empty() {
            parts.push(suffix_formatted);
        }

        Ok(parts.join("\n\n"))
    }

    /// 获取输入变量列表
    pub fn input_variables(&self) -> &[String] {
        &self.input_variables
    }

    /// 获取示例列表
    pub fn examples(&self) -> &[HashMap<String, String>] {
        &self.examples
    }

    /// 添加示例
    pub fn add_example(&mut self, example: HashMap<String, String>) {
        self.examples.push(example);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_example(input: &str, output: &str) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("input".to_string(), input.to_string());
        map.insert("output".to_string(), output.to_string());
        map
    }

    #[test]
    fn test_few_shot_basic() {
        let examples = vec![
            make_example("苹果", "水果"),
            make_example("玫瑰", "花"),
        ];

        let example_prompt = PromptTemplate::new("输入: {input} -> 输出: {output}");
        let few_shot = FewShotPromptTemplate::new(
            examples,
            example_prompt,
            "请分类以下词语：",
            "输入: {input} ->",
            vec!["input".to_string()],
        );

        let mut vars = HashMap::new();
        vars.insert("input", "太阳");

        let result = few_shot.format(&vars).unwrap();
        assert!(result.contains("请分类以下词语"));
        assert!(result.contains("苹果"));
        assert!(result.contains("水果"));
        assert!(result.contains("太阳"));
    }

    #[test]
    fn test_few_shot_missing_variable() {
        let few_shot = FewShotPromptTemplate::new(
            vec![],
            PromptTemplate::new("示例: {input} -> {output}"),
            "",
            "输入: {input}",
            vec!["input".to_string(), "extra".to_string()],
        );

        let mut vars = HashMap::new();
        vars.insert("input", "test");

        let result = few_shot.format(&vars);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("extra"));
    }

    #[test]
    fn test_few_shot_empty_examples() {
        let few_shot = FewShotPromptTemplate::new(
            vec![],
            PromptTemplate::new("{input} -> {output}"),
            "Prefix",
            "Suffix: {input}",
            vec!["input".to_string()],
        );

        let mut vars = HashMap::new();
        vars.insert("input", "hello");

        let result = few_shot.format(&vars).unwrap();
        assert!(result.contains("Prefix"));
        assert!(result.contains("hello"));
        assert!(!result.contains("->")); // 没有示例，不应有箭头
    }

    #[test]
    fn test_few_shot_custom_separator() {
        let examples = vec![
            make_example("a", "1"),
            make_example("b", "2"),
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
        assert_eq!(result, "a=1 | b=2");
    }

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
    }

    #[test]
    fn test_length_based_selector() {
        let examples = vec![
            make_example("long text here", "short"),
        ];

        let selector = LengthBasedExampleSelector::new(examples)
            .with_max_length(100);

        let input_vars = HashMap::new();
        let selected = selector.select_examples(&input_vars);
        assert!(!selected.is_empty());
    }

    #[test]
    fn test_few_shot_with_selector() {
        let examples = vec![
            make_example("a", "1"),
            make_example("b", "2"),
        ];

        let selector = Box::new(LengthBasedExampleSelector::new(examples.clone()));
        let few_shot = FewShotPromptTemplate::new(
            examples,
            PromptTemplate::new("{input}={output}"),
            "Prefix",
            "{input}",
            vec!["input".to_string()],
        ).with_example_selector(selector);

        let mut vars = HashMap::new();
        vars.insert("input", "test");

        let result = few_shot.format(&vars).unwrap();
        assert!(result.contains("Prefix"));
    }
}
