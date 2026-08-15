// crates/lc-prompts/src/prompt_template.rs
//! 简单字符串模板

use crate::template_parser::{
    format_template, parse_template, template_variables, TemplateSegment,
};
use std::collections::HashMap;

/// 提示词模板
///
/// 使用 `{variable}` 格式的模板，支持变量替换；`{{` 与 `}}` 可转义字面花括号。
pub struct PromptTemplate {
    template: String,
    segments: Vec<TemplateSegment>,
}

impl PromptTemplate {
    /// 创建新的提示词模板
    ///
    /// # 参数
    /// * `template` - 模板字符串，使用 `{variable}` 标记变量
    ///
    /// # 示例
    /// ```ignore
    /// let template = PromptTemplate::new("你好，{name}！今天是{day}。");
    /// let mut vars = HashMap::new();
    /// vars.insert("name", "小明");
    /// vars.insert("day", "星期一");
    /// let result = template.format(&vars).unwrap();
    /// ```
    pub fn new(template: impl Into<String>) -> Self {
        let template = template.into();
        let segments = parse_template(&template);
        Self { template, segments }
    }

    /// 格式化模板，替换所有变量
    ///
    /// # 参数
    /// * `variables` - 变量映射表
    ///
    /// # 返回
    /// 替换后的字符串，或缺失变量的错误
    ///
    /// # 错误
    /// 如果模板中有变量但 `variables` 中没有提供对应的值，返回错误
    pub fn format(&self, variables: &HashMap<&str, &str>) -> Result<String, String> {
        format_template(&self.template, &self.segments, variables)
    }

    /// 获取模板中需要的所有变量名
    ///
    /// # 返回
    /// 变量名列表
    pub fn variables(&self) -> Vec<String> {
        template_variables(&self.segments)
    }

    /// 获取原始模板字符串
    pub fn template(&self) -> &str {
        &self.template
    }
}

impl std::fmt::Display for PromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.template)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_template() {
        let template = PromptTemplate::new("你好，{name}！");
        let mut vars = HashMap::new();
        vars.insert("name", "小明");

        let result = template.format(&vars).unwrap();
        assert_eq!(result, "你好，小明！");
    }

    #[test]
    fn test_multiple_variables() {
        let template = PromptTemplate::new("{greeting}，{name}！今天是{day}。");
        let mut vars = HashMap::new();
        vars.insert("greeting", "早上好");
        vars.insert("name", "小红");
        vars.insert("day", "星期一");

        let result = template.format(&vars).unwrap();
        assert_eq!(result, "早上好，小红！今天是星期一。");
    }

    #[test]
    fn test_missing_variable() {
        let template = PromptTemplate::new("你好，{name}！今天是{day}。");
        let mut vars = HashMap::new();
        vars.insert("name", "小明");

        let result = template.format(&vars);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("day"));
    }

    #[test]
    fn test_get_variables() {
        let template = PromptTemplate::new("{a}, {b}, {c}");
        let vars = template.variables();
        assert_eq!(vars, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_escaped_braces() {
        let template = PromptTemplate::new("{{literal}} {name} }}");
        let mut vars = HashMap::new();
        vars.insert("name", "N");

        let result = template.format(&vars).unwrap();
        assert_eq!(result, "{literal} N }");
    }

    #[test]
    fn test_cjk_variable_name() {
        let template = PromptTemplate::new("你好，{姓名}！");
        let mut vars = HashMap::new();
        vars.insert("姓名", "小明");

        let result = template.format(&vars).unwrap();
        assert_eq!(result, "你好，小明！");
    }
}
