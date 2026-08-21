// crates/lc-prompts/src/prompt_template.rs
//! 简单字符串模板

use crate::template_parser::{
    format_template, parse_template, template_variables, TemplateSegment,
};
use async_trait::async_trait;
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
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

// Runnable 形态:让提示词能进 LCEL 链,`prompt.pipe(...)` 成立。
// 接收 owned 变量表(HashMap<String, String>),转引用委托给 `format`。
// 与 `ChatPromptTemplate` 一致,错误走 `LcelError::Chain`。
#[async_trait]
impl Runnable<HashMap<String, String>, String> for PromptTemplate {
    type Error = LcelError;

    async fn invoke(
        &self,
        input: HashMap<String, String>,
        _config: Option<RunnableConfig>,
    ) -> Result<String, LcelError> {
        // `format` 收 &HashMap<&str, &str>,这里从 owned map 转引用
        let vars: HashMap<&str, &str> = input
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        self.format(&vars).map_err(LcelError::Chain)
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

    #[tokio::test]
    async fn test_runnable_invoke() {
        let template = PromptTemplate::new("你好，{name}！");
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "小明".to_string());

        let result = template.invoke(vars, None).await.unwrap();
        assert_eq!(result, "你好，小明！");
    }

    #[tokio::test]
    async fn test_runnable_pipe() {
        use lc_core::runnables::{RunnableExt, RunnableLambda};

        let template = PromptTemplate::new("你好，{name}！");
        let chain = template.pipe(RunnableLambda::new_sync(|s: String| s.contains("小明")));

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "小明".to_string());
        let result = chain.invoke(vars, None).await.unwrap();
        assert!(result);
    }
}
