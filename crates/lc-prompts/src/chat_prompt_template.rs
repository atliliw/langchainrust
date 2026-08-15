// crates/lc-prompts/src/chat_prompt_template.rs
//! 聊天消息模板

use crate::template_parser::{
    format_template, parse_template, template_variables, TemplateSegment,
};
use lc_schema::Message;
use std::collections::HashMap;

/// 聊天提示词模板
///
/// 支持多条消息的模板，每条消息内容可以使用 `{variable}` 格式。
/// 消息内容在构造时被解析一次，`format` 直接替换占位符而不是每次重扫正则。
pub struct ChatPromptTemplate {
    messages: Vec<Message>,
    segments: Vec<Vec<TemplateSegment>>,
}

impl ChatPromptTemplate {
    /// 创建新的聊天模板
    ///
    /// # 参数
    /// * `messages` - 消息列表，消息内容可以使用 `{variable}` 标记变量
    ///
    /// # 示例
    /// ```ignore
    /// let template = ChatPromptTemplate::new(vec![
    ///     Message::system("你是一个{role}助手。"),
    ///     Message::human("你好，我是{name}。"),
    /// ]);
    /// let mut vars = HashMap::new();
    /// vars.insert("role", "编程");
    /// vars.insert("name", "小明");
    /// let messages = template.format(&vars).unwrap();
    /// ```
    pub fn new(messages: Vec<Message>) -> Self {
        let segments = messages
            .iter()
            .map(|msg| parse_template(&msg.content))
            .collect();
        Self { messages, segments }
    }

    /// 格式化模板，替换所有消息中的变量
    ///
    /// # 参数
    /// * `variables` - 变量映射表
    ///
    /// # 返回
    /// 替换后的消息列表，或缺失变量的错误
    ///
    /// # 错误
    /// 如果任何消息中有变量但 `variables` 中没有提供对应的值，返回错误
    pub fn format(&self, variables: &HashMap<&str, &str>) -> Result<Vec<Message>, String> {
        self.messages
            .iter()
            .zip(&self.segments)
            .map(|(msg, segments)| {
                let content = format_template(&msg.content, segments, variables)
                    .map_err(|e| format!("{e} in message {msg:?}"))?;
                let mut formatted = msg.clone();
                formatted.content = content;
                Ok(formatted)
            })
            .collect()
    }

    /// 获取模板中需要的所有变量名
    ///
    /// # 返回
    /// 变量名列表（去重）
    pub fn variables(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        for segments in &self.segments {
            for var in template_variables(segments) {
                seen.insert(var);
            }
        }
        seen.into_iter().collect()
    }

    /// 获取原始消息模板
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// 从消息数组快速创建模板
    ///
    /// # 示例
    /// ```ignore
    /// let template = ChatPromptTemplate::from_messages([
    ///     Message::system("你是一个{role}助手。"),
    ///     Message::human("{question}"),
    /// ]);
    /// ```
    pub fn from_messages(messages: impl Into<Vec<Message>>) -> Self {
        Self::new(messages.into())
    }
}

impl std::fmt::Display for ChatPromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for msg in &self.messages {
            let role = match msg.message_type {
                lc_schema::MessageType::System => "System",
                lc_schema::MessageType::Human => "Human",
                lc_schema::MessageType::AI => "AI",
                lc_schema::MessageType::Tool { .. } => "Tool",
            };
            writeln!(f, "{}: {}", role, msg.content)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chat_template() {
        let template = ChatPromptTemplate::new(vec![
            Message::system("你是一个{role}助手。"),
            Message::human("你好，我是{name}。"),
        ]);

        let mut vars = HashMap::new();
        vars.insert("role", "编程");
        vars.insert("name", "小明");

        let messages = template.format(&vars).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "你是一个编程助手。");
        assert_eq!(messages[1].content, "你好，我是小明。");
        // 非 content 字段应被保留（Q4：不再手工逐字段克隆）
        assert!(matches!(
            messages[0].message_type,
            lc_schema::MessageType::System
        ));
        assert!(matches!(
            messages[1].message_type,
            lc_schema::MessageType::Human
        ));
    }

    #[test]
    fn test_from_messages() {
        let template = ChatPromptTemplate::from_messages([
            Message::system("系统消息"),
            Message::human("用户消息"),
        ]);

        assert_eq!(template.messages().len(), 2);
    }

    #[test]
    fn test_get_variables() {
        let template = ChatPromptTemplate::new(vec![
            Message::system("你是一个{role}，专精于{domain}。"),
            Message::human("我是{name}，请问{question}"),
        ]);

        let vars = template.variables();
        assert!(vars.contains(&"role".to_string()));
        assert!(vars.contains(&"domain".to_string()));
        assert!(vars.contains(&"name".to_string()));
        assert!(vars.contains(&"question".to_string()));
    }

    #[test]
    fn test_missing_variable() {
        let template = ChatPromptTemplate::new(vec![Message::human("你好，{name}！今天是{day}。")]);

        let mut vars = HashMap::new();
        vars.insert("name", "小明");

        let result = template.format(&vars);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("day"));
    }

    #[test]
    fn test_escaped_braces_in_message() {
        let template = ChatPromptTemplate::new(vec![Message::system(
            "请输出 JSON: {{\"key\": \"{value}\"}}",
        )]);

        let mut vars = HashMap::new();
        vars.insert("value", "v");

        let messages = template.format(&vars).unwrap();
        assert_eq!(messages[0].content, "请输出 JSON: {\"key\": \"v\"}");
    }

    #[test]
    fn test_multimodal_fields_preserved_on_format() {
        let msg = Message::human_with_image("{question}", "https://example.com/i.png");
        let template = ChatPromptTemplate::new(vec![msg]);

        let mut vars = HashMap::new();
        vars.insert("question", "看这张图");

        let messages = template.format(&vars).unwrap();
        assert_eq!(messages[0].content, "看这张图");
        assert_eq!(messages[0].images.len(), 1);
        assert_eq!(messages[0].images[0].url, "https://example.com/i.png");
    }
}
