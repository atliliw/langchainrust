// src/memory/summary.rs
//! Conversation Summary Memory
//!
//! 使用 LLM 自动摘要对话历史，解决长对话 token 爆炸问题。

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use super::base::{BaseMemory, ChatMessageHistory, MemoryError};
use crate::schema::Message;
use crate::BaseChatModel;
use crate::Runnable;

/// 默认摘要提示词
const DEFAULT_SUMMARY_PROMPT: &str = "Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

EXAMPLE
Summary of conversation:
Human: 我叫张三，我喜欢编程。
AI: 你好张三，很高兴认识你！你喜欢编程，有什么特别喜欢的语言吗？
Human: 我喜欢 Rust。
AI: Rust 是一门很棒的编程语言，注重安全和性能。

New lines of conversation:
Human: 我还喜欢 Python。
AI: Python 也很受欢迎，语法简洁，适合快速开发。

New summary:
Human 张三喜欢编程，特别喜欢 Rust 和 Python。AI 与张三讨论了这两种语言的特点。

END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:";

/// Conversation Summary Memory
///
/// 使用 LLM 自动摘要对话历史，避免上下文过长。
///
/// # 示例
/// ```ignore
/// use langchainrust::{ConversationSummaryMemory, OpenAIChat};
///
/// let llm = OpenAIChat::new(config);
/// let memory = ConversationSummaryMemory::new(llm);
///
/// // 每轮对话后自动生成摘要
/// memory.save_context(&inputs, &outputs).await?;
///
/// // 加载时返回摘要而非完整历史
/// let vars = memory.load_memory_variables(&HashMap::new()).await?;
/// ```
pub struct ConversationSummaryMemory<M: BaseChatModel> {
    llm: M,

    /// 当前摘要 (M67: removed Mutex - &mut self already guarantees exclusive access)
    buffer: String,

    /// 聊天历史（H29: trimmed after each summary to prevent unbounded growth）
    chat_memory: ChatMessageHistory,

    /// 输入键名
    input_key: String,

    /// 输出键名
    output_key: String,

    /// 记忆变量名
    memory_key: String,

    /// 摘要提示词
    summary_prompt: String,

    /// 是否返回消息对象
    return_messages: bool,

    /// H29: Maximum number of recent message pairs to keep in chat_memory
    /// after summarization. Older messages are discarded since the summary
    /// already captures their content. Default: 2 (last turn only).
    max_recent_turns: usize,
}

impl<M: BaseChatModel> ConversationSummaryMemory<M> {
    /// Create a new summary memory
    pub fn new(llm: M) -> Self {
        Self {
            llm,
            buffer: String::new(),
            chat_memory: ChatMessageHistory::new(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
            return_messages: false,
            max_recent_turns: 2,
        }
    }

    /// Create from existing messages
    pub fn from_messages(llm: M, messages: Vec<Message>) -> Self {
        let chat_memory = ChatMessageHistory::from_messages(messages);
        Self {
            llm,
            buffer: String::new(),
            chat_memory,
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
            return_messages: false,
            max_recent_turns: 2,
        }
    }

    /// 设置输入键名
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// 设置输出键名
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// 设置记忆变量名
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// 设置摘要提示词
    pub fn with_summary_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.summary_prompt = prompt.into();
        self
    }

    /// 设置是否返回消息对象
    pub fn with_return_messages(mut self, return_messages: bool) -> Self {
        self.return_messages = return_messages;
        self
    }

    /// H29: Set maximum recent turns to keep in chat_memory after summarization
    pub fn with_max_recent_turns(mut self, max: usize) -> Self {
        self.max_recent_turns = max;
        self
    }

    /// 获取聊天历史
    pub fn chat_memory(&self) -> &ChatMessageHistory {
        &self.chat_memory
    }

    /// 获取当前摘要
    pub async fn buffer(&self) -> String {
        self.buffer.clone()
    }

    /// 格式化新对话行
    fn format_new_lines(&self, input: &str, output: &str) -> String {
        format!("Human: {}\nAI: {}", input, output)
    }

    /// 生成新摘要
    async fn predict_new_summary(&self, new_lines: &str) -> Result<String, MemoryError> {
        let buffer = self.buffer.clone();

        let prompt = {
            use crate::PromptTemplate;
            let template = PromptTemplate::new(&self.summary_prompt);
            let mut vars: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
            vars.insert("summary", buffer.as_str());
            vars.insert("new_lines", new_lines);
            template.format(&vars).unwrap_or_else(|_| self.summary_prompt.clone())
        };

        let messages = vec![Message::human(&prompt)];

        let result =
            self.llm.invoke(messages, None).await.map_err(|e| {
                MemoryError::SaveError(format!("LLM summary generation failed: {}", e))
            })?;

        Ok(result.content)
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseMemory for ConversationSummaryMemory<M>
where
    <M as Runnable<Vec<Message>, crate::core::language_models::LLMResult>>::Error:
        std::fmt::Display,
{
    fn memory_variables(&self) -> Vec<&str> {
        vec![&self.memory_key]
    }

    async fn load_memory_variables(
        &self,
        _inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, Value>, MemoryError> {
        let mut result = HashMap::new();

        let buffer = self.buffer.clone();

        if self.return_messages {
            let summary_msg = Message::system(&buffer);
            result.insert(
                self.memory_key.clone(),
                serde_json::to_value(&summary_msg).unwrap_or(Value::Null),
            );
        } else {
            result.insert(self.memory_key.clone(), Value::String(buffer));
        }

        Ok(result)
    }

    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        let empty = String::new();
        let input = inputs.get(&self.input_key).unwrap_or(&empty);
        let output = outputs.get(&self.output_key).unwrap_or(&empty);

        self.chat_memory.add_user_message(input);
        self.chat_memory.add_ai_message(output);

        let new_lines = self.format_new_lines(input, output);
        let new_summary = self.predict_new_summary(&new_lines).await?;

        self.buffer = new_summary;

        // H29: Trim chat_memory to prevent unbounded growth.
        // Since the summary already captures all conversation content,
        // only keep the most recent turns for context continuity.
        let max_messages = self.max_recent_turns * 2;
        let current_len = self.chat_memory.len();
        if current_len > max_messages {
            let messages = self.chat_memory.messages().to_vec();
            self.chat_memory.clear();
            // Preserve System messages and the most recent turns
            let start = current_len.saturating_sub(max_messages);
            for msg in messages.iter().take(start) {
                if matches!(msg.message_type, crate::schema::MessageType::System) {
                    self.chat_memory.add_system_message(&msg.content);
                }
            }
            for msg in messages.iter().skip(start) {
                if matches!(msg.message_type, crate::schema::MessageType::Human) {
                    self.chat_memory.add_user_message(&msg.content);
                } else if matches!(msg.message_type, crate::schema::MessageType::AI) {
                    self.chat_memory.add_ai_message(&msg.content);
                } else if matches!(msg.message_type, crate::schema::MessageType::System) {
                    self.chat_memory.add_system_message(&msg.content);
                }
            }
        }

        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        self.buffer = String::new();
        self.chat_memory.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_models::OpenAIChat;
    use crate::OpenAIConfig;

    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig {
            api_key: "sk-test".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            streaming: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_new() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryMemory<OpenAIChat> = ConversationSummaryMemory::new(llm);

        assert_eq!(memory.memory_variables(), vec!["history"]);
    }

    #[test]
    fn test_with_options() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryMemory<OpenAIChat> = ConversationSummaryMemory::new(llm)
            .with_input_key("question")
            .with_output_key("answer")
            .with_memory_key("context");

        assert_eq!(memory.input_key, "question");
        assert_eq!(memory.output_key, "answer");
        assert_eq!(memory.memory_key, "context");
    }

    #[test]
    fn test_from_messages() {
        let llm = OpenAIChat::new(create_test_config());
        let messages = vec![Message::human("你好"), Message::ai("你好！")];
        let memory: ConversationSummaryMemory<OpenAIChat> =
            ConversationSummaryMemory::from_messages(llm, messages);

        assert_eq!(memory.chat_memory().len(), 2);
    }

    #[test]
    fn test_format_new_lines() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryMemory<OpenAIChat> = ConversationSummaryMemory::new(llm);

        let new_lines = memory.format_new_lines("你好", "你好！");
        assert_eq!(new_lines, "Human: 你好\nAI: 你好！");
    }

    #[tokio::test]
    async fn test_buffer_initial_empty() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryMemory<OpenAIChat> = ConversationSummaryMemory::new(llm);

        let buffer = memory.buffer().await;
        assert!(buffer.is_empty());
    }

    #[tokio::test]
    async fn test_load_memory_variables_empty() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryMemory<OpenAIChat> = ConversationSummaryMemory::new(llm);

        let vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();

        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_clear() {
        let llm = OpenAIChat::new(create_test_config());
        let mut memory: ConversationSummaryMemory<OpenAIChat> = ConversationSummaryMemory::new(llm);

        memory.chat_memory.add_user_message("测试");
        memory.chat_memory.add_ai_message("回复");

        memory.buffer = "测试摘要".to_string();

        memory.clear().await.unwrap();

        assert!(memory.buffer().await.is_empty());
        assert_eq!(memory.chat_memory().len(), 0);
    }
}
