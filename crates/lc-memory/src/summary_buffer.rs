// lc-memory/src/summary_buffer.rs
//! Conversation Summary Buffer Memory
//!
//! Combines summary and full conversation, balancing token consumption and conversation quality.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use super::base::{BaseChatMemory, BaseMemory, ChatMessageHistory, MemoryError};
use lc_core::language_models::BaseChatModel;
use lc_core::language_models::LLMResult;
use lc_core::runnables::Runnable;
use lc_core::token_counter::{CharRatioCounter, TiktokenCounter, TokenCounter};
use lc_prompts::PromptTemplate;
use lc_schema::Message;

const DEFAULT_SUMMARY_PROMPT: &str =
    "Progressively summarize the conversation, adding new content to the previous summary.

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:";

/// Conversation Summary Buffer Memory
///
/// Combines summary and full conversation:
/// - Keeps the last k rounds of full conversation (ensuring fluency)
/// - Summarizes older conversations (saving tokens)
///
/// # Example
/// ```ignore
/// use lc_memory::ConversationSummaryBufferMemory;
/// use lc_providers::OpenAIChat;
///
/// let llm = OpenAIChat::new(config);
/// let memory = ConversationSummaryBufferMemory::new(llm, 5); // Keep last 5 rounds
///
/// // After 20 rounds:
/// // - First 15 rounds -> summary
/// // - Last 5 rounds -> full conversation
/// ```
pub struct ConversationSummaryBufferMemory<M: BaseChatModel> {
    llm: M,

    /// M67: Removed `Mutex<String>` - &mut self already guarantees exclusive access
    buffer: String,
    chat_memory: ChatMessageHistory,

    max_token_limit: usize,

    /// P1-2: 可插拔 token 计数器。默认 `TiktokenCounter`(与 `ContextWindow`
    /// 同口径,BPE 预算语义统一);可注入 `CharRatioCounter` 保留零依赖快路径。
    counter: Arc<dyn TokenCounter>,

    input_key: String,
    output_key: String,
    memory_key: String,

    summary_prompt: String,
    return_messages: bool,

    /// P2-4: 最近一次摘要 LLM 失败的原因;成功总结或 `clear()` 后清空。
    /// 摘要失败时保留旧摘要与原始消息,不清空 `chat_memory`,下轮 prune 重试。
    last_summary_error: Option<String>,
}

impl<M: BaseChatModel> ConversationSummaryBufferMemory<M> {
    /// 使用给定的 LLM 与 token 预算创建新的摘要缓冲记忆。
    pub fn new(llm: M, max_token_limit: usize) -> Self {
        Self {
            llm,
            buffer: String::new(),
            chat_memory: ChatMessageHistory::new(),
            max_token_limit,
            counter: Self::default_token_counter(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
            return_messages: false,
            last_summary_error: None,
        }
    }

    /// 设置输入 key,`save_context` 用它从 inputs 中取出用户输入。
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// 设置输出 key,`save_context` 用它从 outputs 中取出 AI 输出。
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// 设置记忆 key,加载的历史将暴露在该 key 下。
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// 设置用于生成摘要的提示词模板。
    pub fn with_summary_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.summary_prompt = prompt.into();
        self
    }

    /// 设置加载的历史是否以消息列表(而非文本)形式返回。
    pub fn with_return_messages(mut self, return_messages: bool) -> Self {
        self.return_messages = return_messages;
        self
    }

    /// 注入自定义 token 计数器。
    ///
    /// 默认 `TiktokenCounter`(BPE 口径,与 `ContextWindow` 一致);需要零依赖
    /// 快路径时注入 `CharRatioCounter::new(4)`。
    pub fn with_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.counter = counter;
        self
    }

    /// P1-3: 从持久化存储回灌摘要状态,保证续写会话摘要链连续。
    pub fn set_summary(&mut self, summary: impl Into<String>) {
        self.buffer = summary.into();
    }

    /// P1-3: 设置 token 预算(持久化 config 的单一来源)。
    pub fn set_max_token_limit(&mut self, max_token_limit: usize) {
        self.max_token_limit = max_token_limit;
    }

    /// 返回底层聊天消息历史的不可变引用。
    pub fn chat_memory(&self) -> &ChatMessageHistory {
        &self.chat_memory
    }

    /// 返回底层聊天消息历史的可变引用。
    pub fn chat_memory_mut(&mut self) -> &mut ChatMessageHistory {
        &mut self.chat_memory
    }

    /// 返回当前配置的 token 预算。
    pub fn max_token_limit(&self) -> usize {
        self.max_token_limit
    }

    /// 返回当前的摘要缓冲内容。
    pub async fn buffer(&self) -> String {
        self.buffer.clone()
    }

    /// P2-4: 最近一次摘要失败的原因(无失败则 `None`)。
    pub fn last_summary_error(&self) -> Option<&str> {
        self.last_summary_error.as_deref()
    }

    /// P1-2: 估算文本 token 数,委托给可插拔计数器(默认 BPE 口径)。
    fn estimate_tokens(&self, text: &str) -> usize {
        self.counter.count_tokens(text) as usize
    }

    fn prune_messages(&self, messages: &[Message]) -> Vec<Message> {
        let total_tokens = messages
            .iter()
            .map(|m| self.estimate_tokens(&m.content))
            .sum::<usize>();

        if total_tokens <= self.max_token_limit {
            return messages.to_vec();
        }

        let mut kept_messages = Vec::new();
        let mut current_tokens = 0;

        for msg in messages.iter().rev() {
            let msg_tokens = self.estimate_tokens(&msg.content);
            if current_tokens + msg_tokens <= self.max_token_limit {
                kept_messages.push(msg.clone());
                current_tokens += msg_tokens;
            } else {
                break;
            }
        }

        kept_messages.reverse();
        kept_messages
    }

    /// 默认 token 计数器:优先 `TiktokenCounter`(BPE 口径,与 `ContextWindow` 一致);
    /// tiktoken 模型加载失败(离线/缺模型)时优雅降级为字符比估算,
    /// 使 `new()` 保持不可失败的签名。
    fn default_token_counter() -> Arc<dyn TokenCounter> {
        TiktokenCounter::new()
            .map(|c| Arc::new(c) as Arc<dyn TokenCounter>)
            .unwrap_or_else(|_| Arc::new(CharRatioCounter::new(4)) as Arc<dyn TokenCounter>)
    }

    async fn predict_new_summary(&self, new_lines: &str) -> Result<String, MemoryError> {
        let buffer = self.buffer.clone();

        let prompt = {
            let template = PromptTemplate::new(&self.summary_prompt);
            let mut vars: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
            vars.insert("summary", buffer.as_str());
            vars.insert("new_lines", new_lines);
            template
                .format(&vars)
                .unwrap_or_else(|_| self.summary_prompt.clone())
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
impl<M: BaseChatModel + Send + Sync + 'static> BaseMemory for ConversationSummaryBufferMemory<M>
where
    <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
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
        let messages = self.chat_memory.messages();
        let pruned = self.prune_messages(messages);

        if self.return_messages {
            let mut all_messages = Vec::new();

            if !buffer.is_empty() {
                all_messages.push(Message::system(&buffer));
            }

            all_messages.extend(pruned);

            let messages_value: Vec<Value> = all_messages
                .iter()
                .map(|m| serde_json::to_value(m).unwrap_or(Value::Null))
                .collect();

            result.insert(self.memory_key.clone(), Value::Array(messages_value));
        } else {
            let mut history = String::new();

            if !buffer.is_empty() {
                history.push_str(&format!("Summary: {}\n\n", buffer));
            }

            for msg in &pruned {
                let role = match msg.message_type {
                    lc_schema::MessageType::Human => "Human",
                    lc_schema::MessageType::AI => "AI",
                    lc_schema::MessageType::System => "System",
                    lc_schema::MessageType::Tool { .. } => "Tool",
                };
                history.push_str(&format!("{}: {}\n", role, msg.content));
            }

            result.insert(self.memory_key.clone(), Value::String(history));
        }

        Ok(result)
    }

    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        // P1-1: 与 Buffer/Window 一致——缺失 key 返回 SaveError,不再静默用空串
        // 存空消息(否则会对空行做无意义摘要,白烧一次 LLM 调用)。
        let input = inputs.get(&self.input_key).ok_or_else(|| {
            MemoryError::SaveError(format!("Missing input key '{}'", self.input_key))
        })?;
        let output = outputs.get(&self.output_key).ok_or_else(|| {
            MemoryError::SaveError(format!("Missing output key '{}'", self.output_key))
        })?;

        self.chat_memory.add_user_message(input);
        self.chat_memory.add_ai_message(output);

        let messages = self.chat_memory.messages();
        let total_tokens = messages
            .iter()
            .map(|m| self.estimate_tokens(&m.content))
            .sum::<usize>();

        if total_tokens > self.max_token_limit {
            let pruned = self.prune_messages(messages);

            let pruned_count = pruned.len();

            if messages.len() > pruned_count {
                let messages_to_summarize: Vec<&Message> = messages
                    .iter()
                    .take(messages.len() - pruned_count)
                    .collect();

                if !messages_to_summarize.is_empty() {
                    let new_lines: String = messages_to_summarize
                        .iter()
                        .map(|m| {
                            let role = match m.message_type {
                                lc_schema::MessageType::Human => "Human",
                                lc_schema::MessageType::AI => "AI",
                                lc_schema::MessageType::System => "System",
                                lc_schema::MessageType::Tool { .. } => "Tool",
                            };
                            format!("{}: {}", role, m.content)
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    // P2-4: 摘要失败时保留旧摘要、不清空 chat_memory(原始消息
                    // 留下,下轮 prune 会再次尝试总结);错误记录到 last_summary_error
                    // 供上层观察,不冒泡打断链。
                    match self.predict_new_summary(&new_lines).await {
                        Ok(new_summary) => {
                            self.buffer = new_summary;
                            self.last_summary_error = None;

                            self.chat_memory.clear();
                            for msg in pruned {
                                if matches!(msg.message_type, lc_schema::MessageType::Human) {
                                    self.chat_memory.add_user_message(&msg.content);
                                } else if matches!(msg.message_type, lc_schema::MessageType::AI) {
                                    self.chat_memory.add_ai_message(&msg.content);
                                } else if matches!(msg.message_type, lc_schema::MessageType::System)
                                {
                                    // H28: Preserve System messages during pruning
                                    self.chat_memory.add_system_message(&msg.content);
                                }
                            }
                        }
                        Err(e) => {
                            self.last_summary_error = Some(e.to_string());
                            log::warn!(
                                "ConversationSummaryBufferMemory summarization failed, keeping old summary and original messages for next retry: {}",
                                e
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        self.buffer = String::new();
        self.chat_memory.clear();
        self.last_summary_error = None;
        Ok(())
    }
}

/// P0-1: `ConversationSummaryBufferMemory` 实现 `BaseChatMemory`。
impl<M: BaseChatModel + Send + Sync + 'static> BaseChatMemory for ConversationSummaryBufferMemory<M>
where
    <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
{
    fn messages(&self) -> &[Message] {
        self.chat_memory.messages()
    }

    fn add_message(&mut self, message: Message) {
        self.chat_memory.add_message(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::MockLlm;
    use lc_providers::{OpenAIChat, OpenAIConfig};

    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig::default()
    }

    #[test]
    fn test_new() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        assert_eq!(memory.memory_variables(), vec!["history"]);
        assert_eq!(memory.max_token_limit(), 1000);
    }

    #[test]
    fn test_with_options() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 500)
                .with_input_key("question")
                .with_output_key("answer")
                .with_memory_key("context")
                .with_return_messages(true);

        assert_eq!(memory.input_key, "question");
        assert_eq!(memory.output_key, "answer");
        assert_eq!(memory.memory_key, "context");
        assert!(memory.return_messages);
    }

    #[test]
    fn test_estimate_tokens_uses_default_counter() {
        // 默认 TiktokenCounter(BPE 口径);离线时降级 CharRatioCounter。
        // 两种实现下,较长文本的估算 token 数都严格大于较短文本。
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        let text1 = "Hello";
        let text2 = "Hello World";
        let text3 = "This is some Chinese text";

        assert!(memory.estimate_tokens(text1) > 0);
        assert!(memory.estimate_tokens(text2) > memory.estimate_tokens(text1));
        assert!(memory.estimate_tokens(text3) > 0);
    }

    #[test]
    fn test_with_counter_injection() {
        // 注入 CharRatioCounter(ratio=4):8 个字符估算 2 token,可复现、不依赖 tiktoken。
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000)
                .with_counter(std::sync::Arc::new(CharRatioCounter::new(4)));

        assert_eq!(memory.estimate_tokens("abcdefgh"), 2);
    }

    #[tokio::test]
    async fn test_set_summary_and_token_limit() {
        // P1-3: 持久化回灌摘要 + 预算单一来源。
        let llm = OpenAIChat::new(create_test_config());
        let mut memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        memory.set_summary("previous summary".to_string());
        assert_eq!(memory.buffer().await, "previous summary");

        memory.set_max_token_limit(500);
        assert_eq!(memory.max_token_limit(), 500);
    }

    #[test]
    fn test_prune_messages_within_limit() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        let messages = vec![
            Message::human("Short message 1"),
            Message::ai("Short reply 1"),
        ];

        let pruned = memory.prune_messages(&messages);

        assert_eq!(pruned.len(), 2);
    }

    #[tokio::test]
    async fn test_buffer_initial_empty() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        let buffer = memory.buffer().await;
        assert!(buffer.is_empty());
    }

    #[tokio::test]
    async fn test_load_memory_variables_empty() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        let vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();

        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_clear() {
        let llm = OpenAIChat::new(create_test_config());
        let mut memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        memory.chat_memory.add_user_message("test");
        memory.chat_memory.add_ai_message("reply");

        memory.buffer = "Test summary".to_string();

        memory.clear().await.unwrap();

        assert!(memory.buffer().await.is_empty());
        assert_eq!(memory.chat_memory().len(), 0);
    }

    /// P2-4: 剪枝触发摘要、但摘要 LLM 失败时——保留旧摘要、不清空 chat_memory、
    /// 记录错误且不冒泡;下轮成功总结后摘要生效、错误清空。
    #[tokio::test]
    async fn test_prune_summary_failure_keeps_messages_and_retries() {
        // MockLlm 按 LIFO 消费:第一次剪枝总结失败,第二次成功。
        let llm = MockLlm::new(vec![
            Ok("summary-b".to_string()),
            Err("summarizer down".to_string()),
        ]);
        // CharRatioCounter 保证 token 估算可复现(不依赖 tiktoken 在线)。
        let mut memory: ConversationSummaryBufferMemory<MockLlm> =
            ConversationSummaryBufferMemory::new(llm, 5)
                .with_counter(std::sync::Arc::new(CharRatioCounter::new(4)));

        let long_input =
            "这是一段足够长的中文消息,用来确保本轮消息总 token 数超过预算并触发剪枝总结逻辑";
        let inputs = HashMap::from([("input".to_string(), long_input.to_string())]);
        let outputs = HashMap::from([("output".to_string(), long_input.to_string())]);

        // 第一轮:总 token 超限 -> 触发剪枝总结 -> LLM 失败
        memory.save_context(&inputs, &outputs).await.unwrap();
        assert!(memory.buffer().await.is_empty(), "失败时不应覆盖旧摘要");
        assert!(memory
            .last_summary_error()
            .unwrap()
            .contains("summarizer down"));
        // 失败时不清空原始消息,下轮 prune 才能重试总结
        assert_eq!(memory.chat_memory().len(), 2);

        // 第二轮:再次触发剪枝总结 -> 成功,摘要生效、错误清空
        memory.save_context(&inputs, &outputs).await.unwrap();
        assert_eq!(memory.buffer().await, "summary-b");
        assert!(memory.last_summary_error().is_none());
    }
}
