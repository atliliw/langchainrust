// src/core/language_models/chat.rs
//! 聊天模型基础 trait
//!
//! 参考 Python 版本: langchain/libs/core/langchain_core/language_models/chat_models.py

use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;
use crate::schema::Message;
use crate::RunnableConfig;
use super::BaseLanguageModel;

/// LLM 结果
#[derive(Debug, Clone)]
pub struct LLMResult {
    /// 生成的内容
    pub content: String,
    
    /// 模型名称
    pub model: String,
    
    /// Token 使用情况
    pub token_usage: Option<TokenUsage>,
}

/// Token 使用情况
#[derive(Debug, Clone)]
pub struct TokenUsage {
    /// 输入 token 数
    pub prompt_tokens: usize,
    
    /// 输出 token 数
    pub completion_tokens: usize,
    
    /// 总 token 数
    pub total_tokens: usize,
}

/// 聊天模型基础 trait
///
/// 继承自 BaseLanguageModel，专门用于聊天场景。
/// 接受消息列表作为输入，返回 AI 消息。
#[async_trait]
pub trait BaseChatModel: BaseLanguageModel<Vec<Message>, LLMResult> {
    /// 与模型聊天
    /// 
    /// # 参数
    /// * `messages` - 消息列表
    /// * `config` - 可选配置
    /// 
    /// # 返回
    /// LLM 结果
    async fn chat(
        &self, 
        messages: Vec<Message>, 
        config: Option<RunnableConfig>
    ) -> Result<LLMResult, Self::Error>;
    
    /// 流式聊天
    /// 
    /// # 参数
    /// * `messages` - 消息列表
    /// * `config` - 可选配置
    /// 
    /// # 返回
    /// 流式输出
    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>;
    
    /// 与系统提示聊天
    /// 
    /// # 参数
    /// * `system` - 系统提示
    /// * `messages` - 消息列表
    /// 
    /// # 返回
    /// LLM 结果
    async fn chat_with_system(
        &self,
        system: String,
        messages: Vec<Message>
    ) -> Result<LLMResult, Self::Error> {
        let full_messages = vec![Message::system(system)]
            .into_iter()
            .chain(messages)
            .collect();
        
        self.chat(full_messages, None).await
    }
}