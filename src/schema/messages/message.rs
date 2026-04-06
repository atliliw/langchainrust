// src/schema/messages/message.rs
//! 核心消息类型
//!
//! 参考 Python 版本: langchain/libs/core/langchain_core/messages/base.py

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// 消息类型分类
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    /// 系统消息
    System,
    /// 用户消息
    Human,
    /// AI 消息
    AI,
    /// 工具消息
    Tool {
        /// 工具调用 ID
        tool_call_id: String,
    },
}

/// 完整的消息结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// 消息内容
    pub content: String,

    /// 消息类型
    #[serde(rename = "type")]
    pub message_type: MessageType,

    /// 消息的可选名称
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// 额外的元数据
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub additional_kwargs: HashMap<String, Value>,

    /// 可选的唯一标识符
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

impl Message {
    /// 创建系统消息
    ///
    /// # 示例
    /// ```
    /// use langchainrust::Message;
    /// let msg = Message::system("你是一个有用的助手");
    /// ```
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            message_type: MessageType::System,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
        }
    }

    /// 创建用户消息
    ///
    /// # 示例
    /// ```
    /// use langchainrust::Message;
    /// let msg = Message::human("你好");
    /// ```
    pub fn human(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
        }
    }

    /// 创建 AI 消息
    ///
    /// # 示例
    /// ```
    /// use langchainrust::Message;
    /// let msg = Message::ai("你好！有什么我可以帮助你的吗？");
    /// ```
    pub fn ai(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            message_type: MessageType::AI,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
        }
    }

    /// 创建工具消息
    ///
    /// # 参数
    /// * `tool_call_id` - 工具调用 ID
    /// * `content` - 工具返回结果
    ///
    /// # 示例
    /// ```
    /// use langchainrust::Message;
    /// let msg = Message::tool("call_123", "结果: 42");
    /// ```
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            message_type: MessageType::Tool {
                tool_call_id: tool_call_id.into(),
            },
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
        }
    }

    /// 设置消息名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// 设置消息 ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// 添加额外的元数据
    pub fn with_additional_kwarg(mut self, key: impl Into<String>, value: Value) -> Self {
        self.additional_kwargs.insert(key.into(), value);
        self
    }

    /// 获取消息类型字符串
    pub fn type_str(&self) -> &str {
        match &self.message_type {
            MessageType::System => "system",
            MessageType::Human => "human",
            MessageType::AI => "ai",
            MessageType::Tool { .. } => "tool",
        }
    }
}
