// src/core/runnables/config.rs
//! Runnable 配置模块
//!
//! 参考 Python 版本: langchain/libs/core/langchain_core/runnables/config.py

use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Runnable 执行配置
///
/// 这个结构体对应 Python 的 RunnableConfig TypedDict
#[derive(Debug, Clone, Default)]
pub struct RunnableConfig {
    /// 标签 - 用于过滤和追踪
    pub tags: Vec<String>,

    /// 元数据 - 自定义数据 (JSON 可序列化)
    pub metadata: HashMap<String, Value>,

    /// 最大并发数 - 用于批量操作
    pub max_concurrency: Option<usize>,

    /// 运行 ID - 用于追踪
    pub run_id: Option<Uuid>,

    /// 运行名称 - 用于调试
    pub run_name: Option<String>,
}

impl RunnableConfig {
    /// 创建空配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加标签
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// 设置最大并发数
    pub fn with_max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = Some(max);
        self
    }

    /// 设置运行 ID
    pub fn with_run_id(mut self, id: Uuid) -> Self {
        self.run_id = Some(id);
        self
    }

    /// 设置运行名称
    pub fn with_run_name(mut self, name: impl Into<String>) -> Self {
        self.run_name = Some(name.into());
        self
    }

    /// 合并两个配置 (后面的配置覆盖前面的)
    pub fn merge(mut self, other: RunnableConfig) -> Self {
        // 合并标签 (取并集)
        self.tags.extend(other.tags);
        self.tags.sort();
        self.tags.dedup();

        // 合并元数据 (覆盖)
        self.metadata.extend(other.metadata);

        // 覆盖其他字段
        if other.max_concurrency.is_some() {
            self.max_concurrency = other.max_concurrency;
        }
        if other.run_id.is_some() {
            self.run_id = other.run_id;
        }
        if other.run_name.is_some() {
            self.run_name = other.run_name;
        }

        self
    }
}
