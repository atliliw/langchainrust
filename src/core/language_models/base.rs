// src/core/language_models/base.rs
//! 语言模型基础 trait
//!
//! 参考 Python 版本: langchain/libs/core/langchain_core/language_models/base.py

use crate::core::runnables::Runnable;
use async_trait::async_trait;

/// 语言模型基础 trait
///
/// 所有语言模型包装器都继承自这个基类。
/// 它继承自 Runnable 接口，提供统一的调用方式。
#[async_trait]
pub trait BaseLanguageModel<Input: Send + Sync + 'static, Output: Send + Sync + 'static>:
    Runnable<Input, Output>
{
    /// 获取模型名称
    fn model_name(&self) -> &str;

    /// 计算文本的 token 数量
    ///
    /// # 参数
    /// * `text` - 要计算的文本
    ///
    /// # 返回
    /// token 数量
    fn get_num_tokens(&self, text: &str) -> usize;

    /// 获取温度参数
    fn temperature(&self) -> Option<f32> {
        None
    }

    /// 获取最大 token 数
    fn max_tokens(&self) -> Option<usize> {
        None
    }

    /// 设置温度参数
    fn with_temperature(self, temp: f32) -> Self
    where
        Self: Sized;

    /// 设置最大 token 数
    fn with_max_tokens(self, max: usize) -> Self
    where
        Self: Sized;
}
