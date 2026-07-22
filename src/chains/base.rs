// src/chains/base.rs
//! Chain 基础 trait

use async_trait::async_trait;
use futures_util::Stream;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;

/// Chain error type
#[derive(Debug, thiserror::Error)]
pub enum ChainError {
    /// Missing input
    #[error("Missing input: {0}")]
    MissingInput(String),

    /// Output error
    #[error("Output error: {0}")]
    OutputError(String),

    /// Execution error
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Stream error
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Other error
    #[error("Chain error: {0}")]
    Other(String),
}

/// Chain 执行结果
pub type ChainResult = HashMap<String, Value>;

/// 流式输出项:逐 token 输出
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// token 文本
    pub token: String,
    /// 是否为最后一个 token
    pub is_final: bool,
}

/// Chain 流式输出类型
pub type ChainStream = Pin<Box<dyn Stream<Item = Result<StreamToken, ChainError>> + Send>>;

/// Base Chain trait
///
/// Chain 是 LangChain 的核心抽象，表示一系列操作的组合。
#[async_trait]
pub trait BaseChain: Send + Sync {
    /// 获取输入键
    fn input_keys(&self) -> Vec<&str>;

    /// 获取输出键
    fn output_keys(&self) -> Vec<&str>;

    /// 执行 Chain
    ///
    /// # 参数
    /// * `inputs` - 输入参数字典
    ///
    /// # 返回
    /// 输出结果字典
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError>;

    /// 流式执行 Chain -- 逐 token 输出
    ///
    /// 默认实现将 invoke 结果包装为单元素流。
    /// 支持 LLM 流式的 Chain(LLMChain / ConversationChain)应覆写此方法,
    /// 内部调 `BaseChatModel::stream_chat`,逐 token 回调。
    ///
    /// # 参数
    /// * `inputs` - 输入参数字典
    ///
    /// # 返回
    /// token 流
    async fn stream(
        &self,
        inputs: HashMap<String, Value>,
    ) -> Result<ChainStream, ChainError> {
        // 默认:将 invoke 结果包装为单元素流
        let result = self.invoke(inputs).await?;
        let output_text = result
            .values()
            .next()
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let stream = futures_util::stream::once(async move {
            Ok(StreamToken {
                token: output_text,
                is_final: true,
            })
        });
        Ok(Box::pin(stream))
    }

    /// 验证输入
    fn validate_inputs(&self, inputs: &HashMap<String, Value>) -> Result<(), ChainError> {
        for key in self.input_keys() {
            if !inputs.contains_key(key) {
                return Err(ChainError::MissingInput(key.to_string()));
            }
        }
        Ok(())
    }

    /// 获取 Chain 名称
    fn name(&self) -> &str {
        "chain"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chain_error_display() {
        let error = ChainError::MissingInput("test".to_string());
        assert!(error.to_string().contains("Missing input"));

        let error = ChainError::ExecutionError("test".to_string());
        assert!(error.to_string().contains("Execution error"));
    }
}