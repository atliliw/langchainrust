// src/chains/base.rs
//! Chain 基础 trait

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

/// Chain 错误类型
#[derive(Debug)]
pub enum ChainError {
    /// 输入缺失
    MissingInput(String),
    
    /// 输出错误
    OutputError(String),
    
    /// 执行错误
    ExecutionError(String),
    
    /// 其他错误
    Other(String),
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainError::MissingInput(key) => write!(f, "缺少输入: {}", key),
            ChainError::OutputError(msg) => write!(f, "输出错误: {}", msg),
            ChainError::ExecutionError(msg) => write!(f, "执行错误: {}", msg),
            ChainError::Other(msg) => write!(f, "Chain 错误: {}", msg),
        }
    }
}

impl std::error::Error for ChainError {}

/// Chain 执行结果
pub type ChainResult = HashMap<String, Value>;

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
        assert!(error.to_string().contains("缺少输入"));
        
        let error = ChainError::ExecutionError("test".to_string());
        assert!(error.to_string().contains("执行错误"));
    }
}