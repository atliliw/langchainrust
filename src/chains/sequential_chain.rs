// src/chains/sequential_chain.rs
//! Sequential Chain
//!
//! 顺序执行多个 Chain。

use async_trait::async_trait;
use std::collections::HashMap;
use serde_json::Value;
use std::sync::Arc;

use super::base::{BaseChain, ChainResult, ChainError};

/// Sequential Chain
/// 
/// 顺序执行多个 Chain，前一个 Chain 的输出可以作为后一个 Chain 的输入。
/// 
/// # 示例
/// ```ignore
/// use langchainrust::{SequentialChain, LLMChain};
/// 
/// let chain1 = LLMChain::new(llm1, "生成一个{topic}相关的词");
/// let chain2 = LLMChain::new(llm2, "用这个词造句: {word}");
/// 
/// let seq_chain = SequentialChain::new()
///     .add_chain(chain1, vec!["topic"], vec!["word"])
///     .add_chain(chain2, vec!["word"], vec!["sentence"]);
/// 
/// let inputs = HashMap::from([("topic".to_string(), "编程".into())]);
/// let result = seq_chain.invoke(inputs).await?;
/// ```
pub struct SequentialChain {
    /// Chain 列表
    chains: Vec<ChainStep>,
    
    /// Chain 名称
    name: String,
}

/// Chain 步骤
struct ChainStep {
    /// Chain 实例
    chain: Arc<dyn BaseChain>,
    
    /// 输入映射（从全局输入或前序输出获取）
    input_mapping: HashMap<String, String>,
    
    /// 输出映射（输出到全局结果）
    output_mapping: HashMap<String, String>,
}

impl SequentialChain {
    /// 创建空的 SequentialChain
    pub fn new() -> Self {
        Self {
            chains: Vec::new(),
            name: "sequential_chain".to_string(),
        }
    }
    
    /// 设置名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
    
    /// 添加 Chain
    /// 
    /// # 参数
    /// * `chain` - 要添加的 Chain
    /// * `input_keys` - 输入键（从全局输入获取）
    /// * `output_keys` - 输出键（输出到全局结果）
    pub fn add_chain(
        mut self,
        chain: Arc<dyn BaseChain>,
        input_keys: Vec<&str>,
        output_keys: Vec<&str>,
    ) -> Self {
        let input_mapping = input_keys
            .into_iter()
            .map(|k| (k.to_string(), k.to_string()))
            .collect();
        
        let output_mapping = output_keys
            .into_iter()
            .map(|k| (k.to_string(), k.to_string()))
            .collect();
        
        self.chains.push(ChainStep {
            chain,
            input_mapping,
            output_mapping,
        });
        
        self
    }
    
    /// 添加带映射的 Chain
    /// 
    /// # 参数
    /// * `chain` - 要添加的 Chain
    /// * `input_mapping` - 输入映射 {chain_input_key: global_key}
    /// * `output_mapping` - 输出映射 {chain_output_key: global_key}
    pub fn add_chain_with_mapping(
        mut self,
        chain: Arc<dyn BaseChain>,
        input_mapping: HashMap<String, String>,
        output_mapping: HashMap<String, String>,
    ) -> Self {
        self.chains.push(ChainStep {
            chain,
            input_mapping,
            output_mapping,
        });
        
        self
    }
}

impl Default for SequentialChain {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseChain for SequentialChain {
    fn input_keys(&self) -> Vec<&str> {
        // 第一个 Chain 的输入键
        if let Some(first) = self.chains.first() {
            first.input_mapping.values().map(|s| s.as_str()).collect()
        } else {
            vec![]
        }
    }
    
    fn output_keys(&self) -> Vec<&str> {
        // 最后一个 Chain 的输出键
        if let Some(last) = self.chains.last() {
            last.output_mapping.values().map(|s| s.as_str()).collect()
        } else {
            vec![]
        }
    }
    
    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let mut current_state = inputs.clone();
        let mut final_output = HashMap::new();
        
        for (step_index, step) in self.chains.iter().enumerate() {
            // 构造当前 Chain 的输入
            let mut chain_inputs = HashMap::new();
            for (chain_key, global_key) in &step.input_mapping {
                if let Some(value) = current_state.get(global_key) {
                    chain_inputs.insert(chain_key.clone(), value.clone());
                } else {
                    return Err(ChainError::MissingInput(format!(
                        "Step {}: 缺少输入 '{}' (映射自 '{}')",
                        step_index, chain_key, global_key
                    )));
                }
            }
            
            // 执行 Chain
            let chain_output = step.chain.invoke(chain_inputs).await.map_err(|e| {
                ChainError::ExecutionError(format!("Step {} ({}) 执行失败: {}", step_index, step.chain.name(), e))
            })?;
            
            // 更新状态
            for (chain_key, global_key) in &step.output_mapping {
                if let Some(value) = chain_output.get(chain_key) {
                    current_state.insert(global_key.clone(), value.clone());
                    final_output.insert(global_key.clone(), value.clone());
                }
            }
        }
        
        Ok(final_output)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Debug for SequentialChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequentialChain")
            .field("steps", &self.chains.len())
            .field("name", &self.name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LLMChain, OpenAIConfig, OpenAIChat};
    
    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig {
            api_key: "sk-l0YYMX65mCYRlTJYH0ptf4BFpqJwm8Xo9Z5IMqSZD0yOafl6".to_string(),
            base_url: "https://api.openai-proxy.org/v1".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            streaming: false,
            organization: None,
            frequency_penalty: None,
            max_tokens: None,
            presence_penalty: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
        }
    }
    
    /// 测试 SequentialChain 的基本逻辑
    #[tokio::test]
    async fn test_sequential_chain_mock() {
        // 创建简单的 Mock Chain
        struct MockChain {
            name: String,
            input_key: String,
            output_key: String,
            transform: fn(&str) -> String,
        }
        
        #[async_trait]
        impl BaseChain for MockChain {
            fn input_keys(&self) -> Vec<&str> {
                vec![&self.input_key]
            }
            
            fn output_keys(&self) -> Vec<&str> {
                vec![&self.output_key]
            }
            
            async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
                let input = inputs.get(&self.input_key)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;
                
                let output = (self.transform)(input);
                
                let mut result = HashMap::new();
                result.insert(self.output_key.clone(), Value::String(output));
                
                Ok(result)
            }
            
            fn name(&self) -> &str {
                &self.name
            }
        }
        
        // 创建两个 mock chain
        let chain1 = Arc::new(MockChain {
            name: "uppercase".to_string(),
            input_key: "text".to_string(),
            output_key: "upper".to_string(),
            transform: |s| s.to_uppercase(),
        });
        
        let chain2 = Arc::new(MockChain {
            name: "reverse".to_string(),
            input_key: "upper".to_string(),
            output_key: "result".to_string(),
            transform: |s| s.chars().rev().collect(),
        });
        
        // 创建 SequentialChain
        let seq_chain = SequentialChain::new()
            .add_chain(chain1, vec!["text"], vec!["upper"])
            .add_chain(chain2, vec!["upper"], vec!["result"]);
        
        // 执行
        let inputs = HashMap::from([
            ("text".to_string(), Value::String("hello".to_string()))
        ]);
        
        let result = seq_chain.invoke(inputs).await.unwrap();
        
        // hello -> HELLO -> OLLEH
        assert_eq!(result.get("result").unwrap().as_str().unwrap(), "OLLEH");
    }
    
    /// 真实 API 测试 - SequentialChain (两步 LLM 调用)
    /// 运行: cargo test test_sequential_chain_real -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_sequential_chain_real() {
        // 创建两个 LLMChain
        let llm1 = OpenAIChat::new(create_test_config());
        let llm2 = OpenAIChat::new(create_test_config());
        
        // 第一个 Chain: 生成一个主题相关的词
        let chain1 = Arc::new(
            LLMChain::new(llm1, "只回复一个与'{topic}'相关的词语，不要其他内容")
                .with_input_key("topic")
                .with_output_key("word")
        );
        
        // 第二个 Chain: 用这个词造句
        let chain2 = Arc::new(
            LLMChain::new(llm2, "用词语'{word}'造一个简单的句子")
                .with_input_key("word")
                .with_output_key("sentence")
        );
        
        // 创建 SequentialChain
        let seq_chain = SequentialChain::new()
            .add_chain(chain1, vec!["topic"], vec!["word"])
            .add_chain(chain2, vec!["word"], vec!["sentence"]);
        
        // 执行
        let inputs = HashMap::from([
            ("topic".to_string(), Value::String("编程".to_string()))
        ]);
        
        println!("\n=== 测试 SequentialChain - 两步 LLM ===");
        let result = seq_chain.invoke(inputs).await.unwrap();
        
        println!("生成的词: {:?}", result.get("word"));
        println!("造句: {:?}", result.get("sentence"));
        
        assert!(result.contains_key("word"));
        assert!(result.contains_key("sentence"));
    }
    
    /// 真实 API 测试 - SequentialChain (三步管道)
    /// 运行: cargo test test_sequential_chain_three_steps -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_sequential_chain_three_steps() {
        let llm1 = OpenAIChat::new(create_test_config());
        let llm2 = OpenAIChat::new(create_test_config());
        let llm3 = OpenAIChat::new(create_test_config());
        
        // 第一步: 提取主题
        let chain1 = Arc::new(
            LLMChain::new(llm1, "从以下文本提取一个关键词，只回复关键词: {text}")
                .with_input_key("text")
                .with_output_key("keyword")
        );
        
        // 第二步: 解释关键词
        let chain2 = Arc::new(
            LLMChain::new(llm2, "用一句话解释'{keyword}'是什么")
                .with_input_key("keyword")
                .with_output_key("explanation")
        );
        
        // 第三步: 生成示例
        let chain3 = Arc::new(
            LLMChain::new(llm3, "为'{keyword}'生成一个简单示例")
                .with_input_key("keyword")
                .with_output_key("example")
        );
        
        // 创建 SequentialChain
        let seq_chain = SequentialChain::new()
            .add_chain(chain1, vec!["text"], vec!["keyword"])
            .add_chain(chain2, vec!["keyword"], vec!["explanation"])
            .add_chain(chain3, vec!["keyword"], vec!["example"]);
        
        // 执行
        let inputs = HashMap::from([
            ("text".to_string(), Value::String("Rust是一门系统编程语言，注重安全和性能".to_string()))
        ]);
        
        println!("\n=== 测试 SequentialChain - 三步管道 ===");
        let result = seq_chain.invoke(inputs).await.unwrap();
        
        println!("关键词: {:?}", result.get("keyword"));
        println!("解释: {:?}", result.get("explanation"));
        println!("示例: {:?}", result.get("example"));
        
        assert!(result.contains_key("keyword"));
        assert!(result.contains_key("explanation"));
        assert!(result.contains_key("example"));
    }
}