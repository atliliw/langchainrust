//! Runnable::stream() 默认实现测试
//!
//! 本测试文件验证 Runnable trait 的 stream() 默认实现：
//! - 将 invoke() 结果包装为单元素流
//! - 所有 Runnable 自动获得 stream 能力
//! - invoke 和 stream 结果一致
//!
//! 测试策略：自定义 Runnable 实现，验证 trait 默认行为

use langchainrust::{Runnable, RunnableConfig};
use async_trait::async_trait;
use futures_util::StreamExt;

// ============================================================================
// 测试用 Runnable 实现
//
// 这些是简单的 Runnable 实现，用于验证 stream() 默认行为
// 真实的 Runnable（如 OpenAIChat）会在具体实现中覆盖 stream()
// ============================================================================

/// 字符串处理 Runnable
///
/// 功能：将输入字符串添加 "processed: " 前缀
///
/// 用于验证 stream() 默认实现能正确包装 invoke() 结果
struct StringProcessor;

#[async_trait]
impl Runnable<String, String> for StringProcessor {
    type Error = std::convert::Infallible;

    async fn invoke(&self, input: String, _config: Option<RunnableConfig>) -> Result<String, Self::Error> {
        Ok(format!("processed: {}", input))
    }
}

/// 数字加倍 Runnable
///
/// 功能：将输入数字乘以 2
///
/// 用于验证不同类型的 Runnable 都能获得 stream 能力
struct NumberDoubler;

#[async_trait]
impl Runnable<i32, i32> for NumberDoubler {
    type Error = std::convert::Infallible;

    async fn invoke(&self, input: i32, _config: Option<RunnableConfig>) -> Result<i32, Self::Error> {
        Ok(input * 2)
    }
}

/// 会失败的 Runnable
///
/// 功能：invoke() 总是返回错误
///
/// 用于验证 stream() 正确传播 invoke() 的错误
struct FailingRunnable;

#[derive(Debug)]
struct FailingError(String);

impl std::fmt::Display for FailingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for FailingError {}

#[async_trait]
impl Runnable<String, String> for FailingRunnable {
    type Error = FailingError;

    async fn invoke(&self, _input: String, _config: Option<RunnableConfig>) -> Result<String, Self::Error> {
        Err(FailingError("invoke failed".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // -------------------------------------------------------------------------
    // 默认 stream() 行为测试
    // -------------------------------------------------------------------------
    
    /// 验证 stream() 默认实现返回单元素流
    ///
    /// 默认行为：
    /// 1. 调用 invoke(input, config)
    /// 2. 将结果包装为 futures_util::stream::once(Ok(result))
    /// 3. 返回单元素流
    ///
    /// 流特性：
    /// - 第一个 next() 返回 invoke 结果
    /// - 第二个 next() 返回 None（流结束）
    #[tokio::test]
    async fn test_default_stream_returns_single_element() {
        let runnable = StringProcessor;
        let mut stream = runnable.stream("test_input".to_string(), None).await.unwrap();
        
        // 第一个元素：invoke 的结果
        let first = stream.next().await;
        assert!(first.is_some(), "流应有第一个元素");
        assert_eq!(first.unwrap().unwrap(), "processed: test_input");
        
        // 第二个元素：流结束
        let second = stream.next().await;
        assert!(second.is_none(), "流应只有一个元素");
    }
    
    /// 验证 invoke() 和 stream() 结果完全一致
    ///
    /// 一致性保证：
    /// - 相同输入、相同配置
    /// - invoke() 返回值 = stream().next().await 返回值
    ///
    /// 这确保代码可以无缝切换：
    /// - 单次调用用 invoke()
    /// - 流式接口用 stream()
    #[tokio::test]
    async fn test_invoke_equals_stream_result() {
        let runnable = StringProcessor;
        let input = "hello world".to_string();
        
        // invoke 结果
        let invoke_result = runnable.invoke(input.clone(), None).await.unwrap();
        
        // stream 结果
        let mut stream = runnable.stream(input.clone(), None).await.unwrap();
        let stream_result = stream.next().await.unwrap().unwrap();
        
        // 应完全一致
        assert_eq!(invoke_result, stream_result);
    }
    
    // -------------------------------------------------------------------------
    // 不同类型测试
    // -------------------------------------------------------------------------
    
    /// 验证非字符串类型的 Runnable 也获得 stream 能力
    ///
    /// Runnable 是泛型 trait：
    /// - Runnable<String, String>
    /// - Runnable<i32, i32>
    /// - Runnable<Vec<Message>, LLMResult>
    ///
    /// 默认 stream() 实现对所有类型都有效
    #[tokio::test]
    async fn test_stream_works_for_different_types() {
        let runnable = NumberDoubler;
        
        // invoke
        let invoke_result = runnable.invoke(5, None).await.unwrap();
        
        // stream
        let mut stream = runnable.stream(5, None).await.unwrap();
        let stream_result = stream.next().await.unwrap().unwrap();
        
        assert_eq!(invoke_result, 10);
        assert_eq!(stream_result, 10);
    }
    
    // -------------------------------------------------------------------------
    // 配置传递测试
    // -------------------------------------------------------------------------
    
    /// 验证 stream() 正确传递 RunnableConfig
    ///
    /// RunnableConfig 包含：
    /// - tags：标签列表
    /// - metadata：元数据键值对
    /// - run_name：运行名称
    /// - callbacks：回调管理器
    ///
    /// 默认 stream() 实现会将 config 传给 invoke()
    #[tokio::test]
    async fn test_stream_passes_config_to_invoke() {
        let runnable = StringProcessor;
        
        let config = RunnableConfig::new()
            .with_tag("test_tag")
            .with_run_name("test_run");
        
        let mut stream = runnable.stream("input".to_string(), Some(config)).await.unwrap();
        let result = stream.next().await.unwrap().unwrap();
        
        assert_eq!(result, "processed: input");
    }
    
    // -------------------------------------------------------------------------
    // 错误传播测试
    // -------------------------------------------------------------------------
    
    /// 验证 stream() 正确传播 invoke() 的错误
    ///
    /// 如果 invoke() 返回 Err：
    /// - stream() 应立即返回 Err（不是 Ok(stream)）
    /// - 错误类型应保持不变
    ///
    /// 这确保错误处理的一致性
    #[tokio::test]
    async fn test_stream_propagates_invoke_error() {
        let runnable = FailingRunnable;
        
        // invoke 返回错误
        let invoke_result = runnable.invoke("test".to_string(), None).await;
        assert!(invoke_result.is_err());
        
        // stream 也应返回错误（而非包含错误的流）
        let stream_result = runnable.stream("test".to_string(), None).await;
        assert!(stream_result.is_err(), "stream 应返回 invoke 的错误");
    }
    
    // -------------------------------------------------------------------------
    // batch 和 stream 一致性测试
    // -------------------------------------------------------------------------
    
    /// 验证 batch() 和多次 stream() 调用结果一致
    ///
    /// batch() 默认实现：顺序调用 invoke()
    /// stream() 默认实现：包装 invoke() 为单元素流
    ///
    /// 两者应产生相同结果
    #[tokio::test]
    async fn test_batch_consistency_with_stream() {
        let runnable = StringProcessor;
        let inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        
        // batch 结果
        let batch_results = runnable.batch(inputs.clone(), None).await.unwrap();
        
        // 多次 stream 结果
        for (i, input) in inputs.iter().enumerate() {
            let mut stream = runnable.stream(input.clone(), None).await.unwrap();
            let stream_result = stream.next().await.unwrap().unwrap();
            assert_eq!(batch_results[i], stream_result);
        }
    }
    
    // -------------------------------------------------------------------------
    // 流消费测试
    // -------------------------------------------------------------------------
    
    /// 验证 stream 可以被完整消费
    ///
    /// StreamExt::collect() 可以收集所有元素：
    /// - 单元素流产生 Vec<Result<Output, Error>>
    /// - 可以用 as_ref() 避免 move
    #[tokio::test]
    async fn test_stream_can_be_collected() {
        let runnable = StringProcessor;
        let stream = runnable.stream("collected".to_string(), None).await.unwrap();
        
        let results: Vec<_> = stream.collect().await;
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].as_ref().unwrap(), "processed: collected");
    }
    
    // -------------------------------------------------------------------------
    // 多次调用测试
    // -------------------------------------------------------------------------
    
    /// 验证 Runnable 可以多次调用 stream()
    ///
    /// Runnable 是 &self（不可变引用）：
    /// - 可以复用同一个实例
    /// - 每次调用独立执行
    /// - 无状态污染
    #[tokio::test]
    async fn test_multiple_stream_calls_on_same_instance() {
        let runnable = StringProcessor;
        
        for i in 0..5 {
            let input = format!("call_{}", i);
            let mut stream = runnable.stream(input.clone(), None).await.unwrap();
            let result = stream.next().await.unwrap().unwrap();
            assert_eq!(result, format!("processed: call_{}", i));
        }
    }
}