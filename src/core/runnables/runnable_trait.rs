// src/core/runnables/runnable_trait.rs
//! Runnable trait - LCEL (LangChain Expression Language) 的基础
//!
//! LangChain 中的每个组件都实现 Runnable，使它们可以
//! 链式调用、组合和互操作。

use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;
use super::RunnableConfig;

/// LangChain 所有组件的基础 trait
///
/// 这个 trait 定义了每个组件必须实现的核心接口：
/// - 单次执行 via `invoke`
/// - 批量处理 via `batch`
/// - 流式输出 via `stream`
///
/// # 示例
/// ```rust
/// use langchainrust::core::runnables::Runnable;
/// use langchainrust::RunnableConfig;
/// use async_trait::async_trait;
///
/// // 定义一个简单的 Runnable：加一
/// struct AddOne;
///
/// #[async_trait]
/// impl Runnable<i32, i32> for AddOne {
///     type Error = std::convert::Infallible;
///
///     async fn invoke(&self, input: i32, _config: Option<RunnableConfig>) -> Result<i32, Self::Error> {
///         Ok(input + 1)
///     }
/// }
/// ```
#[async_trait]
pub trait Runnable<Input: Send + Sync + 'static, Output: Send + Sync + 'static>: Send + Sync {
    /// 错误类型
    type Error: std::error::Error + Send + Sync + 'static;

    /// 将单个输入转换为输出
    ///
    /// 这是单次执行的主要方法。
    ///
    /// # 参数
    /// * `input` - 要处理的输入
    /// * `config` - 可选的执行配置
    ///
    /// # 返回
    /// * `Result<Output, Self::Error>` - 执行结果
    async fn invoke(&self, input: Input, config: Option<RunnableConfig>) -> Result<Output, Self::Error>;

    /// 批量处理 - 将多个输入转换为多个输出
    ///
    /// 默认实现是顺序处理输入。
    /// 可以重写此方法以实现并发执行或优化批处理。
    ///
    /// # 参数
    /// * `inputs` - 输入向量
    /// * `config` - 可选的批处理配置
    ///
    /// # 返回
    /// * `Result<Vec<Output>, Self::Error>` - 结果向量
    async fn batch(
        &self,
        inputs: Vec<Input>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<Output>, Self::Error> {
        let mut results = Vec::with_capacity(inputs.len());

        // 顺序处理每个输入
        for input in inputs {
            let result = self.invoke(input, config.clone()).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// 流式输出 - 用于流式响应 (LLM 等)
    ///
    /// 此方法启用输出的实时流式处理，
    /// 适用于聊天模型、token 生成等场景。
    ///
    /// # 参数
    /// * `input` - 要处理的输入
    /// * `config` - 可选配置
    ///
    /// # 返回
    /// * `Result<Pin<Box<dyn Stream<Item = Result<Output, Self::Error>> + Send>>, Self::Error>` - 输出流
    ///
    /// # 默认实现
    /// 默认将 invoke 结果包装为单元素流。
    /// 支持真正流式的类型应重写此方法。
    async fn stream(
        &self,
        input: Input,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Output, Self::Error>> + Send>>, Self::Error> {
        // 默认实现：将 invoke 结果包装为单元素流
        // 这样所有 Runnable 都自动获得 stream 能力
        // 支持真正流式的类型（如 LLM）应重写此方法
        let result = self.invoke(input, config).await?;
        let stream = futures_util::stream::once(async move { Ok(result) });
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    struct TestRunnable;

    #[async_trait]
    impl Runnable<String, String> for TestRunnable {
        type Error = std::convert::Infallible;

        async fn invoke(&self, input: String, _config: Option<RunnableConfig>) -> Result<String, Self::Error> {
            Ok(format!("processed: {}", input))
        }
    }

    #[tokio::test]
    async fn test_default_stream_returns_single_element() {
        let runnable = TestRunnable;
        let mut stream = runnable.stream("test".to_string(), None).await.unwrap();
        
        let first = stream.next().await;
        assert!(first.is_some());
        assert_eq!(first.unwrap().unwrap(), "processed: test");
        
        let second = stream.next().await;
        assert!(second.is_none());
    }

    #[tokio::test]
    async fn test_invoke_matches_stream_result() {
        let runnable = TestRunnable;
        
        let invoke_result = runnable.invoke("hello".to_string(), None).await.unwrap();
        let mut stream = runnable.stream("hello".to_string(), None).await.unwrap();
        let stream_result = stream.next().await.unwrap().unwrap();
        
        assert_eq!(invoke_result, stream_result);
    }
}