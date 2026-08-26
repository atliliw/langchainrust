//! lc-testkit 的错误类型与到 `ProviderError` 的桥接。

use lc_providers::ProviderError;

/// lc-testkit 统一错误。
///
/// 通过 [`From<TestkitError> for ProviderError`] 桥接,让录播/回放 provider
/// 可以直接喂给 chains 等要求 `L::Error: Into<ProviderError>` 的泛型入口。
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TestkitError {
    /// 录制/回放过程中的 IO 错误(读文件、写文件等)。
    #[error("io error while recording/replaying: {0}")]
    Io(#[from] std::io::Error),
    /// 回放队列耗尽:请求条数超过录制条数。
    #[error("replay queue exhausted (requested {requested} messages, no recording left)")]
    ReplayExhausted { requested: usize },
    /// `ReplayStrategy::Exact` 下请求消息签名在录播中无匹配(显式报错,不做
    /// 静默 FIFO 兜底);`left` 为队列剩余条数,便于排查录播与请求的字段漂移。
    #[error("replay has no recording matching request messages (strategy=Exact, {left} exchange(s) left)")]
    ReplayNoMatch { left: usize },
    /// 内层模型错误,无损透传真实 provider 错误。
    #[error("inner model error: {0}")]
    Inner(#[from] ProviderError),
}

impl From<TestkitError> for ProviderError {
    fn from(e: TestkitError) -> Self {
        match e {
            // 无损透传真实 provider 错误。
            TestkitError::Inner(p) => p,
            // 其余为 testkit 自身错误 → 经 From<String> 落到 ProviderError::Testkit。
            other => other.to_string().into(),
        }
    }
}
