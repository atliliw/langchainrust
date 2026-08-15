//! Token 追踪 LLM 包装器与成本估算

use std::sync::Arc;

use crate::language_models::LLMResult;
use crate::{BaseChatModel, RunnableConfig};
use lc_schema::Message;
use tokio::sync::Mutex;

use super::counter::{TokenCounter, TrackerTokenUsage};
use super::tiktoken::TiktokenCounter;

/// 带 Token 统计的 LLM 包装器
///
/// 包装任意 `BaseChatModel`,自动累计 prompt / completion token 用量,
/// 优先使用 LLM 返回的真实 usage,无则用 tiktoken 估算。
pub struct TokenTrackingLLM<L: BaseChatModel> {
    llm: L,
    counter: Arc<dyn TokenCounter>,
    usage: Arc<Mutex<TrackerTokenUsage>>,
}

impl<L: BaseChatModel> TokenTrackingLLM<L> {
    pub fn new(llm: L, counter: Arc<dyn TokenCounter>) -> Self {
        Self {
            llm,
            counter,
            usage: Arc::new(Mutex::new(TrackerTokenUsage::new())),
        }
    }

    /// 用 Tiktoken(cl100k_base)计数器包装
    pub fn for_openai(llm: L) -> Result<Self, String> {
        let counter = TiktokenCounter::new()?;
        Ok(Self::new(llm, Arc::new(counter)))
    }

    /// 调用 LLM 并统计 token
    pub async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, L::Error> {
        let estimated_prompt = self.counter.count_messages(&messages);
        let result = self.llm.chat(messages, config).await?;

        // 优先用 LLM 返回的真实 usage,否则用估算。
        // `TrackerTokenUsage` 与 `language_models::TokenUsage` 同为 usize，
        // 无需精度损失转换（Q6）。
        let (prompt, completion) = result
            .token_usage
            .as_ref()
            .map(|u| (u.prompt_tokens, u.completion_tokens))
            .unwrap_or((
                estimated_prompt as usize,
                self.counter.count_tokens(&result.content) as usize,
            ));

        self.usage.lock().await.add(prompt, completion);
        Ok(result)
    }

    /// 获取累计用量
    pub async fn get_usage(&self) -> TrackerTokenUsage {
        self.usage.lock().await.clone()
    }

    /// 重置统计
    pub async fn reset(&self) {
        self.usage.lock().await.reset();
    }

    /// 估算成本(美元)
    pub async fn estimate_cost(&self, pricing: &ModelPricing) -> f64 {
        let usage = self.get_usage().await;
        pricing.calculate(usage.prompt_tokens, usage.completion_tokens)
    }
}

/// 模型定价(每 1K token 价格,美元)
pub struct ModelPricing {
    pub prompt_price_per_1k: f64,
    pub completion_price_per_1k: f64,
}

impl ModelPricing {
    pub fn new(prompt: f64, completion: f64) -> Self {
        Self {
            prompt_price_per_1k: prompt,
            completion_price_per_1k: completion,
        }
    }

    /// gpt-4o-mini 定价(美元/1K token)
    pub fn gpt4o_mini() -> Self {
        Self::new(0.15, 0.60)
    }

    /// gpt-4o 定价(美元/1K token)
    pub fn gpt4o() -> Self {
        Self::new(2.50, 10.00)
    }

    /// 计算成本
    pub fn calculate(&self, prompt: usize, completion: usize) -> f64 {
        (prompt as f64 / 1000.0) * self.prompt_price_per_1k
            + (completion as f64 / 1000.0) * self.completion_price_per_1k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_pricing_gpt4o_mini() {
        let p = ModelPricing::gpt4o_mini();
        // 1000 prompt * 0.15/1k + 1000 completion * 0.60/1k = 0.75
        let cost = p.calculate(1000, 1000);
        assert!((cost - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_model_pricing_zero() {
        let p = ModelPricing::gpt4o_mini();
        assert_eq!(p.calculate(0, 0), 0.0);
    }

    #[test]
    fn test_model_pricing_custom() {
        let p = ModelPricing::new(1.0, 2.0);
        // 500 * 1.0/1k + 250 * 2.0/1k = 0.5 + 0.5 = 1.0
        let cost = p.calculate(500, 250);
        assert!((cost - 1.0).abs() < 0.001);
    }

    // NOTE: Tests that require OpenAIChat live in the lc-providers crate
    // because lc-core cannot depend on lc-providers (circular dependency).
    // The TokenTrackingLLM integration is tested there instead.
}
