//! Token-tracking LLM wrapper and cost estimation

use std::sync::Arc;

use crate::language_models::LLMResult;
use crate::{BaseChatModel, RunnableConfig};
use lc_schema::Message;
use tokio::sync::Mutex;

use super::counter::{TokenCounter, TrackerTokenUsage};
use super::tiktoken::TiktokenCounter;
use super::TokenCounterError;

/// LLM wrapper with token statistics
///
/// Wraps any `BaseChatModel`, accumulating prompt / completion token usage automatically,
/// preferring the real usage returned by the LLM, falling back to tiktoken estimates.
pub struct TokenTrackingLLM<L: BaseChatModel> {
    llm: L,
    counter: Arc<dyn TokenCounter>,
    usage: Arc<Mutex<TrackerTokenUsage>>,
}

impl<L: BaseChatModel> TokenTrackingLLM<L> {
    /// Wraps an LLM with a custom counter.
    pub fn new(llm: L, counter: Arc<dyn TokenCounter>) -> Self {
        Self {
            llm,
            counter,
            usage: Arc::new(Mutex::new(TrackerTokenUsage::new())),
        }
    }

    /// Wraps with a Tiktoken (cl100k_base) counter
    pub fn for_openai(llm: L) -> Result<Self, TokenCounterError> {
        let counter = TiktokenCounter::new()?;
        Ok(Self::new(llm, Arc::new(counter)))
    }

    /// Calls the LLM and counts tokens
    pub async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, L::Error> {
        let estimated_prompt = self.counter.count_messages(&messages);
        let result = self.llm.chat(messages, config).await?;

        // prefer the real usage returned by the LLM, otherwise use the estimate.
        // `TrackerTokenUsage` and `language_models::TokenUsage` are both usize,
        // so no precision-loss conversion is needed (Q6).
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

    /// Returns the cumulative usage
    pub async fn get_usage(&self) -> TrackerTokenUsage {
        self.usage.lock().await.clone()
    }

    /// Resets the statistics
    pub async fn reset(&self) {
        self.usage.lock().await.reset();
    }

    /// Estimates the cost (USD)
    pub async fn estimate_cost(&self, pricing: &ModelPricing) -> f64 {
        let usage = self.get_usage().await;
        pricing.calculate(usage.prompt_tokens, usage.completion_tokens)
    }
}

/// Model pricing (per 1K tokens, USD)
pub struct ModelPricing {
    /// Per-1K prompt token price (USD)
    pub prompt_price_per_1k: f64,
    /// Per-1K completion token price (USD)
    pub completion_price_per_1k: f64,
}

impl ModelPricing {
    /// Creates custom model pricing.
    pub fn new(prompt: f64, completion: f64) -> Self {
        Self {
            prompt_price_per_1k: prompt,
            completion_price_per_1k: completion,
        }
    }

    /// gpt-4o-mini pricing (USD / 1K tokens)
    pub fn gpt4o_mini() -> Self {
        Self::new(0.15, 0.60)
    }

    /// gpt-4o pricing (USD / 1K tokens)
    pub fn gpt4o() -> Self {
        Self::new(2.50, 10.00)
    }

    /// Calculates the cost
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
