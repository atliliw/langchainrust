// src/core/language_models/base.rs
//! Language model base trait.

use crate::core::runnables::Runnable;
use async_trait::async_trait;

/// Base trait for all language models.
///
/// All LLM wrappers inherit from this base class.
/// It extends the Runnable interface for unified invocation.
#[async_trait]
pub trait BaseLanguageModel<Input: Send + Sync + 'static, Output: Send + Sync + 'static>:
    Runnable<Input, Output>
{
    /// Returns the model name.
    fn model_name(&self) -> &str;

    /// Calculates token count for text.
    ///
    /// # Arguments
    /// * `text` - Text to count tokens for.
    ///
    /// # Returns
    /// Token count.
    fn get_num_tokens(&self, text: &str) -> usize;

    /// Returns the temperature parameter.
    fn temperature(&self) -> Option<f32> {
        None
    }

    /// Returns the max tokens limit.
    fn max_tokens(&self) -> Option<usize> {
        None
    }

    /// Sets the temperature parameter.
    fn with_temperature(self, temp: f32) -> Self
    where
        Self: Sized;

    /// Sets the max tokens limit.
    fn with_max_tokens(self, max: usize) -> Self
    where
        Self: Sized;
}
