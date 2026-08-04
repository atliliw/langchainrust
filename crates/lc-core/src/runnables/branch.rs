// lc-core/src/runnables/branch.rs
//! RunnableBranch - conditional routing in LCEL pipelines.
//!
//! `RunnableBranch` evaluates conditions sequentially and executes
//! the first matching branch. If no condition matches, the default
//! branch is executed.

use super::any::{into_runnable_any, RunnableAny};
use super::config::RunnableConfig;
use super::error::LcelError;
use super::lambda::RunnableLambda;
use super::runnable_trait::Runnable;
use async_trait::async_trait;
use futures_util::Stream;
use std::any::Any;
use std::pin::Pin;

/// A `Runnable` that routes input to different branches based on conditions.
///
/// Conditions are evaluated in order; the first matching branch wins.
/// If no condition matches, the default branch is used.
///
/// The input type `I` must be `Clone` because the input may need to be
/// evaluated against multiple conditions before a match is found.
///
/// # Example
///
/// ```rust,ignore
/// let branch = RunnableBranch::new(default_handler)
///     .when_fn(|input: &String| input.len() > 100, long_handler)
///     .when_fn(|input: &String| input.starts_with('?'), question_handler);
/// ```
pub struct RunnableBranch<I: Send + Sync + 'static, O: Send + Sync + 'static> {
    branches: Vec<(Box<dyn RunnableAny>, Box<dyn RunnableAny>)>,
    default: Box<dyn RunnableAny>,
    _marker: std::marker::PhantomData<(I, O)>,
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> std::fmt::Debug for RunnableBranch<I, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnableBranch")
            .field("branches", &self.branches.len())
            .field("input", &std::any::type_name::<I>())
            .field("output", &std::any::type_name::<O>())
            .finish()
    }
}

impl<I: Clone + Send + Sync + 'static, O: Send + Sync + 'static> RunnableBranch<I, O> {
    /// Create a new branch with a default runnable.
    pub fn new<R>(default: R) -> Self
    where
        R: Runnable<I, O> + 'static,
        R::Error: Into<LcelError>,
    {
        Self {
            branches: Vec::new(),
            default: into_runnable_any(default),
            _marker: std::marker::PhantomData,
        }
    }

    /// Add a branch with a condition runnable and a branch runnable.
    ///
    /// The condition runnable takes `I` and returns `bool`.
    /// If the condition returns `true`, the branch runnable is executed.
    pub fn when<R1, R2>(mut self, condition: R1, branch: R2) -> Self
    where
        R1: Runnable<I, bool> + 'static,
        R1::Error: Into<LcelError>,
        R2: Runnable<I, O> + 'static,
        R2::Error: Into<LcelError>,
    {
        self.branches.push((into_runnable_any(condition), into_runnable_any(branch)));
        self
    }

    /// Add a branch with a synchronous condition closure.
    ///
    /// Convenience method that wraps the closure in a `RunnableLambda`.
    pub fn when_fn<F, R2>(self, condition: F, branch: R2) -> Self
    where
        F: Fn(&I) -> bool + Send + Sync + 'static,
        R2: Runnable<I, O> + 'static,
        R2::Error: Into<LcelError>,
    {
        let condition_lambda = RunnableLambda::new_sync(move |input: I| condition(&input));
        self.when(condition_lambda, branch)
    }

    /// Number of branches (excluding default).
    pub fn len(&self) -> usize {
        self.branches.len()
    }

    /// Whether there are no branches (only default).
    pub fn is_empty(&self) -> bool {
        self.branches.is_empty()
    }
}

#[async_trait]
impl<I: Clone + Send + Sync + 'static, O: Send + Sync + 'static> Runnable<I, O>
    for RunnableBranch<I, O>
{
    type Error = LcelError;

    /// Evaluate conditions in order, execute the first matching branch.
    /// If no condition matches, execute the default.
    async fn invoke(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<O, LcelError> {
        for (condition, branch) in &self.branches {
            // Clone input for condition evaluation (input is preserved for branch)
            let cond_input: Box<dyn Any + Send> = Box::new(input.clone());
            let cond_result = condition.invoke_any(cond_input, config.clone()).await?;
            let matches: bool = cond_result
                .downcast::<bool>()
                .map(|b| *b)
                .map_err(|_| {
                    LcelError::TypeMismatch("branch condition must return bool".to_string())
                })?;

            if matches {
                let branch_input: Box<dyn Any + Send> = Box::new(input);
                let result = branch.invoke_any(branch_input, config).await?;
                return result.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "branch output downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                });
            }
        }

        // No condition matched, use default
        let default_input: Box<dyn Any + Send> = Box::new(input);
        let result = self.default.invoke_any(default_input, config).await?;
        result
            .downcast::<O>()
            .map(|b| *b)
            .map_err(|_| LcelError::TypeMismatch(format!(
                "branch default output downcast: expected {}",
                std::any::type_name::<O>()
            )))
    }

    /// Stream: invoke and return single-element stream.
    async fn stream(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, LcelError>> + Send>>, LcelError> {
        let result = self.invoke(input, config).await?;
        Ok(Box::pin(futures_util::stream::once(async move { Ok(result) })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RunnableConfig;
    use async_trait::async_trait;
    use futures_util::StreamExt;

    struct EchoDefault;

    #[async_trait]
    impl Runnable<String, String> for EchoDefault {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            input: String,
            _config: Option<RunnableConfig>,
        ) -> Result<String, Self::Error> {
            Ok(format!("default: {}", input))
        }
    }

    struct LongHandler;

    #[async_trait]
    impl Runnable<String, String> for LongHandler {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            input: String,
            _config: Option<RunnableConfig>,
        ) -> Result<String, Self::Error> {
            Ok(format!("long: {}", input))
        }
    }

    #[tokio::test]
    async fn branch_matches_condition() {
        let branch = RunnableBranch::new(EchoDefault).when_fn(
            |input: &String| input.len() > 5,
            LongHandler,
        );

        let result = branch.invoke("hello world".to_string(), None).await.unwrap();
        assert_eq!(result, "long: hello world");
    }

    #[tokio::test]
    async fn branch_falls_to_default() {
        let branch = RunnableBranch::new(EchoDefault).when_fn(
            |input: &String| input.len() > 100,
            LongHandler,
        );

        let result = branch.invoke("hi".to_string(), None).await.unwrap();
        assert_eq!(result, "default: hi");
    }

    #[tokio::test]
    async fn branch_first_match_wins() {
        let branch = RunnableBranch::new(EchoDefault)
            .when_fn(
                |input: &String| input.starts_with('h'),
                RunnableLambda::new_sync(|s: String| format!("starts-h: {}", s)),
            )
            .when_fn(
                |input: &String| input.len() > 3,
                RunnableLambda::new_sync(|s: String| format!("long: {}", s)),
            );

        // "hello" starts with 'h', so first branch wins
        let result = branch.invoke("hello".to_string(), None).await.unwrap();
        assert_eq!(result, "starts-h: hello");
    }

    #[tokio::test]
    async fn branch_stream_works() {
        let branch = RunnableBranch::new(EchoDefault).when_fn(
            |input: &String| input.len() > 5,
            LongHandler,
        );

        let mut stream = branch.stream("hello world".to_string(), None).await.unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result, "long: hello world");
    }
}
