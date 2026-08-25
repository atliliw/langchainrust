// lc-core/src/runnables/sequence.rs
//! RunnableSequence - the core LCEL pipeline type.
//!
//! A `RunnableSequence<I, O>` chains multiple `Runnable` steps together,
//! where the output of each step feeds into the input of the next.
//! Internally, steps are stored as `Box<dyn RunnableAny>` (type-erased),
//! but the `I` and `O` type parameters preserve the pipeline's
//! input and output types at compile time.

use super::any::{into_runnable_any, RunnableAny};
use super::config::RunnableConfig;
use super::error::LcelError;
use super::runnable_trait::Runnable;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use std::any::Any;
use std::marker::PhantomData;
use std::pin::Pin;

/// A sequence of `Runnable` steps composed into a pipeline.
///
/// Created via the `pipe()` method on `RunnableExt`, or directly
/// with `from_single` / `from_pair`.
///
/// # Type Safety
///
/// The `I` and `O` type parameters represent the pipeline's overall
/// input and output types. Intermediate types are erased at runtime
/// via `RunnableAny`, but the compiler guarantees type compatibility
/// at each `pipe()` call site.
///
/// # Flattening
///
/// When two `RunnableSequence` values are piped together, their
/// internal steps are merged (flattened) rather than nested,
/// avoiding unnecessary indirection.
pub struct RunnableSequence<I: Send + Sync + 'static, O: Send + Sync + 'static> {
    steps: Vec<Box<dyn RunnableAny>>,
    _marker: PhantomData<(I, O)>,
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> std::fmt::Debug
    for RunnableSequence<I, O>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnableSequence")
            .field("steps", &self.steps.len())
            .field("input", &std::any::type_name::<I>())
            .field("output", &std::any::type_name::<O>())
            .finish()
    }
}

impl<I: Send + Sync + 'static, O: Send + Sync + 'static> RunnableSequence<I, O> {
    /// Create a sequence from a single runnable step.
    pub fn from_single<R>(runnable: R) -> Self
    where
        R: Runnable<I, O> + 'static,
        R::Error: Into<LcelError>,
    {
        Self {
            steps: vec![into_runnable_any(runnable)],
            _marker: PhantomData,
        }
    }

    /// Create a sequence from two runnable steps.
    ///
    /// The output type of the first must match the input type of the second.
    pub fn from_pair<R1, R2, M>(first: R1, second: R2) -> RunnableSequence<I, O>
    where
        M: Send + Sync + 'static,
        R1: Runnable<I, M> + 'static,
        R1::Error: Into<LcelError>,
        R2: Runnable<M, O> + 'static,
        R2::Error: Into<LcelError>,
    {
        Self {
            steps: vec![into_runnable_any(first), into_runnable_any(second)],
            _marker: PhantomData,
        }
    }

    /// Append a step to this sequence, returning a new sequence
    /// with the updated output type.
    pub fn pipe<O2, R>(self, other: R) -> RunnableSequence<I, O2>
    where
        O2: Send + Sync + 'static,
        R: Runnable<O, O2> + Send + Sync + 'static,
        R::Error: Into<LcelError>,
    {
        let mut steps = self.steps;
        steps.push(into_runnable_any(other));
        RunnableSequence {
            steps,
            _marker: PhantomData,
        }
    }

    /// Number of steps in this sequence.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether this sequence has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Access the steps as a slice.
    pub fn steps(&self) -> &[Box<dyn RunnableAny>] {
        &self.steps
    }
}

#[async_trait]
impl<I: Send + Sync + 'static, O: Send + Sync + 'static> Runnable<I, O> for RunnableSequence<I, O> {
    type Error = LcelError;

    /// Execute the pipeline: feed input through each step sequentially.
    async fn invoke(&self, input: I, config: Option<RunnableConfig>) -> Result<O, LcelError> {
        let mut current: Box<dyn Any + Send> = Box::new(input);
        for step in &self.steps {
            current = step.invoke_any(current, config.clone()).await?;
        }
        current.downcast::<O>().map(|b| *b).map_err(|_| {
            LcelError::TypeMismatch(format!(
                "final downcast failed: expected {}",
                std::any::type_name::<O>()
            ))
        })
    }

    /// Batch processing: each step processes all inputs before
    /// passing to the next step. This allows LLM providers to
    /// optimize batch requests.
    async fn batch(
        &self,
        inputs: Vec<I>,
        config: Option<RunnableConfig>,
    ) -> Result<Vec<O>, LcelError> {
        let mut current: Vec<Box<dyn Any + Send>> = inputs
            .into_iter()
            .map(|i| Box::new(i) as Box<dyn Any + Send>)
            .collect();

        for step in &self.steps {
            current = step.batch_any(current, config.clone()).await?;
        }

        current
            .into_iter()
            .map(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "batch final downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
            .collect()
    }

    /// Streaming: the first step runs through `stream_any` (single input →
    /// item stream) and every following step runs through `transform_any`.
    /// Steps that override `stream` (e.g. LLMs) therefore emit a real token
    /// stream instead of being collapsed to a single `invoke`.
    async fn stream(
        &self,
        input: I,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, LcelError>> + Send>>, LcelError> {
        if self.steps.is_empty() {
            return Ok(Box::pin(futures_util::stream::empty()));
        }

        let mut steps = self.steps.iter();
        let Some(first) = steps.next() else {
            return Ok(Box::pin(futures_util::stream::empty()));
        };

        let input_boxed: Box<dyn Any + Send> = Box::new(input);
        let mut current_stream = first.stream_any(input_boxed, config.clone()).await?;

        // Chain each subsequent step's transform
        for step in steps {
            current_stream = step.transform_any(current_stream, config.clone()).await?;
        }

        // Downcast the final stream from Any to O
        let output_stream = current_stream.map(|result| {
            result.and_then(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "stream final downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
        });

        Ok(Box::pin(output_stream))
    }

    /// Transform: chain each step's transform to enable
    /// stream-to-stream pipeline processing.
    async fn transform(
        &self,
        input: Pin<Box<dyn Stream<Item = Result<I, LcelError>> + Send>>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<O, LcelError>> + Send + '_>>, LcelError> {
        // Upcast input stream from I to Any
        let mut current_stream: Pin<
            Box<dyn Stream<Item = Result<Box<dyn Any + Send>, LcelError>> + Send>,
        > = Box::pin(input.map(|result| result.map(|item| Box::new(item) as Box<dyn Any + Send>)));

        // Chain each step's transform
        for step in &self.steps {
            current_stream = step.transform_any(current_stream, config.clone()).await?;
        }

        // Downcast the final stream from Any to O
        let output_stream = current_stream.map(|result| {
            result.and_then(|boxed| {
                boxed.downcast::<O>().map(|b| *b).map_err(|_| {
                    LcelError::TypeMismatch(format!(
                        "transform final downcast: expected {}",
                        std::any::type_name::<O>()
                    ))
                })
            })
        });

        Ok(Box::pin(output_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    struct Double;

    #[async_trait]
    impl Runnable<i32, i32> for Double {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            input: i32,
            _config: Option<RunnableConfig>,
        ) -> Result<i32, Self::Error> {
            Ok(input * 2)
        }
    }

    struct AddOne;

    #[async_trait]
    impl Runnable<i32, i32> for AddOne {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            input: i32,
            _config: Option<RunnableConfig>,
        ) -> Result<i32, Self::Error> {
            Ok(input + 1)
        }
    }

    struct I32ToString;

    #[async_trait]
    impl Runnable<i32, String> for I32ToString {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            input: i32,
            _config: Option<RunnableConfig>,
        ) -> Result<String, Self::Error> {
            Ok(format!("value={}", input))
        }
    }

    #[tokio::test]
    async fn invoke_two_steps() {
        // Double → AddOne: 5 * 2 + 1 = 11
        let seq = RunnableSequence::from_pair(Double, AddOne);
        let result = seq.invoke(5, None).await.unwrap();
        assert_eq!(result, 11);
    }

    #[tokio::test]
    async fn invoke_three_steps() {
        // Double → AddOne → I32ToString: 3 * 2 + 1 = "value=7"
        let seq = RunnableSequence::from_pair(Double, AddOne).pipe(I32ToString);
        let result = seq.invoke(3, None).await.unwrap();
        assert_eq!(result, "value=7");
    }

    #[tokio::test]
    async fn batch_works() {
        let seq = RunnableSequence::from_pair(Double, AddOne);
        let results = seq.batch(vec![1, 2, 3], None).await.unwrap();
        assert_eq!(results, vec![3, 5, 7]);
    }

    #[tokio::test]
    async fn stream_works() {
        let seq = RunnableSequence::from_pair(Double, AddOne);
        let mut stream = seq.stream(10, None).await.unwrap();
        let result = stream.next().await.unwrap().unwrap();
        assert_eq!(result, 21);
    }

    #[tokio::test]
    async fn transform_works_elementwise() {
        let seq = RunnableSequence::from_pair(Double, AddOne);
        let input = Box::pin(futures_util::stream::iter(vec![
            Ok(1i32),
            Ok(2i32),
            Ok(3i32),
        ])) as Pin<Box<dyn Stream<Item = Result<i32, LcelError>> + Send>>;

        // Default transform maps each item through the chain elementwise
        // (LangChain default semantics): 1→3, 2→5, 3→7.
        let mut output = seq.transform(input, None).await.unwrap();
        let mut results = Vec::new();
        while let Some(item) = output.next().await {
            results.push(item.unwrap());
        }
        assert_eq!(results, vec![3, 5, 7]);
    }

    // A runnable that overrides `stream` to emit several items, mimicking an
    // LLM token stream. `invoke` returns a distinguishable value so tests can
    // prove the streaming path was actually taken.
    struct StreamingTokenLLM;

    #[async_trait]
    impl Runnable<i32, i32> for StreamingTokenLLM {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            input: i32,
            _config: Option<RunnableConfig>,
        ) -> Result<i32, Self::Error> {
            Ok(input * 1000)
        }

        async fn stream(
            &self,
            input: i32,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<i32, Self::Error>> + Send>>, Self::Error>
        {
            let stream =
                futures_util::stream::iter(vec![Ok(input), Ok(input + 100), Ok(input + 200)]);
            Ok(Box::pin(stream))
        }
    }

    #[tokio::test]
    async fn stream_uses_real_streaming_for_first_step() {
        // Single step with a real `stream` override: `sequence.stream` must
        // emit every streamed item (proving it goes through `stream_any`, not
        // a single `invoke`).
        let seq: RunnableSequence<i32, i32> = RunnableSequence::from_single(StreamingTokenLLM);
        let mut stream = seq.stream(5, None).await.unwrap();
        let mut results = Vec::new();
        while let Some(item) = stream.next().await {
            results.push(item.unwrap());
        }
        assert_eq!(results, vec![5, 105, 205]);
    }

    #[tokio::test]
    async fn stream_chains_subsequent_steps_elementwise() {
        // StreamingTokenLLM → AddOne: the token stream [5, 105, 205] is
        // transformed elementwise by AddOne → [6, 106, 206].
        let seq = RunnableSequence::from_pair(StreamingTokenLLM, AddOne);
        let mut stream = seq.stream(5, None).await.unwrap();
        let mut results = Vec::new();
        while let Some(item) = stream.next().await {
            results.push(item.unwrap());
        }
        assert_eq!(results, vec![6, 106, 206]);
    }

    #[tokio::test]
    async fn from_single_works() {
        let seq: RunnableSequence<i32, i32> = RunnableSequence::from_single(Double);
        let result = seq.invoke(4, None).await.unwrap();
        assert_eq!(result, 8);
    }

    #[tokio::test]
    async fn pipe_on_sequence_works() {
        let seq = RunnableSequence::from_single(Double)
            .pipe(AddOne)
            .pipe(I32ToString);
        let result = seq.invoke(5, None).await.unwrap();
        assert_eq!(result, "value=11"); // 5*2+1=11
    }

    #[tokio::test]
    async fn len_and_empty() {
        let seq = RunnableSequence::from_pair(Double, AddOne);
        assert_eq!(seq.len(), 2);
        assert!(!seq.is_empty());
    }
}
