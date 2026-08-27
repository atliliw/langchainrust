//! Fan-out/fan-in orchestrator `FanOutFanIn` (a form of Supervisor, P2-3).

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Semaphore;

use super::{Orchestrator, RunContext};
use crate::task::AgentTask;
use crate::AgentError;

/// Fan-out/fan-in orchestrator (a form of Supervisor, P2-3).
///
/// Broadcasts the same task to N child orchestrators running **in parallel**
/// (rate-limited by `max_concurrency`), then merges their outputs with an
/// aggregator once all complete. Suited to "multi-role review / committee"
/// scenarios: independent perspectives each compute their own, and a unified
/// verdict is reached at the end.
///
/// The default aggregator joins with newlines; swap it via
/// [`FanOutFanIn::with_aggregator`] for voting / best-of custom strategies.
/// Any worker failure fails the whole run (errors are not swallowed).
pub struct FanOutFanIn {
    workers: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>,
    aggregator: Box<dyn Fn(Vec<String>) -> String + Send + Sync>,
    max_concurrency: usize,
    semaphore: Arc<Semaphore>,
}

impl FanOutFanIn {
    /// Construct from a set of homogeneous `AgentTask -> String` child orchestrators;
    /// aggregation defaults to newline-joining.
    ///
    /// Real agents (`Input=String`) should be wrapped with
    /// [`crate::orchestration::task_adapter`] before being put into the worker list.
    pub fn new(workers: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>) -> Self {
        let n = workers.len().max(1);
        Self {
            workers,
            aggregator: Box::new(|results| results.join("\n")),
            max_concurrency: n,
            semaphore: Arc::new(Semaphore::new(n)),
        }
    }

    /// Custom aggregator function (e.g. best-of, voting, join templates).
    pub fn with_aggregator(
        mut self,
        aggregator: impl Fn(Vec<String>) -> String + Send + Sync + 'static,
    ) -> Self {
        self.aggregator = Box::new(aggregator);
        self
    }

    /// Cap the number of parallel workers (at least 1).
    pub fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self.semaphore = Arc::new(Semaphore::new(self.max_concurrency));
        self
    }
}

#[async_trait]
impl Orchestrator for FanOutFanIn {
    type Input = AgentTask;
    type Output = String;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        if self.workers.is_empty() {
            return Err(AgentError::Other(
                "FanOutFanIn requires at least one worker".to_string(),
            ));
        }
        log::debug!(
            target: "lc_agents::orchestrator",
            "FanOutFanIn start workers={} concurrency={} trace_id={}",
            self.workers.len(),
            self.max_concurrency,
            ctx.trace_id
        );

        let futures = self.workers.iter().enumerate().map(|(i, worker)| {
            let worker = worker.clone();
            let input = input.clone();
            let ctx = ctx.clone();
            let sem = self.semaphore.clone();
            async move {
                let _permit = sem
                    .acquire_owned()
                    .await
                    .map_err(|e| AgentError::Other(format!("FanOutFanIn semaphore: {e}")))?;
                worker
                    .run_with_context(input, &ctx)
                    .await
                    .map_err(|e| AgentError::Other(format!("worker {i} failed: {e}")))
            }
        });

        let outputs = futures_util::future::join_all(futures).await;
        let mut results = Vec::with_capacity(outputs.len());
        for output in outputs {
            results.push(output?);
        }
        Ok((self.aggregator)(results))
    }
}
