//! 扇出-聚合编排器 `FanOutFanIn`(Supervisor 的一种形态,P2-3)。

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Semaphore;

use super::{Orchestrator, RunContext};
use crate::task::AgentTask;
use crate::AgentError;

/// 扇出-聚合编排器(Supervisor 的一种形态,P2-3)。
///
/// 把同一任务广播给 N 个子编排器**并行**执行(受 `max_concurrency` 限流),
/// 全部完成后把各自的输出交给聚合函数合并。适合"多角色评审 / 委员会"场景:
/// 多个独立视角各算各的,最后统一裁决。
///
/// 默认聚合是换行拼接;可用 [`FanOutFanIn::with_aggregator`] 换成投票 /
/// 择优等自定义策略。任一 worker 失败即整体失败(不吞错)。
pub struct FanOutFanIn {
    workers: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>,
    aggregator: Box<dyn Fn(Vec<String>) -> String + Send + Sync>,
    max_concurrency: usize,
    semaphore: Arc<Semaphore>,
}

impl FanOutFanIn {
    /// 用一组同构 `AgentTask -> String` 子编排器构造,聚合默认换行拼接。
    ///
    /// 真实 Agent(`Input=String`)用 [`crate::orchestration::task_adapter`] 包装后再放进 worker 列表。
    pub fn new(workers: Vec<Arc<dyn Orchestrator<Input = AgentTask, Output = String>>>) -> Self {
        let n = workers.len().max(1);
        Self {
            workers,
            aggregator: Box::new(|results| results.join("\n")),
            max_concurrency: n,
            semaphore: Arc::new(Semaphore::new(n)),
        }
    }

    /// 自定义聚合函数(如择优、投票、拼接模板)。
    pub fn with_aggregator(
        mut self,
        aggregator: impl Fn(Vec<String>) -> String + Send + Sync + 'static,
    ) -> Self {
        self.aggregator = Box::new(aggregator);
        self
    }

    /// 限制并行 worker 数(至少 1)。
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
