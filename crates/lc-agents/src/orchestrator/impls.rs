//! `Orchestrator` trait 在四个具体 agent 上的实现。
//!
//! `PlanExecuteAgent` / `AdaptiveRAG` / `CorrectiveRAGAgent` / `DeepResearchAgent`
//! 各自把原有的 `run()` / `invoke()` / `research()` 统一桥接到
//! [`Orchestrator::run_with_context`]。

use async_trait::async_trait;
use lc_core::language_models::BaseChatModel;
use lc_rag::RetrieverTrait;

use super::{Orchestrator, RunContext};
use crate::{
    AdaptiveRAG, AdaptiveRAGResult, AgentError, CRAGResult, CorrectiveRAGAgent, DeepResearchAgent,
    PlanExecuteAgent, ResearchReport,
};

#[async_trait]
impl Orchestrator for PlanExecuteAgent {
    type Input = String;
    type Output = String;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        log::debug!(
            target: "lc_agents::orchestrator",
            "PlanExecuteAgent start, trace_id = {}",
            ctx.trace_id
        );
        self.run(&input)
            .await
            .map_err(|e| AgentError::Other(format!("PlanExecute: {e}")))
    }
}

#[async_trait]
impl<M, R> Orchestrator for AdaptiveRAG<M, R>
where
    M: BaseChatModel + Send + Sync,
    M::Error: Send + Sync,
    R: RetrieverTrait + Send + Sync,
{
    type Input = String;
    type Output = AdaptiveRAGResult;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        log::debug!(
            target: "lc_agents::orchestrator",
            "AdaptiveRAG start, trace_id = {}",
            ctx.trace_id
        );
        self.invoke(&input)
            .await
            .map_err(|e| AgentError::Other(format!("AdaptiveRAG: {e}")))
    }
}

#[async_trait]
impl<M, R> Orchestrator for CorrectiveRAGAgent<M, R>
where
    M: BaseChatModel + Send + Sync,
    M::Error: Send + Sync,
    R: RetrieverTrait + Send + Sync,
{
    type Input = String;
    type Output = CRAGResult;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        log::debug!(
            target: "lc_agents::orchestrator",
            "CorrectiveRAGAgent start, trace_id = {}",
            ctx.trace_id
        );
        self.invoke(&input)
            .await
            .map_err(|e| AgentError::Other(format!("CorrectiveRAG: {e}")))
    }
}

#[async_trait]
impl<M> Orchestrator for DeepResearchAgent<M>
where
    M: BaseChatModel + Send + Sync,
    M::Error: Send + Sync,
{
    type Input = String;
    type Output = ResearchReport;

    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError> {
        log::debug!(
            target: "lc_agents::orchestrator",
            "DeepResearchAgent start, trace_id = {}",
            ctx.trace_id
        );
        self.research(&input)
            .await
            .map_err(|e| AgentError::Other(format!("DeepResearch: {e}")))
    }
}
