// src/agents/crag/mod.rs
//! Corrective RAG (CRAG) Agent.
//!
//! Implements the Corrective Retrieval-Augmented Generation pattern:
//!
//! ```text
//! retrieve -> grade_documents -> [transform_query + web_search | keep] -> generate
//! ```
//!
//! When retrieved documents score below a configurable threshold, the agent
//! automatically rewrites the query and optionally falls back to web search
//! before re-retrieving and regenerating the answer.
//!
//! # Example
//!
//! ```rust,ignore
//! use langchainrust::{CorrectiveRAGAgent, OpenAIChat, OpenAIConfig, SimilarityRetriever};
//!
//! let llm = OpenAIChat::new(OpenAIConfig::default());
//! let retriever = SimilarityRetriever::new(store, embeddings);
//!
//! let agent = CorrectiveRAGAgent::new(llm, retriever)
//!     .with_grade_threshold(0.6)
//!     .with_web_fallback(Box::new(DuckDuckGoSearchTool::new()));
//!
//! let result = agent.invoke("What is CRAG?").await?;
//! println!("Answer: {}", result.answer);
//! println!("Grounded: {}", result.grounded);
//! ```

pub mod grader;
pub mod graph;
pub mod rewriter;

mod error;
#[cfg(test)]
mod tests;
mod types;

pub use error::CRAGError;
pub use types::CRAGResult;

use lc_core::language_models::BaseChatModel;
use lc_core::tools::BaseTool;
use lc_rag::RetrieverTrait;

use graph::CRAGGraph;

/// Corrective RAG Agent.
///
/// Implements the CRAG pattern: retrieve documents, grade them for relevance,
/// and if the average score is below a threshold, rewrite the query and
/// optionally use a web search fallback before re-retrieving and generating
/// a new answer.
pub struct CorrectiveRAGAgent<M: BaseChatModel, R: RetrieverTrait> {
    llm: M,
    retriever: R,
    web_fallback: Option<Box<dyn BaseTool>>,
    grade_threshold: f64,
    retrieve_k: usize,
    enable_hallucination_check: bool,
    grader_llm: Option<M>,
    /// Maximum number of tokens for the context in prompts.
    max_context_tokens: Option<usize>,
}

impl<M: BaseChatModel, R: RetrieverTrait> CorrectiveRAGAgent<M, R> {
    /// Creates a new CRAG agent with the given LLM and retriever.
    ///
    /// Default grade threshold is 0.6 and default retrieve count is 4.
    pub fn new(llm: M, retriever: R) -> Self {
        Self {
            llm,
            retriever,
            web_fallback: None,
            grade_threshold: 0.6,
            retrieve_k: 4,
            enable_hallucination_check: true,
            grader_llm: None,
            max_context_tokens: None,
        }
    }

    /// Sets the web search fallback tool.
    ///
    /// When documents score below the threshold, the agent will call this
    /// tool with the rewritten query to supplement the retrieval results.
    pub fn with_web_fallback(mut self, tool: Box<dyn BaseTool>) -> Self {
        self.web_fallback = Some(tool);
        self
    }

    /// Sets the grade threshold for document relevance.
    ///
    /// Documents scoring below this threshold on average will trigger
    /// the corrective path (query rewrite + optional web search).
    /// Must be in [0.0, 1.0]. Values outside this range are clamped.
    pub fn with_grade_threshold(mut self, threshold: f64) -> Self {
        self.grade_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Sets the number of documents to retrieve.
    pub fn with_retrieve_k(mut self, k: usize) -> Self {
        self.retrieve_k = k.max(1);
        self
    }

    /// Enables or disables the hallucination check step.
    ///
    /// When enabled (default), the agent verifies that the generated
    /// answer is grounded in the source documents.
    pub fn with_hallucination_check(mut self, enable: bool) -> Self {
        self.enable_hallucination_check = enable;
        self
    }

    /// Sets a separate LLM for hallucination checking.
    ///
    /// When set, hallucination checks use this LLM instead of the main LLM,
    /// avoiding the self-verification bias where a model tends to endorse
    /// its own output. The grader LLM must be the same concrete type as the
    /// main LLM. When not set, falls back to the main LLM.
    pub fn with_grader_llm(mut self, llm: M) -> Self {
        self.grader_llm = Some(llm);
        self
    }

    /// Sets the maximum number of tokens for the context in prompts.
    ///
    /// When set, source documents are truncated from the lowest-scoring
    /// documents to fit within this budget before being passed to the LLM.
    pub fn with_max_context_tokens(mut self, tokens: usize) -> Self {
        self.max_context_tokens = Some(tokens);
        self
    }

    /// Invokes the CRAG agent on the given query.
    ///
    /// Executes the full CRAG pipeline:
    /// 1. Retrieve documents from the retriever
    /// 2. Grade each document for relevance
    /// 3. If average score < threshold: rewrite query, optionally web search, re-retrieve
    /// 4. Generate answer from filtered documents
    /// 5. Optional: hallucination check
    pub async fn invoke(&self, query: &str) -> Result<CRAGResult, CRAGError> {
        let web_ref: Option<&dyn BaseTool> = self.web_fallback.as_ref().map(|b| b.as_ref());

        let mut graph = CRAGGraph::new(&self.llm, &self.retriever, web_ref, self.grade_threshold)
            .with_retrieve_k(self.retrieve_k)
            .with_hallucination_check(self.enable_hallucination_check);

        if let Some(ref grader) = self.grader_llm {
            graph = graph.with_grader_llm(grader);
        }

        if let Some(tokens) = self.max_context_tokens {
            graph = graph.with_max_context_tokens(tokens);
        }

        graph.run(query).await
    }

    /// Streams the CRAG agent execution, emitting pipeline step events.
    ///
    /// Emits `AgentStreamEvent::PipelineStep` events at each stage of the
    /// CRAG pipeline, and `AgentStreamEvent::FinalAnswer` when the answer
    /// is ready.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use futures_util::StreamExt;
    ///
    /// let mut stream = agent.stream("What is CRAG?").await?;
    /// while let Some(event) = stream.next().await {
    ///     match event {
    ///         AgentStreamEvent::PipelineStep { step, detail } => {
    ///             println!("Step: {} — {:?}", step, detail);
    ///         }
    ///         AgentStreamEvent::FinalAnswer { content } => {
    ///             println!("Answer: {}", content);
    ///         }
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub async fn stream(
        &self,
        query: &str,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures_util::Stream<Item = crate::streaming::AgentStreamEvent> + Send>,
        >,
        CRAGError,
    > {
        use crate::streaming::AgentStreamEvent;
        use graph::CRAGState;

        let web_ref: Option<&dyn BaseTool> = self.web_fallback.as_ref().map(|b| b.as_ref());
        let grade_threshold = self.grade_threshold;
        let retrieve_k = self.retrieve_k;
        let enable_hallucination_check = self.enable_hallucination_check;
        let max_context_tokens = self.max_context_tokens;

        let mut graph = CRAGGraph::new(&self.llm, &self.retriever, web_ref, grade_threshold)
            .with_retrieve_k(retrieve_k)
            .with_hallucination_check(enable_hallucination_check);

        if let Some(ref grader) = self.grader_llm {
            graph = graph.with_grader_llm(grader);
        }
        if let Some(tokens) = max_context_tokens {
            graph = graph.with_max_context_tokens(tokens);
        }

        let mut events: Vec<AgentStreamEvent> = Vec::new();
        let mut state = CRAGState::new(query);

        // Step 1: Retrieve
        events.push(AgentStreamEvent::PipelineStep {
            step: "retrieving".to_string(),
            detail: Some("Retrieving documents...".to_string()),
        });

        graph.retrieve(&mut state).await?;

        events.push(AgentStreamEvent::PipelineStep {
            step: "retrieved".to_string(),
            detail: Some(format!("Retrieved {} documents", state.documents.len())),
        });

        // Step 2: Grade
        events.push(AgentStreamEvent::PipelineStep {
            step: "grading".to_string(),
            detail: Some("Grading document relevance...".to_string()),
        });

        graph.grade_documents(&mut state).await?;

        events.push(AgentStreamEvent::PipelineStep {
            step: "graded".to_string(),
            detail: Some(format!("Average grade score: {:.2}", state.avg_score)),
        });

        // Step 3: Correct if needed
        if state.avg_score < grade_threshold {
            events.push(AgentStreamEvent::PipelineStep {
                step: "correcting".to_string(),
                detail: Some("Score below threshold, rewriting query...".to_string()),
            });

            graph.correct(&mut state).await?;

            events.push(AgentStreamEvent::PipelineStep {
                step: "corrected".to_string(),
                detail: Some(format!("Query rewritten: {}", state.query_rewritten)),
            });
        }

        // Step 4: Filter + Generate
        events.push(AgentStreamEvent::PipelineStep {
            step: "generating".to_string(),
            detail: Some("Generating answer...".to_string()),
        });

        let filtered: Vec<lc_vector_stores::Document> = state
            .documents
            .iter()
            .zip(state.grade_scores.iter())
            .filter(|(_, &score)| score >= grade_threshold)
            .map(|(doc, _)| doc.clone())
            .collect();

        let source_docs = if filtered.is_empty() {
            Vec::new()
        } else {
            filtered
        };

        let reasoning_section = graph::format_reasoning(&state.grade_reasoning);
        graph
            .generate(&mut state, &source_docs, &reasoning_section)
            .await?;

        // Step 5: Hallucination check
        if enable_hallucination_check && state.answer.is_some() {
            events.push(AgentStreamEvent::PipelineStep {
                step: "hallucination_check".to_string(),
                detail: Some("Checking answer grounding...".to_string()),
            });

            let _ = graph.hallucination_check(&mut state, &source_docs).await;
        }

        // Final answer
        events.push(AgentStreamEvent::FinalAnswer {
            content: state.answer.unwrap_or_default(),
        });

        Ok(Box::pin(futures_util::stream::iter(events)))
    }
}
