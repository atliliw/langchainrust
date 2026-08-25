// src/agents/deep_research/mod.rs
//! Deep Research Agent: multi-round research with sub-topic decomposition,
//! parallel search, and comprehensive report synthesis with citations.
//!
//! # Flow
//!
//! 1. **Planner** - LLM decomposes the research topic into sub-topics and
//!    generates search queries for each.
//! 2. **Searcher** - Executes searches in parallel across all configured
//!    search tools, collecting and deduplicating results.
//! 3. **Synthesizer** - Aggregates findings and uses the LLM to write a
//!    comprehensive markdown report with inline citations.
//! 4. **Multi-round** - If information gaps remain after synthesis, the
//!    agent generates follow-up queries and repeats the search-synthesize
//!    cycle up to `max_rounds`.
//!
//! # Example
//!
//! ```ignore
//! use langchainrust::agents::deep_research::DeepResearchAgent;
//! use langchainrust::DuckDuckGoSearchTool;
//!
//! let agent = DeepResearchAgent::new(llm)
//!     .with_searcher(Box::new(DuckDuckGoSearchTool::new()))
//!     .with_max_rounds(3)
//!     .with_max_subtopics(5);
//!
//! let report = agent.research("Impact of AI on healthcare").await?;
//! println!("{}", report.markdown);
//! ```

pub mod planner;
pub mod searcher;
pub mod synthesizer;

mod error;
#[cfg(test)]
mod tests;
mod types;

pub use error::ResearchError;
pub use planner::{ResearchPlan, SubTopic};
pub use searcher::{SearchCollector, SearchResult};
pub use synthesizer::SynthesisOutput;
pub use types::{Citation, ResearchReport};

use lc_core::language_models::BaseChatModel;
use lc_core::tools::BaseTool;
use lc_schema::Message;

use types::parse_gap_queries;

/// Multi-round deep research agent.
///
/// Decomposes a topic into sub-topics, searches in parallel, synthesizes
/// a comprehensive report with citations, and iterates if gaps remain.
pub struct DeepResearchAgent<M: BaseChatModel> {
    llm: M,
    searchers: Vec<Box<dyn BaseTool>>,
    max_rounds: usize,
    max_subtopics: usize,
    /// Maximum number of tokens for source text in synthesis prompts.
    max_source_tokens: Option<usize>,
}

impl<M: BaseChatModel> DeepResearchAgent<M> {
    /// Creates a new `DeepResearchAgent` with the given LLM.
    ///
    /// At least one search tool must be added via `with_searcher` before
    /// calling `research`.
    pub fn new(llm: M) -> Self {
        Self {
            llm,
            searchers: Vec::new(),
            max_rounds: 2,
            max_subtopics: 5,
            max_source_tokens: None,
        }
    }

    /// Adds a search tool to the agent.
    ///
    /// Multiple search tools can be added; all are queried in parallel.
    pub fn with_searcher(mut self, tool: Box<dyn BaseTool>) -> Self {
        self.searchers.push(tool);
        self
    }

    /// Sets the maximum number of research rounds (default: 2).
    pub fn with_max_rounds(mut self, n: usize) -> Self {
        self.max_rounds = n.max(1);
        self
    }

    /// Sets the maximum number of sub-topics to decompose (default: 5).
    pub fn with_max_subtopics(mut self, n: usize) -> Self {
        self.max_subtopics = n.max(1);
        self
    }

    /// Sets the maximum number of tokens for source text in synthesis prompts.
    ///
    /// When set, source snippets are truncated to fit within this budget.
    pub fn with_max_source_tokens(mut self, tokens: usize) -> Self {
        self.max_source_tokens = Some(tokens);
        self
    }

    /// Runs the full deep research pipeline on the given topic.
    ///
    /// Returns a `ResearchReport` containing the markdown report,
    /// citations, sub-topics, and round count.
    pub async fn research(&self, topic: &str) -> Result<ResearchReport, ResearchError> {
        if self.searchers.is_empty() {
            return Err(ResearchError::Search(
                "no search tools configured; add at least one with with_searcher()".to_string(),
            ));
        }

        let mut all_results: Vec<SearchResult> = Vec::new();
        let mut rounds_completed: usize = 0;
        let current_plan = self.plan(topic).await?;
        let mut follow_up_queries: Vec<String> = Vec::new();

        for round in 0..self.max_rounds {
            let queries = if round == 0 {
                current_plan.all_queries()
            } else {
                follow_up_queries.clone()
            };

            if queries.is_empty() {
                break;
            }

            let round_results = self.search(&queries).await?;
            all_results.extend(round_results);

            // Deduplicate after each round
            all_results = SearchCollector::dedup(all_results);

            rounds_completed = round + 1;

            // Synthesize report from accumulated results
            let (markdown, gaps) = self
                .synthesize(topic, &current_plan, &all_results, self.max_source_tokens)
                .await?;

            if gaps.is_empty() || round + 1 >= self.max_rounds {
                let citations = self.build_citations(&all_results);
                return Ok(ResearchReport {
                    markdown,
                    citations,
                    subtopics: current_plan
                        .subtopics
                        .iter()
                        .map(|s| s.name.clone())
                        .collect(),
                    rounds_completed,
                });
            }

            // Generate follow-up queries for the next round
            follow_up_queries = self.generate_follow_ups(topic, &gaps).await?;
        }

        // Final synthesis if we exhausted rounds
        let (markdown, _) = self
            .synthesize(topic, &current_plan, &all_results, self.max_source_tokens)
            .await?;
        let citations = self.build_citations(&all_results);
        Ok(ResearchReport {
            markdown,
            citations,
            subtopics: current_plan
                .subtopics
                .iter()
                .map(|s| s.name.clone())
                .collect(),
            rounds_completed,
        })
    }

    /// Streams the deep research execution, emitting pipeline step events.
    ///
    /// Emits `AgentStreamEvent::PipelineStep` events for planning, searching,
    /// and synthesizing, and `AgentStreamEvent::FinalAnswer` when the report is ready.
    pub async fn stream_research(
        &self,
        topic: &str,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures_util::Stream<Item = crate::streaming::AgentStreamEvent> + Send>,
        >,
        ResearchError,
    > {
        use crate::streaming::AgentStreamEvent;

        if self.searchers.is_empty() {
            return Err(ResearchError::Search(
                "no search tools configured; add at least one with with_searcher()".to_string(),
            ));
        }

        let mut events: Vec<AgentStreamEvent> = Vec::new();

        // Step 1: Plan
        events.push(AgentStreamEvent::PipelineStep {
            step: "planning".to_string(),
            detail: Some("Decomposing topic into subtopics...".to_string()),
        });

        let current_plan = self.plan(topic).await?;

        events.push(AgentStreamEvent::PipelineStep {
            step: "planned".to_string(),
            detail: Some(format!(
                "Subtopics: {}",
                current_plan
                    .subtopics
                    .iter()
                    .map(|s| s.name.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        });

        // Step 2: Multi-round search
        let mut all_results: Vec<SearchResult> = Vec::new();
        let mut rounds_completed: usize = 0;
        let mut follow_up_queries: Vec<String> = Vec::new();
        let mut final_markdown = String::new();
        let mut final_citations: Vec<Citation> = Vec::new();

        for round in 0..self.max_rounds {
            let queries = if round == 0 {
                current_plan.all_queries()
            } else {
                follow_up_queries.clone()
            };

            if queries.is_empty() {
                break;
            }

            events.push(AgentStreamEvent::PipelineStep {
                step: "searching".to_string(),
                detail: Some(format!(
                    "Round {}: searching {} queries",
                    round + 1,
                    queries.len()
                )),
            });

            let round_results = self.search(&queries).await?;
            all_results.extend(round_results);
            all_results = SearchCollector::dedup(all_results);
            rounds_completed = round + 1;

            // Synthesize after each round
            events.push(AgentStreamEvent::PipelineStep {
                step: "synthesizing".to_string(),
                detail: Some(format!("Round {} synthesis", round + 1)),
            });

            let (markdown, gaps) = self
                .synthesize(topic, &current_plan, &all_results, self.max_source_tokens)
                .await?;

            if gaps.is_empty() || round + 1 >= self.max_rounds {
                final_markdown = markdown;
                final_citations = self.build_citations(&all_results);
                break;
            }

            events.push(AgentStreamEvent::PipelineStep {
                step: "gaps_found".to_string(),
                detail: Some(format!("{} information gaps identified", gaps.len())),
            });

            // Generate follow-up queries for the next round
            follow_up_queries = self.generate_follow_ups(topic, &gaps).await?;
        }

        // Final if we exhausted rounds without a final synthesis
        if final_markdown.is_empty() {
            let (markdown, _) = self
                .synthesize(topic, &current_plan, &all_results, self.max_source_tokens)
                .await?;
            final_markdown = markdown;
            final_citations = self.build_citations(&all_results);
        }

        events.push(AgentStreamEvent::PipelineStep {
            step: "completed".to_string(),
            detail: Some(format!(
                "Citations: {}, Rounds: {}",
                final_citations.len(),
                rounds_completed
            )),
        });

        // Final answer
        events.push(AgentStreamEvent::FinalAnswer {
            content: final_markdown,
        });

        Ok(Box::pin(futures_util::stream::iter(events)))
    }

    // -- Private helpers -------------------------------------------------------

    async fn plan(&self, topic: &str) -> Result<ResearchPlan, ResearchError> {
        planner::plan(&self.llm, topic, self.max_subtopics).await
    }

    async fn search(&self, queries: &[String]) -> Result<Vec<SearchResult>, ResearchError> {
        searcher::search(&self.searchers, queries).await
    }

    async fn synthesize(
        &self,
        topic: &str,
        plan: &ResearchPlan,
        results: &[SearchResult],
        max_source_tokens: Option<usize>,
    ) -> Result<(String, Vec<String>), ResearchError> {
        synthesizer::synthesize(&self.llm, topic, plan, results, max_source_tokens).await
    }

    async fn generate_follow_ups(
        &self,
        topic: &str,
        gaps: &[String],
    ) -> Result<Vec<String>, ResearchError> {
        let prompt = format!(
            "Research topic: {}\n\n\
             The following information gaps remain after initial research:\n\
             {}\n\n\
             For each gap, generate 1-3 specific search queries to fill it. \
             Output a JSON array of objects, each with \"gap\" and \"queries\" fields. \
             Every gap listed above must appear as a \"gap\" key in the output.\n\
             Example: [{{\"gap\": \"gap description\", \"queries\": [\"query1\", \"query2\"]}}]\n\
             Only output the JSON array, nothing else.",
            topic,
            gaps.iter()
                .enumerate()
                .map(|(i, g)| format!("{}. {}", i + 1, g))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let messages = vec![
            Message::system("You are a research assistant. Output only valid JSON."),
            Message::human(prompt),
        ];
        let response = crate::retry::retry_chat(
            &self.llm,
            messages,
            None,
            &crate::retry::RetryConfig::default(),
        )
        .await
        .map_err(|e| ResearchError::Llm(format!("{:?}", e)))?;

        parse_gap_queries(&response.content, gaps)
    }

    pub(crate) fn build_citations(&self, results: &[SearchResult]) -> Vec<Citation> {
        results
            .iter()
            .enumerate()
            .map(|(i, r)| Citation {
                index: i + 1,
                source: r.title.clone(),
                url: if r.url.is_empty() {
                    None
                } else {
                    Some(r.url.clone())
                },
                snippet: r.snippet.clone(),
            })
            .collect()
    }
}

impl<M: BaseChatModel> std::fmt::Debug for DeepResearchAgent<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepResearchAgent")
            .field("max_rounds", &self.max_rounds)
            .field("max_subtopics", &self.max_subtopics)
            .field("searchers_count", &self.searchers.len())
            .finish_non_exhaustive()
    }
}
