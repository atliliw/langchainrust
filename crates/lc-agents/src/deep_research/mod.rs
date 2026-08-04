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

pub use planner::{ResearchPlan, SubTopic};
pub use searcher::{SearchCollector, SearchResult};
pub use synthesizer::SynthesisOutput;

use lc_core::language_models::BaseChatModel;
use lc_core::tools::BaseTool;
use lc_schema::Message;

/// Error types for deep research operations.
#[derive(Debug, thiserror::Error)]
pub enum ResearchError {
    /// LLM invocation error.
    #[error("LLM error: {0}")]
    Llm(String),

    /// Search tool execution error.
    #[error("search error: {0}")]
    Search(String),

    /// No search results were found for any query.
    #[error("no results found")]
    NoResults,
}

/// A citation referencing a source used in the research report.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Citation {
    /// 1-based citation index used in the report body (e.g. \[1\]).
    pub index: usize,
    /// Human-readable source title or description.
    pub source: String,
    /// Optional URL for the source.
    pub url: Option<String>,
    /// Short snippet quoted or paraphrased from the source.
    pub snippet: String,
}

/// The final output of a deep research run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResearchReport {
    /// The full report in markdown format with inline citation markers.
    pub markdown: String,
    /// Ordered list of citations referenced in the report.
    pub citations: Vec<Citation>,
    /// Sub-topics that were investigated.
    pub subtopics: Vec<String>,
    /// Number of research rounds completed.
    pub rounds_completed: usize,
}

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
        let response = self
            .llm
            .chat(messages, None)
            .await
            .map_err(|e| ResearchError::Llm(format!("{:?}", e)))?;

        parse_gap_queries(&response.content, gaps)
    }

    fn build_citations(&self, results: &[SearchResult]) -> Vec<Citation> {
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

/// Parses the LLM output for gap→query mapping into a flat list of queries.
///
/// Expected format: `[{"gap": "...", "queries": ["q1", "q2"]}, ...]`
///
/// Validates that every gap in the input list has at least one corresponding query.
/// If a gap has no queries in the parsed output, a warning is logged and a
/// fallback query is generated from the gap text itself.
fn parse_gap_queries(
    content: &str,
    original_gaps: &[String],
) -> Result<Vec<String>, ResearchError> {
    #[derive(serde::Deserialize)]
    struct GapMapping {
        #[allow(dead_code)]
        gap: String,
        queries: Vec<String>,
    }

    let json_str = planner::extract_json(content);
    let mappings: Vec<GapMapping> = serde_json::from_str(&json_str).map_err(|e| {
        let preview: String = content.chars().take(200).collect();
        ResearchError::Llm(format!(
            "failed to parse gap→query mapping: {} | raw: {}",
            e, preview
        ))
    })?;

    // Collect all queries, ensuring each original gap is covered.
    let mut all_queries: Vec<String> = Vec::new();
    let mut covered_gaps: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for mapping in &mappings {
        // Check if this mapping corresponds to any original gap (fuzzy match by substring)
        for (i, gap) in original_gaps.iter().enumerate() {
            if !covered_gaps.contains(&i)
                && (mapping.gap.contains(gap.as_str()) || gap.contains(mapping.gap.as_str()))
            {
                covered_gaps.insert(i);
            }
        }
        all_queries.extend(mapping.queries.iter().filter(|q| !q.is_empty()).cloned());
    }

    // For any uncovered gap, generate a fallback query from the gap text itself.
    for (i, gap) in original_gaps.iter().enumerate() {
        if !covered_gaps.contains(&i) {
            log::warn!(
                "Deep Research: gap '{}' has no follow-up queries, using gap as query",
                gap
            );
            all_queries.push(gap.clone());
        }
    }

    Ok(all_queries)
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::language_models::{BaseLanguageModel, LLMResult};
    use lc_core::runnables::Runnable;
    use lc_core::runnables::RunnableConfig;
    use lc_core::tools::ToolError;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    // -- Mock LLM that returns a sequence of responses -------------------------

    #[derive(Debug)]
    struct MockError(String);

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockError: {}", self.0)
        }
    }

    impl std::error::Error for MockError {}

    /// Mock LLM that returns responses from a preset list in order.
    /// Each call to `chat` advances to the next response.
    struct SequentialMockLLM {
        responses: Arc<Mutex<Vec<String>>>,
    }

    impl SequentialMockLLM {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses)),
            }
        }

        fn next_response(&self) -> String {
            let mut guard = self.responses.lock().unwrap();
            if guard.is_empty() {
                return r#"["follow-up query"]"#.to_string();
            }
            guard.remove(0)
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for SequentialMockLLM {
        type Error = MockError;

        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: self.next_response(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for SequentialMockLLM {
        fn model_name(&self) -> &str {
            "mock"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len() / 4
        }

        fn temperature(&self) -> Option<f32> {
            None
        }

        fn max_tokens(&self) -> Option<usize> {
            None
        }

        fn with_temperature(self, _temp: f32) -> Self
        where
            Self: Sized,
        {
            self
        }

        fn with_max_tokens(self, _max: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for SequentialMockLLM {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: self.next_response(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
        {
            let content = self.next_response();
            let stream = futures_util::stream::once(async move { Ok(content) });
            Ok(Box::pin(stream))
        }
    }

    // -- Mock search tool ------------------------------------------------------

    struct MockSearchTool {
        results: Vec<SearchResult>,
    }

    impl MockSearchTool {
        fn new(results: Vec<SearchResult>) -> Self {
            Self { results }
        }
    }

    #[async_trait]
    impl BaseTool for MockSearchTool {
        fn name(&self) -> &str {
            "mock_search"
        }

        fn description(&self) -> &str {
            "A mock search tool for testing"
        }

        async fn run(&self, _input: String) -> Result<String, ToolError> {
            let output = serde_json::json!({
                "results": self.results,
            });
            Ok(output.to_string())
        }
    }

    // -- Tests -----------------------------------------------------------------

    fn sample_search_results() -> Vec<SearchResult> {
        vec![
            SearchResult {
                query: "AI healthcare".to_string(),
                title: "AI in Medicine".to_string(),
                snippet: "AI is transforming healthcare diagnostics.".to_string(),
                url: "https://example.com/ai-medicine".to_string(),
            },
            SearchResult {
                query: "AI diagnostics".to_string(),
                title: "Diagnostic AI Tools".to_string(),
                snippet: "New AI tools improve diagnostic accuracy.".to_string(),
                url: "https://example.com/diagnostic-ai".to_string(),
            },
        ]
    }

    #[tokio::test]
    async fn test_research_single_round_no_gaps() {
        // Plan response: sub-topics with queries
        let plan_json = r#"[
            {"name": "AI in Diagnostics", "queries": ["AI diagnostics healthcare"]},
            {"name": "AI in Treatment", "queries": ["AI treatment planning"]}
        ]"#
        .to_string();

        // Synthesis response: report with no gaps
        let synthesis_json = "{\"report\": \"# AI in Healthcare\\n\\nAI is transforming healthcare [1]. New tools improve diagnostics [2].\", \"gaps\": []}".to_string();

        let llm = SequentialMockLLM::new(vec![plan_json, synthesis_json]);
        let search_results = sample_search_results();
        let mock_search = MockSearchTool::new(search_results);

        let agent = DeepResearchAgent::new(llm)
            .with_searcher(Box::new(mock_search))
            .with_max_rounds(1)
            .with_max_subtopics(3);

        let report = agent.research("AI in healthcare").await.unwrap();
        assert!(!report.markdown.is_empty());
        assert_eq!(report.rounds_completed, 1);
        assert_eq!(report.subtopics.len(), 2);
        assert_eq!(report.citations.len(), 2);
        assert_eq!(report.citations[0].index, 1);
        assert_eq!(report.citations[0].source, "AI in Medicine");
    }

    #[tokio::test]
    async fn test_research_multi_round_with_gaps() {
        // Round 1: plan
        let plan_json = r#"[
            {"name": "AI Ethics", "queries": ["AI ethics healthcare"]}
        ]"#
        .to_string();

        // Round 1: synthesis with gaps
        let synthesis1_json = "{\"report\": \"# AI Ethics in Healthcare\\n\\nSome info [1].\", \"gaps\": [\"Regulatory frameworks for AI in healthcare\"]}".to_string();

        // Follow-up queries (gap→query mapping format)
        let follow_up_json = r#"[{"gap": "Regulatory frameworks for AI in healthcare", "queries": ["AI healthcare regulation 2024"]}]"#.to_string();

        // Round 2: synthesis with no gaps
        let synthesis2_json = "{\"report\": \"# AI Ethics in Healthcare\\n\\nSome info [1]. Regulatory frameworks are evolving [2].\", \"gaps\": []}".to_string();

        let llm = SequentialMockLLM::new(vec![
            plan_json,
            synthesis1_json,
            follow_up_json,
            synthesis2_json,
        ]);

        let search_results = sample_search_results();
        let mock_search = MockSearchTool::new(search_results);

        let agent = DeepResearchAgent::new(llm)
            .with_searcher(Box::new(mock_search))
            .with_max_rounds(2)
            .with_max_subtopics(3);

        let report = agent.research("AI ethics in healthcare").await.unwrap();
        assert!(!report.markdown.is_empty());
        assert_eq!(report.rounds_completed, 2);
    }

    #[tokio::test]
    async fn test_research_no_search_tools() {
        let llm = SequentialMockLLM::new(vec![]);
        let agent: DeepResearchAgent<SequentialMockLLM> = DeepResearchAgent::new(llm);

        let result = agent.research("test topic").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("no search tools"));
    }

    #[tokio::test]
    async fn test_citation_building() {
        let results = sample_search_results();
        let llm = SequentialMockLLM::new(vec![]);
        let agent = DeepResearchAgent::new(llm);

        let citations = agent.build_citations(&results);
        assert_eq!(citations.len(), 2);
        assert_eq!(citations[0].index, 1);
        assert_eq!(citations[0].source, "AI in Medicine");
        assert_eq!(
            citations[0].url,
            Some("https://example.com/ai-medicine".to_string())
        );
        assert_eq!(citations[1].index, 2);
        assert!(citations[1].snippet.contains("diagnostic accuracy"));
    }

    #[test]
    fn test_research_error_display() {
        let err = ResearchError::Llm("timeout".to_string());
        assert_eq!(format!("{}", err), "LLM error: timeout");

        let err = ResearchError::Search("no tools".to_string());
        assert_eq!(format!("{}", err), "search error: no tools");

        let err = ResearchError::NoResults;
        assert_eq!(format!("{}", err), "no results found");
    }

    #[test]
    fn test_citation_serialization() {
        let citation = Citation {
            index: 1,
            source: "Test Source".to_string(),
            url: Some("https://example.com".to_string()),
            snippet: "A test snippet".to_string(),
        };
        let json = serde_json::to_string(&citation).unwrap();
        let deserialized: Citation = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.index, 1);
        assert_eq!(deserialized.source, "Test Source");
    }

    #[test]
    fn test_research_report_serialization() {
        let report = ResearchReport {
            markdown: "# Test Report\n\nContent [1].".to_string(),
            citations: vec![Citation {
                index: 1,
                source: "Source".to_string(),
                url: None,
                snippet: "snippet".to_string(),
            }],
            subtopics: vec!["Topic A".to_string()],
            rounds_completed: 1,
        };
        let json = serde_json::to_string(&report).unwrap();
        let deserialized: ResearchReport = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.rounds_completed, 1);
        assert_eq!(deserialized.subtopics.len(), 1);
    }

    #[test]
    fn test_parse_gap_queries_valid() {
        let content = r#"[{"gap": "Regulatory frameworks", "queries": ["AI regulation 2024", "FDA AI policy"]}, {"gap": "Cost analysis", "queries": ["AI healthcare cost savings"]}]"#;
        let gaps = vec![
            "Regulatory frameworks".to_string(),
            "Cost analysis".to_string(),
        ];
        let queries = parse_gap_queries(content, &gaps).unwrap();
        assert_eq!(queries.len(), 3);
        assert!(queries.contains(&"AI regulation 2024".to_string()));
        assert!(queries.contains(&"AI healthcare cost savings".to_string()));
    }

    #[test]
    fn test_parse_gap_queries_uncovered_gap_fallback() {
        let content = r#"[{"gap": "Regulatory frameworks", "queries": ["AI regulation"]}]"#;
        let gaps = vec![
            "Regulatory frameworks".to_string(),
            "Cost analysis".to_string(),
        ];
        let queries = parse_gap_queries(content, &gaps).unwrap();
        // "Cost analysis" has no mapping, so the gap text itself is used as a query
        assert!(queries.contains(&"Cost analysis".to_string()));
        assert!(queries.contains(&"AI regulation".to_string()));
    }

    #[test]
    fn test_parse_gap_queries_invalid_json() {
        let content = "not json";
        let gaps = vec!["Some gap".to_string()];
        let result = parse_gap_queries(content, &gaps);
        assert!(result.is_err());
    }

    /// Verify that cross-round citation numbering is consistent:
    /// citation[1] from round 1 should still map to the same source after round 2.
    #[tokio::test]
    async fn test_cross_round_citation_numbering() {
        // Plan response
        let plan_json = r#"[
            {"name": "AI Ethics", "queries": ["AI ethics healthcare"]}
        ]"#
        .to_string();

        // Round 1 synthesis with 2 sources
        let synthesis1_json = "{\"report\": \"# AI Ethics\\n\\nEthics matter [1]. Privacy concerns [2].\", \"gaps\": [\"Regulatory frameworks\"]}".to_string();

        // Follow-up queries
        let follow_up_json =
            r#"[{"gap": "Regulatory frameworks", "queries": ["AI regulation 2024"]}]"#.to_string();

        // Round 2 synthesis — should still reference [1] and [2] from round 1
        let synthesis2_json = "{\"report\": \"# AI Ethics\\n\\nEthics matter [1]. Privacy concerns [2]. Regulatory frameworks are evolving [3].\", \"gaps\": []}".to_string();

        let llm = SequentialMockLLM::new(vec![
            plan_json,
            synthesis1_json,
            follow_up_json,
            synthesis2_json,
        ]);

        let search_results = sample_search_results();
        let mock_search = MockSearchTool::new(search_results);

        let agent = DeepResearchAgent::new(llm)
            .with_searcher(Box::new(mock_search))
            .with_max_rounds(2)
            .with_max_subtopics(3);

        let report = agent.research("AI ethics in healthcare").await.unwrap();

        // Verify citations are built from accumulated results across rounds
        assert!(
            !report.citations.is_empty(),
            "should have citations from accumulated results"
        );

        // Verify citation numbering is 1-based and sequential
        for (i, citation) in report.citations.iter().enumerate() {
            assert_eq!(
                citation.index,
                i + 1,
                "citation index should be sequential starting at 1"
            );
        }

        // Verify that the first citation still maps to the original source
        assert_eq!(
            report.citations[0].source, "AI in Medicine",
            "citation [1] should still map to the first source from round 1"
        );
    }
}
