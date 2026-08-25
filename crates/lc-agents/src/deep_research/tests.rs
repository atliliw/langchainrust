// lc-agents/src/deep_research/tests.rs
//! Unit tests for `DeepResearchAgent`.

use super::*;
use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::BaseChatModel;
use lc_core::language_models::{BaseLanguageModel, LLMResult};
use lc_core::runnables::Runnable;
use lc_core::runnables::RunnableConfig;
use lc_core::tools::{BaseTool, ToolError};
use lc_schema::Message;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use super::types::parse_gap_queries;

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
        let mut guard = self.responses.lock().unwrap_or_else(|e| e.into_inner());
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
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
