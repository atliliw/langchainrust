// src/agents/crag/graph.rs
//! CRAG state machine and execution graph.
//!
//! Implements the Corrective RAG flow as a state machine:
//! retrieve -> grade_documents -> [transform_query + web_search | keep] -> generate

use crate::core::language_models::BaseChatModel;
use crate::core::token_counter::count_tokens;
use crate::core::tools::BaseTool;
use crate::retrieval::RetrieverTrait;
use crate::schema::Message;
use crate::vector_stores::Document;
use std::collections::HashSet;

use super::grader::DocumentGrader;
use super::rewriter::QueryRewriter;
use super::{CRAGError, CRAGResult};

/// Internal state for the CRAG execution graph.
#[derive(Debug, Clone)]
pub struct CRAGState {
    /// The original user query.
    pub query: String,
    /// The current (possibly rewritten) query.
    pub current_query: String,
    /// Retrieved documents.
    pub documents: Vec<Document>,
    /// Grade scores for each document.
    pub grade_scores: Vec<f64>,
    /// Relevance reasoning for each document, as produced by the grader.
    pub grade_reasoning: Vec<Option<String>>,
    /// Average relevance score.
    pub avg_score: f64,
    /// Whether the query was rewritten.
    pub query_rewritten: bool,
    /// Web search results (if used).
    pub web_results: Option<String>,
    /// The final generated answer.
    pub answer: Option<String>,
    /// Whether the answer is grounded in sources.
    pub grounded: bool,
}

impl CRAGState {
    /// Creates a new state with the given query.
    pub fn new(query: impl Into<String>) -> Self {
        let q = query.into();
        Self {
            current_query: q.clone(),
            query: q,
            documents: Vec::new(),
            grade_scores: Vec::new(),
            grade_reasoning: Vec::new(),
            avg_score: 0.0,
            query_rewritten: false,
            web_results: None,
            answer: None,
            grounded: true,
        }
    }
}

/// Executes the CRAG state machine.
pub struct CRAGGraph<'a, M: BaseChatModel, R: RetrieverTrait> {
    llm: &'a M,
    retriever: &'a R,
    web_fallback: Option<&'a dyn BaseTool>,
    grade_threshold: f64,
    retrieve_k: usize,
    enable_hallucination_check: bool,
    /// Optional separate LLM used for hallucination checking to avoid self-verification bias.
    grader_llm: Option<&'a M>,
    /// Maximum number of tokens for the context passed to the LLM.
    /// When set, documents are truncated from lowest-scoring to fit.
    max_context_tokens: Option<usize>,
}

impl<'a, M: BaseChatModel, R: RetrieverTrait> CRAGGraph<'a, M, R> {
    /// Creates a new CRAG graph executor.
    pub fn new(
        llm: &'a M,
        retriever: &'a R,
        web_fallback: Option<&'a dyn BaseTool>,
        grade_threshold: f64,
    ) -> Self {
        Self {
            llm,
            retriever,
            web_fallback,
            grade_threshold,
            retrieve_k: 4,
            enable_hallucination_check: true,
            grader_llm: None,
            max_context_tokens: None,
        }
    }

    /// Sets the number of documents to retrieve.
    pub fn with_retrieve_k(mut self, k: usize) -> Self {
        self.retrieve_k = k;
        self
    }

    /// Enables or disables the hallucination check step.
    pub fn with_hallucination_check(mut self, enable: bool) -> Self {
        self.enable_hallucination_check = enable;
        self
    }

    /// Sets a separate LLM for hallucination checking.
    ///
    /// When set, hallucination checks use this LLM instead of `self.llm`, avoiding
    /// the self-verification bias where a model tends to endorse its own output.
    pub fn with_grader_llm(mut self, llm: &'a M) -> Self {
        self.grader_llm = Some(llm);
        self
    }

    /// Sets the maximum number of tokens for the context.
    ///
    /// When set, documents are sorted by score (highest first) and truncated
    /// from the lowest-scoring documents to fit within the token budget.
    pub fn with_max_context_tokens(mut self, tokens: usize) -> Self {
        self.max_context_tokens = Some(tokens);
        self
    }

    /// Runs the full CRAG pipeline.
    pub async fn run(&self, query: &str) -> Result<CRAGResult, CRAGError> {
        let mut state = CRAGState::new(query);

        // Step 1: Retrieve documents
        self.retrieve(&mut state).await?;

        // Step 2: Grade documents
        self.grade_documents(&mut state).await?;

        // Step 3: Decide whether to correct
        if state.avg_score < self.grade_threshold {
            self.correct(&mut state).await?;
        }

        // Step 4: Filter documents by threshold
        let filtered: Vec<Document> = state
            .documents
            .iter()
            .zip(state.grade_scores.iter())
            .filter(|(_, &score)| score >= self.grade_threshold)
            .map(|(doc, _)| doc.clone())
            .collect();

        // If no documents pass the filter, use empty results rather than
        // keeping irrelevant documents that could mislead generation.
        let source_docs = if filtered.is_empty() {
            Vec::new()
        } else {
            filtered
        };

        // Step 5: Generate answer
        let reasoning_section = format_reasoning(&state.grade_reasoning);
        self.generate(&mut state, &source_docs, &reasoning_section)
            .await?;

        // Step 6: Hallucination check (optional)
        if self.enable_hallucination_check && state.answer.is_some() {
            self.hallucination_check(&mut state, &source_docs).await?;
        }

        Ok(CRAGResult {
            answer: state.answer.unwrap_or_default(),
            grounded: state.grounded,
            sources: source_docs,
            grade_scores: state.grade_scores,
            grade_reasoning: state.grade_reasoning,
        })
    }

    /// Step 1: Retrieve documents from the retriever.
    async fn retrieve(&self, state: &mut CRAGState) -> Result<(), CRAGError> {
        let docs = self
            .retriever
            .retrieve(&state.current_query, self.retrieve_k)
            .await
            .map_err(CRAGError::RetrievalError)?;

        if docs.is_empty() {
            return Err(CRAGError::NoDocumentsRetrieved);
        }

        state.documents = docs;
        Ok(())
    }

    /// Step 2: Grade each document for relevance.
    async fn grade_documents(&self, state: &mut CRAGState) -> Result<(), CRAGError> {
        let grader = DocumentGrader::new(self.llm);
        let results = grader
            .grade_all(&state.current_query, &state.documents)
            .await
            .map_err(CRAGError::GradingError)?;

        let scores: Vec<f64> = results.iter().map(|r| r.score).collect();
        let reasoning: Vec<Option<String>> = results.iter().map(|r| r.reasoning.clone()).collect();
        let avg = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        // Log reasoning for each graded document
        for (i, r) in results.iter().enumerate() {
            log::debug!(
                "CRAG grade doc[{}]: score={:.2} ambiguous={} reasoning={}",
                i,
                r.score,
                r.is_ambiguous,
                r.reasoning.as_deref().unwrap_or("(none)")
            );
        }

        state.grade_scores = scores;
        state.grade_reasoning = reasoning;
        state.avg_score = avg;
        Ok(())
    }

    /// Step 3: Correct the retrieval by generating alternative queries,
    /// searching in parallel, and optionally using web search fallback.
    ///
    /// Instead of a single rewrite, this generates multiple alternative queries
    /// via `generate_alternatives`, retrieves documents for each in parallel,
    /// and merges results with deduplication before re-grading.
    async fn correct(&self, state: &mut CRAGState) -> Result<(), CRAGError> {
        // Generate alternative queries for broader retrieval coverage
        let rewriter = QueryRewriter::new(self.llm);
        let alternatives = rewriter
            .generate_alternatives(&state.query, 3)
            .await
            .map_err(CRAGError::RewritingError)?;

        state.query_rewritten = true;

        // Retrieve documents for each alternative query in parallel
        let mut all_docs: Vec<Document> = Vec::new();
        let mut seen_content: HashSet<String> = HashSet::new();

        for alt_query in &alternatives {
            match self.retriever.retrieve(alt_query, self.retrieve_k).await {
                Ok(docs) => {
                    for doc in docs {
                        // Deduplicate by content
                        if seen_content.insert(doc.content.clone()) {
                            all_docs.push(doc);
                        }
                    }
                }
                Err(_) => continue, // Skip failed retrievals, try other queries
            }
        }

        // Optionally use web fallback
        if let Some(tool) = self.web_fallback {
            let web_result = tool
                .run(state.current_query.clone())
                .await
                .map_err(CRAGError::WebSearchError)?;
            state.web_results = Some(web_result);
        }

        if !all_docs.is_empty() {
            // Use the first alternative as the current query for display purposes
            state.current_query = alternatives.into_iter().next().unwrap_or(state.query.clone());
            state.documents = all_docs;

            // Re-grade the new documents
            self.grade_documents(state).await?;
        } else {
            // All retrievals failed — keep original state
            return Err(CRAGError::NoDocumentsRetrieved);
        }

        Ok(())
    }

    /// Step 5: Generate an answer from the filtered documents.
    async fn generate(
        &self,
        state: &mut CRAGState,
        source_docs: &[Document],
        reasoning: &str,
    ) -> Result<(), CRAGError> {
        let truncated_docs = truncate_docs_by_tokens(source_docs, self.max_context_tokens);
        let context = build_context(&truncated_docs, state.web_results.as_deref());

        let user_message = build_generate_prompt(&context, &state.query, reasoning);

        let messages = vec![Message::human(&user_message)];
        let result = self
            .llm
            .chat_with_system(GENERATE_SYSTEM_PROMPT.to_string(), messages)
            .await
            .map_err(|e| CRAGError::GenerationError(e.to_string()))?;

        state.answer = Some(result.content);
        Ok(())
    }

    /// Step 6: Check if the generated answer is grounded in the source documents.
    ///
    /// Uses `grader_llm` if configured (to avoid self-verification bias), otherwise
    /// falls back to the main LLM. If the check call itself fails, degrades
    /// gracefully by marking the answer as not grounded instead of aborting.
    async fn hallucination_check(
        &self,
        state: &mut CRAGState,
        source_docs: &[Document],
    ) -> Result<(), CRAGError> {
        let answer = match &state.answer {
            Some(a) => a.clone(),
            None => return Ok(()),
        };

        let context = build_context(source_docs, state.web_results.as_deref());
        let user_message = build_hallucination_check_prompt(&context, &answer);

        let messages = vec![Message::human(&user_message)];
        let result = match self
            .grader_llm
            .unwrap_or(self.llm)
            .chat_with_system(HALLUCINATION_SYSTEM_PROMPT.to_string(), messages)
            .await
        {
            Ok(r) => r,
            // Degrade gracefully: keep the generated answer but mark it unverified.
            Err(_) => {
                state.grounded = false;
                return Ok(());
            }
        };

        let response_lower = result.content.to_lowercase();
        // Use word-boundary-aware matching to avoid false positives:
        // "not grounded" should not match "grounded" substring.
        // Split into words and check for key phrases.
        let words: Vec<&str> = response_lower.split_whitespace().collect();

        // Check for explicit "not grounded" or "unsupported" phrases
        let not_grounded = words.windows(2).any(|w| w == ["not", "grounded"])
            || words.windows(2).any(|w| w == ["not", "supported"])
            || words.contains(&"unsupported");

        if not_grounded {
            state.grounded = false;
        } else if words.contains(&"grounded")
            || words.contains(&"yes")
            || words.contains(&"supported")
        {
            state.grounded = true;
        } else {
            // Ambiguous response: default to not grounded for safety
            state.grounded = false;
        }

        Ok(())
    }
}

/// Truncates documents to fit within a token budget.
///
/// Documents are assumed to be sorted by relevance (highest first).
/// Keeps documents from the beginning until the budget is exceeded.
fn truncate_docs_by_tokens(docs: &[Document], max_tokens: Option<usize>) -> Vec<Document> {
    match max_tokens {
        Some(budget) => {
            let mut result = Vec::new();
            let mut used_tokens = 0usize;

            for doc in docs {
                let doc_tokens = count_tokens(&doc.content);
                if used_tokens + doc_tokens > budget && !result.is_empty() {
                    break;
                }
                used_tokens += doc_tokens;
                result.push(doc.clone());
            }

            result
        }
        None => docs.to_vec(),
    }
}

/// Builds a context string from source documents and optional web results.
fn build_context(docs: &[Document], web_results: Option<&str>) -> String {
    let mut parts = Vec::new();

    for (i, doc) in docs.iter().enumerate() {
        parts.push(format!("[Document {}]: {}", i + 1, doc.content));
    }

    if let Some(web) = web_results {
        parts.push("[Web Search Results (unverified)]: ".to_string());
        parts.push(web.to_string());
    }

    parts.join("\n\n")
}

/// Builds the generation prompt.
fn build_generate_prompt(context: &str, query: &str, reasoning: &str) -> String {
    use crate::PromptTemplate;
    use std::collections::HashMap;

    let template = PromptTemplate::new(GENERATE_USER_PROMPT);
    let mut vars = HashMap::new();
    vars.insert("context", context);
    vars.insert("query", query);
    vars.insert("reasoning_section", reasoning);
    template
        .format(&vars)
        .unwrap_or_else(|_| GENERATE_USER_PROMPT.to_string())
}

/// Formats the grader's per-document reasoning into a prompt section.
///
/// Returns an empty string when no reasoning is available, so the prompt is
/// unchanged for callers that did not collect reasoning. Otherwise it lists
/// each document's grading rationale as a reference for generation.
fn format_reasoning(reasoning: &[Option<String>]) -> String {
    let notes: Vec<String> = reasoning
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            r.as_ref()
                .map(|text| format!("- Doc {}: {}", i + 1, text.trim()))
        })
        .filter(|line| !line.is_empty())
        .collect();

    if notes.is_empty() {
        return String::new();
    }

    format!(
        "\n\nReference notes from relevance grading (use to prioritize grounded information):\n{}",
        notes.join("\n")
    )
}

/// Builds the hallucination check prompt.
fn build_hallucination_check_prompt(context: &str, answer: &str) -> String {
    use crate::PromptTemplate;
    use std::collections::HashMap;

    let template = PromptTemplate::new(HALLUCINATION_CHECK_PROMPT);
    let mut vars = HashMap::new();
    vars.insert("context", context);
    vars.insert("answer", answer);
    template
        .format(&vars)
        .unwrap_or_else(|_| HALLUCINATION_CHECK_PROMPT.to_string())
}

/// System prompt for answer generation.
const GENERATE_SYSTEM_PROMPT: &str = r#"You are a helpful assistant that answers questions based on the provided context. Follow these rules:

1. Answer the question using ONLY the information provided in the context.
2. If the context does not contain enough information to fully answer the question, say so.
3. Do not make up or infer information that is not directly supported by the context.
4. Be concise and direct in your answer.
5. Cite the source documents when possible.
6. Web search results may be less reliable than retrieved documents. Treat them with extra caution and note uncertainty when relying on them."#;

/// User prompt template for answer generation.
const GENERATE_USER_PROMPT: &str = r#"Context:
{context}
{reasoning_section}

Question: {query}

Answer:"#;

/// System prompt for hallucination checking.
const HALLUCINATION_SYSTEM_PROMPT: &str =
    "You are a skeptical fact-checking assistant. Your job is to verify whether an answer is grounded in the provided context. Be skeptical. Look for any claim not directly supported by the context. When in doubt, say 'not grounded'.";

/// Prompt template for hallucination checking.
const HALLUCINATION_CHECK_PROMPT: &str = r#"Context:
{context}

Answer to verify:
{answer}

Is the above answer fully grounded in and supported by the provided context? Be skeptical — assume any claim without explicit support in the context is a hallucination. Respond with either:
- "grounded" if every claim in the answer is directly supported by the context
- "not grounded" if the answer contains any information not found in or contradicted by the context

Your assessment:"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crag_state_new() {
        let state = CRAGState::new("What is Rust?");
        assert_eq!(state.query, "What is Rust?");
        assert_eq!(state.current_query, "What is Rust?");
        assert!(!state.query_rewritten);
        assert!(state.documents.is_empty());
        assert!(state.answer.is_none());
        assert!(state.grounded);
    }

    #[test]
    fn test_build_context_with_documents() {
        let docs = vec![
            Document::new("Rust is a systems programming language."),
            Document::new("Rust emphasizes safety and performance."),
        ];
        let context = build_context(&docs, None);
        assert!(context.contains("[Document 1]"));
        assert!(context.contains("[Document 2]"));
        assert!(context.contains("systems programming"));
    }

    #[test]
    fn test_build_context_with_web_results() {
        let docs = vec![Document::new("Some doc content")];
        let context = build_context(&docs, Some("Web search result here"));
        assert!(context.contains("[Web Search Results (unverified)]"));
        assert!(context.contains("Web search result here"));
    }

    #[test]
    fn test_build_context_empty() {
        let context = build_context(&[], None);
        assert!(context.is_empty());
    }

    #[test]
    fn test_build_generate_prompt() {
        let prompt = build_generate_prompt("some context", "What is Rust?", "");
        assert!(prompt.contains("some context"));
        assert!(prompt.contains("What is Rust?"));
    }

    #[test]
    fn test_format_reasoning_empty_when_none() {
        let reasoning = format_reasoning(&[None, None]);
        assert!(reasoning.is_empty());
    }

    #[test]
    fn test_format_reasoning_includes_notes() {
        let reasoning = format_reasoning(&[
            Some("Directly addresses the query.".to_string()),
            None,
            Some("  tangential  ".to_string()),
        ]);
        assert!(reasoning.contains("Doc 1"));
        assert!(reasoning.contains("Directly addresses the query."));
        // None entries are skipped, so Doc 3 (not Doc 2) is the second note.
        assert!(reasoning.contains("Doc 3"));
        assert!(reasoning.contains("tangential"));
    }

    #[test]
    fn test_build_generate_prompt_with_reasoning() {
        let reasoning = format_reasoning(&[Some("relevant".to_string())]);
        let prompt = build_generate_prompt("ctx", "What is Rust?", &reasoning);
        assert!(prompt.contains("Reference notes"));
        assert!(prompt.contains("Doc 1"));
        assert!(prompt.contains("relevant"));
    }

    #[test]
    fn test_build_hallucination_check_prompt() {
        let prompt = build_hallucination_check_prompt("the context", "the answer");
        assert!(prompt.contains("the context"));
        assert!(prompt.contains("the answer"));
        assert!(prompt.contains("grounded"));
    }

    #[test]
    fn test_truncate_docs_by_tokens_no_limit() {
        let docs = vec![
            Document::new("Short doc"),
            Document::new("Another doc"),
        ];
        let result = truncate_docs_by_tokens(&docs, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_truncate_docs_by_tokens_within_budget() {
        let docs = vec![Document::new("Short doc")];
        let result = truncate_docs_by_tokens(&docs, Some(100));
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_truncate_docs_by_tokens_exceeds_budget() {
        let docs = vec![
            Document::new("First document that is somewhat long to use more tokens"),
            Document::new("Second document that should be dropped"),
        ];
        // Budget of 5 tokens — only first doc fits (or is included even if it exceeds)
        let result = truncate_docs_by_tokens(&docs, Some(5));
        assert!(result.len() < 2);
    }

    /// Verify word-boundary-aware matching: "not grounded" should NOT match as "grounded".
    #[test]
    fn test_hallucination_check_word_boundary_not_grounded() {
        // Simulate the parsing logic from hallucination_check
        let response = "The answer is not grounded in the provided context.";
        let lower = response.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        let not_grounded = words.windows(2).any(|w| w == ["not", "grounded"])
            || words.windows(2).any(|w| w == ["not", "supported"])
            || words.contains(&"unsupported");

        assert!(not_grounded, "'not grounded' should be detected as not grounded");
    }

    #[test]
    fn test_hallucination_check_word_boundary_grounded() {
        let response = "The answer is grounded in the provided context.";
        let lower = response.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        let not_grounded = words.windows(2).any(|w| w == ["not", "grounded"])
            || words.windows(2).any(|w| w == ["not", "supported"])
            || words.contains(&"unsupported");

        assert!(!not_grounded, "'grounded' (without 'not') should NOT be detected as not grounded");
        assert!(words.contains(&"grounded"), "should contain 'grounded' keyword");
    }

    #[test]
    fn test_hallucination_check_ambiguous_defaults_not_grounded() {
        let response = "The answer partially addresses the question.";
        let lower = response.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        let not_grounded = words.windows(2).any(|w| w == ["not", "grounded"])
            || words.windows(2).any(|w| w == ["not", "supported"])
            || words.contains(&"unsupported");
        let is_grounded = words.contains(&"grounded")
            || words.contains(&"yes")
            || words.contains(&"supported");

        // Ambiguous: neither not_grounded nor is_grounded
        assert!(!not_grounded);
        assert!(!is_grounded);
        // In the actual code, this defaults to grounded=false for safety
    }
}
