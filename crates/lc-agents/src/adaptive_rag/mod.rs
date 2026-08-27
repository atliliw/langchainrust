// src/agents/adaptive_rag.rs → adaptive_rag/ (mod.rs + types.rs + prompts.rs + tests.rs)
//! Adaptive RAG implementation.
//!
//! Uses an LLM to decide whether retrieval is needed and what strategy to use.
//! Three decision branches:
//! - **NoRetrieval**: The query can be answered from general knowledge.
//! - **SingleSearch**: A single search is sufficient.
//! - **MultiQuery**: The query is complex and needs multiple search angles.

use lc_core::language_models::BaseChatModel;
use lc_core::tools::ToolDefinition;
use lc_rag::RetrieverTrait;
use lc_schema::Message;
use lc_vector_stores::Document;
use serde_json::json;

mod prompts;
#[cfg(test)]
mod tests;
mod types;

pub use types::{AdaptiveRAGError, AdaptiveRAGResult, RagDecision};

use prompts::{GENERATE_SYSTEM_PROMPT, MULTI_QUERY_PROMPT, ROUTING_PROMPT};

/// Adaptive RAG that routes queries to the most appropriate strategy.
///
/// # Overview
///
/// 1. The LLM classifies the query into one of three buckets:
///    `no_retrieval`, `single_search`, or `multi_query`.
/// 2. Based on the decision:
///    - **NoRetrieval**: call the LLM directly.
///    - **SingleSearch**: retrieve documents, then generate.
///    - **MultiQuery**: generate multiple query variants, retrieve for each,
///      merge results, then generate.
///
/// # Example
///
/// ```ignore
/// use langchainrust::agents::adaptive_rag::{AdaptiveRAG, RagDecision};
/// use langchainrust::OpenAIChat;
/// use langchainrust::retrieval::SimilarityRetriever;
///
/// let rag = AdaptiveRAG::new(llm, retriever);
/// let result = rag.invoke("What is the capital of France?").await?;
/// assert_eq!(result.decision, RagDecision::NoRetrieval);
/// ```
pub struct AdaptiveRAG<M: BaseChatModel, R: RetrieverTrait> {
    llm: M,
    retriever: R,
    /// Number of documents to retrieve per query.
    retrieve_k: usize,
    /// Number of alternative queries to generate for multi-query mode.
    multi_query_count: usize,
}

impl<M: BaseChatModel, R: RetrieverTrait> AdaptiveRAG<M, R> {
    /// Creates a new `AdaptiveRAG` with the given LLM and retriever.
    pub fn new(llm: M, retriever: R) -> Self {
        Self {
            llm,
            retriever,
            retrieve_k: 4,
            multi_query_count: 3,
        }
    }

    /// Sets the number of documents to retrieve per query.
    pub fn with_retrieve_k(mut self, k: usize) -> Self {
        self.retrieve_k = k;
        self
    }

    /// Sets the number of alternative queries for multi-query mode.
    pub fn with_multi_query_count(mut self, count: usize) -> Self {
        self.multi_query_count = count;
        self
    }

    /// Invokes the adaptive RAG pipeline for the given query.
    pub async fn invoke(&self, query: &str) -> Result<AdaptiveRAGResult, AdaptiveRAGError> {
        // Step 1: Route the query.
        let decision = self.route(query).await?;

        match decision {
            RagDecision::NoRetrieval => self.generate_no_retrieval(query).await,
            RagDecision::SingleSearch => self.generate_single_search(query).await,
            RagDecision::MultiQuery => self.generate_multi_query(query).await,
        }
    }

    /// Streams the AdaptiveRAG execution, emitting pipeline step events.
    ///
    /// Emits `AgentStreamEvent::PipelineStep` events for routing and generation,
    /// and `AgentStreamEvent::FinalAnswer` when the answer is ready.
    pub async fn stream(
        &self,
        query: &str,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures_util::Stream<Item = crate::streaming::AgentStreamEvent> + Send>,
        >,
        AdaptiveRAGError,
    > {
        use crate::streaming::AgentStreamEvent;

        let mut events: Vec<AgentStreamEvent> = Vec::new();

        // Step 1: Route
        events.push(AgentStreamEvent::PipelineStep {
            step: "routing".to_string(),
            detail: Some("Classifying query...".to_string()),
        });

        let decision = self.route(query).await?;

        events.push(AgentStreamEvent::PipelineStep {
            step: "routed".to_string(),
            detail: Some(format!("Decision: {}", decision)),
        });

        // Step 2: Execute based on decision
        let result = match decision {
            RagDecision::NoRetrieval => {
                events.push(AgentStreamEvent::PipelineStep {
                    step: "generating".to_string(),
                    detail: Some("No retrieval needed, generating directly...".to_string()),
                });
                self.generate_no_retrieval(query).await?
            }
            RagDecision::SingleSearch => {
                events.push(AgentStreamEvent::PipelineStep {
                    step: "retrieving".to_string(),
                    detail: Some("Single search retrieval...".to_string()),
                });
                let result = self.generate_single_search(query).await?;
                events.push(AgentStreamEvent::PipelineStep {
                    step: "generating".to_string(),
                    detail: Some(format!("Sources: {} documents", result.sources.len())),
                });
                result
            }
            RagDecision::MultiQuery => {
                events.push(AgentStreamEvent::PipelineStep {
                    step: "multi_query".to_string(),
                    detail: Some("Generating multiple queries...".to_string()),
                });
                let result = self.generate_multi_query(query).await?;
                events.push(AgentStreamEvent::PipelineStep {
                    step: "generating".to_string(),
                    detail: Some(format!("Sources: {} documents", result.sources.len())),
                });
                result
            }
        };

        // Final answer
        events.push(AgentStreamEvent::FinalAnswer {
            content: result.answer,
        });

        Ok(Box::pin(futures_util::stream::iter(events)))
    }

    // -- Routing -----------------------------------------------------------

    /// Asks the LLM to classify the query.
    async fn route(&self, query: &str) -> Result<RagDecision, AdaptiveRAGError> {
        let prompt = ROUTING_PROMPT.replace("{query}", query);
        let messages = vec![Message::human(&prompt)];

        // P1-3: prefer structured routing via tool_calls, falling back to text parsing.
        let structured = crate::structured::chat_structured(
            &self.llm,
            Some(route_tool()),
            messages,
            None,
            &crate::retry::RetryConfig::default(),
        )
        .await
        .map_err(|e| AdaptiveRAGError::Llm(e.to_string()))?;

        if let Some(args) = &structured.tool_args {
            if let Some(decision) = args.get("decision").and_then(|v| v.as_str()) {
                return parse_decision(decision);
            }
        }
        parse_decision(&structured.content)
    }

    // -- No retrieval ------------------------------------------------------

    /// Generates an answer without any retrieval.
    async fn generate_no_retrieval(
        &self,
        query: &str,
    ) -> Result<AdaptiveRAGResult, AdaptiveRAGError> {
        let messages = vec![Message::human(query)];
        let result = self
            .llm
            .chat_with_system(GENERATE_SYSTEM_PROMPT.to_string(), messages)
            .await
            .map_err(|e| AdaptiveRAGError::Llm(e.to_string()))?;

        Ok(AdaptiveRAGResult {
            answer: result.content,
            decision: RagDecision::NoRetrieval,
            sources: Vec::new(),
        })
    }

    // -- Single search -----------------------------------------------------

    /// Retrieves documents with a single query and generates an answer.
    async fn generate_single_search(
        &self,
        query: &str,
    ) -> Result<AdaptiveRAGResult, AdaptiveRAGError> {
        let docs = self.retriever.retrieve(query, self.retrieve_k).await?;

        let context = build_context(&docs);
        let user_msg = format!("Context:\n{}\n\nQuestion: {}\n\nAnswer:", context, query);
        let messages = vec![Message::human(&user_msg)];

        let result = self
            .llm
            .chat_with_system(GENERATE_SYSTEM_PROMPT.to_string(), messages)
            .await
            .map_err(|e| AdaptiveRAGError::Llm(e.to_string()))?;

        Ok(AdaptiveRAGResult {
            answer: result.content,
            decision: RagDecision::SingleSearch,
            sources: docs,
        })
    }

    // -- Multi query -------------------------------------------------------

    /// Generates multiple query variants, retrieves for each, merges, and
    /// generates an answer.
    async fn generate_multi_query(
        &self,
        query: &str,
    ) -> Result<AdaptiveRAGResult, AdaptiveRAGError> {
        let alternative_queries = self.generate_queries(query).await?;

        // Combine original + alternatives.
        let all_queries: Vec<String> = std::iter::once(query.to_string())
            .chain(alternative_queries)
            .collect();

        // Retrieve and merge.
        let docs = self.retrieve_and_merge(&all_queries).await?;

        let context = build_context(&docs);
        let user_msg = format!("Context:\n{}\n\nQuestion: {}\n\nAnswer:", context, query);
        let messages = vec![Message::human(&user_msg)];

        let result = self
            .llm
            .chat_with_system(GENERATE_SYSTEM_PROMPT.to_string(), messages)
            .await
            .map_err(|e| AdaptiveRAGError::Llm(e.to_string()))?;

        Ok(AdaptiveRAGResult {
            answer: result.content,
            decision: RagDecision::MultiQuery,
            sources: docs,
        })
    }

    /// Asks the LLM to generate alternative query variants.
    async fn generate_queries(&self, query: &str) -> Result<Vec<String>, AdaptiveRAGError> {
        let prompt = MULTI_QUERY_PROMPT
            .replace("{count}", &self.multi_query_count.to_string())
            .replace("{question}", query);

        let messages = vec![Message::human(&prompt)];
        let result = crate::retry::retry_chat(
            &self.llm,
            messages,
            None,
            &crate::retry::RetryConfig::default(),
        )
        .await
        .map_err(|e| AdaptiveRAGError::Llm(e.to_string()))?;

        let queries: Vec<String> = result
            .content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .collect();

        Ok(queries)
    }

    /// Retrieves documents for each query and merges/deduplicates by content.
    async fn retrieve_and_merge(
        &self,
        queries: &[String],
    ) -> Result<Vec<Document>, AdaptiveRAGError> {
        // M11: Parallel retrieval instead of sequential.
        let futures: Vec<_> = queries
            .iter()
            .map(|q| self.retriever.retrieve(q, self.retrieve_k))
            .collect();
        let all_results = futures_util::future::join_all(futures).await;

        let mut seen_content: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut merged: Vec<Document> = Vec::new();

        for result in all_results {
            let docs = result?;
            for doc in docs {
                // M6 fix: deduplicate by full content hash instead of first 80 chars
                // to avoid false collisions on documents with common prefixes.
                let key = {
                    use std::hash::Hasher;
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    hasher.write(doc.content.as_bytes());
                    format!("{:016x}", hasher.finish())
                };
                if seen_content.insert(key) {
                    merged.push(doc);
                }
            }
        }

        Ok(merged)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Routing tool definition: forces the LLM to emit a three-way decision (P1-3).
fn route_tool() -> ToolDefinition {
    ToolDefinition::new(
        "route_decision",
        "判断查询是否需要检索及检索策略:no_retrieval / single_search / multi_query",
    )
    .with_parameters(json!({
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["no_retrieval", "single_search", "multi_query"]
            }
        },
        "required": ["decision"]
    }))
}

/// Parses the routing decision from the LLM response.
fn parse_decision(response: &str) -> Result<RagDecision, AdaptiveRAGError> {
    let lower = response.to_lowercase();

    // Check for exact or contained keywords, with precedence.
    if lower.contains("no_retrieval") {
        return Ok(RagDecision::NoRetrieval);
    }
    if lower.contains("multi_query") {
        return Ok(RagDecision::MultiQuery);
    }
    if lower.contains("single_search") {
        return Ok(RagDecision::SingleSearch);
    }

    Err(AdaptiveRAGError::DecisionParse(response.to_string()))
}

/// Builds a context string from source documents.
fn build_context(docs: &[Document]) -> String {
    docs.iter()
        .enumerate()
        .map(|(i, doc)| format!("[Document {}]: {}", i + 1, doc.content))
        .collect::<Vec<_>>()
        .join("\n\n")
}
