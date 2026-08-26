// lc-chains/src/document_chains/map_rerank.rs
//! MapRerankDocumentsChain - processes documents in parallel then ranks by relevance.

use async_trait::async_trait;
use futures_util::future::try_join_all;
use futures_util::StreamExt;
use lc_core::BaseChatModel;
use lc_providers::{wrap_chat_model, ProviderError};
use lc_schema::Message;
use lc_shared::document::Document;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::LazyLock;

use crate::base::{BaseChain, ChainError, ChainResult, ChainStream, StreamToken};
use crate::BoxedChatModel;

/// Default Map + Rerank prompt template.
pub(crate) const DEFAULT_MAP_RERANK_PROMPT: &str = "Answer the question based on the following document, and provide a relevance score (0-100, higher is more relevant).

Document content:
{context}

Question: {input}

Please output in the following format:
Relevance score: <score>
Answer: <your answer>";

/// MapRerankDocumentsChain
///
/// First calls LLM independently for each document to generate an answer and score,
/// then ranks by relevance score and returns the highest-scoring answer.
pub struct MapRerankDocumentsChain {
    llm: BoxedChatModel,
    map_prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
    /// Return top k results (default 1, i.e. only the highest score).
    top_k: usize,
    /// Fallback score for LLM output without a parseable score (P1-3).
    /// `None` = skip the document; `Some(n)` = rank it with score n.
    default_score: Option<u32>,
}

// Pre-compiled regex patterns for score extraction.
static SCORE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:relevance\s*score|相关性评分)\s*[:：]\s*(\d+)").unwrap());
static SCORE_RE2: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)score\s*[:：]\s*(\d+)").unwrap());

/// Truncate a string to at most `max_len` characters, respecting char boundaries.
fn truncate_str(s: &str, max_len: usize) -> &str {
    if s.chars().count() <= max_len {
        s
    } else {
        let end = s
            .char_indices()
            .nth(max_len)
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        &s[..end]
    }
}

/// Extract score and answer from LLM output.
///
/// Returns `None` when no parseable score is present — the caller then decides
/// how to treat unscored output (skip the document or use a configured default)
/// instead of silently assigning a middle score that pollutes ranking (P1-3).
pub fn extract_score(text: &str) -> Option<(u32, String)> {
    for re in [&*SCORE_RE, &*SCORE_RE2] {
        if let Some(caps) = re.captures(text) {
            if let Ok(score) = caps[1].parse::<u32>() {
                let cleaned = re.replace(text, "").trim().to_string();
                let cleaned = cleaned
                    .trim_start_matches("Answer")
                    .trim_start_matches("答案")
                    .trim_start_matches(&[':', '：'][..])
                    .trim()
                    .to_string();
                return Some((
                    std::cmp::min(score, 100),
                    if cleaned.is_empty() {
                        text.to_string()
                    } else {
                        cleaned
                    },
                ));
            }
        }
    }
    None
}

impl MapRerankDocumentsChain {
    /// Create a new [`MapRerankDocumentsChain`] with the given LLM.
    pub fn new<L>(llm: L) -> Self
    where
        L: BaseChatModel + Send + Sync + 'static,
        L::Error: Into<ProviderError>,
    {
        Self {
            llm: wrap_chat_model(llm),
            map_prompt_template: DEFAULT_MAP_RERANK_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "map_rerank_documents".to_string(),
            verbose: false,
            top_k: 1,
            default_score: None,
        }
    }

    /// Set the map-phase prompt template.
    pub fn with_map_prompt(mut self, template: impl Into<String>) -> Self {
        self.map_prompt_template = template.into();
        self
    }

    /// Set the document variable name used in the map prompt.
    pub fn with_document_variable(mut self, name: impl Into<String>) -> Self {
        self.document_variable_name = name.into();
        self
    }

    /// Set the input key.
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Set the output key.
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// Set the chain name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the number of top results to return.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Configure the fallback score for LLM output without a parseable score.
    ///
    /// `None` (default) skips such documents; `Some(n)` ranks them with score `n`
    /// instead of silently assigning the old middle score 50 (P1-3).
    pub fn with_default_score(mut self, score: u32) -> Self {
        self.default_score = Some(score);
        self
    }

    /// Build Map stage prompt.
    pub fn build_map_prompt(&self, context: &str, input: &str) -> String {
        self.map_prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
    }

    async fn map_document(
        &self,
        doc: &Document,
        input: &str,
        index: usize,
    ) -> Result<Option<(u32, String)>, ChainError> {
        let prompt = self.build_map_prompt(&doc.content, input);
        if self.verbose {
            println!("\n--- Map document {} ---", index + 1);
        }
        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await.map_err(|e| {
            ChainError::ExecutionError(format!("Map call failed (document {}): {}", index + 1, e))
        })?;

        self.rank_output(&response.content, index)
    }

    /// Score an LLM output for one document: parse the relevance score and
    /// answer, or apply the configured default / skip (P1-3). Shared by the
    /// invoke and streaming map paths so the scoring rules never drift.
    fn rank_output(&self, output: &str, index: usize) -> Result<Option<(u32, String)>, ChainError> {
        // P1-3: no more silent middle score 50. Unscored output either uses the
        // configured default_score or is excluded from ranking entirely.
        let scored = match extract_score(output) {
            Some(pair) => Some(pair),
            None => match self.default_score {
                Some(n) => Some((n, output.trim().to_string())),
                None => {
                    log::warn!(
                        "MapRerank: document {} output has no parseable score; excluded from ranking",
                        index + 1
                    );
                    None
                }
            },
        };

        if self.verbose {
            if let Some((score, answer)) = &scored {
                println!(
                    "Document {} score: {}, answer: {}",
                    index + 1,
                    score,
                    truncate_str(answer, 80)
                );
            } else {
                println!("Document {} excluded (no score)", index + 1);
            }
        }
        Ok(scored)
    }

    /// Map-phase variant for the streaming path: tokens are collected from
    /// `stream_chat` so the full document answer is available for scoring
    /// (ranking requires the complete output).
    async fn map_document_stream(
        &self,
        doc: &Document,
        input: &str,
        index: usize,
    ) -> Result<Option<(u32, String)>, ChainError> {
        let prompt = self.build_map_prompt(&doc.content, input);
        if self.verbose {
            println!("\n--- Map document {} (stream) ---", index + 1);
        }
        let messages = vec![Message::human(&prompt)];
        let mut llm_stream = self.llm.stream_chat(messages, None).await.map_err(|e| {
            ChainError::StreamError(format!("Map stream failed (document {}): {}", index + 1, e))
        })?;

        let mut text = String::new();
        while let Some(chunk) = llm_stream.next().await {
            match chunk {
                Ok(chunk) => text.push_str(&chunk.text),
                Err(e) => {
                    return Err(ChainError::StreamError(format!(
                        "Map stream token error (document {}): {}",
                        index + 1,
                        e
                    )));
                }
            }
        }
        self.rank_output(&text, index)
    }

    /// Invoke with documents and input directly.
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<Vec<(u32, String)>, ChainError> {
        if documents.is_empty() {
            return Err(ChainError::ExecutionError(
                "Document list is empty".to_string(),
            ));
        }

        if self.verbose {
            println!("\n=== MapRerankDocumentsChain ===");
            println!("Document count: {}, Input: {}", documents.len(), input);
            println!("\n--- Map phase ---");
        }

        let mut map_futures = Vec::new();
        for (i, doc) in documents.iter().enumerate() {
            map_futures.push(self.map_document(doc, input, i));
        }
        // P1-3: drop documents whose output carried no score and had no default.
        let mut results: Vec<(u32, String)> = try_join_all(map_futures)
            .await?
            .into_iter()
            .flatten()
            .collect();

        if results.is_empty() {
            return Err(ChainError::ExecutionError(
                "All documents were excluded: no document produced a parseable score".to_string(),
            ));
        }

        results.sort_by(|a, b| b.0.cmp(&a.0));

        if self.verbose {
            println!("\n--- Rerank phase ---");
            for (i, (score, answer)) in results.iter().enumerate() {
                println!(
                    "Rank {}: score={}, answer={}",
                    i + 1,
                    score,
                    truncate_str(answer, 100)
                );
            }
        }

        let top_results: Vec<(u32, String)> = results.into_iter().take(self.top_k).collect();
        if self.verbose {
            println!("Selected {} best results", top_results.len());
            println!("=== MapRerankDocumentsChain complete ===\n");
        }
        Ok(top_results)
    }
}

#[async_trait]
impl BaseChain for MapRerankDocumentsChain {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key, "documents"]
    }
    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        // P2-8: validate inputs on the invoke path too, matching stream.
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents = crate::base::documents_from_input(inputs.get("documents"))?;

        let results = self.invoke_with_documents(documents, input).await?;
        let output_json: Vec<serde_json::Value> = results
            .iter()
            .map(|(score, answer)| serde_json::json!({"score": score, "answer": answer}))
            .collect();

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::Array(output_json));
        Ok(result)
    }

    /// Stream execution for MapRerankDocumentsChain.
    ///
    /// P2-2: the map phase runs via `stream_chat` per document (tokens
    /// accumulated for scoring), then the reranked top answer(s) are emitted.
    /// Raw token streaming of the final answer is impossible here — ranking
    /// requires each document's complete output — so the ranked result is the
    /// stream payload, produced without the base default's silent `unwrap_or("")`.
    async fn stream(&self, inputs: HashMap<String, Value>) -> Result<ChainStream, ChainError> {
        // P2-8: validate inputs on the stream path too (this was the one
        // document chain that skipped it entirely), matching the others.
        self.validate_inputs(&inputs)?;

        let input = inputs
            .get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents = crate::base::documents_from_input(inputs.get("documents"))?;
        if documents.is_empty() {
            return Err(ChainError::ExecutionError(
                "Document list is empty".to_string(),
            ));
        }

        let mut map_futures = Vec::new();
        for (i, doc) in documents.iter().enumerate() {
            map_futures.push(self.map_document_stream(doc, input, i));
        }
        let mut results: Vec<(u32, String)> = try_join_all(map_futures)
            .await?
            .into_iter()
            .flatten()
            .collect();

        if results.is_empty() {
            return Err(ChainError::ExecutionError(
                "All documents were excluded: no document produced a parseable score".to_string(),
            ));
        }

        results.sort_by(|a, b| b.0.cmp(&a.0));
        let top_results: Vec<(u32, String)> = results.into_iter().take(self.top_k).collect();

        let stream = futures_util::stream::once(async move {
            let text = top_results
                .iter()
                .map(|(_, answer)| answer.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            Ok(StreamToken {
                token: text,
                is_final: true,
            })
        });

        Ok(Box::pin(stream))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::language_models::{LLMResult, StreamChunk};
    use lc_core::runnables::RunnableConfig;
    use lc_core::{BaseLanguageModel, Runnable};
    use std::pin::Pin;

    /// Mock chat model whose stream returns a fixed scored answer per document.
    struct MockLLM;

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockLLM {
        type Error = ProviderError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: "Relevance score: 90\nAnswer: best answer".to_string(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockLLM {
        fn model_name(&self) -> &str {
            "mock"
        }
        fn get_num_tokens(&self, t: &str) -> usize {
            t.len()
        }
        fn with_temperature(self, _: f32) -> Self {
            self
        }
        fn with_max_tokens(self, _: usize) -> Self {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for MockLLM {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: "Relevance score: 90\nAnswer: best answer".to_string(),
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
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            let tokens = [Ok(StreamChunk::new(
                "Relevance score: 90\nAnswer: best answer",
            ))];
            Ok(Box::pin(futures_util::stream::iter(tokens)))
        }
    }

    #[tokio::test]
    async fn test_map_rerank_stream_emits_top_answer() {
        let chain = MapRerankDocumentsChain::new(MockLLM);
        let docs = vec![Document::new("doc one"), Document::new("doc two")];
        let docs_value = serde_json::to_value(docs).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Value::String("question".to_string()));
        inputs.insert("documents".to_string(), docs_value);

        let mut stream = chain.stream(inputs).await.unwrap();
        let mut tokens = Vec::new();
        while let Some(item) = stream.next().await {
            tokens.push(item.unwrap());
        }
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].is_final);
        assert!(
            tokens[0].token.contains("best answer"),
            "top answer should be streamed, got {:?}",
            tokens[0].token
        );
    }

    #[tokio::test]
    async fn test_map_rerank_stream_empty_documents() {
        let chain = MapRerankDocumentsChain::new(MockLLM);
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Value::String("q".to_string()));
        inputs.insert("documents".to_string(), serde_json::json!([]));
        let err = match chain.stream(inputs).await {
            Ok(_) => panic!("expected an execution error"),
            Err(e) => e,
        };
        assert!(matches!(err, ChainError::ExecutionError(_)));
    }

    #[test]
    fn test_rank_output_uses_default_score_for_unscored() {
        let chain = MapRerankDocumentsChain::new(MockLLM).with_default_score(40);
        let scored = chain.rank_output("plain answer without score", 0).unwrap();
        assert_eq!(scored, Some((40, "plain answer without score".to_string())));
    }

    #[test]
    fn test_rank_output_skips_unscored_without_default() {
        let chain = MapRerankDocumentsChain::new(MockLLM);
        assert_eq!(chain.rank_output("no score here", 0).unwrap(), None);
    }
}
