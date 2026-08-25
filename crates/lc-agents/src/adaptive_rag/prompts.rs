// lc-agents/src/adaptive_rag/prompts.rs
//! Prompt templates used by [`crate::adaptive_rag::AdaptiveRAG`].

/// Routing prompt: ask the LLM to classify the query into one of three
/// strategies.
pub(crate) const ROUTING_PROMPT: &str = r#"Given the following query, decide whether retrieval is needed:
- "no_retrieval": The query can be answered from general knowledge
- "single_search": A single search is sufficient
- "multi_query": The query is complex and needs multiple search angles

Query: {query}

Respond with exactly one of: no_retrieval, single_search, multi_query"#;

/// System prompt used for the final generation step.
pub(crate) const GENERATE_SYSTEM_PROMPT: &str = r#"You are a helpful assistant. Answer the user's question based on the provided context when available. If no context is provided, use your general knowledge. Be concise and accurate."#;

/// Prompt used to generate alternative query variants in multi-query mode.
pub(crate) const MULTI_QUERY_PROMPT: &str = r#"You are an AI language model assistant. Your task is to generate {count} different versions of the given user question to retrieve relevant documents from a vector database.

By generating multiple perspectives on the user question, your goal is to help overcome some of the limitations of distance-based similarity search.

Provide these alternative questions separated by newlines.

Original question: {question}

Alternative questions:"#;
