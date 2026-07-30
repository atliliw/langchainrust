// src/chains/document_chains/tests.rs
//! Tests for document processing chains.

use super::*;
use crate::language_models::OpenAIChat;
use crate::retrieval::Document;
use crate::OpenAIConfig;
use std::collections::HashMap;

fn create_test_config() -> OpenAIConfig {
    OpenAIConfig {
        api_key: "sk-6eb65fcf5d17491ca10b984efe1f43e7".to_string(),
        base_url: "https://llm-8xo1b7o30z27y2xc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
            .to_string(),
        model: "glm-5.2".to_string(),
        streaming: false,
        organization: None,
        frequency_penalty: None,
        max_tokens: None,
        presence_penalty: None,
        temperature: None,
        top_p: None,
        tools: None,
        tool_choice: None,
    }
}

#[test]
fn test_stuff_documents_format_documents() {
    let llm = OpenAIChat::new(create_test_config());
    let chain: StuffDocumentsChain<OpenAIChat> = StuffDocumentsChain::new(llm);

    let docs = vec![
        Document {
            content: "Hello world".to_string(),
            metadata: HashMap::new(),
            id: None,
        },
        Document {
            content: "Rust programming".to_string(),
            metadata: HashMap::new(),
            id: None,
        },
    ];

    let formatted = chain.format_documents(&docs);
    assert!(formatted.contains("Document 1:"));
    assert!(formatted.contains("Hello world"));
    assert!(formatted.contains("Document 2:"));
    assert!(formatted.contains("Rust programming"));
}

#[test]
fn test_stuff_documents_build_prompt() {
    let llm = OpenAIChat::new(create_test_config());
    let chain: StuffDocumentsChain<OpenAIChat> = StuffDocumentsChain::new(llm);

    let prompt = chain.build_prompt("some context", "what is Rust?");
    assert!(prompt.contains("some context"));
    assert!(prompt.contains("what is Rust?"));
}

#[test]
fn test_stuff_documents_truncation() {
    let llm = OpenAIChat::new(create_test_config());
    let chain: StuffDocumentsChain<OpenAIChat> =
        StuffDocumentsChain::new(llm).with_max_doc_length(5);

    let docs = vec![Document {
        content: "Hello world this is a long string".to_string(),
        metadata: HashMap::new(),
        id: None,
    }];

    let formatted = chain.format_documents(&docs);
    assert!(formatted.contains("[document truncated]"));
}

#[test]
fn test_refine_documents_build_prompts() {
    let llm = OpenAIChat::new(create_test_config());
    let chain: RefineDocumentsChain<OpenAIChat> = RefineDocumentsChain::new(llm);

    let initial = chain.build_initial_prompt("context1", "question");
    assert!(initial.contains("context1"));
    assert!(initial.contains("question"));

    let refine = chain.build_refine_prompt("context2", "question", "existing answer");
    assert!(refine.contains("context2"));
    assert!(refine.contains("question"));
    assert!(refine.contains("existing answer"));
}

#[test]
fn test_map_rerank_extract_score() {
    let text1 = "Relevance score: 85\nAnswer: The answer is 42";
    let (score1, answer1) = extract_score(text1);
    assert_eq!(score1, 85);
    assert!(answer1.contains("42"));

    let text2 = "Score: 70\nSome answer text";
    let (score2, _) = extract_score(text2);
    assert_eq!(score2, 70);

    let text3 = "No score here, just an answer";
    let (score3, _) = extract_score(text3);
    assert_eq!(score3, 50);
}

#[test]
fn test_map_rerank_build_map_prompt() {
    let llm = OpenAIChat::new(create_test_config());
    let chain: MapRerankDocumentsChain<OpenAIChat> = MapRerankDocumentsChain::new(llm);

    let prompt = chain.build_map_prompt("doc content", "what is this?");
    assert!(prompt.contains("doc content"));
    assert!(prompt.contains("what is this?"));
}

#[test]
fn test_map_reduce_build_prompts() {
    let llm = OpenAIChat::new(create_test_config());
    let chain: MapReduceDocumentsChain<OpenAIChat> = MapReduceDocumentsChain::new(llm);

    let map_prompt = chain.build_map_prompt("doc content", "question");
    assert!(map_prompt.contains("doc content"));
    assert!(map_prompt.contains("question"));

    let summaries = vec!["Answer 1".to_string(), "Answer 2".to_string()];
    let reduce_prompt = chain.build_reduce_prompt(&summaries, "question");
    assert!(reduce_prompt.contains("Answer 1"));
    assert!(reduce_prompt.contains("Answer 2"));
    assert!(reduce_prompt.contains("question"));
}

/// Real API test - StuffDocumentsChain
/// Run: cargo test test_stuff_documents_chain_invoke -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn test_stuff_documents_chain_invoke() {
    let llm = OpenAIChat::new(create_test_config());
    let chain: StuffDocumentsChain<OpenAIChat> = StuffDocumentsChain::new(llm);

    let docs = vec![
        Document {
            content: "Rust is a systems programming language.".to_string(),
            metadata: HashMap::new(),
            id: None,
        },
        Document {
            content: "Rust emphasizes safety and performance.".to_string(),
            metadata: HashMap::new(),
            id: None,
        },
    ];

    println!("\n=== Test StuffDocumentsChain ===");
    let result = chain
        .invoke_with_documents(docs, "What is Rust?")
        .await
        .unwrap();
    println!("Output: {}", result);
    assert!(!result.is_empty());
}

/// Real API test - RefineDocumentsChain
/// Run: cargo test test_refine_documents_chain_invoke -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn test_refine_documents_chain_invoke() {
    let llm = OpenAIChat::new(create_test_config());
    let chain: RefineDocumentsChain<OpenAIChat> = RefineDocumentsChain::new(llm);

    let docs = vec![
        Document {
            content: "Rust is a systems programming language.".to_string(),
            metadata: HashMap::new(),
            id: None,
        },
        Document {
            content: "Rust was created by Mozilla.".to_string(),
            metadata: HashMap::new(),
            id: None,
        },
    ];

    println!("\n=== Test RefineDocumentsChain ===");
    let result = chain
        .invoke_with_documents(docs, "What is Rust and who created it?")
        .await
        .unwrap();
    println!("Output: {}", result);
    assert!(!result.is_empty());
}

/// Real API test - MapRerankDocumentsChain
/// Run: cargo test test_map_rerank_documents_chain_invoke -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn test_map_rerank_documents_chain_invoke() {
    let llm = OpenAIChat::new(create_test_config());
    let chain: MapRerankDocumentsChain<OpenAIChat> = MapRerankDocumentsChain::new(llm);

    let docs = vec![
        Document {
            content: "Python is a high-level programming language.".to_string(),
            metadata: HashMap::new(),
            id: None,
        },
        Document {
            content: "Rust is a systems programming language focused on safety.".to_string(),
            metadata: HashMap::new(),
            id: None,
        },
    ];

    println!("\n=== Test MapRerankDocumentsChain ===");
    let result = chain
        .invoke_with_documents(docs, "What is Rust?")
        .await
        .unwrap();
    println!("Output: {:?}", result);
    assert!(!result.is_empty());
}

/// Real API test - MapReduceDocumentsChain
/// Run: cargo test test_map_reduce_documents_chain_invoke -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn test_map_reduce_documents_chain_invoke() {
    let llm = OpenAIChat::new(create_test_config());
    let chain: MapReduceDocumentsChain<OpenAIChat> = MapReduceDocumentsChain::new(llm);

    let docs = vec![
        Document {
            content: "Rust is a systems programming language.".to_string(),
            metadata: HashMap::new(),
            id: None,
        },
        Document {
            content: "Rust emphasizes memory safety without garbage collection.".to_string(),
            metadata: HashMap::new(),
            id: None,
        },
    ];

    println!("\n=== Test MapReduceDocumentsChain ===");
    let result = chain
        .invoke_with_documents(docs, "What is Rust?")
        .await
        .unwrap();
    println!("Output: {}", result);
    assert!(!result.is_empty());
}
