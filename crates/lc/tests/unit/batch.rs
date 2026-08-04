//! Batch API unit tests - v0.5.0 #11

use langchainrust::{
    BatchClient, BatchError, BatchId, BatchProvider, BatchRequest, BatchResult, BatchStatus,
    LLMResult, Message,
};

#[test]
fn batch_request_round_trip() {
    let req = BatchRequest {
        custom_id: "req-1".into(),
        messages: vec![Message::human("Hello"), Message::system("Be helpful")],
        model: "gpt-4o".into(),
        temperature: Some(0.7),
        max_tokens: Some(256),
    };
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("req-1"));
    assert!(json.contains("gpt-4o"));

    let back: BatchRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.custom_id, "req-1");
    assert_eq!(back.model, "gpt-4o");
    assert_eq!(back.temperature, Some(0.7));
    assert_eq!(back.max_tokens, Some(256));
    assert_eq!(back.messages.len(), 2);
}

#[test]
fn batch_request_optional_fields_none() {
    let req = BatchRequest {
        custom_id: "req-2".into(),
        messages: vec![Message::human("Hi")],
        model: "claude-3-5-sonnet-20241022".into(),
        temperature: None,
        max_tokens: None,
    };
    let json = serde_json::to_string(&req).unwrap();
    let back: BatchRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.temperature, None);
    assert_eq!(back.max_tokens, None);
}

#[test]
fn batch_id_clone_and_equality() {
    let id = BatchId("batch_abc".into());
    let id2 = id.clone();
    assert_eq!(id.0, id2.0);
}

#[test]
fn batch_status_variants() {
    assert_eq!(BatchStatus::InProgress, BatchStatus::InProgress);
    assert_ne!(BatchStatus::Completed, BatchStatus::Failed);
    assert_ne!(BatchStatus::Expired, BatchStatus::Cancelled);
}

#[test]
fn batch_error_display_messages() {
    let e = BatchError::Api("something went wrong".to_string());
    assert!(e.to_string().contains("something went wrong"));

    let e = BatchError::NotFound("batch_123".to_string());
    assert!(e.to_string().contains("batch_123"));

    let e = BatchError::Expired;
    assert!(e.to_string().contains("expired"));

    let e = BatchError::Timeout(60_000);
    assert!(e.to_string().contains("60000"));

    let e = BatchError::Failed("internal error".to_string());
    assert!(e.to_string().contains("internal error"));

    // Serialization variant
    let bad_json = "not json";
    let e: BatchError = serde_json::from_str::<serde_json::Value>(bad_json)
        .map_err(BatchError::from)
        .unwrap_err();
    assert!(e.to_string().contains("serialization error"));
}

#[test]
fn batch_result_ok_serialization() {
    let ok_result = BatchResult {
        custom_id: "r1".into(),
        result: Ok(LLMResult {
            content: "Hello world".into(),
            model: "gpt-4o".into(),
            token_usage: None,
            tool_calls: None,
            thinking_content: None,
        }),
    };
    let json = serde_json::to_string(&ok_result).unwrap();
    assert!(json.contains("r1"));
    assert!(json.contains("Hello world"));
}

#[test]
fn batch_result_err_serialization() {
    let err_result = BatchResult {
        custom_id: "r2".into(),
        result: Err("request failed".into()),
    };
    let json = serde_json::to_string(&err_result).unwrap();
    assert!(json.contains("r2"));
    assert!(json.contains("request failed"));
}

#[test]
fn batch_client_default_urls() {
    let _openai = BatchClient::new(BatchProvider::OpenAI, "test-key");
    let _anthropic = BatchClient::new(BatchProvider::Anthropic, "test-key");
    // Clients created successfully with default URLs
}

#[test]
fn batch_client_with_base_url_override() {
    let _client = BatchClient::new(BatchProvider::OpenAI, "key")
        .with_base_url("https://proxy.example.com/v1");
    // Client created successfully with custom base URL
}

#[test]
fn batch_provider_equality() {
    assert_eq!(BatchProvider::OpenAI, BatchProvider::OpenAI);
    assert_eq!(BatchProvider::Anthropic, BatchProvider::Anthropic);
    assert_ne!(BatchProvider::OpenAI, BatchProvider::Anthropic);
}
