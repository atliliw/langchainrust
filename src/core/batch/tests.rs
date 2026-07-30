// src/core/batch/tests.rs
//! Tests for the batch module.

#[cfg(test)]
mod tests {
    use super::super::client::BatchClient;
    use super::super::types::*;
    use crate::core::language_models::LLMResult;
    use crate::schema::Message;

    #[test]
    fn batch_request_serialization() {
        let req = BatchRequest {
            custom_id: "req-1".into(),
            messages: vec![Message::human("Hello")],
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
    }

    #[test]
    fn batch_id_clone_eq() {
        let id = BatchId("batch_abc".into());
        let id2 = id.clone();
        assert_eq!(id.0, id2.0);
    }

    #[test]
    fn batch_status_eq() {
        assert_eq!(BatchStatus::InProgress, BatchStatus::InProgress);
        assert_ne!(BatchStatus::Completed, BatchStatus::Failed);
    }

    #[test]
    fn batch_error_display() {
        let e = BatchError::Api("something went wrong".to_string());
        assert!(e.to_string().contains("something went wrong"));

        let e = BatchError::NotFound("batch_123".to_string());
        assert!(e.to_string().contains("batch_123"));

        let e = BatchError::Expired;
        assert!(e.to_string().contains("expired"));

        let e = BatchError::Timeout(60_000);
        assert!(e.to_string().contains("60000"));
    }

    #[test]
    fn openai_result_line_parsing() {
        let json = r#"{
            "custom_id": "req-1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": "Hello!"}}],
                    "model": "gpt-4o",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }
            }
        }"#;
        let line: OpenAIResultLine = serde_json::from_str(json).unwrap();
        assert_eq!(line.custom_id, "req-1");
        let body = line.response.unwrap().body.unwrap();
        assert_eq!(body.model.unwrap(), "gpt-4o");
        assert_eq!(
            body.choices[0].message.as_ref().unwrap().content.as_deref(),
            Some("Hello!")
        );
    }

    #[test]
    fn openai_result_line_with_error() {
        let json = r#"{
            "custom_id": "req-2",
            "error": {
                "message": "rate limit exceeded",
                "code": "rate_limit"
            }
        }"#;
        let line: OpenAIResultLine = serde_json::from_str(json).unwrap();
        assert_eq!(line.custom_id, "req-2");
        let err = line.error.unwrap();
        assert_eq!(err.message.unwrap(), "rate limit exceeded");
    }

    #[test]
    fn anthropic_result_line_parsing() {
        let json = r#"{
            "custom_id": "req-1",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [{"type": "text", "text": "Hi there!"}],
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {"input_tokens": 8, "output_tokens": 4}
                }
            }
        }"#;
        let line: AnthropicResultLine = serde_json::from_str(json).unwrap();
        assert_eq!(line.custom_id, "req-1");
        assert_eq!(line.result.result_type, "succeeded");
        let msg = line.result.message.unwrap();
        assert_eq!(msg.model, "claude-3-5-sonnet-20241022");
        assert_eq!(msg.content[0].text.as_deref(), Some("Hi there!"));
    }

    #[test]
    fn anthropic_result_line_errored() {
        let json = r#"{
            "custom_id": "req-2",
            "result": {
                "type": "errored",
                "error": {
                    "type": "invalid_request",
                    "message": "max_tokens too large"
                }
            }
        }"#;
        let line: AnthropicResultLine = serde_json::from_str(json).unwrap();
        assert_eq!(line.result.result_type, "errored");
        let err = line.result.error.unwrap();
        assert_eq!(err.message.unwrap(), "max_tokens too large");
    }

    #[test]
    fn parse_openai_jsonl() {
        let client = BatchClient::new(BatchProvider::OpenAI, "test-key");
        let jsonl = r#"{"custom_id":"r1","response":{"status_code":200,"body":{"choices":[{"message":{"content":"A1"}}],"model":"gpt-4o","usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}}}
{"custom_id":"r2","error":{"message":"fail","code":"err"}}"#;
        let results = client.parse_openai_results_jsonl(jsonl).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].custom_id, "r1");
        assert!(results[0].result.is_ok());
        let llm = results[0].result.as_ref().unwrap();
        assert_eq!(llm.content, "A1");
        assert_eq!(llm.model, "gpt-4o");
        assert!(llm.token_usage.is_some());
        assert_eq!(results[1].custom_id, "r2");
        assert!(results[1].result.is_err());
    }

    #[test]
    fn parse_anthropic_jsonl() {
        let client = BatchClient::new(BatchProvider::Anthropic, "test-key");
        let jsonl = r#"{"custom_id":"r1","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"B1"}],"model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":6,"output_tokens":2}}}}
{"custom_id":"r2","result":{"type":"errored","error":{"type":"invalid","message":"bad"}}}"#;
        let results = client.parse_anthropic_results_jsonl(jsonl).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].custom_id, "r1");
        assert!(results[0].result.is_ok());
        let llm = results[0].result.as_ref().unwrap();
        assert_eq!(llm.content, "B1");
        assert!(llm.token_usage.is_some());
        assert_eq!(results[1].custom_id, "r2");
        assert!(results[1].result.is_err());
    }

    #[test]
    fn message_to_openai_format() {
        let msg = Message::human("Hello");
        let val = BatchClient::message_to_openai(&msg);
        assert_eq!(val["role"], "user");
        assert_eq!(val["content"], "Hello");

        let sys = Message::system("You are helpful");
        let val = BatchClient::message_to_openai(&sys);
        assert_eq!(val["role"], "system");
    }

    #[test]
    fn message_to_anthropic_format() {
        let msg = Message::human("Hello");
        let val = BatchClient::message_to_anthropic(&msg);
        assert_eq!(val["role"], "user");
        assert_eq!(val["content"], "Hello");

        let ai = Message::ai("Response");
        let val = BatchClient::message_to_anthropic(&ai);
        assert_eq!(val["role"], "assistant");

        // H40: System messages now mapped to "system" role (extracted to top-level by caller)
        let sys = Message::system("You are helpful");
        let val = BatchClient::message_to_anthropic(&sys);
        assert_eq!(val["role"], "system");
        assert_eq!(val["content"], "You are helpful");
    }

    #[test]
    fn batch_client_new_default_urls() {
        let openai = BatchClient::new(BatchProvider::OpenAI, "key");
        assert!(openai.base_url.contains("openai.com"));

        let anthropic = BatchClient::new(BatchProvider::Anthropic, "key");
        assert!(anthropic.base_url.contains("anthropic.com"));
    }

    #[test]
    fn batch_client_with_base_url() {
        let client = BatchClient::new(BatchProvider::OpenAI, "key")
            .with_base_url("https://custom.api.com/v1");
        assert_eq!(client.base_url, "https://custom.api.com/v1");
    }

    #[test]
    fn openai_batch_response_deserialize() {
        let json = r#"{
            "id": "batch_abc123",
            "status": "in_progress",
            "input_file_id": "file-xyz",
            "output_file_id": null,
            "error_file_id": null
        }"#;
        let resp: OpenAIBatchResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "batch_abc123");
        assert_eq!(resp.status, "in_progress");
        assert_eq!(resp.input_file_id.unwrap(), "file-xyz");
        assert!(resp.output_file_id.is_none());
    }

    #[test]
    fn anthropic_batch_response_deserialize() {
        let json = r#"{
            "id": "msgbatch_abc",
            "processing_status": "in_progress",
            "request_counts": {
                "processing": 5,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0
            }
        }"#;
        let resp: AnthropicBatchResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "msgbatch_abc");
        assert_eq!(resp.processing_status.as_deref(), Some("in_progress"));
        let counts = resp.request_counts.unwrap();
        assert_eq!(counts.processing, 5);
    }

    #[test]
    fn batch_result_serialization() {
        let ok_result = BatchResult {
            custom_id: "r1".into(),
            result: Ok(LLMResult {
                content: "Hello".into(),
                model: "gpt-4o".into(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            }),
        };
        let json = serde_json::to_string(&ok_result).unwrap();
        assert!(json.contains("r1"));
        assert!(json.contains("Hello"));

        let err_result = BatchResult {
            custom_id: "r2".into(),
            result: Err("failed".into()),
        };
        let json = serde_json::to_string(&err_result).unwrap();
        assert!(json.contains("failed"));
    }
}
