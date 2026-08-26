// lc-providers/src/openai/chat/tests.rs

use super::*;

mod tests_env {
    use super::*;

    use std::env;

    fn save_and_set(key: &str, value: &str) -> Option<String> {
        let old = env::var(key).ok();
        env::set_var(key, value);
        old
    }

    fn restore(key: &str, old: Option<String>) {
        match old {
            Some(v) => env::set_var(key, v),
            None => env::remove_var(key),
        }
    }

    #[test]
    fn test_from_env_result_ok_when_key_set() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = save_and_set("OPENAI_API_KEY", "test-key-123");
        assert!(OpenAIChat::from_env_result().is_ok());
        restore("OPENAI_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_err_when_key_missing() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = env::var("OPENAI_API_KEY").ok();
        env::remove_var("OPENAI_API_KEY");
        assert!(OpenAIChat::from_env_result().is_err());
        restore("OPENAI_API_KEY", old);
    }
}

mod tests_q3_q4 {
    use super::*;

    fn message(content: Option<&str>, reasoning: Option<&str>) -> OpenAIMessage {
        OpenAIMessage {
            role: "assistant".to_string(),
            content: content.map(|s| s.to_string()),
            reasoning_content: reasoning.map(|s| s.to_string()),
            tool_calls: None,
        }
    }

    #[test]
    fn test_llm_result_keeps_content_when_non_empty() {
        let msg = message(Some("Hello"), Some("hidden chain-of-thought"));
        let result = OpenAIChat::llm_result_from_message(
            &msg,
            "gpt-test".to_string(),
            Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        );

        assert_eq!(result.content, "Hello");
        assert_eq!(
            result.thinking_content.as_deref(),
            Some("hidden chain-of-thought")
        );
        assert_eq!(result.model, "gpt-test");
        let usage = result.token_usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn test_llm_result_reasoning_does_not_leak_into_content() {
        // Q3: reasoning-only responses keep `content` empty — no fallback.
        let msg = message(Some(""), Some("reasoning only"));
        let result = OpenAIChat::llm_result_from_message(&msg, "gpt-test".to_string(), None);

        assert_eq!(result.content, "");
        assert_eq!(result.thinking_content.as_deref(), Some("reasoning only"));
    }

    #[test]
    fn test_llm_result_empty_content_no_thinking() {
        let msg = message(None, Some(""));
        let result = OpenAIChat::llm_result_from_message(&msg, "gpt-test".to_string(), None);

        assert_eq!(result.content, "");
        assert!(result.thinking_content.is_none());
    }

    #[tokio::test]
    async fn test_aggregate_stream_concatenates_tokens_in_order() {
        // Q4: the aggregation helper produces the full content in order.
        let stream: Pin<Box<dyn Stream<Item = Result<StreamChunk, OpenAIError>> + Send>> =
            Box::pin(futures_util::stream::iter(vec![
                Ok(StreamChunk::new("Hello")),
                Ok(StreamChunk::new(", ")),
                Ok(StreamChunk::new("world")),
            ]));

        let content = OpenAIChat::aggregate_stream(stream).await.unwrap();
        assert_eq!(content, "Hello, world");
    }

    #[tokio::test]
    async fn test_aggregate_stream_stops_on_error() {
        let stream: Pin<Box<dyn Stream<Item = Result<StreamChunk, OpenAIError>> + Send>> =
            Box::pin(futures_util::stream::iter(vec![
                Ok(StreamChunk::new("Hello")),
                Err(OpenAIError::Api("boom".to_string())),
                Ok(StreamChunk::new("never")),
            ]));

        let err = OpenAIChat::aggregate_stream(stream).await.unwrap_err();
        assert!(matches!(err, OpenAIError::Api(_)));
    }
}
