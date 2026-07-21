//! 示范:用 wiremock 测 OpenAIChat,不打真实网络。
//!
//! v0.4.1 #1(wiremock 改造网络测试)基础设施示范。
//! 真实 API 测试应标 `#[ignore]`,默认测试走 mock(见 `common::mock_openai_chat_server`)。

#[path = "../common/mod.rs"]
mod common;
use common::mock_openai_chat_server;
use langchainrust::{BaseChatModel, Message, OpenAIChat, OpenAIConfig};

#[tokio::test]
async fn test_openai_chat_with_mock() {
    // 起 mock server,base_url 指向它(不打真实网络)
    let (_server, base_url) = mock_openai_chat_server("你好,这是 mock 回复").await;

    let config = OpenAIConfig {
        api_key: "test-key".to_string(),
        base_url,
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    };
    let llm = OpenAIChat::new(config);

    let result = llm.chat(vec![Message::human("hi")], None).await.unwrap();
    assert_eq!(result.content, "你好,这是 mock 回复");
}
