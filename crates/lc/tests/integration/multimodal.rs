//! 多模态 vision 集成测试 - 需要 API Key
//!
//! 测试 OpenAI Vision 识别图片内容。
//!
//! 手动运行:
//! ```bash
//! cargo test --test integration_multimodal -- --ignored
//! ```

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::schema::Message;
use langchainrust::{BaseChatModel, OpenAIChat, OpenAIConfig};

/// 构造支持 vision 的 LLM(gpt-4o-mini)
fn vision_llm() -> OpenAIChat {
    let cfg = TestConfig::get();
    OpenAIChat::new(OpenAIConfig {
        api_key: cfg.api_key.clone(),
        base_url: cfg.base_url.clone(),
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    })
}

#[tokio::test]
#[ignore = "需要 API Key 和网络访问图片"]
async fn test_vision_with_image_url() {
    let llm = vision_llm();
    let messages = vec![
        Message::system("你是图像分析助手,用一句话回答。"),
        Message::human_with_image(
            "描述这张图片里有什么。",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/240px-Cat03.jpg",
        ),
    ];
    let response = llm.chat(messages, None).await.expect("chat 失败");
    println!("Vision 回答: {}", response.content);
    assert!(!response.content.is_empty());
}

#[tokio::test]
#[ignore = "需要 API Key"]
async fn test_vision_message_has_images() {
    // 验证 human_with_image 构造的消息携带 images
    let msg = Message::human_with_image("看图", "https://example.com/img.png");
    assert!(msg.has_images());
    assert_eq!(msg.images.len(), 1);
}
