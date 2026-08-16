//! 多模态 vision 集成测试
//!
//! 验证 Message 的多模态构造能力(纯内存,不触网)。

use langchainrust::schema::Message;

#[tokio::test]
async fn test_vision_message_has_images() {
    // 验证 human_with_image 构造的消息携带 images
    let msg = Message::human_with_image("看图", "https://example.com/img.png");
    assert!(msg.has_images());
    assert_eq!(msg.images.len(), 1);
}
