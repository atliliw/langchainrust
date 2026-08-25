//! RecordingProvider → JSONL → ReplayProvider 往返:同一响应一致。
//!
//! 这是 harness 的核心闭环:真调一次落盘,再从文件回放得到相同结果。

mod common;

use common::FakeModel;
use lc_core::language_models::BaseChatModel;
use lc_schema::Message;
use lc_testkit::{RecordingProvider, ReplayProvider};

#[tokio::test]
async fn record_then_replay_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.jsonl");

    // 1. 真调一次(假模型),响应写入录制文件
    let recorded =
        RecordingProvider::new(FakeModel::new("Rust 是一门系统编程语言。"), &path).unwrap();
    let result = recorded
        .chat(
            vec![Message::system("测试"), Message::human("什么是 Rust?")],
            None,
        )
        .await
        .unwrap();
    assert_eq!(result.content, "Rust 是一门系统编程语言。");

    // 2. 文件恰好 1 行合法 JSONL
    let raw = std::fs::read_to_string(&path).unwrap();
    assert_eq!(raw.lines().count(), 1);

    // 3. 从文件回放:内容与 token 计数一致
    let replay = ReplayProvider::from_file(&path).unwrap();
    assert_eq!(replay.len(), 1);

    let replayed = replay
        .chat(vec![Message::human("什么是 Rust?")], None)
        .await
        .unwrap();
    assert_eq!(replayed.content, result.content);
    assert_eq!(replayed.model, result.model);
    // 假模型的 token 计数是确定的:2 条消息输入 / 1 输出 / 3 总计
    let tokens = replayed.token_usage.as_ref().expect("回放应带 token 计数");
    assert_eq!(tokens.prompt_tokens, 2);
    assert_eq!(tokens.completion_tokens, 1);
    assert_eq!(tokens.total_tokens, 3);
}
