//! 录播落地:真 `LLMChain` + `ReplayProvider`,离线、确定性,不依赖任何 key。
//!
//! 对应 EXECUTION_PLAN §4.4「至少 1 类已知在线失败(如 chains 的 AccessDenied)
//! 转为录播回放通过」:在线版 `chains::f01` 需要真实模型(无 key 跳过),
//! 这里用录制 fixture 把同一条链离线跑通。

use std::collections::HashMap;

use lc_chains::BaseChain;
use lc_chains::LLMChain;
use lc_testkit::ReplayProvider;

#[tokio::test]
async fn chain_replay_answers_offline() {
    // cargo test 的 CWD = crates/lc-testkit,fixture 用包内相对路径
    let llm = ReplayProvider::from_file("fixtures/llm_chain_f01.jsonl")
        .expect("fixture 存在: fixtures/llm_chain_f01.jsonl");
    let chain = LLMChain::new(llm, "用一句话回答:{question}");

    let mut inputs = HashMap::new();
    inputs.insert(
        "question".to_string(),
        serde_json::Value::String("什么是 Rust?".to_string()),
    );

    let result = chain.invoke(inputs).await.expect("回放链调用失败");
    let answer = result
        .values()
        .next()
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    assert!(!answer.trim().is_empty(), "回放回答不能为空");
    println!("[chain_replay] 回放回答: {:?}", answer);
}
