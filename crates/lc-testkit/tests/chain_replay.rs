//! Recording lands: a real `LLMChain` + `ReplayProvider`, offline, deterministic, no key needed.
//!
//! Corresponds to EXECUTION_PLAN §4.4 "turn at least 1 kind of known online failure (e.g. chains'
//! AccessDenied) into a pass via record/replay": the online `chains::f01` needs a real model
//! (skipped without a key); here a recorded fixture runs the same chain offline.

use std::collections::HashMap;

use lc_chains::BaseChain;
use lc_chains::LLMChain;
use lc_testkit::ReplayProvider;

#[tokio::test]
async fn chain_replay_answers_offline() {
    // cargo test's CWD = crates/lc-testkit; fixtures use a package-relative path
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
