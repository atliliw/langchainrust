//! A5: agent-level offline record/replay — a real `FunctionCallingAgent` + a real `Calculator`
//! tool + `ReplayProvider`, running the tool-call loop over zero network.
//!
//! This is the real consumer of lc-testkit's second-phase capabilities:
//! - tool recording (A2 `RecordingProvider::bind_tools`) records the request-side `tools`;
//! - the replay side (A3 `ReplayProvider::bind_tools`) satisfies the agent's "tools-bound loop" precondition;
//! - FIFO replay replays "request a tool call → feed the tool result back → final answer",
//!   asserting the ToolStart/ToolEnd event sequence and the final answer.
//!
//! Recording-count semantics (since S2): the streaming executor goes through `plan_stream`, and a
//! tool-call round's model returns no text (`stream_chat` chunks only carry text/usage, not
//! `tool_calls`), so `plan_stream` falls back to the non-streaming `plan()` for the native
//! `tool_calls` — **one tool round consumes 2 recordings** (a streaming attempt + a non-streaming
//! replan), the final-answer round consumes only 1. Hence the fixture has 3 recordings:
//! [tool round × 2, answer round × 1], with the two tool-round recordings identical.

use std::sync::Arc;

use futures_util::StreamExt;
use lc_agents::{AgentExecutor, AgentStreamEvent, FunctionCallingAgent};
use lc_core::language_models::LLMResult;
use lc_core::tools::ToolCall;
use lc_testkit::{RecordedExchange, ReplayProvider};
use lc_tools::Calculator;

#[tokio::test]
async fn agent_offline_replays_tool_call_loop() {
    // 1. Hand-written deterministic recordings (equivalent to a fixture):
    //    - tool round (streaming attempt): the model requests a calculator call (tool_calls in the response, empty text)
    //    - tool round (non-streaming replan): `plan_stream` falls back to `plan()` on empty text,
    //      returning the same calculator call — identical to the previous round (see the module doc's
    //      count-semantics note)
    //    - answer round: after the tool result is fed back, the model gives the final answer "4"
    let tool_call_round = || RecordedExchange {
        messages: vec![],
        response: LLMResult {
            content: String::new(),
            model: "replay".to_string(),
            tool_calls: Some(vec![ToolCall::builder("call_1")
                .name("calculator")
                .arguments("{\"expression\":\"2+2\"}".to_string())
                .build()]),
            ..Default::default()
        },
        tools: None,
    };
    let exchanges = vec![
        tool_call_round(),
        tool_call_round(),
        RecordedExchange {
            messages: vec![],
            response: LLMResult {
                content: "4".to_string(),
                model: "replay".to_string(),
                ..Default::default()
            },
            tools: None,
        },
    ];
    let replay = ReplayProvider::from_exchanges(exchanges);

    // 2. Real agent (with the real Calculator tool) + real executor
    let agent = FunctionCallingAgent::new(replay, vec![Arc::new(Calculator::new())], None);
    let executor = AgentExecutor::new(Arc::new(agent), vec![Arc::new(Calculator::new())]);

    // 3. Streaming execution: the tool loop should emit ToolStart → ToolEnd, ending with FinalAnswer
    let mut stream = executor.stream("2+2=?".to_string());
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("回放 agent 执行不应失败"));
    }

    // 4. Assert: the tool event sequence + the final answer comes from the second round of replay
    assert!(events.len() >= 4, "事件数不足: {events:?}");
    assert!(
        matches!(&events[0], AgentStreamEvent::ToolStart { name, .. } if name == "calculator"),
        "第一个事件应为 calculator 的 ToolStart: {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentStreamEvent::ToolEnd { .. })),
        "应有 ToolEnd 事件: {events:?}"
    );
    match events.last().expect("有事件") {
        AgentStreamEvent::FinalAnswer { content } => {
            assert_eq!(content.trim(), "4", "最终答案应来自第二轮回放");
        }
        other => panic!("期望 FinalAnswer 终态,got {other:?}"),
    }
}
