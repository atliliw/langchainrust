//! A5:agent 级离线录播——真 `FunctionCallingAgent` + 真 `Calculator` 工具 +
//! `ReplayProvider`,零网络跑通工具调用循环。
//!
//! 这是 lc-testkit 二期新能力的真实消费方:
//! - 工具录制(A2 `RecordingProvider::bind_tools`)录下请求侧 `tools`;
//! - 回放侧(A3 `ReplayProvider::bind_tools`)让 agent"绑了工具的循环"前提成立;
//! - FIFO 回放把「请求工具调用 → 工具结果喂回 → 最终答案」放回,
//!   断言 ToolStart/ToolEnd 事件序列与最终答案落地。
//!
//! 录播条数的语义(S2 起):流式 executor 走 `plan_stream`,而工具调用轮模型
//! 不回文本(`stream_chat` chunk 只带 text/usage、不携带 tool_calls),`plan_stream`
//! 在空文本时回退非流式 `plan()` 拿原生 tool_calls —— **一个工具轮消费 2 条
//! 录播**(流式尝试 + 非流式重规划),最终答案轮只消费 1 条。故 fixture 为
//! 3 条:[工具轮×2, 答案轮×1],两条工具轮录播内容相同。

use std::sync::Arc;

use futures_util::StreamExt;
use lc_agents::{AgentExecutor, AgentStreamEvent, FunctionCallingAgent};
use lc_core::language_models::LLMResult;
use lc_core::tools::ToolCall;
use lc_testkit::{RecordedExchange, ReplayProvider};
use lc_tools::Calculator;

#[tokio::test]
async fn agent_offline_replays_tool_call_loop() {
    // 1. 手写确定性录播(等价 fixture):
    //    - 工具轮(流式尝试):模型请求调用 calculator(响应带 tool_calls,文本为空)
    //    - 工具轮(非流式重规划):`plan_stream` 空文本回退 `plan()`,同样返回
    //      calculator 调用 —— 与上一轮内容一致(见模块 doc 的条数语义说明)
    //    - 答案轮:工具结果喂回后,模型给出最终答案 "4"
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

    // 2. 真 agent(带真 Calculator 工具)+ 真 executor
    let agent = FunctionCallingAgent::new(replay, vec![Arc::new(Calculator::new())], None);
    let executor = AgentExecutor::new(Arc::new(agent), vec![Arc::new(Calculator::new())]);

    // 3. 流式执行:工具循环应发出 ToolStart → ToolEnd,最终以 FinalAnswer 收尾
    let mut stream = executor.stream("2+2=?".to_string());
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("回放 agent 执行不应失败"));
    }

    // 4. 断言:工具事件序列 + 最终答案来自第二轮回放
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
