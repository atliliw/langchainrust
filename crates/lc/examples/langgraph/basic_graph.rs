//! 基础图工作流示例
//!
//! 展示 LangGraph 线性图:START -> greet -> reply -> END(无需 API Key)。
//!
//! # 运行
//! ```bash
//! cargo run --example langgraph_basic_graph
//! ```

use langchainrust::{AgentState, GraphBuilder, StateUpdate, END, START};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compiled = GraphBuilder::<AgentState>::new()
        .add_node_fn("greet", |state: &AgentState| {
            Ok(StateUpdate::full(AgentState::new(format!(
                "你好:{}",
                state.input
            ))))
        })
        .add_node_fn("reply", |state: &AgentState| {
            let mut s = state.clone();
            s.set_output(format!("回复:{}", state.input));
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "greet")
        .add_edge("greet", "reply")
        .add_edge("reply", END)
        .compile()?;

    let result = compiled.invoke(AgentState::new("世界".to_string())).await?;
    println!("input:  {}", result.final_state.input);
    println!("output: {:?}", result.final_state.output);
    Ok(())
}
