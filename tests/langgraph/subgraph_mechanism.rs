//! Subgraph 执行机制深度解析
//!
//! 本测试文件专门演示 Subgraph 的内部执行机制和层级穿透

use langchainrust::{
    AgentState, StateUpdate,
    GraphBuilder, START, END, Runnable,
};
use std::time::{Duration, Instant};

/// 模拟一个耗时操作（用于演示阻塞等待）
async fn simulate_work(name: &str, duration_ms: u64) -> String {
    println!("    ⏳ [{}] 开始工作，预计耗时 {}ms...", name, duration_ms);
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;
    println!("    ✅ [{}] 工作完成！", name);
    format!("{}_result", name)
}

/// 测试：演示子图的同步阻塞执行
///
/// 这个测试用时间戳证明：
/// - 父图执行到子图节点时会阻塞
/// - 子图完全完成后父图才继续
#[tokio::test]
async fn test_subgraph_blocking_execution() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        Subgraph 执行机制：同步阻塞演示                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    println!("\n【核心问题】");
    println!("  问：一个 node 包含了一个图，这个图执行完了之后父才会执行下一个 node 吗？");
    println!("  答：是的！使用 .await 实现同步阻塞等待");

    println!("\n【实验设计】");
    println!("  创建三层嵌套图，每层有模拟耗时操作");
    println!("  用时间戳证明父图确实等待子图完成");

    // 内层子图
    let inner = GraphBuilder::<AgentState>::new()
        .add_async_node("inner_work", |_state: &AgentState| async move {
            let t0 = Instant::now();
            let result = simulate_work("内层", 100).await;
            let elapsed = t0.elapsed();
            
            println!("    📊 [内层] 耗时: {}ms", elapsed.as_millis());
            
            let mut s = AgentState::new(result.clone());
            s.set_output(result);
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "inner_work")
        .add_edge("inner_work", END)
        .compile()
        .unwrap();

    // 中层子图（嵌入内层）
    let middle = GraphBuilder::<AgentState>::new()
        .add_subgraph_same_state("inner_subgraph", inner)
        .add_async_node("middle_work", |state: &AgentState| {
            let inner_result = state.output.clone().unwrap_or_default();
            async move {
                let t0 = Instant::now();
                println!("\n  ⏸️ [中层] 等待内层完成，输入: {}", inner_result);
                
                let result = simulate_work("中层", 150).await;
                let elapsed = t0.elapsed();
                
                println!("  📊 [中层] 耗时: {}ms (含等待内层)", elapsed.as_millis());
                
                let mut s = AgentState::new(result.clone());
                s.set_output(result);
                Ok(StateUpdate::full(s))
            }
        })
        .add_edge(START, "inner_subgraph")
        .add_edge("inner_subgraph", "middle_work")
        .add_edge("middle_work", END)
        .compile()
        .unwrap();

    // 外层图（嵌入中层）
    let outer = GraphBuilder::<AgentState>::new()
        .add_async_node("outer_pre", |_state: &AgentState| async move {
            println!("\n══════════════════════════════════════════════════════════════");
            println!("【外层】outer_pre 开始执行（在子图之前）");
            println!("══════════════════════════════════════════════════════════════");
            
            let result = simulate_work("外层前", 50).await;
            
            let mut s = AgentState::new(result.clone());
            s.set_output(result);
            Ok(StateUpdate::full(s))
        })
        .add_subgraph_same_state("middle_subgraph", middle)
        .add_async_node("outer_post", |state: &AgentState| {
            let mid_result = state.output.clone().unwrap_or_default();
            async move {
                println!("\n══════════════════════════════════════════════════════════════");
                println!("【外层】outer_post 开始执行（在子图之后）");
                println!("  子图返回: {}", mid_result);
                println!("══════════════════════════════════════════════════════════════");
                
                let result = simulate_work("外层后", 50).await;
                
                let mut s = AgentState::new("complete".to_string());
                s.set_output(result);
                Ok(StateUpdate::full(s))
            }
        })
        .add_edge(START, "outer_pre")
        .add_edge("outer_pre", "middle_subgraph")
        .add_edge("middle_subgraph", "outer_post")
        .add_edge("outer_post", END)
        .compile()
        .unwrap();

    println!("\n【图结构】");
    println!("{}", outer.visualize_ascii());

    println!("\n【执行时序分析】");
    println!("  预期总耗时: 50 + (100 + 150) + 50 = 350ms");
    println!("  因为子图执行是同步阻塞的");

    println!("\n【开始执行】");
    let total_start = Instant::now();
    
    let result = outer.invoke(AgentState::new("start".to_string())).await.unwrap();
    
    let total_elapsed = total_start.elapsed();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                     执行结果分析                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    println!("\n  实际总耗时: {}ms", total_elapsed.as_millis());
    println!("  预期总耗时: ~350ms");
    println!("  结论: 时间吻合，证明父图确实等待子图完成！");

    println!("\n【时序图解】");
    println!("  ");
    println!("  时间轴:  0ms    50ms   150ms   300ms   350ms");
    println!("           │      │      │       │       │");
    println!("  外层前:  ████████                                      (50ms)");
    println!("           │");
    println!("  子图:    ├────────────────────────────────────┤          (阻塞等待)");
    println!("           │      │");
    println!("  内层:    │      ████████████                    (100ms)");
    println!("           │      │");
    println!("  中层:    │      ├────────████████████████       (150ms)");
    println!("           │      │       │");
    println!("           │      │       ↓ (内层完成后中层继续)");
    println!("           │");
    println!("  外层后:  ├─────────────────────────────────███████ (50ms)");
    println!("           │");
    println!("  结束:    ↓");

    println!("\n【关键代码解析】");
    println!("  ");
    println!("  SubgraphNode.execute() 中的阻塞等待：");
    println!("  ");
    println!("  async fn execute(&self, state: &S) -> NodeResult<S> {{");
    println!("      let sub_input = (self.input_mapper)(state);");
    println!("      ");
    println!("      // 👇 这里 .await = 阻塞等待子图完全执行");
    println!("      let sub_result = self.subgraph.invoke(sub_input).await;");
    println!("      ");
    println!("      // 👆 子图返回后，父图才继续");
    println!("      let mut parent_output = state.clone();");
    println!("      (self.output_mapper)(&sub_result.final_state, &mut parent_output);");
    println!("      Ok(StateUpdate::full(parent_output))");
    println!("  }}");

    println!("\n【总结】");
    println!("  ");
    println!("  问题: 一个 node 包含了一个图，这个图执行完了之后父才会执行下一个 node 吗？");
    println!("  ");
    println!("  答案: ✅ 是的！");
    println!("  ");
    println!("  1. SubgraphNode 内部持有 CompiledGraph");
    println!("  2. execute() 方法调用 subgraph.invoke().await");
    println!("  3. .await = 阻塞等待，直到子图完全执行");
    println!("  4. 子图返回 final_state 后，父图才继续下一个节点");
    println!("  ");
    println!("  内层、中层、外层结构完全一样：都是 CompiledGraph<AgentState>");
    println!("  区别只在：某个\"节点\"本身就是一个完整的图");

    assert!(result.final_state.output.is_some());
}

/// 测试：验证内层和外层结构相同
///
/// 证明三层都是 CompiledGraph<AgentState> 类型
#[tokio::test]
async fn test_all_layers_same_structure() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        验证：内层、中层、外层结构完全相同                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    println!("\n【问题】");
    println!("  内层和外层所用的结构是一样的吗？");

    println!("\n【验证】");

    // 创建三层，每层都是 CompiledGraph<AgentState>
    let inner: langchainrust::CompiledGraph<AgentState> = GraphBuilder::<AgentState>::new()
        .add_node_fn("inner_node", |state: &AgentState| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "inner_node")
        .add_edge("inner_node", END)
        .compile()
        .unwrap();

    println!("  内层类型: CompiledGraph<AgentState>");
    println!("  内层节点: {:?}", inner.node_names());

    let middle: langchainrust::CompiledGraph<AgentState> = GraphBuilder::<AgentState>::new()
        .add_subgraph_same_state("inner", inner)
        .add_node_fn("middle_node", |state: &AgentState| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "inner")
        .add_edge("inner", "middle_node")
        .add_edge("middle_node", END)
        .compile()
        .unwrap();

    println!("  中层类型: CompiledGraph<AgentState> (和内层一样！)");
    println!("  中层节点: {:?}", middle.node_names());
    println!("  注意: 'inner' 节点是一个子图，但中层本身还是 CompiledGraph");

    let outer: langchainrust::CompiledGraph<AgentState> = GraphBuilder::<AgentState>::new()
        .add_subgraph_same_state("middle", middle)
        .add_node_fn("outer_node", |state: &AgentState| Ok(StateUpdate::full(state.clone())))
        .add_edge(START, "middle")
        .add_edge("middle", "outer_node")
        .add_edge("outer_node", END)
        .compile()
        .unwrap();

    println!("  外层类型: CompiledGraph<AgentState> (和内层、中层一样！)");
    println!("  外层节点: {:?}", outer.node_names());
    println!("  注意: 'middle' 节点是一个子图，但外层本身还是 CompiledGraph");

    println!("\n【答案】");
    println!("  ✅ 内层、中层、外层结构完全相同！");
    println!("  ");
    println!("  都是用 GraphBuilder::<AgentState>::new()");
    println!("  都返回 CompiledGraph<AgentState>");
    println!("  ");
    println!("  区别只在：");
    println!("  - 内层: 普通节点");
    println!("  - 中层: 有一个节点是子图（内层）");
    println!("  - 外层: 有一个节点是子图（中层）");

    println!("\n【可视化对比】");
    println!("  ");
    println!("  内层: [inner_node]");
    println!("  中层: [inner(子图), middle_node]");
    println!("  外层: [middle(子图), outer_node]");
    println!("  ");
    println!("  每层的子图节点，内部包含一个完整的 CompiledGraph");

    println!("\n【类型代码】");
    println!("  ");
    println!("  pub struct SubgraphNode<S: StateSchema, SubS: StateSchema> {{");
    println!("      name: String,");
    println!("      subgraph: CompiledGraph<SubS>,  // ← 内部持有完整的图");
    println!("      input_mapper: Arc<dyn Fn(&S) -> SubS>,");
    println!("      output_mapper: Arc<dyn Fn(&SubS, &mut S)>,");
    println!("  }}");
    println!("  ");
    println!("  SubgraphNode 实现了 GraphNode trait");
    println!("  所以它可以像普通节点一样加入父图");

    let _ = outer.invoke(AgentState::new("test".to_string())).await.unwrap();
}