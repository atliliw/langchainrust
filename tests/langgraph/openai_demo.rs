//! LangGraph 高级功能 + OpenAI 真实调用集成测试
//!
//! 本测试文件演示 LangGraph 的三个高级功能与 OpenAI API 的真实集成：
//!
//! 1. Subgraph（子图）功能：
//!    - 将复杂工作流封装成可复用的子图组件
//!    - 子图可以嵌入到父图中作为一个节点执行
//!    - 支持多层嵌套，实现模块化设计
//!
//! 2. Persistence（持久化）功能：
//!    - 将编译后的图定义保存为 JSON 文件
//!    - 从文件加载图定义，实现跨会话复用
//!    - 适用于生产部署和版本控制
//!
//! 3. 多节点顺序执行：
//!    - 多个异步节点按顺序执行
//!    - 每个节点可独立调用 OpenAI
//!    - 状态在节点间正确传递
//!
//! 运行方式：
//! cargo test --test langgraph_openai_demo -- --ignored --nocapture

use langchainrust::language_models::openai::OpenAIChat;
use langchainrust::schema::Message;
use langchainrust::{AgentState, GraphBuilder, MessageEntry, Runnable, StateUpdate, END, START};

/// OpenAI API 调用辅助函数
///
/// 使用项目中已配置的 API Key（在 config.rs 中设置）：
/// - api_key: sk-l0YYMX65mCYRlTJYH0ptf4BFpqJwm8Xo9Z5IMqSZD0yOafl6
/// - base_url: https://api.openai-proxy.org/v1
/// - model: gpt-3.5-turbo
///
/// 参数：
/// - prompt: 用户提示词
/// - system: 系统提示词（设定 AI 角色）
///
/// 返回：OpenAI 的响应内容
async fn call_openai(prompt: &str, system: &str) -> String {
    let client = OpenAIChat::from_env();
    let messages = vec![Message::system(system), Message::human(prompt)];
    client.invoke(messages, None).await.unwrap().content
}

/// 测试 1：Subgraph + OpenAI 真实调用
///
/// 场景：创建一个"翻译工作流"子图，然后在父图中使用它
///
/// 子图流程：
/// 1. detect 节点：调用 OpenAI 检测输入文本的语言
/// 2. translate 节点：调用 OpenAI 翻译文本
///
/// 父图流程：
/// 1. 嵌入子图 translator
/// 2. 整个子图作为单个节点执行
///
/// 打印输出：
/// - 每个节点的执行结果
/// - 最终翻译输出
#[tokio::test]
#[ignore = "需要真实 OpenAI API"]
async fn test_subgraph_with_openai() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         测试 1: Subgraph（子图）+ OpenAI 真实调用           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n┌─────────────────────────────────────────┐");
    println!("│         【第 1 部分】创建子图            │");
    println!("└─────────────────────────────────────────┘");
    println!("子图是一个独立的工作流，可以嵌入到其他图中：");

    println!("\n  子图内部结构：");
    println!("  ┌─────────────────────────────────────┐");
    println!("  │      【子图: translator】           │");
    println!("  │                                     │");
    println!("  │   START → detect → translate → END │");
    println!("  │                                     │");
    println!("  │   • detect: 检测语言（调用OpenAI）  │");
    println!("  │   • translate: 翻译文本（调用OpenAI)│");
    println!("  └─────────────────────────────────────┘");

    let subgraph = GraphBuilder::<AgentState>::new()
        .add_async_node("detect", |_state: &AgentState| async move {
            println!("\n    >>> [子图内部] detect 节点执行 <<<");
            let lang = call_openai("'Hello' 是什么语言？", "只回答语言名").await;
            println!("    >>> [子图内部] detect 完成: {} <<<", lang);

            let mut s = AgentState::new(lang.clone());
            s.add_message(MessageEntry::ai(lang));
            Ok(StateUpdate::full(s))
        })
        .add_async_node("translate", |_state: &AgentState| async move {
            println!("\n    >>> [子图内部] translate 节点执行 <<<");
            let result = call_openai("把 'Hello' 翻译成中文", "只返回翻译结果").await;
            println!("    >>> [子图内部] translate 完成: {} <<<", result);

            let mut s = AgentState::new("Hello".to_string());
            s.set_output(result);
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "detect")
        .add_edge("detect", "translate")
        .add_edge("translate", END)
        .compile()
        .unwrap();

    println!("\n  子图可视化输出:");
    println!("{}", subgraph.visualize_ascii());

    println!("\n┌─────────────────────────────────────────┐");
    println!("│         【第 2 部分】创建父图            │");
    println!("└─────────────────────────────────────────┘");
    println!("父图将子图作为一个节点嵌入：");

    println!("\n  父图结构：");
    println!("  ┌─────────────────────────────────────┐");
    println!("  │      【父图: main_workflow】        │");
    println!("  │                                     │");
    println!("  │   START → [translator] → END       │");
    println!("  │                                     │");
    println!("  │   • translator = 整个子图作为节点   │");
    println!("  │     (包含 detect + translate)       │");
    println!("  └─────────────────────────────────────┘");

    let parent = GraphBuilder::<AgentState>::new()
        .add_subgraph_same_state("translator", subgraph)
        .add_edge(START, "translator")
        .add_edge("translator", END)
        .compile()
        .unwrap();

    println!("\n  父图可视化输出:");
    println!("{}", parent.visualize_ascii());

    println!("\n┌─────────────────────────────────────────┐");
    println!("│     【第 3 部分】执行时的层级关系        │");
    println!("└─────────────────────────────────────────┘");

    println!("\n  执行流程图解：");
    println!("  ");
    println!("  ┌──────────┐                         ");
    println!("  │  START   │                         ");
    println!("  └────┬─────┘                         ");
    println!("       │                               ");
    println!("       ▼                               ");
    println!("  ┌──────────────────────────────────┐ ← 父图层");
    println!("  │     节点: translator             │   ");
    println!("  │  ┌────────────────────────────┐  │   ");
    println!("  │  │    【子图内部执行】        │  │ ← 子图层");
    println!("  │  │                            │  │   ");
    println!("  │  │  detect → translate       │  │   ");
    println!("  │  │  (各调用一次OpenAI)        │  │   ");
    println!("  │  └────────────────────────────┘  │   ");
    println!("  └──────────────────────────────────┘   ");
    println!("       │                               ");
    println!("       ▼                               ");
    println!("  ┌──────────┐                         ");
    println!("  │   END    │                         ");
    println!("  └──────────┘                         ");

    println!("\n【开始执行】...");
    let result = parent
        .invoke(AgentState::new("test".to_string()))
        .await
        .unwrap();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                     【最终结果】                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!(
        "\n  父图最终输出: {}",
        result.final_state.output.clone().unwrap_or_default()
    );
    println!("  子图执行步骤: {}", result.steps.len());
    println!("\n  层级总结:");
    println!("    - 父图只有 1 个节点: translator（子图）");
    println!("    - 子图有 2 个节点: detect + translate");
    println!("    - 子图内部的 2 次调用都发生在父图的 1 个节点内");
}

/// 测试 2：Persistence（持久化）+ OpenAI 真实调用
///
/// 场景：创建工作流，保存到 JSON 文件，然后从文件加载
///
/// 流程：
/// 1. 创建单节点图（调用 OpenAI 分析 Rust）
/// 2. 保存图定义到临时文件
/// 3. 从文件加载图定义
/// 4. 执行图，真实调用 OpenAI
///
/// 打印输出：
/// - 保存的 JSON 内容
/// - 加载后的图信息
/// - OpenAI 调用结果
#[tokio::test]
#[ignore = "需要真实 OpenAI API"]
async fn test_persistence_with_openai() {
    println!("\n========================================");
    println!("测试 2: Persistence（持久化）+ OpenAI 真实调用");
    println!("========================================\n");

    println!("【步骤 1】创建分析工作流...");

    let graph = GraphBuilder::<AgentState>::new()
        .add_async_node("analyze", |_state: &AgentState| async move {
            println!("  → analyze 节点开始执行");
            let result = call_openai("用一句话解释 Rust 编程语言", "简洁专业").await;
            println!("  → analyze 节点完成");

            let mut s = AgentState::new("Rust".to_string());
            s.set_output(result);
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "analyze")
        .add_edge("analyze", END)
        .compile()
        .unwrap();

    println!("图结构:");
    println!("{}", graph.visualize_ascii());

    println!("\n【步骤 2】保存图定义到文件...");

    let definition = graph.to_definition();
    let temp = std::env::temp_dir().join("langgraph_persistence_test.json");
    let json = serde_json::to_string_pretty(&definition).unwrap();
    std::fs::write(&temp, json).unwrap();
    println!("保存路径: {}", temp.display());

    println!("\n【步骤 3】查看保存的 JSON 内容...");
    let content = std::fs::read_to_string(&temp).unwrap();
    println!("{}", content);

    println!("\n【步骤 4】验证 JSON 可正确解析...");
    let loaded: langchainrust::langgraph::GraphDefinition = serde_json::from_str(&content).unwrap();
    println!("解析成功!");
    println!("  入口节点: {}", loaded.entry_point);
    println!("  节点数量: {}", loaded.nodes.len());
    println!("  边数量: {}", loaded.edges.len());
    println!("  递归限制: {}", loaded.recursion_limit);

    println!("\n【步骤 5】执行工作流（真实调用 OpenAI）...");
    let result = graph
        .invoke(AgentState::new("test".to_string()))
        .await
        .unwrap();

    println!("\n========================================");
    println!("【最终结果】");
    println!("========================================");
    println!(
        "OpenAI 分析结果: {}",
        result.final_state.output.clone().unwrap_or_default()
    );

    std::fs::remove_file(&temp).ok();
    println!("\n已清理临时文件: {}", temp.display());
}

/// 测试 3：多节点顺序执行 + OpenAI
///
/// 场景：三个节点按顺序执行，每个节点独立调用 OpenAI
///
/// 流程：
/// 1. step1: 用一个词描述 Rust
/// 2. step2: 基于 step1 的结果，问 Rust 的特点
/// 3. step3: 综合前两步结果，给出最终总结
///
/// 状态传递：
/// - step1 → step2: 通过 messages 传递
/// - step2 → step3: 通过 messages 传递
///
/// 打印输出：
/// - 每个步骤的 OpenAI 调用结果
/// - 最终综合结果
#[tokio::test]
#[ignore = "需要真实 OpenAI API"]
async fn test_multi_node_with_openai() {
    println!("\n========================================");
    println!("测试 3: 多节点顺序执行 + OpenAI");
    println!("========================================\n");

    println!("【创建图】三步顺序执行...");

    let graph = GraphBuilder::<AgentState>::new()
        .add_async_node("step1", |_state: &AgentState| async move {
            println!("\n  ▶ 步骤 1 开始: 用一个词描述 Rust");
            let r = call_openai("用一个词描述 Rust 语言", "只回答一个词").await;
            println!("  ✓ 步骤 1 完成: {}", r);

            let mut s = AgentState::new(r.clone());
            s.add_message(MessageEntry::ai(r));
            Ok(StateUpdate::full(s))
        })
        .add_async_node("step2", |state: &AgentState| {
            let prev = state
                .messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            async move {
                println!("\n  ▶ 步骤 2 开始: 分析 {} 的特点", prev);
                let r = call_openai(&format!("{} 语言的主要特点是什么？", prev), "简洁回答").await;
                println!("  ✓ 步骤 2 完成: {}", r);

                let mut s = AgentState::new(prev);
                s.add_message(MessageEntry::ai(r));
                Ok(StateUpdate::full(s))
            }
        })
        .add_async_node("step3", |state: &AgentState| {
            let msgs = state.messages.clone();
            async move {
                let all = msgs
                    .iter()
                    .filter(|m| m.role == langchainrust::MessageRole::AI)
                    .map(|m| m.content.clone())
                    .collect::<Vec<_>>()
                    .join(" → ");

                println!("\n  ▶ 步骤 3 开始: 综合总结");
                println!("    输入链: {}", all);
                let r = call_openai(
                    &format!("综合以下信息给出一句话总结: {}", all),
                    "一句话总结",
                )
                .await;
                println!("  ✓ 步骤 3 完成: {}", r);

                let mut s = AgentState::new("done".to_string());
                s.set_output(r);
                Ok(StateUpdate::full(s))
            }
        })
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .add_edge("step2", "step3")
        .add_edge("step3", END)
        .compile()
        .unwrap();

    println!("\n图结构:");
    println!("{}", graph.visualize_ascii());

    println!("\n【执行】开始顺序执行三步...");
    let result = graph
        .invoke(AgentState::new("start".to_string()))
        .await
        .unwrap();

    println!("\n========================================");
    println!("【最终结果】");
    println!("========================================");
    println!(
        "综合总结: {}",
        result.final_state.output.clone().unwrap_or_default()
    );
    println!("执行路径: start → step1 → step2 → step3 → end");
    println!("总步骤数: {}", result.steps.len());
}

/// 测试 4：三层嵌套子图 + OpenAI
///
/// 场景：演示子图的多层嵌套能力
///
/// 结构：
/// - 内层子图 (inner): 调用 OpenAI 获取一个编程语言名
/// - 中层子图 (middle): 包含内层子图 + 分析语言特点
/// - 外层图 (outer): 包含中层子图 + 最终总结
///
/// 执行流程：
/// outer.start → middle → inner → inner_task → middle_task → outer_task → end
///
/// 打印输出：
/// - 每层的执行状态
/// - 嵌套层级信息
/// - 最终综合结果
#[tokio::test]
#[ignore = "需要真实 OpenAI API"]
async fn test_nested_subgraph_with_openai() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         测试 4: 三层嵌套子图 + OpenAI 真实调用              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\n┌─────────────────────────────────────────┐");
    println!("│        【层级结构预览】                 │");
    println!("└─────────────────────────────────────────┘");

    println!("\n  三层嵌套结构图解：");
    println!("  ");
    println!("  ╔═════════════════════════════════════════════════════╗");
    println!("  ║           【外层图: outer】顶层                    ║");
    println!("  ║                                                     ║");
    println!("  ║   START → middle(子图) → outer_task → END         ║");
    println!("  ║                                                     ║");
    println!("  ║   ┌─────────────────────────────────────────────┐ ║");
    println!("  ║   │       【中层子图: middle】第2层             │ ║");
    println!("  ║   │                                             │ ║");
    println!("  ║   │   START → inner(子图) → middle_task → END │ ║");
    println!("  ║   │                                             │ ║");
    println!("  ║   │   ┌───────────────────────────────────┐   │ ║");
    println!("  ║   │   │     【内层子图: inner】第3层     │   │ ║");
    println!("  ║   │   │                                   │   │ ║");
    println!("  ║   │   │   START → inner_task → END      │   │ ║");
    println!("  ║   │   │                                   │   │ ║");
    println!("  ║   │   │   inner_task: 获取语言名        │   │ ║");
    println!("  ║   │   │   (调用OpenAI)                  │   │ ║");
    println!("  ║   │   └───────────────────────────────────┘   │ ║");
    println!("  ║   │                                             │ ║");
    println!("  ║   │   middle_task: 分析语言特点              │ ║");
    println!("  ║   │   (调用OpenAI)                            │ ║");
    println!("  ║   └─────────────────────────────────────────────┘ ║");
    println!("  ║                                                     ║");
    println!("  ║   outer_task: 最终总结 (调用OpenAI)               ║");
    println!("  ╚═════════════════════════════════════════════════════╝");

    println!("\n┌─────────────────────────────────────────┐");
    println!("│      【第 1 部分】创建内层子图           │");
    println!("└─────────────────────────────────────────┘");
    println!("  层级: 第 3 层（最内层）");
    println!("  功能: 获取一个编程语言名");

    let inner = GraphBuilder::<AgentState>::new()
        .add_async_node("inner_task", |_state: &AgentState| async move {
            println!("\n      >>> [第3层-内层] inner_task 执行 <<<");
            let r = call_openai("说出一个编程语言的名字", "只回答一个词").await;
            println!("      >>> [第3层-内层] 完成: {} <<<", r);

            let mut s = AgentState::new(r.clone());
            s.set_output(r);
            Ok(StateUpdate::full(s))
        })
        .add_edge(START, "inner_task")
        .add_edge("inner_task", END)
        .compile()
        .unwrap();

    println!("\n  内层子图结构:");
    println!("{}", inner.visualize_ascii());

    println!("\n┌─────────────────────────────────────────┐");
    println!("│      【第 2 部分】创建中层子图           │");
    println!("└─────────────────────────────────────────┘");
    println!("  层级: 第 2 层");
    println!("  功能: 嵌入内层子图 + 分析语言特点");

    let middle = GraphBuilder::<AgentState>::new()
        .add_subgraph_same_state("inner", inner)
        .add_async_node("middle_task", |state: &AgentState| {
            let inner_result = state.output.clone().unwrap_or_default();
            async move {
                println!("\n    >>> [第2层-中层] middle_task 执行 <<<");
                println!("    >>> [第2层-中层] 输入来自内层: {} <<<", inner_result);
                let r =
                    call_openai(&format!("{} 编程语言的主要特点", inner_result), "简洁回答").await;
                println!("    >>> [第2层-中层] 完成 <<<");

                let mut s = AgentState::new(inner_result);
                s.set_output(r);
                Ok(StateUpdate::full(s))
            }
        })
        .add_edge(START, "inner")
        .add_edge("inner", "middle_task")
        .add_edge("middle_task", END)
        .compile()
        .unwrap();

    println!("\n  中层子图结构:");
    println!("{}", middle.visualize_ascii());
    println!("  注意: 'inner' 节点是嵌入的内层子图");

    println!("\n┌─────────────────────────────────────────┐");
    println!("│      【第 3 部分】创建外层图             │");
    println!("└─────────────────────────────────────────┘");
    println!("  层级: 第 1 层（最外层）");
    println!("  功能: 嵌入中层子图 + 最终总结");

    let outer = GraphBuilder::<AgentState>::new()
        .add_subgraph_same_state("middle", middle)
        .add_async_node("outer_task", |state: &AgentState| {
            let mid = state.output.clone().unwrap_or_default();
            async move {
                println!("\n  >>> [第1层-外层] outer_task 执行 <<<");
                println!("  >>> [第1层-外层] 输入来自中层 <<<");
                let r = call_openai(&format!("用一句话总结: {}", mid), "一句话").await;
                println!("  >>> [第1层-外层] 完成 <<<");

                let mut s = AgentState::new("done".to_string());
                s.set_output(r);
                Ok(StateUpdate::full(s))
            }
        })
        .add_edge(START, "middle")
        .add_edge("middle", "outer_task")
        .add_edge("outer_task", END)
        .compile()
        .unwrap();

    println!("\n  外层图结构:");
    println!("{}", outer.visualize_ascii());
    println!("  注意: 'middle' 节点是嵌入的中层子图");

    println!("\n┌─────────────────────────────────────────┐");
    println!("│        【执行时的层级穿透】             │");
    println!("└─────────────────────────────────────────┘");

    println!("\n  执行顺序（从外到内，再从内到外）：");
    println!("  ");
    println!("  ① 外层 START → 进入 middle 子图");
    println!("      ↓");
    println!("  ② 中层 START → 进入 inner 子图");
    println!("      ↓");
    println!("  ③ 内层 START → inner_task → END");
    println!("      ↓");
    println!("  ④ 中层继续 → middle_task → END");
    println!("      ↓");
    println!("  ⑤ 外层继续 → outer_task → END");

    println!("\n【开始执行】...");
    let result = outer
        .invoke(AgentState::new("start".to_string()))
        .await
        .unwrap();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                     【最终结果】                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!(
        "\n  三层嵌套最终输出: {}",
        result.final_state.output.clone().unwrap_or_default()
    );
    println!("\n  层级关系总结:");
    println!("    ┌─────────────────────────────────────┐");
    println!("    │ 外层图 (outer)                      │");
    println!("    │   ├─ 节点: middle (中层子图)       │");
    println!("    │   │   ├─ 节点: inner (内层子图)   │");
    println!("    │   │   │   └─ 节点: inner_task    │");
    println!("    │   │   └─ 节点: middle_task        │");
    println!("    │   └─ 节点: outer_task              │");
    println!("    └─────────────────────────────────────┘");
    println!("\n  嵌套深度: 3 层");
    println!("  总 OpenAI 调用: 3 次（每层各 1 次）");
}
