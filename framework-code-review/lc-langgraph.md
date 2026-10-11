# lc-langgraph

> 硬性约束声明:本次审查只读不改任何代码,唯一落盘为本报告。每条发现均带 `文件:行号` + 触发条件 + 后果。重点核查了任务点名的 6 项风险。

## 1. 概览

`crates/lc-langgraph/src/` 是图状编排层(对标 LangChain 的 LangGraph):`state.rs`(StateSchema/StateUpdate + field-level reducer)、`graph.rs`(StateGraph/EdgeBuilder)、`node.rs`/`edge.rs`(节点与边的种类/FanOut 汇聚)、`checkpointer.rs` / `checkpointer_*` / `persistence.rs`(checkpoint 持久化)、`compiled/`(编译后执行器:invoke / parallel / stream / validate / visualize / graph / types)。

核心执行模型:编译图在 `CompiledGraph` 内(`compiled/graph.rs`),提供 `invoke / invoke_with_execution / invoke_parallel / invoke_dynamic / stream` 等多入口;`checkpointer` 支持中断后 `resume`,用 `__checkpoint_id` 标注节点、在中断前落 checkpoint(M-15/H2 设计)。

共 21 个 `.rs`。本次对执行为主线逐行通读 17 个,`checkpointer_{postgres,redis,sqlite}` 与 `compiled/tests.rs` 未逐行(列全)。

## 2. 遍历文件清单(21 个,全部列出)

| 文件 | 审查程度 |
|---|---|
| `src/lib.rs` | 通读 |
| `src/state.rs` | 通读 — StateSchema / 字段级 reducer |
| `src/graph.rs` | 通读 — StateGraph/GraphBuilder |
| `src/node.rs` | 通读 — GraphNode / NodeConfig / NodeResult |
| `src/edge.rs` | 通读 — 边/FanOut/FanIn |
| `src/subgraph.rs` | 通读 — 子图节点 |
| `src/errors.rs` | 通读 — GraphError(DynamicInterrupt 等) |
| `src/persistence.rs` | 通读 |
| `src/checkpointer.rs` | 通读 — checkpoint 存根与恢复逻辑(1975 行) |
| `src/compiled/mod.rs` | 通读 |
| `src/compiled/graph.rs` | 通读 — CompiledGraph 主结构 |
| `src/compiled/invoke.rs` | 通读 — invoke / invoke_with_execution / rerun_interrupted / 中断编解码 |
| `src/compiled/parallel.rs` | 通读 — invoke_dynamic(含 FanOut/FanIn/field-union)/ invoke_parallel |
| `src/compiled/stream.rs` | 通读 — stream / execute_parallel_branches |
| `src/compiled/validate.rs` | 通读 — 编译期校验(DAG 无环 / 悬挂边) |
| `src/compiled/types.rs` | 通读 — ExecutionStep / ExecutionInterrupted 等类型 |
| `src/compiled/visualize.rs` | 通读 — Mermaid 可视化 |
| `src/checkpointer_sqlite.rs` / `checkpointer_postgres.rs` / `checkpointer_redis.rs` | 仅 glob/grep 定位,未逐行 |
| `src/compiled/tests.rs` | 未展开(测试) |

## 3. bug / 高风险点

### 3.1 HIGH — `invoke_dynamic` 的中断不落 checkpoint(非主路径中断)
- `文件:行号`:`compiled/parallel.rs:245-250`(在动态遍历循环终止处 `if self.interrupt_after.contains(&current_node) { return Err(GraphError::ExecutionInterrupted(...)) }`)。
- 触发条件:通过 `invoke_dynamic`(动态路由入口)执行,且某节点命中 `interrupt_after`(或 `interrupt_before`,`parallel.rs:223-225`)。
- 后果:与 `invoke / invoke_with_execution / invoke_parallel / invoke_from_node_with_count`(这些在中断前都 `save_checkpoint` + 打 `__checkpoint_id`)不同,`invoke_dynamic` **在中断前不写 checkpoint**。崩溃后 `resume` 拿不到"中断前已完成的节点状态",既无法从真实边界续跑,也会重新执行已产生副作用/已写状态的节点 → **双倍副作用 + 恢复丢状态**。这是本 crate 唯一一处"主路径 checkpoint-before-interrupt 与动态路径不一致"的真实缺口。
- 修复方向(建议,不动代码):在 `parallel.rs:245-250`(及 `223-225` 的 interrupt_before)返回中断前,复用 `invoke` 用的 `save_checkpoint(state, current_node, __checkpoint_id)`。

> 注:此处"非主路径"(from_node/fork/并行口径)是任务点名项,经代码核认确实存在;其余主路径系列均为"先 checkpoint 后中断",未发现同类缺口。

### 3.2 MEDIUM — 子图节点不携带父图 config / checkpoint 恢复值
- `文件:行号`:`subgraph.rs:91`(`async fn execute(&self, state, _config: Option<NodeConfig>)`,配置参数被 `_config` 丢弃),`subgraph.rs:96`(`self.subgraph.invoke(sub_input)` 每次用**全新**上下文执行)。
- 触发条件:父图内嵌子图,且父图带 checkpointer / 采样配置 / 取消令牌 / recursion_limit,或子图中途 `DynamicInterrupt` 需恢复值跨子图边界传导。
- 后果:
  1. 父图 config(取消/采样/回调/recursion_limit)不穿透到子图执行 → 子图在无取消认知、无回调的情况下裸跑。
  2. 子图内中断(`subgraph.rs:103` 已把 `DynamicInterrupt` 透传给父)后,恢复值 `value` 无法经 `invoke` 注入(用的是无参 `invoke`)→ 跨子图的 `resume_with_value` 实际不可达。
- 改良方向:让 `execute` 接收并转发 `NodeConfig`(含 checkpointer/resume value)给 `subgraph.invoke_with_execution / resume_with_value`。

### 3.3 MEDIUM — `stream()` 无条件 `tokio::spawn` + 并行 fan-out 无并发上限
- `文件:行号`:`compiled/stream.rs:33`(`tokio::spawn`)与 `compiled/stream.rs`(再结合 `compiled/parallel.rs` 的 `execute_parallel_branches` 用 `join_all`,无 `max_concurrency` 概念)。
- 触发条件 1:在非 tokio 运行时调用 `stream()`(图本身 invoke 路径不依赖 tokio,只有 stream 会 spawn)→ panic("there is no reactor running")。任务点名的"parallel invoke 无条件 tokio::spawn"经核认是 **stream 路径的 spawn**,invoke 路径未见 spawn。
- 后果:stream 必须运行在 tokio 上下文中,与 invoke 的运行时无关性不一致。
- 触发条件 2:动态 FanOut 分支数无上限,`join_all` 一次性拉起全部 → 大扇出时资源峰值高。建议加 `max_concurrency`(对标 lc-agents 的 `concurrency_sem`)。

### 3.4 复核为正的项(任务点名,已修/无 bug)

- **并行 fan-in「最后分支赢」—— 已解决**:`compiled/parallel.rs` 用 `merge_parallel_states` / `union_states` / `patch_field` 做**字段级合并**,不是整状态覆盖;`compiled/parallel.rs:276-278` 在 pre-fan-out 基态上 merge,不再"后写覆盖先写"。
- **crash-resume 重跑已折叠节点** 属 M-15 设计模型(中断在节点边界,先 checkpoint),仅在 §3.1 的 `invoke_dynamic` 缺口处失真。
- **递归上限**:`recursion_limit` 用 `>=` 前置检查(`recursion_count += 1` 前),防止无限环;`compiled/validate.rs` 编译期做无环/悬挂边校验。

## 4. 不合理 / 可改进

1. **`invoke_dynamic` 与其余入口的 checkpoint 语义分叉**(承接 §3.1):应在文档/设计层统一"任何入口中断前必落 checkpoint"的不变式,并补 `compiled/parallel.rs` 动态路径与主路径的差集测试。
2. **子图 config 不透传**(承接 §3.2):子图应共享父图 `RunnableConfig` / checkpointer,否则"取消/恢复"在跨层场景失效。建议在 `subgraph.rs` 暴露 `invoke_with_execution`。
3. **`compiled/stream.rs` 的 spawn** 建议改成"由调用方传 runtime handle"或显式返回错误,避免静默依赖 tokio。
4. **`compiled/visualize.rs` / `compiled/parallel.rs` 的命名**:`visualize` 集中在 `compiled/` 欠独立;可out到工具模块提升可测性(风格建议,非功能缺陷)。

## 5. 可加的新功能(对标 LangGraph/SDK)

1. **子图 checkpoint 贯通 + `resume_with_value` 跨层**:对标 LangGraph 的子图嵌套恢复,让父图把 checkpoint 与 human-value 注入子图。
2. **`interrupt()` + `interrupt_before/after` 的显式原语**:目前靠 `DynamicInterrupt`/`__checkpoint_id` 表示;可对标 LangGraph 的 `interrupt()` 内联 await 人类输入。
3. **time-travel / 分支重放(state history)** 已在 README 标注支持;建议补 `get_state_history / update_state` 公开 API,对标 LangGraph 的 `graph.state_history`。
4. **并行批量并发上限**(承接 §3.3)+ `JoinSet` 流式收集,对标 lc-agents `execute_many` 的 `concurrency_sem + abort_all + partial` 语义。
5. **teritorial 子图**:`subgraph.rs` 目前是"同步 invoke 型";可加"子图即节点"的外部续跑(suspend/resume)模式。

## 6. 对标主流

- **LangGraph**:checkpointer + interrupt/resume + 子图 + time-travel 是最贴近的对应;本 crate 的 `union_states/patch_field` 字段级并联合并甚至优于 LangGraph 默认"per-key reducer 覆盖"。
- **Claude Agent SDK / OpenAI Agents SDK**:本 crate 是"图即控制流",而这两个是"agent 即控制流";lc-langgraph 的价值在把 `interrupt/resume` 做成图级原语,可与 lc-agents 的 `handoffs/approval` 双向接线。
- **相较于 lc-chains**:ch 是线性/树状定向流,langgraph 是任意 DAG + 循环 + checkpoint,两者互补,可把 chains 作为子图节点复用。

## 7. 一句总评

lc-langgraph 的图执行核心(字段级并联合并、interrupt/resume 主路径、先 checkpoint 后中断、编译期校验)设计精良,但**动态路由入口 `invoke_dynamic` 在 `interrupt_after/interrupt_before` 时不落 checkpoint**(HIGH),子图 config/恢复值未透传(MEDIUM),stream 依赖 tokio 且并行 fan-out 无上限(MEDIUM)——这三处是相对主路径的一致性缺口,其余任务点名项已确认修复。