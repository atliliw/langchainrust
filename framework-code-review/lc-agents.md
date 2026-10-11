# lc-agents

> 硬性约束声明:本次审查只读不改任何代码,唯一落盘为本报告。每条发现均带 `文件:行号` + 触发条件 + 后果。重点核查了任务点名的 4 块 executor 风险区,并如实给出"已解决/仍存在"两类结论。

## 1. 概览

`crates/lc-agents/src/` 是"Agent 执行协调层":`BaseAgent`(规划)+ `AgentExecutor`(执行循环)分离。核心在 `executor/` 子目录:决策循环(`agent_loop`)、预算门禁(`budget`)、统一工具执行闸口(`tool_gate`)、上下文压缩(`compaction`)、持久化恢复(`resume`/`store`)、流式(`stream`);`types.rs` 定义 `AgentAction/AgentFinish/AgentStep/AgentOutput`。外围还有 ReAct / function-calling 两种规划器、`orchestrator/`(supervisor/sequential/review/fan-out-fan-in)、`react/`、`streaming/`、`hooks/`(审批/内容过滤/限流/日志/注入)、`crag/`、`adaptive_rag/`、`deep_research/`、`plan_execute/`、`handoffs/` 等。

共 81 个 `.rs`。本次对核心执行路径(executor 全部 + react + supervisor + streaming)逐行通读,外围子模块列全并给出模块清单;通读集合见 §2 标记。

## 2. 遍历文件清单(81 个,全部列出)

**✦ = 已逐行通读**(覆盖全部既定风险区与其直接关联件):

| 文件 | 程度 |
|---|---|
| `src/lib.rs` | ✦ |
| `src/types.rs` | ✦ |
| `src/executor/mod.rs` | ✦ |
| `src/executor/engine.rs` | ✦ |
| `src/executor/agent_loop.rs` | ✦ |
| `src/executor/budget.rs` | ✦ |
| `src/executor/tool_gate.rs` | ✦ |
| `src/executor/tools.rs` | ✦ |
| `src/executor/stream.rs` | ✦ |
| `src/executor/compaction.rs` | ✦ |
| `src/react/agent.rs` | ✦ |
| `src/orchestrator/supervisor.rs` | ✦ |
| `src/streaming/mod.rs` | ✦ |
| `src/executor/hooks.rs` / `semantic_memory.rs` / `file_memory_tool.rs` / `tests.rs` | 列全未逐行展开 |
| `src/react/{mod,parser,prompt}.rs` | 未展开 |
| `src/function_calling/{mod,agent}.rs` | 未展开 |
| `src/adapter.rs, builder.rs, cache.rs, policy.rs, retry.rs, task.rs, approval.rs, graph_approval.rs, resume.rs, metrics.rs, memory_extractor.rs, structured.rs` | 未展开 |
| `src/hooks/{mod,approval,content_filter,logging,injection,rate_limit}.rs` | 未展开 |
| `src/orchestrator/{mod,impls,sequential,review,task_adapter,fan_out_fan_in,tests}.rs` | 未展开 |
| `src/crag/{mod,rewriter,grader,types,graph,error,tests}.rs` | 未展开 |
| `src/adaptive_rag/{mod,prompts,types,tests}.rs` | 未展开 |
| `src/deep_research/{mod,planner,searcher,synthesizer,types,error,tests}.rs` | 未展开 |
| `src/plan_execute/{mod,plan,planner,agent}.rs` | 未展开 |
| `src/handoffs/{mod,handoff,manager}.rs` | 未展开 |
| `src/streaming/{sse,state,tool_call_stream}.rs` | 未展开 |

## 3. bug / 高风险点

### 3.1 预判风险区的复核结论(任务点名 4 项)

**(1) cost 增量并发污染 —— 已解决。** `agent_loop.rs:122-129`:cost 门禁读取 `tracker.run_cost_usd(&scope)` 且 `scope = 本 run 的 root_run.id`,并非全局基线;`engine.rs`(`build_plan_config`)把 `RUN_META_COST_SCOPE` 打进每轮配置。并发 run 共享同一 `Arc<CostTracker>` 时,**增量不会互相串账**。触发条件=两个 run 并发且共用 tracker;结果=各自按 root_run.id 记账,预算独立。此项非现行 bug。

**(2) budget 门禁边界语义统一 —— 已解决。** `budget.rs:173-179`(tool,`>= limit` 允许恰 `limit` 个)、`budget.rs:122`(token,`> limit` 允许恰 `limit` 个 token)、`budget.rs:148`(cost,`> limit`)、`budget.rs:89`(iteration,`>= limit`)。四类门禁统一为"**允许恰好打满 limit,超过才停**"语义,`budget.rs` 顶部注释与 M12/G5 测试都对这一边界做了显式断言。此项非现行 bug。

**(3) 工具调用计数只计实际执行 —— 已解决。** 单条 `agent_loop.rs:218-220` 与并行 `agent_loop.rs:247-251` 都用 `is_non_execution_observation`(`tools.rs:75-80` 的 DENIED / Tool not found / Skipped by hook / BLOCKED by Rule of Two 前缀)过滤,**只有真执行才 `tool_calls += 1`**;`server_executed` 视为已执行(`tool_gate.rs:226-228` 直接返回参数原样)。不再预增计数,`limit` 不再被拒绝/未命中的调用提前吃掉。此项非现行 bug。

**(4) server-tool 被本地二次执行 —— 已解决。** `tool_gate.rs:226-228`:`if action.server_executed { return Ok(ToolOutcome::Done(server_executed_observation(action))) }` —— 本地执行链整体跳过,参数原样回观测(`tool_gate.rs:549`),`types.rs:34-37` 的 `server_executed` 标志由 M7 从 `ToolCall` 传播。此项非现行 bug。

> 结论:任务点名的 executor 4 项风险在现行代码中**全部已落地解决**,不是现行 bug。这是本次审查最重要的如实结论——不能因为"任务点名"就假装它们还存在。

### 3.2 现行真实问题(1 个 MEDIUM)

- **并行 batch 可超 `max_tool_calls` 限值。**
  - `文件:行号`:`budget.rs:173-179`(`tool_calls >= limit`)+ `agent_loop.rs:237-241`(batch 前检查一次)+ `agent_loop.rs:243`(一次性整批执行)+ `agent_loop.rs:247-251`(整批执行完之后再一次性 `tool_calls += executed`)。
  - 触发条件:`limit - 1 <= tool_calls < limit` 时,LLM 规划出**一批含 N 个**真执行工具(N>1)的 `AgentOutput::Actions`。门禁只在 round 起点检查一次,批内不再复查。
  - 后果:单批能把 `tool_calls` 从 `limit-1` 打到 `limit-1+N`,**超额幅度可达批大小 N-1**,`max_tool_calls` 只在批边界生效而非恰好打满。对于"严格不超预算"的硬约束不是精确满足(单工具 Action 路径因一次 `+1` + round 起点的 `>=` 复查,能停在恰 `limit`;并行路径做不到)。这是文档化语句"Cumulative tool-call cap (parallel calls included; stops when exceeded)"在"批量内"欠严谨的部分。

### 3.3 已确认修复(审计期处理,证据收纳,非现行 bug)

- `react/agent.rs:153-189`(H-A5):`PARSE_ERROR_TOOL` 伪工具修复 ReAct 解析失败,连续两次失败才 hard-fail,防死循环。
- `tools.rs:75-80`:非执行观测判定与 §3.1-(3) 配套。
- `tool_gate.rs:442-520`:并行走 `JoinSet` + 共享 `concurrency_sem` 限流;框架级硬错误 `set.abort_all()` 返回 `AgentError::BatchAborted { partial }` 保留已完成的兄弟工具观测(不静默丢弃)。
- `agent_loop.rs:140-154`:伪工具不执行、不回执门禁。
- `engine.rs cache_key:690` 对输入键排序保证确定性(A10)。
- `stream.rs:743` 通过 watch 完成 Drop 取消,`stream.rs:797` 先发 metrics 再发终态。

## 4. 不合理 / 可改进

1. **SF 门禁在批内不复查**(承接 §3.2):建议在 `execute_tools_parallel` 内按完成顺序用 `JoinSet::join_next` 边完成边复查 `budget_tool_gate`,超限即 `abort_all` —— 既满足"恰好打满"又能保留已完成的 `partial` 观测。

2. **`engine.rs` resume 的 `tool_calls_consumed` 用 `metrics.tool_calls + 1`**(`agent_loop.rs:185`):写死 +1 假定恢复的是单工具。若并行批(多工具)挂起恢复,恢复计数会与实际执行数不一致导致 budget 基准偏差。建议把入批时待执行工具数记入 checkpoint。此点因未逐行读 `resume.rs`/store 落盘格式,标注为"待核对"(不是确认 bug)。

3. **cost 门禁只测 LLM 侧开销**:`agent_loop.rs:111/122-129` 的 token/cost 都从 LLM `token_usage` 来,工具自身(如网页抓取、外部 API 计时)不计成本。若 `max_cost_usd` 想覆盖工具副作用开销,当前测量面只在 LLM。属设计取舍,建议文档明示。

4. **`last_token_usage` 用 `std::sync::Mutex`**(`react/agent.rs:49,235`):`.lock()` 阻塞在 async 上下文内,极端低概率卡调度;换 `tokio::sync::Mutex` 或原子更契合异步。

5. **`structured.rs` 为私有 `mod`**(`lib.rs`):结构化输出实现被封装不可用,若用户需 external-facing structured-agent 需 `pub`。可能是"死接口/未露面能力",建议按需开放。

## 5. 可加的新功能(对标 Claude/OpenAI Agents SDK)

1. **Human-in-the-loop 同步审批节点**(已有 `hooks/approval` + `approval.rs` + `graph_approval.rs` 骨架):可内联为 Agent 循环原语 `await_decision`,对标 Claude Agent SDK `HumanLoopOptions`。
2. **`run_cost_usd` 分层聚合**:对标 OpenAI Agents SDK 每条 `Run` 的 cost breakdown,对外暴露按 run/tool 维度的明细,便于预算审计。
3. **多 Agent 横向编排** 更强:现 supervisor 是单层 worker;可对标 LangGraph Supervisor 的递归子 agent 与 `handoffs/` 天然衔接成"Agent 交接环"语义。
4. **流式 + 审批组合**:把 `streaming/` events(plan/action/observation)与 `tool_gate` 的 pending-approval 对接成 SSE 可交互审批面板。
5. **`on_max_iterations` 支持 Placeholder 之外的"转交给另一个 agent/do-come-back"策略**,对标 Claude Agent SDK `must_interrupt_at_turn`。

## 6. 对标主流

- **OpenAI Agents SDK**:executor 的 `ToolGate`(hooks→approval→policy→Rule of Two→timeout→spotlight)≈ Agents SDK 的 `Runner` + `tool_use` callback 链;`run_cost_usd` per-scope 记账是对其 `run` cost 的中文复刻。
- **Claude Agent SDK**:`server_executed` / 审批 / 交接环三处机制与本 crate 的 `tool_gate`/`approval`/`handoffs` 高度同构;区别在 Claude SDK 是"客户端原语",本 crate 把决策循环完全暴露为可替换 `BaseAgent`。
- **LangChain AgentExecutor**:本 crate 是"react_agent + max_iterations + verbose"的现代再实现,额外有 budget 门禁、compaction、resume、独立 stream 线程。整体比 LangChain 老牌 executor 更工程化。

## 7. 一句总评

lc-agents 的 executor 把任务点名的 4 项高风险(并发 cost 串账 / budget 边界 / 计数只计执行 / server-tool 二次执行)**全部修到位**,是硬核、审计痕迹密集的实现;唯一的现行缺口是**并行批对 `max_tool_calls` 只做批边界而非批内精确控制**(MEDIUM),外加上几处配置/记账的可打磨点。

## 8. 补充审阅:未展开文件通读

> 本次补齐上一轮标为「未展开」的全部非 tests 文件,**逐行通读**,只读不改。tests.rs(`orchestrator/tests.rs` / `crag/tests.rs` / `adaptive_rag/tests.rs` / `deep_research/tests.rs`)按要求略过。共实读 **57 个文件**;`executor/{budget,tool_gate,agent_loop,engine,stream,compaction,tools}`、`react/agent.rs`、`orchestrator/supervisor.rs`、`streaming/mod.rs` 属 §2 已✦通读,本次未重复。结论:半成品子模块(crag / adaptive_rag / deep_research / plan_execute / handoffs / streaming)确有**未接线死代码**与**文档-行为不一致**,但均非崩溃级;两处真实 bug 见 §8.2。

### 8.1 本次实读清单(57 个,全部逐行)

**配置/协调顶层(12)** `adapter.rs` `builder.rs` `cache.rs` `policy.rs` `retry.rs` `task.rs` `approval.rs` `graph_approval.rs` `resume.rs` `metrics.rs` `memory_extractor.rs` `structured.rs`
**hooks(6)** `hooks/mod.rs` `hooks/approval.rs` `hooks/content_filter.rs` `hooks/logging.rs` `hooks/injection.rs` `hooks/rate_limit.rs`
**orchestrator 非 supervisor/tests(6)** `orchestrator/mod.rs` `orchestrator/impls.rs` `orchestrator/sequential.rs` `orchestrator/review.rs` `orchestrator/task_adapter.rs` `orchestrator/fan_out_fan_in.rs`
**react(3)** `react/mod.rs` `react/parser.rs` `react/prompt.rs`
**function_calling(2)** `function_calling/mod.rs` `function_calling/agent.rs`
**crag 非 tests(6)** `crag/mod.rs` `crag/rewriter.rs` `crag/grader.rs` `crag/types.rs` `crag/graph.rs` `crag/error.rs`
**adaptive_rag 非 tests(3)** `adaptive_rag/mod.rs` `adaptive_rag/prompts.rs` `adaptive_rag/types.rs`
**deep_research 非 tests(6)** `deep_research/mod.rs` `deep_research/planner.rs` `deep_research/searcher.rs` `deep_research/synthesizer.rs` `deep_research/types.rs` `deep_research/error.rs`
**plan_execute(4)** `plan_execute/mod.rs` `plan_execute/plan.rs` `plan_execute/planner.rs` `plan_execute/agent.rs`
**handoffs(3)** `handoffs/mod.rs` `handoffs/handoff.rs` `handoffs/manager.rs`
**streaming 非 mod(3)** `streaming/sse.rs` `streaming/state.rs` `streaming/tool_call_stream.rs`
**executor 外围(3)** `executor/hooks.rs` `executor/semantic_memory.rs` `executor/file_memory_tool.rs`

### 8.2 真实 bug / 高风险点(2 处 MEDIUM)

**8.2.1 `ContentFilterHook::replace_sensitive` 字节索引错位 → 越界 panic / 脱敏错乱。**
- `文件:行号`:`hooks/content_filter.rs:66-93`(`lower = text.to_lowercase()` 只在循环外算一次 L70;按 `lower` 算出的字节偏移打到**已被占位符改长**的 `result` 上切片 L75-79;`start` 用 `placeholder.len()` 前进却去搜 `lower`,占位符长度 ≠ 词长或一处文本多词命中即失步 L72-81)。
- 触发条件:一段文本里**同一个敏感词出现多次**,或敏感词被替换后占位符与原词字节长度不同;或含非 ASCII(`to_lowercase()` 改变字节数)。
- 后果:`&result[..actual_pos]` / `&result[end..]` 的 `actual_pos/end` 是基于原始 `lower` 计算的,与当前 `result` 字节流不对齐,可能落在 UTF-8 字符中间 → `byte index ... is not a char boundary` **panic**(流式脱敏 `on_stream_chunk` 里直接炸掉整条流);即便不 panic,多命中场景也会把占位符嵌到错误位置(`result[..actual_pos]` 取到的是改写后字符串,而非原字符串)。建议改为逐字节替换或先 `find` 再按 `result` 自身字节校验边界。此项为现行 bug。

**8.2.2 `AgentBuilder::max_iterations` 是静默死配置(用户以为生效,实际被丢弃)。**
- `文件:行号`:`builder.rs:51`(字段)、`builder.rs:107-115`(`max_iterations()` 存取+clamp 到 [1,100])、`builder.rs:145-147`(`get_max_iterations()`);但 `build()` `builder.rs:122-132` 与 `build_as_agent()` `builder.rs:139-142` 把 `llm/tools/system_prompt` 传给 `FunctionCallingAgent::from_arc`,**从不传 `self.max_iterations`**(`FunctionCallingAgent` 该构造路径无此参数)。
- 触发条件:用户 `.max_iterations(50)` 后 `.build()`。
- 后果:该限制**完全不生效**,迭代次数的默认/上限只能靠 `AgentExecutor` 侧 budget(`budget.rs:89` iteration 门禁)兜底;API 面存在一个有文档、有 getter 却无效果的误导性配置。属静默失效,建议要么接入 agent/executor 的 iteration 上限,要么删掉并明确标注由 executor budget 负责。此项为现行 bug(配置层)。

### 8.3 死代码 / 未接线能力(半成品子模块重点核实)

1. **crag 的 `HallucinationCheckError` 变体不可达。** `graph.rs:362-414` `hallucination_check` 返回 `Result<(), CRAGError>` 但**全程不构造 `Err`**——失败路径只 `state.grounded=false` 后 `Ok(())`。`error.rs:37` 该变体成为死代码;任何调用方按 `Err` 处理幻觉检查失败的分支实际永不触发。
2. **crag web 兜底用「原始 query」而非「改写 query」(文档-行为不一致)。** `graph.rs:309` `correct()` 调 web 工具时传 `state.current_query.clone()`,而 `current_query` 直到 `graph.rs:317-319` 才被替换成 alternative → 检索用的是原查询;`mod.rs:86` 文档却承诺「with the rewritten query」。属述而不作。
3. **crag `adapt` 冗余条件 + 吞错。** `mod.rs:286-290` `let source_docs = if filtered.is_empty() { Vec::new() } else { filtered }` 两分支同值,纯冗余;`mod.rs:304` `let _ = graph.hallucination_check(...)` 静默丢弃错误(`hallucination_check` 固然不 Err,但 `?` 语义被注释成吞掉)。
4. **`StreamingFunctionCallingAgent` 名不符实:只流式吐字,从不执行工具。** `tool_call_stream.rs:68-110` 全流程只有 `stream_chat` → 逐 chunk 发 `Text` → 收尾发 `FinalAnswer{full}`,**不发任何 `ToolCall/ToolStart/ToolEnd`**,也不具备工具循环/工具注册。文档 `tool_call_stream.rs:18-21` 声称「Tool-call state is exposed via AgentStreamEvent::ToolCall」——该状态机(`state.rs` 六态)与 `sse.rs:143-193` 精心序列化的 `tool_call` 事件,**没有任何 producer** 实际产生,属「仪表全装、发动机未接」的未接线能力。
5. **`structured.rs` 为私有 `mod`。** `structured.rs`(§4.5 复核成立,本次逐行确认)`StructuredChatResult`/`chat_structured` 未 `pub` 出 crate,外部无法使用该结构化输出能力。

### 8.4 不合理 / 可改进

1. **`review.rs:96-104` `parse_review_verdict` 反馈分支前后大小写不一致:** 判定用 `upper.starts_with("FAIL")`(大小写不敏感),剔前缀却只用 `trim_start_matches("FAIL")`(仅大写)→ 小写 `fail: …` 反馈会保留 `fail` 前缀。低危,建议统一。
2. **`adaptive_rag` 多路查询不完全剥离编号前缀 + 参数不钳制下限:** `adaptive_rag/mod.rs:317-322` `generate_queries` 原样返回每个非空行(不剥 `1. 2.` 之类前缀,`crag` 的 `parse_alternatives` 会剥),多路检索可能拿「`1. xxx`」作检索词;且 `with_retrieve_k` / `with_multi_query_count` 不 `<rel=1` 钳制(允许 0),与 crag `.max(1)` 风格不一致。低危。
3. **`tool_call_stream.rs:96-109` 出错仍补发 `FinalAnswer`:** `stream_chat` 中途 `Err` 时已发 `Error` 事件,但 break 后仍会把**残缺 `full`** 当最终答案发出(`FinalAnswer{content: full}`)。前端若同时消费 `error` 与 `final_answer` 会拿到半截答案;建议错误路径不发 FinalAnswer。
4. **`HandoffManager` 链守卫仅 `run()` 作用域内生效:** `manager.rs:110-146` 用 `tokio::task_local!` + `try_with(...).unwrap_or(Ok(()))`,直接(out-of-band)调用 `execute_handoff`(不经 `run()`)会跳过环/深度守卫——文档已注明(设计取舍),但外部 API 暴露了「绕开死循环保护」的旁路,建议注释明示或把守卫上移到工具入口。非现行 bug。
5. **`function_calling/agent.rs:36` `last_token_usage` 用 `std::sync::Mutex`**,阻塞式 `.lock()` 出现在 async 上下文(与 §4.4 同型,本 agent 亦有);低概率卡调度,建议 `tokio::sync` 或原子。

### 8.5 已核对无问题(诚实清单)

以下文件**逐行读完,未发现 bug**,仅列出(均已计入 §8.1):`state.rs`、`sse.rs`(帧编号/重连/心跳/断开取消均有测试,质量高)、`semantic_memory.rs`(detached 提取不阻塞主循环,设计正确)、`file_memory_tool.rs`(默认不挂载,Rule of Two 符合)、`handoffs`(环+深度守卫经 task-local 隔离,并发正确)、`executor/hooks.rs`、`cache.rs`、`policy.rs`、`retry.rs`、`task.rs`、`approval.rs`、`graph_approval.rs`、`resume.rs`、`metrics.rs`、`memory_extractor.rs`、`react/*`、`function_calling/agent.rs`、`orchestrator/{mod,impls,sequential,task_adapter,fan_out_fan_in}`、`deep_research/*`(searcher 有 8 并发信号量上限,健壮)、`plan_execute/*`。

### 8.6 小结

本次把 §2 中「未展开」的 57 个文件逐行读完。真实 bug **2 处**(`content_filter` UTF-8 字节切片 panic 风险、`builder.max_iterations` 静默死配置),均为 MEDIUM、非崩溃级;半成品子模块(crag / streaming / adaptive_rag)存在**4 类未接线死代码/文档不一致**(§8.3),符合「框架多层但部分能力未贯通」的整体画像。整体实现仍属工程化、审计密集,新增发现不推翻 §7 总评。