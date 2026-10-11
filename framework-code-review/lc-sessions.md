# lc-sessions 代码审查报告

> 审查方式:只读,通读 `crates/lc-sessions/src/**/*.rs` 全部功能源码;未改动任何代码。
> 结论基于代码实际情况,行号以当前工作区为准。

## 1 概览

lc-sessions 提供两套会话管理 API:

1. 传统可变异步 API(已废弃,`manager.rs` / `session.rs` / `store.rs` / `memory_store.rs` / `session_runnable.rs`):`SessionManager::chat` = 用户消息 → LLM → 追加回复 → 持久化,消息存在 `Session.messages` 中。
2. 事件溯源 API(推荐,`events/`):`EventSessionManager` + 追加式 `EventStore`(idempotent append、分支/fork、紧凑化 `Snapshot`)。历史是日志的投影(纯函数 `project`),崩溃安全,`AutoCompaction` 默认 1024 回合。

事件 API 设计质量高:幂等追加(`(session_id,branch,id)` 键去重)、分支隔离、快照紧凑化、按会话分片锁、`C7` 把「读 next_id → 追加用户消息 → LLM → 追加回复 → 紧凑化」整段用会话锁串行。`create_session` 与 `delete_session` 构成生命周期闭环。审查焦点(过期清理 / 并发 / 隔离 / 无界增长)逐一核验:并发已由分片锁 + 幂等追加双保险;隔离有 `memory_factory`(H-M1);无界增长只在「存储侧」仍未闭合(见下)。

## 2 遍历文件清单

实际通读的功能源文件(12 个):

- `lib.rs` — 新旧两套 API 的组织与导出。
- `session.rs` — `Session`(状态 Active/Archived/Deleted、recent_messages)。
- `store.rs` — `SessionError`、`SessionStore` trait。
- `manager.rs` — 废弃 `SessionManager`(striped 锁、`memory_factory`、`max_context_messages`)。
- `session_runnable.rs` — `SessionManagerRunnable`(Runnable 包装)。
- `memory_store.rs` — `MemorySessionStore`(更新缺键返回 NotFound、按用户过滤 Deleted)。
- `events/mod.rs` — 事件模块装配。
- `events/model.rs` — `SessionEvent` / `EventPayload`(异构枚举)/ `Turn`。
- `events/store.rs` — `EventStore` trait + `MemoryEventStore`(BTreeMap `(String,String,u64)` 幂等、分支隔离)。
- `events/manager.rs` — `EventSessionManager`(chat / fork / delete / auto_compaction / record_tool_use)。
- `events/replay.rs` — 日志投影 `project`、紧凑化 `compact_to_snapshot`、孤儿 Tool 结果校验。
- `events/checkpoint.rs` — `SessionCheckpoint` trait + `NoopCheckpoint`(占位)。

## 3 bug/高风险点(按严重度排序)

### 3.1 【中】事件日志「只增不删」,自动紧凑化不回收存储,事件日志真·无界增长

- 位置:`events/replay.rs:146-174`(`compact_to_snapshot` 只**追加**一个 `Snapshot`)、`events/manager.rs:410-418`(chat 末尾追加快照)、`events/store.rs`(`MemoryEventStore` 除 `clear_session` 全删外无按 id 区间的 GC/delete-older)。
- 触发条件:活跃会话长期聊天(超过 `AutoCompaction.max_turns`,默认 1024)。
- 后果:紧凑化只把「投影 context」压到近 `max_turns` 回合,**被紧凑化的旧事件仍原样留在日志里**。于是:1) 存储(含持久后端)随回合数无限增长;2) 每次 `chat()` / `history()` 都要 `store.read` 全量日志再 `project`(O(n) 每回合,长日志读入内存整段),越聊越慢。任务焦点「无界增长」在此只闭合了一半——LLM context 有界,存储无界。
- 建议:`compact_to_snapshot` 追加快照后,对已受快照覆盖的旧前缀事件做物理裁剪(在日志的纯前缀段落删掉 `id <= 快照起点` 之前的事件,因 `(session,branch,id)` 幂等,删前缀不影响后续 `next_id` 单调);或引入 `delete_older(session, branch, upto_id)` + 定期 GC。注意裁剪需保证快照 `compact_to_snapshot` 的 `next_id` 与裁剪边界一致。

### 3.2 【中】`EventSessionManager::chat`「先落库、后存记忆」,save_context 失败造成日志/记忆不一致且可重复

- 位置:`events/manager.rs:382-407`:追加 `UserMessage`+`AssistantMessage`(**步骤 2**)之后才执行 `memory.save_context`(**步骤 3**)。
- 触发条件:LLM 已成功返回,但 `save_context`(或该步骤)抛错。
- 后果:chat 返回 `Err`,但这一回合(用户消息 + AI 回复)已经**幂等且持久**地写进事件日志。调用方若按错误重试,会追加出**重复回合**(每次 append 都是全新自增 id,无内容去重)。这与同方法刻意坚持的「成功后写回」(E2/M-s1:LLM 失败则不落孤儿用户消息)的纪律不一致——「LLM 失败」保护了,**「记忆保存失败」没保护**。
- 建议:把 `save_context` 提到两笔 append 之前(记忆先落),失败则不写日志;或对重复回合做内容/`call_id` 幂等判重。

### 3.3 【低】`recovery` 闭环仍是占位,长日志恢复 O(n)

- 位置:`events/checkpoint.rs:37-61`(`NoopCheckpoint`)、`events/mod.rs` / `events/manager.rs` 未接入真实 checkpoint。
- 触发条件:进程崩溃后按事件日志恢复会话。
- 后果:永远全量重放整条日志,无 `up_to_id` 快照加速;叠加 3.1,长会话恢复退化为 O(n) 且占用峰值内存。接口与恢复语义(第 9-8 行注释)已定义好,但 0.23.0 交付物只是占位。属「半成品」而非 bug,列此说明。

### 3.4 【低】废弃 `SessionManager` 的消息存储同样无界

- 位置:`manager.rs` `chat()` 把每条用户+AI 消息**无条件** push 进 `session.messages` 并持久化(仅 LLM 喂给窗口 `max_context_messages` 受限)。
- 触发条件:用废弃 API 的会话反复聊天且不删会话。
- 后果:`MemorySessionStore` 中该会话历史无上限增长。对照:事件 API 已给默认自动紧凑化(有界 context),废弃 API 没有任何有界守卫。建议迁移到事件 API,或在废弃 API 上按 `max_context_messages` 同步裁剪存储。

## 4 不合理/可改进

- **分片锁表与 `session_memories` 表**:`events/manager.rs:86` 与 `:71` 两处 `HashMap`,条目按会话惰性创建,`delete_session`(`:251-265`) 会清理——生命周期闭环做对了。但对**永不删除**的会话,条目常驻;量级=活跃会话数,一般可接受。废弃 `manager.rs` 同形(`session_lock` 仅在 delete_session 回收)。
- **记忆窗口下丢弃工具消息**:事件 API 的 `chat()` 在 memory 路径只喂 `memory_variables` + 当前用户消息;`last_n_turns_messages`(`events/manager.rs:480-510`) 在窗口内也只取 UserMessage/AssistantMessage/Snapshot,**工具调用/结果不进 LLM 上下文**(只进回合计数使窗口按回合完整)。对 agent 类会话,工具证据丢失,模型看不到中间工具结果。与废弃 API 的「只存用户/AI 消息」同源,属设计取舍,但值得明确文档化。
- `project()` 的 `current_turn!` 宏内用 `.expect()`(`events/replay.rs:65`):有上面 INVARIANT 注释保证必为 `Some`,实际安全,仅风格上可在审慎前提下保留。

## 5 可加的新功能

- 事件日志**物理 GC / 时间分片归档**:按 `(session, branch, id)` 前缀裁剪 + 定期归档冷会话到对象存储,真正闭合 3.1。
- **真实 `SessionCheckpoint`**:落地 Mongo/Redis,`latest()` 返回最近投影 + 尾重放,配合幂等 append 实现秒级恢复(接口已备好)。
- 事件日志字段化查询:`project` 目前全内存,可暴露 `turns_range(branch, from_turn, to_turn)` 做增量投影,避免每回合全量投影。
- Agent 工具循环(record_tool_use 已支持)可加**工具编辑/撤销**事件类型,支持分支改写。

## 6 对标主流

- 对标 LangGraph Checkpointer:事件 API 的追加日志 + 快照 + fork 已覆盖 LangGraph checkpointer 的「历史 / 时间旅行 / 分支」大半;缺真实持久后端(目前 `MemoryEventStore`)与物理裁剪。
- 对标 LangChain `ConversationBufferWindowMemory` + 会话:lc-sessions 事件 API 以 turn 为单位(而非消息数)切窗口,工具配对更稳,是比 LangChain 消息级窗口更干净的切法。
- `AutoCompaction` 的确定性快照(无 LLM)对标 LangGraph 的 CompactionStrategy,但 LangGraph 紧凑化同样不释放存储——这是社区普遍未解决的问题,lc-sessions 只是随了大流,并非特例缺陷(但仍是真实风险)。

## 7 一句总评

事件 API 设计优雅、并发与幂等打磨到位,唯一硬伤是「紧凑化只压 context 不裁存储」导致日志存储 O(n) 无界增长;另有一处「落库先于存记忆」的一致性小缝,以及 checkpoint 仍为占位的半闭合闭环。