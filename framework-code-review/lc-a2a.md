# lc-a2a

Agent-to-Agent (A2A) 协议实现 crate 的完整只读代码审查。覆盖 `crates/lc-a2a/src/` 下全部 29 个 `.rs`(26 个功能源文件 + 3 个外部测试文件 3390 行)。仅阅读,未改动任何代码;唯一写入的文件就是本报告。

---

## 1. 概览

lc-a2a 是一个相当完整、工程成熟度很高的 A2A 服务端 + 客户端实现,遵循 A2A v0.3(兼容 draft 1.0.1)线协议。

架构分层:

- **协议层** `protocol/`:线类型——`AgentCard`(含签名)、`A2ATask`(`TaskStatus` 状态机)、`Part`/`FilePart`/`A2AArtifact`(v0.3 parts 模型)、`TaskPushNotification`(`kind` 判别)、`TraceContext`(W3C traceparent)、`MessageEnvelope`、`TaskFilter`、`A2AWorkflow`、JSON-RPC `A2ARequest`/`A2AResponse`/`A2AErrorData`。
- **服务端核心** `server/`:`A2AServer` 持有 `TaskStore`(默认内存 LRU)、幂等 `message_id` 表(有界)、`inflight_resumes` 续跑串行化、skill 路由、事件总线、身份鉴权、`cancellations`(task_id→Notify)。`handle_a2a_request*` 是纯逻辑层,可接任何 HTTP 框架。
- **HTTP 层** `server_impl.rs`(axum feature):挂载 `/.well-known/agent-card.json`(及 `agent.json` 别名)、`POST /`、`GET /events` SSE。含 loopback-默认绑定即拒绝公网开放、CORS 精确源白名单、bearer 鉴权、SSE 跨租户 403 守卫。
- **客户端** `client/`:RPC + SSE 双通道,`send_task_and_wait`/`subscribe`/`send_task_streaming`,卡片签名校验,沙箱,回退到 `agent.json`。
- **周边扩展** `router`/`security`/`resilient`/`rate_limiter`/`agent_adapter`:skill 路由、信任注册表、L1-L4 弹性客户端、限流、AgentExecutor 适配。
- **"规模化/联邦"契约层** `discovery`/`gateway`/`scale`:AgentRegistry、FederationGateway、以及 Scale ~1000 智能体的全套构件——**均无生产调用点(仅自测)**。

并发与安全方面做得非常扎实:任务终态用 `compare_and_update`(CAS)防竞态覆盖、取消通过 `select!` 真正终止在途 invoke future(有 Drop guard 单测实证)、C-A2A-1 以已鉴权主体为准的 owner 判定、共享 token 不泄漏跨租户 SSE。整体是"审查后反复打磨过的生产级代码"。

但存在一个系统性缺陷模式:**大量"只建契约层、未接线"的半成品**(见 §4),以及两处影响正确性的真实问题(续跑丢 skill 路由、工作流聚合乱序,见 §3 B1/B2)。

---

## 2. 遍历文件清单(29 个,全部读完)

**核心 (top)** — `src/lib.rs`
**协议层** — `src/protocol/{mod,card,message,model,part,task,tests}.rs`
**服务端逻辑** — `src/server/{mod,routes,execution,handlers,auth,message,tests}.rs`
**服务端 HTTP** — `src/server_impl.rs`
**存储** — `src/store.rs`
**客户端** — `src/client/{mod,sse,signing,tests}.rs`
**扩展** — `src/{router,discovery,rate_limiter,agent_adapter,security,resilient,gateway,scale}.rs`

行数(功能源文件,不含注释统计):`server_impl.rs`(845+,含大测试)、`server/mod.rs`(841,含测试)、`routes.rs`(546)、`protocol/model.rs`(~660)、`scale.rs`(~790,含测试)、`gateway.rs`(~750,含测试)、`security.rs`(~660,含测试)、`store.rs`(583,含测试)等。三个外部测试文件:`protocol/tests.rs`(719)、`server/tests.rs`(1720)、`client/tests.rs`(951)。

---

## 3. bug / 高风险点

### B1 【高】续跑(`tasks/send` + `taskId`)丢失 skill 路由
- **位置**:`src/server/routes.rs:231` (`handle_tasks_send_continue`)
- **触发**:任务最初以 `skillId=math` 通过 `tasks/send` 提交(由 `resolve_chain(skill_id)` 路由到 math 链,`routes.rs:110-111`);随后进入 `input-required`,客户端用 `tasks/send {taskId, message}` 续跑。
- **后果**:续跑路径固定 `let chain = self.resolve_chain(None);`——**永远回到默认链**,忽略原任务的 skill 路由。同一多轮对话前后两段在**不同模型/链**上执行,语义断裂;skill 路由对多轮任务完全不生效,与 `routes.rs:110-111` 的新建路径行为不一致。链选择没有持久化到 `StoredTask`,续跑无法还原原链。
- **修复方向**:在 `StoredTask` 记录 `skill_id`(或选定链),续跑时按原 skill 解析。

### B2 【高】工作流聚合顺序不保证 + 失败时"部分结果"其实被丢弃
- **位置**:`src/server/execution.rs:217-221`;对照 `src/server/routes.rs:431-433` 的文档注释
- **触发**:`tasks/runWorkflow` 执行完所有 step 后,把各 step 输出写入 `serde_json::Map<"step_id", output>`,再 `results.values().join("\n")` 拼成最终 output。
- **后果(顺序)**:step 执行是按序的(execution.rs:203 的 zip),但**聚合顺序取决于 `serde_json::Map` 的键序**。不启用 `preserve_order` feature 时 `serde_json::Map` 是 `BTreeMap`,按 step_id 字母序排列——`["s2","s1"]` 会被拼成 `"second\nfirst"`,与执行顺序相反。当前测试(`run_workflow_executes_steps_in_order_and_aggregates` 断言 `"first\nsecond"`)恰好是字母序,掩盖了该 bug。一旦某 step_id 命名顺序与执行顺序不同(如 `s10` vs `s2`,`s10` 字母序在 `s2` 前),输出即错乱。这是依赖 Map 迭代序的脆弱正确性问题。
- **后果(部分结果)**:`routes.rs:432` 注释声称"失败时已聚合的结果仍会返回",但 `execution.rs:227-244` 的失败分支只写入 `finalize.error`,**从不写入 `finalize.result`**——已执行的 step 产出被整体丢弃,与文档描述的契约不符。
- **修复方向**:用 `Vec<(step_id, output)>` 按下标聚合(保证执行序),并让失败分支把先前的部分输出作为 result 一并返回。

### B3 【中】客户端 SSE 无读超时、无 lag/reset 自愈,订阅路径可能静默丢终态
- **位置**:`src/client/sse.rs:32-48`(`A2ASseStream::next` 直接 `self.response.chunk().await`,**无超时、无 recv 超时**);`src/client/mod.rs:81-83`(`A2ASubscription::next` 纯委托);对照服务端 `src/server_impl.rs:373-375`(广播 lag 时发 `event: reset data: lagged`)
- **触发**:P2-9 `subscribe()`(或 `connect_sse`)建立 SSE 后,若服务端 `broadcast` 通道因慢消费者 lag,服务端会切出一帧"reset"，客户端 `A2ASseStream` 解析器(`sse.rs:84-91`)只跳过 `data=="[DONE]"` 与解析失败帧,**不认识 `reset` 语义**;若 lag 导致连接关闭,`next()` 直接返回 `None`,流结束。
- **后果**:客户端既不会读超时报错,也不会在 reset/断流后回退到 `tasks/get` 轮询补齐终态——`completed`/`artifact` 事件可被静默吞掉,调用方等不到任务结束(卡在 `working` 视角)。相比 `send_task_and_wait`(有 1s 轮询 + 超时)的健壮性,这是一个明显的健壮性缺口。
- **修复方向**:`next()` 加读超时;识别 `reset`/`lagged` 帧或流中断时,按任务 id 回退 `tasks/get` 拉齐状态。

### B4 【中】`AuthRequired` 是"死的契约态",且状态机文档与其 `can_transition_to` 相互矛盾
- **位置**:`src/protocol/task.rs:50-82`。状态机黑板注释(50-59 行)画出 `auth-required ──→ working`;但 `can_transition_to`(63-82)只有 `(AuthRequired, Submitted)` 与 `(AuthRequired, Expired)`,**没有 `(AuthRequired, Working)`**。
- **后果**:全代码库没有任何 handler 把任务置为 `AuthRequired`(搜索不到生产者),该状态是**永不产生的死契约态**;同时一旦某个自定义链/agent 层真的置成 `auth-required`,`tasks/send_continue` 只会检查 `InputRequired`(routes.rs:199),既无法续跑、也无法合法转入 `working`——该任务只能等 TTL 过期。状态机自洽性欠佳。
- **修复方向**:要么补 `(AuthRequired, Working)` 并让续跑路径接受 `AuthRequired`,要么删掉这个黑板上画着但代码不达的状态。

### B5 【低/边缘】`message_id` 幂等在有界表 + LRU 驱逐双重作用下可被静默打破
- **位置**:`src/server/mod.rs:110-149`(有界 `MessageIdTable`,上限 `MAX_MESSAGE_IDS=10_000`)+ `src/server/routes.rs:44-55`(reserve 命中后再查 store,驱逐则 `abort` 并当作新发送)
- **触发**:1) `message_id` 被表 LRU 顶出(>10k 个去重键)→ 重试找不到 → 重新 claim → **新建任务、链再跑一次**;2) 对应任务被 store 的 LRU(`DEFAULT_MAX_TASKS=10_000`)驱逐,`reserve_message_id`(mod.rs:761-764)把它当作可回收 → 下一次同 `message_id` 又建新任务重跑。
- **后果**:幂等去重只在有界窗口内成立;重试老请求可重复执行副作用型链。这算有界存储的固有代价,但无任何日志/提示,调用方无从得知幂等已被削弱。
- **修复方向**:至少在驱逐时打 warning;文档化"message_id 幂等范围受表容量限制"。

### B6 【低】鉴权失败的 JSON-RPC 错误 `id` 恒为 0,未回显请求 id
- **位置**:`src/server/mod.rs:601`(`A2AResponse::error(0, 401, ...)`);经 `server_impl.rs:280-285` 转成 HTTP 401 透出
- **触发**:带 token 服务上任何未带/错误 bearer 的 `POST /`。
- **后果**:JSON-RPC 规范要求错误响应的 `id` 回显请求的 `id`(至少为可关联的 request id);这里恒为 0,客户端无法把 401 与具体请求/批次对齐(多请求并发时尤甚)。属轻微不规约。
- **修复方向**:把请求 `id` 透传进 `check_auth` 的错误响应。

### B7 【低】`tasks/sendSubscribe` 服务端几乎等于 `tasks/send` 的别名
- **位置**:`src/server/routes.rs:146-152`
- **触发**:调用 `tasks/sendSubscribe`。
- **后果**:handler 原样委托 `handle_tasks_send`,除返回同款 `submitted` ack 外**不做任何"订阅"登记/资源跟踪**。订阅语义完全落在客户端(先开 SSE、再 POST)与服务端广播总线上,服务端无法感知/约束谁在订阅、也无法按连接回传。不算 bug,但与 A2A 1.0 `sendSubscribe` 的"ack 与增量事件走同一条流"语义有明显差距(见 §6)。

### B8 【低/信息】非身份 SSE 连接在鉴权服务器上静默收不到任何东西
- **位置**:`src/server_impl.rs:648-656`(`notification_visible` 对 `principal=None` 且 `has_auth()` 时仅放行无 owner 任务)
- **触发**:鉴权服务器上,一个用共享 token(解析不出 principal)或未带 token 的 `GET /events` 连接订阅。
- **后果**:行为与拉取侧(`tasks/get`/`list`)一致、是刻意收紧(防共享 token 泄漏跨租户)的正确安全取舍;但这类连接会**静默地永远收不到事件**且无任何提示,调用方难排查。建议在这类连接上给一条说明性通知/日志。

---

## 4. 不合理 / 可改进(含死代码·未接线)

### 4.1 系统性模式:"只建契约层、未接线"——多个整模块无生产调用点(仅自测)

这是本 crate 最突出的结构性问题,与集中式的 A2A 服务/客户端主干脱节:

| 模块 | 构件 | 位置 | 生产调用点 |
|---|---|---|---|
| `scale.rs`(约 790 行) | `SkillIndex`、`AgentTier`/`HierarchyPolicy`、`DelegationGuard`、`TaskSharder`、`CircuitBreaker`、`StickyRouter`、`TaskGraph` | `src/scale.rs:78-543` | **无**——仅测试(scale.rs:578+)。目标"~1000 agent 规模化"整层未接入客户端发起或服务端路由 |
| `gateway.rs`(约 750 行) | `FederationGateway`、`CallPolicy`、`DataContract` | `gateway.rs:188-210` | **无**——仅测试(gateway.rs:388+) |
| `discovery.rs` | `AgentRegistry`、`RegistryClient` | `discovery.rs:43,141` | **无**——仅测试(discovery.rs:219,310) |
| `signing.rs` | `sign_card_jws`/`verify_card_jws`(JWS HS256)+ `canonical_json` | `signing.rs:150,172` | **无**——仅测试;生产接线的是另一套 HMAC `sign_agent_card`/`verify_card_signature`(供 `get_agent_card` 校验) |
| `security.rs` | `verify_chain`/`issue_card`(信任链) | `security.rs:244,301` | **无**——仅测试;`TrustRegistry.verify_card` 已接入客户端 `get_agent_card`,但链式/签发未接线 |
| `protocol/message.rs` | `MessageEnvelope`、`A2ARequest::send_envelope` | `model.rs:610,195`(message.rs) | **无**——仅 `protocol/tests.rs` 使用。统一消息模型未进入收发主干 |

这些构件都**被 `lib.rs:63-75` 公开 re-export**(`AgentRegistry`,`FederationGateway`,`SkillIndex`,…),所以编译器不会报 dead_code 警告——它们作为"公共 API"存在,但**没有任何真实代码路径调用**。评审建议要么把 `scale`(至少 `CircuitBreaker`/`StickyRouter`——后者已在本仓库 LangGraph supervisor 路由找到对应)接入客户端路由,要么标注为 `#[doc(hidden)]` 的实验层,避免给使用者"已支持多智能体规模化/联邦"的错误印象。`CircuitBreaker` 实现本身质量很高(HalfOpen 单探针、失败重熔断、测试齐全),是这些构件里最值得接线的。

### 4.2 双套卡片签名方案并存,仅其一接线
HMAC 签名有真实使用路径(`sign_agent_card`→`get_agent_card` 校验,`client/mod.rs:45`)与 TrustRegistry 校验结合;JWS 方案完整实现了 RFC-8785 近似 canonical JSON + HS256 紧凑 JWS,却只活在测试里。两套并存徒增维护面,应明确二选一。

### 4.3 `run_task` 终态 CAS 失败时结果被静默丢弃
`execution.rs:115-133`:链完成后若 `compare_and_update` 因并发 cancel 返回 `false`,已完成的结果不落库、不发事件——行为上正确(终态是 cancelled),但无日志,排障时可能困惑"结果去哪了"。

### 4.4 SSE 是"进程内广播"而非按连接跟踪
事件总线是 `broadcast`(`mod.rs:312-325`),SSE 只是对广播的过滤投影(`server_impl.rs:349-380`)。慢消费者即 lag→reset。缺:
- 事件回放 / 断线重连补拉(客户端侧,见 B3);
- 每订阅连接的去重与资源上限(无连接数、无事件速率限流)。

### 4.5 文档与代码契约的少量偏差
- `routes.rs:432-433` 声称失败时仍返回已聚合结果,实际丢弃(见 B2);
- `task.rs` 状态机黑板画了 `auth-required→working` 但代码不允许(见 B4);
- `A2AServer` 详注由早期查询(workflow 同步执行)演化到现在的后台执行,部分 `runWorkflow` 注释仍暗示"返回 then poll",仔细看是 G2 后已是 poll,可视为陈旧。

---

## 5. 可加的新功能(A2A spec 1.0 + 主流多智能体框架)

1. **`sendSubscribe` 单流复用(HTTP/2)**:现为"POST + 独立 SSE"两段式,客户端注释已自认与规范有差距(`client/mod.rs:677-680`)。完整实现 ack 与增量事件共享一个流,可省掉" SST 先开再发、丢早帧"问题(当前靠先开 SSE 规避)。
2. **客户端 SSE 自愈**:读超时 + `reset`/断流时自动回退 `tasks/get` 轮询补齐终态(配 B3 修复),把 `subscribe()` 的可靠性与 `send_task_and_wait` 拉平。
3. **把 `scale.rs` 接线**:`StickyRouter`(hash 钉住有状态 agent)+ `CircuitBreaker` 接入多 agent 客户端;`TaskGraph` 对接父/子委托,与 LangGraph Supervisor 子智能体路由对齐。
4. **续跑/任务的 skill 亲和性**:`StoredTask` 记录 `skill_id`,续跑沿用原链(修 B1),并让 `tasks/get` 返回该元信息。
5. **事件回放 / at-least-once 推送**:广播 lag 或断连后,按 `message_id` 或任务 id 提供增量回放端点,配合客户端补拉。
6. **`AuthRequired` 落地**:把"需身份才能继续"真正映射到状态(例如链抛授挑战),并补 `AuthRequired→Working` 转移与续跑支持。
7. **可观测性**:任务驱逐、CAS 竞争失败、SSE lag/reset 加结构化日志(tracing spans),已有多处 `log::debug`/`log::warn` 但未成体系。
8. **`tasks/list` 分页 + 游标**:现全量返回(`routes.rs:417`),上千任务时无分页与上限。

---

## 6. 对标主流

**对 Google A2A 官方参考实现 / TS SDK:**
- 对齐:核心 endpooint(`tasks/send|get|cancel|list`)、任务生命周期、JSON-RPC 2.0、`/.well-known/agent-card.json`、bearer 鉴权、SSE 推送、卡片签名,均与官方线协议一致;`owner` 授权、幂等 `message_id` 属官方之外的加固。
- 差距:官方 `sendSubscribe` 约定 ack+事件单流复用(本实现为 POST+SSE 分叉,`client/mod.rs:677-680` 自认);官方 draft 1.0/1.0.1 的 parts 模型在**主任务/消息/结果**上仍未落地(`A2ATaskResult` 仍是 `{output:String}`,parts 只用于推送 ArtifactUpdate),跨端解析力不均;
- 特色超出官方:身份绑定 token 的按租户 SSE 过滤、CAS 终态保护 + 真正终止在途计算、沙箱 payload 限制、L1-L4 弹性客户端——这些在官方参考实现里没有。

**对 LangGraph 多智能体(supervisor / handoff / 子智能体):**
- 本仓库 6128624 提交已为 LangGraph supervisor 加了动态子智能体路由,并新增 `CandleEmbeddings` 量化等。
- lc-a2a 的 `scale.rs`(`SkillIndex`/`TaskGraph`/`StickyRouter`/`CircuitBreaker`)正是"同类、但未接线"的多智能体编排构件:LangGraph 有真实的 supervisor+handoff 运行时把路由/委托串起来,lc-a2a 这边只有孤立的便携构件。`TaskGraph` 的父/子委托 + 环检测恰好可作为 supervisor 编排的底层,建议接线打通。

---

## 7. 一句总评

**lc-a2a 的服务端核心(状态机、鉴权、幂等、SSE、取消语义)是打磨扎实的准生产级代码,但 crate 里"只建契约层未接线"的规模化/联邦/发现层与两处真实正确性问题(续跑丢 skill 路由、工作流聚合依赖 Map 键序)让"宣称能力 > 实际可用能力"的落差成为本次审查最值得修复的点。**

---

*(报告范围:`crates/lc-a2a/src/` 下 29 个 `.rs`,全部只读;未修改任何代码、未做任何 git 操作。)*