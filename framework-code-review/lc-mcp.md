# lc-mcp

> 只读代码评审报告。评审对象:`crates/lc-mcp/src/` 全部功能 `.rs`(`#[cfg(test)]` 测试模块已通读但不计入 bug 论据)。未改动任何代码。

## 1. 概览

`lc-mcp` 是 langchainrust 里 MCP(Model Context Protocol)支持 crate,一条 crate 同时覆盖三条传输轨道:

- `client_stateless`(StatelessMcpClient)—— 2026-07-28 自研无握手、每次请求自带 `_meta` + `Mcp-Method`/`Mcp-Name` 头的 HTTP POST 扩展轨道,
  配合 `transport/stateless.rs` 与 `transport/streamable_http_server.rs` 里的 stateless 服务端;
- `client_stdio`(StdioMcpClient)—— 走官方 `initialize` 握手的新行分隔 JSON-RPC stdio 轨道;
- `client_streamable`(StreamableMcpClient)—— 官方 Streamable HTTP 轨道(JSON-RPC POST + SSE 返回流、`Mcp-Session-Id` 会话、OAuth 2.1 Bearer 鉴权)。

三条轨道都实现 `McpToolClient`(`tool_client.rs`),可经 `MCPToolAdapter`(`tool_adapter.rs`)挂进 Agent。

除纯类型/协议模块外,还包含一批自有工程化能力:服务端 `MCPServer`(`server.rs`)与手写 HTTP/SSE 传输(`transport/streamable_http_server.rs`)、网关 `MCPGateway`(`gateway/`)、多租户 `TenantGateway`、按 Server 的隔离沙盒 `ServerSandbox`(`sandbox.rs`)、工具编排 DAG `ToolOrchestrator`(`orchestrate.rs`)、静态+动态分层工具发现 `ToolDiscovery`、全新型号受限的熔断器 `CircuitBreaker`(`health.rs`)、MRTR 多轮往返状态机(`protocol.rs`)、OAuth2.1 鉴权发现(`oauth2.rs`)、OpenTelemetry 埋点(`instrument.rs`)。

安全基调总体很扎实且一致:跨主机重定向一律 fail-closed(stateless `Policy::none`、streamable 同源、OAuth 同级同 Scheme/端口)、A15 先验签后看 claim、A16 默认失败关闭的执行门、A17 常量时间比对、审计与埋点参数级脱敏、SSRF 走 `lc_core::ssrf::guarded_*`。下面结论分"确认 bug → 改进点 → 新功能 → 对标"。

## 2. 遍历的文件清单(全部功能 .rs)

按 `lib.rs` 模块顺序,(含 `lib.rs` 共 **39 个**文件,嵌套模块 `gateway/`、`transport/` 各 5 个文件):

**顶层(29):**
1. `lib.rs`
2. `auth.rs`
3. `client_stateless.rs`
4. `client_stdio.rs`
5. `client_streamable.rs`
6. `completion.rs`
7. `connection_manager.rs`
8. `elicitation.rs`
9. `execution.rs`
10. `health.rs`
11. `instrument.rs`
12. `oauth2.rs`
13. `orchestrate.rs`
14. `prompts.rs`
15. `protocol.rs`
16. `resources.rs`
17. `roots.rs`
18. `sampling.rs`
19. `sandbox.rs`
20. `server.rs`
21. `subscriptions.rs`
22. `tasks.rs`
23. `tenant.rs`
24. `tool_adapter.rs`
25. `tool_client.rs`
26. `tool_discovery.rs`
27. `tool_namespace.rs`
28. `tool_timeout.rs`
29. `types.rs`

**gateway/(5):** `gateway/mod.rs`、`gateway/policy.rs`、`gateway/rate_limiter.rs`、`gateway/method_rate_limiter.rs`、`gateway/audit.rs`

**transport/(5):** `transport/mod.rs`、`transport/stateless.rs`、`transport/stdio.rs`、`transport/streamable_http.rs`、`transport/streamable_http_server.rs`

> `test_support.rs` 仅在 `#[cfg(test)]` 下存在,属测试设施,不计入功能清单。

## 3. 发现的 bug / 高风险点(按严重度)

### 3-1 [MEDIUM] `Gateway::as_base_tools` 产出的工具适配器绕过了每-Server 限流与统一审计

- **位置:** `gateway/mod.rs:366-387`(`as_base_tools`);对照 `gateway/mod.rs:268-358`(`call` 入口)。
- **触发:** 用 `gateway.as_base_tools().await` 把统一注册表转成 `Arc<dyn BaseTool>` 列表挂到 Agent 之后,直接调用这些 adapter(`MCPToolAdapter::run`,`tool_adapter.rs:217-239`)。
- **后果:** 统一入口 `call` 里做三件事:(a) 固定窗口每-Server 限流(`gateway/mod.rs:288-303`,超限 `record` 后拒绝);(b) A16 执行门(`authorize_call`，先于任何网络 IO);(c) 统一审计 `record`。而 `as_base_tools`(377-384 行)只给 adapter 接了 `with_execution_policy` + `with_approver` + `with_timeout`,**没有**接 `rate_limiters`,也没有把调用写进格林 `GatewayAuditRecord` 环形缓冲。`MCPToolAdapter` 体内也没有 rate-limiter 或审计字段(`tool_adapter.rs:217-239`)。
- 即:同一个 Server 的同一批工具,走 `gateway.call("srv:tool",…)` 受限流+审计,走 `as_base_tools` 生成的工具则不受限流、不留统一审计。文档(360-365 行)宣称“invoking an adapter directly cannot bypass it”,只对“执行门”成立,对限流与审计是**过当承诺**。
- **建议:** `as_base_tools` 里把该 Server 的 `RateLimiter` 与统一审计钩子注进 adapter(或让 `MCPToolAdapter` 支持可选注入),或把这类直接-adapter 调用也收敛进 `call` 的统一路径。

### 3-2 [LOW/MEDIUM] `transport/streamable_http_server.rs` 的 `q_is_zero` 对带尾零的十进制 `q=0` 判不出来

- **位置:** `transport/streamable_http_server.rs:348-351`。
  ```rust
  v.trim().trim_start_matches('-').trim_end_matches('0').trim_end_matches('.').is_empty()
  ```
- **触发:** 服务端做内容协商(`response_mode`,340-390 行)时,收到形如 `text/event-stream;q=0.0` / `q=0.000` / `q=0.0000` 的 Accept(即客户端显式把 SSE 的接受度标为 0)。
- **后果:** 这些拼写归一化后是 `"0."` → 再 `trim_end_matches('.')` → `"0"`,**非空**,`q_is_zero` 返回 `false`,于是该 range 被当作“可接受”处理;若另有 wildcard 或直接命中,服务端仍会当 SSE 流式回给已明言拒收 SSE 的客户端。与 345-347 行注释“`text/event-stream;q=0, */*` …必须不被选中”及 349 行注释“`0.0`、`0.000` 均表 q=0”自相矛盾——只有 `"0"` 和/或 `"-0"` 能测中,带尾零的小数拼写全部漏检。
- **附带说明:** 该 bug 只产生**漏检(把零当成非零)**,不会产生把非零判成零的误检(`"0.50"→"0.5"` 非空、`"1.0"→"1"` 非空都对),因此影响面收敛为“q=0 的小数拼写被流式推送”。
- **为何测试没拦住:** `streamable_http_server.rs` 约 620 行的用例用的是 `Q=0.000` **且与 `application/json` 并存**,命中 `listed("application/json")` 直接走 Json 分支,未触达 `q_is_zero` 对 SSE 单体的判断。
- **建议:** 用数值解析(`v.trim_end_matches('%').parse::<f64>() == 0.0`)或规范化(`&v` 先 `f64` 再比较),并补一个“只给 `text/event-stream;q=0.000` 应回 Json/非 SSE”的用例。

### 3-3 [边界敞口,如实说明非 bug] `server/discover` 订阅推送已接线,但 Streamable HTTP 传输未内置物理推送投递

- **位置:** `server.rs:240-249`(`notify_resource_changed` 用 `subscriptions.emit_list_changed` + 遍历 `subscribers_of` 逐个 push_sink 投递)、`subscriptions.rs`(注册表 + `ResourcePushSink::deliver`)、`transport/streamable_http_server.rs`(会话/请求处理)。
- **现状:** `ResourceSubscriptions` 是纯内存注册表,`emit_list_changed` 只产出信封并交给注入的 `ResourcePushSink`。`server.rs` 的 `RESOURCES_SUBSCRIBE/UNSUBSCRIBE` 分派是通的;但 `transport/streamable_http_server.rs` **没有挂 push_sink,也没实现 GET SSE 长连接推送**。也就是说:订阅可注册、通知信封可生成,但“把枚举实时推到已订阅的流式 HTTP 客户端”这条物理路径当前没有内置实现,交付留给调用方注入 sink。
- **定性:** 这是按注释明示的设计边界(“Delivery of that envelope is left to the transport layer”),不是隐藏缺陷;但若要宣称“资源订阅在 Streamable HTTP 轨道开箱可用”,**尚未闭环**。

### 已核对无问题、可放心的风险点(任务清单要求逐条验证)

- `tools/call` 超时是否释放并发额度 — **已修(无问题)**。`server.rs` 的 `handle_tools_call`(约 621-718 行)在分派前把 `(session, req.id) → Notify` 取消表注册,`tokio::select!` 的 timeout 分支会丢工具 future、由 Drop 释放信号量额度,不会把额度永久卡死。默认 `tool_timeout = Some(60s)`(`server.rs:148`)。
- Streamable 服务端会话表是否只按年龄 TTL、无数量上限 — **已修(无问题)**。`streamable_http_server.rs` 的 `insert_session`(约 413-433 行)先做 TTL 清扫,再 `if sessions.len() >= MAX_SESSIONS { return Err(()) } → 503`;`MAX_SESSIONS=1024`(约 49 行)、`SESSION_TTL=60*60`(约 41 行)。数量与年龄双上限都在。
- Stateless 无 id 通知是否被误当 `-32700` — **已修(无问题)**。`server.rs:936` 起 `if msg.id.is_none()` → `handle_notification` → 返回 202,而非 `-32700`。
- 客户端重定向是否跨原点透传 / 泄漏 session 头 — **已修(无问题)**。`transport/stateless.rs:47` 用 `Policy::none`;`transport/streamable_http.rs:76-94` 只跟随同源重定向,否则直接停(失败关闭)。OAuth 元数据发现同样限同源同 Scheme 端口(`oauth2.rs` `ensure_same_host`)。
- MRTR 状态机越界跳转是否被拒、`round_trips` 超限是否报错 — **已修(无问题)**。`protocol.rs` 的 `can_transition_to`(约 165-174 行)只允许 `Pending→AwaitingInput/Resolved`、`AwaitingInput→Pending/Resolved`;`client_stateless.rs` 的 MRTR 循环命中非法跳转返 `-32003`,且 `round_trips` 超限(默认 3)直接报错。
- 密钥/令牌是否泄进错误信息或日志 — **已修(无问题,且做得好)**。`oauth2.rs` 的 `redact_token_body_error`(572-591 行)从不回显可能含 `client_secret`/`refresh_token`/`access_token` 的原始 body;`sandbox.rs` 的审计写入前参数级脱敏(296-331 行,`password/secret/token/apikey/…` key 标记为 `[REDACTED]`);`instrument.rs` 的工具参数/结果默认不落 span,需显式 `with_tool_payloads(true)` 才记录并截断到 2048 字符。

## 4. 不合理 / 可改进(非 bug)

1. **`tasks.rs` 是占位 stub。** 2026 规范里的 Tasks 扩展(`tasks/get`/`tasks/result`/中途输入)当前只有 trait 壳,`status()`/`result()` 直接返回“implementation pending”。这是与 2026 spec 最明显的功能缺口(详见第 5 节)。
2. **`SamplingGuard` 的令牌预算只增不减、不随 lease 释放。** `sampling.rs:264` 进入即 `tokens_used += request_tokens` 且没有“退还”路径(深度 depth 会随 lease Drop 释放,预算不会)。保序语义上是“按链累积 max_tokens 的悲观预算”,注释已写明是设计意图;但对“兄弟并行 + 结果远小于 maxTokens”的场景会偏保守,可考虑按真实用量(messages/输出长度)结算或提供预算释放钩子。
3. **`server.rs` 里的 `handle_request_scoped` 对未注册的 `prompts/*`/`resources/*`/`completion/*` 返回 `method_not_found` 的“诚实边界”思路好,但各 provider 均为调用方注入 trait(`PromptProvider`/`ResourceProvider`/`CompletionProvider`),**没有给客户端侧预留对应的方法分派**——即服务器能力取决于注入,而 `client_streamable`/`client_stdio` 侧是否有对应的 `prompts/get`、`resources/read` 调用入口需要确认,否则易出现“服务端能但客户端没入口”的不对称。
4. **`EgressPolicy::allows` 子域匹配是手写字符串拼接**(`sandbox.rs:154-156`,`host.ends_with(&format!(".{a}"))`),对 `evil-example.com` 一类能正确拒绝(开头非点分隔),边界正确;但后缀匹配与 IDN/大小写/尾点(`example.com.`)未统一,属可加固点而非缺陷。
5. **`ToolNamespace` / `ToolDiscovery` 的 `HashMap` 顺序不稳定**,而 `wire-order-determinism` 记忆指出本项目对 serde_json Map 键序漂移很敏感;`names()` 与 `select()` 的返回顺序在网关里被“显式排序/独立收集”规避,但依赖城堡较弱——建议在需要稳定序处显式 `sort`。
6. **`instrument.rs` 的 GLSS/semconv 是“手抄镜像”**(注释写明“if the registry moves a name, update both places”),`lc-mcp` 因不依赖 `lc-callbacks` 只能镜像;这是有意的解耦,但存在双份定义漂移的风险,值得在代码里留同步检查或注释锚点。
7. **`MCP_VERSION` 仍是 `2024-11-05` 而 stateless 轨道用 `MCP_VERSION_STATELESS=2026-07-28`**,且 `SUPPORTED_PROTOCOL_VERSIONS` 需三方一起维护;两条轨道版本号的双轨制在文档里讲清楚了,但对外暴露上建议统一成一份“机制版本→轨道”对照,避免调用方误判。

## 5. 可加的新功能(MCP 官方 spec 2026 角度 / 主流 MCP SDK)

1. **补齐 MCP Tasks 扩展**。`tasks.rs` 已留好 trait 壳,直接实现 `tasks/get`、`tasks/result`、任务级取消与中途输入(`inputRequired`),让 stateless/streamable 轨道支持长时任务(可参考 Anthropic TS SDK 的 `McpAsyncTask`)。
2. **Streamable HTTP 的资源推送闭环**。给 `transport/streamable_http_server.rs` 挂上实际 `ResourcePushSink` 并实现 GET SSE 长连接(或提供 `Server-Sent-Events` 订阅频道),把第 3-3 节的可注入“边界”变成开箱即用的推送。
3. **客户端侧 prompts/resources/completion/roots 调用入口**。既然服务端能注入这些 provider,建议在 `McpToolClient` 的狭义接口之外,为 `client_streamable`/`client_stdio` 提供 `prompts/get`、`resources/read`、`completion/complete` 的对称客户端方法,提升与官方 SDK 的互操作面。
4. **分页(`nextCursor`)**。`tools/list`、prompts/resources 列表在官方 spec 里允许分页与增量;当前 `MCPToolDefinition` 列表与注册表是全量拉取,100+ server 场景下建议加 `ListResult` 游标分页。
5. **授权与 SSRF 之外的 Annotated/访问控制瘦身**:`server.rs` 的 per-request 鉴权可按 `Mcp-Method`/命名空间做更细的 ACL(现为全局 validator),便于多租户/多团队共享一个端点。
6. **`WebSocket` 传输**。主流 TS SDK 已在实验性支持 WS;作为 HTTP 之外的第三官方传输,可作为后续增量(现有 `ResourceSubscriptions` 就绪,WS 是天然的 sink)。
7. **`logging` / `notifications/resources/*` 之外的通知面**:实现 `notifications/message`(log)与 `resources/subscribed` 确认通知,完善官方 2026 通知矩阵。
8. **Exa/web 无关、纯工程**:为 `MCPGateway::as_base_tools` 补上第 3-1 节的限流+审计注入,并把 `TenantGateway` 的 `sync_all` 失败语义(哪个 server 挂、部分成功)对外暴露更细。

## 6. 对标主流(官方 MCP 用)

对比官方 `modelcontextprotocol/typescript-sdk` 与官方 Rust SDK:

| 维度 | 官方 TS/Rust SDK | lc-mcp | 结论 |
|---|---|---|---|
| 传输轨道 | stdio + Streamable HTTP(附 WS 实验) | stdio + Streamable HTTP + **自研 stateless** | lc-mcp 多一条可选的扩展轨道,官方无;兼容面持平 |
| 握手表 | `initialize`/能力协商 | stdio/streamable 均走握手;stateless 无握手(设计差异) | 语义不同但各自自洽 |
| 会话/资源推送 | 服务器端会内置 SSE/WS 通道投递订阅通知 | 订阅注册 + push_sink 已就绪,但 HTTP 轨道未内置物理推送(3-3) | lc-mcp 略欠一个“开箱推送” |
| 鉴权 | OAuth 2.1(资源元数据 + DCR + 握手内联) | OAuth2.1(含 `WWW-Authenticate` 挑战解析、PKCE、DCR 元数据) + JWT `iss`/`aud`/验签 + 静态 bearer | lc-mcp 更宽(多了 JWT 与静态 bearer),且带 SSRF/跨主机加固,官方不带 |
| 失效/重试 | 无内建熔断 | `CircuitBreaker` 指数退避 + 半开单探针 + 413 拒线 | lc-mcp 明显更强 |
| 工具编排/命名空间 | 官方不提供 | 多 server `server:tool` 命名空间、冲突策略、DAG 编排、分层发现、租户隔离 | lc-mcp 明显更强 |
| 可观测性 | 官方有 semconv span | `McpInstrumentation` 按 GenAI/mcp semconv 打标、payload 默认脱敏、截断 | 相当/更保守(默认不开 payload) |
| 执行安全门 | 无 | A16 失败关闭批准/沙盒/无值守三层门 + 参数级 least-privilege | lc-mcp 明显更强 |

总体:lc-mcp 在“工程化广播面”(多轨道、网关、熔断、编排、沙盒、租户、可观测)远超官方 SDK;官方 SDK 唯一的相对优势是**紧跟最新 spec 的扩展实现完整度**(Tasks 等,lc-mcp 尚为占位)与经社区验证的握手/能力协商细节广度。

## 7. 一句总评

工程化与安全硬化和体系化程度领先主流官方 SDK,代码整体干净扎实;主要薄弱点在“与最新 spec 的扩展特色能力闭环度”(Tasks 占位、Streamable 资源推送未闭口)和一条误报率很低的协商小 bug(“q=0 小数拼写漏检”),外加一个需要正视的 `as_base_tools` 直调绕过限流/审计的过当承诺。