# lc-observability 代码审查报告

> 审查方式:只读,通读 `crates/lc-observability/src/**/*.rs` 全部功能源码;未改动任何代码。
> 结论基于代码实际情况,行号以当前工作区为准。

## 1 概览

lc-observability 是观测性 sink 插件层,实现 `lc-core::observability::{MetricsSink, ObsEvent, ObsError}`,事件由 `TokenTrackingLLM`、`AgentExecutor` 经 callback 写入,对外提供三种导出:

- `JsonLinesSink`(默认 `jsonl`):每事件一行 JSON 追加到文件,可 tail / jq / 重放。
- `MongoSink`(`mongodb`):每事件一条 document,保留 serde tag `"kind"` 区分 `token_usage` / `cost` / `agent_metrics`。
- `OtlpSink`(`otlp`):每事件一个 OTel span,带 GenAI 语义约定属性,经 OTLP/HTTP-JSON 交给 batch exporter 推送 collector。

设计基线正确:导出失败返回 `ObsError`,框架只 `warn` 不中断主体(sinks never interrupt)。OTLP 属性名与 `lc-callbacks::semconv` 同拼写、本地常量不引依赖,干净。审查焦点(gen_ai.* 语义 / 半成品 / 闭环)逐一核验:gen_ai.* 键名与当前 semconv 拼写吻合;但**追踪闭环未闭合、token span 缺模型归因**两点值得修。

## 2 遍历文件清单

实际通读的功能源文件(4 个):

- `lib.rs` — 模块装配与实例导出,feature 门控(mongodb / otlp)。
- `otlp.rs` — `OtlpSink` / `OtlpConfig`(读 `OTEL_EXPORTER_OTLP_*` 环境变量)、span 名称与属性映射、header 百分号解码。
- `mongo.rs` — `MongoSink`。
- `jsonl.rs` — `JsonLinesSink`。

## 3 bug/高风险点(按严重度排序)

### 3.1 【中高】OTLP 每次 `export` 都是独立根 span,单次运行的 llm / cost / agent 之间无追踪血缘

- 位置:`otlp.rs:153-160`:`self.tracer().start(span_name_for(event))` 每次都开一个**新根 span**,没有继承任何父 span/trace 上下文。`AgentMetrics.trace_id`(字符串)只在 `agent_attributes` 里当普通属性贴上去(`otlp.rs:214-216`,常量 `otlp.rs:51`),**从未真正用来设置 OTel 的 trace ID 或作为父子连接**。
- 触发条件:一次 agent 运行产生 ≥2 类事件(TokenUsage + Cost + AgentMetrics)。
- 后果:后端(Jaeger / Grafana Tempo)看到的是 3 条互不相关的根 trace,无法把「agent 总览」「每次 LLM 调用的 token 用量」「成本」聚合成一次运行的完整调用链。跨参照只能靠 `langchainrust.trace_id` 属性手动 join,违背 OTel「trace 把你的 span 串成一次请求」的闭环目标。任务焦点「闭环 / empty instrumentation」主要落在这里。
- 建议:在 `OtlpSink` 外层接收一个 OTel `Context`(或从 `lc-core` 的 run 级 `trace_id` 构造一个含该 trace id 的 `TraceContext`),`export` 时用 `tracer.start_with_context(name, &parent_ctx)`;agent span 作为该 trace 的父节点,llm/cost span 挂其下。

### 3.2 【中】TokenUsage(llm) span 缺模型 / provider 归因,无法按模型统计 token

- 位置:`otlp.rs:196-202` `usage_attributes`:只写 `gen_ai.operation.name="chat"` + `gen_ai.usage.input_tokens/output_tokens`,**没有 `gen_ai.request.model`,也没有 `gen_ai.provider.name`**。Cost span 才带 model/provider(`otlp.rs:177-188`)。
- 触发条件:任何 `ObsEvent::TokenUsage`(由惯用的 TokenTrackingLLM 在每个 LLM 调用后触发,频率最高)。
- 后果:占事件量最大头的 llm span 无模型属性,后端无法按模型汇总 token/成本;要按模型看用量必须等成本事件,而成本事件未必每次都发。
- 建议:若 `TokenUsage` 里拿不到模型名,可在 `usage_attributes` 的调用处并入当时已知的 model;或扩展 `ObsEvent::TokenUsage` 携带 `model` 字段。

### 3.3 【低】失败即静默,C 侧无 error/exception span

- 位置:`lib.rs:13-16` 文档「sinks never interrupt the agent」;`otlp.rs:152-161` `export` 只 `start/set/end`,失败时不写任何错误 span。
- 触发条件:某个 agent run 中途出错(LLM/tool 异常)。
- 后果:出错的那次运行在 OTLP / jsonl / mongo 里可能**什么都看不到**(只有成功的 token/cost/agent 事件),失败对观测面不可见,DevOps 需靠应用日志而非 trace 定位。这与其说是 lc-observability 自身的 bug(事件面由上层决定是否发布),不如说整个观测通路没有为「失败」预留 span 类型/语义约定。列「低」,结合 3.1 一并改进:失败同样需要一个挂父子血缘的 error span。

## 4 不合理/可改进

- **`OtlpSink::install` 必须跑在 Tokio runtime 内**(`otlp.rs:120-122`,batch exporter 会 `runtime::Tokio` spawn;测试 `otlp.rs:426` 也加注 multi_thread)。在纯同步/非 Tokio 线程里调用会死锁或 panic——文档已写明,但这是易踩的 API 陷阱,建议在 `install` 内先探测 runtime 并在文档/错误信息更强提示。
- **`gen_ai.operation.name="chat"` / span 名 "llm"/"cost"/"agent"**:最新 GenAI OTel semconv 建议 span 名用 `{gen_ai.operation.name} {gen_ai.provider.name}` 模板。当前 span 名过于简陋,演进空间小。属性键拼写本身与当前主流 semconv 吻合,这部分 OK。
- **mongo `/ jsonl` 无运行级 join 键**:两者都直接 `serde_json::to_value(event)` 落原始事件,若不带 `trace_id`,单次运行的 token/cost/agent 文档同样无法快捷聚合。建议 jsonl/mongo 也统一写入 `trace_id` 字段(与 OTel 侧 `langchainrust.trace_id` 对齐)。
- **mongo 无索引/TTL 建议**:观测事件是高频低价值,落 Mongo 建议按 `kind + trace_id` 建索引、config 里给 TTL 兜底,避免 collection 无界。目前 `mongo.rs` 不做任何索引。

## 5 可加的新功能

- **trace 汇聚的 `trace_id` 贯通**:jsonl/mongo/otlp 三端统一携带同一 run `trace_id`,供外部聚合 + 关联业务日志(trace_id ↔ 应用日志 span)。
- **失败/异常事件类型**:新增 `ObsEvent::RunError{ scope, error }`,OTLp 以 error span(携带 `error.type` / `error.message` semconv)发布,闭合「失败可见」。
- **采样与批量化**:OTLP 已由 SDK batch exporter 处理;jsonl 侧可加按 `trace_id` 的写缓冲/批量 flush。
- **成本聚合 KPI**:在 `cost` span 已带 `langchainrust.cost_usd` 基础上,可加 OTel 指标(Meter,RuntimeMetric)输出 `gen_ai.client.token.usage` 面向 Prometheus。

## 6 对标主流

- 对标 OpenTelemetry GenAI semantic conventions:lc-observability 使用的 `gen_ai.operation.name` / `gen_ai.request.model` / `gen_ai.provider.name` / `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` 与当前 semconv 拼写一致,方向正确。落后点:1) 未按新规范把 span 名模板化;2) llm span 缺 model/provider;3) 未把 trace 血缘打通(3.1)。
- 对标 LangChain LangSmith / OpenLLMetry:这些产品有运行级 trace(整次 run 一条,内部 LLM/tool/成本为子 span)。lc-observability 的 OtlpSink 目前「每事件一根根 span」,间距还明显——这正是 3.1 要修的核心,修后即对标 LangSmith 的 trace 形态。
- jsonl/mongo 侧对标 LangSmith 的 run 记录落库/导出,已完成度较好(语义事件已结构化)。

## 7 一句总评

三个 sink 实现干净、gen_ai.* 键名对齐、失败不影响主链路,但 OTLP 侧「每事件一根独立根 span」关不上追踪闭环(需父子血缘)、TokenUsage span 缺模型归因,是值得优先修的两处。