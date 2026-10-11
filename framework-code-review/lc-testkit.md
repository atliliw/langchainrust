# lc-testkit

## 1. 概览

`crates/lc-testkit/` 是框架的记录/回放(record/replay)测试马具 crate,使框架无 API key 自测。

核心构成:

- **RecordingProvider** (recording.rs) — 包装任意 `BaseChatModel`,真实调用后把请求/响应对 append 到 JSONL 文件。pass-through:失败不写,写失败只 `log::warn!`。
- **ReplayProvider** (replay.rs) — 零网络回放,支持三种策略 `ReplayStrategy::{Fifo, ByToolName, Exact}`。
- **golden.rs** (N2/v0.25) — 录播 → `lc-evaluation` golden 数据集 → 离线打分回归闭环。
- **canon.rs** — 顺序无关的 JSON 规范比较断言。error.rs —— `TestkitError`,bridge 到 `ProviderError`。
- **wire.rs** (`wire` feature) — 可复用 HTTP/1.1 环回 mock server,供 embeddings/vector-stores/rag 的 dev-dependencies 做 wire 级回放。

代码质量总体较高:错误路径大多显式、三种回放策略设计清晰(尤其 `Exact` 用了"无匹配即报错、绝不静默 FIFO 兜底"的正确原则)、N2 闭环的 FIFO 对齐有文档且有测试支撑。

## 2. 遍历文件清单(7 个功能 .rs,全部通读)

| 文件 | 内容 |
|---|---|
| `D:\...\crates\lc-testkit\src\lib.rs` | 模块声明 + `pub use` 聚合 |
| `D:\...\crates\lc-testkit\src\error.rs` | `TestkitError` + `From<TestkitError> for ProviderError` |
| `D:\...\crates\lc-testkit\src\canon.rs` | `CanJson` 规范值 + `assert_json_canonical` |
| `D:\...\crates\lc-testkit\src\wire.rs` | 环回 mock HTTP server |
| `D:\...\crates\lc-testkit\src\recording.rs` | `RecordingProvider`/`Recorder`/`RecordedExchange` |
| `D:\...\crates\lc-testkit\src\replay.rs` | `ReplayProvider` + 三种策略 + 测试 |
| `D:\...\crates\lc-testkit\src\golden.rs` | N2 golden 数据集构建/回放闭环 |

未发现 lc-testkit 有独立功能文件被遗漏(无 `errors/`、`utils/` 等子目录)。`wire.rs` 仅在 `wire` feature 下编译(lib.rs:20-21)。

## 3. bug / 高风险点

### [HIGH] recording.rs:279-300 — 流式调用中途出错仍会记录"截断的"部分响应,与声明契约矛盾

触发条件:`stream_chat` 的底层流在产出若干 chunk 后产出一个 `Err`(真实 provider 中途失败)。

机制:unfold 状态机里,`Some(Err(e))` 分支(`:279`)**与** `Ok` 分支一样返回 `Some((Err(...), ...))` 并**保持累积状态**,把控制权交回消费方;消费方下一次 poll 会再次进入闭包执行 `inner.next()`。底层流在报错后再 poll 通常返回 `None`,于是落到 `:282-299` 的 `None` 分支 —— 把**已累积的部分内容**打包成 `LLMResult` **写进 recording 文件**。

后果:

1. 直接违背该文件注释(lib.rs:12-13、recording.rs:243-249)声明的"mid-stream error 只转发、不记录,与 `chat` 的失败契约一致"。`chat`(非流式)失败时确实不写,但稳定的 `stream_chat` 则不保证。
2. 是否触发取决于消费方行为:消费方若在收到 `Err` 后 `break`/`return` 丢弃流,`None` 分支不会执行 → 侥幸不写;若消费方用通用 `collect`/`to_vec` 等"poll 到 None 为止"的方式消费(它们不会因其中一项是 Err 就提前停),流会继续 poll 到 `None` → 写出部分响应。即**非确定性**,同一错误在不同消费路径下产生不同录制结果。
3. 被写进录播的部分响应在回放时会原样吐出截断文本,掩盖真实失败,并污染 N2 golden 数据集(截断 reference 会参与回归打分)。

修复方向:在 `Some(Err)` 分支置一个 `failed` 标志,`None` 分支据其跳过记录;或在收到 Err 后直接终结流(记录与否二选一,但要确定化)。

### [MEDIUM] error.rs:34-43 — `From<TestkitError> for ProviderError` 对非 Inner 错误做有损降级

触发条件:任一非 `Inner` 变体(`ReplayExhausted`/`ReplayNoMatch`/`Io`/`Golden`)经此 From 转换。

后果:除 `Inner`(真实 provider 错误原样透传)外,其余全部折叠成 `ProviderError::Testkit(String)`,丢失了变体区分。例如"队列耗尽"与"精确匹配失败"在泛化入口里无法用类型区分,只能解析字符串错误消息。这是测试工具,严重度可接受,但若消费方依赖错误类型做分支(如区分"无匹配→重试"),会失效。

### [MEDIUM] replay.rs:210-219 — `ByToolName` 对"绑定工具取第一个名"有顺序耦合

触发条件:并发/多工具场景下,一个 `ReplayProvider` 实例通过 `bind_tools` 绑了多个 tool,`want` 只取 `.first()`(`:210`)。

后果:匹配键只取绑定工具列表的第一个工具名。若同一实例绑了多个工具(agent 级联绑定),后续工具名不会被扫描;同名工具多次绑定的第二次交换无法命中对应录播,退化为 FIFO 或 `ReplayNoMatch`。单工具分支(README 所述场景)没问题,但设计上掩盖了"按工具名路由只解决单工具对应"的限制——文档(README/注释)已部分说明,但 `want` 的 `.first()` 语义未在用户可见注释点明。

### [LOW] wire.rs:105 — accept 失败即终止整个 accept 循环

`let Ok((mut sock, _)) = listener.accept().await else { return; };` 任一临时 accept 错误(EINTR/连接洪峰)会让整个 mock server 的监听循环退出,后续请求全部得不到响应。测试用 mock 可接受,但在长时 wire 测试套件里一个尖峰即可让后续断言全 501。

### [LOW] wire.rs:151 — 未知状态码 reason 写死 "OK"

`_ => "OK"`,非 2xx 的自定义状态码(如 308、507)会返回 `HTTP/1.1 507 OK`,语义误导。仅影响原始报文观测,不影响解析。

## 4. 不合理 / 可改进

- **golden.rs:201-208 `replay_golden_from_file` 的 strategy 参数是伪参数/脚枪**:该函数无论传什么 strategy,`ReplayPredictor::predict`(golden.rs:181)只发 `[Message::human(input)]` 单条消息。若调用方想用 `Exact` 对齐多轮录播,单条 human 与多轮 `exchange.messages` 永远不匹配 → 必然 `ReplayNoMatch`,N2 离线冒烟被不必要地打断。文档说默认 `Fifo` 是预期模式,但对非默认 strategy 无任何拦截或提示。建议对非 `Fifo` 显式 warn 或直接不接受。
- **`scoring_exchanges`(golden.rs:124)与 `golden_dataset`(golden.rs:107)的口径不一致**:`golden_dataset`(裸函数)不过滤、首个不可评分交换即报错;只有 `*_from_file` 才过滤。调用方可混用两套口径,需要读注释才能分清。是一个易踩的 API 不一致(minor)。
- **wire.rs 的 `Pattern` 被标 `#[doc(hidden)]`(replay 侧无,** wire.rs:58)但仍是 `pub`,且 `WireRoutes` 公开别名里暴露了它。作为测试设施可接受,但文档态不完整。
- **recording.rs `RecordedExchange.tools` 用 `Option<Vec<ToolDefinition>>`(recording.rs:37)**:`#[serde(default, skip_serializing_if)]` 已做旧文件兼容,但 `bind_tools` 后 tools 必然 `Some`,而对象本身沿用旧格式语义,序列化模型显冗——可考虑 `#[serde(default)]` 的 `Vec`(空即无)。非功能问题。

## 5. 可加的新功能

- **完整消息签名法下的字段白名单/黑名单**:`Exact` 要求字段级全等(连运行时 `id` 都算漂移,replay.rs:48-49 已点明),但实际追踪常只关心 `content`/`message_type`。给 `Exact` 加"忽略字段集(如 `id`)"可大幅提升可用性。
- **`ReplayNoMatch` 差分提示**:报错时除 `left: usize` 外,给出与请求最接近的一条录播及其字段差异,调试字段漂移快很多。
- **录制侧指纹 / 重放资格检校**:`read_exchanges` 读取时校验每行走 JSON 结构(目前只要 `serde_json::from_str` 成功即可,recording.rs:54),可加最小 schema 校验(如 response.content 存在)防脏录播。
- **`ByToolName` 多工具匹配**:将 `.first()` 改为遍历所有绑定工具名尝试匹配,覆盖级联绑定的场景。

## 6. 对标主流

- Python `langchain` 的测试靠 `langchain_core` 的 mock model 与离线 fixtures(**自身工具链内嵌而非独立 crate**;本 crate 独立成 crate,隔离更干净)。
- Rust 生态无同体量等价物;本 crate 的 `Recorder/ReplayProvider` 等价于 JS `nock` + `VCR.py` 的"录制回放"思路,但额外支持 `ByToolName`/`Exact` 两种确定性路由,比 VCR 的单一 regex 匹配更贴合 agent-loop;`Exact`"无匹配即报错"比 nock 的宽松回退更不易掩盖回归,值得褒奖。
- `canon.rs` 的 json 顺序无关比较对标 `assert-json-diff`(支持 array 顺序保持),功能相当但实现更轻(无折叠数组/补丁 diff 输出)。缺失点是**:无结构化 diff 输出**,`assert_json_canonical` 只 panic 一条消息,调试大对象比对靠 `{:?}`。

## 7. 一句总评

骨架干净、`Exact` 路由原则正确,N2 闭环自洽;主要缺陷是 stream_chat 在 provider 中途出错时**不确定地写入截断录播**,违背"失败不记录"的声明契约,应在交付前修复。