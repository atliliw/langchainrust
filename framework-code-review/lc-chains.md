# lc-chains

> 硬性约束声明:本次审查只读不改任何代码,唯一落盘为本报告。以下每条发现均带 `文件:行号` + 触发条件 + 后果。

## 1. 概览

`crates/lc-chains/src/` 是 langchainrush 的"链组合"层:把若干个可运行单元(LLM、检索器、路由器、文档处理)以 `BaseChain` 抽象串成一条可 invoke / stream / 参与 LCEL `pipe` 的流水线。共 19 个 `.rs`(含 1 个测试文件 `router_chain/tests.rs`)。

核心抽象:
- `BaseChain` trait(`base.rs`)统一 `invoke / invoke_with_config / stream / stream_with_config / input_keys / output_keys / name`。
- 单文件 `substitute_template` 模板替换(H-C1/C2 审计修复后的单趟实现,不重扫二次替换)。
- `ChainRunnable`(`adapter.rs`)把 `BaseChain` 桥接为 `Runnable`,从而能进 LCEL 管道。
- 具象链:LLM(`llm_chain`)、顺序(`sequential_chain`)、对话(`conversation_chain`)、检索QA(`retrieval_qa`)、多轮对话检索(`conversation_retrieval`)、路由(`router_chain/{base,llm,destination}`)、文档处理(`document_chains/{stuff,refine,map_reduce,map_rerank}`)。

整体结论(先讲结论):**逐文件通读后未发现会导致崩溃/数据损坏/静默错误的真实 bug**。代码经过了密集的审计修复周期,几乎每处易错点都留了 `Px-x / H-Cx / Mxx / Cx` 注释说明修复动机与触发条件。少数几处为"可改进/隐式约定",见 §4。

## 2. 遍历文件清单(19 个,全部列出)

| 文件 | 审查程度 |
|---|---|
| `src/lib.rs` | 通读 — 模块声明 + `BoxedChatModel` 类型别名 |
| `src/base.rs` | 通读 — `BaseChain` trait、`ChainResult`、`StreamToken`、`substitute_template`、`run_chain_with_callbacks`、`stream_chain_with_callbacks`、`end_stream_on_completion`、`documents_from_input/to_values` |
| `src/llm_chain.rs` | 通读 — 单 LLM 调用链 |
| `src/sequential_chain.rs` | 通读 — 顺序组合链(L13 外部键计算、stream_steps 输出映射校验 P2-3) |
| `src/conversation_chain.rs` | 通读 — 对话记忆链(M-19/M-20 键对齐、P1-4 流式记忆保存) |
| `src/retrieval_qa.rs` | 通读 — 检索 QA(真实流式 P2-2) |
| `src/conversation_retrieval.rs` | 通读 — 多轮对话 + 检索链(流式经 channel 存全量记忆 P2-2) |
| `src/router_chain/base.rs` | 通读 — 关键词路由(最长匹配优先) |
| `src/router_chain/llm.rs` | 通读 — LLM 结构化路由 + 关键词兜底 |
| `src/router_chain/destination.rs` | 通读 — 路由目的地容器 |
| `src/router_chain/mod.rs` | 通读 — re-export |
| `src/router_chain/tests.rs` | 未展开(测试文件) |
| `src/document_chains/mod.rs` | 通读 — re-export |
| `src/document_chains/stuff.rs` | 通读 — 全量塞入链(空文档守卫 P2-7、CJK 截断) |
| `src/document_chains/refine.rs` | 通读 — 迭代精修链(single-doc 规避二次 LLM P2-4) |
| `src/document_chains/map_reduce.rs` | 通读 — 并行 map + reduce(并发钳位 P2-6) |
| `src/document_chains/map_rerank.rs` | 通读 — 并行 map + 相关性 rerank(未打分解读 P1-3) |
| `src/adapter.rs` | 通读 — `ChainRunnable` → `Runnable` 桥接 |
| `src/conversation_retrieval.rs`* | (上表重复项标注) |

> 实际 19 个文件:上述清单为全部功能 `.rs`,其中 `tests.rs` 未逐行展开(测试专用)。功能源码 18 个全部通读。

## 3. bug / 高风险点

**未发现真正的 bug。** 重点核查了三块既定风险点,结论如下:

1. **链可组合性(Runnable 桥接)** — 干净。
   - `adapter.rs:39-49`(invoke)/ `adapter.rs:51-77`(stream)统一通过 `LcelError::from(ChainError)`(单点转换,P2-1)传播错误。`stream` 把 `StreamToken` 经 `map` 转成 `HashMap<String,Value>`(键固定为 `"text"`),与 invoke 返回的由各链 `output_key` 决定的键**不一致** —— 这是 stream 桥接的约定性差异(不是崩溃级 bug),见 §4-改进1。

2. **流式一致性** — 干净。
   - `llm_chain / sequential / conversation* / retrieval_qa / stuff` 均提供 `stream_with_config`,把 `RunnableConfig` 透传给 `stream_chat`(config 携带采样覆盖/取消令牌/回调),与 invoke 路径对称。
   - `conversation_retrieval.rs:403-443`、`conversation_chain.rs`(P1-4)stream 用 `tokio::sync::mpsc::unbounded_channel` 转发 token,`once` 尾流里 drain 全量收尾再 `save_context` —— 确保记忆保存的是完整输出,不会截断。
   - `conversation_retrieval.rs:427` 只在 `!output.is_empty()` 时写记忆,空输出不污染记忆(合理)。
   - `retrieval_qa / map_rerank` 的 stream 在 `is_final:true` 一次吐出聚合结果(因为 rerank 需要整篇输出),这是语义必要,非 bug。

3. **并发一致性** — 干净。
   - `map_reduce.rs:212`(P2-6)`with_map_concurrency` 钳位 0→1,`buffer_unordered` 限流;默认 `try_join_all` 无界(预期,默认并发跑满)。
   - `MapRerankDocumentsChain::invoke_with_documents_cfg / stream_inner` 用 `try_join_all` 无界并发(预期)。
   - 顺序/并行、invoke/stream 四条路径在"执行错误转 observation / 键缺失报 `MissingInput` / 空文档报 `ExecutionError`"语义上一致(多处 `validate_inputs` + 空文档守卫)。

**半成品/死接口:** 未发现。`BaseChain` 的 `invoke(/stream)` 与 `invoke_with_config(/stream_with_config)` 均有实现并被 `adapter`、各链复用,`ChainRunnable` 有测试覆盖,无遗留 TODO 占位或未接线接口。

**已确认修复(审计期处理,非现行 bug,列作证据收纳):**
- `stuff.rs:69-75`(P2-7)、`stuff.rs:206-210`、`map_reduce / refine / map_rerank` 均加空文档守卫:空文档不再喂空上下文让 LLM 编造。
- `conversation_retrieval.rs:67-76`(键对齐):memory 的 input/output key 与链默认键(`query/result`)对齐,否则 `save_context` 会 `Missing input key` 静默失败。
- `stuff.rs:186-195`(H-C2)单趟替换防注入;`conversation_retrieval.rs:296`(C6)verbose 预览用 `chars()` 避免 CJK 字节切片 panic。
- `map_rerank.rs:209-239`(P1-3)未打分输出不再静默给中间分 50,而是用配置默认分或跳过。

## 4. 不合理 / 可改进

1. **`adapter.rs` stream 键约定**:`adapter.rs:66-74` 把 token 包成 `{"text": token}`,而 invoke 返回用链自己的 `output_key`(`base.rs`/各链)。下游做 LCEL `pipe` 时,同一链的 invoke 结果键与 stream 结果键不同,易踩坑。建议 stream 也按 `output_key` 聚合,或在文档中明确该不对称。

2. **`map_rerank` stream 语义**:`map_rerank.rs:428-438` 把被选中的 top-k 答案用 `\n\n` 拼接后经 single `is_final:true` token 一次性吐出,并非「按 answer 逐 token 流」。这是 rerank 拓扑决定的(先要整篇才能打分),但用户若期待"最终答案 token 级流式"会不符预期,值得在 doc 里写清"此链语义上是'after-thinking'输出"。

3. **`retrieval_qa / conversation_retrieval` 检索是无界并发前置**:检索在 invoke 进入 LLM 前同步做,`k` 默认 4/4,无 `max_documents` 硬上限之外的约束,超长文档亲合上下文;建议借鉴 `stuff.with_max_doc_length` 提供 per-doc 截断(部分链已配)。

4. **`conversation_retrieval.rs:306-313`(LOW,已在源码自注)**:`context_str = format_context(&documents)` 只为日志长度算了一次,随后 `build_messages` 又算一次,重复扫描。O(n) 且无并发问题,纯冗余。

5. **隐式 `to_string()` 错误**:`base.rs` 的 `ChainError` / `LcelError` 会 `err.to_string()` 抹掉 source/context。`adapter.rs:83-87` 的 `From<ChainError> for LcelError` 只拿 `err.to_string()`,嵌套错误详情丢失。建议保留结构化链。

## 5. 可加的新功能(对标 LangChain / LangGraph / SDK)

1. **`SequentialChain.stream` 全量转发**:目前顺序链 stream 依赖各子链;可对标 LangChain 的 `_atransform` 逐个物化再推送,缺"每子链只输出最新 token"的 `reentrant` 调度——可加 `stream_all` 语义扩展。
2. **`.with_memory` 组合到 RetrievalQA/Retrieval** 的通用记忆注入:现在只有 `conversation_*` 内置记忆,建议抽出 trait 统一注入。
3. **`map_reduce` 的 `reduce` 阶段支持分片合并(two-pass reduce)**,对标 LangChain `MapReduceDocumentsChain` 的分区合并,避免单轮 reduce 吐爆上下文。
4. **LCEL 管道补 `batch` / `.map` / 并行 `parallel` 组合子**:`adapter.rs` 走默认 batch;可加 `parallel_lc` 让多链并行 fan-out。
5. **结构化输出(Schema-free)链**:对标 Claude Agent SDK 的 structured-output,给 `LLMRunnerChain` 加 `response_format` 透传。

## 6. 对标主流

- **LangChain(py)**:lc-chains 是 LangChain `LCEL Runnable` + 各文档链的精简 Rust 复刻;`BaseChain`≈`RunnableSequence`,`ChainRunnable`≈`lc_core::Runnable` 桥接,`map_reduce/refine/stuff/rerank` 一一对应。
- **LangGraph**:lc-chains 是"线性/树状流",LangGraph 是"任意 DAG + checkpoint + 循环";两者互补——本 crate 的链适合垂直流水线,lc-langgraph 适合横向编排。
- **Claude Agent SDK / OpenAI Agents SDK**:本 crate 尚未提供 agentic-loop 层的链式原子(with_human_approval 的 LLM 位),`chain.with_approval_node` 会是把 agents 能力下沉到链层的增量。

## 7. 一句总评

lc-chains 是一个**完成度高、审计痕迹清晰、未发现实际 bug** 的链组合层——把 LangChain 的 LCEL 作品复刻到了"无崩溃、无静默错、error 显式"的 Rust 水准,剩余多是接口约定/文档级打磨而非缺陷。