# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **GraphRAG 社区摘要用实体 ID 而非 name** (`retrieval::graph_rag::community`): `summarize_community` 直接拼接 `r.source`/`r.target`(值为 `e_xxx` 形式的实体 ID),LLM 收到无意义 ID 导致社区摘要质量极差,直接拖垮 Global/Hybrid 查询。改为通过 `store.get_entity()` 查实体 name,与 `query.rs` 的 `format_relation` 逻辑一致
- **Deep Research 综合报告嵌入 JSON 字符串导致转义失败** (`agents::deep_research::synthesizer`): 要求 LLM 把完整 markdown 报告作为 JSON 字符串字段 `"report"` 输出,markdown 含 `\n`、`"`、`\` 需 JSON 转义,LLM 转义出错率高导致 `serde_json` 解析失败,整个综合步骤报错。改用分隔符格式 `<<<REPORT>>>...<<<END_REPORT>>><<<GAPS>>>[...]<<<END_GAPS>>>`,report 部分直接取原始文本无需转义;保留旧 JSON 格式作为 fallback 兼容
- **document_store 在 tokio runtime 内 panic** (`vector_stores::document_store`): `InMemoryDocumentStore` 和 `InMemoryChunkedDocumentStore` 内部用 `tokio::sync::RwLock`,但 `_blocking` 方法调 `blocking_read()`/`blocking_write()`,在 `#[tokio::test]` async 上下文里触发 `Cannot block the current thread from within a runtime` panic。改为 `std::sync::RwLock`,所有锁操作 `.read().unwrap()`/`.write().unwrap()`,同步异步上下文通用;锁持有时间短(无跨 await 持锁),不会死锁

## [0.5.0] - 2026-07-23

### Added
- **模型路由 + Fallback + 负载均衡 (`core::router_llm`)** (#2): `RouterLLM` 实现 `BaseChatModel`,在异构 provider 池上按策略选模型并在失败时 fallback
  - `RoutingStrategy`: `Fallback`(primary-first) / `RoundRobin` / `LeastLatency`(EMA 延迟统计) / `LowestCost` / `InputDirected`(闭包按输入选模型)
  - `RouterError` 统一错误,经内部 `ModelAdapter` 把各 provider 异构错误收敛为单一类型
  - `with_fallbacks(primary, fallbacks)` / `with_model` / `with_named_model` / `with_cost` builder
- **Agentic / Corrective RAG (`agents::crag`)** (#1): `CorrectiveRAGAgent` LangGraph 状态机: retrieve -> grade -> [rewrite+web|keep] -> generate,带幻觉检测
- **MCP 全协议 (`mcp`)** (#3): 补 resources / prompts / completion / elicitation / roots / sampling 六大原语,Client/Server 对应 handler,39 个单元测试
- **Code Interpreter 沙箱 (`tools::sandbox`)** (#4): `CodeSandbox` trait + `SandboxTool<BaseTool>` + `LocalSandbox`(子进程) + E2B/WASM 后端(feature gate)
- **OpenAI Responses API (`language_models::openai::responses`)** (#5): `ResponsesModel` 实现 `BaseChatModel`,走 `/v1/responses`,内置 web_search/file_search/code_interpreter/computer_use,流式事件解析
- **GraphRAG 知识图谱 RAG (`retrieval::graph_rag`)** (#6): LLM 抽实体+关系->建图->Label Propagation 社区检测+摘要->Global/Local/Hybrid 查询,无 petgraph 依赖
- **Anthropic extended thinking (`language_models::providers::anthropic`)** (#7): `ThinkingConfig` + `with_thinking(budget_tokens)`,thinking block 解析,`on_llm_thinking` 回调,`LLMResult.thinking_content`,流式 thinking
- **Deep Research Agent (`agents::deep_research`)** (#8): 多轮搜索(query 生成->并行检索->去重->子主题聚合->综合报告+引用),`ResearchReport` + `Citation`
- **Streaming Structured Output (`core::structured_output`)** (#9): `PartialJsonParser` 增量 JSON 解析 + `stream_structured_output<T>` 流式部分结构 + `StreamingStructuredOutputExt` trait
- **Adaptive RAG (`agents::adaptive_rag`)** (#10): LLM 路由判 `NoRetrieval`/`SingleSearch`/`MultiQuery`,复用现有 Retriever
- **Batch API (`core::batch`)** (#11): `BatchClient` submit/poll/results/cancel,OpenAI + Anthropic batch 端点,`submit_and_wait` 便捷方法
- **Agent Observability / 深度 Tracing (`callbacks::tracing`)** (#12): `Tracer` + `SpanGuard`(RAII) + `TracingBackend` trait + `InMemoryTracingBackend`/`ConsoleTracingBackend`/`OtelTracingBackend`,parent-child span tree

### Changed
- `LLMResult` 新增 `thinking_content: Option<String>` 字段(#[serde(default)] 向后兼容)
- `CallbackHandler` trait 新增 `on_llm_thinking` 默认方法
- `AnthropicChat` 新增 `with_thinking(budget_tokens)` builder
- Cargo.toml 新增 feature gates: `sandbox-e2b`, `sandbox-wasm`

### Fixed
- **安全**: PythonREPLTool 加危险 import 检查(os/subprocess/socket 等 17 模块);HTTPTool SSRF 改 async DNS 防止 rebinding;URLFetchTool 加内网 IP 过滤;SQLTool 阻止分号/注释/子查询绕过;Gemini API key 从 URL 移至 header(C1-C5)
- **Panic 修复**: choices[0] 改 `.first().ok_or()`(OpenAI+Ollama);from_env() 返回 Result 不再 expect panic(ResponsesConfig);Regex 改 LazyLock 编译一次;Mutex poison 改 `into_inner()` 恢复(C7-C11)
- **SSE 流式**: Ollama/Anthropic/Gemini 三个 provider 加跨 chunk buffer,不再丢 token;回调从 `drop()` 改 `.then()` async 执行;Gemini stream_chat 加回调;激活 responses.rs 死模块(H1-H7)
- **多轮 Function Calling**: Anthropic system 消息填入 top-level `system` 字段;Ollama AI 消息包含 tool_calls;Gemini tool_result 走 functionResponse;Anthropic tool_result 走 content block 格式(H42-H45)
- **并发安全**: langgraph/compiled 改 tokio::sync::RwLock;sessions/memory_store 改 tokio::sync::Mutex;mongo_memory 避免 blocking_write 死锁;HandoffManager 合并为单 Mutex;MCP Transport 加请求级互斥(C17-C19,C23,H9-H13)
- **数据正确性**: parent_id 分隔符从 `_` 改 `::`;错误传播替代 `.ok()` 静默吞掉;stream finalizer 重写为可靠机制;UTF-8 按字符边界切分(非字节);负分文档过滤;RRF 文档 ID 用内容 hash(C12-C16,H23-H27,H46)
- **Runnable::stream() 假流式**: OpenAI/Anthropic/Ollama 改为逐 token 发射 LLMResult(H4)
- **Batch API**: Anthropic 消息映射修正 — system 消息提取到 top-level `system` 字段,tool 消息走 tool_result 格式(H40)
- **RouterLLM**: Mutex poison 改 into_inner;RoundRobin 加除零检查;stream_chat 更新延迟统计;Arc 共享 messages 减少 clone(H33-H38)
- **JSON 修复**: repair_partial_json 正确跟踪字符串内花括号;处理转义引号;UTF-8 字符边界检查(C20-C21,M37)
- **其他**: cosine_similarity 用 epsilon 浮点零判断;不同长度向量返回错误;缓存过期条目清理;structured_output parse 支持 markdown 包裹;score 范围校验 0-1;A2A 错误响应修正;thiserror 替代手动 Error 实现;多处 Regex 改 LazyLock;vector store 操作优化(C22,M21-M23,M30-M34,M7-M8)

## [0.4.2] - 2026-07-22

### Added
- **共享数学工具 `core::math`**: 新增 `src/core/math.rs`,提取 `cosine_similarity` 共享实现(带 doctest + 单元测试),供 vector_stores / retrieval / embeddings / evaluation 等 12 处复用,去除各模块内联重复实现
- **Calculator 安全表达式求值**: `Calculator` 工具接入 `meval` crate(`meval::eval_str`),支持算术 / 幂 / 函数(sin/cos/tan/sqrt/log/exp/abs)/ 常量(pi/e),替代手写解析
- **HTTP 工具 URL 解析**: `HTTPTool` 接入 `url` crate,用 `url::Url::parse` 规范化 URL

### Changed
- **reqwest 0.11 -> 0.12**(breaking): 全量迁移 HTTP 客户端代码,涉及 providers / embeddings / tools / mcp / a2a / vector_stores 等模块
- **内部重构与去重**: chains(document_chains / conversation_chain / retrieval_qa / llm_chain / router_chain)、tools(calculator / http / url_fetch / python_repl)、embeddings(deepseek / qwen)、vector_stores(memory / file_store / chunked)、mcp/transport、a2a/server、pinecone 等模块代码整理,统一复用 `core::math::cosine_similarity`

### Fixed
- `MapRerankDocumentsChain::extract_score` 集成测试调用泛型 `M` 推断失败(`tests/unit/conversation_retrieval_chains.rs`),对齐源码写法显式指定类型

## [0.4.1] - 2026-07-20

### Added
- **Assistants requires_action 工具调度**: `OpenAIAssistant` 轮询遇 `requires_action` → 解析 `tool_calls` → 经 `ToolRegistry` 执行 → `submit_tool_outputs` → 继续轮询至 `completed`/`failed`/`cancelled`
- **A2A Agent 协议**: 新增 `src/a2a/` 模块
  - `AgentCard` / `A2ATask` / `A2AMessage` / `TaskStatus` / `A2ARequest` / `A2AResponse` / `A2AErrorData` 协议类型
  - `A2AServer`: handler 函数(`tasks/send`/`tasks/get`/`tasks/cancel`),可插入任意 HTTP 框架(axum/actix/warp),含 `RwLock<HashMap>` 内存 task persistence
  - `A2AClient`: reqwest HTTP 客户端,`get_agent_card()`/`send_task()`/`get_task()`/`cancel_task()`
- **with_structured_output**: `StructuredOutputExt` trait + 独立函数,按 provider 走 function calling 或 `JsonOutputParser` 降级,11 个测试
- **Chain 流式**: `BaseChain::stream()` 默认实现 + `LLMChain`/`ConversationChain` 覆写,逐 token 回调 `on_llm_new_token`
- **ContextWindow 长上下文管理**: `ContextWindow<M: BaseChatModel>`,两种策略:
  - `Strategy::Truncate`: 按 token 数截断旧消息
  - `Strategy::Summarize`: 超限时用 LLM 摘要压缩旧对话
  - `TiktokenCounter` 集成,18 个测试
- **FileVectorStore**: JSON 持久化向量存储,原子写入(tmp+rename),跨实例持久化,维度校验,`VectorStore` trait 完整实现
- **ComputerUseTool**: Anthropic computer use API 接入 + Native 截图/键盘/鼠标(feature gate `computer-use-native`)
- **DocxLoader**: ZIP 解压 + XML 解析 `<w:t>` 文本节点
- **WebScraperLoader**: 网页爬取,递归链接跟踪,同域过滤,可配最大深度/页面数
- **SitemapLoader**: 解析 sitemap.xml,批量爬取页面
- **LocalEmbeddings ort**: ONNX Runtime 神经网络嵌入(feature gate `local-embeddings`,依赖 `ort` + `ndarray`),替代原 bag-of-words 占位实现
- **wiremock 测试基础设施**: `wiremock` 作为 dev-dependency,mock 辅助函数,示范测试,默认测试不打真实网络
- **MSRV 声明**: `rust-version = "1.82"`,CI 矩阵含 1.82
- **criterion benchmark**: `benches/` 下 retrieval(6)/splitter(4)/embedding(4) 组基准
- **12+ 新 examples**: evaluation / mcp_server / guardrails / sessions / context_window / vectorstore_memory / semantic_splitter / file_vectorstore / otel / assistants / handoffs / plan_execute / token_counter

### Changed
- **unused import 修复**: `evaluation/pairwise.rs` 中 `async_trait` 移入 `#[cfg(test)]`
- **LocalEmbeddings**: 原 bag-of-words 实现保留为默认,ort 实现在 `local-embeddings` feature 下
- **VectorStoreProvider**: `provider.rs` 新增 `FileVectorStore` 工厂方法
- **lib.rs**: 导出 A2A 模块、`ContextWindow`、`FileVectorStore`、`StructuredOutputExt`、新 loaders 等公开 API

### Fixed
- **Examples 编译修复**: 全部 25 个 example 编译通过(修复 API 名不匹配/类型推断/未用导入/async 缺失等)
- **A2A server task persistence**: `tasks/get` 和 `tasks/cancel` 原本总是返回"not found",现已实现内存存储和状态查询/转换

## [0.4.0] - 2026-07-14

### Added
- **Evaluation 评估模块**: 10 种评测器(5 类),框架含 `Score` / `Example` / `Dataset` / `Evaluator` / `Predictor` / `EvalRunner` / `Report`
  - 字面: `ExactMatch` / `StringDistance`(Levenshtein 编辑距离归一)
  - 语义: `EmbeddingSimilarity` / `LLMAsJudge` / `PairwiseJudge`(交换 A/B 消位置偏差)
  - 规则: `ContainsKeyword` / `RegexMatch` / `LengthCheck`
  - 经典 NLP: `Bleu`(字符级分词 + 平滑)
  - RAG: `Faithfulness`(拆主张逐条验证抓幻觉,`join_all` 并发,`llm_split` / `empty_score` 可配)
- **MCP Server**: `MCPServer` 把本地 `BaseTool` 暴露为 MCP Server(`initialize` / `tools/list` / `tools/call`),供 Claude Desktop / Cursor 等 host 调用,与 `MCPClient` 对称
- **VectorStoreRetrieverMemory**: 向量检索记忆,每轮对话嵌入存向量库,按当前输入语义召回 top-k 历史
- **OpenAIAssistant**: 封装 OpenAI Assistants API(`Assistants` / `Threads` / `Run`,服务端会话状态)
- **SemanticSplitter**: 语义分块器,相邻句相似度骤降处断块,中英文分句
- **LocalEmbeddings**: 轻量本地嵌入(词频 hash + L2 归一,纯 Rust 无外部依赖)
- **OtelHandler**: OpenTelemetry callback handler,执行事件转 OTel span(feature `opentelemetry`)

### Changed
- **依赖**: 新增可选依赖 `opentelemetry` + feature flag `opentelemetry`(默认关闭,不影响默认编译)
- **导出**: `lib.rs` 导出 evaluation 模块、`MCPServer`、`OpenAIAssistant`、`VectorStoreRetrieverMemory`、`LocalEmbeddings`、`SemanticSplitter`、`OtelHandler`

## [0.3.0] - 2026-07-12

### Added
- **examples 目录**: 12 个可运行示例(basic / agent / rag / langgraph / memory / chains)
- **MCP 协议支持**: `MCPClient`(Stdio + SSE 传输,`tools/list` + `tools/call`,MCPTool -> BaseTool 适配)
- **多模态 vision**: `ImageContent` + `Message::human_with_image`(OpenAI / Ollama Vision 序列化)
- **Sessions 会话管理**: `SessionManager` + `SessionStore`(Memory)+ 多轮对话记忆
- **Token 计数器**: `TiktokenCounter` + `TokenTrackingLLM` + `ModelPricing`(成本估算)
- **Guardrails 安全护栏**: `InputGuardrail` / `OutputGuardrail` + `SensitiveInfoGuardrail` + `GuardedAgent`
- **Plan-Execute Agent**: `Planner` + `PlanExecuteAgent`(规划 - 执行 - 重规划)
- **Handoffs 多 Agent 交接**: `HandoffManager` + `HandoffTool`
- **Streaming Tool Calls**: `StreamingFunctionCallingAgent`(`invoke_stream`)
- **工具扩展**: `SQLTool`(只读 + 表白名单)+ `HTTPTool` + `FileTool`(沙箱 + 扩展名白名单)
- **PGVector 向量库**: `PGVectorStore`(feature `pgvector-storage`,需用户配置 sqlx/pgvector 依赖)
- **Pinecone 向量库**: `PineconeStore`(reqwest HTTP API)
- **HTML Loader**: `HTMLLoader`(regex 提取文本,去 script/style)

### Changed
- **OpenAIChat**: 加 `Clone` derive(支持 PlanExecuteAgent / Streaming 复用)
- **Message**: 加 `images` 字段(多模态)+ `additional_kwargs` 加 `serde(default)` 向后兼容
- **清理**: `compiled.rs` clippy(type_complexity / collapsible_match / unnecessary_lazy_evaluations)

## [0.2.20] - 2026-05-05

### Fixed
- **create_resume_execution**: 修复 strip `after_` 前缀问题

### Changed
- **文档**: 更新 HTML interrupt/checkpointer API

## [0.2.19] - 2026-05-05

### Added
- **Interrupt/Resume 支持**: LangGraph 中断/恢复执行
  - `last_checkpoint_state` 状态保存
  - `create_resume_execution` 从中断点恢复执行

### Changed
- **文档**: 更新 interrupt/resume API 文档(中英文)

## [0.2.18] - 2026-04-30

### Added
- **Output Parsers**: StrOutputParser + CommaSeparatedListOutputParser + JsonOutputParser + StructuredOutputParser + TypedOutputParser
- **Document Chains**: StuffDocumentsChain + RefineDocumentsChain + MapReduceDocumentsChain + MapRerankDocumentsChain
- **ConversationRetrievalChain**: 带记忆的检索增强对话
- **Google Gemini**: GeminiChat (native API)
- **ChromaDB**: 轻量级向量数据库 HTTP API
- **LLM Cache**: 内存缓存 + TTL
- **Redis/SQLite 存储**: RedisDocumentStore + SQLiteDocumentStore
- **Tools 扩展**: Wikipedia + DuckDuckGo + PythonREPL
- **FewShotPrompt + ExampleSelectors**: 少样本提示模板 + 示例选择器
- **LCEL 组合操作符**: RunnableSequence + RunnableParallel + RunnablePassthrough + RunnableLambda + BitOr trait
- **Qdrant**: `delete_by_metadata` 方法
- **MongoPersistentMemory**: 条件编译(仅在 `mongodb-persistence` feature 启用时可用)

## [0.2.17] - 2025-04-24

### Added
- **Memory 持久化**: 新增 PersistentMemory trait 和 MongoPersistentMemory 实现
  - `PersistentMemory` trait: 框架层持久化接口，支持 load_from_store/save_to_store/delete_session
  - `MongoPersistentMemory`: MongoDB 存储，组合 ConversationSummaryBufferMemory 压缩逻辑
  - `PersistenceConfig`: 配置 auto_save/auto_load/token_limit
  - `MemoryData`: 内存数据序列化结构
  - 框架负责压缩算法，业务层负责存储实现
- **ConversationSummaryBufferMemory**: 添加 `chat_memory_mut()` 方法支持可变访问

## [0.2.16] - 2025-04-24

### Fixed
- **BM25 分割算法**: 修复 Parent-Child 分割使用简单字符切分导致语义边界破坏的问题
  - `InMemoryChunkedDocumentStore`: 使用 `RecursiveCharacterSplitter` 替代 `chars[start..end]`
  - `MongoChunkedDocumentStore`: 同样修改，MongoDB 存储也使用语义分割
  - 分隔符优先级：段落 > 行 > 句号 > 空格 > 字符
  - 添加 chunk_overlap（默认 chunk_size / 10）避免边界信息丢失

### Added
- **文档**: `docs/bm25_split_fix.md` 详细说明分割算法修复

## [0.2.15] - 2025-04-23

### Fixed
- **MongoChunkedDocumentStore**: 修复 blocking 方法在 tokio runtime 内部的兼容性问题
  - 使用 `tokio::task::block_in_place` + `Handle::current().block_on` 替代创建新 runtime
  - 解决 "Cannot block the current thread from within a runtime" 错误

## [0.2.14] - 2025-04-23

### Changed
- **ChunkedDocumentStoreTrait**: 添加 blocking 方法支持
  - `add_parent_document_blocking`: 同步添加父文档
  - `get_parent_document_blocking`: 同步获取父文档
  - `get_chunk_blocking`: 同步获取 chunk
  - `blocking_get_chunks_for_parent`: 同步获取父文档的所有 chunks
- **MongoChunkedDocumentStore**: 实现 blocking 方法（使用 tokio runtime 桥接）
- **ChunkedBM25Retriever/ChunkedBM25Index**: 改为泛型结构，支持多种 DocumentStore 后端
  - 默认类型参数：`ChunkedBM25Retriever<S: ChunkedDocumentStoreTrait = ChunkedDocumentStore>`
  - 向后兼容：现有代码无需修改即可继续使用

### Fixed
- BM25 MongoDB 持久化支持：现在可以使用 `MongoChunkedDocumentStore` 作为 BM25 检索器的存储后端

## [0.2.13] - 2025-04-22

### Added
- **LLM Providers**: 统一的第三方 LLM 提供者支持
  - `DeepSeekChat`: DeepSeek API 支持
  - `MoonshotChat`: Moonshot (Kimi) API 支持
  - `QwenChat`: 阿里云通义千问 API 支持
  - `ZhipuChat`: 智谱 ChatGLM API 支持
  - `AnthropicChat`: Anthropic Claude API 支持
  - 所有 providers 使用 OpenAI 兼容接口或原生 API
- **Embeddings 扩展**: 新增向量生成服务
  - `DeepSeekEmbeddings`: DeepSeek 向量生成
  - `QwenEmbeddings`: 通义千问向量生成
- **Ollama 增强**: 多模态和工具调用改进
  - Vision 参数支持：`with_image()`, `with_images()`
  - 工具调用改进：更好的 function calling 支持
  - 配置增强：新增 `OllamaConfig` 配置项

### Changed
- **LangSmith Client**: `request_id` 追踪增强
  - 优化请求追踪链路
  - 支持多层级 run 追踪
- **Qdrant Vector Store**: 重构优化
  - 更好的错误处理
  - 改进的连接管理
- **LangGraph Compiled**: 状态管理改进
- **MultiQuery Retriever**: 错误处理优化

### Configuration
- **Cargo.toml**: demo 目录已 exclude (不上传 crates.io)

## [0.2.12] - 2025-04-19

### Documentation
- **Callbacks 文档**: LangSmith 追踪完整说明
- **README**: 更新多 Provider 支持列表

## [0.2.11] - 2025-04-17

### Added
- **Document Loaders**: 文档加载器系列
  - `TextLoader`: 纯文本加载
  - `JSONLoader`: JSON 文档加载
  - `MarkdownLoader`: Markdown 文档加载
  - `PDFLoader`: PDF 文档提取
  - `CSVLoader`: CSV 数据加载
- **MultiQuery Retriever**: 多查询扩展检索
  - 自动生成多个相关查询
  - 合并多查询结果
  - 提升检索召回率
- **HyDE (Hypothetical Document Embeddings)**: 假设文档嵌入
  - 基于问题生成假设答案
  - 使用假设答案检索相关文档
- **Reranking**: 重排序模块
  - 支持多种重排序策略
  - 提升检索精准度

## [0.2.6] - 2025-04-18

### Added
- **LangGraph**: 图状工作流框架
  - `StateGraph`: 状态图构建器
  - `CompiledGraph`: 编译后的可执行图
  - `GraphNode` trait + `SyncNode` + `AsyncNode`: 节点抽象
  - `GraphEdge` + `ConditionalEdge`: 边和条件路由
  - `StateSchema` trait + `AgentState`: 状态管理
  - `Reducer` trait + `AppendReducer`: 状态更新策略
- **LangGraph Checkpointer**: 执行状态持久化
  - `MemoryCheckpointer`: 内存持久化
  - `ThreadSafeMemoryCheckpointer`: 线程安全版本
  - `FileCheckpointer`: 文件持久化
- **LangGraph 可视化**: 图结构可视化输出
  - `visualize_ascii()`: ASCII 图形
  - `visualize_mermaid()`: Mermaid 图表格式
  - `visualize_json()`: JSON 结构输出
- **LangGraph Human-in-the-loop**: 人工干预机制
  - `interrupt_before`: 执行前中断
  - `interrupt_after`: 执行后中断
  - `resume()`: 从中断点恢复执行
- **LangGraph Graph 验证**: 图完整性验证
  - `validate_cycles()`: 死循环检测
  - `validate_unreachable_nodes()`: 孤立节点检测
  - `validate_duplicate_edges()`: 重复边检测
- **LangGraph Subgraph**: 子图嵌套支持
  - `SubgraphNode`: 子图节点封装
  - 状态映射器: 父子图状态转换
- **LangGraph Parallel**: 并行节点执行
  - `invoke_parallel()`: 并行执行多个节点
  - FanOut/FanIn 模式支持
- **LangGraph Persistence**: 图定义持久化
  - `GraphDefinition`: 图定义结构
  - `NodeRegistry`: 节点注册表
  - `save_to_file()` / `load_from_file()`: 序列化/反序列化

### Tests
- 新增 `tests/langgraph/` 目录 (10+ 测试文件)
- LangGraph 基础测试、条件边、状态管理
- 异步节点、Checkpointer、可视化测试
- Human-in-the-loop、Subgraph、Parallel 测试

### Documentation
- README.md 更新核心特性列表
- ROADMAP.md 添加 LangGraph 模块详情

## [0.2.5] - 2025-04-15

### Added
- **RetrievalQA**: 一站式检索问答 Chain
  - 自动检索相关文档（RAG 核心）
  - 组装 Prompt（上下文 + 问题）
  - LLM 基于上下文生成答案
  - `query()` 化接口，一行完成问答
  - `with_return_source_documents(true)` 返回来源文档
  - `with_prompt_template()` 自定义 Prompt
  - `with_k()` 配置检索数量
- **RouterChain**: 条件路由 Chain
  - 根据输入关键词自动路由到不同 Chain
  - `LLMRouterChain` 使用 LLM 智能判断路由
  - 支持默认 Chain（未匹配时使用）
- **ConversationChain**: 带记忆的对话 Chain
  - 自动保存和加载对话历史
  - 支持多轮对话记忆
  - `predict()` 简化接口
- **Memory 系统完善**: 完整的对话记忆管理
  - `ConversationBufferMemory`: 无压缩，保存全部对话历史
  - `ConversationBufferWindowMemory`: 窗口截断，只保留最近 k 轮
  - `ConversationSummaryMemory`: LLM 智能摘要，压缩旧对话
  - `ConversationSummaryBufferMemory`: 混合策略，摘要 + 最近对话
  - `ChatMessageHistory`: 底层消息存储容器
- **流式输出增强**: LLM stream_chat 完整实现
  - `stream_chat()`: 逐 token 实时输出
  - 打字机效果，用户感知延迟更低
  - 支持流式部分收集、中途停止

### Tests
- 新增 `tests/unit/memory.rs` (Memory 基础测试)
- 新增 `tests/unit/summary_buffer_memory.rs` (压缩触发测试)
- 新增 `tests/unit/llm_stream.rs` (流式输出测试)
- 新增 `tests/unit/retrieval_qa.rs` (RetrievalQA 测试)
- 新增 `tests/unit/router_chain.rs` (RouterChain 测试)

### Documentation
- USAGE.md 添加 Memory 详细说明
- USAGE.md 添加流式输出使用示例
- README.md 更新 Memory 特性列表

## [0.2.4] - 2025-04-13

### Added
- **FunctionCallingAgent**: 使用原生 Function Calling 的 Agent
  - 不依赖文本解析，直接处理 `tool_calls`
  - 类型安全：通过 JSON Schema 定义工具参数
  - 更可靠：利用模型原生支持，不依赖 Prompt Engineering
  - 更高效：Token 消耗更低
- **to_tool_definition()**: 将 BaseTool 转为 ToolDefinition 的转换函数
  - 自动从 `args_schema()` 生成 JSON Schema
  - 简化工具绑定流程
- **测试目录**: 新增 `tests/function_calling/` 专门用于 Function Calling 测试
  - 5 个测试用例覆盖单工具、多工具、系统提示词等场景
  - 对比测试：ReActAgent vs FunctionCallingAgent

### Changed
- **OpenAI 响应解析**: 修复 Function Calling 时 `content` 为 null 的解析错误
  - `OpenAIMessage.content` 改为 `Option<String>`
  - `OpenAIMessage.finish_reason` 改为 `Option<String>`
- **项目结构**: `agents/` 目录新增 `function_calling/` 子模块
- **导出**: 新增 `FunctionCallingAgent` 和 `to_tool_definition` 公开导出

### Documentation
- README 添加 FunctionCallingAgent 使用示例
- 新增内部文档解释两种 Agent 的区别和适用场景

## [0.2.3] - 2025-04-11

### Changed
- 移除源码中的 Python 参考注释，保持代码整洁

## [0.2.2] - 2025-04-11

### Added
- **回调系统 (Callback System)**: 完整的执行追踪和监控框架
  - `CallbackHandler` trait: 定义 LLM/Chain/Tool/Retriever 回调接口
  - `CallbackManager`: 多处理器管理和分发
  - `StdOutHandler`: 控制台日志输出
  - `LangSmithHandler`: LangSmith 平台追踪集成
  - `RunTree`: 运行层次结构和追踪 ID 管理
  - `RunType`: LLM/Chain/Tool/Retriever 类型枚举
- **工具回调 (Tool Callbacks)**: 工具执行全生命周期追踪
  - `on_tool_start`: 工具开始时记录输入
  - `on_tool_end`: 工具完成时记录输出
  - `on_tool_error`: 工具失败时记录错误
- **Tool Calling 增强**: OpenAI function calling 完整支持
  - `bind_tools()`: LLM 绑定工具定义
  - `ToolDefinition`: 工具定义结构 (name, description, parameters)
  - `ToolCall` / `ToolCallResult`: 工具调用解析
  - `with_structured_output<T>()`: 结构化输出方法
  - `StructuredOutput<T>`: 泛型结构化输出包装
  - `StructuredTool<T>`: 泛型结构化工具包装
- **Runnable 接口**: LCEL 基础 trait
  - `Runnable<Input, Output>`: 统一执行接口
  - `RunnableConfig`: 配置支持回调、标签、元数据
  - `invoke()` / `batch()` 方法

### Changed
- `OpenAIChat` 实现 `Runnable<Vec<Message>, String>` trait
- `RunnableConfig` 支持回调系统集成 (`with_callbacks()`)
- AgentExecutor 自动触发工具回调

### Documentation
- 新增 `docs/internal/ROADMAP.md`: 功能开发路线图
- 新增 `docs/internal/FEATURE_PLAN.md`: 详细实现计划
- README 更新回调系统使用示例

## [0.2.1] - 2025-04-09

### Changed
- **Project Structure Cleanup**: Reorganized documentation and tests
  - Moved internal docs to `docs/internal/` (not published)
  - Moved test files from root to `tests/` directory
  - Removed `examples/` directory (examples now in tests)
- **Git Configuration**: Updated `.gitignore` to exclude AI tool directories
  - Added `.sisyphus/` to gitignore
  - Added `docs/internal/` to gitignore
- **Documentation**: Updated README with complete RAG + LLM examples

### Removed
- Removed `examples/` directory and Cargo.toml example configurations
- Removed internal documentation from git tracking

## [0.2.0] - 2025-04-07

### Added
- **Complete RAG + LLM Integration**: Full retrieval-augmented generation pipeline
  - `OpenAIEmbeddings`: Real AI-powered vector generation
  - Automatic vector generation in `add_documents()`
  - Batch embedding API calls for efficiency
- **Qdrant Vector Database Support**: Production-ready vector storage
  - `QdrantVectorStore`: Full integration with Qdrant
  - `QdrantConfig`: Configurable vector size, distance metrics
  - Feature-gated: `qdrant-integration` feature
- **Comprehensive RAG Test Suite**: 6 complete tests with real API calls
  - `test_inmemory_embeddings_real`
  - `test_rag_inmemory_full_pipeline`
  - `test_rag_with_document_splitting`
  - `test_rag_qdrant_full_pipeline`
  - `test_compare_memory_vs_qdrant`
  - `test_rag_multi_turn_conversation`

### Changed
- **Vector Generation**: Now automatic via `retriever.add_documents(docs)`
- **README**: Added complete RAG + LLM examples with real embeddings

## [0.1.2] - 2024-04-07

### Added
- **Prompts Module**: New `PromptTemplate` and `ChatPromptTemplate` for flexible prompt engineering
  - `PromptTemplate`: String template with `{variable}` placeholders
  - `ChatPromptTemplate`: Multi-message template for chat scenarios
- **OpenAIError Export**: `OpenAIError` is now publicly accessible from `langchainrust::language_models::openai`
- **Example Configuration**: All examples configured in `Cargo.toml` for easy running
- **LICENSE File**: MIT License for open source distribution

### Changed
- **Refactored Examples**: All examples updated to match current API
  - Fixed `chain_pipeline.rs` to use proper LLMChain API
  - Fixed `memory_conversation.rs` to use `ChatMessageHistory`
  - Fixed `full_pipeline.rs` to work with current components
  - Removed unused imports in `multi_tool_agent.rs` and `rag_demo.rs`
- **Removed Reference Comments**: Cleaned up "参考 Python 版本" comments from all source files
- **Improved Documentation**: 
  - Rewritten README with bilingual support (English/Chinese)
  - Updated examples/README with clearer structure
  - Added comprehensive API usage examples

### Fixed
- All examples now compile and run successfully
- Proper trait imports (`BaseChain`, `BaseMemory`) in examples
- Type annotation issues resolved in chain examples

### Documentation
- Bilingual README (English + Chinese)
- Improved code examples with error handling
- Added configuration tables and feature descriptions
- Cleaner project structure documentation

## [0.1.1] - 2024-03-XX

### Added
- Initial release with core features
- LLM support (OpenAI compatible)
- ReActAgent with tool calling
- Memory management
- Chains (LLMChain, SequentialChain)
- RAG components
- Built-in tools (Calculator, DateTime, Math, URLFetch)

[0.5.0]: https://github.com/atliliw/langchainrust/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/atliliw/langchainrust/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/atliliw/langchainrust/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/atliliw/langchainrust/compare/v0.3.0...v0.4.0
[0.2.14]: https://github.com/atliliw/langchainrust/compare/v0.2.13...v0.2.14
[0.2.13]: https://github.com/atliliw/langchainrust/compare/v0.2.12...v0.2.13
[0.2.12]: https://github.com/atliliw/langchainrust/compare/v0.2.11...v0.2.12
[0.2.11]: https://github.com/atliliw/langchainrust/compare/v0.2.6...v0.2.11
[0.2.6]: https://github.com/atliliw/langchainrust/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/atliliw/langchainrust/compare/v0.2.4...v0.2.5
[0.2.3]: https://github.com/atliliw/langchainrust/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/atliliw/langchainrust/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/atliliw/langchainrust/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/atliliw/langchainrust/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/atliliw/langchainrust/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/atliliw/langchainrust/releases/tag/v0.1.1