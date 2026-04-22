# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.15]: https://github.com/atliliw/langchainrust/compare/v0.2.14...v0.2.15
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