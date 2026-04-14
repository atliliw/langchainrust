# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.6] - 2025-04-14

### Added
- **RetrievalQA**: 一站式检索问答 Chain
  - 自动检索相关文档（RAG 核心）
  - 组装 Prompt（上下文 + 问题）
  - LLM 基于上下文生成答案
  - `query()` 简化接口，一行完成问答
  - `with_return_source_documents(true)` 返回来源文档
  - `with_prompt_template()` 自定义 Prompt
  - `with_k()` 配置检索数量
- **ConversationSummaryBufferMemory**: 智能摘要 + 完整对话
  - 保留最近对话完整内容（确保流畅性）
  - 对旧对话进行摘要（节省 token）
  - `max_token_limit` 触发摘要机制
  - 平衡效率和对话质量
- **RouterChain**: 条件路由 Chain（已完成）
  - 根据输入关键词自动路由到不同 Chain
  - `LLMRouterChain` 使用 LLM 智能判断路由
  - 支持默认 Chain（未匹配时使用）

### Changed
- Chain 类型完成度提升到 40%（新增 RetrievalQA）
- Memory 类型完成度提升到 80%（新增 SummaryBuffer）

### Tests
- 新增 `tests/unit/retrieval_qa.rs` (4 个 LLM 测试)
- 新增 `tests/unit/summary_buffer_memory.rs` (5 个 LLM 测试)
- RetrievalQA 5 个单元测试
- SummaryBufferMemory 7 个单元测试

### Added
- **RouterChain**: 条件路由 Chain
  - 根据输入关键词自动路由到不同 Chain
  - `add_route_with_keywords()` 配置关键词匹配
  - `LLMRouterChain` 使用 LLM 智能判断路由
  - 支持默认 Chain（未匹配时使用）
  - `with_verbose()` 打印路由详情
- **ConversationChain**: 带记忆的对话 Chain
  - 自动保存和加载对话历史
  - 支持多轮对话记忆
  - `predict()` 简化接口，直接传入字符串
  - `clear_memory()` 清空记忆方法
  - 支持系统提示词配置
  - 支持自定义输入/输出/记忆键名
  - `ConversationChainBuilder` 方便构建
- **ConversationSummaryMemory**: 智能摘要记忆
  - 使用 LLM 自动摘要对话历史
  - 解决长对话 token 爆炸问题
  - 每轮对话后更新摘要
  - 支持自定义摘要提示词
  - 可配置返回消息或字符串格式
- **并行工具调用**: Agent 支持一次调用多个工具并并行执行
  - `AgentOutput::Actions(Vec<AgentAction>)` 新枚举变体
  - `execute_tools_parallel()` 并行执行方法
- **FileCallbackHandler**: 文件日志回调处理器
  - 支持 JSON 和纯文本格式
  - `LogFormat` 枚举选择日志格式
- **Runnable::stream()**: 流式处理默认实现
  - 所有 Runnable 自动获得 stream 能力
  - 将 invoke 结果包装为单元素流

### Changed
- **AgentOutput**: 新增 `Actions` 变体和 `actions()`、`is_action()` 方法
- **FunctionCallingAgent**: 支持解析多个 tool_calls

### Tests
- 新增 `tests/unit/parallel_tool_calls.rs`
- 新增 `tests/unit/file_handler.rs`
- 新增 `tests/unit/runnable_stream.rs`
- 新增 `tests/unit/conversation_chain.rs` (14 个测试)
- 新增 `tests/unit/summary_memory.rs` (4 个 LLM 测试)
- 新增 `tests/unit/router_chain.rs` (3 个 LLM 测试)
- RouterChain 7 个单元测试

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

[0.2.4]: https://github.com/atliliw/langchainrust/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/atliliw/langchainrust/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/atliliw/langchainrust/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/atliliw/langchainrust/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/atliliw/langchainrust/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/atliliw/langchainrust/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/atliliw/langchainrust/releases/tag/v0.1.1