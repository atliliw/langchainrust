# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.2]: https://github.com/atliliw/langchainrust/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/atliliw/langchainrust/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/atliliw/langchainrust/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/atliliw/langchainrust/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/atliliw/langchainrust/releases/tag/v0.1.1