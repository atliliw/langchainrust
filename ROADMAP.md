# LangChainRust 开发路线图

> 版本: 0.2.6 | 更新日期: 2025-04-18

---

## 项目概述

LangChainRust 是一个用 Rust 实现的 LLM 应用框架，提供 Agent、Chain、Memory、RAG、LangGraph 等核心组件。

| 统计 | 数值 |
|------|------|
| 源码文件 | 88 个 |
| 测试文件 | 54 个 |
| 核心模块 | 14 个 |
| crates.io 版本 | 0.2.6 |

---

## 模块状态

| 模块 | 文件数 | 核心组件 | 状态 |
|------|--------|----------|------|
| **core** | 13 | Runnable、BaseTool、ToolDefinition、StructuredOutput | ✅ 完成 |
| **language_models** | 8 | OpenAIChat、OllamaChat、BaseChatModel | ✅ 完成 |
| **agents** | 8 | ReActAgent、FunctionCallingAgent、AgentExecutor | ✅ 完成 |
| **memory** | 6 | Buffer、Window、Summary、SummaryBuffer | ✅ 完成 |
| **chains** | 7 | LLMChain、SequentialChain、RouterChain、RetrievalQA | ✅ 完成 |
| **langgraph** | 10 | StateGraph、Subgraph、Parallel、Checkpointer | ✅ 完成 |
| **retrieval** | 6 | TextSplitter、PDFLoader、CSVLoader | ✅ 完成 |
| **embeddings** | 3 | OpenAIEmbeddings、MockEmbeddings | ✅ 完成 |
| **vector_stores** | 5 | InMemoryVectorStore、QdrantVectorStore | ✅ 完成 |
| **callbacks** | 9 | LangSmithHandler、RunTree、CallbackManager | ✅ 完成 |
| **prompts** | 3 | PromptTemplate、ChatPromptTemplate | ✅ 完成 |
| **tools** | 5 | Calculator、DateTimeTool、URLFetchTool | ✅ 完成 |
| **schema** | 3 | Message、MessageType | ✅ 完成 |

---

## 已完成功能 (v0.2.6)

### LLM 支持
- ✅ OpenAI Chat API（含流式输出、Function Calling）
- ✅ Ollama Chat API（本地模型支持）
- ✅ SSE 流式解析器
- ✅ Tool Binding (bind_tools)
- ✅ 结构化输出 (StructuredOutput)

### Agent 框架
- ✅ ReActAgent（文本解析，兼容不支持 FC 的模型）
- ✅ FunctionCallingAgent（原生 Function Calling）
- ✅ AgentExecutor（执行器，含迭代限制）
- ✅ 工具调用完整流程

### Memory 系统
- ✅ ConversationBufferMemory（完整历史）
- ✅ ConversationBufferWindowMemory（窗口截断）
- ✅ ConversationSummaryMemory（LLM 摘要）
- ✅ ConversationSummaryBufferMemory（混合策略）
- ✅ ChatMessageHistory（消息存储）

### Chain 工作流
- ✅ LLMChain（单步链）
- ✅ SequentialChain（顺序链）
- ✅ ConversationChain（对话链）
- ✅ RouterChain（路由链）
- ✅ RetrievalQA（检索问答）

### LangGraph 图工作流
- ✅ StateGraph（状态图构建）
- ✅ CompiledGraph（编译执行）
- ✅ 条件边路由（ConditionalEdge）
- ✅ Human-in-the-loop（人工干预）
- ✅ Subgraph（子图嵌套）
- ✅ Parallel（并行执行）
- ✅ Checkpointer（状态持久化）
- ✅ 可视化输出（ASCII/Mermaid/JSON）
- ✅ Graph 验证（死循环/孤立节点检测）

### RAG 检索增强
- ✅ TextSplitter（文本分割）
- ✅ RecursiveCharacterSplitter（递归分割）
- ✅ PDFLoader（PDF 文档加载）
- ✅ CSVLoader（CSV 文档加载）
- ✅ SimilarityRetriever（语义检索）
- ✅ Document 数据结构

### Embeddings 嵌入
- ✅ OpenAIEmbeddings（OpenAI 嵌入）
- ✅ MockEmbeddings（测试用嵌入）
- ✅ cosine_similarity（相似度计算）

### Vector Stores 向量存储
- ✅ InMemoryVectorStore（内存存储）
- ✅ QdrantVectorStore（Qdrant 集成，需 feature）
- ✅ VectorStoreBuilder（构建器模式）

### Tools 工具
- ✅ Calculator（计算器）
- ✅ DateTimeTool（日期时间）
- ✅ SimpleMathTool（数学运算）
- ✅ URLFetchTool（URL 抓取）
- ✅ to_tool_definition（工具定义转换）

### Callbacks 回调
- ✅ StdOutHandler（标准输出）
- ✅ LangSmithHandler（LangSmith 集成）
- ✅ FileCallbackHandler（文件日志）
- ✅ RunTree（运行追踪）

---

## 待完成功能

### 🔴 高优先级（核心缺失）

| 功能 | 文件位置 | 问题类型 | 预估时间 |
|------|----------|----------|----------|
| OpenAI stream_chat 实现 | `src/language_models/openai/chat.rs:182` | unimplemented! | 2 小时 |
| Ollama stream_chat 实现 | `src/language_models/ollama/chat.rs:202` | unimplemented! | 2 小时 |
| MemoryCheckpointer load/delete | `src/langgraph/checkpointer.rs:66,74` | unimplemented! | 1 小时 |
| Qdrant 实际调用 | `src/vector_stores/qdrant_impl.rs:38,86,122` | TODO 占位符 | 3 小时 |

### 🟡 中优先级（LLM 扩展）

| 功能 | 说明 | 预估时间 |
|------|------|----------|
| Claude API 支持 | Anthropic Claude 模型集成 | 2-3 天 |
| Gemini API 支持 | Google Gemini 模型集成 | 2-3 天 |
| Token 精确计数 | 使用 tiktoken 或官方方案 | 1 天 |
| 更多本地模型 | llama.cpp、vLLM 等 | 2-3 天 |

### 🟡 中优先级（RAG 扩展）

| 功能 | 说明 | 预估时间 |
|------|------|----------|
| Markdown Loader | .md 文档加载 | 2 小时 |
| JSON Loader | .json 文档加载 | 2 小时 |
| HTML Loader | .html 网页加载 | 2 小时 |
| Pinecone VectorStore | Pinecone 云向量库 | 2-3 天 |
| Weaviate VectorStore | Weaviate 向量库 | 2-3 天 |
| Milvus VectorStore | Milvus 向量库 | 2-3 天 |
| Reranker | 重排序支持 | 1-2 天 |
| 混合检索 | BM25 + 语义检索 | 2-3 天 |

### 🟢 低优先级（增强功能）

| 功能 | 说明 | 预估时间 |
|------|------|----------|
| Plan-and-Execute Agent | 规划型 Agent | 3-4 天 |
| Multi-Agent 协作 | 多 Agent 系统 | 5-7 天 |
| Agent 工具动态加载 | 运行时添加工具 | 1 天 |
| LCEL 完整实现 | LangChain Expression Language | 3-5 天 |
| TransformChain | 数据转换链 | 1 天 |
| BranchChain | 分支链 | 1 天 |

### 🟢 低优先级（优化完善）

| 功能 | 文件位置 | 说明 |
|------|----------|------|
| LangSmith multipart upload | `src/callbacks/langsmith_client.rs:197` | 性能优化 |
| 清理 unused imports | 多个文件 | 消除 14 warnings |
| API 导出补充 | `src/lib.rs` | LangGraph 导出完善 |
| 减少 #[ignore] 测试 | `tests/` | 添加 mock 测试 |

---

## 版本规划

### v0.2.6 (当前版本) ✅

**LangGraph 完整实现**
- StateGraph、条件边、Human-in-the-loop
- Subgraph、Parallel、Checkpointer
- 可视化输出、Graph 验证
- MongoDB Persistence（可选）

### v0.3.0 (下一版本) 📋

**LLM 扩展**
- Claude API 支持
- Gemini API 支持
- OpenAI/Ollama stream_chat 完善
- Token 精确计数

### v0.3.1 📋

**RAG 增强**
- Markdown/JSON/HTML Loader
- Qdrant 实际调用完善
- Pinecone 或 Weaviate 支持

### v0.4.0 📋

**进阶 Agent**
- Plan-and-Execute Agent
- Multi-Agent 协作框架
- Human-in-the-loop 增强

---

## 测试状态

| 测试类型 | 文件数 | #[ignore] 数量 | 说明 |
|----------|--------|----------------|------|
| langgraph | 17 | 5 | 需要 OPENAI_API_KEY |
| unit | 18 | 0 | 纯单元测试 |
| integration | 14 | 28 | 需要 API Key/外部服务 |
| function_calling | 1 | 5 | 需要 API Key |
| e2e | 1 | 2 | 需要 API Key |
| **总计** | **54** | **~40** | 大部分需要环境变量 |

**改进方向**：
- 添加更多 mock 测试，减少 #[ignore]
- CI 环境变量配置指南
- 测试覆盖率统计

---

## 依赖关系

```
core (基础层)
  ├── Runnable trait
  ├── BaseTool / ToolDefinition
  └── StructuredOutput
      │
      ▼
language_models (LLM 实现)
  ├── OpenAIChat → core/BaseChatModel
  ├── OllamaChat → core/BaseChatModel
  └── bind_tools → core/ToolDefinition
      │
      ▼
agents (Agent 框架)
  ├── ReActAgent → language_models + tools
  ├── FunctionCallingAgent → language_models + tools
  └── AgentExecutor → agents + memory
      │
      ▼
chains (链式调用)
  ├── LLMChain → language_models + prompts
  ├── SequentialChain → chains
  ├── RetrievalQA → retrieval + chains
      │
      ▼
langgraph (图工作流)
  ├── StateGraph → core/Runnable
  ├── Checkpointer → persistence
  ├── Subgraph → langgraph
      │
      ▼
retrieval (RAG 组件)
  ├── Loaders → Document
  ├── Retriever → vector_stores + embeddings
      │
      ▼
vector_stores + embeddings
  ├── InMemoryVectorStore
  ├── QdrantVectorStore
  ├── OpenAIEmbeddings
```

---

## 开发指南

### 快速贡献

1. **修复 unimplemented!**：优先处理高优先级缺失
2. **添加新 Loader**：实现 DocumentLoader trait
3. **新增 LLM Provider**：实现 BaseChatModel trait
4. **添加测试**：减少 #[ignore] 测试

### 代码风格

- 遵循 Rust 标准命名规范
- 所有公开 API 需文档注释
- 新功能需配套测试
- 使用 feature flag 控制可选依赖

---

## 相关链接

- **GitHub**: https://github.com/atliliw/langchainrust
- **crates.io**: https://crates.io/crates/langchainrust
- **docs.rs**: https://docs.rs/langchainrust
- **LangChain Python**: https://github.com/langchain-ai/langchain