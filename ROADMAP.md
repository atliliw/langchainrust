# LangChain Rust ROADMAP

## 项目概况

| 统计 | 数值 |
|------|------|
| 源码文件 | 89 个 (+3) |
| 测试文件 | 48 个 (+3) |
| 版本 | 0.2.6 |
| 核心模块 | 14 个 (+3) |

---

## 已完成模块 ✅

### 1. LLM (language_models)
| 功能 | 状态 | 文件 |
|------|------|------|
| OpenAIChat | ✅ 完成 | `src/language_models/openai/chat.rs` |
| OpenAIConfig | ✅ 完成 | `src/language_models/openai/config.rs` |
| 流式输出 (stream_chat) | ✅ 完成 | 同上 |
| Function Calling | ✅ 完成 | 同上 |
| OllamaChat | ✅ 完成 | `src/language_models/ollama/chat.rs` |
| OllamaConfig | ✅ 完成 | `src/language_models/ollama/config.rs` |

### 2. Agents (agents)
| 功能 | 状态 | 文件 |
|------|------|------|
| ReActAgent (文本解析) | ✅ 完成 | `src/agents/react/agent.rs` |
| FunctionCallingAgent (原生FC) | ✅ 完成 | `src/agents/function_calling/agent.rs` |
| AgentExecutor | ✅ 完成 | `src/agents/executor.rs` |
| BaseAgent trait | ✅ 完成 | `src/agents/base.rs` |

### 3. Memory (memory)
| 功能 | 状态 | 文件 |
|------|------|------|
| ChatMessageHistory | ✅ 完成 | `src/memory/history.rs` |
| ConversationBufferMemory | ✅ 完成 | `src/memory/buffer.rs` |
| ConversationBufferWindowMemory | ✅ 完成 | `src/memory/window.rs` |
| ConversationSummaryMemory | ✅ 完成 | `src/memory/summary.rs` |
| ConversationSummaryBufferMemory | ✅ 完成 | `src/memory/summary_buffer.rs` |

### 4. Chains (chains)
| 功能 | 状态 | 文件 |
|------|------|------|
| LLMChain | ✅ 完成 | `src/chains/llm_chain.rs` |
| SequentialChain | ✅ 完成 | `src/chains/sequential_chain.rs` |
| ConversationChain | ✅ 完成 | `src/chains/conversation_chain.rs` |
| RouterChain | ✅ 完成 | `src/chains/router_chain.rs` |
| RetrievalQA | ✅ 完成 | `src/chains/retrieval_qa.rs` |

### 5. Prompts (prompts)
| 功能 | 状态 | 文件 |
|------|------|------|
| PromptTemplate | ✅ 完成 | `src/prompts/template.rs` |
| ChatPromptTemplate | ✅ 完成 | `src/prompts/chat_template.rs` |

### 6. RAG (retrieval)
| 功能 | 状态 | 文件 |
|------|------|------|
| TextSplitter | ✅ 完成 | `src/retrieval/splitter.rs` |
| RecursiveCharacterSplitter | ✅ 完成 | 同上 |
| DocumentLoader trait | ✅ 完成 | `src/retrieval/loader.rs` |
| PDFLoader | ✅ 完成 | 同上 |
| CSVLoader | ✅ 完成 | 同上 |
| SimilarityRetriever | ✅ 完成 | `src/retrieval/retriever.rs` |

### 7. Embeddings (embeddings)
| 功能 | 状态 | 文件 |
|------|------|------|
| Embeddings trait | ✅ 完成 | `src/embeddings/mod.rs` |
| OpenAIEmbeddings | ✅ 完成 | `src/embeddings/openai.rs` |
| MockEmbeddings | ✅ 完成 | `src/embeddings/mock.rs` |
| cosine_similarity | ✅ 完成 | 同上 |

### 8. Vector Stores (vector_stores)
| 功能 | 状态 | 文件 |
|------|------|------|
| VectorStore trait | ✅ 完成 | `src/vector_stores/mod.rs` |
| InMemoryVectorStore | ✅ 完成 | `src/vector_stores/inmemory.rs` |
| QdrantVectorStore | ✅ 完成 | `src/vector_stores/qdrant.rs` (feature-gated) |
| VectorStoreBuilder | ✅ 完成 | `src/vector_stores/provider.rs` |

### 9. Tools (tools)
| 功能 | 状态 | 文件 |
|------|------|------|
| BaseTool trait | ✅ 完成 | `src/core/tools/mod.rs` |
| Calculator | ✅ 完成 | `src/tools/calculator.rs` |
| DateTimeTool | ✅ 完成 | `src/tools/datetime.rs` |
| SimpleMathTool | ✅ 完成 | `src/tools/math.rs` |
| URLFetchTool | ✅ 完成 | `src/tools/url_fetch.rs` |
| ToolRegistry | ✅ 完成 | `src/core/tools/mod.rs` |
| to_tool_definition() | ✅ 完成 | 同上 |

### 10. Callbacks (callbacks)
| 功能 | 状态 | 文件 |
|------|------|------|
| CallbackHandler trait | ✅ 完成 | `src/callbacks/base.rs` |
| CallbackManager | ✅ 完成 | `src/callbacks/manager.rs` |
| StdOutHandler | ✅ 完成 | `src/callbacks/handlers/stdout_handler.rs` |
| LangSmithHandler | ✅ 完成 | `src/callbacks/handlers/langsmith_handler.rs` |
| FileCallbackHandler | ✅ 完成 | `src/callbacks/handlers/file_handler.rs` |
| RunTree | ✅ 完成 | `src/callbacks/run_tree.rs` |

### 11. LangGraph (langgraph)
| 功能 | 状态 | 文件 |
|------|------|------|
| StateGraph | ✅ 完成 | `src/langgraph/graph.rs` |
| GraphBuilder | ✅ 完成 | 同上 |
| CompiledGraph | ✅ 完成 | `src/langgraph/compiled.rs` |
| GraphNode trait | ✅ 完成 | `src/langgraph/node.rs` |
| SyncNode | ✅ 完成 | 同上 |
| FunctionNode (异步节点) | ✅ 完成 | 同上 |
| AsyncNode | ✅ 完成 | 同上 |
| AsyncFn trait | ✅ 完成 | 同上 |
| add_async_node() | ✅ 完成 | `src/langgraph/graph.rs` |
| GraphEdge | ✅ 完成 | `src/langgraph/edge.rs` |
| ConditionalEdge | ✅ 完成 | 同上 |
| FunctionRouter | ✅ 完成 | 同上 |
| StateSchema trait | ✅ 完成 | `src/langgraph/state.rs` |
| AgentState | ✅ 完成 | 同上 |
| StateUpdate | ✅ 完成 | 同上 |
| Reducer trait | ✅ 完成 | 同上 |
| ReplaceReducer | ✅ 完成 | 同上 |
| AppendReducer | ✅ 完成 | 同上 |
| AppendMessagesReducer | ✅ 完成 | 同上 |
| AppendStepsReducer | ✅ 完成 | 同上 |
| Checkpointer trait | ✅ 完成 | `src/langgraph/checkpointer.rs` |
| MemoryCheckpointer | ✅ 完成 | 同上 |
| ThreadSafeMemoryCheckpointer | ✅ 完成 | 同上 |
| FileCheckpointer | ✅ 完成 | 同上 |
| GraphError | ✅ 完成 | `src/langgraph/errors.rs` |
| invoke() | ✅ 完成 | `src/langgraph/compiled.rs` |
| stream() | ✅ 完成 | 同上 |
| START / END sentinel | ✅ 完成 | `src/langgraph/graph.rs` |
| 条件边路由 | ✅ 完成 | `src/langgraph/edge.rs` |
| 循环执行 + recursion_limit | ✅ 完成 | `src/langgraph/compiled.rs` |
| Graph 可视化 | ✅ 完成 | `src/langgraph/compiled.rs` |
| visualize_ascii() | ✅ 完成 | 同上 |
| visualize_mermaid() | ✅ 完成 | 同上 |
| visualize_json() | ✅ 完成 | 同上 |
| Human-in-the-loop | ✅ 完成 | `src/langgraph/compiled.rs` |
| interrupt_before | ✅ 完成 | 同上 |
| interrupt_after | ✅ 完成 | 同上 |
| GraphExecution | ✅ 完成 | 同上 |
| invoke_with_execution() | ✅ 完成 | 同上 |
| resume() | ✅ 完成 | 同上 |
| Graph 验证增强 | ✅ 完成 | `src/langgraph/compiled.rs` |
| validate_duplicate_edges() | ✅ 完成 | 同上 |
| validate_unreachable_nodes() | ✅ 完成 | 同上 |
| validate_cycles() | ✅ 完成 | 同上 |
| compute_reachable_nodes() | ✅ 完成 | 同上 |
| compute_end_reachable_nodes() | ✅ 完成 | 同上 |
| 错误类型增强 | ✅ 完成 | `src/langgraph/errors.rs` |
| InfiniteCycleError | ✅ 完成 | 同上 |
| OrphanNodeError | ✅ 完成 | 同上 |
| DuplicateEdgeError | ✅ 完成 | 同上 |
| ExecutionInterrupted | ✅ 完成 | 同上 |
| LLM 集成示例 | ✅ 完成 | `tests/langgraph/llm_integration.rs` |
| 异步节点测试 | ✅ 完成 | `tests/langgraph/async_node.rs` |
| 可视化测试 | ✅ 完成 | `tests/langgraph/visualize.rs` |
| Human-in-the-loop 测试 | ✅ 完成 | `tests/langgraph/human_loop.rs` |
| Graph 验证测试 | ✅ 完成 | `tests/langgraph/validation.rs` |
| **Subgraph 测试** | ✅ 完成 | `tests/langgraph/subgraph.rs` |
| **Parallel 执行测试** | ✅ 完成 | `tests/langgraph/parallel.rs` |
| **Persistence 测试** | ✅ 完成 | `tests/langgraph/persistence.rs` |

---

## 待完成/增强功能 📋

### LangGraph 增强 (高优先级)
| 功能 | 状态 | 说明 |
|------|------|------|
| 异步节点 API 优化 | ✅ 完成 | add_async_node() 方法 |
| AppendReducer | ✅ 完成 | AppendMessagesReducer + AppendStepsReducer |
| FileCheckpointer | ✅ 完成 | 文件持久化 Checkpointer |
| Graph 可视化 | ✅ 完成 | visualize_ascii/mermaid/json |
| Human-in-the-loop | ✅ 完成 | interrupt_before/interrupt_after + resume() |
| GraphExecution 状态结构 | ✅ 完成 | 保存中断执行进度 |
| Graph 验证增强 - 死循环检测 | ✅ 完成 | validate_cycles() |
| Graph 验证增强 - 孤立节点检测 | ✅ 完成 | validate_unreachable_nodes() |
| Graph 验证增强 - 重复边检测 | ✅ 完成 | validate_duplicate_edges() |
| 条件边目标完整性验证 | ✅ 完成 | 验证 targets 存在性 |
| **Subgraph 支持** | ✅ 完成 | SubgraphNode + 状态映射器 + 嵌套子图 |
| **Parallel Node 执行** | ✅ 完成 | invoke_parallel() + FanOut/FanIn 真正并行执行 |
| **Graph persistence** | ✅ 完成 | GraphDefinition + NodeRegistry + save_to_file/load_from_file |

### LLM 增强
| 功能 | 状态 | 说明 |
|------|------|------|
| Claude API | 📋 待完成 | Anthropic Claude 支持 |
| Gemini API | 📋 待完成 | Google Gemini 支持 |
| 本地模型 | 📋 待完成 | 更多本地模型适配 |
| Token 计数 | 📋 待完成 | 精确 token 使用统计 |

### RAG 增强
| 功能 | 状态 | 说明 |
|------|------|------|
| 更多文档加载器 | 📋 待完成 | Markdown、JSON、HTML 等 |
| 更多向量数据库 | 📋 待完成 | Pinecone、Weaviate、Milvus |
| 重排序 | 📋 待完成 | Reranker 支持 |
| 混合检索 | 📋 待完成 | BM25 + 语义检索混合 |

### Agents 增强
| 功能 | 状态 | 说明 |
|------|------|------|
| Plan-and-Execute Agent | 📋 待完成 | 规划型 Agent |
| Multi-Agent | 📋 待完成 | 多 Agent 协作 |
| Agent 工具动态加载 | 📋 待完成 | 运行时添加工具 |

### Chains 增强
| 功能 | 状态 | 说明 |
|------|------|------|
| TransformChain | 📋 待完成 | 数据转换 Chain |
| BranchChain | 📋 待完成 | 分支 Chain |
| LCEL 完整实现 | 📋 待完成 | LangChain Expression Language |

---

## 测试覆盖

### 已有测试文件
| 目录 | 测试文件数 | 覆盖范围 |
|------|-----------|----------|
| `tests/unit/` | 13 | 单元测试（Memory、Chain、Callback等） |
| `tests/integration/` | 17 | 集成测试（Agent、RAG、LLM等） |
| `tests/langgraph/` | 8 | LangGraph 测试（basic、conditional、LLM集成等） |
| `tests/function_calling/` | 1 | Function Calling Agent 测试 |
| `tests/e2e/` | 1 | 完整应用测试 |

### 测试类型
- 单元测试：无 API Key，纯逻辑验证
- 集成测试：部分需要 API Key (`#[ignore]`)
- LangGraph 测试：包含真实 LLM 调用示例

---

## 版本规划

### v0.2.6 (当前版本)
- LangGraph 异步节点 API 优化
- AppendReducer 实现
- Subgraph 支持 ✅
- Parallel Node 执行 ✅
- Graph persistence ✅

### v0.3.0
- Claude API 支持
- Graph 可视化增强
- 更多 LangGraph 示例

### v0.4.0
- Multi-Agent 协作
- Human-in-the-loop 增强
- 更多持久化选项

---