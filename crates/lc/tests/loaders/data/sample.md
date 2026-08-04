# LangChainRust 文档

LangChainRust 是一个 Rust 实现的 LangChain 框架。

## 核心模块

### LLM 模块

LLM 模块提供与语言模型的交互能力：

- OpenAI: 支持 GPT-3.5、GPT-4
- Ollama: 支持本地模型

使用示例：
```rust
let llm = OpenAIChat::new(config);
let response = llm.chat(messages).await?;
```

### Agent 模块

Agent 模块提供智能代理功能：

- ReActAgent: 文本解析式代理
- FunctionCallingAgent: 原生函数调用代理

Agent 可以自动选择并执行工具，完成复杂任务。

### Memory 模块

Memory 模块管理对话上下文：

- ConversationBufferMemory: 保存完整对话
- ConversationSummaryMemory: 自动摘要压缩
- ConversationSummaryBufferMemory: 混合策略

## RAG 功能

### BM25 检索

BM25 是关键词匹配检索算法，适合精确搜索场景。

参数配置：
- k1: 词频饱和参数（默认 1.5）
- b: 文档长度归一化（默认 0.75）

### Hybrid 检索

Hybrid 结合 BM25 和向量检索，使用 RRF 算法融合结果。

优势：
- 关键词精确匹配 + 语义理解
- 更高的召回率和精确度

## 安装使用

添加依赖：
```toml
[dependencies]
langchainrust = "0.2.6"
```

更多详情见 USAGE.md。