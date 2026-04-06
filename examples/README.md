# LangChain Rust 示例

本目录包含 LangChain Rust 的示例代码，按难度分为三个级别。

## 目录结构

```
examples/
├── basic/           # 基础示例 - 适合初学者
│   ├── hello_llm.rs
│   ├── streaming.rs
│   ├── prompt_template.rs
│   └── tools.rs
├── intermediate/    # 中级示例 - Agent、Memory、Chains
│   ├── agent_with_tools.rs
│   ├── memory_conversation.rs
│   └── chain_pipeline.rs
└── advanced/        # 高级示例 - RAG、完整流程
    ├── rag_demo.rs
    ├── multi_tool_agent.rs
    └── full_pipeline.rs
```

## 环境配置

运行示例前，请设置以下环境变量：

```bash
# OpenAI API 配置
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选，用于自定义 API 端点
```

或在 Windows PowerShell 中：

```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

## 基础示例 (Basic)

### 1. hello_llm.rs - 简单 LLM 调用
```bash
cargo run --example hello_llm
```
演示如何创建 OpenAI 客户端并进行简单的对话。

### 2. streaming.rs - 流式输出
```bash
cargo run --example streaming
```
演示如何使用流式 API 逐字接收 LLM 输出。

### 3. prompt_template.rs - 提示词模板
```bash
cargo run --example prompt_template
```
演示如何使用 PromptTemplate 和 ChatPromptTemplate（无需 API Key）。

### 4. tools.rs - 使用工具
```bash
cargo run --example tools
```
演示如何直接调用内置工具：Calculator、DateTimeTool、SimpleMathTool。

---

## 中级示例 (Intermediate)

### 1. agent_with_tools.rs - Agent 与工具
```bash
cargo run --example agent_with_tools
```
演示如何创建 ReActAgent 并使用工具回答问题。

**核心概念：**
- ReActAgent - Reasoning + Acting
- AgentExecutor - 执行器和迭代控制
- 工具自动选择

### 2. memory_conversation.rs - 记忆与多轮对话
```bash
cargo run --example memory_conversation
```
演示如何使用 Memory 实现多轮对话。

**核心概念：**
- ConversationBufferMemory
- 对话历史管理
- 上下文保持

### 3. chain_pipeline.rs - Chain 链式调用
```bash
cargo run --example chain_pipeline
```
演示如何使用 LLMChain 和 SequentialChain。

**核心概念：**
- LLMChain - 单步链
- SequentialChain - 多步顺序链
- Chain + Memory 组合

---

## 高级示例 (Advanced)

### 1. rag_demo.rs - RAG 检索增强生成
```bash
cargo run --example rag_demo
```
演示完整的 RAG 流程。

**流程：**
1. 准备知识库文档
2. 文档分割
3. 生成嵌入向量
4. 存储到向量数据库
5. 检索相关文档
6. 结合检索结果生成答案

**核心概念：**
- Document - 文档结构
- RecursiveCharacterSplitter - 文本分割
- OpenAIEmbeddings - 嵌入模型
- InMemoryVectorStore - 向量存储
- SimilarityRetriever - 相似度检索

### 2. multi_tool_agent.rs - 多工具 Agent
```bash
cargo run --example multi_tool_agent
```
演示 Agent 如何自动选择和使用多个工具。

**核心概念：**
- 多工具组合
- 自动工具选择
- 复杂任务分解

### 3. full_pipeline.rs - 完整 AI 应用
```bash
cargo run --example full_pipeline
```
演示一个完整的 AI 应用，结合所有组件。

**包含：**
- LLM 调用
- Agent + 工具
- Memory 对话记忆
- RAG 知识库检索
- 智能问答系统

---

## 运行所有示例

```bash
# 基础示例
cargo run --example hello_llm
cargo run --example streaming
cargo run --example prompt_template
cargo run --example tools

# 中级示例
cargo run --example agent_with_tools
cargo run --example memory_conversation
cargo run --example chain_pipeline

# 高级示例
cargo run --example rag_demo
cargo run --example multi_tool_agent
cargo run --example full_pipeline
```

## 示例难度说明

| 级别 | 适合人群 | 需要的前置知识 |
|------|---------|--------------|
| Basic | 初学者 | Rust 基础、异步编程概念 |
| Intermediate | 中级开发者 | Basic 级别 + Agent/Memory 概念 |
| Advanced | 高级开发者 | Intermediate 级别 + RAG/向量检索 |

## 注意事项

1. **API Key 安全**: 不要将 API Key 提交到代码仓库
2. **API 费用**: 部分示例会调用真实 API，可能产生费用
3. **网络要求**: 某些示例需要网络连接访问 API

## 故障排除

### 错误: "Invalid API Key"
确保正确设置了 `OPENAI_API_KEY` 环境变量。

### 错误: "Connection refused"
检查网络连接和 `OPENAI_BASE_URL` 设置。

### 示例运行缓慢
首次运行时需要编译，后续运行会快很多。