# langchainrust

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

一个受 LangChain 启发的 Rust 框架，用于构建基于大模型（LLM）的应用。

📖 **详细文档**: [docs/USAGE.md](docs/USAGE.md)

## ✨ 特性

### LLM 支持
- **OpenAI 兼容接口** - 支持流式与非流式输出
- **Qwen** - 通义千问接口
- **模型路由** - 根据问题难度自动选择合适的模型

### Agent
- **ReActAgent** - 支持 Tool 调用的智能代理
- **PlannedExecutor** - 自动分解复杂任务、执行、汇总
- **工具调用** - 可选的工具使用，LLM 自主判断
- **RAG 检索** - 结合向量数据库的知识增强

### 提示词工程
- **PromptTemplate** - 字符串模板 `{var}` 替换
- **ChatPromptTemplate** - 多角色消息模板

### 其他
- **Memory** - 对话记忆管理
- **Chains** - 链式调用组合
- **Retrieval** - 文档分割、向量存储、语义检索
- **Tools** - 内置计算器、天气、日期时间等工具

## 📦 安装

```toml
[dependencies]
langchainrust = "0.1"
```

## 🚀 快速开始

### 基础 LLM 调用

```rust
use langchainrust::llms::{LLM, OpenAIConfig};

let llm = LLM::new(OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY").unwrap(),
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-3.5-turbo".to_string(),
    streaming: false,
    factor: 3,
});

let response = llm.invoke("你好，介绍一下 Rust 语言").await?;
println!("{}", response);
```

### Agent + Tools

```rust
use langchainrust::agent::{AgentExecutor, ReActAgent};
use langchainrust::tools::Calculator;
use std::sync::Arc;

let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Calculator)];
let agent = ReActAgent::new(llm, tools.clone(), None);
let executor = AgentExecutor::new(Box::new(agent), tools);

let result = executor.run("计算 37 + 48").await?;
```

### RAG 检索增强

```rust
use langchainrust::agent::ReActAgent;
use langchainrust::retrieval::{
    Document, InMemoryVectorStore, MockEmbeddingModel,
    RecursiveCharacterSplitter, SimilarityRetriever, TextSplitter,
};

// 准备文档并创建检索器
let retriever = SimilarityRetriever::new(vector_store, embedding_model);
retriever.add_documents(chunks).await?;

// 创建带 RAG 的 Agent
let agent = ReActAgent::with_retriever(
    llm, tools, memory, retriever, 3
);
```

### 任务规划

```rust
use langchainrust::agent::{PlannedExecutor, ReActAgent};

let executor = PlannedExecutor::new(llm, agent, tools)
    .with_max_sub_tasks(3)
    .with_verbose(true);

// 自动分解、执行、汇总
let result = executor
    .run("分析项目代码，写测试用例，运行测试")
    .await?;
```

## 📚 文档

- [详细使用文档](docs/USAGE.md)
- [Retrieval 模块原理](src/retrieval/README.md)

## 🧪 运行测试

```bash
# 运行全部测试
cargo test

# 运行特定测试
cargo test --test planner_test -- --nocapture
cargo test --test retrieval_test -- --nocapture
```

## 📁 项目结构

```
src/
├── llms/          # LLM 实现（OpenAI、Qwen、模型路由）
├── agent/         # Agent 框架
│   ├── react/     # ReActAgent 实现
│   └── planner/   # 任务规划模块
├── retrieval/     # RAG 检索组件
├── tools/         # 工具接口与内置工具
├── prompts/       # 提示词模板
├── memory/        # 记忆管理
└── chains/        # 链式调用
```

## 🔧 配置与安全

- **不要**将真实 API Key 提交到 Git 仓库
- 推荐使用环境变量：`OPENAI_API_KEY`
- 支持 OpenAI 代理地址配置

## 📄 License

MIT OR Apache-2.0
