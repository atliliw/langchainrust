# langchainrust

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/langchainrust.svg)](https://crates.io/crates/langchainrust)
[![Documentation](https://docs.rs/langchainrust/badge.svg)](https://docs.rs/langchainrust)

一个受 LangChain 启发的 Rust 框架，用于构建基于大模型（LLM）的应用。

## ✨ 特性

### LLM 支持
- **OpenAI 兼容接口** - 支持流式与非流式输出
- **Qwen** - 通义千问接口
- **模型路由** - 根据问题难度自动选择合适的模型

### Agent
- **ReActAgent** - 支持 Tool 调用的智能代理（Reasoning + Acting）
- **AgentExecutor** - Agent 执行器，管理迭代和错误处理
- **工具调用** - Agent 可自主选择和调用工具

### 提示词工程
- **PromptTemplate** - 字符串模板 `{var}` 变量替换
- **ChatPromptTemplate** - 多角色消息模板

### 其他核心组件
- **Memory** - 对话记忆管理（ConversationBufferMemory、ChatMessageHistory）
- **Chains** - 链式调用（LLMChain、SequentialChain）
- **Retrieval** - 文档分割、向量存储、语义检索（RAG）
- **Tools** - 内置计算器、日期时间、数学运算、URL抓取等工具
- **Embeddings** - 文本嵌入（OpenAI、Mock）
- **VectorStore** - 向量存储（InMemoryVectorStore）

## 📦 安装

```toml
[dependencies]
langchainrust = "0.1.2"
```

## 🚀 快速开始

### 基础 LLM 调用

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

let config = OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY").unwrap(),
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-3.5-turbo".to_string(),
    streaming: false,
    temperature: Some(0.7),
    max_tokens: Some(500),
    ..Default::default()
};

let llm = OpenAIChat::new(config);
let messages = vec![
    Message::system("你是一个友好的助手。"),
    Message::human("什么是 Rust 语言？"),
];

let response = llm.chat(messages, None).await?;
println!("{}", response.content);
```

### 使用提示词模板

```rust
use langchainrust::prompts::{PromptTemplate, ChatPromptTemplate};
use langchainrust::schema::Message;
use std::collections::HashMap;

// 简单字符串模板
let template = PromptTemplate::new("你好，{name}！今天是{day}。");
let mut vars = HashMap::new();
vars.insert("name", "小明");
vars.insert("day", "星期一");
let prompt = template.format(&vars)?;

// 聊天消息模板
let chat_template = ChatPromptTemplate::new(vec![
    Message::system("你是一个{role}，专精于{domain}。"),
    Message::human("你好，我是{name}。"),
    Message::human("{question}"),
]);
```

### Agent + Tools

```rust
use langchainrust::{
    ReActAgent, AgentExecutor, BaseAgent, BaseTool,
    Calculator, DateTimeTool, SimpleMathTool,
};
use std::sync::Arc;

let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(DateTimeTool::new()),
    Arc::new(SimpleMathTool::new()),
];

let agent = ReActAgent::new(llm, tools.clone(), None);
let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
    .with_max_iterations(5);

let result = executor.invoke("计算 37 + 48".to_string()).await?;
```

### Chain 链式调用

```rust
use langchainrust::{LLMChain, SequentialChain, BaseChain};
use std::sync::Arc;

// 单步 Chain
let chain1 = LLMChain::new(llm1, "分析以下主题: {topic}");

// 多步顺序 Chain
let chain2 = LLMChain::new(llm2, "根据分析生成总结: {analysis}");

let seq_chain = SequentialChain::new()
    .add_chain(Arc::new(chain1), vec!["topic"], vec!["analysis"])
    .add_chain(Arc::new(chain2), vec!["analysis"], vec!["summary"]);

let result = seq_chain.invoke(inputs).await?;
```

### RAG 检索增强生成

```rust
use langchainrust::{
    Document, InMemoryVectorStore, MockEmbeddings,
    SimilarityRetriever, RetrieverTrait, RecursiveCharacterSplitter, TextSplitter,
};

// 准备文档
let docs = vec![
    Document::new("文档内容..."),
];

// 文档分割
let splitter = RecursiveCharacterSplitter::new(200, 50);
let chunks = splitter.split_document(&docs[0]);

// 创建检索器
let store = Arc::new(InMemoryVectorStore::new());
let embeddings = Arc::new(MockEmbeddings::new(128));
let retriever = SimilarityRetriever::new(store.clone(), embeddings);

// 添加文档
retriever.add_documents(chunks).await?;

// 检索相关文档
let relevant = retriever.retrieve("查询问题", 3).await?;
```

## 📚 示例

查看 [examples/](examples/) 目录获取完整示例：

### 基础示例
- `hello_llm.rs` - 基础 LLM 调用
- `streaming.rs` - 流式输出
- `prompt_template.rs` - 提示词模板
- `tools.rs` - 工具使用

### 中级示例
- `agent_with_tools.rs` - Agent 与工具
- `memory_conversation.rs` - 多轮对话记忆
- `chain_pipeline.rs` - Chain 链式调用

### 高级示例
- `rag_demo.rs` - RAG 检索增强生成
- `multi_tool_agent.rs` - 多工具 Agent
- `full_pipeline.rs` - 完整 AI 应用

运行示例：

```bash
# 无需 API Key
cargo run --example prompt_template
cargo run --example tools

# 需要 API Key
export OPENAI_API_KEY="your-api-key"
cargo run --example hello_llm
cargo run --example agent_with_tools
```

## 🧪 测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test prompts:: --lib -- --nocapture

# 运行测试并显示输出
cargo test -- --nocapture
```

## 📁 项目结构

```
src/
├── language_models/    # LLM 实现（OpenAI）
├── agents/             # Agent 框架
│   └── react/          # ReActAgent 实现
├── core/               # 核心抽象
│   ├── language_models/
│   ├── runnables/
│   └── tools/
├── prompts/            # 提示词模板
├── memory/             # 记忆管理
├── chains/             # 链式调用
├── retrieval/          # RAG 检索组件
├── embeddings/         # 文本嵌入
├── vector_stores/      # 向量存储
├── tools/              # 内置工具
└── schema/             # 数据结构
```

## 🔧 配置与安全

- **不要**将真实 API Key 提交到 Git 仓库
- 推荐使用环境变量：`OPENAI_API_KEY`
- 支持 OpenAI 代理地址配置：`OPENAI_BASE_URL`

## 📖 文档

- [API 文档](https://docs.rs/langchainrust)
- [使用示例](examples/)
- [贡献指南](CONTRIBUTING.md)

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 License

MIT OR Apache-2.0

## 🙏 致谢

本项目受 [LangChain](https://github.com/langchain-ai/langchain) 启发，使用 Rust 实现。