# langchainrust

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/langchainrust.svg)](https://crates.io/crates/langchainrust)
[![Documentation](https://docs.rs/langchainrust/badge.svg)](https://docs.rs/langchainrust)

A LangChain-inspired Rust framework for building LLM applications. Provides abstractions for agents, chains, memory, RAG pipelines, and tool-calling workflows.

一个受 LangChain 启发的 Rust 框架，用于构建 LLM 应用。提供 Agent、Chain、Memory、RAG 和工具调用等核心抽象。

[English](#english-documentation) | [中文文档](#中文文档)

---

# 中文文档

## ✨ 核心特性

| 组件 | 功能 |
|------|------|
| **LLM** | OpenAI 兼容接口，支持流式输出、function calling |
| **Agents** | ReActAgent（文本解析）+ FunctionCallingAgent（原生 FC）|
| **Prompts** | PromptTemplate 和 ChatPromptTemplate |
| **Memory** | 四种记忆类型：Buffer、Window、Summary、SummaryBuffer |
| **Chains** | LLMChain、SequentialChain、ConversationChain、RetrievalQA |
| **RAG** | 文档分割、向量存储、语义检索 |
| **Loaders** | 支持 PDF 和 CSV 文档加载 |
| **Tools** | 内置工具：计算器、日期时间、数学运算、URL抓取 |
| **Callbacks** | 执行追踪、LangSmith 集成、日志输出 |
| **LangGraph** | 图状工作流：StateGraph、可视化、Human-in-the-loop、Subgraph、Parallel |
| **Tool Calling** | bind_tools()、结构化输出、ToolDefinition、to_tool_definition() |
| **Streaming** | stream_chat() 逐 token 实时输出，打字机效果 |

### 关键优势

- 🚀 **完全异步** - 基于 Tokio 的 async/await 支持
- 🔒 **类型安全** - 利用 Rust 类型系统确保代码可靠性
- 📦 **零成本抽象** - 高性能设计
- 🎯 **简洁 API** - 直观易用的接口
- 🔌 **易于扩展** - 方便添加自定义工具和组件

## 📦 安装

在 `Cargo.toml` 中添加：

```toml
[dependencies]
langchainrust = "0.2.6"
tokio = { version = "1.0", features = ["full"] }
```

## 🚀 快速开始

### 基础对话

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")?,
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
    
    Ok(())
}
```

### 提示词模板

```rust
use langchainrust::prompts::{PromptTemplate, ChatPromptTemplate};
use langchainrust::schema::Message;
use std::collections::HashMap;

// 字符串模板
let template = PromptTemplate::new("你好，{name}！今天是{day}。");
let mut vars = HashMap::new();
vars.insert("name", "小明");
vars.insert("day", "星期一");
let prompt = template.format(&vars)?;

// 聊天模板
let chat_template = ChatPromptTemplate::new(vec![
    Message::system("你是一个{role}，专精于{domain}。"),
    Message::human("你好，我是{name}。"),
    Message::human("{question}"),
]);

let mut vars = HashMap::new();
vars.insert("role", "Rust 专家");
vars.insert("domain", "系统编程");
vars.insert("name", "小红");
vars.insert("question", "解释 Rust 的所有权机制");

let messages = chat_template.format(&vars)?;
```

### Agent 与工具调用

LangChainRust 提供两种 Agent：

| Agent | 方式 | 适用场景 |
|-------|------|----------|
| **ReActAgent** | 文本解析（正则提取） | 不支持 Function Calling 的模型 |
| **FunctionCallingAgent** | 原生 Function Calling | 支持 FC 的模型（推荐） |

#### 使用 FunctionCallingAgent（推荐）

```rust
use langchainrust::{
    FunctionCallingAgent, AgentExecutor, BaseAgent, BaseTool,
    Calculator, DateTimeTool, to_tool_definition,
};
use std::sync::Arc;

let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(DateTimeTool::new()),
];

// FunctionCallingAgent 自动绑定工具到 LLM
let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
    .with_max_iterations(5);

let result = executor.invoke("计算 37 + 48".to_string()).await?;
println!("结果: {}", result);
```

#### 使用 ReActAgent（兼容旧模型）

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
println!("结果: {}", result);
```

#### 两种 Agent 的区别

| 维度 | ReActAgent | FunctionCallingAgent |
|------|------------|---------------------|
| 工具调用方式 | 文本解析（正则） | 原生 Function Calling |
| 可靠性 | 依赖 Prompt 格式 | 类型安全，模型原生支持 |
| Token 消耗 | 高（需要格式说明） | 低（不需要格式说明） |
| 适用模型 | 所有模型 | 支持 FC 的模型（GPT-4、Claude、Gemini） |

### 对话记忆

LangChainRust 提供四种 Memory 类型：

| Memory 类型 | 压缩方式 | 适用场景 |
|-------------|----------|----------|
| **BufferMemory** | 无压缩 | 短对话、需要完整历史 |
| **WindowMemory** | 窗口截断 | 简单控制、接受丢失 |
| **SummaryMemory** | LLM 摘要 | 长对话、节省 token |
| **SummaryBufferMemory** | 混合策略 | 平衡方案（推荐） |

#### ConversationBufferMemory

```rust
use langchainrust::{ConversationBufferMemory, BaseMemory};
use std::collections::HashMap;

let mut memory = ConversationBufferMemory::new();

// 保存对话
let inputs = HashMap::from([("input".to_string(), "我叫张三".to_string())]);
let outputs = HashMap::from([("output".to_string(), "你好张三！".to_string())]);
memory.save_context(&inputs, &outputs).await?;

// 加载历史
let vars = memory.load_memory_variables(&HashMap::new()).await?;
let history = vars.get("history").unwrap().as_str().unwrap();
// 输出: "Human: 我叫张三\nAI: 你好张三！"
```

#### ConversationBufferWindowMemory

```rust
use langchainrust::ConversationBufferWindowMemory;

// k=2，保留最近 2 轮（4 条消息）
let mut memory = ConversationBufferWindowMemory::new(2);

for i in 1..=5 {
    let inputs = HashMap::from([("input".to_string(), format!("问题{}", i))]);
    let outputs = HashMap::from([("output".to_string(), format!("答案{}", i))]);
    memory.save_context(&inputs, &outputs).await?;
}

// 只返回最近 2 轮，问题1-3 被丢弃
let vars = memory.load_memory_variables(&HashMap::new()).await?;
```

#### ConversationSummaryBufferMemory（推荐）

```rust
use langchainrust::{ConversationSummaryBufferMemory, OpenAIChat};

let llm = OpenAIChat::new(config);

// max_token_limit = 100，超过时触发压缩
let mut memory = ConversationSummaryBufferMemory::new(llm, 100);

for i in 1..=10 {
    memory.save_context(&inputs, &outputs).await?;
}

// 返回: "摘要: 用户讨论了...\n\nHuman: 最近消息\nAI: 回复"
let vars = memory.load_memory_variables(&HashMap::new()).await?;
```

### 流式输出

实时看到生成过程，感知延迟更低：

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;
use futures_util::StreamExt;

let config = OpenAIConfig {
    streaming: true,  // 启用流式
    ..Default::default()
};

let llm = OpenAIChat::new(config);
let messages = vec![Message::human("写一段短文")];

// 获取流式输出
let mut stream = llm.stream_chat(messages, None).await?;

// 逐 token 接收并实时打印
while let Some(chunk) = stream.next().await {
    if let Ok(token) = chunk {
        print!("{}", token);  // 打字机效果
    }
}
```

### 基础对话

```rust
use langchainrust::{LLMChain, SequentialChain, BaseChain};
use std::sync::Arc;
use std::collections::HashMap;
use serde_json::Value;

// 单步 Chain
let chain1 = LLMChain::new(llm1, "分析以下主题: {topic}");

// 多步顺序 Chain
let chain2 = LLMChain::new(llm2, "根据分析生成总结: {analysis}");

let pipeline = SequentialChain::new()
    .add_chain(Arc::new(chain1), vec!["topic"], vec!["analysis"])
    .add_chain(Arc::new(chain2), vec!["analysis"], vec!["summary"]);

let mut inputs = HashMap::new();
inputs.insert("topic".to_string(), Value::String("2024年人工智能发展".to_string()));

let results = pipeline.invoke(inputs).await?;
```

### RAG 检索增强生成

```rust
use langchainrust::{
    Document, InMemoryVectorStore, MockEmbeddings,
    SimilarityRetriever, RetrieverTrait, 
    RecursiveCharacterSplitter, TextSplitter,
};
use std::sync::Arc;

// 创建文档
let docs = vec![
    Document::new("Rust 是一门系统编程语言..."),
];

// 文档分割
let splitter = RecursiveCharacterSplitter::new(200, 50);
let chunks = splitter.split_document(&docs[0]);

// 创建检索器
let store = Arc::new(InMemoryVectorStore::new());
let embeddings = Arc::new(MockEmbeddings::new(128));
let retriever = SimilarityRetriever::new(store.clone(), embeddings);

// 索引文档
retriever.add_documents(chunks).await?;

// 检索
let relevant_docs = retriever.retrieve("什么是 Rust？", 3).await?;
```

### 文档加载器

LangChainRust 现在支持从多种格式加载文档，包括 PDF 和 CSV 文件。

#### PDF Loader

```rust
use langchainrust::retrieval::{PDFLoader, DocumentLoader};

// 加载 PDF 文件
let pdf_loader = PDFLoader::new("path/to/document.pdf");
let documents = pdf_loader.load().await?;

// 提取的文档包含文本内容和元数据
for doc in documents {
    println!("Content: {}", &doc.content[..100.min(doc.content.len())]);
    println!("Metadata: {:?}", doc.metadata);
}
```

#### CSV Loader

```rust
use langchainrust::retrieval::{CSVLoader, DocumentLoader};

// 加载 CSV 文件，指定内容列为"description"
let csv_loader = CSVLoader::new("path/to/data.csv", "description");
let documents = csv_loader.load().await?;

// 每一行数据转换为单独的文档，具有对应元数据
for doc in documents {
    println!("Content: {}", doc.content);
    println!("Row Metadata: {:?}", doc.metadata);
}
```

## 📚 完整示例

查看 [examples/](examples/) 目录：

### 基础示例
- `hello_llm` - 基础 LLM 对话
- `streaming` - 流式输出
- `prompt_template` - 提示词模板
- `tools` - 内置工具

### 中级示例
- `agent_with_tools` - Agent 工具调用
- `memory_conversation` - 多轮对话记忆
- `chain_pipeline` - Chain 工作流

### 高级示例
- `rag_demo` - 完整 RAG 流程
- `multi_tool_agent` - 多工具 Agent
- `full_pipeline` - 完整 AI 应用

运行示例：

```bash
# 无需 API Key
cargo run --example prompt_template
cargo run --example tools

# 需要 API Key
export OPENAI_API_KEY="your-key"
cargo run --example hello_llm
cargo run --example agent_with_tools
```

## 🧪 测试

```bash
# 运行所有测试
cargo test

# 运行特定模块测试
cargo test prompts:: --lib -- --nocapture

# 显示测试输出
cargo test -- --nocapture
```

## 📁 项目结构

```
src/
├── core/                # 核心抽象
│   ├── language_models/ # 基础 LLM trait
│   ├── runnables/       # Runnable trait
│   └── tools/           # Tool trait + to_tool_definition()
├── language_models/     # LLM 实现
│   └── openai/          # OpenAI 客户端（支持 Function Calling）
├── agents/              # Agent 框架
│   ├── react/           # ReActAgent（文本解析）
│   └── function_calling/ # FunctionCallingAgent（原生 FC）
├── prompts/             # 提示词模板
├── memory/              # 记忆管理
├── chains/              # 链式调用
├── retrieval/           # RAG 组件
├── embeddings/          # 文本嵌入
├── vector_stores/       # 向量存储
├── tools/               # 内置工具
└── schema/              # 数据结构
```

## 🔧 配置

### 环境变量

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选：自定义端点
```

### OpenAIConfig 配置项

| 字段 | 类型 | 说明 |
|------|------|------|
| `api_key` | `String` | OpenAI API 密钥 |
| `base_url` | `String` | API 端点（支持代理） |
| `model` | `String` | 模型名称（如 "gpt-3.5-turbo"） |
| `streaming` | `bool` | 启用流式响应 |
| `temperature` | `Option<f32>` | 采样温度 (0.0-2.0) |
| `max_tokens` | `Option<usize>` | 最大生成 token 数 |

## 🔐 安全提示

- **切勿**将 API Key 提交到版本控制
- 使用环境变量存储密钥
- 支持代理/自定义端点

## 📖 文档

- [API 文档](https://docs.rs/langchainrust)
- [示例代码](examples/)
- [贡献指南](CONTRIBUTING.md)

## 🤝 贡献

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

Apache License, Version 2.0 或 MIT License，任选其一。

## 🙏 致谢

本项目受 [LangChain](https://github.com/langchain-ai/langchain) 启发，使用 Rust 实现。

---

# English Documentation

## ✨ Features

| Component | Description |
|-----------|-------------|
| **LLM** | OpenAI-compatible API with streaming support |
| **Agents** | ReActAgent (text parsing) + FunctionCallingAgent (native FC) |
| **Prompts** | PromptTemplate and ChatPromptTemplate |
| **Memory** | Conversation history management |
| **Chains** | LLMChain and SequentialChain workflows |
| **RAG** | Document splitting, vector stores, semantic retrieval |
| **Loaders** | PDF and CSV document loading support |
| **Tools** | Built-in: Calculator, DateTime, Math, URLFetch |
| **LangGraph** | Graph-based workflows: StateGraph, visualization, Human-in-the-loop, Subgraph, Parallel |
| **Tool Calling** | bind_tools(), to_tool_definition(), ToolDefinition, structured output |

### Key Benefits

- 🚀 **Fully Async** - Tokio-based async/await support
- 🔒 **Type-Safe** - Leverage Rust's type system
- 📦 **Zero-Cost Abstractions** - High-performance design
- 🎯 **Simple API** - Intuitive interfaces
- 🔌 **Extensible** - Easy to add custom tools

## 📦 Installation

Add to `Cargo.toml`:

```toml
[dependencies]
langchainrust = "0.2.6"
tokio = { version = "1.0", features = ["full"] }
```

## 🚀 Quick Start

### Basic Chat

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")?,
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        temperature: Some(0.7),
        max_tokens: Some(500),
        ..Default::default()
    };
    
    let llm = OpenAIChat::new(config);
    
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::human("What is Rust?"),
    ];
    
    let response = llm.chat(messages, None).await?;
    println!("{}", response.content);
    
    Ok(())
}
```

### Prompt Templates

```rust
use langchainrust::prompts::{PromptTemplate, ChatPromptTemplate};
use langchainrust::schema::Message;
use std::collections::HashMap;

// String template
let template = PromptTemplate::new("Hello, {name}! Today is {day}.");
let mut vars = HashMap::new();
vars.insert("name", "Alice");
vars.insert("day", "Monday");
let prompt = template.format(&vars)?;

// Chat template
let chat_template = ChatPromptTemplate::new(vec![
    Message::system("You are a {role} specializing in {domain}."),
    Message::human("Hello, I'm {name}."),
    Message::human("{question}"),
]);

let mut vars = HashMap::new();
vars.insert("role", "Rust expert");
vars.insert("domain", "systems programming");
vars.insert("name", "Bob");
vars.insert("question", "Explain ownership in Rust");

let messages = chat_template.format(&vars)?;
```

### Agent with Tools

LangChainRust provides two types of Agents:

| Agent | Method | Use Case |
|-------|--------|----------|
| **ReActAgent** | Text parsing (regex) | Models without Function Calling support |
| **FunctionCallingAgent** | Native Function Calling | Models with FC support (recommended) |

#### Using FunctionCallingAgent (Recommended)

```rust
use langchainrust::{
    FunctionCallingAgent, AgentExecutor, BaseAgent, BaseTool,
    Calculator, DateTimeTool,
};
use std::sync::Arc;

let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(DateTimeTool::new()),
];

// FunctionCallingAgent automatically binds tools to LLM
let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
    .with_max_iterations(5);

let result = executor.invoke("What is 37 + 48?".to_string()).await?;
println!("Answer: {}", result);
```

#### Using ReActAgent (Legacy Support)

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

let result = executor.invoke("What is 37 + 48?".to_string()).await?;
println!("Answer: {}", result);
```

### Memory

```rust
use langchainrust::{ChatMessageHistory, Message};

let mut history = ChatMessageHistory::new();

// Add messages
history.add_message(Message::human("Hello!"));
history.add_message(Message::ai("Hi there!"));

// Retrieve messages
for msg in history.messages() {
    println!("{:?}: {}", msg.message_type, msg.content);
}
```

### Chain Pipelines

```rust
use langchainrust::{LLMChain, SequentialChain, BaseChain};
use std::sync::Arc;
use std::collections::HashMap;
use serde_json::Value;

// Single chain
let chain1 = LLMChain::new(llm1, "Analyze this topic: {topic}");

// Sequential chains
let chain2 = LLMChain::new(llm2, "Summarize: {analysis}");

let pipeline = SequentialChain::new()
    .add_chain(Arc::new(chain1), vec!["topic"], vec!["analysis"])
    .add_chain(Arc::new(chain2), vec!["analysis"], vec!["summary"]);

let mut inputs = HashMap::new();
inputs.insert("topic".to_string(), Value::String("AI in 2024".to_string()));

let results = pipeline.invoke(inputs).await?;
```

### RAG Pipeline

```rust
use langchainrust::{
    Document, InMemoryVectorStore, MockEmbeddings,
    SimilarityRetriever, RetrieverTrait, 
    RecursiveCharacterSplitter, TextSplitter,
};
use std::sync::Arc;

// Create documents
let docs = vec![
    Document::new("Rust is a systems programming language..."),
];

// Split documents
let splitter = RecursiveCharacterSplitter::new(200, 50);
let chunks = splitter.split_document(&docs[0]);

// Create retriever
let store = Arc::new(InMemoryVectorStore::new());
let embeddings = Arc::new(MockEmbeddings::new(128));
let retriever = SimilarityRetriever::new(store.clone(), embeddings);

// Index documents
retriever.add_documents(chunks).await?;

// Search
let relevant_docs = retriever.retrieve("What is Rust?", 3).await?;
```

## 📚 Examples

See [examples/](examples/) for complete code:

### Basic
- `hello_llm` - Basic LLM chat
- `streaming` - Streaming output
- `prompt_template` - Using templates
- `tools` - Built-in tools

### Intermediate
- `agent_with_tools` - Agent with tool calling
- `memory_conversation` - Multi-turn conversations
- `chain_pipeline` - Chain workflows

### Advanced
- `rag_demo` - Full RAG pipeline
- `multi_tool_agent` - Agent with multiple tools
- `full_pipeline` - Complete AI application

Run examples:

```bash
# Without API key
cargo run --example prompt_template
cargo run --example tools

# With API key
export OPENAI_API_KEY="your-key"
cargo run --example hello_llm
cargo run --example agent_with_tools
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run specific module
cargo test prompts:: --lib -- --nocapture

# Show test output
cargo test -- --nocapture
```

## 📁 Project Structure

```
src/
├── core/                # Core abstractions
│   ├── language_models/ # Base LLM traits
│   ├── runnables/       # Runnable trait
│   └── tools/           # Tool trait + to_tool_definition()
├── language_models/     # LLM implementations
│   └── openai/          # OpenAI client (Function Calling support)
├── agents/              # Agent framework
│   ├── react/           # ReActAgent (text parsing)
│   └── function_calling/ # FunctionCallingAgent (native FC)
├── prompts/             # Prompt templates
├── memory/              # Memory management
├── chains/              # Chain workflows
├── retrieval/           # RAG components
├── embeddings/          # Text embeddings
├── vector_stores/       # Vector databases
├── tools/               # Built-in tools
└── schema/              # Data structures
```

## 🔧 Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional: custom endpoint
```

### OpenAIConfig Options

| Field | Type | Description |
|-------|------|-------------|
| `api_key` | `String` | OpenAI API key |
| `base_url` | `String` | API endpoint (supports proxies) |
| `model` | `String` | Model name (e.g., "gpt-3.5-turbo") |
| `streaming` | `bool` | Enable streaming responses |
| `temperature` | `Option<f32>` | Sampling temperature (0.0-2.0) |
| `max_tokens` | `Option<usize>` | Maximum tokens to generate |

## 🔐 Security

- **Never** commit API keys to version control
- Use environment variables for secrets
- Support for proxy/custom endpoints

## 📖 Documentation

- [API Documentation](https://docs.rs/langchainrust)
- [Examples](examples/)
- [Contributing](CONTRIBUTING.md)

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

Licensed under either of:
- Apache License, Version 2.0
- MIT License

## 🙏 Acknowledgments

Inspired by [LangChain](https://github.com/langchain-ai/langchain), implemented in Rust.