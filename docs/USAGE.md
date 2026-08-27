# 使用指南

本文档提供详细的使用说明。如需快速概览，请参阅 [README.md](../README.md)。

---

## 目录

- [LLM](#llm)
  - 多 Provider 支持
  - 统一客户端与自动发现 ✨ v0.15.0
  - OpenAI Chat
  - 流式输出
  - 函数调用
  - Ollama（本地 LLM）
  - Google Gemini
  - 多模态视觉
  - Message 结构 ✨ v0.15.0
  - MultimodalModel ✨ v0.15.0
  - OpenAI Assistants API
- [嵌入](#embeddings)
  - OpenAI Embeddings
  - DeepSeek Embeddings
  - Qwen Embeddings
  - LocalEmbeddings
- [提示词](#prompts)
  - FewShotPrompt + ExampleSelectors
- [输出解析器](#output-parsers)
- [记忆](#memory)
  - VectorStoreRetrieverMemory
  - MongoPersistentMemory ✨ v0.15.0
  - ContextWindow（长上下文管理） ✨ v0.4.1
- [LLM 缓存](#llm-cache)
- [链](#chains)
  - ConversationRetrievalChain
  - RouterChain ✨ v0.14.0
  - 链流式输出 ✨ v0.4.1
  - ConversationChain ✨ v0.13.0
  - invoke_with_config ✨ v0.15.0
- [LCEL (LangChain Expression Language)](#lcel-langchain-expression-language-) ✨ v0.9.0
  - RunnableWithFallbacks ✨ v0.10.0
  - RunnableAssign ✨ v0.10.0
  - RunnableRetry ✨ v0.11.0
  - CancellationToken ✨ v0.11.0
  - 适配器 (AgentEventRunnable / OrchestratorRunnable) ✨ v0.13.0
  - 统一组合 (v0.15.0)
- [文档链](#document-chains)
- [智能体](#agents)
  - Agent Hooks ✨ v0.11.0
  - Agent 流式输出 ✨ v0.12.0
  - AgentBuilder ✨ v0.14.0
  - Orchestrator ✨ v0.14.0
  - ToolPolicy ✨ v0.14.0
- [Plan-Execute 智能体](#plan-execute-agent)
- [Handoffs](#handoffs)
- [流式工具调用](#streaming-tool-calls)
- [护栏](#guardrails)
  - Guardable ✨ v0.15.0
  - 流式护栏 ✨ v0.15.0
  - 审计持久化 ✨ v0.15.0
- [Token 计数器](#token-counter)
- [会话](#sessions)
  - 会话生命周期 ✨ v0.15.0
  - 接入记忆系统 ✨ v0.15.0
- [MCP](#mcp)
  - MCPServer
  - ConnectionManager ✨ v0.15.0
  - SamplingGuard ✨ v0.15.0
  - MCPGateway ✨ v0.15.0
- [工具](#tools)
  - WikipediaTool
  - DuckDuckGoSearchTool
  - PythonREPLTool
  - 扩展工具 (HTTPTool / FileTool / SQLTool)
  - `#[tool]` 过程宏 ✨ v0.10.0
  - ToolRegistry ✨ v0.15.0
  - StructuredTool ✨ v0.15.0
  - SSRF 防护 ✨ v0.15.0
- [RAG](#rag)
  - RAGPipeline ✨ v0.15.0
  - ChromaDB
  - PGVectorStore
  - PineconeStore
  - SemanticSplitter
  - 统一 VectorStore trait ✨ v0.15.0
  - MetadataFilter ✨ v0.18.0
- [BM25](#bm25)
- [混合检索](#hybrid-retrieval)
- [文档加载器](#document-loaders)
  - HTMLLoader
  - DocxLoader ✨ v0.4.1
  - WebScraperLoader ✨ v0.4.1
  - SitemapLoader ✨ v0.4.1
- [MultiQueryRetriever](#multiqueryretriever)
- [HyDE 检索器](#hyde-retriever)
- [SelfQueryRetriever](#selfqueryretriever) ✨ v0.18.0
- [重排序](#reranking)
- [回调](#callbacks)
  - OtelHandler
- [评估](#evaluation)
  - 评估器（10 种类型）
  - EvalRunner
  - LLMAsJudge ✨ v0.15.0
  - PairwiseJudge ✨ v0.15.0
- [LangGraph](#langgraph)
  - Reducer ✨ v0.15.0
  - 边类型 ✨ v0.15.0
  - Checkpointer 家族 ✨ v0.15.0
  - 子图 / 动态规划 / 流式 ✨ v0.15.0
- [A2A 智能体协议](#a2a-agent-protocol) ✨ v0.4.1
- [with_structured_output](#with_structured_output) ✨ v0.4.1
- [FileVectorStore](#filevectorstore) ✨ v0.4.1
- [ComputerUseTool](#computerusetool) ✨ v0.4.1
- [v0.5.0 新特性](#v050-new-features) ✨ v0.5.0
  - RouterLLM（模型路由 + 回退）
  - CorrectiveRAG
  - AdaptiveRAG
  - GraphRAG（知识图谱 RAG）
  - Deep Research 智能体
  - MCP 协议原语
  - 代码解释器沙箱
  - OpenAI Responses API
  - Anthropic Extended Thinking
  - 流式结构化输出
  - Batch API
  - 追踪（分布式追踪）
  - v0.5.0 质量加固（176 项修复）
- [v0.5.2 修复](#v052-fixes) ✨ v0.5.2
- [测试](#testing)
- [MongoDB 存储](#mongodb-storage)
- [Redis / SQLite 存储](#redis--sqlite-storage)

---

## 快速上手

> 这个教程带你从零搭一个 LLM 应用:先能**对话**,再能**记住上下文**,然后能**检索文档**,最后能**调用工具**。它是一段连续的程序,每一节在前一节的基础上加一块能力,照着往下读就能跑通。

### 1. 安装与环境变量

在 `Cargo.toml` 中加入:

```toml
[dependencies]
langchainrust = "0.15"
tokio = { version = "1", features = ["full"] }
```

设置环境变量(以 OpenAI 为例;换 Provider 只需换环境变量,见 [LLM](#llm)):

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # 可选,默认官方地址
```

### 2. 第一次聊天

最直接的用法:构造一个 LLM,传进系统提示词和用户消息,拿到回复。

```rust
use langchainrust::{BaseChatModel, OpenAIChat, OpenAIConfig};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = OpenAIChat::new(OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")?,
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    });

    let response = llm.chat(
        vec![
            Message::system("你是一个简洁的 Rust 助手。"),
            Message::human("用一句话说明什么是 Rust。"),
        ],
        None,
    ).await?;

    println!("{}", response.content);
    Ok(())
}
```

要点:
- 11 家 Provider 都实现同一个 `BaseChatModel` trait,换 Provider 只改 `XxxConfig` 一处;
- `Message::system` / `Message::human` 构造消息,`chat(Vec<Message>, None)` 返回完整 `LLMResult`;
- 想边生成边看?用 `stream_chat()` 或 `config.streaming = true`,见 [流式输出](#流式输出)。

### 3. 提示词模板

把系统/用户消息模板化,运行期用变量填充,避免写死字符串:

```rust
use langchainrust::{ChatPromptTemplate, Message, Runnable};
use std::collections::HashMap;

// 复用第 2 步的 llm
let prompt = ChatPromptTemplate::from_messages([
    Message::system("你是一个翻译助手。"),
    Message::human("请把「{text}」翻译成英文。"),
]);

let mut vars = HashMap::new();
vars.insert("text".to_string(), "你好,世界".to_string());

// ChatPromptTemplate 本身是 Runnable,可单独执行,输出 Vec<Message>
let messages = prompt.invoke(vars, None).await?;
```

要点:
- 消息用 `{变量名}` 占位,`invoke` 时传入 `HashMap<String, String>` 填充;
- 变量缺失会**响亮报错**,不会静默产出坏提示词;
- 完整介绍见 [提示词](#提示词)。

### 4. 用 LCEL 组合成一条链

`Runnable` 之间用 `.pipe()` 组合。`prompt.pipe(llm).pipe(parser)` 就是一条完整链,调用它一次拿到最终答案:

```rust
use langchainrust::{ChatPromptTemplate, Message, OpenAIChat, Runnable, StrOutputParser};
use std::collections::HashMap;

// 复用第 2 步的 llm
let prompt = ChatPromptTemplate::from_messages([
    Message::system("你是一个简洁的 Rust 助手。"),
    Message::human("{question}"),
]);

let chain = prompt.pipe(llm).pipe(StrOutputParser::new());

let mut vars = HashMap::new();
vars.insert("question".to_string(), "什么是所有权系统?".to_string());
let answer: String = chain.invoke(vars, None).await?;
println!("{answer}");
```

要点:
- `StrOutputParser` 从 `LLMResult` 里取 `content`,链的输出类型变成 `String`;
- 四个基础操作统一:`invoke` / `batch` / `stream` / `transform`;
- LCEL 全部操作符见 [LCEL 章节](#lcel-langchain-expression-language-)。

### 5. 加记忆:让它记住你

`RunnableWithMessageHistory` 把「读记忆 → 拼输入 → LLM → 写回」整个封装成一个 Runnable,多轮对话不用自己拼历史:

```rust
use langchainrust::{
    ConversationBufferMemory, OpenAIChat, Runnable, RunnableWithMessageHistory, StrOutputParser,
};

// 复用第 2 步的 llm
let memory = ConversationBufferMemory::new().with_return_messages(true);

let chat = RunnableWithMessageHistory::new(llm, memory).pipe(StrOutputParser::new());

let r1: String = chat.invoke("我叫小明,请记住我。".to_string(), None).await?;
let r2: String = chat.invoke("我叫什么名字?".to_string(), None).await?;
// r2 会答出"小明"
```

要点:
- 输入直接是 `String`,记忆的读写被封装在 Runnable 内部;
- 四种记忆各有取舍(全量 / 滑动窗口 / 摘要 / 摘要+原文),见 [记忆](#memory);
- 跨进程持久化用 `MongoPersistentMemory`,见 [MongoPersistentMemory](#mongopersistentmemory)。

### 6. 接上检索:RAG

`RAGPipelineBuilder` 组装「检索 + 生成」,`RagRunnable` 把它变成链的一段。下面用 **BM25 本地关键词检索**,不依赖任何向量数据库:

```rust
use langchainrust::{BM25Retriever, Document, OpenAIChat, RAGPipelineBuilder, RagRunnable, Runnable};
use std::sync::Arc;

// 复用第 2 步的 llm
let mut retriever = BM25Retriever::new();
retriever.add_documents_sync(vec![
    Document::new("Rust 是一门系统编程语言,注重安全和性能。").with_id("rust_intro"),
    Document::new("Rust 的核心特性包括所有权系统、借用检查和零成本抽象。").with_id("rust_features"),
]);

let pipeline = RAGPipelineBuilder::new()
    .llm(llm)
    .retriever(retriever)
    .retrieve_k(2)
    .build()?;

let rag_chain = RagRunnable::new(Arc::new(pipeline));

let answer: String = rag_chain.invoke("Rust 有哪些核心特性?".to_string(), None).await?;
```

要点:
- 只有回答生成走 LLM,检索全部本地完成,零向量库也能跑;
- 想要引用来源,`RAGPipeline::query_with_sources()` 返回每条引文,见 [端到端 RAGPipeline](#end-to-end-ragpipeline);
- 向量 / BM25 / 混合检索怎么选,见 [检索模式对比](#检索模式对比)。

### 7. 给它工具:变成智能体

让应用不只是问答,还能**决定调用哪个工具**。`FunctionCallingAgent` 读模型的 `tool_calls`,`AgentExecutor` 负责执行:

```rust
use langchainrust::tools::Calculator;
use langchainrust::{AgentExecutor, BaseAgent, BaseTool, FunctionCallingAgent, OpenAIChat};
use std::sync::Arc;

// 复用第 2 步的 llm
let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
let agent = FunctionCallingAgent::new(llm, tools.clone(), None);

let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
    .with_max_iterations(3)
    .with_verbose(true);

let result = executor.invoke("25 + 17 等于多少?".to_string()).await?;
```

要点:
- `FunctionCallingAgent` 是推荐路径(原生 tool_calls);不支持函数调用的模型用 `ReActAgent`;
- `max_iterations` 上限保护、工具超时、LLM 重试都由 Executor 兜底,见 [智能体](#agents)。

### 下一步

| 想做什么 | 去读 |
|----------|------|
| 换别家模型 / 流式 / 结构化输出 | [LLM](#llm) |
| 多轮记忆与持久化 | [记忆](#memory) |
| 完整 RAG 与检索策略 | [RAG](#rag) · [BM25](#bm25) · [混合检索](#hybrid-retrieval) |
| 接入 MCP 工具生态 | [MCP](#mcp) |
| 生产级护栏 / 评估 / 追踪 | [护栏](#guardrails) · [评估](#evaluation) · [回调](#callbacks) |

---

## LLM

本节讲怎么接入大模型:实例化任意一家 Provider、流式输出、函数调用、多模态。所有 Provider 都实现同一个 `BaseChatModel` trait,API 完全一致——先用一家把流程跑通,后面随时可换,不用改业务代码。初次上手请看[快速上手](#快速上手)第 2 节。

### 多 Provider 支持

LangChainRust 支持多个 LLM Provider，提供统一的 API：

| Provider | 类 | 特性 |
|----------|-------|----------|
| **OpenAI** | `OpenAIChat` | GPT-4, GPT-4o, GPT-3.5-turbo |
| **DeepSeek** | `DeepSeekChat` | DeepSeek-V3，高性价比 |
| **Moonshot** | `MoonshotChat` | Kimi，长上下文 |
| **Qwen** | `QwenChat` | 阿里云 |
| **Zhipu** | `ZhipuChat` | ChatGLM |
| **Anthropic** | `AnthropicChat` | Claude，注重安全 |
| **Ollama** | `OllamaChat` | 本地部署 |
| **Gemini** | `GeminiChat` | Google Gemini，多模态 |
| **Azure** | `AzureChat` | Azure OpenAI，企业合规 |
| **Cohere** | `CohereChat` | Command R+，RAG 场景 |
| **Mistral** | `MistralChat` | Mistral Large/Medium |

#### 统一客户端与自动发现 ✨ v0.15.0

`LLMClient::from_env()` 自动识别 11 家 Provider 的环境变量,零配置切换;`LLMClient::from_llm(provider)` 手动包装任意 `BaseChatModel`。`ChatModelWrapper` / `wrap_chat_model` 提供 trait 对象包装。

```rust
use langchainrust::LLMClient;
use langchainrust::language_models::ProviderError;

// 自动探测:哪个 Provider 配了环境变量就用哪个
let llm = LLMClient::from_env()?;
let response = llm.chat(vec![Message::human("Hello")], None).await?;

// 原生 Provider 也可以被包装
let client = LLMClient::from_llm(DeepSeekChat::from_env());
```

错误类型统一为 `ProviderError`,按供应商区分变体(OpenAI / Anthropic / Gemini / Azure / Cohere / Ollama / DeepSeek / Qwen / Moonshot / Zhipu / Mistral),`config.streaming` 决定 `chat()` 走流式还是普通路径。

#### DeepSeek（高性价比）

```rust
use langchainrust::{DeepSeekChat, BaseChatModel};
use langchainrust::schema::Message;

// 从环境变量读取
let llm = DeepSeekChat::from_env();

// 或手动配置
let llm = DeepSeekChat::with_model("deepseek-chat");

let response = llm.chat(vec![
    Message::human("Explain Rust ownership"),
], None).await?;
```

#### Moonshot（长上下文）

```rust
use langchainrust::MoonshotChat;

let llm = MoonshotChat::with_model("moonshot-v1-128k");  // 128K 上下文

let response = llm.chat(vec![
    Message::human("Analyze this long document..."),
], None).await?;
```

#### Qwen

```rust
use langchainrust::QwenChat;

let llm = QwenChat::from_env();  // 或 QwenChat::with_model("qwen-plus")

let response = llm.chat(vec![
    Message::human("Explain microservices in Chinese"),
], None).await?;
```

#### Zhipu（ChatGLM）

```rust
use langchainrust::ZhipuChat;

let llm = ZhipuChat::from_env();  // 或 ZhipuChat::with_model("glm-4")

let response = llm.chat(vec![
    Message::human("Write Rust concurrent code"),
], None).await?;
```

#### Anthropic Claude

```rust
use langchainrust::{AnthropicChat, AnthropicConfig};

let config = AnthropicConfig {
    api_key: std::env::var("ANTHROPIC_API_KEY")?,
    model: "claude-3-opus-20240229".to_string(),
    ..Default::default()
};
let llm = AnthropicChat::new(config);

let response = llm.chat(vec![
    Message::human("Analyze this code safely"),
], None).await?;
```

### Google Gemini

```rust
use langchainrust::{GeminiChat, GeminiConfig, BaseChatModel};
use langchainrust::schema::Message;

let config = GeminiConfig {
    api_key: std::env::var("GEMINI_API_KEY")?,
    model: "gemini-2.0-flash".to_string(),
    ..Default::default()
};

let llm = GeminiChat::new(config);

let response = llm.chat(vec![
    Message::human("Explain Rust enums"),
], None).await?;
```

### OpenAI Chat

使用 OpenAI GPT 系列模型。支持自定义 base_url（兼容所有 OpenAI API 格式的服务），temperature 控制随机性。

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

let config = OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY")?,
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-3.5-turbo".to_string(),
    temperature: Some(0.7),
    ..Default::default()
};

let llm = OpenAIChat::new(config);

let response = llm.chat(vec![
    Message::system("You are a helpful assistant."),
    Message::human("What is Rust?"),
], None).await?;

println!("{}", response.content);
```

### 流式输出

LLM 生成文本是逐 token 的，流式输出让你实时看到每个 token，而不是等整个回答完成。适合聊天界面、实时展示等场景。

```rust
use futures_util::StreamExt;

let config = OpenAIConfig {
    streaming: true,
    ..Default::default()
};

let llm = OpenAIChat::new(config);

let mut stream = llm.stream_chat(vec![
    Message::human("Write a short story"),
], None).await?;

while let Some(chunk) = stream.next().await {
    if let Ok(chunk) = chunk {
        print!("{}", chunk.text);  // 实时输出(StreamChunk 的文本字段)
        // 流结束时 chunk.token_usage 携带 token 用量(provider 支持时)
    }
}
```

### 函数调用

让 LLM 决定何时调用工具。`bind_tools` 将工具定义附加到 LLM，LLM 返回 `tool_calls` 而非纯文本。框架负责解析参数、调用工具、返回结果。

```rust
use langchainrust::ToolDefinition;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(JsonSchema, Deserialize)]
struct CalculatorInput {
    expression: String,
}

let tool = ToolDefinition::from_type::<CalculatorInput>(
    "calculator",
    "Evaluate mathematical expressions"
);

let llm_with_tools = llm.bind_tools(vec![tool]);

let response = llm_with_tools.chat(vec![
    Message::human("Calculate 25 + 17"),
], None).await?;

if let Some(tool_calls) = response.tool_calls {
    for call in tool_calls {
        println!("Tool: {}", call.function.name);
        println!("Args: {}", call.function.arguments);
    }
}
```

### Ollama（本地 LLM）

Ollama 让你在本地运行开源模型（Llama、Mistral 等），无需 API Key，数据不出本机。适合隐私敏感场景或离线使用。

```rust
use langchainrust::{OllamaChat, OllamaConfig};

let config = OllamaConfig {
    base_url: "http://localhost:11434".to_string(),
    model: "llama2".to_string(),
    ..Default::default()
};

let llm = OllamaChat::new(config);

let response = llm.chat(vec![
    Message::human("Hello!"),
], None).await?;
```

### 多模态视觉

`ImageContent` 表示一张图片（URL 或 base64 数据 URI）。使用 `Message::human_with_image` 构建包含图片的消息；`OpenAIChat` 和 `OllamaChat` 会自动将其序列化为各自原生的多模态格式。

```rust
use langchainrust::schema::{ImageContent, Message};
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};

let msg = Message::human_with_image("Describe this image", "https://example.com/cat.jpg");
// 或多张图片：
// let msg = Message::human_with_images("Compare these two", vec![
//     ImageContent::from_url("https://example.com/a.jpg"),
//     ImageContent::from_base64_with_mime(base64_str, "image/png"),
// ]);

let llm = OpenAIChat::new(OpenAIConfig::default());
let resp = llm.chat(vec![msg], None).await?;
println!("{}", resp.content);
```

`ImageContent::from_url(url)` / `from_base64(data)` / `from_base64_with_mime(data, mime)`；也可以链式调用 `Message::human(text).with_image(ImageContent)`。`OllamaChat` 同样适用。

### Message 结构 ✨ v0.15.0

`Message` 是统一的对话消息结构，除文本 `content` 外还携带多模态附件与工具调用：

| 字段 | 类型 | 说明 |
|------|------|------|
| `content` | `String` | 文本内容 |
| `images` / `audio` / `files` | `Vec<...>` | 图片 / 音频 / 文件附件 |
| `message_type` | `MessageType` | System / Human / Ai |
| `tool_calls` | `Option<Vec<ToolCall>>` | AI 消息携带的待执行工具调用 |
| `name` / `id` / `additional_kwargs` | — | 角色名 / 消息 ID / 额外字段 |

```rust
use langchainrust::schema::{Message, AudioContent, FileContent, ToolCall};

// 带音频 / 文件的消息
let msg = Message::human_with_audio("转录这段音频", AudioContent::from_base64(data));
let msg = Message::human_with_file("读取这个文件", FileContent::from_url("file:///tmp/doc.pdf"));

// AI 发起工具调用
let msg = Message::ai_with_tool_calls("", vec![
    ToolCall::builder("call_1")
        .name("calculator")
        .arguments(r#"{"expression":"25+17"}"#)
        .build(),
]);
```

构造器覆盖常见组合：`Message::system/human/ai`、`human_with_image(s)`、`human_with_audio`、`human_with_file`、`ai_with_tool_calls`；serde 向后兼容（附件字段带 `#[serde(default)]`，旧数据可反序列化）。

### MultimodalModel（多模态能力） ✨ v0.15.0

`MultimodalModel` trait（`BaseChatModel` 扩展）声明语音识别 / 语音合成 / 文生图三个能力接口。**默认实现返回 `MultimodalError::Unsupported`**——只有真正支持该能力的 Provider 才覆盖，避免"看似可用实则报错"的假多模态。OpenAI 系列已实现；其余 Provider 调用对应方法会得到明确的 Unsupported 错误而非静默失败。

```rust
use langchainrust::MultimodalModel;

let text = llm.transcribe(AudioContent::from_base64(data)).await?; // 仅支持 Provider
// let audio = llm.generate_speech("hello").await?;
// let img = llm.generate_image("一只猫").await?; → 不支持时 Err(Unsupported)
```

---

### OpenAI Assistants API

`OpenAIAssistant` 封装了官方 OpenAI Assistants API（Assistants / Threads / Run），具有服务端会话状态，适合多轮复杂任务。需要 OpenAI 官方端点；部分兼容模式端点可能不支持。

```rust
use langchainrust::{OpenAIAssistant, OpenAIConfig};

let config = OpenAIConfig::default();
let assistant = OpenAIAssistant::create(&config, "gpt-4o", "You are a translator").await?;
// 或复用已有助手：OpenAIAssistant::from_id(config, "asst_xxx")

let answer = assistant.run_once("Translate: Hello").await?;
```

**Run 状态**：带工具调用的 Run（`requires_action`）在 `run_once` 的轮询循环内自动处理——`handle_requires_action` 会向 Assistant API 提交工具输出后继续轮询，直至 `completed` 或 `failed`。

<a id="prompts"></a>
## 提示词

提示词模板将变量占位符（`{name}`）替换为实际值，避免手动拼接字符串。框架提供三种模板，覆盖从简单到复杂的所有场景。

### PromptTemplate

最基础的模板——单条文本，用 `{variable}` 占位。适合不需要区分角色、只需拼一段 prompt 的场景。

```rust
use langchainrust::prompts::PromptTemplate;
use std::collections::HashMap;

let template = PromptTemplate::new("Hello, {name}! Today is {day}.");

let vars = HashMap::from([
    ("name", "Alice"),
    ("day", "Monday"),
]);

let prompt = template.format(&vars)?;
// 输出："Hello, Alice! Today is Monday."
```

**模板语法细节**（`PromptTemplate` / `ChatPromptTemplate` 通用）：

- **花括号转义**：`{{` → 字面 `{`，`}}` → 字面 `}`（写 JSON 模板时常用：`"请输出 JSON: {{\"key\": \"{value}\"}}"`）
- **变量命名**：支持中文/下划线开头等宽字符：`{中文名}`、`{_private}`、`{a1}` 均可
- **缺失变量报错**：模板引用了变量但未提供时，`format` 返回包含 `missing` 的明确错误，**不会**静默保留 `{var}` 原文——避免脏 prompt 悄悄进 LLM
- **FewShot 后缀同样校验**：后缀中的未声明变量也报错（原先会静默保留 `{answer}` 原文，已修复）

### ChatPromptTemplate

多轮消息模板——每条消息有角色（system/human/ai），变量在消息文本中替换。适合需要设定系统角色、区分对话轮次的场景，是 Agent 和 Chain 中最常用的模板。

```rust
use langchainrust::prompts::ChatPromptTemplate;
use langchainrust::schema::Message;

let template = ChatPromptTemplate::new(vec![
    Message::system("You are a {role} expert in {domain}."),
    Message::human("Hello, I'm {name}."),
    Message::human("{question}"),
]);

let vars = HashMap::from([
    ("role", "programming"),
    ("domain", "Rust"),
    ("name", "Bob"),
    ("question", "Explain ownership"),
]);

let messages = template.format(&vars)?;
```

### FewShotPromptTemplate

少样本模板——在 prompt 前插入若干"输入→输出"示例，教 LLM 按特定格式回答。适合需要引导输出格式（如翻译、情感分析、格式转换）的场景。LLM 看到示例后，会模仿示例的格式来回答。

**工作原理**：将前缀 + 每个示例（通过 `example_prompt` 格式化）+ 后缀拼接成完整 prompt，LLM 看到的是一段包含例子的完整文本。

```rust
use langchainrust::prompts::{FewShotPromptTemplate, PromptTemplate};
use std::collections::HashMap;

let examples = vec![
    HashMap::from([("input", "happy"), ("output", "sad")]),
    HashMap::from([("input", "tall"), ("output", "short")]),
];

let example_prompt = PromptTemplate::new("Input: {input}\nOutput: {output}");

let prompt = FewShotPromptTemplate::new(
    examples,
    example_prompt,
    "Input: {input}\nOutput:",
);
```

### ExampleSelectors

当示例很多时，不需要全部塞给 LLM——选择器按策略挑选最相关的示例，节省 token 并提高质量。

```rust
use langchainrust::prompts::LengthBasedExampleSelector;

// 基于长度：选择不超过最大长度的示例
let selector = LengthBasedExampleSelector::new(examples) // examples: Vec<HashMap<String, String>>
    .with_max_length(50);
```

---

<a id="output-parsers"></a>
## 输出解析器

LLM 返回的是纯文本字符串，输出解析器将其转换为结构化数据。选择哪个解析器取决于你需要什么格式：

| 解析器 | 输入 | 输出 | 适用场景 |
|--------|------|------|----------|
| `StrOutputParser` | 任意文本 | 原样字符串 | 只需文本，不做转换 |
| `CommaSeparatedListOutputParser` | 逗号分隔文本 | `Vec<String>` | LLM 输出列表 |
| `JsonOutputParser` | JSON 文本 | `serde_json::Value` | 需要灵活的 JSON 结构 |
| `StructuredOutputParser` | `key: value` 文本 | `HashMap<String, String>` | 简单键值对，无需 JSON |
| `TypedOutputParser<T>` | JSON 文本 | 强类型 `T` | 需要类型安全的结构化输出 |

> **提示**：如果 LLM 支持 Function Calling，优先使用 `with_structured_output()`——它比解析器更可靠。

### StrOutputParser

最简单的解析器——原样返回文本。通常作为 LCEL 管道的最后一步，确保输出类型是 `String`。

```rust
use langchainrust::output_parsers::{StrOutputParser, BaseOutputParser};

let parser = StrOutputParser::new();
let result = parser.parse("Hello world")?;
```

### CommaSeparatedListOutputParser

将逗号分隔的文本解析为字符串列表。适合让 LLM 列举项目、标签、关键词等场景。

```rust
use langchainrust::output_parsers::CommaSeparatedListOutputParser;

let parser = CommaSeparatedListOutputParser::new();
let result = parser.parse("apple, banana, cherry")?;
```

### JsonOutputParser

从 LLM 输出中提取 JSON。支持完整 JSON 和从 markdown 代码块中提取部分 JSON（LLM 经常把 JSON 包在 ` ```json ``` ` 里）。

```rust
use langchainrust::output_parsers::JsonOutputParser;
use serde_json::Value;

// 完整 JSON 解析
let parser = JsonOutputParser::<Value>::new();
let result: Value = parser.parse(r#"{"name": "Rust"}"#)?;

// 部分解析（从 markdown 中提取 JSON）
let partial = parser.parse_partial("Here is the JSON:\n```json\n{\"name\": \"Rust\"\n}")?;
```

### StructuredOutputParser

将 `key: value` 格式的文本解析为 HashMap。比 JsonOutputParser 更宽松——LLM 不需要输出严格的 JSON 格式，只需按行写 `key: value` 即可。

```rust
use langchainrust::output_parsers::StructuredOutputParser;
use std::collections::HashMap;

let parser = StructuredOutputParser::new(vec![
    ("name".to_string(), "string".to_string()),
    ("age".to_string(), "integer".to_string()),
]);

let result: HashMap<String, String> = parser.parse(
    "name: Alice\nage: 30"
)?;
```

### TypedOutputParser\<T\>

将 JSON 文本反序列化为强类型结构体。需要 `T` 实现 `Deserialize`。比 `JsonOutputParser<Value>` 更安全——编译时就能检查字段类型。

```rust
use langchainrust::output_parsers::TypedOutputParser;
use serde::Deserialize;

#[derive(Deserialize)]
struct Person {
    name: String,
    age: u32,
}

let parser = TypedOutputParser::<Person>::new();
let person: Person = parser.parse(
    r#"{"name": "Alice", "age": 30}"#
)?;
```

---

<a id="memory"></a>
## 记忆

记忆给链或智能体加"上下文":让多轮对话记住前面说了什么,而不用每次把整段历史塞进提示词。四种内置记忆各有取舍:

| 记忆 | 行为 | 适合 | 代价 |
|------|------|------|------|
| `ConversationBufferMemory` | 保留全部对话 | 短对话、信息不能丢 | token 随轮次线性增长 |
| `ConversationBufferWindowMemory` | 只留最近 k 轮 | 长对话、旧细节不重要 | 旧内容直接丢弃 |
| `ConversationSummaryBufferMemory`(推荐) | 旧消息摘要 + 近期原文 | 长对话且要近期细节 | 摘要消耗一次 LLM 调用 |
| `VectorStoreRetrieverMemory` | 按相似度检索记忆 | 知识型、联想式记忆 | 需要向量库 |

- 想跨进程持久化 → 用 `MongoPersistentMemory`(见下方);
- 想限制单次上下文长度 → 用 `ContextWindow` 自动截断/摘要;
- 所有记忆实现统一 `BaseChatMemory` trait,可即插即换。

### ConversationBufferMemory

保留所有对话历史：

```rust
use langchainrust::{ConversationBufferMemory, BaseMemory};

let mut memory = ConversationBufferMemory::new();

memory.save_context(
    HashMap::from([("input", "My name is Alice")]),
    HashMap::from([("output", "Hello Alice!")]),
).await?;

let vars = memory.load_memory_variables(&HashMap::new()).await?;
// 输出："Human: My name is Alice\nAI: Hello Alice!"
```

### ConversationBufferWindowMemory

仅保留最近 k 轮对话。当对话很长、不需要完整历史时使用，避免 token 超限。

```rust
use langchainrust::ConversationBufferWindowMemory;

// k=2，保留最近 2 轮（4 条消息）
let mut memory = ConversationBufferWindowMemory::new(2);

for i in 1..=5 {
    memory.save_context(
        HashMap::from([("input", format!("Question {}", i))]),
        HashMap::from([("output", format!("Answer {}", i))]),
    ).await?;
}

// 仅返回最近 2 轮，Q1-Q3 被丢弃
let vars = memory.load_memory_variables(&HashMap::new()).await?;
```

### ConversationSummaryBufferMemory（推荐）

对旧消息进行摘要压缩，保留近期消息原文。结合了 BufferMemory（保留近期细节）和 SummaryMemory（压缩旧内容）的优点，是长对话场景的最佳选择。

```rust
use langchainrust::ConversationSummaryBufferMemory;

let llm = OpenAIChat::new(config);

// max_token_limit = 100，超出时触发压缩
let mut memory = ConversationSummaryBufferMemory::new(llm, 100);

for i in 1..=10 {
    memory.save_context(&inputs, &outputs).await?;
}

// 返回："Summary: User discussed...\n\nHuman: Recent\nAI: Response"
let vars = memory.load_memory_variables(&HashMap::new()).await?;
```

| 记忆类型 | 压缩方式 | Token 控制 | 适用场景 |
|-------------|-------------|---------------|----------|
| BufferMemory | 无 | 无限制 | 短对话 |
| WindowMemory | 硬删除 | 固定 k | 简单控制 |
| SummaryMemory | LLM 摘要 | 动态 | 长对话 |
| SummaryBufferMemory | 混合 | 动态 + 保留近期 | 均衡（推荐） |

---

### VectorStoreRetrieverMemory

将每轮对话嵌入向量存储，根据当前输入的语义相似度召回 top-k 相关历史。与固定窗口的缓冲记忆相比，在长对话/跨会话场景中能保留更多有用的上下文。

```rust
use langchainrust::{VectorStoreRetrieverMemory, MockEmbeddings, BaseMemory};
use langchainrust::vector_stores::InMemoryVectorStore;
use std::collections::HashMap;

let mut memory = VectorStoreRetrieverMemory::new(
    InMemoryVectorStore::new(),
    MockEmbeddings::new(1536),
    4,
);

memory.save_context(&inputs, &outputs).await?;
let vars = memory.load_memory_variables(&HashMap::new()).await?;
```

**权衡**：语义召回在长对话中保留关键信息；但依赖向量存储 + 嵌入模型（额外成本）。

### 统一 BaseChatMemory trait ✨ v0.15.0

所有对话记忆实现统一的 `BaseChatMemory` trait（`save_context` / `load_memory_variables` / `clear`），可互换、可进入 LCEL 管道（`RunnableWithMessageHistory::new(llm, memory)` 直接收任意 `BaseMemory`）。

<a id="mongopersistentmemory"></a>
### MongoPersistentMemory（跨进程持久化） ✨ v0.15.0

把对话历史持久化到 MongoDB，服务重启不丢，多实例共享同一份记忆。内部组合 `ConversationSummaryBufferMemory`，自带 token 预算；并发写入用乐观锁防丢更新。

```rust
use langchainrust::memory::MongoPersistentMemory;

let mut memory = MongoPersistentMemory::new(
    "mongodb://localhost:27017",
    "chatdb",
    "sessions",
    llm,        // 任意 BaseChatModel,泛型 M
    2000,       // token 上限
).await?;

memory.set_session_id_async("user-123".to_string()).await;  // 绑定会话
memory.save_context(&inputs, &outputs).await?;
```

### 摘要失败的可见性

`ConversationSummaryMemory` / `ConversationSummaryBufferMemory` 提供 `last_summary_error() -> Option<&str>`：LLM 摘要步骤失败时不会吞错，调用方可读取上次失败原因并决定降级策略。

### ContextWindow（长上下文管理） ✨ v0.4.1

`ContextWindow` 管理长对话的 token 预算，提供两种策略：截断（Truncate）和摘要（Summarize）。

```rust
use langchainrust::{ContextWindow, Message, OpenAIChat, Strategy};

// 策略 1：Truncate — 超出 token 预算时丢弃最旧的消息
let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096)?;
let fitted = cw.fit(messages).await?;

// 策略 2：Summarize — 超出预算时使用 LLM 压缩旧对话
let cw = ContextWindow::with_strategy(4096, Strategy::summarize(llm))?;
let fitted = cw.fit(messages).await?;
```

| 策略 | 行为 | 适用场景 |
|----------|----------|----------|
| `Truncate` | 超出预算时丢弃最旧的消息 | 简单场景 |
| `Summarize` | LLM 将旧对话压缩为摘要 | 需要保留关键信息的长对话 |

> **细节**：`Truncate` 策略总是保留 `System` 消息（角色/指令不因截断而丢失）；`Summarize` 策略生成的摘要计入 token 预算，避免压缩后再超限。

<a id="llm-cache"></a>
## LLM 缓存

### 概念引入：为什么要缓存

LLM 调用是应用中最慢、最贵的部分——一次请求要跨网络、排队、生成，既耗时又花钱。当用户反复问同一个问题、或批处理里出现大量重复请求时，每次都真实调用 API 既慢又费预算。缓存的思路很朴素：**同样的输入直接复用上次的结果**，不再发起真实 LLM 调用。

什么时候用：

- 高频重复查询（如同一个问题发给多用户、同一批文档反复摘要）
- 批处理 / 评测中大量相似或相同的输入
- 结果确定性要求高、可容忍偶发陈旧

工作流程（2 步）：

1. 构造 `CacheConfig`，声明 TTL（存活时间）与容量上限
2. 创建 `LLMCache`，用 `build_key` 判键 + `get`/`put` 读写缓存（`LLMCache` 是独立组件，不自动挂到模型上）

### 工作机制

#### 判键逻辑

缓存以调用输入作为判键：输入完全一致（如相同的消息序列）时命中，直接复用上次结果；输入不同则视为新条目。判键决定了缓存"认不认识"这次请求——想让缓存生效，重复调用的输入就要保持一致。

#### TTL 过期

每条记录带 TTL（存活时间），超过多久自动失效。`with_ttl` 设置全局过期时间。过期条目在访问 / 淘汰时被清理，避免"旧答案一直占着位置"。

#### 容量限制 + LRU 淘汰

`with_max_entries` 声明最多缓存多少条。缓存满了要淘汰旧条目——采用 **LRU（最久未用）淘汰**：淘汰"最久没被访问过"的那条，而不是"最早插入"的那条。这样频繁命中的热点不会被新条目挤掉，缓存命中率才稳。v0.14.0 起，命中缓存会刷新条目的"最近使用时间"，淘汰语义为真正的 LRU。

#### 命中刷新

每次缓存命中（get）时刷新条目的"最近使用时间"，保证 LRU 语义正确——刚被用过的条目"变年轻"，不会被紧接着的插入挤掉。

### 带 TTL 的内存缓存

```rust
use langchainrust::core::cache::{CacheConfig, LLMCache};
use std::time::Duration;

let config = CacheConfig::new()
    .with_ttl(Duration::from_secs(3600))  // 1 小时
    .with_max_entries(1000);              // 1000 条记录

let cache = LLMCache::with_config(config);

// LLMCache 是独立组件，手动接入调用路径：build_key 判键，get/put 读写
let messages = vec![Message::human("Hello")];
let key = LLMCache::build_key(&messages, "gpt-4o")?;

if let Some(hit) = cache.get(&key).await {
    // 缓存命中，直接用缓存结果
    let result = hit.result;
} else {
    // 未命中，真实调用后写入缓存
    let result = llm.chat(messages, None).await?;
    cache.put(key, result).await;
}
```

### 关键行为一览

| 行为 | 说明 |
|---|---|
| 判键 | 相同输入直接复用上次结果，不再真实调用 |
| TTL 过期 | 超过存活时间自动失效 |
| 容量上限 | `with_max_entries` 控制最多缓存条数，满了触发淘汰 |
| LRU 淘汰 | 淘汰"最久未用"的条目，热点不被挤掉 |
| 命中刷新 | get 命中刷新最近使用时间，保证 LRU 语义正确 |

### 怎么选 / 使用建议

- 结果越稳定、请求越重复，缓存收益越大；`with_ttl` 设长一点能覆盖更多重复请求。
- 对实时性敏感的数据（价格、库存、最新状态）不建议缓存，或把 TTL 调短。
- 这是**内存级**缓存——进程重启即清空；需要跨进程 / 重启持久化时，应把"判键 → 结果"落到外部存储。

---

<a id="lcel-langchain-expression-language-"></a>
## LCEL (LangChain Expression Language) ✨ v0.9.0

LCEL 提供类似 Python LangChain 的管道组合语法：把 `Runnable` 组件通过 `.pipe()` 串联成流水线。与手写"取结果 → 传给下一步"的胶水代码不同，LCEL 里拼出来的**整条链本身还是一个 Runnable**——可以继续拼接、可以批量执行、可以流式输出、可以自动重试降级。

打个比方：如果把一次 LLM 调用比作一个加工工位，LCEL 就是**传送带**——把提示词模板、模型、解析器、记忆、检索器这些"工位"用 `.pipe()` 连起来，数据自动流过去；每加一段管道，就多一层能力。

### 为什么需要 LCEL

v0.14.0 之前，框架里每类组件"能不能进链"不一致，用户被迫到处写 `.content` 提取、手动拼消息、套包装：

| 组件 | v0.14.0 前 | 症状 |
|---|---|---|
| 解析器 | `Runnable<String, String>` | 接不住 LLM 的 `LLMResult`，`llm.pipe(parser)` 编译不过 |
| 提示词 | 无 Runnable | `ChatPromptTemplate` 只能手动 `format`，进不了链 |
| 记忆 | 无 Runnable | 手写"读记忆 → 调模型 → 写回"胶水 |
| 原生 Provider | 错误不进 `LcelError` | 只有套 `LLMClient` 才能 pipe |

v0.15.0 统一后，提示词、记忆、原生 Provider、解析器、RAG 全部是 Runnable，一条链跑通，见[统一组合 (v0.15.0)](#unified-lcel)。

### Runnable：统一的"可执行单元"

框架里各种东西——提示词模板、聊天模型、输出解析器、Agent——调用方式原本各不相同。`Runnable` trait 给它们一个统一接口，任何组件都能用同一套姿势执行、互相拼接。

**四个基本动作**：

| 动作 | 作用 | 例子 |
|---|---|---|
| `invoke` | 单次执行：给一个输入，拿一个输出 | 问一次模型，拿一次回答 |
| `batch` | 批量执行：一批输入 → 一批输出 | 100 条评论一次性打标签 |
| `stream` | 流式执行：结果一个接一个吐出来 | 模型边生成边显示文字 |
| `transform` | 流到流：输入流 → 输出流，边进边出 | 长文档分块处理，不等全部读完 |

关键行为：
- **所有组件自动具备这四个能力**——即便某个单元只实现了 `invoke`，其余三个也有默认实现兜底（默认行为是"调一次 invoke，把结果包成流 / 循环多次"）。
- `batch` 支持**并发**：通过 `RunnableConfig.max_concurrency` 控制同时跑多少个，不设就是按顺序跑。
- 真正的逐字流式（首 token 秒回）需要组件自己覆写 `stream`——语言模型做了，普通函数没有。

### RunnableConfig：一次执行的"设置单"

每次执行都想带点附加信息——这条执行属于哪个业务、要不要挂回调、能不能取消。不用改代码，传一张配置单 `RunnableConfig` 就行。

| 配置项 | 作用 |
|---|---|
| `tags` | 给这次执行打标，方便筛选/追踪（如 `"user-123"`、`"rag-流程"`） |
| `metadata` | 任意键值对，业务信息随便塞 |
| `max_concurrency` | 批量执行时的并行度 |
| `run_id` / `run_name` | 这次运行的唯一 ID 和名字，用于链路追踪 |
| `callbacks` | 挂回调管理器，监听执行过程中的事件 |
| `cancellation_token` | 取消令牌，长任务可以从外部叫停 |

层级配置合并规则：
- **标签**：保序去重合并（父的顺序保持，重复的去掉，不排序）。
- **其它字段**：子配置有值的**覆盖**父配置。
- **回调**：父子**叠加**，两个都会触发。

### 组合算子：把 Runnable 拼成管道

真实应用不是"调一次模型"这么简单，而是"检索 → 拼提示词 → 问模型 → 解析结果"这种多步骤流程。这些算子就是**搭积木**：把简单单元拼成复杂流程，拼出来的东西本身还是个 Runnable，还能继续拼。串联时**类型安全**——编译器保证"上一步的输出类型 == 下一步的输入类型"，写错了编译不过。

| 算子 | 作用 | 例子 |
|---|---|---|
| `pipe`（串联） | A 的输出喂给 B，一步接一步 | 提示词 → 模型 → 解析器，一行写完 |
| `RunnableLambda`（包函数） | 把普通函数塞进管道当单元 | 清洗文本等自定义处理逻辑 |
| `RunnablePassthrough`（透传） | 输入原样传下去，不做处理 | RAG 里"问题"要一路带到最后 |
| `RunnableParallel`（并行） | 同一份数据同时走多条线，结果合成一个 map | 一条线检索文档、一条线保留原问题 |
| `RunnableBranch`（分支） | 按输入内容选分支，都不匹配走默认 | 退款问题走售后分支，技术问题走技术分支 |
| `RunnableBinding`（绑定） | 给单元绑死固定配置 | 整条链固定走某套模型配置 |
| `RunnableAssign`（附加） | 在数据上追加新字段 | 原问题上再附一份大写版本 |
| `RunnableWithFallbacks`（降级） | 主单元失败，自动换备用单元 | 主模型不可用自动切备用模型 |
| `with_retry`（重试） | 失败自动再试，带退避等待 | 网络抖动、临时限流不至于整个流程挂掉 |

### 基本管道

```rust
use langchainrust::{
    RunnableExt, RunnableLambda, RunnablePassthrough,
};

// 创建简单的管道: 输入 -> 加倍 -> 转字符串
let doubler = RunnableLambda::new_sync(|x: i32| x * 2);
let formatter = RunnableLambda::new_sync(|x: i32| format!("Result: {}", x));

let chain = doubler.pipe(formatter);
let result = chain.invoke(5, None).await?;
// result = "Result: 10"
```

### 三步管道 (Prompt | LLM | Parser)

```rust
use langchainrust::{RunnableExt, RunnableLambda, StrOutputParser};

let prompt = RunnableLambda::new_sync(|query: String| {
    format!("请回答以下问题：{}", query)
});
let parser = RunnableLambda::new_sync(|output: String| {
    output.trim().to_string()
});

// prompt.pipe(llm).pipe(parser) — LLM 步骤需要真实 API
let chain = prompt.pipe(parser);
let result = chain.invoke("什么是Rust?".to_string(), None).await?;
```

### RunnableLambda (包函数)

普通函数也能进管道：`new_sync` 包同步闭包、`new_sync_fallible` 包可失败的同步闭包、`new_async` 包异步闭包。这样清洗文本、拼字符串、发请求等任意自定义逻辑都能成为链的一段，返回值自动包成 `Result<_, LcelError>`。

```rust
use langchainrust::{LcelError, RunnableExt, RunnableLambda};

// new_sync：同步闭包，输出自动包成 Ok
let clean = RunnableLambda::new_sync(|s: String| s.trim().to_string());

// new_async：异步闭包，返回 Result<O, LcelError>
let fetch = RunnableLambda::new_async(|url: String| async move {
    Ok(format!("fetched {}", url.trim()))
});

let chain = clean.pipe(fetch);
let result = chain.invoke("  https://example.com  ".to_string(), None).await?;
// result = "fetched https://example.com"
```

### RunnablePassthrough (透传)

```rust
use langchainrust::RunnablePassthrough;

// Passthrough 直接传递输入，不修改
let passthrough = RunnablePassthrough::<String>::new();
let result = passthrough.invoke("hello".to_string(), None).await?;
// result = "hello"

// 真流式: transform 直接传递输入流，不缓冲
let stream = passthrough.transform(input_stream, None).await;
```

### RunnableParallel (扇出/扇入)

```rust
use langchainrust::{RunnableExt, RunnableLambda, RunnableParallel};

let doubler = RunnableLambda::new_sync(|x: i32| x * 2);
let tripler = RunnableLambda::new_sync(|x: i32| x * 3);

let parallel = RunnableParallel::new()
    .with("double", doubler)
    .with("triple", tripler);

let result = parallel.invoke(5, None).await?;
// result = {"double": 10, "triple": 15}
```

### RunnableBranch (条件路由)

```rust
use langchainrust::{RunnableExt, RunnableLambda, RunnableBranch};

let short_handler = RunnableLambda::new_sync(|s: String| format!("短: {}", s));
let long_handler = RunnableLambda::new_sync(|s: String| format!("长: {}", s));
let default_handler = RunnableLambda::new_sync(|s: String| format!("默认: {}", s));

let branch = RunnableBranch::new(default_handler)
    .when(
        RunnableLambda::new_sync(|s: String| s.len() < 5),
        short_handler,
    )
    .when(
        RunnableLambda::new_sync(|s: String| s.len() >= 10),
        long_handler,
    );

let result = branch.invoke("hi".to_string(), None).await?;
// result = "短: hi"
```

### RunnableBinding (配置绑定)

```rust
use langchainrust::{RunnableBinding, RunnableConfig};

// 预绑定配置和 kwargs
let bound = runnable
    .bind("temperature", serde_json::json!(0.7))
    .with_config(RunnableConfig::new().with_tag("production"));
let result = bound.invoke(input, None).await?;
```

### Batch 批量执行

```rust
let results = chain.batch(vec![1, 2, 3], None).await?;
// results = ["Result: 2", "Result: 4", "Result: 6"]
```

### Stream 流式执行

```rust
use futures_util::StreamExt;

let mut stream = chain.stream("hello".to_string(), None).await?;
while let Some(item) = stream.next().await {
    println!("Token: {}", item?);
}
```

### RunnableWithFallbacks (降级回退) ✨ v0.10.0

```rust
use langchainrust::{RunnableExt, RunnableLambda};

let primary = RunnableLambda::new_sync(|x: i32| -> i32 {
    if x < 0 { panic!("negative") } else { x * 2 }
});
let fallback = RunnableLambda::new_sync(|x: i32| x.abs() * 2);

// primary 失败时自动切换到 fallback
let chain = primary.with_fallbacks(vec![fallback.into_runnable_any()]);
let result = chain.invoke(-5, None).await?;
// result = 10 (fallback 执行)
```

### RunnableAssign (字段注入) ✨ v0.10.0

```rust
use langchainrust::{
    RunnableExt, RunnableLambda, RunnableParallel, RunnablePassthrough,
    core::runnables::RunnableAssign,
};
use std::collections::HashMap;
use serde_json::Value;

// RunnableParallel.assign() — 在 parallel 输出的 HashMap 中注入新字段
let parallel = RunnableParallel::new()
    .with("question", RunnablePassthrough::<String>::new())
    .with("context", RunnableLambda::new_sync(|_: String| "some context".to_string()));

// assign 在 parallel 输出后追加字段
let chain = parallel.assign("answer", RunnableLambda::new_sync(|map: HashMap<String, Value>| {
    let ctx = map.get("context").unwrap().as_str().unwrap();
    format!("Based on: {}", ctx)
}));

let result = chain.invoke("What is Rust?".to_string(), None).await?;
// result = {"question": "What is Rust?", "context": "some context", "answer": "Based on: some context"}
```

### RunnableRetry (自动重试) ✨ v0.11.0

`with_retry(RetryConfig)` 包装任意 Runnable，失败时按指数退避自动重试。

```rust
use langchainrust::{
    RunnableExt, RunnableLambda, core::runnables::{RetryConfig, RetryOn},
};
use std::time::Duration;

let flaky = RunnableLambda::new_sync(|x: i32| {
    if rand::random::<f32>() < 0.3 { panic!("transient") } else { x }
});

// 默认:最多 3 次,指数退避 0.5s→10s,仅对瞬时错误重试
let chain = flaky.with_retry(RetryConfig::default());

// 自定义:最多 5 次,初始 100ms,倍增 2.0,对所有错误重试
let config = RetryConfig::new(5)
    .with_initial_delay(Duration::from_millis(100))
    .with_max_delay(Duration::from_secs(5))
    .with_backoff_multiplier(2.0)
    .with_retry_on(RetryOn::AllErrors);
let chain = flaky.with_retry(config);
```

- `RetryOn::TransientErrors`（默认）—— 只重试瞬时错误：HTTP 429 / 500 / 502 / 503 / 504，以及 rate limit、timeout、connection reset 等
- `RetryOn::AllErrors` —— 全部错误都重试
- `RetryOn::Custom(predicate)` —— 自定义判定

### CancellationToken (取消信号) ✨ v0.11.0

跨任务共享的取消标记：`cancel()` 后所有 clone 同时变为取消态，长任务轮询 `is_cancelled()` 优雅退出。

```rust
use langchainrust::core::runnables::CancellationToken;

let token = CancellationToken::new();
let cloned = token.clone();

// 超时自动取消
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(30)).await;
    cloned.cancel();
});

// 注入 Runnable 配置
let config = RunnableConfig::default().with_cancellation_token(token.clone());
let result = chain.invoke(input, Some(config)).await?;

// 循环中主动检查
if token.is_cancelled() {
    return Ok("stopped by cancellation".to_string());
}
```

`await token.cancelled()` 可挂起直到取消触发（轻量自旋，不阻塞线程）。

### 适配器 (将现有组件接入 LCEL)

```rust
use langchainrust::{ChainRunnable, AgentRunnable, RagRunnable};

// Chain 适配器
let chain_runnable = ChainRunnable::new(arc_chain);
let result = chain_runnable.invoke(input_map, None).await?;

// Agent 适配器
let agent_runnable = AgentRunnable::new(arc_agent_executor);
let result = agent_runnable.invoke("query".to_string(), None).await?;

// RAG 适配器
let rag_runnable = RagRunnable::new(arc_rag_pipeline);
let result = rag_runnable.invoke("query".to_string(), None).await?;
```

**AgentEventRunnable** ✨ v0.13.0

与 `AgentRunnable` 不同，`AgentEventRunnable` 的 `stream()` 保留**全部** `AgentStreamEvent` 事件变体（`Text` / `ToolCall` / `ToolStart` / `ToolEnd` / `PipelineStep` / `FinalAnswer` / `Error`），而非只过滤出最终答案；非流式 `invoke()` 则返回单个 `FinalAnswer` 事件。

```rust
use langchainrust::{
    AgentEventRunnable, AgentExecutor, AgentStreamEvent, BaseAgent, FunctionCallingAgent,
    OpenAIChat, OpenAIConfig, Runnable,
};
use std::sync::Arc;
use futures_util::StreamExt;

let llm = OpenAIChat::new(OpenAIConfig::default());
let executor = AgentExecutor::new(
    Arc::new(FunctionCallingAgent::new(llm, vec![], None)) as Arc<dyn BaseAgent>,
    vec![],
);
let agent = AgentEventRunnable::new(Arc::new(executor));

// stream 保留全部事件变体
let mut stream = agent.stream("What is Rust?".to_string(), None).await?;
while let Some(item) = stream.next().await {
    match item? {
        AgentStreamEvent::Text { content } => println!("[text] {}", content),
        AgentStreamEvent::ToolStart { name, .. } => println!("[tool] {name} start"),
        AgentStreamEvent::ToolEnd { name, .. } => println!("[tool] {name} end"),
        AgentStreamEvent::FinalAnswer { content } => println!("[answer] {}", content),
        AgentStreamEvent::Error { message } => eprintln!("[error] {}", message),
        _ => {} // ToolCall / PipelineStep
    }
}

// invoke 返回单个 FinalAnswer
if let AgentStreamEvent::FinalAnswer { content } =
    agent.invoke("What is Rust?".to_string(), None).await?
{
    println!("{}", content);
}
```

**OrchestratorRunnable** ✨ v0.13.0

将高层编排器（`PlanExecuteAgent` / `AdaptiveRAG` / `CorrectiveRAG` / `DeepResearch` / `FanOutFanIn` / `SequentialPipeline` / `TaskAdapter` / `ReviewOrchestrator`）包装为 `Runnable`，让它们能进入 LCEL 管道。`config.metadata["trace_id"]` 会贯通到编排器的 `RunContext`。

```rust
use langchainrust::{BaseTool, OrchestratorRunnable, PlanExecuteAgent, Runnable, RunnableConfig};
use std::sync::Arc;

let tools: Vec<Arc<dyn BaseTool>> = vec![];
let plan_exec = PlanExecuteAgent::new(llm, tools);
let runnable = OrchestratorRunnable::new(plan_exec);

// trace_id 贯通到 RunContext
let config = RunnableConfig::new()
    .with_metadata("trace_id".to_string(), serde_json::json!("trace-001"));
let result: String = runnable.invoke("Research Rust async runtimes".to_string(), Some(config)).await?;
```

<a id="unified-lcel"></a>
### 统一组合 (v0.15.0) —— 提示词 / 记忆 / LLM / 解析器 / RAG 全部可 pipe

v0.15.0 把全框架核心能力统一成 `Runnable`,一条链跑通「提示词 + 记忆 + LLM + 解析器 + RAG」,不再需要手写胶水代码。四件事的变化:

1. **5 个输出解析器输入改为 `LLMResult`** —— `StrOutputParser` / `JsonOutputParser` / `CommaSeparatedListOutputParser` / `StructuredOutputParser` / `TypedOutputParser` 的 invoke 直接取 `input.content` 再走原 `parse`,`llm.pipe(parser)` 编译通过。
2. **`ChatPromptTemplate` 实现 `Runnable`** —— 作为链首段,输入变量表 → 输出 `Vec<Message>`。
3. **`RunnableWithMessageHistory`** —— 把「LLM + 记忆」整体封装成一个 `Runnable<String, LLMResult>`,自动读历史 → 拼输入 → 调 LLM → 写回。
4. **原生 Provider 错误收口** —— `OpenAIChat` / `QwenChat` / `DeepSeekChat` 的错误统一进 `LcelError`,直接 `pipe` 不再套 `LLMClient`。

**现在能 pipe 什么**:

| 组件 | Runnable 形态 | 在链里的位置 |
|---|---|---|
| 提示词 | `ChatPromptTemplate` | 链首段,输入变量表 → 输出消息列表 |
| 记忆 | `RunnableWithMessageHistory` | 把「LLM + 记忆」整体封装,自动读历史 → 拼输入 → 调 LLM → 写回 |
| LLM | 原生 `OpenAIChat` / `QwenChat` / `DeepSeekChat` | 中段,错误统一收口进 `LcelError`,不必再套 `LLMClient` |
| 解析器 | Str / Json / List / Structured / Typed | 尾段,直接接住 `LLMResult`,自动取 `content` |
| RAG | `RagRunnable` | 整段,输入问题输出答案 |
| 错误 | `LcelError` | 全链统一,解析器 / Provider / 链的错误收敛进同一类型 |

**典型组合形态**:

- **纯问答链**:提示词 → LLM → 解析器。输入变量表,输出字符串答案。
- **多轮对话链**:记忆 → LLM → 解析器。输入用户话,输出回答,历史自动读/写。
- **RAG 链**:检索 → 生成。输入问题,结合检索到的资料生成答案。
- **组合**:以上形态可在一个程序里并存,共享同一个 LLM 实例,组成完整的会话式 RAG 助手。

**P0 核心链:提示词 + LLM + 解析器**

```rust
use langchainrust::{
    ChatPromptTemplate, Message, OpenAIChat, OpenAIConfig, RunnableExt, StrOutputParser,
};
use std::collections::HashMap;

let llm = OpenAIChat::new(OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY")?,
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-4o-mini".to_string(),
    ..Default::default()
});

let prompt = ChatPromptTemplate::from_messages([
    Message::system("你是一个简洁的 Rust 助手,只输出结论。"),
    Message::human("{question}"),
]);
let chain = prompt.pipe(llm).pipe(StrOutputParser::new());

let mut vars = HashMap::new();
vars.insert("question".to_string(), "一句话说明什么是 Rust".to_string());
let answer = chain.invoke(vars, None).await?;
```

**多轮对话链:记忆 + LLM + 解析器**

```rust
use langchainrust::{
    ConversationBufferMemory, RunnableExt, RunnableWithMessageHistory, StrOutputParser,
};

let memory = ConversationBufferMemory::new().with_return_messages(true);
let chat_chain = RunnableWithMessageHistory::new(llm.clone(), memory)
    .pipe(StrOutputParser::new());

let r1 = chat_chain.invoke("我叫小明,请记住我。".to_string(), None).await?;
let r2 = chat_chain.invoke("我叫什么名字?".to_string(), None).await?; // 能记住上轮
```

**RAG 链:本地 BM25 检索 + LLM 生成**

```rust
use langchainrust::{
    BM25Retriever, Document, RAGPipelineBuilder, RagRunnable, Runnable, RunnableExt,
};
use std::sync::Arc;

let retriever = BM25Retriever::new();
retriever.add_documents_sync(vec![
    Document::new("Rust 是一门系统编程语言,由 Mozilla 开发,注重安全和性能。").with_id("rust_intro"),
    Document::new("Rust 的核心特性包括所有权系统、借用检查和零成本抽象。").with_id("rust_features"),
]);

let pipeline = RAGPipelineBuilder::new()
    .llm(llm)
    .retriever(retriever)
    .retrieve_k(2)
    .build()?;
let rag_chain = RagRunnable::new(Arc::new(pipeline));

let answer = rag_chain.invoke("Rust 有哪些核心特性?".to_string(), None).await?;
```

**完整五段组合(提示词 + 记忆 + LLM + 解析器 + RAG 一条链)**

五个能力放进同一个可运行程序、共享同一个 LLM 实例,组成完整的会话式 RAG 助手——以下就是 `crates/lc/examples/lcel/lcel_compose.rs` 的完整内容:

```rust
use langchainrust::{
    BM25Retriever, ChatPromptTemplate, ConversationBufferMemory, Document, Message, OpenAIChat,
    OpenAIConfig, RAGPipelineBuilder, RagRunnable, Runnable, RunnableExt, RunnableWithMessageHistory,
    StrOutputParser,
};
use std::collections::HashMap;
use std::sync::Arc;

let api_key = std::env::var("OPENAI_API_KEY").expect("请设置 OPENAI_API_KEY 环境变量");
let llm = OpenAIChat::new(OpenAIConfig {
    api_key,
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-4o-mini".to_string(),
    ..Default::default()
});

// 1. 提示词 + LLM + 解析器 —— Runnable<HashMap<String, String>, String>
let prompt = ChatPromptTemplate::from_messages([
    Message::system("你是一个简洁的 Rust 助手,只输出结论,不要多余文字。"),
    Message::human("{question}"),
]);
let qa_chain = prompt.pipe(llm.clone()).pipe(StrOutputParser::new());
let answer = qa_chain.invoke(HashMap::from([(
    "question".to_string(),
    "一句话说明什么是 Rust 语言".to_string(),
)]), None).await?;

// 2. 记忆 + LLM + 解析器 —— Runnable<String, String>,历史自动读/写
let memory = ConversationBufferMemory::new().with_return_messages(true);
let chat_chain = RunnableWithMessageHistory::new(llm.clone(), memory)
    .pipe(StrOutputParser::new());
let r1 = chat_chain.invoke("我叫小明,请记住我。".to_string(), None).await?;
let r2 = chat_chain.invoke("我叫什么名字?".to_string(), None).await?; // 能记住上轮

// 3. RAG 链:BM25 本地检索 + LLM 生成 —— Runnable<String, String>
let retriever = BM25Retriever::new();
retriever.add_documents_sync(vec![
    Document::new("Rust 是一门系统编程语言,由 Mozilla 开发,注重安全和性能。").with_id("rust_intro"),
    Document::new("Rust 的核心特性包括所有权系统、借用检查和零成本抽象。").with_id("rust_features"),
]);
let pipeline = RAGPipelineBuilder::new()
    .llm(llm)
    .retriever(retriever)
    .retrieve_k(2)
    .build()?;
let rag_chain = RagRunnable::new(Arc::new(pipeline));
let answer = rag_chain.invoke("Rust 有哪些核心特性?".to_string(), None).await?;
```

运行方式:`cargo run --example lcel_compose`(环境变量 `OPENAI_API_KEY` 必需,`OPENAI_BASE_URL` / `TEST_CHAT_MODEL` 可选)。

**统一错误类型 `LcelError`**

整条链的错误都收敛进 `LcelError`:解析器错误实现 `From<OutputParserError>`,原生 OpenAI 错误实现 `From<OpenAIError>`,所以 `prompt.pipe(llm).pipe(parser)` 整条链返回 `Result<T, LcelError>`,一个 `?` 处理全链错误,不需要每段各自 match。

> **边界说明**:其余 Provider(非 OpenAI/Qwen/DeepSeek)维持 `LLMClient` 收口;不做 Rust `|` 运算符重载(用 `.pipe()`);Retriever 的 `Runnable<String, Vec<Document>>` 适配器留待 v0.16。

---

## Chains

Chain 将 LLM 与提示词、记忆、检索等组件组合成可复用的流水线。每个 Chain 接收输入、执行一系列步骤、返回输出。

### LLMChain

最基础的链——一个提示词模板 + 一个 LLM。输入变量替换到模板中，发送给 LLM，返回结果。是构建更复杂链的积木。

```rust
use langchainrust::{LLMChain, BaseChain};

let chain = LLMChain::new(
    llm,
    "Translate the following to {language}: {text}"
);

let result = chain.invoke(HashMap::from([
    ("language", "French"),
    ("text", "Hello world"),
])).await?;
```

### SequentialChain

将多个 Chain 串联——前一个 Chain 的输出作为后一个 Chain 的输入。适合多步骤任务，如"先分析，再总结"。

```rust
use langchainrust::{SequentialChain, LLMChain};
use std::sync::Arc;

let chain1 = LLMChain::new(llm1, "Analyze: {topic}");
let chain2 = LLMChain::new(llm2, "Summarize: {analysis}");

let pipeline = SequentialChain::new()
    .add_chain(Arc::new(chain1), vec!["topic"], vec!["analysis"])
    .add_chain(Arc::new(chain2), vec!["analysis"], vec!["summary"]);

let result = pipeline.invoke(HashMap::from([
    ("topic", "AI trends in 2024"),
])).await?;
```

### RetrievalQA

检索增强问答——先从向量存储中检索相关文档，再把文档和问题一起发给 LLM 回答。是 RAG 的最简形式。

```rust
use langchainrust::{RetrievalQA, SimilarityRetriever};

let retriever = SimilarityRetriever::new(store, embeddings);
let qa = RetrievalQA::new(llm, retriever, 3);

let answer = qa.invoke(HashMap::from([
    ("query", "What is BM25?"),
])).await?;
```

**返回来源**：`.with_return_source_documents(true)` 让检索命中的原始文档随答案一并返回，便于展示依据 / 审计：

```rust
let qa = RetrievalQA::new(llm, retriever, 3).with_return_source_documents(true);
let result = qa.invoke(HashMap::from([("query", "What is BM25?")])).await?;
// result.source_documents 携带命中的 Document 列表
```

### RouterChain（路由链） ✨ v0.14.0

按规则把不同输入分派给不同子链。`RouterChain` 用关键词匹配；`LLMRouterChain` 用 LLM 判断。

```rust
use langchainrust::chains::RouterChain;
use std::sync::Arc;

let router = RouterChain::new()
    .add_route_with_keywords("math", "数学运算", Arc::new(math_chain), vec!["加", "减", "乘"])
    .add_route("general", "通用问答", Arc::new(general_chain))
    .with_default(Arc::new(fallback_chain));

let answer = router.invoke(HashMap::from([("input", "3 加 5 等于几")])).await?;
```

```rust
use langchainrust::chains::LLMRouterChain;

// LLM 版本:按描述让模型自行判断路由目标
let router = LLMRouterChain::new(llm)
    .add_route("translation", "翻译类请求", Arc::new(trans_chain))
    .add_route("code", "编程相关问题", Arc::new(code_chain))
    .with_default(Arc::new(general_chain));
let answer = router.invoke(HashMap::from([("input", "用 Rust 写一个冒泡排序")])).await?;
```

> `add_route_with_keywords` 可给每个路由附带关键词做快速命中；无匹配时走 `with_default` 兜底。

### ConversationRetrievalChain

带记忆的检索增强对话：每次提问时，自动检索相关文档 + 加载对话历史，让 LLM 既能参考知识库，又能记住之前的对话。

```rust
use langchainrust::{ConversationRetrievalChain, ConversationBufferMemory};
use std::sync::Arc;

let memory = Arc::new(ConversationBufferMemory::new());

let chain = ConversationRetrievalChain::new(
    llm,
    retriever,
    memory,
).with_k(3);

let answer = chain.invoke(HashMap::from([
    ("question", "What is BM25?"),
])).await?;
```

### ConversationChain ✨ v0.13.0

带可插拔记忆的对话链——`from_memory` 接受任何实现了 `BaseMemory` 的记忆（窗口 / 摘要 / 向量库 / 持久化），或用 `ConversationChainBuilder` 组装并自定义系统提示词与键名。

```rust
use langchainrust::{
    ConversationChain, ConversationChainBuilder, ConversationBufferWindowMemory,
    OpenAIChat, OpenAIConfig,
};
use std::sync::Arc;
use tokio::sync::Mutex;

let llm = OpenAIChat::new(OpenAIConfig::default());

// 方式一:from_memory 传入任意 BaseMemory
let memory = Arc::new(Mutex::new(ConversationBufferWindowMemory::new(4)));
let chain = ConversationChain::from_memory(llm.clone(), memory);
let answer = chain.predict("Hello!").await?;

// 方式二:Builder(同样可插拔 + 自定义系统提示词/键)
let chain = ConversationChainBuilder::new(llm)
    .memory(ConversationBufferWindowMemory::new(6))
    .system_prompt("You are a helpful assistant.")
    .build();
let answer = chain.predict("What is Rust?").await?;
```

---

## Document Chains

当文档太多、无法一次性塞入 prompt 时，Document Chain 提供不同的策略来处理多文档场景：

| Chain | 策略 | 适用场景 |
|-------|------|----------|
| **StuffDocumentsChain** | 所有文档塞入一个 prompt | 文档少、总长度在 token 限制内 |
| **RefineDocumentsChain** | 逐个文档迭代优化答案 | 需要逐步精炼、文档间有依赖 |
| **MapReduceDocumentsChain** | 每个文档独立处理，再合并 | 文档多、可并行处理 |
| **MapRerankDocumentsChain** | 每个文档独立评分，选最佳 | 需要从多个文档中选最相关的 |

### StuffDocumentsChain

将所有文档与提示词组合，一次性发给 LLM。最简单直接，但文档总量不能超过 LLM 的 token 限制。

```rust
use langchainrust::chains::{StuffDocumentsChain, LLMChain};
use std::sync::Arc;

let llm_chain = Arc::new(LLMChain::new(
    llm,
    "Summarize the following documents:\n{documents}"
));

let chain = StuffDocumentsChain::new(llm_chain);
let result = chain.invoke(documents).await?;
```

### RefineDocumentsChain

逐个文档迭代优化：先用第一个文档生成初始答案，再用后续文档逐步精炼。适合需要综合多个文档信息的场景，但无法并行。

```rust
use langchainrust::chains::RefineDocumentsChain;

let initial_llm = Arc::new(LLMChain::new(llm.clone(), "Summarize: {text}"));
let refine_llm = Arc::new(LLMChain::new(llm, "Refine summary with: {text}"));

let chain = RefineDocumentsChain::new(initial_llm, refine_llm);
let result = chain.invoke(documents).await?;
```

### MapReduceDocumentsChain

Map 阶段对每个文档独立处理（可并行），Reduce 阶段将所有结果合并。适合文档量大、各文档可独立处理的场景。

```rust
use langchainrust::chains::MapReduceDocumentsChain;

let map_chain = Arc::new(LLMChain::new(llm.clone(), "Summarize: {text}"));
let reduce_chain = Arc::new(LLMChain::new(llm, "Combine: {summaries}"));

let chain = MapReduceDocumentsChain::new(map_chain, reduce_chain);
let result = chain.invoke(documents).await?;
```

### MapRerankDocumentsChain

对每个文档独立评分，按分数排序选最佳。适合"从多个候选中选最相关"的场景。

```rust
use langchainrust::chains::MapRerankDocumentsChain;

let map_chain = Arc::new(LLMChain::new(llm, "{text}\nScore (0-10):"));

let chain = MapRerankDocumentsChain::new(map_chain);
let (best_doc, score) = chain.invoke(documents).await?;
```

---

### Chain Streaming ✨ v0.4.1

`BaseChain::stream()` 提供逐 token 的流式输出。`LLMChain` 和 `ConversationChain` 有自定义的实现。

```rust
use langchainrust::{LLMChain, BaseChain};
use futures_util::StreamExt;

let chain = LLMChain::new(llm, "You are a helpful assistant");
let mut stream = chain.stream(inputs).await?;

while let Some(token) = stream.next().await {
    match token {
        Ok(t) => print!("{}", t),
        Err(e) => eprintln!("Stream error: {}", e),
    }
}
```

### invoke_with_config（回调透传） ✨ v0.15.0

`BaseChain::invoke_with_config(inputs, config)` 在调用时注入 `RunnableConfig`（含回调处理器 / 元数据）。复合链（`SequentialChain` / `RouterChain`）会把 config **透传给子链**，不会静默丢弃——整个链路的回调保持一致。

```rust
use langchainrust::{CallbackManager, StdOutHandler, ChainResult};

let config = RunnableConfig::new()
    .with_callbacks(Arc::new(CallbackManager::new().add_handler(Arc::new(StdOutHandler::new()))));
let result: ChainResult = chain.invoke_with_config(inputs, config).await?;
```

---

## Agents

Agent 是能自主调用工具、多步推理的 LLM 应用。与 Chain 不同，Agent 不是固定流程，而是 LLM 根据输入动态决定调用哪些工具、执行多少步。

**什么时候用 Agent，什么时候用 Chain / RAGPipeline？**

| 需求 | 用什么 |
|------|--------|
| 固定流程：提示词 → 模型 → 解析 | Chain / LCEL |
| 基于私有文档回答问题 | RAGPipeline |
| 需要决定调用哪个工具、多步推理 | Agent |
| 检索质量不确定、要自我纠错 / 深度研究 | `CorrectiveRAGAgent` / `DeepResearchAgent` |

**三个基础 Agent 怎么选？**

| Agent | 机制 | 适用模型 | 场景 |
|-------|------|----------|------|
| `FunctionCallingAgent`（推荐） | 原生 tool_calls | GPT-4 / Claude / Gemini 等 | 大部分场景 |
| `ReActAgent` | 文本"思考/行动"正则解析 | 不支持函数调用的模型 | 兼容老模型 |
| `PlanExecuteAgent` | 先规划再逐步执行、失败重规划 | 任何 | 复杂任务分解 |

### FunctionCallingAgent (推荐)

使用 LLM 原生的 Function Calling 能力来调用工具。类型安全、可靠性高，是支持 FC 的模型（GPT-4、Claude、Gemini）的首选。

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

let agent = FunctionCallingAgent::new(llm, tools.clone(), None);

let executor = AgentExecutor::new(
    Arc::new(agent) as Arc<dyn BaseAgent>,
    tools,
).with_max_iterations(5);

let result = executor.invoke("Calculate 37 + 48".to_string()).await?;
```

### ReActAgent (旧版)

使用 ReAct（Reasoning + Acting）模式：LLM 输出"思考→行动→观察"文本，框架解析后调用工具。兼容性好，但依赖文本解析，可靠性不如 FunctionCallingAgent。适合不支持 FC 的模型。

```rust
use langchainrust::{ReActAgent, SimpleMathTool};

let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(DateTimeTool::new()),
    Arc::new(SimpleMathTool::new()),
];

let agent = ReActAgent::new(llm, tools.clone(), None);

let executor = AgentExecutor::new(
    Arc::new(agent) as Arc<dyn BaseAgent>,
    tools,
).with_max_iterations(5);
```

| Agent | 工具调用 | 可靠性 | 适用场景 |
|-------|----------|--------|----------|
| FunctionCallingAgent | 原生 FC | 高（类型安全） | GPT-4, Claude, Gemini |
| ReActAgent | 文本解析 | 中等 | 不支持 FC 的模型 |

### Agent 流式输出 ✨ v0.12.0

CRAG、AdaptiveRAG、DeepResearch 支持 `stream()` 方法，逐步返回管道事件，让你可以实时展示 Agent 的执行进度。

**CRAG 流式输出：**

```rust
use langchainrust::agents::crag::CorrectiveRAGAgent;

let agent = CorrectiveRAGAgent::new(llm, retriever);
let stream = agent.stream("What is Rust ownership?").await?;

// 逐步接收事件：
// PipelineStep { step: "retrieving", detail: "Retrieving documents..." }
// PipelineStep { step: "retrieved", detail: "Retrieved 4 documents" }
// PipelineStep { step: "grading", detail: "Grading documents..." }
// PipelineStep { step: "graded", detail: "Average score: 0.85" }
// PipelineStep { step: "generating", detail: "Generating answer..." }
// FinalAnswer { content: "Rust ownership is..." }
while let Some(event) = stream.next().await {
    match event {
        AgentStreamEvent::PipelineStep { step, detail } => {
            println!("[{}] {}", step, detail.unwrap_or_default());
        }
        AgentStreamEvent::FinalAnswer { content } => {
            println!("Answer: {}", content);
        }
    }
}
```

**AdaptiveRAG 流式输出：**

```rust
use langchainrust::agents::adaptive_rag::AdaptiveRAG;

let agent = AdaptiveRAG::new(llm, retriever);
let stream = agent.stream("Compare tokio vs async-std").await?;

// 事件流：
// PipelineStep { step: "routing", detail: "Deciding retrieval strategy..." }
// PipelineStep { step: "routed", detail: "Decision: MultiQuery" }
// PipelineStep { step: "retrieving", ... }
// PipelineStep { step: "generating", ... }
// FinalAnswer { content: "..." }
```

**DeepResearch 流式输出：**

```rust
use langchainrust::agents::deep_research::DeepResearchAgent;

let agent = DeepResearchAgent::new(llm)
    .with_searcher(Box::new(DuckDuckGoSearchTool::new()));

let stream = agent.stream_research("Rust async runtimes comparison").await?;

// 事件流（多轮搜索）：
// PipelineStep { step: "planning", detail: "Decomposing topic into subtopics..." }
// PipelineStep { step: "searching", detail: "Round 1/3: Searching 3 subtopics..." }
// PipelineStep { step: "searched", detail: "Found 12 results" }
// PipelineStep { step: "synthesizing", detail: "Synthesizing findings..." }
// PipelineStep { step: "gaps_found", detail: "Found 2 knowledge gaps" }
// PipelineStep { step: "searching", detail: "Round 2/3: Searching gaps..." }
// PipelineStep { step: "completed", detail: "Research completed in 2 rounds" }
// FinalAnswer { content: "..." }
```

### AgentBuilder（链式构造） ✨ v0.14.0

`AgentBuilder` 提供链式构造,一次性装配 LLM、工具与执行参数;`max_iterations` 强制 clamp 到 `[1, 100]`,避免死循环。

```rust
use langchainrust::agents::AgentBuilder;
use langchainrust::{Calculator, DateTimeTool, OpenAIChat};

let executor = AgentBuilder::new()
    .llm(OpenAIChat::new(config))
    .tool(Calculator::new())
    .tool(DateTimeTool::new())
    .max_iterations(10)
    .build()
    .await?;

let result = executor.invoke("Calculate 37 + 48".to_string()).await?;
```

`build()` 返回 `AgentExecutor`。健壮性兜底已内置:工具执行超时(`tool_timeout`)、LLM 指数退避重试、Actions 并发 `Semaphore` 限流。

### Orchestrator（编排器） ✨ v0.14.0

`Orchestrator` trait 把多个 Agent 组织成工作流:

- **FanOutFanIn** —— 分发到多个子 Agent 并行执行,再用自定义聚合器(投票/拼接)合并结果
- **SequentialPipeline** —— 串行执行,前一步输出喂给下一步

```rust
use langchainrust::agents::{FanOutFanIn, SequentialPipeline};

// 串行:两个 Agent 依次执行
let pipeline = SequentialPipeline::new()
    .add(researcher_agent)
    .add(writer_agent);
let result = pipeline.run("Rust async runtimes".to_string()).await?;
```

`OrchestratorRunnable` 把它们包装成 LCEL `Runnable`,可进入 `pipe()` 管道(见 LCEL 适配器一节)。

### Agent Hooks（五类安全控制） ✨ v0.11.0

Hooks 在 Agent 执行生命周期插入安全控制:

```rust
use langchainrust::agents::hooks::{AgentHook, PromptInjectionHook, TokenBudgetHook, ContentFilterHook};

let hook = AgentHook::new()
    .on_before_tool_call(approval_callback)   // 允许 / 拒绝 / 跳过
    .with_hook(Arc::new(PromptInjectionHook::new())) // 注入检测
    .with_hook(Arc::new(TokenBudgetHook::new(100_000))) // 预算限制
    .with_hook(Arc::new(ContentFilterHook::new()));    // 内容过滤
```

### ToolPolicy（工具风险分级） ✨ v0.14.0

`ToolPolicy` + `ToolRisk` 给工具分级:高风险工具需要更严格的审批路径,防止越权调用。

```rust
use langchainrust::agents::policy::{ToolPolicy, ToolRisk};

let mut policy = ToolPolicy::new();
policy.set_risk("delete_file", ToolRisk::High);
// 高风险工具调用会走审批,而非直接执行
```

### Agent 人审门（ApprovalHandler） ✨ v0.16.0

工具执行前异步审批：`Allow` 放行 / `Deny` 拒绝（理由作为 observation 喂回循环，不执行工具）/ `Modify` 改参后执行。默认关（`None` = 原样放行）。

```rust
use langchainrust::agents::hooks::ToolCallContext;
use langchainrust::{ApprovalHandler, ApprovalDecision};
use std::sync::Arc;

struct MyApproval;

#[async_trait::async_trait]
impl ApprovalHandler for MyApproval {
    async fn approve(&self, ctx: &ToolCallContext) -> ApprovalDecision {
        if ctx.name == "delete_file" {
            ApprovalDecision::Deny { reason: "manual review required".into() }
        } else {
            ApprovalDecision::Allow
        }
    }
}

let executor = executor.with_approval(Arc::new(MyApproval));
```

谁审批由调用方实现 trait（CLI 交互 / Webhook / 自动策略路由），框架只提供闸 + 参考实现 `AllowAll`。`approve(ctx).await` 是异步的——挂起后信号到即从同一行续跑，同进程 resume 天然成立。

### Agent 预算门（BudgetConfig） ✨ v0.16.0

给 Agent 循环设硬上限，超限返回 `AgentError::BudgetExceeded`（带精确 `limit` / `actual`）。默认关。

```rust
use langchainrust::BudgetConfig;
use std::time::Duration;

let executor = executor.with_budget(BudgetConfig {
    max_tool_calls: Some(50),                          // 累计工具调用上限
    max_tokens: Some(20_000),                          // 累计 LLM token 上限
    max_duration: Some(Duration::from_secs(120)),      // 循环总时长
    max_iterations: Some(10),                          // 覆盖/收紧默认迭代上限
    ..Default::default()
});

### 跨进程 resume（ResumeStore / FileResumeStore） ✨ v0.18.0

人审门 / 预算门的挂起点可以**落盘**：进程死亡不再丢等待中的审批，重启后从磁盘恢复续跑，而不是从头重放 agent 循环。

```rust
use langchainrust::{FileResumeStore, ResumeStore};
use std::sync::Arc;

// 进程 A：给 executor 挂上磁盘挂起点存储 + 审批门
let store = Arc::new(FileResumeStore::new("/var/checkpoints/app")?);
let executor = AgentExecutor::new(agent, tools)
    .with_resume_store(store)
    .with_approval(handler);

// 进程 B（重启后）：读挂起点、向操作员展示待审批调用、用审批决定续跑
if let Some(pending) = executor.pending_approval().await? {
    println!("待审批: {} {}", pending.tool_name, pending.arguments);
    let answer = executor.resume(decision).await?;   // Allow / Deny / Modify
}
```

框架在每次工具调用进入审批**之前**把 `PendingApproval` 快照写入 store（工具名 / 参数 / 中间步骤 / 迭代序号 / 预算累计），审批决定落地后清除——原子写（先 `pending.json.tmp` 再 rename），崩溃不产生半截 checkpoint。`ApprovalHandler` 接口不变，调用方零改动；`MemoryResumeStore` 是内存版（单进程演示 / 测试用）。并发 executor 须用各自独立目录。
```

## Plan-Execute Agent

**解决什么问题**：普通单循环 Agent（`FunctionCallingAgent` / `ReActAgent`）适合"一步能想清楚"的任务——想一步、干一步、再看结果。但像"先调研、再写代码、最后解释要点"这类复杂多步骤任务，模型一步想不出完整方案，直接动手又容易走偏。Plan-Execute Agent 把大任务拆成"先规划 → 逐步执行 → 失败重规划"的循环：先用 LLM 把任务拆成若干可执行步骤，每步交给一个单循环 Agent 执行，某一步失败就重新规划（而不是硬着头皮继续），全部完成后总结出最终结果。适用于复杂、多步骤、允许中途调整计划的任务。

> 注意：每个步骤通过 `FunctionCallingAgent` + 工具执行；`llm` 目前必须是 `OpenAIChat`。

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, PlanExecuteAgent, BaseTool};
use std::sync::Arc;

let llm = OpenAIChat::new(OpenAIConfig::default());
let tools: Vec<Arc<dyn BaseTool>> = vec![]; // 传入实际工具

let agent = PlanExecuteAgent::new(llm, tools)
    .with_max_replans(2); // 失败时最多重新规划 2 次

let result = agent
    .run("Research Rust async runtimes, write example code, explain key points")
    .await?;
println!("{}", result);
```

### 工作流程

| 阶段 | 做什么 |
|---|---|
| 规划（Plan） | 用 LLM 把任务拆成若干可执行步骤 |
| 逐步执行（Execute） | 每步交给一个 `FunctionCallingAgent` + 工具执行，拿回步骤结果 |
| 失败重规划（Re-plan） | 某步失败时重新规划，重规划次数由 `with_max_replans` 控制，避免空转 |
| 总结（Answer） | 全部步骤完成后，汇总生成最终回答 |

### 与 FunctionCallingAgent / ReActAgent 的区别

| | FunctionCallingAgent / ReActAgent | PlanExecuteAgent |
|---|---|---|
| 定位 | 单循环执行者（实现 `BaseAgent`，可塞进 `AgentExecutor`） | 高层编排器（不实现 `BaseAgent`，有自己的 `run()`） |
| 任务形态 | 单步"想 → 干 → 看"，直到模型说完成 | 先把复杂任务拆成步骤，再逐步驱动执行者 |
| 失败处理 | 工具失败把结果喂回 LLM 再想 | 步骤失败触发重新规划 |
| 适用 | 单任务、决策清晰 | 多步骤、需要先拆解的复杂任务 |

**怎么选（人话）**：简单任务直接用 `FunctionCallingAgent` / `ReActAgent`；任务大到"一眼看不出先做什么"、需要拆步骤时再用 Plan-Execute。它就是"老板"，自己不上手干，把活派给单循环 Agent。

### 关键行为与注意

- **每步都是独立执行（冷启动）**：每个步骤由新的 `FunctionCallingAgent` + `Executor` 执行，跑完即弃，步骤之间默认不共享上下文。对步骤结果强依赖的任务（第 2 步要用第 1 步的产出），需要把上一步结果写进下一步的步骤描述里带上。
- **失败重规划有上限**：`with_max_replans` 限制重规划次数，防止失败后无限重规划。
- **执行器可配置**：v0.14 起执行步骤所用的 Agent 可通过 `agent_factory` 配置，不再写死成 `FunctionCallingAgent`。
- **和 DeepResearch 的取舍**：PlanExecute 适合步骤相对独立的任务（搜索 → 排行程 → 写报告）；研究型任务（研究 → 再研究 → 综合）需要步骤间串联上下文、避免丢中间结论时，用 `DeepResearch` 更合适。

---

## Handoffs

**解决什么问题**：让一个 Agent 包办所有事——既当研究又当写作——模型容易顾此失彼，也不利于复用。Handoffs（交接）让主 Agent 干到一半发现"这事该另一个专家 Agent 干"时，把控制权转交过去。受 OpenAI Agents SDK 启发：主 Agent 通过 `HandoffTool` 将任务委托给已注册的专家 Agent。适合"分工明确的专家 Agent 团队"——主 Agent 只负责判断谁来干，具体活交给对应专家，交接完成后由专家继续往下走。

```rust
use langchainrust::agents::HandoffManager;
use langchainrust::{BaseAgent, AgentExecutor, FunctionCallingAgent, OpenAIChat, OpenAIConfig};
use std::sync::Arc;

let llm = OpenAIChat::new(OpenAIConfig::default());

let mgr = HandoffManager::new();
let writer = Arc::new(AgentExecutor::new(
    Arc::new(FunctionCallingAgent::new(llm.clone(), vec![], None)) as Arc<dyn BaseAgent>,
    vec![],
));
let researcher = Arc::new(AgentExecutor::new(
    Arc::new(FunctionCallingAgent::new(llm.clone(), vec![], None)) as Arc<dyn BaseAgent>,
    vec![],
));
mgr.register_agent("writer", writer)?;
mgr.register_agent("researcher", researcher)?;
mgr.set_primary("researcher")?;

// 运行主 Agent
let result = mgr.run("Research and write an article".to_string()).await?;

// 为每个已注册的 Agent 生成 HandoffTool（命名为 handoff_to_{agent}）
let mgr = Arc::new(mgr);
let handoff_tools = mgr.handoff_tools();
let history = mgr.history(); // 委托历史
```

### 怎么用

1. **注册专家 Agent**：`register_agent("writer", writer)` 给每个专家起名并登记。
2. **设置主 Agent**：`set_primary("researcher")` 指定入口 Agent，任务从它开始跑。
3. **跑主 Agent**：`mgr.run(...)` 执行任务。
4. **生成交接工具**：`handoff_tools()` 为每个已注册的 Agent 生成名为 `handoff_to_{agent}` 的工具；把这些工具绑给 Agent 后，模型就能主动选择"交给谁"。也可以不绑工具，在代码里用 `execute_handoff(Handoff)` 直接发起交接。

### 关键行为

| 行为 | 说明 |
|---|---|
| `handoff_tools()` | 返回一批 `handoff_to_{agent}` 工具，命名与注册名一一对应 |
| `execute_handoff(Handoff)` | 不经工具，在代码里直接发起交接 |
| `history()` | 委托历史，可追溯谁把任务交给了谁 |
| `max_handoff_depth` | 交接深度上限（默认 10），防止 Agent A↔B 无限互相交接陷入死循环；达到上限时终止交接并返回明确错误 |

### 什么时候用 / 什么时候别用

- **用**：任务边界清晰、每个子领域有一个专门 Agent（如 writer / researcher / coder），主 Agent 只做调度。
- **别用**：需要"派活收结果"（把任务分发给多个 Agent 并行干、再聚合结果）时，Handoffs 是**单一控制权转移**，不是分发聚合——这种场景用 `FanOutFanIn` 更合适。

---

## Streaming Tool Calls

**解决什么问题**：普通 Agent 等整个执行完才返回结果，用户只能干等，无法判断是卡死还是在工作。流式工具调用让 `StreamingFunctionCallingAgent` 逐 token 流式输出 LLM 文本，并通过事件流暴露工具调用状态——用户能实时看到 Agent "正在想、正在调哪个工具、工具调用走到哪个阶段、最后给出答案"的全过程。适合聊天式体验（打字机效果）、长任务进度展示、以及调试 Agent 行为。

```rust
use langchainrust::StreamingFunctionCallingAgent;
use langchainrust::agents::streaming::AgentStreamEvent;
use futures_util::StreamExt;

let agent = StreamingFunctionCallingAgent::new(llm);
let mut stream = agent.invoke_stream("Describe Rust in one sentence".to_string()).await;

while let Some(event) = stream.next().await {
    match event {
        AgentStreamEvent::Text { content } => print!("{}", content),
        AgentStreamEvent::ToolCall { state } => {
            // state: Started / ArgumentsStreaming / Completed / Failed ...
        }
        AgentStreamEvent::FinalAnswer { content } => println!("\n[done] {}", content),
    }
}
```

### 事件流

`invoke_stream` 返回一个异步 Stream，逐个产出 `AgentStreamEvent`：

| 事件 | 含义 |
|---|---|
| `Text { content }` | LLM 逐 token 文本，直接打印即是打字机效果 |
| `ToolCall { state }` | 工具调用状态变化（见下） |
| `FinalAnswer { content }` | 最终答案，一般出现在流末尾 |

`ToolCallState` 覆盖工具调用的生命周期：

| 状态 | 含义 |
|---|---|
| `Started` | 工具调用开始 |
| `ArgumentsStreaming` | 工具参数正在逐段生成 |
| `Completed` | 工具调用完成 |
| `Failed` | 工具调用失败 |

### 适用场景

| 场景 | 为什么用流式 |
|---|---|
| 聊天式 UI | 文字逐字显示，等待不煎熬 |
| 长任务 / 多步任务 | 实时展示"正在做什么"，用户知道没卡死 |
| 调试 Agent | 直接观察思考文本 + 工具调用时序，快速定位问题 |

### 注意

- 事件流给到的是 LLM 文本与工具调用的**状态变化**；`ToolCall` 事件描述"调用走到哪一步"，工具执行的结果会回流给 LLM 做下一步决策，但结果正文本身不出现在事件流里。
- 该 Agent 的流式面聚焦于 LLM 输出 + 工具调用状态；如果只需要工具执行层面的细粒度事件，可另看 `Executor::stream` 的能力面——两者视角不同。

---

## Guardrails

输入/输出验证，用于阻止恶意输入和敏感信息泄露。实现 `InputGuardrail` / `OutputGuardrail`，或使用内置验证器，然后用 `GuardedAgent` 包装 Agent。

```rust
use langchainrust::guardrails::{
    GuardrailsConfig, MaxLengthGuardrail, SensitiveInfoGuardrail, GuardedAgent,
};
use langchainrust::{BaseAgent, AgentExecutor, FunctionCallingAgent, OpenAIChat, OpenAIConfig};
use std::sync::Arc;

let config = GuardrailsConfig::new()
    .with_input(Arc::new(MaxLengthGuardrail::new(1000)))    // 限制输入长度
    .with_output(Arc::new(SensitiveInfoGuardrail::new()));  // 阻止敏感输出

let agent = FunctionCallingAgent::new(OpenAIChat::new(OpenAIConfig::default()), vec![], None);
let executor = Arc::new(AgentExecutor::new(
    Arc::new(agent) as Arc<dyn BaseAgent>,
    vec![],
));

let mut guarded = GuardedAgent::new(executor, config);
let result = guarded.invoke("Summarize this content".to_string()).await?; // 验证输入 -> Agent -> 验证输出
let violations = guarded.violations();
```

内置验证器：`MaxLengthGuardrail`（输入长度）、`ForbiddenWordsGuardrail`（禁用词）、`SensitiveInfoGuardrail`（API 密钥 / 邮箱 / 信用卡 / 关键词，可通过 `with_keywords` 扩展）。也可以使用 `GuardrailRunner` 手动驱动验证。

### 类型分离的护栏结果 ✨ v0.15.0

护栏结果按输入/输出分开成两个类型,由类型系统强制安全规则:

- **`InputGuardrailResult`** —— 只有 `Pass` / `Block`(输入侧不存在 `Modify`)
- **`OutputGuardrailResult`** —— `Pass` / `Block` / `Modify`

「Modify 只适用于输出」由编译器保证,而非运行时约定。`GuardrailError::Blocked` 携带 `reason` / `partial` / `suggestion`,失败时可带部分内容做降级展示。

### Guardable（解耦包装目标） ✨ v0.15.0

`GuardedAgent` 不再只认 `AgentExecutor`。`Guardable` trait(`invoke_str` / `stream_str`)让**任意可执行单元**都能被护栏包裹:

- `AgentExecutor` 直接实现 `Guardable`
- 任意 `BaseChain` 经 `ChainGuardable` 适配
- `GuardedAgent::from_chain` 提供链式入口

```rust
use langchainrust::guardrails::{GuardedAgent, GuardrailsConfig, MaxLengthGuardrail};
use langchainrust::LLMChain;

let chain = LLMChain::new(llm, "You are a helpful assistant");
let mut guarded = GuardedAgent::from_chain(
    Arc::new(chain),
    GuardrailsConfig::new().with_input(Arc::new(MaxLengthGuardrail::new(1000))),
);
let result = guarded.invoke("Summarize this".to_string()).await?;
```

### 流式护栏 ✨ v0.15.0

`StreamingOutputGuardrail` trait(`validate_chunk -> ChunkAction::{Pass, Replace, Block}`)配合 `GuardedAgent::invoke_stream` 做两阶段检查:增量关键词检查 + 24 字符滑动窗口(防止跨块切断)+ 完整输出复查。

```rust
use futures_util::StreamExt;

let mut stream = guarded.invoke_stream("Write a long summary".to_string()).await?;
while let Some(chunk) = stream.next().await {
    let chunk = chunk?;   // GuardableChunk { token, is_final }
    print!("{}", chunk.token);
    if chunk.is_final {
        break;            // 最后一个分块,结束流式输出
    }
}
```

### 审计持久化 ✨ v0.15.0

`AuditSink` trait + `FileAuditSink`(JSON Lines 追加式)把违规记录落盘,供事后分析:

```rust
use langchainrust::guardrails::audit::FileAuditSink;

let config = GuardrailsConfig::new()
    .with_output(Arc::new(SensitiveInfoGuardrail::new()))
    .with_audit_sink(Arc::new(FileAuditSink::new("guardrails.log")?));
```

`violations` 有界(`MAX_VIOLATIONS = 1000`),可 `clear_violations()` 清空。`SensitiveInfoGuardrail` 支持挂 LLM 裁判(`with_judge`,复用 `SensitiveJudge` / `LlmSensitiveJudge`)做上下文敏感检测,并对高误报词(`password`/`密码`/`token`/`secret`)降级为仅告警。

---

## Token Counter

**解决什么问题**：LLM 按 token 计费，但 token 数不等于字符数——一段中文、一段代码、一段英文的 token 密度各不相同。想回答"这一轮花了多少 token、多少钱"、发请求前估算 token 数（用来截断超长文本）、或自动累计每次调用的用量，都需要专门的工具。Token Counter 相关组件把"计数 → 追踪 → 计价"串成一条链。

```rust
use langchainrust::{TokenTrackingLLM, ModelPricing, OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

let tracked = TokenTrackingLLM::for_openai(OpenAIChat::new(OpenAIConfig::default()))?;

let result = tracked.chat(vec![Message::human("hi")], None).await?;

let usage = tracked.get_usage();                               // prompt / completion / total tokens
let cost = tracked.estimate_cost(&ModelPricing::gpt4o_mini()); // USD
```

### 四个组件各管什么

| 组件 | 作用 | 什么时候用 |
|---|---|---|
| `TiktokenCounter` | 用 OpenAI 同款分词算法精确计数（cl100k_base） | 需要精确 token 数（计费、对齐上下文窗口） |
| `CharRatioCounter` | 没有 tiktoken 时的粗略估算——按字符数量比例推算（中文场景常用） | 快速估算、拿不到精确分词器时兜底 |
| `TokenTrackingLLM` | 包装任意模型，自动记录每次调用的 token 用量并累计 | 想自动追踪累计用量，不想手动算 |
| `ModelPricing` | 给每个模型配定价（每千 token 价格），按累计用量算费用 | 估算成本 / 按用量算钱 |

> 导入路径：`TiktokenCounter` / `TokenTrackingLLM` / `ModelPricing` 在根级；`CharRatioCounter` 需从 `langchainrust::core::token_counter::CharRatioCounter` 导入。

### 精确计数 vs 估算

| 场景 | 选哪个 |
|---|---|
| 计费、限额、对齐窗口 | `TiktokenCounter` 精确计数 |
| 快速估算、中文长文本裁剪 | `CharRatioCounter` 粗略估算 |
| 自动记录每次调用用量 | `TokenTrackingLLM`（包装模型） |
| 按用量算 USD 成本 | `ModelPricing` + `estimate_cost` |

### 关键行为

- **真实用量优先**：`TokenTrackingLLM` 优先使用模型 API 返回的真实 usage 统计；模型没返回时才用估算。
- **不侵入原模型**：统计对象是包装器自己，原模型行为不变——包一层就能自动累计用量。
- **内置定价**：`ModelPricing::gpt4o()` / `gpt4o_mini()` 为内置定价；用 `ModelPricing::new(prompt_per_1k, completion_per_1k)` 可自定义其它模型的定价。
- **算钱**：`get_usage()` 拿累计的 prompt / completion / total tokens，`estimate_cost(&pricing)` 按定价换算成 USD。

### 怎么选（人话）

- 只是想"每次调用记个数、最后报个总价"，直接用 `TokenTrackingLLM::for_openai(...)` 包装模型 + `estimate_cost`，一条链搞定。
- 要在发请求**前**估算一段文本的 token 数（比如判断要不要截断），用 `TiktokenCounter`（精确）或 `CharRatioCounter`（估算）。
- 中文内容占比高、且不追求精确时，`CharRatioCounter` 够用；正式计费请用精确分词。

---

## Sessions

**解决什么问题**：多轮对话必须记住上下文——用户上一轮说了什么、助手怎么回的。但"记在哪、怎么存、怎么取"是每个应用都要重复实现的样板。`SessionManager` 把会话抽象成生命周期管理：创建会话 → 往里写对话 → 随时取历史 → 归档/清理。同时天然支持**多会话隔离**：每个会话有独立 id 和归属用户，不同用户、不同话题的对话互不串扰。

**核心三件套**：`Session`（会话本身）← `SessionStore`（怎么存）← `SessionManager`（怎么用）。

```rust
use langchainrust::sessions::{SessionManager, MemorySessionStore};
use langchainrust::{OpenAIChat, OpenAIConfig};
use std::sync::Arc;

let manager = SessionManager::new(Arc::new(MemorySessionStore::new()));
let id = manager.create_session_for("user_1").await?;

let llm = OpenAIChat::new(OpenAIConfig::default());
let r1 = manager.chat(&id, &llm, "My name is Tom".to_string()).await?;
let r2 = manager.chat(&id, &llm, "What is my name?".to_string()).await?; // 记住上一轮对话

let history = manager.history(&id).await?;  // Vec<Message>
manager.clear(&id).await?;                   // 清除历史（保留会话）
manager.archive(&id).await?;                 // 归档
let sessions = manager.list_by_user("user_1").await?;
```

### 会话模型

| 字段 | 作用 |
|---|---|
| `id` | 会话唯一标识 |
| `user_id` | 归属用户（可空，支持匿名） |
| `messages` | 对话消息列表（`Vec<Message>`），追加式增长 |
| `status` | 生命周期状态（`Active` / `Archived` / `Deleted`） |
| `metadata` | 自由键值扩展属性 |

### SessionManager 方法

| 方法 | 作用 |
|---|---|
| `create_session_for(user)` | 建新会话，返回会话 id |
| `chat(&id, &llm, msg)` | 核心：追加用户消息 → 拿历史喂 LLM → 把回复追加回会话（自动维护历史） |
| `history(&id)` | 取完整对话历史（`Vec<Message>`） |
| `clear(&id)` | 清除历史（保留会话） |
| `archive(&id)` | 归档（不再活跃，但保留） |
| `list_by_user(user)` | 列出某用户的全部会话 |

`chat()` 是核心——调用方只传"会话 id + LLM + 用户消息"，历史读写都由 `SessionManager` 包办，不需要手动维护 `Vec<Message>`。

### 会话级历史管理

- 每个会话的历史独立维护；`chat()` 每次基于该会话已有历史喂给 LLM，所以同一会话第二句才能"记住上一轮"。
- 默认用内部缓冲直接维护完整历史，简单直接；长会话 token 会随轮数线性增长，需要控制成本时挂记忆组件（见下）。
- `SessionStore` trait 包含 `create/get/update/delete/list_by_user`；`MemorySessionStore` 为内置实现（进程内存 + tokio 锁），适用于测试和单进程使用；可实现自己的后端（Redis / 数据库）。

### 会话生命周期 ✨ v0.15.0

`SessionStatus` 状态机闭环：`Active` → `Archived` → `Deleted`。删除为**软删除**：记录保留（可审计/恢复），但不再出现在用户会话列表中。

### 接入记忆系统 ✨ v0.15.0

`SessionManager` 默认用内部缓冲维护历史；`with_memory` 可挂接任意 `BaseMemory`（如 `ConversationSummaryBufferMemory` / `MongoPersistentMemory`），让会话历史走摘要压缩或跨进程持久化：

```rust
let mut manager = SessionManager::new(Arc::new(MemorySessionStore::new()));
manager = manager.with_memory(Arc::new(Mutex::new(
    ConversationSummaryBufferMemory::new(llm, 2000),
)));
let r = manager.chat(&id, &llm, "问题".to_string()).await?;
```

**怎么选记忆（人话）**：不挂，保持全量历史，语义最简单但长会话会膨胀；挂 `ConversationSummaryBufferMemory`，长会话被压成"摘要 + 近期窗口"，控制 token 成本；挂持久化记忆，历史能跨进程存活、多实例共享。挂上后对话历史由记忆组件处理，而不是全量透传。

### 注意

- **并发写同一会话**：`chat()` 内部是"读取 → 追加 → 写回"三步，同一会话的并发写入需要自行串行化（如按会话加锁），否则可能丢消息。
- **会话 vs 长期记忆**：lc-sessions 管"一次对话的过程记录"，跨会话的长期记忆（人设、偏好）交给 lc-memory；会话是按时间组织的对话上下文。
- **会话 vs 检查点**：会话存"聊了什么"；检查点（Checkpointer）存"图执行到哪一步"（lc-langgraph）。两者都涉及持久化，但语义不同。
- **存储选型**：测试、单进程场景用 `MemorySessionStore` 足够；多实例 / 需要跨进程共享历史时，换数据库或 Redis 后端实现 `SessionStore`。

---

## MCP

[MCP](https://modelcontextprotocol.io)（Model Context Protocol）是 Anthropic 推出的工具协议标准。`MCPClient` 连接任意 MCP Server 获取工具，并将其适配为 `BaseTool` 供 Agent 使用。

```rust
use langchainrust::mcp::{MCPClient, MCPConfig};
use langchainrust::{BaseAgent, AgentExecutor, FunctionCallingAgent, OpenAIChat, OpenAIConfig};
use std::sync::Arc;

// Stdio：启动 MCP Server 子进程
let config = MCPConfig::stdio(
    "npx",
    vec!["@anthropic/mcp-server-filesystem".to_string(), "/tmp".to_string()],
);
// 或 SSE：MCPConfig::sse("http://localhost:3001/sse");

let mut client = MCPClient::connect(config).await?;
let tools = client.list_tools().await?;           // tools/list
println!("MCP tool count: {}", tools.len());

// 适配为 BaseTool 列表并交给 Agent
// P0-3: as_tools 自动发现工具,无需先手动调用 list_tools
let mcp_tools = client.as_tools().await?;
let agent = FunctionCallingAgent::new(
    OpenAIChat::new(OpenAIConfig::default()),
    mcp_tools,
    None,
);
let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, vec![]);
let result = executor.invoke("Read /tmp/notes.txt".to_string()).await?;

client.close().await?;
```

`MCPConfig::stdio(command, args)` / `MCPConfig::sse(url)` / `.with_env(k, v)`；`client.call_tool(name, arguments)` 直接调用工具；`as_tools()` 将工具包装为 `MCPToolAdapter`（实现 `BaseTool`）。

---

### MCPServer

与 `MCPClient` 对称：将本地 `BaseTool` 暴露为 MCP Server，供 Claude Desktop / Cursor 等宿主使用。支持 `initialize` / `tools/list` / `tools/call`。

```rust
use langchainrust::{MCPServer, Calculator, BaseTool};
use std::sync::Arc;

let tool: Arc<dyn BaseTool> = Arc::new(Calculator::new());
let server = MCPServer::new()
    .with_tool(tool)
    .with_server_info("my-tools", "0.1.0");

server.serve_stdio().await?;
```

`server.handle_request(req)` 用于自定义传输层的单步 JSON-RPC 处理。

### 传输层韧性 ✨ v0.15.0

MCP 连接自动重连:断线后按指数退避(0.5s 起,上限 30s)重试,Server 重启后自动恢复。`MCPServer` 侧同样具备热重启能力,宿主重连即恢复会话。

### ConnectionManager（连接池） ✨ v0.15.0

管理多个 `MCPClient` 生命周期,自动重连、统一关闭:

```rust
use langchainrust::mcp::{ConnectionManager, ServerSpec};

let manager = ConnectionManager::new();
manager.register(ServerSpec::new("files", MCPConfig::sse("http://localhost:3001/sse"))).await?;
manager.register(ServerSpec::new("tools", MCPConfig::stdio("npx", vec!["...".into()]))).await?;

let client = manager.client("files").await?;  // 取某个 server 的连接
manager.reap_idle().await;                     // 回收空闲连接
// manager.shutdown().await;                     // 统一关闭
```

### 工具命名空间 / 发现 / 超时 ✨ v0.15.0

- **`ToolNamespace`**:工具名自动加 `server:tool` 前缀,多 server 同名工具不冲突;`register(server, tools, conflict)` 返回命名空间化结果,可据此构造 `MCPToolAdapter::namespaced(...)`
- **`ToolDiscovery`**:批量发现 + 健康检查,过滤掉不可用 server 的工具
- **`ToolSpec`**:`timeout`(单次工具调用超时)、`max_retries` 等执行策略,超时命中熔断

### ServerHealth / CircuitBreaker（健康与熔断） ✨ v0.15.0

每个 server 有健康状态(`HealthStatus`)与熔断器:

```rust
use langchainrust::mcp::{CircuitBreaker, HealthStatus};

let breaker = CircuitBreaker::new(5); // 连续 5 次失败 -> 熔断
if !breaker.allow_request() {
    // 熔断打开:直接短路,不再打后端
} else {
    match call_tool().await {
        Ok(v) => breaker.record_success(), // 成功,自动重置计数
        Err(_) => breaker.record_failure(),
    }
}
```

`ServerHealth` 记录延迟、错误率、最近一次探测时间,供上层做路由决策。

### SamplingGuard（采样保护） ✨ v0.15.0

对服务端采样请求(resources/sampling/createMessage)的递归防护:限制嵌套深度、整条采样链的 token 预算与总时长,防模型自行递归采样耗尽资源:

```rust
use langchainrust::mcp::SamplingGuard;

let guard = SamplingGuard::new(5, 100_000) // 最大嵌套深度 5,整条链 token 预算 100k
    .with_timeout(std::time::Duration::from_secs(60)); // 整条链总时长上限
let lease = guard.enter(4000)?; // 进入一次采样,返回 SamplingLease,drop 时自动释放深度
```

### MCPGateway（网关） ✨ v0.15.0

把多个 MCP server 聚合为一个统一入口,按 `server` 参数路由:

```rust
use langchainrust::mcp::{MCPGateway, GatewayServerSpec, MCPConfig};

let gateway = MCPGateway::new();
gateway.register(GatewayServerSpec::new("files", MCPConfig::stdio("npx", vec!["filesystem".into(), "/tmp".into()]))).await?;
gateway.register(GatewayServerSpec::new("db", MCPConfig::sse("http://localhost:9000/sse"))).await?;
gateway.sync_all().await?; // 拉取全部 server 的工具

let tools = gateway.as_base_tools().await?; // 自动加 server 前缀,互不冲突
```

配套能力:
- **`ServerSandbox`**:`ParamRule`(参数白名单/黑名单/类型校验)、`EgressPolicy`(出站策略,限制工具调用的网络/文件范围)
- **`PartialContent`**:流式工具结果分块返回,`stream_tool_call` 边执行边推送
- **`TenantGateway`**:多租户隔离,每租户独立的工具命名空间 + 配额 + 访问控制
- **`ToolOrchestrator`**:工具 DAG 编排,声明依赖关系后自动排序/并行执行
- **`VersionPolicy`**:多版本 MCP 协议协商(`VersionPolicy::Latest` / `Pin("2024-11-05")`)

### MCP Server 原语接线 ✨ v0.18.0

client→server 原语(`resources/*` / `prompts/*` / `completion/complete`)为**注册制**:给 `MCPServer` 注册数据源后,对应方法返回真实数据;未注册仍返回 `method_not_found`(-32601,诚实边界)。`initialize` 握手时 `capabilities` 按实际注册项补齐(`tools` 恒声明)。

```rust
use langchainrust::mcp::{
    ElicitationAction, MCPError, MCPServer, Resource, ResourceContent, ResourceProvider,
};
use std::sync::Arc;

struct StaticResources;

#[async_trait::async_trait]
impl ResourceProvider for StaticResources {
    async fn list_resources(&self) -> Result<Vec<Resource>, MCPError> {
        Ok(vec![Resource {
            uri: "file:///README.md".into(),
            name: "README".into(),
            description: None,
            mime_type: Some("text/markdown".into()),
        }])
    }
    async fn read_resource(&self, uri: &str) -> Result<Vec<ResourceContent>, MCPError> {
        let text = format!("content of {uri}");
        Ok(vec![ResourceContent {
            uri: uri.into(),
            mime_type: Some("text/plain".into()),
            text: Some(text),
            blob: None,
        }])
    }
}

let server = MCPServer::new()
    .with_tool(Arc::new(Calculator))
    .with_resource_provider(Arc::new(StaticResources))
    .with_prompt_provider(Arc::new(my_prompts))      // prompts/list + prompts/get
    .with_completion_provider(Arc::new(my_completions)); // completion/complete
```

server→host 方向的 `sampling::create_message` / `elicitation::create` 由 Server 发起、Host 执行: `MCPServer` 提供发起方法(`create_message` / `create_elicitation`),需注入回调(`with_sampling_handler` / `with_elicitation_handler`);未注入回调时返回明确错误,不静默。真实交互依赖宿主 UI/模型环境,由使用者经回调接入(测试用注入 mock 覆盖)。

## Tools

工具是 Agent 的"手"——让 LLM 能执行计算、搜索、读写文件等操作。每个工具实现 `BaseTool` trait，定义名称、描述、参数 schema 和执行逻辑。

### 内置工具

| 工具 | 描述 | 参数 |
|------|------|------|
| Calculator | 数学运算 | `expression` |
| DateTimeTool | 日期/时间查询 | `operation`, `datetime` |
| SimpleMathTool | 幂运算、开方、三角函数 | `operation`, `value` |
| URLFetchTool | 获取 URL 内容 | `url` |
| WikipediaTool | Wikipedia 搜索 | `query` |
| DuckDuckGoSearchTool | 网页搜索 | `query` |
| PythonREPLTool | 执行 Python 代码 | `code` |

### 自定义工具

当内置工具不够用时，实现 `BaseTool` trait 创建自己的工具。需要定义输入结构体（`JsonSchema` + `Deserialize`）和 `run` 方法。

```rust
use langchainrust::{BaseTool, ToolError};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(JsonSchema, Deserialize)]
struct EchoInput {
    text: String,
}

pub struct EchoTool;

#[async_trait::async_trait]
impl BaseTool for EchoTool {
    fn name(&self) -> &str { "echo" }
    
    fn description(&self) -> &str { "Echo the input text" }
    
    async fn run(&self, input: String) -> Result<String, ToolError> {
        let args: EchoInput = serde_json::from_str(&input)?;
        Ok(args.text)
    }
    
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(EchoInput)).unwrap())
    }
}
```

### `#[tool]` 过程宏 ✨ v0.10.0

用 `#[tool]` 宏自动生成 `BaseTool` + `Tool` 实现，无需手写样板代码：

```rust
use langchainrust::{BaseTool, Tool, ToolError, tools::tool};

// 一行宏 = 上面 ~20 行手写代码
#[tool(description = "Echo the input text back")]
fn echo(
    #[param(desc = "The text to echo back")]
    text: String,
) -> Result<String, ToolError> {
    Ok(text)
}

// 自动生成:
// - EchoTool struct (BaseTool + Tool impl)
// - EchoInput struct (Deserialize + JsonSchema)
// - args_schema() 从 JsonSchema 自动生成

// 使用方式与手写 Tool 完全一致
let tool = EchoTool::new();
let schema = BaseTool::args_schema(&tool);  // JSON Schema
let result = tool.run(r#"{"text":"hello"}"#.to_string()).await?;
// result = "\"hello\""

// 支持 Option<T> 可选参数
#[tool(description = "Greet someone")]
fn greet(
    #[param(desc = "Person's name")]
    name: String,
    #[param(desc = "Greeting style")]
    style: Option<String>,
) -> Result<String, ToolError> {
    let style = style.unwrap_or_else(|| "Hello".to_string());
    Ok(format!("{}, {}!", style, name))
}
```

### ToolRegistry（工具注册表） ✨ v0.15.0

按名称管理一组工具的注册表：注册、查找、移除、批量描述，可直接喂给 LLM 展示可用工具。

```rust
use langchainrust::ToolRegistry;
use std::sync::Arc;

let mut registry = ToolRegistry::new();
registry.register(Arc::new(Calculator::new()));
registry.register(Arc::new(DateTimeTool::new()));

registry.get("calculator");                 // Option<&Arc<dyn BaseTool>>
registry.contains("datetime_tool");
registry.tool_names();                       // Vec<&str>
let description = registry.describe_tools(); // 供 LLM 阅读的工具清单
registry.remove("calculator");
```

### StructuredTool（结构化包装） ✨ v0.15.0

把实现 `Tool` trait 的通用工具包装成 `BaseTool`，自动完成 JSON 输入解析与输出序列化：

```rust
use langchainrust::{Tool, core::tools::StructuredTool};

let tool = StructuredTool::new(my_tool, Some("my_tool"), Some("描述"));
let result = tool.run(r#"{"k": "v"}"#.to_string()).await?; // 内部自动解析/序列化
```

### SSRF 防护（网络工具） ✨ v0.15.0

`URLFetchTool` / `HTTPTool` **默认开启 SSRF 防护**：请求前与重定向的每一跳都检查目标是否为内网/回环地址，命中即拒绝并提示 `.with_allow_private_ips(true)` 显式放行。

```rust
let tool = URLFetchTool::new();                 // 默认拦截内网
let tool = URLFetchTool::new().with_allow_private_ips(true); // 显式放行
```

实现要点：`is_private_ip` 是全 crate 唯一实现（禁止复制逻辑），覆盖 127.0.0.0/8、10/8、172.16/12、192.168/16、169.254.169.254、IPv6 内网段及 IPv4-mapped IPv6（`::ffff:127.0.0.1`）；自动重定向被禁用，改为 `guarded_get` 逐跳重查，堵住"首跳公网、重定向进内网"的绕过。

### WikipediaTool

搜索 Wikipedia 文章摘要。适合 Agent 需要查询百科知识的场景。

```rust
use langchainrust::WikipediaTool;

let tool = WikipediaTool::new();
let result = tool.run(r#"{"query": "Rust programming"}"#).await?;
```

### DuckDuckGoSearchTool

使用 DuckDuckGo 搜索网页。无需 API Key，适合 Agent 需要实时网络信息的场景。

```rust
use langchainrust::DuckDuckGoSearchTool;

let tool = DuckDuckGoSearchTool::new();
let result = tool.run(r#"{"query": "langchain rust"}"#).await?;
```

### PythonREPLTool

在子进程中执行 Python 代码并返回输出。适合需要动态计算、数据处理、科学计算的场景。注意：代码在本地执行，确保运行环境安全。

```rust
use langchainrust::PythonREPLTool;

let tool = PythonREPLTool::new();
let result = tool.run(r#"{"code": "print(sum(range(10)))"}"#).await?;
```

> **安全边界**：内置的"危险 import 黑名单"（`os` / `sys` / `subprocess` / `__import__` / `eval` / `exec` 等）只是**噪音过滤，不是安全边界**——`__import__`、`"o"+"s"` 拼接、`().__class__` 反射、unicode 混淆等编码绕过挡不住，还会误伤字符串字面量。真正的隔离必须走 [代码解释器沙箱](#v050-new-features)（`LocalSandbox` 子进程 + 超时）；黑名单只用于减少误入沙箱的噪音。不要在不可信输入上依赖 `PythonREPLTool` 做隔离。

### 扩展工具 (HTTPTool / FileTool / SQLTool)

v0.3.0 新增的三个面向生产环境的工具，均实现 `BaseTool`。

**HTTPTool** -- 发送 GET/POST 请求：

```rust
use langchainrust::HTTPTool;
use serde_json::json;

let http = HTTPTool::new();
let body = http.post("https://httpbin.org/post", json!({"k": "v"})).await?;
// 作为 BaseTool：输入 JSON {"url":"...","method":"get|post","body":{...}}
```

**FileTool** -- 沙箱文件读写（限制在 `base_path` 内，扩展名白名单，大小上限，路径遍历防护）：

```rust
use langchainrust::FileTool;
use std::path::PathBuf;

let file = FileTool::new(PathBuf::from("./workspace"))
    .with_allowed_extensions(vec!["txt".into(), "md".into(), "json".into()])
    .with_max_size(10 * 1024 * 1024);
let content = file.read("notes.txt").await?;
file.write("out.txt", "hello").await?;
// 作为 BaseTool：输入 JSON {"op":"read|write|list","path":"...","content":"..."}
```

**SQLTool** -- 只读 SQL 查询（仅 SELECT，表白名单；支持参数化查询，防注入；需要 `sqlite-storage` feature）：

```rust
use langchainrust::tools::extended::SQLTool;

let sql = SQLTool::new("data.db")?
    .with_allowed_tables(vec!["users".into()]);
let rows = sql.execute("SELECT id, name FROM users")?; // Vec<HashMap<String,String>>
// 非 SELECT 语句（如 DROP/INSERT）会被拒绝

// 参数化查询（推荐,防 SQL 注入）
let rows = sql.execute_parameterized("SELECT * FROM users WHERE name = ?", &["Alice".into()])?;
```

作为工具调用时优先解析 `{"sql": "...", "params": [...]}` 参数化形式。

> `SQLTool` 在 `sqlite-storage` feature 下可用；`HTTPTool` / `FileTool` 默认可用。

---

## Embeddings

**Embeddings** 将文本转换为固定维度的浮点向量，使语义相近的文本在向量空间中距离更近。是语义检索、相似度计算、RAG 的基础。

### 支持的 Embeddings

| 提供商 | 类 | 维度 | 特性 |
|----------|-------|-----------|----------|
| **OpenAI** | `OpenAIEmbeddings` | 1536 | 高质量 |
| **DeepSeek** | `DeepSeekEmbeddings` | 1536 | 高性价比 |
| **Qwen** | `QwenEmbeddings` | 1536 | 中文优化 |
| **Cohere** | `CohereEmbeddings` | 自定义 | RAG 场景、多语言 |
| **FastEmbed** | `FastEmbedEmbeddings` | 384 | 本地 ONNX 加速 |
| **BagOfWords** | `BagOfWordsEmbeddings` | 自定义 | 纯本地词袋 |
| **Mock** | `MockEmbeddings` | 自定义 | 测试用 |
| **Local** | `LocalEmbeddings` | 默认 | 纯 Rust,离线 |

### OpenAI 嵌入

使用 OpenAI 的 text-embedding-ada-002 模型，1536 维，质量最高但需要 API 调用。

```rust
use langchainrust::{OpenAIEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(OpenAIEmbeddings::new(
    std::env::var("OPENAI_API_KEY")?
));

// 单文本嵌入
let vector = embeddings.embed("Rust is a systems language").await?;
println!("Dimension: {}", vector.len());  // 1536

// 批量嵌入
let texts = vec![
    "Rust is a systems language",
    "Python is a scripting language",
];
let vectors = embeddings.embed_batch(texts).await?;
```

### DeepSeek 嵌入

DeepSeek 的嵌入模型，1536 维，价格比 OpenAI 低。

```rust
use langchainrust::{DeepSeekEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(DeepSeekEmbeddings::from_env());

let vector = embeddings.embed("Deep learning fundamentals").await?;
```

### Qwen 嵌入

阿里云 Qwen 的嵌入模型，1536 维，中文效果更好。

```rust
use langchainrust::{QwenEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(QwenEmbeddings::from_env());

let vector = embeddings.embed("Qwen vector generation").await?;
```

### Mock 嵌入（测试用）

生成固定维度的随机向量，不调用任何 API。仅用于测试和开发，不用于生产。

```rust
use langchainrust::{MockEmbeddings, Embeddings};
use std::sync::Arc;

// 自定义维度
let embeddings = Arc::new(MockEmbeddings::new(128));

let vector = embeddings.embed("Test text").await?;
println!("Dimension: {}", vector.len());  // 128
```

---

### LocalEmbeddings

纯 Rust 实现的轻量级本地嵌入（词频哈希 + L2 归一化），无需 API 调用。适用于离线/隐私/零成本的粗粒度检索。

```rust
use langchainrust::LocalEmbeddings;

let emb = LocalEmbeddings::default_dim();
let vec = emb.embed_query("hello world").await?;
```

**限制**：基于词袋哈希，语义质量有限。如需高质量嵌入，请使用 `OpenAIEmbeddings` 等。

### 统一 Embeddings trait ✨ v0.15.0

所有嵌入 Provider 统一实现 `Embeddings` trait（`embed` / `embed_batch` / `embed_query`），可直接替换、组合成 `EmbeddingMatcher` 做相似度检索。统一错误语义：

- `EmptyInput` —— 空文本
- `EmptyVectorInBatch` —— 批次中某条返回空向量
- `BatchMismatch` —— 输入条数与返回向量数不一致

错误不静默吞掉:任一条嵌入失败即返回明确错误,不做静默降级。

### 重试与并发 ✨ v0.15.0

内建请求韧性:对 429 / 5xx 自动重试(默认 3 次);批量嵌入并发度 8、批次上限 2048;向量统一 L2 归一化,便于余弦相似度比对。

```rust
use langchainrust::{OpenAIEmbeddings, Embeddings, retrieval::graph_rag::EmbeddingMatcher};

let emb = Arc::new(OpenAIEmbeddings::new("sk-..."));
let docs = vec!["Rust ownership".into(), "Borrow checker".into()];
let matcher = EmbeddingMatcher::new(emb, docs);
let top = matcher.query("memory safety in Rust", 2).await?; // 语义最相近的 2 篇
```

## RAG

RAG（Retrieval-Augmented Generation）让 LLM 基于你的私有数据回答问题，而不是只靠训练时的知识。流程：文档 → 分割 → 嵌入 → 存入向量库 → 检索相关文档 → 连同问题发给 LLM。

**三条实现路径怎么选：**

| 路径 | 做法 | 适合 |
|------|------|------|
| `RAGPipeline` | 一条龙「检索 + 生成」封装，builder 构造 | 快速起步、开箱即用 |
| LCEL 手动链 | 检索器 + `prompt \| llm` 自己 pipe | 想精细控制提示词与中间步骤 |
| RAG 智能体 | `CorrectiveRAGAgent` / `AdaptiveRAG` | 检索质量不确定、要自我纠错 |

**检索方式怎么选：** 关键词精确匹配用 BM25，语义相似用向量检索，两者都要用混合检索——对比见[检索模式对比](#检索模式对比)。

<a id="end-to-end-ragpipeline"></a>
### 端到端 RAGPipeline ✨ v0.15.0

`RAGPipeline` 把「检索 + 生成」封装成开箱即用的完整管道。`RAGPipelineBuilder` 提供链式构造:LLM、检索器(或 嵌入+向量库 组合)、召回数 `retrieve_k`、System 提示词 `system`。

```rust
use langchainrust::{
    BM25Retriever, Document, RAGPipelineBuilder, RetrieverTrait,
};

let retriever = BM25Retriever::new();
retriever.add_documents_sync(vec![
    Document::new("Rust 是一门系统编程语言,注重安全与性能。").with_id("intro"),
    Document::new("所有权系统与借用检查是 Rust 的核心。").with_id("ownership"),
]);

// 检索器方案(零依赖,本地)
let pipeline = RAGPipelineBuilder::new()
    .llm(llm)
    .retriever(retriever)
    .retrieve_k(2)
    .system("请基于提供的上下文回答,不要编造。")
    .build()?;

// 或嵌入 + 向量库方案(语义检索)
let pipeline = RAGPipelineBuilder::new()
    .llm(llm)
    .embeddings(OpenAIEmbeddings::new(api_key))
    .vector_store(ChromaDBVectorStore::new(
        ChromaDBConfig::new("http://localhost:8000", "docs", 1536),
    ).await?)
    .retrieve_k(3)
    .build()?;
```

三种调用方式:

```rust
// 1. 只取生成结果
let answer: String = pipeline.query("Rust 有哪些核心特性?").await?;

// 2. 带来源引用(审计 / 展示依据)
let answer_with_sources = pipeline.query_with_sources("Rust 有哪些核心特性?").await?;
println!("{}", answer_with_sources.answer);
for src in &answer_with_sources.sources { /* 每个来源 Document 与相似度 */ }

// 3. 进入 LCEL 管道(RagRunnable 包装)
let rag_chain = RagRunnable::new(Arc::new(pipeline));
let answer = rag_chain.invoke("Rust 有哪些核心特性?".to_string(), None).await?;
```

> **设计要点**:`RetrieverTrait` 统一了 Similarity / BM25 / UnifiedHybrid 三类检索器,`RAGPipeline` 只依赖 trait 而非具体实现——换检索策略不改业务代码。

### 文档分割

长文档需要先分割成小块，才能有效检索。`RecursiveCharacterSplitter` 按字符数分割，在段落/句子边界处优先断开，保持语义完整性。

```rust
use langchainrust::{RecursiveCharacterSplitter, TextSplitter};

let splitter = RecursiveCharacterSplitter::new(200, 50);

let chunks = splitter.split_document(&Document::new(
    "Long text to split..."
))?;
```

### SemanticSplitter

按语义相关性分割：句子分词 + 嵌入，在相邻相似度急剧下降处断开。比字符级分割具有更好的语义完整性。支持中英文句子边界（`。!?;` / `.!?\n`）。

```rust
use langchainrust::SemanticSplitter;
use langchainrust::OpenAIEmbeddings;

let splitter = SemanticSplitter::with_defaults(OpenAIEmbeddings::new(config));
// or: SemanticSplitter::new(emb, 0.5, 1000)

let chunks = splitter.split_text(long_text).await;  // Vec<String>
```

**注意**：嵌入是异步的，而 `TextSplitter` 是同步的；为避免破坏同步 trait，此分割器暴露异步的 `split_text` / `split_document`，不实现同步的 `TextSplitter`。

### 向量存储

将文档嵌入后存入向量存储，支持相似度检索。`InMemoryVectorStore` 适合开发和测试；生产环境使用 ChromaDB、Qdrant、PGVector 等持久化存储。

```rust
use langchainrust::{InMemoryVectorStore, SimilarityRetriever};
use std::sync::Arc;

let store = Arc::new(InMemoryVectorStore::new());
let embeddings = Arc::new(OpenAIEmbeddings::new(api_key));

let retriever = SimilarityRetriever::new(store.clone(), embeddings);

retriever.add_documents(vec![
    Document::new("Rust is a systems language"),
    Document::new("Python is a scripting language"),
]).await?;

let docs = retriever.retrieve("systems programming", 3).await?;
```

### ChromaDB

使用 Chroma 的持久化向量存储。需要运行 Chroma 服务（默认端口 8000），适合需要持久化和生产级检索的场景。

```toml
[dependencies]
langchainrust = { version = "0.8", features = ["chromadb"] }
```

```rust
use langchainrust::{ChromaDBConfig, ChromaDBVectorStore, SimilarityRetriever};
use std::sync::Arc;

let store = Arc::new(ChromaDBVectorStore::new(
    ChromaDBConfig::new("http://localhost:8000", "my_collection", 1536),
).await?);

let retriever = SimilarityRetriever::new(store.clone(), embeddings);

retriever.add_documents(vec![
    Document::new("Rust is a systems language"),
]).await?;

let docs = retriever.retrieve("systems programming", 3).await?;
```

### PGVectorStore

PostgreSQL + pgvector 扩展向量存储。适合已有 PostgreSQL 基础设施、需要关系型数据库 + 向量检索合一的场景。需要 `pgvector-storage` feature（框架在 feature 内已内置 `sqlx` + `pgvector` 依赖，无需自行添加）。建库前需由管理员执行 `CREATE EXTENSION vector;`。

```rust
use langchainrust::vector_stores::PGVectorStore;

let store = PGVectorStore::connect(
    "postgres://user:pass@localhost/db",
    "docs",
    1536, // 向量维度
).await?;
// 建表（CREATE TABLE IF NOT EXISTS，幂等）；需先由管理员执行 CREATE EXTENSION vector
store.initialize().await?;
// docs: Vec<Document>；embeddings: Vec<Vec<f32>>（来自 Embeddings::embed_documents）
let ids = store.add_documents(docs, embeddings).await?;
// query_embedding: Vec<f32>（来自 Embeddings::embed_query）
let found = store.similarity_search(&query_embedding, 5).await?;
store.delete_document("doc-id").await?;
```

`connect` 只建连接池不建表；`initialize()` 建表（幂等）；`build_table_sql(table, dim)` 是用于表 DDL 的纯函数。检索支持 [`MetadataFilter`](#metadatafilter) 过滤（`similarity_search_with_filter`）。

### PineconeStore

Pinecone 云向量存储（reqwest HTTP API，无需 feature，默认可用）。适合需要托管向量服务、不想自建数据库的场景。

```rust
use langchainrust::vector_stores::PineconeStore;
use langchainrust::embeddings::Embeddings;

// host 格式：https://{index-name}.svc.{environment}.pinecone.io
let store = PineconeStore::new("your-api-key", "https://my-index.svc.prod.pinecone.io");

// embeddings: impl Embeddings
store.upsert(&docs, &embeddings).await?;       // 自动嵌入文档
let qvec: Vec<f32> = embeddings.embed_query("query").await?; // 查询接受已嵌入的向量
let found = store.query(qvec, 5).await?;
store.delete(&["id1".to_string()]).await?;
```

`upsert` 自动调用 `embed_documents`；`query` 接受已嵌入的向量（`embed_query` 的结果）。

### 统一 VectorStore trait ✨ v0.15.0

所有后端统一实现 10 方法的 `VectorStore` trait，接口一致、可即插即用：

| 方法 | 说明 |
|------|------|
| `add_documents` | 批量写入（文档 + 向量） |
| `similarity_search` | 向量相似检索（降序） |
| `embed_query` / `similarity_search_text` | 自带嵌入器的后端可直接传文本 |
| `similarity_search_with_min_score` | 带最低分数阈值 |
| `get_document` / `get_embedding` | 按 ID 读取 |
| `delete_document` / `count` / `clear` | 管理 |

```rust
use langchainrust::vector_stores::{VectorStore, VectorStoreBuilder};

// 统一工厂:同一个 trait 下切换后端
let store: Arc<dyn VectorStore> = VectorStoreBuilder::in_memory().build().await?;
let store = VectorStoreBuilder::file_backed("kb.bin", 384).build().await?;
let store = VectorStoreBuilder::qdrant("http://localhost:6334", "kb").build().await?;
```

**错误类型** `VectorStoreError` 四种变体：`DocumentNotFound` / `EmbeddingError` / `StorageError` / `ConnectionError`。

**后端清单**：`InMemoryVectorStore`、`ChromaDBVectorStore`、`PGVectorStore`、`PineconeStore`、`LanceDBVectorStore`、`Neo4jVectorStore`、`QdrantVectorStore`、`FileVectorStore`、`ChunkedVectorStore`，以及 `DocumentStore` 家族（`InMemoryDocumentStore` / `MongoChunkedDocumentStore` / `RedisDocumentStore` / `SQLiteDocumentStore`）。

> **诚实报错，拒绝静默降级**：Qdrant 等需要 feature 的后端在未启用 feature 时返回显式错误（提示开启 `qdrant-integration`），**不会**悄悄回退到内存存储——否则生产代码以为在写持久化，进程重启数据即丢。

<a id="metadatafilter"></a>
### MetadataFilter 元数据过滤 ✨ v0.18.0

`VectorStore` 从 0.18 起支持**跨后端一致的元数据过滤**：`similarity_search_with_filter(&query_embedding, k, Some(&filter))`。`filter: None` 等价旧 `similarity_search`；后端未覆写过滤时返回明确 `VectorStoreError::UnsupportedFilter`，**不静默吞掉过滤返回全量**。

```rust
use langchainrust::{FilterOp, MetadataFilter};

// 单条件：字段等于
let f = MetadataFilter::field("category", FilterOp::Eq, "news");
// AND / OR 组合
let f2 = MetadataFilter::and(vec![
    MetadataFilter::field("year", FilterOp::Gte, 2024),
    MetadataFilter::field("author", FilterOp::In, vec!["alice", "bob"]),
]);

let found = store
    .similarity_search_with_filter(&qvec, 5, Some(&f2))
    .await?;
```

| 操作符 | 含义 |
|--------|------|
| `Eq` / `Ne` | 等于 / 不等于 |
| `Gt` / `Gte` / `Lt` / `Lte` | 数值 / 日期范围比较 |
| `In` / `Nin` | 在集合内 / 不在（value 为数组） |

支持过滤的后端：内存 / 文件 / Qdrant / Pinecone / Chroma / LanceDB / Neo4j / PGVector（`pgvector-storage` feature）。第三方 `VectorStore` 实现想支持过滤，覆写 `similarity_search_with_filter` 把 `MetadataFilter` 翻译成原生查询即可；不需要的后端依赖默认实现（有过滤请求时报 `UnsupportedFilter`）。`SelfQueryRetriever` 就是建立在这层过滤之上的（见下节）。

---

## BM25

BM25 是经典的关键词检索算法，根据词频和文档长度计算相关性分数。与向量检索（语义相似）不同，BM25 擅长精确关键词匹配，如搜索"Rust ownership"会优先返回包含这些词的文档。不需要嵌入模型，零成本，速度快。

### BM25Retriever（关键词搜索）

```rust
use langchainrust::{BM25Retriever, Document};

let retriever = BM25Retriever::new();

retriever.add_documents_sync(vec![
    Document::new("Rust is a systems programming language"),
    Document::new("Python is a scripting language"),
    Document::new("JavaScript is for web development"),
]);

let results = retriever.search("systems programming", 3);

for result in results {
    println!("Document: {}", result.document.content);
    println!("Score: {}", result.score);
}
```

### BM25 参数

k1 控制词频饱和度（越大，高频词权重越高），b 控制文档长度归一化（越大，长文档惩罚越重）。默认值 k1=1.5, b=0.75 适合大多数场景。

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| k1 | 1.5 | 词频饱和度 |
| b | 0.75 | 文档长度归一化 |

```rust
let retriever = BM25Retriever::with_params(2.0, 0.5);
```

### ChunkedBM25Retriever（父子结构）

解决"小块匹配但丢失上下文"的问题：文档先分割为叶子块建立 BM25 索引，检索时如果同一父文档的多个叶子块都匹配，就自动合并为完整的父文档返回。

```rust
use langchainrust::{ChunkedBM25Retriever, AutoMergingConfig, ChunkedDocumentStore};

let config = AutoMergingConfig::new()
    .with_leaf_size(400)      // 叶子块大小
    .with_threshold(0.5);     // 当 50%+ 叶子匹配时合并

let store = Arc::new(ChunkedDocumentStore::new());
let mut retriever = ChunkedBM25Retriever::with_config(store, config);

retriever.add_document(Document::new("Long document..."));

let results = retriever.search("keyword", 5);

for result in results {
    if result.is_merged() {
        println!("Merged: {}", result.content());
    } else {
        println!("Leaf: {}", result.content());
    }
}
```

---

<a id="hybrid-retrieval"></a>
## 混合检索

向量检索擅长语义相似，BM25 擅长关键词匹配——两者互补。混合检索同时使用两种方式，用 RRF（Reciprocal Rank Fusion）算法合并结果，比单一检索方式召回率更高。

### RRF 融合算法

```
RRF_score(d) = Σ 1/(k + rank(d))
```

其中 k=60，rank(d) 是文档在各结果列表中的排名。

### UnifiedHybridIndex

一站式混合检索：内部同时维护 BM25 索引和向量索引，添加文档时自动双索引，查询时自动双检索 + RRF 合并。无需手动管理两个索引。

```rust
use langchainrust::{
    UnifiedHybridIndex, HybridIndexConfig, OpenAIEmbeddings, InMemoryVectorStore, VectorStore,
};

let config = HybridIndexConfig::new()
    .with_chunk_size(500)
    .with_top_k(10, 10)        // BM25_k, Vector_k
    .with_rrf_k(60);

let embeddings = Arc::new(OpenAIEmbeddings::new(api_key));
let vector_store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
let index = UnifiedHybridIndex::with_config(embeddings, vector_store, 1536, config);

// 自动构建双索引
index.add_document(Document::new("Document content")).await?;

// 混合搜索
let results = index.retrieve("query", 5).await?;

for result in results {
    println!("Content: {}", result.document.content);
    println!("RRF Score: {}", result.score);
}
```

### 检索模式对比

| 模式 | 内容存储 | 查找 | 使用场景 |
|------|------------------|--------|----------|
| SimpleVector | InMemoryVectorStore | 无查找 | 纯向量，简单场景 |
| BM25 Only | ChunkedDocumentStore | 查找 | 纯关键词 |
| Hybrid | ChunkedDocumentStore（共享） | 查找 | 组合检索（推荐） |

---

## LangGraph

LangGraph 用有向图定义复杂工作流：每个节点是一个处理步骤，边定义执行顺序。比 Chain 更灵活——支持条件分支、循环、人工介入、子图。适合需要精细控制执行流程的场景。

### StateGraph

最基础的图——定义节点和边，状态在节点间传递。`AgentState` 是内置的状态结构，包含 `messages`、`steps` 等字段。

```rust
use langchainrust::langgraph::{StateGraph, AgentState, START, END};

let mut graph = StateGraph::new();

graph.add_node_fn("analyze", |state: AgentState| {
    let mut new_state = state.clone();
    new_state.steps.push("analyzed".to_string());
    new_state
});

graph.add_node_fn("process", |state: AgentState| {
    state
});

graph.add_edge(START, "analyze");
graph.add_edge("analyze", "process");
graph.add_edge("process", END);

let compiled = graph.compile();

let result = compiled.invoke(AgentState::new("用户问题".to_string())).await?;
```

### 条件边

根据当前状态动态选择下一个节点。`FunctionRouter` 接收一个闭包，返回目标节点名称。适合"消息多就总结，少就继续"这类分支逻辑。

```rust
use std::collections::HashMap;
use langchainrust::langgraph::FunctionRouter;

let router = FunctionRouter::new(|state: &AgentState| {
    if state.messages.len() > 5 { "summarize" } else { "continue" }
});
graph.set_conditional_router("route", router);

graph.add_conditional_edges(
    "analyze",
    "route",
    HashMap::from([
        ("summarize".to_string(), "summarize".to_string()),
        ("continue".to_string(), "continue".to_string()),
    ]),
    None, // 默认目标;路由返回值不在 targets 时使用
);
```

### 人工介入 / 中断与恢复

在关键节点前暂停执行，等待人工确认后继续。`with_interrupt_before` 指定哪些节点前中断；`MemoryCheckpointer` 保存执行状态，支持跨会话恢复。

```rust
use langchainrust::langgraph::{GraphError, MemoryCheckpointer};

let compiled = graph.compile()
    .map_err(|e| ...)?
    .with_checkpointer(MemoryCheckpointer::new())
    .with_interrupt_before(vec!["output", "analyze"]);

match compiled.invoke(state).await {
    Ok(result) => { /* 完成 */ }
    Err(GraphError::ExecutionInterrupted(node)) => {
        println!("暂停于: {}", node);
        if let Some(exec) = compiled.create_resume_execution(&node).await {
            let result = compiled.resume(exec).await?;
        }
    }
    Err(e) => { /* 错误 */ }
}
```

### Reducer（状态合并规则） ✨ v0.15.0

子节点返回的状态如何合并进共享状态，由 Reducer 决定：

- `ReplaceReducer` —— 直接覆盖字段（默认）
- `AppendReducer` —— 追加（`messages` 数组用它在每步累积）

```rust
use langchainrust::langgraph::{StateGraph, AppendMessagesReducer};

let mut graph = StateGraph::new();
graph.set_reducer("messages", std::sync::Arc::new(AppendMessagesReducer));
```

### 边类型 ✨ v0.15.0

`GraphEdge` 四种边：

| 边 | 语义 |
|----|------|
| `Fixed` | 固定跳转 `source → target` |
| `Conditional` | 按路由函数动态选择 |
| `FanOut` | 一个节点并行分发到多个目标 |
| `FanIn` | 多个节点汇入一个汇聚点 |

```rust
graph.add_fan_out("query", vec!["crag".to_string(), "graph".to_string(), "vector".to_string()]);
graph.add_fan_in(vec!["crag".to_string(), "graph".to_string(), "vector".to_string()], "merge");
```

### Checkpointer 家族 ✨ v0.15.0

- `MemoryCheckpointer` —— 进程内（单线程）
- `ThreadSafeMemoryCheckpointer` —— 并发安全
- `FileCheckpointer::new(path)` —— 落盘持久化（不实现 `Default`，必须显式给路径，失败可传播）

配合 `with_checkpointer` + `with_interrupt_before` 实现「暂停 → 恢复」工作流。

### 图定义持久化 ✨ v0.15.0

`GraphPersistence` trait 把图定义（节点/边/reducer）存下来复用：`MemoryPersistence` / `FilePersistence` / `MongoPersistence`。

### 子图 / 动态规划 / 流式 ✨ v0.15.0

- `SubgraphNode` —— 图嵌图，把子流程封装成节点复用
- `DynamicPlanner` / `DynamicInjection` / `DynamicTask` —— 运行时动态构造任务、注入并行分支
- `compiled.stream_collected(input)` —— 返回 `Vec<StreamEvent<S>>`，逐步观测节点执行进度

---

<a id="document-loaders"></a>
## 文档加载器

从各种文件格式加载文档，统一转为 `Document` 结构（`content` + `metadata`），供后续分割和检索使用。

### Document 家族 ✨ v0.15.0

统一的数据结构贯穿加载 → 分割 → 存储 → 检索全链路：

| 类型 | 用途 |
|------|------|
| `Document` | 原始文档：`content` + `metadata`（`with_id` / `with_metadata` 链式构建） |
| `VectorDocument` | 带向量的文档（向量库内部存储） |
| `SearchResult` | 检索结果：`document` + `score` |
| `ChunkDocument` | 父子结构的叶子块，持有父文档引用 |

`RecursiveCharacterSplitter` 按优先级选择分隔符：**段落 → 行 → 句子 → 字符**，在前一级分割后仍超限时才降级到下一级，尽量保持语义完整。

### 支持的格式

| 加载器 | 格式 | 特性 |
|--------|--------|----------|
| **TextLoader** | .txt | 按行分割 |
| **JSONLoader** | .json | 指定 content_key |
| **MarkdownLoader** | .md | 按标题级别分割 |
| **PDFLoader** | .pdf | 提取 PDF 文本 |
| **CSVLoader** | .csv | 每行作为一个文档 |

### TextLoader

加载纯文本文件。支持整文件加载和按行分割加载。

```rust
use langchainrust::{TextLoader, DocumentLoader};

let loader = TextLoader::new("document.txt");
let docs = loader.load().await?;

// 按行分割
let loader = TextLoader::new_with_line_split("document.txt");
let docs = loader.load().await?;
```

### JSONLoader

加载 JSON 文件。默认提取整个 JSON 字符串作为内容；指定 `content_key` 后只提取特定字段的值。

```rust
use langchainrust::{JSONLoader, DocumentLoader};

let loader = JSONLoader::new("data.json");
let docs = loader.load().await?;

// 指定内容字段
let loader = JSONLoader::new_with_content_key("data.json", "content");
let docs = loader.load().await?;
```

### MarkdownLoader

加载 Markdown 文件。支持按标题级别分割——每个标题下的内容作为一个独立文档，保持章节的语义完整性。

```rust
use langchainrust::{MarkdownLoader, DocumentLoader};

// 按标题级别分割
let loader = MarkdownLoader::new_with_heading_split("guide.md", 1);
let docs = loader.load().await?;
```

### HTMLLoader

去除 `<script>`/`<style>`，移除标签，解码常见 HTML 实体，折叠空白，从 HTML 字符串或 URL 中提取纯文本。

```rust
use langchainrust::retrieval::HTMLLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

// 从 HTML 字符串
let loader = HTMLLoader::new("<p>Hello <b>world</b></p>");
let docs = loader.load().await?; // content: "Hello world"

// 从 URL（异步获取后解析）
let loader = HTMLLoader::from_url("https://example.com");
let docs = loader.load().await?;

// 纯函数：直接提取文本
let text = HTMLLoader::extract_text("<script>x</script><p>a &amp; b</p>");
// -> "a & b"
```

### DocxLoader ✨ v0.4.1

解析 Word `.docx` 文件：ZIP 解压 + XML `<w:t>` 文本节点解析。

```rust
use langchainrust::retrieval::loaders::DocxLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

let loader = DocxLoader::new("document.docx");
let docs = loader.load().await?;
```

### WebScraperLoader ✨ v0.4.1

网页抓取：提取页面文本，支持递归同域链接跟踪。

```rust
use langchainrust::retrieval::loaders::WebScraperLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

let loader = WebScraperLoader::new("https://example.com")
    .with_max_depth(2)
    .with_max_pages(10);
let docs = loader.load().await?;
```

### SitemapLoader ✨ v0.4.1

解析 `sitemap.xml` 并批量抓取页面。

```rust
use langchainrust::retrieval::loaders::SitemapLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

let loader = SitemapLoader::new("https://example.com/sitemap.xml")
    .with_max_pages(50);
let docs = loader.load().await?;
```

---

## MultiQueryRetriever

用户的查询可能措辞与文档不一致，导致检索不到。MultiQueryRetriever 用 LLM 将一个查询改写为多个变体，分别检索后合并去重，提高召回率。

它的定位是"查询扩展"型的增强检索器：把一次检索变成多次检索，用不同问法去"捞"同一份语料，专门应对**召回率不足**的问题。典型场景是用户提问自由、文档术语不统一——文档写"DB 连接超时"，用户问"database timeout"，单路检索关键词对不上，Top-K 里就找不到相关段落。多路变体并行检索，相当于同时用多种问法开卷，漏掉的可能性就小很多。适合对召回率敏感、宁可多返回再交给下游精排的场景。

### 使用场景

| 场景 | 现象 | 建议 |
|---|---|---|
| 用户提问措辞与文档术语不一致 | 检索结果相关度低、漏召回 | 用 MultiQueryRetriever，多路变体覆盖不同措辞 |
| 文档里同一概念有多种叫法 | 同义词、别名命中率低 | 用 MultiQueryRetriever，LLM 改写可生成同义表达 |
| 查询简短含糊、意图没展开 | 检索结果发散、不聚焦 | 用 MultiQueryRetriever，多路改写把意图拆开 |
| 不想额外调用 LLM、预算有限 | 检索够用，不想多花一次生成 | 用 StaticQueryGenerator 或普通检索器 |
| 检索要求精确、返回量要小 | 更看重精确度而非召回 | 多路召回后接重排序精排 |

### 工作方式

```
用户查询 → LLM 生成 N 个变体 → 分别检索 → 合并去重 → 返回结果
```

关键行为：

- **查询改写**：LLM 把原始查询改写成 N 个变体，数量由 `with_num_queries` 控制。改写不止换措辞，还会从不同角度拆解意图，覆盖同义词、缩写、口语化表达，让每一路都能命中不同类型的文档。
- **并行检索**：每个变体分别调用底层检索器，每路返回 `k_per_query` 条结果。底层只要是实现了 `RetrieverTrait` 的检索器即可——`SimilarityRetriever`、`BM25Retriever`、`UnifiedHybridIndex` 都能接，不限于向量检索。
- **合并去重**：把 N 路结果汇总，同一文档被多路查到只保留一份。
- **截断返回**：合并后的结果按 `final_k` 截断，返回最终 Top-K。多路召回会放大返回量，`final_k` 是最终出口，控制喂给下游的结果数。

### 使用方法

```rust
use langchainrust::{MultiQueryRetriever, SimilarityRetriever, OpenAIChat};
use std::sync::Arc;

let llm = OpenAIChat::new(config);
let retriever = Arc::new(SimilarityRetriever::new(store, embeddings));

let multi_query = MultiQueryRetriever::new(llm, retriever)
    .with_num_queries(3)
    .with_k_per_query(5)
    .with_final_k(10);

let docs = multi_query.retrieve_multi("database timeout").await?;
```

参数说明：

| 参数 | 作用 | 示例取值 |
|---|---|---|
| `with_num_queries` | LLM 生成的查询变体数量，变体越多覆盖越广，但 LLM 调用越贵 | `3` |
| `with_k_per_query` | 每个变体各自检索返回的结果数，决定每路的召回深度 | `5` |
| `with_final_k` | 合并去重后最终返回的结果数，是喂给下游的最终数量 | `10` |

注意：

- **LLM 不限定 OpenAI**：MultiQueryRetriever 内部持有实现了 `BaseChatModel` 的聊天模型（trait object 形式），示例里的 `OpenAIChat` 只是其中一种，任意 provider 都能接。
- **LLM 输出的解析是脆弱点**：查询变体来自 LLM 的自由文本输出，按行切分解析。如果模型输出带编号（"1. xxx"）、引号或多余解释，脏文本可能被当成查询，导致某一路召回奇怪的结果。生产环境建议给模型明确的输出格式要求。
- **增强器不是检索器**：MultiQueryRetriever 消费 `Arc<dyn RetrieverTrait>` 但自身不实现该 trait，所以它不能再被另一层增强检索器包装。

### StaticQueryGenerator（无需 LLM）

不需要 LLM 的查询生成器——通过同义词表扩展查询。适合不想额外调用 LLM、或查询模式可预测的场景。

```rust
use langchainrust::StaticQueryGenerator;
use std::collections::HashMap;

let synonyms: HashMap<String, Vec<String>> = HashMap::from([
    ("database".to_string(), vec!["DB".to_string(), "storage".to_string()),
]);

let generator = StaticQueryGenerator::new()
    .with_synonym_expansion(synonyms);

let queries = generator.generate("database connection failed");
```

关键行为：

- **词级扩展**：`generate` 拿查询词去查同义词表，把命中的词替换或扩展成多个变体。不走 LLM，零额外调用、零延迟。
- **与 MultiQueryRetriever 的取舍**：StaticQueryGenerator 是"字典式"扩展，只处理预先登记的同义词，不会生成全新的自然语言问法；MultiQueryRetriever 是"生成式"扩展，变体更灵活但更贵。同义词明确、查询模式可预测时用前者，语料术语复杂、需要生成式改写时用后者。
- **返回查询列表**：`generate` 返回展开后的查询列表，可自行决定如何交给检索器使用。

---

<a id="hyde-retriever"></a>
## HyDE 检索器

**HyDE（Hypothetical Document Embeddings）** 解决"查询太短、与文档不匹配"的问题：先用 LLM 生成一个假设性答案（可能不准确），再用这个假设答案的嵌入去检索真实文档。假设答案的措辞更接近真实文档，所以检索效果更好。

它的思路是"先把答案写出来再找"：查询太短是向量检索的老问题，像 "Rust concurrency" 这种短语，嵌入只刻画了几个关键词的语义，与文档里"async/await、线程安全、数据竞争"这种展开的表述距离很远，相似度打分就低。HyDE 让 LLM 就查询生成一段假设性的回答文档，这段文本的措辞、句长、信息密度都和真实文档更接近，再用它的嵌入去检索，命中率自然更高。注意假设答案本身可以是错的——它只是用来"对齐措辞"，真正返回的还是检索到的真实文档。适合查询过短、过于口语、与文档长文风格差距大的场景。

### 使用场景

| 场景 | 现象 | 建议 |
|---|---|---|
| 查询太短（几个关键词） | 嵌入只刻画关键词，与长文文档相似度低 | 用 HyDE，先生成假设文档再检索 |
| 查询口语化、文档是书面长文 | 措辞风格不匹配，检索效果差 | 用 HyDE，假设文档把口语转成书面长文风格 |
| 用户提问与文档表述差距大 | 向量相似度打不准 | 用 HyDE 提高召回 |
| 担心假设答案带偏检索 | 生成内容质量不稳定 | 打开 `with_include_original_query`，把原始查询一起并入检索 |
| 召回已够、只求精确 | 检索结果足够相关 | 不需要 HyDE，直接检索 + 重排序 |

### 工作方式

```
用户查询 → LLM 生成假设文档 → 使用假设文档检索 → 返回真实文档
```

关键行为：

- **生成假设文档**：LLM 就查询写一段像模像样的回答（假设答案）。这一步的价值不在"答得对"，而在"写得像文档"——把短查询补成与真实文档同风格的长文。
- **用假设文档检索**：把假设文档交给底层检索器。示例中底层是 `SimilarityRetriever`，检索器内部会先对假设文档做嵌入，再与库里文档算相似度。因此 HyDE 本身不再需要单独的嵌入模型参数，假设文档的向量化由底层检索器处理。
- **返回真实文档**：检索命中、排序都发生在"假设文档 ↔ 真实文档"之间，最终返回的是真实文档而不是假设文档。假设文档只在检索那一刻出现，用完即弃。

### 使用方法

```rust
use langchainrust::{HyDERetriever, SimilarityRetriever, OpenAIChat, OpenAIEmbeddings};
use std::sync::Arc;

let llm = OpenAIChat::new(config);
let embeddings = Arc::new(OpenAIEmbeddings::new(api_key));
let base_retriever = Arc::new(SimilarityRetriever::new(store, embeddings));

let hyde = HyDERetriever::new(llm, base_retriever)
    .with_k(5)
    .with_include_original_query(true);

let docs = hyde.retrieve("Rust concurrency").await?;
```

参数说明：

| 参数 | 作用 |
|---|---|
| `with_k(5)` | 最终返回的结果数，把前 k 条真实文档返回给下游 |
| `with_include_original_query(true)` | 检索时是否把原始查询与假设文档一起作为检索入口。打开后相当于"原始问法 + 假设答法"双路检索，降低假设答案带偏的风险 |

注意：

- **LLM 不限定 OpenAI**：HyDERetriever 持有实现了 `BaseChatModel` 的聊天模型（trait object），示例中的 `OpenAIChat` 只是其中一种。
- **底层检索器是接口**：HyDE 消费 `Arc<dyn RetrieverTrait>`，`SimilarityRetriever`、`BM25Retriever`、`UnifiedHybridIndex` 都能作为底层。对关键词检索来说，假设文档也比短查询包含更全的关键词，同样有帮助。
- **增强器不是检索器**：和 MultiQueryRetriever 一样，HyDERetriever 自身不实现 `RetrieverTrait`，不能再被另一层增强包装。

---

<a id="selfqueryretriever"></a>
## SelfQueryRetriever ✨ v0.18.0

用户问的是自然语言，但文档里的字段是结构化元数据——"去年的科技新闻"其实隐含了 `year >= 2024 AND category = tech` 的过滤条件。SelfQueryRetriever 让 LLM 把查询拆成 `{query, filter}`：清洗后的 query 走向量检索，解析出的 `MetadataFilter` 交给 `similarity_search_with_filter` 做元数据过滤（建立在统一过滤之上）。

```rust
use langchainrust::{SelfQueryRetriever, OpenAIChat};
use std::sync::Arc;

let retriever = Arc::new(SelfQueryRetriever::new(
    OpenAIChat::new(config),
    store,          // Arc<dyn VectorStore>
    embeddings,     // Arc<dyn Embeddings>
    vec!["category".to_string(), "year".to_string()], // allowed_attributes 白名单
));

let docs = retriever.retrieve("去年的科技新闻", 5).await?;
```

关键行为：

- **白名单防乱用字段**：`allowed_attributes` 是唯一允许出现在 filter 里的字段集合；LLM 构造了白名单外的字段时，整个 filter 被丢弃并记 warning，回退无过滤检索（避免 LLM 用不存在的字段过滤导致空结果）。
- **结构化优先、文本回落**：拆解走 `structured_call`（与 Guardrails / Evaluation 同源）拿结构化参数；模型不支持结构化输出时，把整段文本当查询词回落。
- **可进 LCEL**：实现 `RetrieverTrait`，用 `RetrieverRunnable` 包进链中与其他检索器一样组合。

对比：MultiQuery / HyDE 解决的是**召回不足**（问法变体、假设文档），SelfQuery 解决的是**查询里隐含的过滤意图**——语料带结构化元数据、用户查询里带筛选条件时用它。

---

<a id="reranking"></a>
## 重排序

初次检索可能返回不太相关的结果。重排序器对检索结果重新评分，把最相关的排到前面，提高精确度。

它的定位是"召回之后、喂给模型之前"的一道精排工序。第一次检索（召回）追求"别漏掉"，宁可多返回一些；重排序在召回结果上再做一次更严格的打分，把不相关的压下去、最相关的提到最前，再保留 `top_n` 条交给下游。它解决的问题是**精确度不足**——召回结果里混着不相关段落，直接全量喂给 LLM 会稀释注意力、浪费上下文。适合与 MultiQueryRetriever、HyDE 这类"扩大召回"的增强器搭配使用：增强器负责多捞，重排序负责精选。

### 使用场景

| 场景 | 现象 | 建议 |
|---|---|---|
| 召回结果多、相关性参差不齐 | 相关文档埋在不相关结果里 | 用重排序，把最相关的提到最前 |
| 与 MultiQuery/HyDE 搭配 | 多路召回放大返回量、掺杂噪声 | 重排序精排，只留 `top_n` 条给下游 |
| 只做一次检索、结果已够精准 | 前几条就是想要的 | 不需要重排序，省一次打分 |
| 想要可控的返回数量 | 每路召回返回量不可控 | 用 `with_top_n` 固定最终数量 |

### 支持的重排序器

| 重排序器 | 说明 |
|----------|-------------|
| **KeywordReranker** | 关键词匹配重排序 |
| **BM25Reranker** | BM25 公式重排序 |

两者都不需要额外模型调用，直接对传入的检索结果打分，速度快、成本低。区别在打分公式的精细程度，怎么选见下表：

| 怎么选 | KeywordReranker | BM25Reranker |
|---|---|---|
| 打分依据 | 查询关键词在文档中出现的位置与次数 | BM25 公式：词频 + 稀有度 + 文档长度归一化 |
| 复杂度 | 简单，关键词命中即高分 | 更精确，区分度更好 |
| 是否需要嵌入模型 | 不需要 | 不需要 |
| 可调参数 | 无 | `with_params(k1, b)`，控制词频饱和与长度惩罚强度 |
| 适合场景 | 快速、粗略、结果集小 | 结果集大、需要更细的区分度 |

### KeywordReranker

基于关键词匹配重排序——查询中的关键词在文档中出现越多、越靠前，分数越高。简单快速，不需要嵌入模型。

```rust
use langchainrust::{KeywordReranker, RerankingExecutor};

let reranker = Box::new(KeywordReranker::new());

let executor = RerankingExecutor::new(reranker)
    .with_top_n(5)
    .with_min_score(0.5);

let reranked = executor.rerank("Rust programming", search_results)?;
```

关键行为：

- **打分机制**：对每条检索结果统计查询关键词的出现次数与位置——出现越多、越靠前，分数越高。这是"关键词命中"式的打分，不涉及语义。
- **保留 top_n**：`with_top_n(5)` 表示重排后只保留前 5 条，其余丢弃，`rerank` 返回的就是这 5 条，下游拿到的数量是确定的。
- **最小分数过滤**：`with_min_score(0.5)` 设置分数下界，低于该分数的结果会被过滤掉，用于剔除明显不相关的结果；不设则不过滤。
- **与检索解耦**：`rerank` 接收传入的 `search_results` 列表，不关心结果来自哪个检索器，所以可以接在 `SimilarityRetriever`、MultiQueryRetriever、HyDE 等任意检索结果之后。

### BM25Reranker

使用 BM25 公式重排序——比 KeywordReranker 更精确，考虑了词频饱和度和文档长度归一化。可调 k1/b 参数。

```rust
use langchainrust::{BM25Reranker, RerankingExecutor};

let reranker = Box::new(BM25Reranker::new()
    .with_params(2.0, 0.5));

let executor = RerankingExecutor::new(reranker).with_top_n(5);

let reranked = executor.rerank("Rust programming", results)?;
```

关键行为：

- **打分机制**：用 BM25 公式打分，比关键词命中多考虑了三点——词频饱和度（词出现到一定程度后边际收益递减）、文档长度归一化（长文档里多出现一次不稀奇）、逆文档频率（越稀有的词越重要）。
- **可调参数**：`with_params(k1, b)` 两个参数分别控制词频饱和度和长度归一化的强度，示例 `(2.0, 0.5)` 是常见起点，可在实际数据上微调。
- **保留 top_n**：`with_top_n(5)` 决定最终保留条数，重排后只返回前 5 条。示例里没设 `with_min_score`，即默认不按分数过滤。
- **同为无模型重排**：和 KeywordReranker 一样不需要嵌入模型，直接在已检索结果上打分，成本可控。

---

<a id="callbacks"></a>
## 回调

回调系统让你在 LLM 调用的关键节点（开始、结束、出错、流式 token）插入自定义逻辑，用于日志、追踪、监控。`CallbackManager` 管理多个处理器，按顺序触发。

### CallbackManager

管理多个回调处理器，支持组合使用（如同时输出到控制台和 LangSmith）：

```rust
use langchainrust::{CallbackManager, StdOutHandler, LangSmithHandler};
use std::sync::Arc;

let manager = CallbackManager::new()
    .add_handler(Arc::new(StdOutHandler::new()))
    .add_handler(Arc::new(LangSmithHandler::from_env()?));
```

### StdOutHandler

输出到标准输出（用于调试）。最简单的回调，直接打印 LLM 的输入输出。

```rust
use langchainrust::StdOutHandler;

let handler = StdOutHandler::new();
```

### FileCallbackHandler

输出到文件。支持 JSON 格式（便于程序解析）和文本格式（便于人阅读）。

```rust
use langchainrust::{FileCallbackHandler, LogFormat};

// JSON 格式
let handler = FileCallbackHandler::new("trace.json", LogFormat::Json);

// 文本格式
let handler = FileCallbackHandler::new("trace.log", LogFormat::Text);
```

### CallbackHandler 生命周期 ✨ v0.15.0

实现 `CallbackHandler` 即可接入回调系统。每个 Run 有三段生命周期回调：`on_run_start` → `on_run_end` / `on_run_error`；组件级钩子（`on_llm_start/end/new_token/thinking/error`、`on_chain_*`、`on_tool_*`、`on_retriever_*`）可选覆盖，默认空实现。`StdOutHandler` 的 `verbose` 开关控制是否打印组件级细节。

### LangSmith 追踪

LangSmith 是 LangChain 的官方追踪平台，用于监控和调试 LLM 应用。

#### 环境变量

```bash
export LANGSMITH_API_KEY="ls_xxxxx"       # 必填
export LANGSMITH_PROJECT="my-project"      # 项目名称
export LANGSMITH_TRACING="true"            # 启用追踪
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
```

#### 使用 LangSmithHandler

```rust
use langchainrust::{CallbackManager, LangSmithHandler, StdOutHandler};
use std::sync::Arc;

// 从环境变量自动配置
let langsmith = LangSmithHandler::from_env()?;

let manager = CallbackManager::new()
    .add_handler(Arc::new(StdOutHandler::new()))
    .add_handler(Arc::new(langsmith));
```

#### 手动配置

```rust
use langchainrust::{LangSmithHandler, LangSmithConfig};

let config = LangSmithConfig {
    api_key: "ls_xxxxx".to_string(),
    project: "my-project".to_string(),
    endpoint: "https://api.smith.langchain.com".to_string(),
    tracing: true,
    workspace_id: None,
};

let handler = LangSmithHandler::new(config);
```

#### LangSmith 功能

| 功能 | 说明 |
|---------|-------------|
| **追踪** | 记录每次 LLM 调用 |
| **监控** | 查看 token 用量、延迟 |
| **调试** | 比较不同版本输出 |
| **评估** | 测试集评估 |
| **分享** | 分享追踪链接 |

---

### OtelHandler

将 LLM / Chain / Tool / Retriever 的开始/结束/错误事件转换为 OpenTelemetry span。需要 `opentelemetry` feature 和已配置的全局 tracer provider。

```toml
[dependencies]
langchainrust = { version = "0.8", features = ["opentelemetry"] }
```

```rust
use langchainrust::{CallbackManager, OtelHandler};
use std::sync::Arc;

// set tracer provider first: opentelemetry::global::set_tracer_provider(...)
let manager = CallbackManager::new()
    .add_handler(Arc::new(OtelHandler::from_global("langchainrust")));
// llm.with_callbacks(Arc::new(manager));
```

嵌套 span；导出到 Jaeger / Tempo / Grafana。

---

<a id="evaluation"></a>
## 评估

量化 LLM 输出质量：在更改提示词 / 模型 / 添加 RAG 之后，运行评估集并查看分数是否提升。5 个类别共 10 个评估器，覆盖从字面匹配到 RAG 幻觉检测：

| 类别 | 评估器 | 描述 |
|----------|-----------|-------------|
| 字面匹配 | `ExactMatch` / `StringDistance` | 精确相等 / 归一化 Levenshtein 距离 |
| 语义 | `EmbeddingSimilarity` / `LLMAsJudge` / `PairwiseJudge` | 余弦相似度 / LLM 评判 / 成对比较（交换 A/B 以消除位置偏差） |
| 规则 | `ContainsKeyword` / `RegexMatch` / `LengthCheck` | 关键词 / 正则 / 长度 |
| 经典 NLP | `Bleu` | n-gram 精确率（字符级 + 平滑） |
| RAG | `Faithfulness` | 拆分声明，逐一验证，检测幻觉 |

### EvalRunner

对 `Dataset` 运行一组评估器，生成 `Report`（每个示例的分数 + 每个评估器的平均值）。支持从 JSONL 文件加载评估集。

```rust
use langchainrust::evaluation::*;
use async_trait::async_trait;

let dataset = Dataset::new(vec![
    Example::new("2+2=?", "4"),
    Example::new("capital of China?", "Beijing"),
]);
// or: Dataset::from_jsonl("eval.jsonl")?

struct MyLLM;
#[async_trait]
impl Predictor for MyLLM {
    async fn predict(&self, input: &str) -> Result<String, EvalError> {
        Ok("4".to_string())
    }
}

let runner = EvalRunner::new(vec![
    Box::new(ExactMatch),
    Box::new(StringDistance),
]);
let report = runner.run(&dataset, &MyLLM).await?;
println!("{:?}", report.summary);
// {"ExactMatch": 1.0, "StringDistance": 1.0}
```

### Faithfulness

将预测拆分为原子声明，并逐一对照参考（上下文）进行验证，检测捏造内容。对 RAG 最为有用。

```rust
use langchainrust::evaluation::{Faithfulness, Evaluator};
use langchainrust::OpenAIChat;

let judge = Faithfulness::new(OpenAIChat::new(config));
// reference is context: "annual leave 15 days"
let ok = judge.eval("", "annual leave 15 days, accruable", "annual leave 15 days").await?;
assert_eq!(ok.value, 1.0); // faithful

let halluc = judge.eval("", "annual leave 20 days", "annual leave 15 days").await?;
assert_eq!(halluc.value, 0.0); // fabricated, caught
```

`with_llm_split(true)` 使用 LLM 拆分声明（默认：按句号拆分）；`with_empty_score(x)` 设置无声明时的分数。验证并发执行（`join_all`）。

### LLMAsJudge（LLM 裁判） ✨ v0.15.0

用 LLM 按 0-10 打分，可自定义评分标准（`with_rubric`）和满分（`with_max_score`）。

```rust
use langchainrust::evaluation::LLMAsJudge;

let judge = LLMAsJudge::new(OpenAIChat::new(config))
    .with_rubric("从正确性、完整性、清晰性三方面评分")
    .with_max_score(10);
let score = judge.eval(input, output, reference).await?; // 0.0 ~ 10.0
```

### PairwiseJudge（成对比较） ✨ v0.15.0

竞技场模式：让 LLM 裁判在两个回答中二选一，返回 `Verdict::{AWins, BWins, Tie}`。

```rust
use langchainrust::evaluation::{PairwiseJudge, Verdict};

let judge = PairwiseJudge::new(OpenAIChat::new(config));
match judge.compare("问题是?", &answer_a, &answer_b).await? {
    Verdict::AWins => { /* A 更好 */ }
    Verdict::BWins => { /* B 更好 */ }
    Verdict::Tie    => { /* 平局 */ }
}
```

> **位置偏差缓解**：自动交换 A/B 顺序跑两次，两次都选同一个才算真赢，否则判平局；两次调用并发发起，不增加串行往返。

### Report 容错 ✨ v0.15.0

`EvalRunner.run` 逐条容错：单条 `predict` 失败或某个评估器打分失败，只记入 `Report::failures`（含 `index` 与 `stage`），其余样例照常出分——一条坏数据不会拖垮整次评估。

```rust
let report = runner.run(&dataset, &MyLLM).await?;
if !report.failures.is_empty() {
    eprintln!("{} 条失败", report.failures.len());
}
```

> 底层复用 `core::judge::structured_call` 的结构化判定路径（强制 LLM 输出 JSON 后解析，错误统一为 `StructuredJudgeError`），保证裁判结果可机读。

---

<a id="mongodb-storage"></a>
## MongoDB 存储

MongoDB 存储解决两类问题：一是把**文档库**落到 MongoDB，让长文档的「父文档 + 子块」关系**跨进程共享、跨重启保留**；二是把**对话记忆**落到 MongoDB，让多轮记忆**多实例共享**。默认的内存存储（如 `InMemoryChunkedDocumentStore`）进程一退出就没了，专用向量库又太重；当应用要起多个实例、或需要真持久化时，MongoDB 是生产级的中间选项。

本节会出现两类对象，用途不同，别混淆：

| 对象 | 归属 | 存什么 | 一句话 |
|---|---|---|---|
| `MongoChunkedDocumentStore` | 文档库（DocumentStore 家族） | 文档正文 + 父子分块关系 | 长文档切块后，正文落 MongoDB |
| `MongoPersistentMemory` | 持久化记忆（记忆家族） | 对话历史 / 压缩摘要 | 多实例共享同一份对话记忆 |

工作流程（以文档库为例）：先 `create_indexes()` 建好查询索引 → `add_parent_document(doc, 500)` 把长文档按分块大小切成子块入库 → 按父文档 ID 用 `get_chunks_for_parent` 取回全部子块。检索命中小块后，再用子块 ID 回文档库取父块正文，这是分块检索的标准回源路径。

### 适用场景（什么时候选 MongoDB）

| 场景 | 是否推荐 | 原因 |
|---|---|---|
| 多实例部署，要共享同一份文档/记忆 | ✅ 推荐 | 所有实例连同一个 MongoDB，读到同一份数据 |
| 服务重启后数据要保留 | ✅ 推荐 | 数据落库，不依赖进程内存 |
| 单机演示、数据量小、想零依赖 | ⚠️ 可换轻量后端 | 本地可用 SQLite / 文件存储替代 |
| 可接受进程重启丢数据 | ❌ 不必要 | 内存实现更简单，不用引入服务 |

### 启用 Feature

```toml
[dependencies]
langchainrust = { version = "0.8", features = ["mongodb-persistence"] }
```

### 用法

```rust
use langchainrust::{MongoChunkedDocumentStore, MongoStoreConfig, ChunkedDocumentStoreTrait};

let config = MongoStoreConfig::new(
    "mongodb://localhost:27017",
    "langchainrust_db"
);

let store = MongoChunkedDocumentStore::new(config).await?;
store.create_indexes().await?;

// 与 InMemory 相同的接口
let (parent_id, chunk_ids) = store.add_parent_document(doc, 500).await?;
let chunks = store.get_chunks_for_parent(&parent_id).await?;
```

`MongoStoreConfig::new` 的两个参数：

| 参数 | 含义 | 例子 |
|---|---|---|
| 第一个参数 | MongoDB 连接串 | `mongodb://localhost:27017` |
| 第二个参数 | 数据库名 | `langchainrust_db` |

### MongoPersistentMemory（对话记忆持久化）

`MongoChunkedDocumentStore` 管「文档正文」，`MongoPersistentMemory` 管「对话记忆」，两者是不同层的持久化。`MongoPersistentMemory`（详见 [MongoPersistentMemory](#mongopersistentmemory)）内部组合 `ConversationSummaryBufferMemory`、自带 token 预算，把「历史 + 摘要」写进 MongoDB，多个实例连同一个库就能共享同一份记忆。

| 行为 | 说明 |
|---|---|
| 持久化 | 记忆存 MongoDB，服务重启不丢 |
| 多实例共享 | 同一库同一集合，多实例读写同一份记忆 |
| token 预算 | 内部是摘要缓冲，超出预算自动压缩 |
| 乐观锁 | 并发写不互相覆盖，防「后写覆盖先写」 |
| 会话绑定 | `set_session_id_async` 绑定当前会话 |

### 关键行为

- `create_indexes()` 建索引：首次建库建议先调用，为后续按父 ID / 子块 ID 查询做准备。
- 父子分块关系持久化：删父文档会连带删掉它的全部子块。
- 接口与 InMemory 实现一致：同一 `ChunkedDocumentStoreTrait`，换后端只换构造一行。
- 存正文不存向量：向量由配套的向量库（如 `ChunkedVectorStore`）索引，`MongoChunkedDocumentStore` 只负责「正文 + 分块关系」。
- 提供 `_blocking` 同步方法，供 BM25 这类同步检索路径使用。

### 怎么选

什么时候用 MongoDB 文档库？一句话：需要**多进程/多实例共享同一份文档正文**，或需要**真持久化**时。如果只是单机、数据量小，SQLite 文档库更轻（见下节）；如果数据量大、要专业向量检索，就再配一个向量库——向量放 vector store，正文放这里。

---

<a id="redis--sqlite-storage"></a>
## Redis / SQLite 存储

这两个都是 `ChunkedDocumentStoreTrait` 的轻量实现，管「文档正文 + 父子分块」，和 MongoDB 文档库干同一件事，但取舍相反：**Redis 走分布式共享，SQLite 走本地零依赖**。选谁，取决于你手里已有什么基础设施、数据要不要多实例共享。

工作流程和 MongoDB 文档库完全一样：`add_parent_document(doc, 500)` 切块入库 → `get_chunks_for_parent` 按父 ID 取回子块。接口统一，换后端只换构造这一行。

### 适用场景与取舍

| 后端 | 数据存哪 | 需要外部服务 | 数据生命周期 | 适用场景 |
|---|---|---|---|---|
| `RedisDocumentStore` | Redis 服务器内存 | 是，先起 Redis | 常驻 Redis；是否落盘取决于 Redis 服务自身的持久化配置（RDB/AOF），不归本库管 | 多实例共享、已有 Redis 基础设施、要跨进程一致 |
| `SQLiteDocumentStore` | 本地 `.db` 文件 | 否，零依赖 | 直接写本地文件，进程退出数据保留 | 单机、本地开发、免服务 |

一句话记法：Redis 是「多人共用的共享仓库」，SQLite 是「这台机器自己的抽屉」。

### 启用 Feature

```toml
[dependencies]
langchainrust = { version = "0.8", features = ["redis-storage"] }
```

或

```toml
[dependencies]
langchainrust = { version = "0.8", features = ["sqlite-storage"] }
```

### RedisDocumentStore

```rust
use langchainrust::{RedisDocumentStore, ChunkedDocumentStoreTrait};

let store = RedisDocumentStore::new("redis://127.0.0.1:6379").await?;

let (parent_id, chunk_ids) = store.add_parent_document(doc, 500).await?;
let chunks = store.get_chunks_for_parent(&parent_id).await?;
```

### SQLiteDocumentStore

```rust
use langchainrust::{SQLiteDocumentStore, ChunkedDocumentStoreTrait};

let store = SQLiteDocumentStore::new("langchain.db").await?;

let (parent_id, chunk_ids) = store.add_parent_document(doc, 500).await?;
let chunks = store.get_chunks_for_parent(&parent_id).await?;
```

注意两者构造参数的不同语义：

| 后端 | 构造参数 | 含义 |
|---|---|---|
| `RedisDocumentStore::new(uri)` | `redis://127.0.0.1:6379` | Redis 连接串，指向一个已在运行的服务 |
| `SQLiteDocumentStore::new(path)` | `langchain.db` | 本地文件路径，文件不存在会自动创建 |

### 关键行为

- 两者都实现 `ChunkedDocumentStoreTrait`，都是**文档库**——存文档正文和父子分块关系，**不存向量**；向量另放向量库。
- `add_parent_document` 自动切块，返回 `(parent_id, chunk_ids)`；`get_chunks_for_parent` 按父 ID 取回全部子块。
- 删父文档会连带删掉它的全部子块。
- 提供 `_blocking` 同步方法，供 BM25 这类同步检索路径使用。
- 接口与 InMemory 实现一致，换后端只换构造一行。
- `RedisDocumentStore` 的数据可见性取决于所有实例是否连同一个 Redis；`SQLiteDocumentStore` 的数据就在本地文件里，单机使用最自然。

### 怎么选

| 你的情况 | 推荐 |
|---|---|
| 单机 / 本地开发 / 不想装任何服务 | `SQLiteDocumentStore` |
| 多实例部署 / 团队已用 Redis 基础设施 | `RedisDocumentStore` |
| 数据量大、要生产级可靠与更复杂查询 | `MongoChunkedDocumentStore`（上一节） |

### Feature 门控

| Feature 标志 | 存储后端 | 依赖 |
|-------------|-----------------|--------------|
| `redis-storage` | Redis | redis crate |
| `sqlite-storage` | SQLite | rusqlite crate |
| `mongodb-persistence` | MongoDB | mongodb crate |

---

<a id="testing"></a>
## 测试

### 概念引入：为什么要测试

在 21 个 crate 组成的 workspace 里，测试是保证每个模块行为正确的最后防线——解析器、缓存、状态机这类纯逻辑一旦写错，会悄悄影响所有上层功能。用 `cargo test` 从 workspace 根目录跑一遍，能把各 crate 的单元测试、集成测试与文档测试一起执行，尽早发现问题。

```bash
cargo test
```

### 测试覆盖范围

| 层级 | 位置 | 覆盖对象 |
|---|---|---|
| 单元测试 | 各 crate 内部 | 解析器、缓存、淘汰策略、状态流转等纯逻辑 |
| 集成测试 | facade crate（`langchainrust`）的 tests 目录 | 跨模块组装、对外行为 |
| 文档测试（doctest） | 公开 API 的文档注释示例 | 示例代码可运行、API 真实存在 |

### 如何运行

- 根目录 `cargo test`：跑遍全部 crate
- `cargo test --workspace`：显式指定 workspace 全量
- 单独跑某个 crate：`cargo test -p langchainrust`（facade crate 的包名），或进入对应 crate 目录再跑

### 如何为 lib 写测试

**单元测试**：在实现文件里用 `#[cfg(test)]` 模块把测试代码圈起来，只测本模块内部逻辑，不依赖网络与真实模型。判键、解析、LRU 淘汰这类行为适合在这里逐条覆盖正反例。

**集成测试**：把 lib 当作外部使用者调用公开 API，验证跨模块的组装是否正确。涉及真实 API 的用例建议用 **mock 实现替换**，既快又不花预算，还能稳定断言结果——比如"敏感输出确实被拦截"这类行为，应该用明确的断言验证，而不是只"跑通不 panic"。

### 要点

- 纯逻辑优先用单元测试覆盖，快且好定位；覆盖正反例，别只测正常路径。
- 对外组合行为用集成测试覆盖；集成测试要有明确断言，不能只验证不 panic。
- 写文档示例时让它能作为 doctest 运行——示例即测试，示例里示范的 API 必须真实存在。

### 离线录播测试（lc-testkit） ✨ v0.16.0

`lc-testkit` 是独立的录播测试 harness crate（不走 facade，作为 dev-dependency 引入）：`RecordingProvider` 包住任意 `BaseChatModel` 真调一次、把请求/响应逐行录成 JSONL；`ReplayProvider` 零网络、确定性地回放——没 key 的 CI 也能测链。

```toml
[dev-dependencies]
lc-testkit = "0.18.0"
```

```rust
// 录：真调一次，成功后写入 fixture 文件
let recorded = RecordingProvider::new(real_llm, "fixtures/llm_chain_f01.jsonl")?;

// 回放：零网络、FIFO、逐字节稳定
let llm = ReplayProvider::from_file("fixtures/llm_chain_f01.jsonl")?;
let chain = LLMChain::new(llm, "用一句话回答:{question}");
let result = chain.invoke(inputs).await?;
```

录制是旁路不是拦截：真调失败不写录播；写盘失败仅告警不阻断真实结果。内置 round-trip（录→回放逐字节一致）与真链回放测试。

**三档回放策略（v0.18.0 起）**：

| 策略 | 匹配方式 | 何时用 |
|------|----------|--------|
| `Fifo`（默认） | 按录制顺序逐个出队 | 单请求顺序回放、简单场景 |
| `ByToolName` | 请求侧工具名命中即取 | 多工具并行、按工具路由 |
| `Exact` ✨ v0.18.0 | 请求 `messages` 完整签名逐条严格匹配 | 并行乱序下精确对应；无匹配返回 `TestkitError::ReplayNoMatch`（不做静默 FIFO 兜底） |

```rust
use lc_testkit::{ReplayProvider, ReplayStrategy};

let llm = ReplayProvider::from_file("fixtures/llm_chain_f01.jsonl")?
    .with_strategy(ReplayStrategy::Exact);
```

---

<a id="a2a-agent-protocol"></a>
## A2A 智能体协议 ✨ v0.4.1

### 概念引入：A2A 是什么

[A2A](https://github.com/google/A2A)（Agent-to-Agent）是 Google 推出的智能体间互操作协议，解决"不同团队、不同厂商开发的智能体如何互相调用"的问题。LangChainRust 提供完整的 A2A 支持：Server 用于暴露智能体，Client 用于调用远程智能体，使用 JSON-RPC 2.0 风格的消息传递。

什么时候用：

- 要把自己的 Agent 开放给外部（跨组织 / 跨服务）调用
- 要编排调用远程 Agent，而不是本地函数调用
- 需要标准化的"发现 → 派活 → 查进度 → 取消"协议，而不是自己造轮子

分层定位：**A2A 管"Agent ↔ Agent 的通信"，MCP 管"Agent ↔ 工具 / 数据源的连接"**。两者常配合使用——MCP 提供工具，Agent 通过 A2A 互相协作，职责更清晰。

### 协议流程

典型的 A2A 调用流程（4 步）：

| 步骤 | 操作 | 角色 |
|---|---|---|
| 1. 发现 | `get_agent_card` 获取远程智能体的 Agent Card | Client |
| 2. 派活 | `send_task` 提交一个任务（`A2AMessage`） | Client → Server |
| 3. 查进度 | `get_task` 按任务 ID 查询状态与结果 | Client |
| 4. 取消 | `cancel_task` 取消未完成的任务 | Client |

Server 侧对应两个端点：

- `GET /.well-known/agent-card.json` → 返回 Agent Card（智能体的自我描述，供发现）
- `POST /` → 接收并处理 JSON-RPC 请求（`handle_a2a_request`）

### A2AServer（暴露你的智能体）

`A2AServer` 提供可插入任何 HTTP 框架（axum、actix、warp）的处理函数——它不会启动自己的 HTTP 监听器。

```rust
use langchainrust::a2a::{A2AServer, AgentCard};
use langchainrust::LLMChain;
use std::sync::Arc;

let chain = Arc::new(LLMChain::new(llm, "You are a helpful assistant"));
let server = A2AServer::new(chain)
    .with_card(AgentCard::new("my-agent", "A helpful agent", "http://localhost:8080"));

// 在你的 HTTP 处理函数中：
// GET  /.well-known/agent-card.json → server.get_agent_card()
// POST /                       → server.handle_a2a_request(body).await
```

**任务持久化**：来自 `tasks/send` 的任务存储在内存中的 `RwLock<HashMap>` 中。`tasks/get` 检索任务，`tasks/cancel` 转换其状态。生产环境中，请使用自己的数据库支持的存储进行包装。

### A2AClient（调用远程智能体）

```rust
use langchainrust::a2a::{A2AClient, A2AMessage};

let client = A2AClient::new("http://remote-agent:8080".to_string()).unwrap();

// 发现智能体
let card = client.get_agent_card().await?;

// 发送任务
let task = client.send_task(A2AMessage::user("hello")).await?;

// 获取任务
let task = client.get_task(&task.id).await?;

// 取消任务
let task = client.cancel_task(&task.id).await?;
```

**当前边界**：`tasks/send` / `tasks/get` / `tasks/cancel` 与 `AgentCard` 已实现；任务状态机扩展、鉴权（token）、流式推送为规划项（⏳）。部署示例见仓库 `crates/lc/examples/a2a_http_server.rs`（axum HTTP 封装）。

### 关键行为与边界

| 能力 | 状态 |
|---|---|
| `tasks/send` | 已实现（任务存内存 `RwLock<HashMap>`） |
| `tasks/get` | 已实现（按任务 ID 检索） |
| `tasks/cancel` | 已实现（转换任务状态） |
| Agent Card 发现 | 已实现 |
| 任务状态机扩展（submitted → working → terminal） | ⏳ 规划 |
| 鉴权（token） | ⏳ 规划 |
| 流式推送 | ⏳ 规划 |
| TLS / 速率限制 | ⏳ 规划，生产部署前建议在 HTTP 层自补 |

### 生产注意点

- 任务存储是内存级，进程重启即丢失；生产环境建议换成数据库支持的存储（可事务、可恢复）。
- 规范要求鉴权；在自建的 HTTP 层补上 token 校验，别让任意任务 ID 可被查询 / 取消。
- 部署示例见仓库 `crates/lc/examples/a2a_http_server.rs`（axum HTTP 封装）。

### 怎么选

- 跨组织 / 跨服务、需要标准协议互操作时用 A2A。
- 只是进程内多个 Agent 协作，优先用本地编排（交接、派活收结果），别引入网络协议。
- Agent 要干活（调工具 / 数据源）时配合 MCP 分层，一个管工具、一个管 Agent 间通信。

---

<a id="with_structured_output"></a>
## with_structured_output ✨ v0.4.1

### 概念引入：给个 schema，一步拿到强类型对象

传统做法是"提示模型输出 JSON → 手动解析 → 出错容错 → 再转成类型"，又烦又容易坏——模型输出常常裹着 JSON 代码块、带点废话、偶尔格式不标准。`with_structured_output` 让框架替你完成整套流程：**给个 schema，一步拿到强类型对象**。

什么时候用：

- 需要把模型输出解析成程序可直接使用的结构体
- 想省掉手写"提示 JSON → 解析 → 容错 → 转换"的胶水
- 对输出字段有强类型要求，编译期就想要类型安全

### 工作机制

`StructuredOutputExt` trait 提供 `with_structured_output`，核心是"两级优先"：

1. **优先函数调用**：模型支持函数调用时，框架把 schema 作为工具声明绑给模型；模型返回 tool_calls，框架直接从结构化参数里解析出目标类型——一步到位，不依赖文本格式。
2. **回落 JsonOutputParser**：模型不支持函数调用、或没有返回工具调用时，自动回落到 `JsonOutputParser`——把模型输出的文本 JSON（自动剥掉 markdown 代码块、容错修复）解析成目标类型。

一次调用，两条路径自动切换，对调用方完全透明。

### 流式版本 stream_structured_output

`with_structured_output` 拿的是"完整结果"；`stream_structured_output` 则是流式版——JSON 一边生成一边解析，基于 `PartialJsonParser` 增量解析器，**字段一出来就能用**。适合前端边收边渲染的场景，比如"书名先出来、作者后出来、年份最后"。

### 代码示例

```rust
use langchainrust::StructuredOutputExt;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(JsonSchema, Deserialize)]
struct Answer {
    city: String,
    population: u64,
}

let llm = OpenAIChat::new(config);
let answer: Answer = llm.with_structured_output::<Answer>().await?;
```

### 关键行为

| 行为 | 说明 |
|---|---|
| schema 定义 | 用 Rust 结构体 + `JsonSchema` / `Deserialize` 派生声明输出结构 |
| 函数调用优先 | 支持函数调用时走原生 tool_calls，直接解析结构化参数 |
| 自动回落 | 不支持函数调用 / 无 tool_calls 时回落 `JsonOutputParser` |
| 类型安全 | 返回编译期确定的强类型，不手写解析 |
| 流式版本 | `stream_structured_output` + `PartialJsonParser` 边生成边解析 |

### 怎么选

- 要"一次拿完整结果、类型确定"→ 用 `with_structured_output`。
- 要"边生成边用、逐字段渲染"→ 用 `stream_structured_output`。
- 只是想解析模型已有的输出（不是让模型按 schema 返回）→ 直接用 `JsonOutputParser` 或 `TypedOutputParser` 等解析器。

---

<a id="filevectorstore"></a>
## FileVectorStore ✨ v0.4.1

基于 JSON 持久化的向量存储。填补了 InMemory（不持久化）与外部数据库（过于重量级）之间的空白。

它把**向量 + 文档**序列化成 JSON 落到磁盘文件：进程退出数据不丢，重启后用同一个 `new(path, dim)` 就能把上次的数据加载回来；不需要装任何数据库，也没有网络依赖。适合演示、离线、小规模本地知识库这类「要持久化、但不想上服务」的场景。

工作流：`FileVectorStore::new(path, 4).await` 指定落盘路径和向量维度（**创建是 async，需 `.await`**）→ `add_documents(docs, embeddings)` 存一批「文档 + 向量」→ `similarity_search(&query, k)` 给查询向量取 top-k。写操作自动落盘，不用手动保存。

### 创建与加载

| 参数 | 含义 |
|---|---|
| `path` | JSON 文件路径（如 `./vectors.json`） |
| `dim` | 向量维度（如 4），建库时固定 |

加载方式：用相同的 `path` + `dim` 再调一次 `FileVectorStore::new(path, dim).await`，即可把上次落盘的数据读回来。维度在建库时固定，之后 `add_documents` 塞入维度不一致的向量会直接报错，防止污染同一份索引。

### 用法

```rust
use langchainrust::{FileVectorStore, VectorStore, Document, MockEmbeddings};
use std::path::PathBuf;

let path = PathBuf::from("./vectors.json");
let store = FileVectorStore::new(path, 4).await?;  // 4 维(async 创建)

let docs = vec![
    Document::new("Rust focuses on safety and performance").with_id("rust"),
    Document::new("Python is great for rapid development").with_id("python"),
];
let embeddings = vec![
    vec![1.0, 0.0, 0.0, 0.0],
    vec![0.0, 1.0, 0.0, 0.0],
];
let ids = store.add_documents(docs, embeddings).await?;

let query = vec![0.9, 0.1, 0.0, 0.0];
let results = store.similarity_search(&query, 2).await?;

// 持久化：文件自动写入；重启时使用 new(path, dim) 加载
store.clear().await?;
```

**特性**：原子写入（tmp+rename）、维度验证、跨实例持久化。

### 关键行为

| 行为 | 说明 |
|---|---|
| JSON 落盘 | 向量 + 文档写入 JSON 文件，跨重启保留 |
| 原子写入 | 先写临时文件再 rename，断电/崩溃不损坏已有文件 |
| 维度验证 | 建库时固定维度，塞错维度的向量直接报错 |
| 删除诚实 | 删不存在的文档返回 `DocumentNotFound`，不假装成功 |
| 跨实例持久化 | 同一路径的文件，多个实例都能读（共享磁盘 / 演示场景） |
| 纯存储 | 向量由调用方生成后传入，`similarity_search` 也直接收查询向量 |

### 适用场景

- 演示 / 原型：不想为演示装一个数据库服务。
- 离线小规模知识库：本地数据量小，JSON 文件足够。
- 免外部服务：没有任何网络依赖，开箱即用。
- 数据量小但重启要保留：内存会丢，文件不会。

### 怎么选

| 你的情况 | 推荐 |
|---|---|
| 数据可丢、进程内即可 | `InMemoryVectorStore` |
| 要持久化、数据量小、免服务 | `FileVectorStore` |
| 长文档要切块检索 | `ChunkedVectorStore` + 文档库 |
| 数据量大 / 高并发 / 上生产 | 专业向量库（如 Chroma / Qdrant / Pinecone） |

---

<a id="computerusetool"></a>
## ComputerUseTool ✨ v0.4.1

### 概念引入：让 Agent 操控浏览器 / 桌面

普通工具让 Agent "调用 API、查数据"；`ComputerUseTool` 让 Agent **像人一样操作屏幕**——截屏看界面、移动鼠标点击、敲键盘输入。它对齐 Anthropic 的 computer use API，适合"没有现成接口、只能靠界面操作"的自动化任务。

什么时候用：

- 网页 / 桌面应用没有可用 API，只能靠 UI 操作
- 自动化"填表单 → 点按钮 → 读结果"这类 GUI 流程
- 让 Agent 通过"看屏幕 + 操作"完成数据录入、界面巡检等任务

### 能力

| 能力 | 作用 |
|---|---|
| 截图 | Agent 先"看"当前屏幕，知道界面长什么样 |
| 鼠标点击 | 点击目标位置 / 元素，完成选择、提交等操作 |
| 键盘输入 | 填文本框、按快捷键 |

### 作为工具接入 Agent

`ComputerUseTool` 实现 `BaseTool`，可以塞进 Agent 的工具列表。Agent 在"想 → 干 → 看结果 → 再想"的循环里按需调用它：先截图观察 → 决定点哪里 → 执行点击 → 再截图确认结果。

```rust
use langchainrust::ComputerUseTool;
use std::sync::Arc;

// Anthropic API 模式（默认）
let tool = ComputerUseTool::new();

// 作为 BaseTool 使用
let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(tool)];
```

### 使用注意

- 这是有真实副作用的工具——会操作真实屏幕，接入前先在可控环境（测试页面 / 隔离桌面）验证，注意权限与安全边界。
- Agent 通过截图"看"界面，截图质量直接影响它选点、输入的准确度。
- 默认是 Anthropic API 模式（见代码注释）；是否可换其它后端，以该工具版本的能力为准。

---

<a id="v050-new-features"></a>
## v0.5.0 新特性 ✨ v0.5.0

### RouterLLM（模型路由 + 回退）

`RouterLLM` 实现了 `BaseChatModel`，在异构模型池中路由调用，并在失败时回退。

**五种路由策略：**

| 策略 | 行为 | 使用场景 |
|----------|----------|----------|
| `Fallback` | 主模型失败 → 尝试下一个 | 生产环境容错 |
| `RoundRobin` | 在模型间轮转 | 负载均衡、避免速率限制 |
| `LeastLatency` | 选择最近最快的模型 | 延迟敏感场景 |
| `LowestCost` | 选择最便宜的模型 | 成本优化 |
| `InputDirected` | 基于输入文本的自定义闭包 | 按查询复杂度路由 |

```rust
use langchainrust::{RouterLLM, RoutingStrategy, BaseChatModel};

// 1. Fallback：主模型 + 备用模型
let router = RouterLLM::with_fallbacks(gpt4, vec![claude, local_model]);
let result = router.chat(messages, None).await?;

// 2. 最低成本路由
let router = RouterLLM::new(RoutingStrategy::LowestCost)
    .with_cost(cheap_model, 0.01)
    .with_cost(powerful_model, 0.03);

// 3. 输入定向路由
let router = RouterLLM::new(RoutingStrategy::InputDirected(Arc::new(|input| {
    if input.contains("code") { 1 } else { 0 }
})))
.with_model(general_model)
.with_model(code_model);

// 4. 最低延迟路由
let router = RouterLLM::new(RoutingStrategy::LeastLatency)
    .with_model(fast_model)
    .with_model(slow_but_smart_model);

// 作为普通 BaseChatModel 使用——即插即用替换
let result = router.chat(messages, None).await?;
let stream = router.stream_chat(messages, None).await?;
```

---

### CorrectiveRAG

标准 RAG 可能检索到不相关的文档，而 LLM 仍会幻觉出看似合理的答案。CorrectiveRAG 添加了三道关卡：评估文档 -> 重写查询或用网络搜索补充 -> 幻觉检查。

```rust
use langchainrust::agents::crag::CorrectiveRAGAgent;

let agent = CorrectiveRAGAgent::new(llm, retriever)
    .with_web_fallback(Box::new(web_tool))  // 可选：网络搜索回退
    .with_hallucination_check(true)       // 可选：幻觉检测（默认：true）
    .with_grade_threshold(0.6)            // 可选：相关性阈值（默认：0.6）
    .with_retrieve_k(4)                   // 可选：检索文档数量（默认：4）
    .with_grader_llm(grader_llm)          // 可选：独立的评分 LLM（避免自我验证偏差）
    .with_max_context_tokens(4000);       // 可选：截断低分文档以适应 token 预算

let answer = agent.invoke("What is Rust ownership?").await?;
```

**流程：** 查询 -> 检索 -> 评分 -> [不相关？ -> 重写/网络搜索 -> 重新检索] -> 生成 -> 幻觉检查 -> 输出

**Builder 方法：**

| 方法 | 默认值 | 描述 |
|-------|---------|-------------|
| `with_web_fallback(tool)` | None | 网络搜索工具（`Box<dyn BaseTool>`），用于补充较差的检索结果 |
| `with_hallucination_check(bool)` | `true` | 启用/禁用幻觉检测 |
| `with_grade_threshold(f64)` | `0.6` | 平均相关性分数低于此值时触发纠正路径（限制在 0.0-1.0） |
| `with_retrieve_k(usize)` | `4` | 检索的文档数量 |
| `with_grader_llm(llm)` | None | 用于幻觉检查的独立 LLM；避免模型倾向于认可自身输出的自我验证偏差 |
| `with_max_context_tokens(usize)` | None | 截断最低分文档以适应此 token 预算 |

---

### AdaptiveRAG

LLM 根据每个查询决定检索策略：NoRetrieval（跳过检索）、SingleSearch（单次查询）、MultiQuery（多角度查询）。

```rust
use langchainrust::agents::adaptive_rag::AdaptiveRAG;

let agent = AdaptiveRAG::new(llm, retriever);

// 复杂问题 -> LLM 选择 MultiQuery，生成多个查询
let answer = agent.invoke("Compare tokio vs async-std scheduling").await?;

// 简单问候 -> LLM 选择 NoRetrieval，完全跳过检索
let answer = agent.invoke("Hello").await?;
```

---

### GraphRAG（知识图谱 RAG）

向量搜索会遗漏关系。GraphRAG 提取实体 + 关系 -> 构建图 -> Label Propagation 社区检测 -> 社区摘要 -> 按社区查询。

```rust
use langchainrust::{GraphRAG, GraphQueryMode};

let mut graph_rag = GraphRAG::new(llm);
graph_rag.add_documents(&documents).await?;

// 全局查询：搜索社区摘要（宏观问题）
let result = graph_rag.query("overall tech stack architecture", GraphQueryMode::Global).await?;

// 局部查询：搜索实体邻居（具体问题）
let result = graph_rag.query("Alice's advisor's students", GraphQueryMode::Local).await?;

// 混合：结合两者
let result = graph_rag.query("...", GraphQueryMode::Hybrid).await?;
```

**流水线：** 文档 -> LLM 实体+关系提取 -> 图构建 -> Label Propagation 社区检测 -> LLM 社区摘要 -> 查询（Global/Local/Hybrid）。无外部图库依赖。

---

### Deep Research 智能体

多轮深度研究：将主题分解为子主题 -> 跨多个工具并行搜索 -> 去重 -> 综合 -> 发现空白 -> 重新搜索 -> 带引用的报告。

```rust
use langchainrust::agents::deep_research::DeepResearchAgent;

let agent = DeepResearchAgent::new(llm)
    .with_searcher(Box::new(DuckDuckGoSearchTool::new()))  // 添加搜索工具（至少需要一个）
    .with_max_rounds(3)           // 最大研究轮次（默认：2）
    .with_max_subtopics(5)        // 最大分解子主题数（默认：5）
    .with_max_source_tokens(8000);// 可选：截断来源片段以适应 token 预算

let report = agent.research("Compare Rust async runtimes: tokio vs async-std vs smol").await?;
println!("{}", report.markdown);           // 带内联引用的完整 markdown 报告
println!("Rounds: {}", report.rounds_completed);
for citation in &report.citations {
    println!("[{}] {} - {}", citation.index, citation.source, citation.snippet);
}
```

**Builder 方法：**

| 方法 | 默认值 | 描述 |
|-------|---------|-------------|
| `with_searcher(tool)` | None（必填） | 添加搜索工具；多个工具并行查询 |
| `with_max_rounds(n)` | `2` | 最大搜索-综合迭代次数 |
| `with_max_subtopics(n)` | `5` | 分解的最大子主题数 |
| `with_max_source_tokens(n)` | None | 截断来源片段以适应此 token 预算 |

**ResearchReport 字段：** `markdown`（带内联 `[1]` 引用的完整报告）、`citations`（按顺序排列，含 `index`/`source`/`url`/`snippet`）、`subtopics`（已调查的子主题）、`rounds_completed`。

---

### MCP 协议原语

MCP 规范定义了 6 类原语。LangChainRust 中 **已实现调用逻辑** 的是：`initialize`（握手）、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list`、`prompts/get`、`completion/complete`，以及流式工具结果（`notifications/tool_partial`）与取消（`notifications/cancelled`）。client→server 原语均为**注册制**：注册数据源后才返回真实数据，未注册仍返回 `method_not_found`。server→host 方向的 `sampling::create_message` / `elicitation::create` 由 `MCPServer` 发起，需注入回调。

| 原语 | 状态 | 说明 |
|-----------|---------|-------------|
| **Resources** | ✅ server 已接线 | `with_resource_provider` 注册数据源；`resources/list` / `resources/read` |
| **Prompts** | ✅ server 已接线 | `with_prompt_provider` 注册数据源；`prompts/list` / `prompts/get` |
| **Completion** | ✅ server 已接线 | `with_completion_provider` 注册数据源；`completion/complete` |
| **Elicitation** | ✅ 发起方法已接 | server→host；`MCPServer::create_elicitation` 需注入 `ElicitationHandler` 回调 |
| **Roots** | ⏳ 类型已定义 | 发现客户端根目录（client 能力，未接入） |
| **Sampling** | ✅ 发起方法 + 防护 | server→host；`create_message` 需注入 `SamplingHandler`；`SamplingGuard` 防护 |

> 服务端采样有独立的 `SamplingGuard`（深度 / token 预算 / 超时三重防护），见 [MCP](#mcp) 章节。client→server 原语未注册数据源时仍返回 `method_not_found`；server→host 原语未注入回调时返回明确错误。真实交互（采样 / elicitation）依赖宿主 UI、模型环境，由使用者经回调接入（测试用注入 mock 覆盖）。

---

### 代码解释器沙箱

使用 `LocalSandbox`（子进程 + 超时）进行安全代码执行。

```rust
use langchainrust::tools::sandbox::{LocalSandbox, CodeSandbox, SandboxTool, Language};

// 直接使用沙箱
let sandbox = LocalSandbox::new()
    .with_python_path("python3");  // 可选：自定义解释器路径

let result = sandbox.run("print(2 + 2)", Language::Python, 30_000).await?;
assert_eq!(result.stdout.trim(), "4");

// 或包装为 BaseTool 供智能体使用
let tool = SandboxTool::new(LocalSandbox::new(), Language::Python)
    .with_timeout(30_000);  // 30 秒超时
```

- **LocalSandbox**：子进程执行，超时自动终止，捕获 stdout/stderr，Python 危险导入检查（唯一内置后端）

---

### OpenAI Responses API

连接到 `/v1/responses`，使用内置工具：WebSearch、FileSearch、CodeInterpreter、ComputerUse——一次请求，模型自动处理工具调用。

```rust
use langchainrust::language_models::openai::responses::{ResponsesModel, ResponsesConfig, BuiltinTool};

let config = ResponsesConfig::new("your-api-key")
    .with_model("gpt-4o")
    .with_builtin_tool(BuiltinTool::WebSearch)
    .with_builtin_tool(BuiltinTool::CodeInterpreter);

let model = ResponsesModel::new(config);

let result = model.chat(messages, None).await?;
// result.content 包含工具执行后的最终答案
```

---

### Anthropic Extended Thinking

配置 `budget_tokens` 让 Claude 在回答前先思考。思考块通过 `LLMResult` 中的 `thinking_content` 暴露；流式输出通过 `on_llm_thinking` 回调。

```rust
use langchainrust::{AnthropicChat, AnthropicConfig};

let config = AnthropicConfig::new("your-api-key")
    .with_model("claude-sonnet-5");
let model = AnthropicChat::new(config)
    .with_thinking(10000); // 最多 10000 个思考 token

let result = model.chat(messages, None).await?;
println!("Thinking: {:?}", result.thinking_content);
println!("Answer: {}", result.content);
```

---

### 流式结构化输出

`PartialJsonParser` 增量地将流式 JSON 解析为部分结构体——无需等待所有 token。

```rust
use langchainrust::core::structured_output::StreamingStructuredOutputExt;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(JsonSchema, Deserialize, Clone, PartialEq, Default)]
struct UserInfo {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    age: Option<u32>,
    #[serde(default)]
    email: Option<String>,
}

let schema = serde_json::to_value(schemars::schema_for!(UserInfo)).unwrap();
let stream = model.stream_structured_output::<UserInfo>(schema, "Tell me about Alice, age 30").await?;
pin_mut!(stream);
while let Some(result) = stream.next().await {
    let partial = result?;
    if let Some(name) = &partial.name {
        println!("Got name: {}", name); // 在所有字段到达之前即可获取
    }
}
```

---

### Batch API

`BatchClient` 统一了 OpenAI 和 Anthropic 的批处理工作流：提交 → 轮询 → 结果，成本降低 50%。

```rust
use langchainrust::batch::{BatchClient, BatchProvider, BatchRequest};

let client = BatchClient::new(BatchProvider::OpenAI, "your-api-key");

let requests = vec![
    BatchRequest {
        custom_id: "req-1".to_string(),
        model: "gpt-4o".to_string(),
        messages: vec![Message::human("Translate: Hello")],
        temperature: None,
        max_tokens: None,
    },
    BatchRequest {
        custom_id: "req-2".to_string(),
        model: "gpt-4o".to_string(),
        messages: vec![Message::human("Translate: World")],
        temperature: None,
        max_tokens: None,
    },
];

let results = client.submit_and_wait(requests, 5000, 300_000).await?;
for result in results {
    println!("{}: {:?}", result.custom_id, result.result?.content);
}
```

---

### 追踪（分布式追踪）

`Tracer` + `SpanGuard`（RAII）自动管理父子 span。后端：InMemory / Console / OTel。

```rust
use langchainrust::callbacks::tracing::{Tracer, ConsoleTracingBackend, SpanKind};
use std::sync::Arc;

let tracer = Tracer::new(Arc::new(ConsoleTracingBackend));
let span = tracer.start("agent_run", SpanKind::Internal);
{
    let _retrieve = tracer.start_child("retrieve", SpanKind::Internal);
    let docs = retriever.retrieve(&query).await?;
} // _retrieve drop -> 子 span 自动记录结束时间
{
    let _generate = tracer.start_child("generate", SpanKind::Internal);
    let answer = llm.chat(messages, None).await?;
}
span.end(); // span 自动记录持续时间、token 数等
```

---

### v0.5.0 质量加固（176 项修复）

在实现 12 个新特性后，对 223 个文件进行了两轮全代码库审查，发现并修复了 176 个问题（23 CRITICAL / 63 HIGH / 75 MEDIUM / 15 LOW）。

**关键修复：**

- **安全**：PythonREPL 危险导入检查、HTTPTool/URLFetchTool SSRF 防护（私有 IP + DNS 重绑定）、SQLTool 注入防护、Gemini API 密钥移至 header
- **多轮函数调用**：Anthropic/Gemini/Ollama 工具消息映射错误导致多轮 FC 中断——全部修正
- **流式输出**：Ollama/Anthropic/Gemini SSE 跨 chunk token 丢失已修复；`Runnable::stream()` 从伪流式改为真实流式（逐 token 发射）
- **并发**：异步上下文中的 `std::sync::Mutex` 替换为 `tokio::sync::Mutex`；MCP Transport 请求级互斥锁；HandoffManager 锁合并
- **Panic 修复**：`choices[0]` 越界 → `.first().ok_or()`；`from_env()` 返回 `Result`；Regex → LazyLock；Mutex poison → `into_inner()` 恢复
- **数据正确性**：UTF-8 字符边界切片；RRF 文档 ID 使用内容哈希；错误传播替代静默吞没

**验证：** 826 个单元测试通过 · clippy 零警告 · cargo fmt 干净

---

<a id="v052-fixes"></a>
## v0.5.2 修复 ✨ v0.5.2

v0.5.2 是一个稳定性和正确性版本，包含对多个 v0.5.0 特性的关键错误修复。

### GraphRAG 社区摘要修复

社区摘要之前拼接的是原始实体 ID（`e_xxx`）而非实体名称，导致生成无意义的摘要，降低了 Global/Hybrid 查询质量。已通过 `store.get_entity()` 查找实体名称修复。

### Deep Research 报告格式修复

合成器之前要求 LLM 将完整 markdown 报告输出为 JSON 字符串字段，由于 markdown 中未转义的 `\n`、`"`、`\` 导致频繁的 `serde_json` 解析失败。替换为基于分隔符的格式：

```
<<<REPORT>>>
...markdown report...
<<<END_REPORT>>>
<<<GAPS>>>
[...gap descriptions...]
<<<END_GAPS>>>
```

报告部分现在是原始文本，无需转义。旧的 JSON 格式作为向后兼容的回退保留。

### DocumentStore 异步 Panic 修复

`InMemoryDocumentStore` 和 `InMemoryChunkedDocumentStore` 之前使用 `tokio::sync::RwLock` 的 `blocking_read()`/`blocking_write()`，在异步上下文中会因 "Cannot block the current thread from within a runtime" 而 panic。已切换为 `std::sync::RwLock`，在同步和异步上下文中均可工作。

### CRAG 评分改进

**阈值修复**：默认 `grade_threshold` 从 `0.5` 改为 `0.6`。旧阈值处于 LLM 评分最不稳定的区域，且模糊解析默认值（`0.5`）恰好等于阈值——使纠正触发近乎随机。现在模糊默认值为 `0.4`，远低于 `0.6` 阈值。

**幻觉检测偏差修复**：添加了 `with_grader_llm()` builder，注入独立的 LLM 进行幻觉检测，防止模型认可自身输出：

```rust
use langchainrust::agents::crag::CorrectiveRAGAgent;

let agent = CorrectiveRAGAgent::new(llm.clone(), retriever)
    .with_grader_llm(claude_llm)  // 使用不同的 LLM 进行评分
    .with_grade_threshold(0.6);    // 新默认值：0.6（原为 0.5）
```

其他改进：
- `GradeResult` 现在有 `is_ambiguous` 字段，指示分数是否来自模糊解析
- 幻觉检测提示词现在包含对抗性框架（"Be skeptical"）
- 幻觉检查 LLM 失败时优雅降级（返回 `grounded: false`）而非中止

### 其他 v0.5.2 变更

- **Feature gate 声明**：`sandbox-e2b` 和 `sandbox-wasm` feature 在代码中被引用但未在 `Cargo.toml` `[features]` 中声明——现已正确声明
- **Clippy 零警告**：所有 clippy 警告已解决

---

## 更多资源

| 资源 | 内容 |
|----------|---------|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | 贡献指南 |
| [API Docs](https://docs.rs/langchainrust) | Rust API reference |