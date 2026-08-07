# 使用指南

本文档提供详细的使用说明。如需快速概览，请参阅 [README.md](../README.md)。

---

## 目录

- [LLM](#llm)
  - 多 Provider 支持
  - OpenAI Chat
  - 流式输出
  - 函数调用
  - Ollama（本地 LLM）
  - Google Gemini
  - 多模态视觉
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
  - ContextWindow（长上下文管理） ✨ v0.4.1
- [LLM 缓存](#llm-cache)
- [链](#chains)
  - ConversationRetrievalChain
  - 链流式输出 ✨ v0.4.1
- [LCEL (LangChain Expression Language)](#lcel-langchain-expression-language-) ✨ v0.9.0
  - RunnableWithFallbacks ✨ v0.10.0
  - RunnableAssign ✨ v0.10.0
  - RunnableRetry ✨ v0.11.0
  - CancellationToken ✨ v0.11.0
- [文档链](#document-chains)
- [智能体](#agents)
  - Agent Hooks ✨ v0.11.0
  - Agent 流式输出 ✨ v0.12.0
- [Plan-Execute 智能体](#plan-execute-agent)
- [Handoffs](#handoffs)
- [流式工具调用](#streaming-tool-calls)
- [护栏](#guardrails)
- [Token 计数器](#token-counter)
- [会话](#sessions)
- [MCP](#mcp)
  - MCPServer
- [工具](#tools)
  - WikipediaTool
  - DuckDuckGoSearchTool
  - PythonREPLTool
  - 扩展工具 (HTTPTool / FileTool / SQLTool)
  - `#[tool]` 过程宏 ✨ v0.10.0
- [RAG](#rag)
  - ChromaDB
  - PGVectorStore
  - PineconeStore
  - SemanticSplitter
- [BM25](#bm25)
- [混合检索](#hybrid-retrieval)
- [文档加载器](#document-loaders)
  - HTMLLoader
  - DocxLoader ✨ v0.4.1
  - WebScraperLoader ✨ v0.4.1
  - SitemapLoader ✨ v0.4.1
- [MultiQueryRetriever](#multiqueryretriever)
- [HyDE 检索器](#hyde-retriever)
- [重排序](#reranking)
- [回调](#callbacks)
  - OtelHandler
- [评估](#evaluation)
  - 评估器（10 种类型）
  - EvalRunner
- [LangGraph](#langgraph)
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
  - MCP 完整协议（6 个原语）
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

## LLM

### 多 Provider 支持

LangChainRust 支持多个 LLM Provider，提供统一的 API：

| Provider | 类 | 特性 |
|----------|-------|----------|
| **OpenAI** | `OpenAIChat` | GPT-4, GPT-3.5-turbo |
| **DeepSeek** | `DeepSeekChat` | DeepSeek-V3，高性价比 |
| **Moonshot** | `MoonshotChat` | Kimi，长上下文 |
| **Qwen** | `QwenChat` | 阿里云 |
| **Zhipu** | `ZhipuChat` | ChatGLM |
| **Anthropic** | `AnthropicChat` | Claude，注重安全 |
| **Ollama** | `OllamaChat` | 本地部署 |
| **Gemini** | `GeminiChat` | Google Gemini，多模态 |

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
    if let Ok(token) = chunk {
        print!("{}", token);  // 实时输出
    }
}
```

### 函数调用

让 LLM 决定何时调用工具。`bind_tools` 将工具定义附加到 LLM，LLM 返回 `tool_calls` 而非纯文本。框架负责解析参数、调用工具、返回结果。

```rust
use langchainrust::{ToolDefinition, bind_tools};
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

**限制**：带工具调用的 Run（`requires_action`）尚未实现；会返回 `AssistantError::RequiresAction`。如需工具调用，请使用 `FunctionCallingAgent`。

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
use langchainrust::prompts::{LengthBasedExampleSelector, SemanticExampleSelector};

// 基于长度：选择不超过最大长度的示例
let selector = LengthBasedExampleSelector::new(examples, example_prompt, 50);

// 基于语义：通过嵌入选择最相似的示例
let selector = SemanticExampleSelector::new(embeddings, examples, 2);
```

---

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

## 记忆

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

### ContextWindow（长上下文管理） ✨ v0.4.1

`ContextWindow` 管理长对话的 token 预算，提供两种策略：截断（Truncate）和摘要（Summarize）。

```rust
use langchainrust::{ContextWindow, Message, OpenAIChat, Strategy};
use langchainrust::BaseChatModel;

// 策略 1：Truncate — 超出 token 预算时丢弃最旧的消息
let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096);

// 策略 2：Summarize — 超出预算时使用 LLM 压缩旧对话
let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096)
    .with_strategy(Strategy::Summarize)
    .with_llm(OpenAIChat::new(config));

cw.add_message(Message::human("hello")).await;
cw.add_message(Message::ai("Hi! How can I help?")).await;

let messages = cw.get_messages().await;
```

| 策略 | 行为 | 适用场景 |
|----------|----------|----------|
| `Truncate` | 超出预算时丢弃最旧的消息 | 简单场景 |
| `Summarize` | LLM 将旧对话压缩为摘要 | 需要保留关键信息的长对话 |

## LLM 缓存

LLM 调用是应用中最慢、最贵的部分。缓存对相同输入返回之前的结果，避免重复调用。支持 TTL 过期和容量上限。

### 带 TTL 的内存缓存

```rust
use langchainrust::cache::{LLMCache, CacheConfig};
use std::time::Duration;

let config = CacheConfig::new()
    .with_ttl(Duration::from_secs(3600))  // 1 小时
    .with_max_size(1000);                 // 1000 条记录

let cache = LLMCache::new(config);

// 与 LLM 配合使用
let llm = OpenAIChat::new(config)
    .with_cache(cache);

// 后续相同的调用返回缓存结果
let r1 = llm.chat(vec![Message::human("Hello")], None).await?;
let r2 = llm.chat(vec![Message::human("Hello")], None).await?;  // 缓存命中
```

---

## LCEL (LangChain Expression Language) ✨ v0.9.0

LCEL 提供类似 Python LangChain 的管道组合语法，将 `Runnable` 组件通过 `pipe()` 串联成流水线。

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
    RunnableExt, RunnableLambda, RunnableParallel, RunnableAssign,
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

---

## Agents

Agent 是能自主调用工具、多步推理的 LLM 应用。与 Chain 不同，Agent 不是固定流程，而是 LLM 根据输入动态决定调用哪些工具、执行多少步。

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

## Plan-Execute Agent

Plan-Execute Agent 先用 LLM 规划任务步骤，逐步执行，失败时重新规划，最后总结。适用于复杂的多步骤任务。

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

流程：规划 -> 逐步执行（FunctionCallingAgent + 工具）-> 失败时重新规划 -> 总结。

---

## Handoffs

受 OpenAI Agents SDK 启发：主 Agent 可以通过 `HandoffTool` 将任务委托给已注册的专家 Agent。

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

`handoff_tools()` 返回名为 `handoff_to_{agent}` 的工具；也可以通过 `execute_handoff(Handoff)` 直接委托。

---

## Streaming Tool Calls

普通 Agent 等整个执行完才返回结果。`StreamingFunctionCallingAgent` 逐 token 流式输出 LLM 文本，并通过事件流暴露工具调用状态——用户可以实时看到 Agent 的思考和操作过程。

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

事件：`AgentStreamEvent`（`Text` / `ToolCall` / `FinalAnswer`）和 `ToolCallState`。

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

---

## Token Counter

LLM 按 token 计费，但 token 数不等于字符数。`TiktokenCounter` 使用 OpenAI 的分词器精确计数；`TokenTrackingLLM` 包装 LLM 自动累计用量；`ModelPricing` 根据用量估算成本。

```rust
use langchainrust::{TokenTrackingLLM, ModelPricing, OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

let tracked = TokenTrackingLLM::for_openai(OpenAIChat::new(OpenAIConfig::default()))?;

let result = tracked.chat(vec![Message::human("hi")], None).await?;

let usage = tracked.get_usage();                               // prompt / completion / total tokens
let cost = tracked.estimate_cost(&ModelPricing::gpt4o_mini()); // USD
```

`ModelPricing::gpt4o()` / `gpt4o_mini()` 为内置定价；使用 `ModelPricing::new(prompt_per_1k, completion_per_1k)` 可自定义定价。

---

## Sessions

`SessionManager` 管理多轮对话会话的生命周期：创建/获取/归档，每次聊天自动维护历史，支持可插拔存储（`SessionStore` trait）。

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

`SessionStore` trait 包含 `create/get/update/delete/list_by_user`；可实现自己的后端（Redis/数据库）。`MemorySessionStore` 为内置实现，适用于测试和单进程使用。

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
let mcp_tools = client.as_tools().await;
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
use langchainrust::{tool, BaseTool, Tool, ToolError};

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

**SQLTool** -- 只读 SQL 查询（仅 SELECT，表白名单；需要 `sqlite-storage` feature）：

```rust
use langchainrust::tools::extended::SQLTool;

let sql = SQLTool::new("data.db")?
    .with_allowed_tables(vec!["users".into()]);
let rows = sql.execute("SELECT id, name FROM users")?; // Vec<HashMap<String,String>>
// 非 SELECT 语句（如 DROP/INSERT）会被拒绝
```

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
| **Mock** | `MockEmbeddings` | 自定义 | 测试用 |

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

## RAG

RAG（Retrieval-Augmented Generation）让 LLM 基于你的私有数据回答问题，而不是只靠训练时的知识。流程：文档 → 分割 → 嵌入 → 存入向量库 → 检索相关文档 → 连同问题发给 LLM。

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
use langchainrust::{ChromaVectorStore, SimilarityRetriever};
use std::sync::Arc;

let store = Arc::new(ChromaVectorStore::new(
    "http://localhost:8000",
    "my_collection",
).await?);

let retriever = SimilarityRetriever::new(store.clone(), embeddings);

retriever.add_documents(vec![
    Document::new("Rust is a systems language"),
]).await?;

let docs = retriever.retrieve("systems programming", 3).await?;
```

### PGVectorStore

PostgreSQL + pgvector 扩展向量存储。适合已有 PostgreSQL 基础设施、需要关系型数据库 + 向量检索合一的场景。需要 `pgvector-storage` feature；由于 `sqlx` / `pgvector` 依赖未在 crate 内启用，需自行在 `Cargo.toml` 中添加 `sqlx和 pgvector。

```rust
use langchainrust::vector_stores::PGVectorStore;
use langchainrust::embeddings::Embeddings;

let store = PGVectorStore::new(
    "postgres://user:pass@localhost/db",
    "docs",
    1536, // 向量维度
).await?;
// embeddings: impl Embeddings (e.g. OpenAIEmbeddings); docs: &[Document]
store.add_documents(&docs, &embeddings).await?;
let found = store.similarity_search("query", 5, &embeddings).await?;
store.delete("doc-id").await?;
```

`PGVectorStore::new` 会执行 `CREATE EXTENSION IF NOT EXISTS vector` 并创建表；`build_table_sql(table, dim)` 是用于表 DDL 的纯函数。

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

---

## BM25

BM25 是经典的关键词检索算法，根据词频和文档长度计算相关性分数。与向量检索（语义相似）不同，BM25 擅长精确关键词匹配，如搜索"Rust ownership"会优先返回包含这些词的文档。不需要嵌入模型，零成本，速度快。

### BM25Retriever（关键词搜索）

```rust
use langchainrust::{BM25Retriever, Document};

let mut retriever = BM25Retriever::new();

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
use langchainrust::{UnifiedHybridIndex, HybridIndexConfig, OpenAIEmbeddings};

let config = HybridIndexConfig::new()
    .with_chunk_size(500)
    .with_top_k(10, 10)        // BM25_k, Vector_k
    .with_rrf_k(60);

let embeddings = Arc::new(OpenAIEmbeddings::new(api_key));
let index = UnifiedHybridIndex::with_config(embeddings, 1536, config);

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

graph.add_node("analyze", |state: AgentState| {
    let mut new_state = state.clone();
    new_state.steps.push("analyzed".to_string());
    new_state
});

graph.add_node("process", |state: AgentState| {
    state
});

graph.add_edge(START, "analyze");
graph.add_edge("analyze", "process");
graph.add_edge("process", END);

let compiled = graph.compile();

let result = compiled.invoke(AgentState::new()).await?;
```

### 条件边

根据当前状态动态选择下一个节点。`FunctionRouter` 接收一个闭包，返回目标节点名称。适合"消息多就总结，少就继续"这类分支逻辑。

```rust
use langchainrust::langgraph::{ConditionalEdge, FunctionRouter};

let router = FunctionRouter::new(|state: &AgentState| {
    if state.messages.len() > 5 { "summarize" } else { "continue" }
});

graph.add_conditional_edge(
    "analyze",
    ConditionalEdge::new(router, vec!["summarize", "continue"]),
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

---

## 文档加载器

从各种文件格式加载文档，统一转为 `Document` 结构（`content` + `metadata`），供后续分割和检索使用。

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

### 工作方式

```
用户查询 → LLM 生成 N 个变体 → 分别检索 → 合并去重 → 返回结果
```

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

---

## HyDE 检索器

**HyDE（Hypothetical Document Embeddings）** 解决"查询太短、与文档不匹配"的问题：先用 LLM 生成一个假设性答案（可能不准确），再用这个假设答案的嵌入去检索真实文档。假设答案的措辞更接近真实文档，所以检索效果更好。

### 工作方式

```
用户查询 → LLM 生成假设文档 → 使用假设文档检索 → 返回真实文档
```

### 使用方法

```rust
use langchainrust::{HyDERetriever, SimilarityRetriever, OpenAIChat, OpenAIEmbeddings};
use std::sync::Arc;

let llm = OpenAIChat::new(config);
let embeddings = Arc::new(OpenAIEmbeddings::new(api_key));
let base_retriever = Arc::new(SimilarityRetriever::new(store, embeddings));

let hyde = HyDERetriever::new(llm, embeddings, base_retriever)
    .with_k(5)
    .with_include_original_query(true);

let docs = hyde.retrieve("Rust concurrency").await?;
```

---

## 重排序

初次检索可能返回不太相关的结果。重排序器对检索结果重新评分，把最相关的排到前面，提高精确度。

### 支持的重排序器

| 重排序器 | 说明 |
|----------|-------------|
| **KeywordReranker** | 关键词匹配重排序 |
| **BM25Reranker** | BM25 公式重排序 |

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

### BM25Reranker

使用 BM25 公式重排序——比 KeywordReranker 更精确，考虑了词频饱和度和文档长度归一化。可调 k1/b 参数。

```rust
use langchainrust::{BM25Reranker, RerankingExecutor};

let reranker = Box::new(BM25Reranker::new()
    .with_params(2.0, 0.5));

let executor = RerankingExecutor::new(reranker).with_top_n(5);

let reranked = executor.rerank("Rust programming", results)?;
```

---

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

---

## MongoDB 存储

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

---

## Redis / SQLite 存储

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

### Feature 门控

| Feature 标志 | 存储后端 | 依赖 |
|-------------|-----------------|--------------|
| `redis-storage` | Redis | redis crate |
| `sqlite-storage` | SQLite | rusqlite crate |
| `mongodb-persistence` | MongoDB | mongodb crate |

---

## 测试

```bash
cargo test
```

---

## A2A 智能体协议 ✨ v0.4.1

[A2A](https://github.com/google/A2A)（Agent-to-Agent）是 Google 的智能体间通信协议。LangChainRust 提供完整的 A2A 支持：Server 用于暴露智能体，Client 用于调用远程智能体，使用 JSON-RPC 2.0 风格的消息传递。

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
// GET  /.well-known/agent.json → server.get_agent_card()
// POST /                       → server.handle_a2a_request(body).await
```

**任务持久化**：来自 `tasks/send` 的任务存储在内存中的 `RwLock<HashMap>` 中。`tasks/get` 检索任务，`tasks/cancel` 转换其状态。生产环境中，请使用自己的数据库支持的存储进行包装。

### A2AClient（调用远程智能体）

```rust
use langchainrust::a2a::{A2AClient, A2AMessage};

let client = A2AClient::new("http://remote-agent:8080".to_string());

// 发现智能体
let card = client.get_agent_card().await?;

// 发送任务
let task = client.send_task(A2AMessage::user("hello")).await?;

// 获取任务
let task = client.get_task(&task.id).await?;

// 取消任务
let task = client.cancel_task(&task.id).await?;
```

---

## with_structured_output ✨ v0.4.1

`StructuredOutputExt` trait 让你通过一次调用从 LLM 获取强类型输出。在可用时使用函数调用，否则回退到 JsonOutputParser。

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

---

## FileVectorStore ✨ v0.4.1

基于 JSON 持久化的向量存储。填补了 InMemory（不持久化）与外部数据库（过于重量级）之间的空白。

```rust
use langchainrust::{FileVectorStore, VectorStore, Document, MockEmbeddings};
use std::path::PathBuf;

let path = PathBuf::from("./vectors.json");
let store = FileVectorStore::new(path, 4)?;  // 4 维

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

---

## ComputerUseTool ✨ v0.4.1

与 Anthropic computer use API 对齐的计算机使用工具。提供截图、鼠标点击和键盘输入功能。

```rust
use langchainrust::ComputerUseTool;
use std::sync::Arc;

// Anthropic API 模式（默认）
let tool = ComputerUseTool::new();

// 或 Native 模式（需要 feature computer-use-native）
// let tool = ComputerUseTool::new_native();

// 作为 BaseTool 使用
let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(tool)];
```

---

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
use langchainrust::retrieval::graph_rag::{GraphRAG, GraphQueryMode};

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

### MCP 完整协议（6 个原语）

v0.5.0 完成了 MCP 规范的全部 6 个原语，包括 Client 和 Server：

| 原语 | 用途 | 典型用途 |
|-----------|---------|-------------|
| **Resources** | 浏览/读取服务器资源 | Claude Desktop 读取本地文件 |
| **Prompts** | 获取预定义提示词模板 | 标准化提示词管理 |
| **Completion** | 自动补全建议 | 参数自动补全 |
| **Elicitation** | 向用户的交互式提示 | 需要用户确认 |
| **Roots** | 发现客户端根目录 | 服务器需要知道可访问的路径 |
| **Sampling** | 服务器通过客户端代理 LLM 请求 | 服务器需要 LLM 能力 |

```rust
use langchainrust::mcp::MCPClient;

// 客户端：浏览资源
let resources = client.list_resources().await?;
let content = client.read_resource("file:///data/report.pdf").await?;

// 获取提示词模板
let prompts = client.list_prompts().await?;
let prompt = client.get_prompt("code_review", arguments).await?;

// 补全建议
let completions = client.complete("file:///src/", "main").await?;
```

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

- **LocalSandbox**：子进程执行，超时自动终止，捕获 stdout/stderr，Python 危险导入检查
- **E2B 云沙箱**（feature gate `sandbox-e2b`）：远程微虚拟机，完全隔离
- **WASM 沙箱**（feature gate `sandbox-wasm`）：浏览器级沙箱，零网络访问

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