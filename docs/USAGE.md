# 使用文档

这份文档承载 README 的详细内容；GitHub 首页 README 会保持简短。

## 目录

- [LLM](#llm)
  - 多 Provider 支持
  - 直接调用（纯文本）
  - 流式输出（streaming）
  - Function Calling（bind_tools）
  - OpenAIConfig 配置项
  - 多模态 Vision
  - OpenAI Assistants API
- [Prompts](#prompts)
  - PromptTemplate
  - ChatPromptTemplate
  - FewShotPromptTemplate
  - ExampleSelector
- [Memory](#memory)
  - ConversationBufferMemory
  - ConversationBufferWindowMemory
  - ConversationSummaryMemory
  - ConversationSummaryBufferMemory
  - VectorStoreRetrieverMemory（向量检索）
  - ContextWindow（长上下文管理）✨ v0.4.1
  - Memory 类型对比
- [Chains](#chains)
  - SequentialChain（带 memory 注入）
  - ConversationRetrievalChain（对话检索）
  - Chain 流式输出 ✨ v0.4.1
- [Agent](#agent)
  - FunctionCallingAgent（推荐）
  - ReActAgent（兼容旧模型）
  - 两种 Agent 对比
  - 模型路由（智能选择模型）
  - Tool 调用输出格式约定
- [Plan-Execute Agent](#plan-execute-agent)
- [Handoffs 多 Agent 交接](#handoffs-多-agent-交接)
- [Streaming Tool Calls](#streaming-tool-calls)
- [Guardrails 安全护栏](#guardrails-安全护栏)
- [Token 计数器与成本估算](#token-计数器与成本估算)
- [Sessions 会话管理](#sessions-会话管理)
- [MCP 协议](#mcp-协议)
  - MCPServer（暴露本地工具）
- [Tools](#tools)
  - 内置工具
  - WikipediaTool
  - DuckDuckGoSearchTool
  - PythonREPLTool
  - 自定义 Tool
  - to_tool_definition()
  - 扩展工具（HTTPTool / FileTool / SQLTool）
- [Embeddings](#embeddings)
  - OpenAI Embeddings
  - DeepSeek Embeddings
  - Qwen Embeddings
  - LocalEmbeddings（离线）
- [RAG](#rag)
  - 完整 RAG 流程
  - 文档分割配置
  - SemanticSplitter（语义分块）
- [BM25](#bm25)
  - BM25Retriever 基础用法
  - BM25 参数配置
  - ChunkedBM25Retriever（Parent-Child 结构）
- [Hybrid（混合检索）](#hybrid混合检索)
  - RRF 融合算法
  - HybridRetriever 使用
  - UnifiedHybridIndex（统一混合索引）
  - 三种检索模式架构对比
- [Document Loaders](#document-loaders)
  - TextLoader
  - JSONLoader
  - MarkdownLoader
  - HTMLLoader
  - DocxLoader ✨ v0.4.1
  - WebScraperLoader ✨ v0.4.1
  - SitemapLoader ✨ v0.4.1
- [MultiQueryRetriever](#multiqueryretriever)
  - 多查询检索原理
  - StaticQueryGenerator
- [HyDE Retriever](#hyde-retriever)
  - 假设文档生成
  - 工作流程
- [Reranking](#reranking)
  - KeywordReranker
  - BM25Reranker
  - RerankingExecutor
- [Output Parsers](#output-parsers)
  - StrOutputParser
  - CommaSeparatedListOutputParser
  - JsonOutputParser
  - StructuredOutputParser
  - TypedOutputParser
- [Document Chains](#document-chains)
  - StuffDocumentsChain
  - RefineDocumentsChain
  - MapReduceDocumentsChain
  - MapRerankDocumentsChain
- [Vector Stores](#vector-stores)
  - ChromaDB
  - PGVectorStore
  - PineconeStore
- [LLM Cache](#llm-cache)
  - 内存缓存 + TTL
  - CacheConfig 配置
- [存储后端](#存储后端)
  - RedisDocumentStore
  - SQLiteDocumentStore
- [Callbacks](#callbacks)
  - CallbackManager
  - StdOutHandler
  - LangSmith 追踪
  - OtelHandler（OpenTelemetry）
- [Evaluation 评估](#evaluation-评估)
  - 评测器（10 种）
  - EvalRunner（批量评测）
- [配置与安全](#配置与安全)
- [LangGraph](#langgraph)
  - StateGraph 基础用法
  - 条件边路由
  - Human-in-the-loop
  - Subgraph 子图嵌套
  - Parallel 并行执行
  - Checkpointer 持久化
  - 可视化输出
- [A2A Agent 协议](#a2a-agent-协议) ✨ v0.4.1
  - A2AServer（暴露 agent）
  - A2AClient（调用远程 agent）
- [with_structured_output](#with_structured_output) ✨ v0.4.1
- [FileVectorStore](#filevectorstore) ✨ v0.4.1
- [ComputerUseTool](#computerusetool) ✨ v0.4.1
- [测试](#测试)
- [模块结构](#模块结构)
- [最佳实践](#最佳实践)

---

## LLM

### 多 Provider 支持

LangChainRust 支持多种 LLM 提供商，使用统一的 API 接口：

| Provider | 类名 | 特点 |
|----------|------|------|
| **OpenAI** | `OpenAIChat` | GPT-4、GPT-3.5-turbo |
| **DeepSeek** | `DeepSeekChat` | DeepSeek-V3、性价比高 |
| **Moonshot** | `MoonshotChat` | Kimi、长上下文 |
| **Qwen** | `QwenChat` | 通义千问、阿里云 |
| **Zhipu** | `ZhipuChat` | ChatGLM、智谱 |
| **Anthropic** | `AnthropicChat` | Claude、安全性强 |
| **Ollama** | `OllamaChat` | 本地部署、开源模型 |
| **Gemini** | `GeminiChat` | Google Gemini、多模态 |

#### DeepSeek（推荐性价比）

```rust
use langchainrust::{DeepSeekChat, DeepSeekConfig, BaseChatModel};
use langchainrust::schema::Message;

// 从环境变量创建
let llm = DeepSeekChat::from_env();

// 或手动配置
let config = DeepSeekConfig {
    api_key: "sk-xxx".to_string(),
    base_url: "https://api.deepseek.com/v1".to_string(),
    model: "deepseek-chat".to_string(),  // 或 "deepseek-reasoner"
    ..Default::default()
};
let llm = DeepSeekChat::new(config);

let response = llm.chat(vec![
    Message::human("解释 Rust 的所有权机制"),
], None).await?;
```

#### Moonshot（长上下文）

```rust
use langchainrust::{MoonshotChat, MoonshotConfig};

let llm = MoonshotChat::with_model("moonshot-v1-128k");  // 128K 上下文

let response = llm.chat(vec![
    Message::human("分析这篇长文档..."),
], None).await?;
```

#### Qwen（通义千问）

```rust
use langchainrust::{QwenChat, QwenConfig};

let llm = QwenChat::from_env();  // 或 QwenChat::with_model("qwen-plus")

let response = llm.chat(vec![
    Message::human("用中文解释微服务架构"),
], None).await?;
```

#### Zhipu（ChatGLM）

```rust
use langchainrust::{ZhipuChat, ZhipuConfig};

let llm = ZhipuChat::from_env();  // 或 ZhipuChat::with_model("glm-4")

let response = llm.chat(vec![
    Message::human("写一段 Rust 并发代码"),
], None).await?;
```

#### Anthropic Claude

```rust
use langchainrust::{AnthropicChat, AnthropicConfig};

let config = AnthropicConfig {
    api_key: std::env::var("ANTHROPIC_API_KEY")?,
    model: "claude-3-opus-20240229".to_string(),  // 或 claude-3-sonnet
    ..Default::default()
};
let llm = AnthropicChat::new(config);

let response = llm.chat(vec![
    Message::human("安全地分析这段代码"),
], None).await?;
```

#### Ollama（本地部署）

```rust
use langchainrust::{OllamaChat, OllamaConfig};

// 基础用法
let llm = OllamaChat::new("llama3.2");  // 默认 localhost:11434

// 自定义配置
let config = OllamaConfig {
    base_url: "http://localhost:11434".to_string(),
    model: "llama3.2".to_string(),
    ..Default::default()
};
let llm = OllamaChat::with_config(config);

// 多模态支持（vision）
let llm = OllamaChat::new("llava");
let response = llm.chat_with_image(
    vec![Message::human("描述这张图片")],
    "path/to/image.png",
).await?;
```

#### Gemini（Google）

```rust
use langchainrust::{GeminiChat, GeminiConfig, BaseChatModel};
use langchainrust::schema::Message;

// 从环境变量创建（读取 GEMINI_API_KEY 或 GOOGLE_API_KEY）
let llm = GeminiChat::from_env();

// 或手动配置
let config = GeminiConfig {
    api_key: std::env::var("GEMINI_API_KEY")?,
    model: "gemini-1.5-flash".to_string(),  // 或 gemini-1.5-pro
    ..Default::default()
};
let llm = GeminiChat::new(config);

let response = llm.chat(vec![
    Message::human("介绍 Google Gemini 模型的特点"),
], None).await?;
```

#### Gemini 配置项

| 字段 | 类型 | 说明 |
|------|------|------|
| `api_key` | String | API 密钥 |
| `base_url` | String | API 基础 URL |
| `model` | String | 模型名称 |
| `temperature` | Option<f32> | 采样温度 (0.0-2.0) |
| `max_output_tokens` | Option<usize> | 最大生成 token 数 |
| `top_p` | Option<f32> | Top-p 采样 |
| `top_k` | Option<i32> | Top-k 采样 |

### 直接调用 LLM（纯文本）

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")?,
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
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

### 流式输出（streaming）

流式输出让用户实时看到生成过程，感知延迟更低：

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")?,
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: true,
        ..Default::default()
    };
    
    let llm = OpenAIChat::new(config);
    
    let messages = vec![Message::human("写一段 100 字的短文")];
    
    // 获取流式输出
    let mut stream = llm.stream_chat(messages, None).await?;
    
    let mut full_response = String::new();
    
    // 逐 token 接收并打印
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(token) => {
                print!("{}", token);  // 实时打印（打字机效果）
                full_response.push_str(&token);
            }
            Err(e) => {
                println!("流式错误: {}", e);
                break;
            }
        }
    }
    
    println!("\n完整响应: {}", full_response);
    
    Ok(())
}
```

#### 流式 vs 非流式对比

| 方式 | 用户感知延迟 | 适用场景 |
|------|--------------|----------|
| `chat()` | 等待完整响应（可能5-10秒） | 批量处理、后台任务 |
| `stream_chat()` | 首token立即显示（约0-500ms） | 实时交互、聊天界面 |

#### 流式部分收集

流式输出可以中途停止：

```rust
let mut stream = llm.stream_chat(messages, None).await?;

let mut partial = String::new();
let mut count = 0;

while let Some(chunk) = stream.next().await {
    if let Ok(token) = chunk {
        count += 1;
        partial.push_str(&token);
        
        if count >= 10 {
            break;  // 只收集前 10 个 token
        }
    }
}

println!("部分收集: {}", partial);
```

### Function Calling（bind_tools）

通过 `bind_tools()` 将工具绑定到 LLM，让模型能够调用外部函数：

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, ToolDefinition, BaseChatModel};
use langchainrust::schema::Message;
use schemars::JsonSchema;
use serde::Deserialize;

// 定义工具输入类型
#[derive(JsonSchema, Deserialize)]
struct CalculatorInput {
    expression: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")?,
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        ..Default::default()
    };
    
    let llm = OpenAIChat::new(config);
    
    // 创建工具定义（自动生成 JSON Schema）
    let tool = ToolDefinition::from_type::<CalculatorInput>(
        "calculator",
        "计算数学表达式"
    );
    
    // 绑定工具到 LLM
    let llm_with_tools = llm.bind_tools(vec![tool]);
    
    let messages = vec![Message::human("计算 25 + 17")];
    let response = llm_with_tools.chat(messages, None).await?;
    
    // 检查是否有工具调用
    if let Some(tool_calls) = response.tool_calls {
        let call = &tool_calls[0];
        println!("工具: {}", call.function.name);
        println!("参数: {}", call.function.arguments);
        
        // 解析参数
        let args: CalculatorInput = call.parse_arguments()?;
        println!("表达式: {}", args.expression);
    }
    
    Ok(())
}
```

### OpenAIConfig 配置项

| 字段 | 类型 | 说明 |
|------|------|------|
| `api_key` | String | API 密钥 |
| `base_url` | String | API 基础 URL |
| `model` | String | 模型名称 |
| `streaming` | bool | 是否启用流式输出 |
| `temperature` | Option<f32> | 采样温度 (0.0-2.0) |
| `max_tokens` | Option<usize> | 最大生成 token 数 |
| `tools` | Option<Vec<ToolDefinition>> | 绑定的工具定义 |

### 多模态 Vision

`ImageContent` 表示图片（URL 或 base64 data URI），通过 `Message::human_with_image` 构造带图消息；OpenAIChat 与 OllamaChat 均会自动序列化为对应的多模态格式。

```rust
use langchainrust::schema::{ImageContent, Message};
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};

let msg = Message::human_with_image("描述这张图", "https://example.com/cat.jpg");
// 或多图：
// let msg = Message::human_with_images("对比这两张", vec![
//     ImageContent::from_url("https://example.com/a.jpg"),
//     ImageContent::from_base64_with_mime(base64_str, "image/png"),
// ]);

let llm = OpenAIChat::new(OpenAIConfig::default());
let resp = llm.chat(vec![msg], None).await?;
println!("{}", resp.content);
```

`ImageContent::from_url(url)` / `from_base64(data)` / `from_base64_with_mime(data, mime)`；也可用 `Message::human(text).with_image(ImageContent)` 链式追加。Ollama 用 `OllamaChat` 同理。

---

### OpenAI Assistants API

`OpenAIAssistant` 封装 OpenAI 官方 Assistants API（Assistants / Threads / Run），会话状态托管在服务端，适合多轮复杂任务。需 OpenAI 官方端点，部分 compatible-mode 端点可能不支持。

```rust
use langchainrust::{OpenAIAssistant, OpenAIConfig};

let config = OpenAIConfig::default();
// 创建 Assistant（模型 + 系统指令）
let assistant = OpenAIAssistant::create(&config, "gpt-4o", "你是一个翻译助手").await?;
// 或复用已有：OpenAIAssistant::from_id(config, "asst_xxx")

// run_once：创建线程 + 发消息 + 运行到终态，一步到位
let answer = assistant.run_once("请翻译：Hello").await?;
```

**限制**：Run 遇到工具调用（`requires_action`）当前未实现调度，返回 `AssistantError::RequiresAction`；需要工具调用请用 `FunctionCallingAgent`。

> ✨ **v0.4.1 更新**：Assistants API 已完整支持 `requires_action` 工具调度。遇到 `requires_action` 状态时，自动解析 `tool_calls` → 经 `ToolRegistry` 执行 → `submit_tool_outputs` → 继续轮询至 `completed`。可搭配任意 `BaseTool` 使用。

```rust
use langchainrust::{OpenAIAssistant, OpenAIConfig, ToolRegistry, Calculator, BaseTool};
use std::sync::Arc;

let config = OpenAIConfig::default();
let mut assistant = OpenAIAssistant::create(&config, "gpt-4o", "你是一个计算助手").await?;

// 注册工具，requires_action 时自动调度
let mut registry = ToolRegistry::new();
registry.register(Arc::new(Calculator::new()));
assistant.with_tool_registry(registry);

let answer = assistant.run_once("计算 37 + 48").await?;
```

## Prompts

### PromptTemplate（字符串模板）

```rust
use langchainrust::prompts::PromptTemplate;
use std::collections::HashMap;

fn main() {
    let t = PromptTemplate::new("请用{tone}风格解释{topic}。");
    let values = HashMap::from([("tone", "简洁"), ("topic", "Rust的所有权")]);
    let prompt = t.format(&values).unwrap();
    assert_eq!(prompt, "请用简洁风格解释Rust的所有权。");
}
```

### ChatPromptTemplate（多消息模板）

```rust
use langchainrust::messages::Message;
use langchainrust::prompts::ChatPromptTemplate;
use std::collections::HashMap;

fn main() {
    let tpl = ChatPromptTemplate::new(vec![
        Message::system("你是一个{role}助手。"),
        Message::human("你好，{name}！"),
    ]);

    let values = HashMap::from([("role", "编程"), ("name", "Alice")]);
    let messages = tpl.format(&values).unwrap();
    assert_eq!(messages.len(), 2);
}
```

### FewShotPromptTemplate（少样本提示模板）

嵌入示例到 Prompt 中，帮助 LLM 理解期望的输出格式：

```rust
use langchainrust::prompts::{FewShotPromptTemplate, PromptTemplate};
use std::collections::HashMap;

let examples = vec![
    HashMap::from([("input", "开心"), ("output", "😊 开心")]),
    HashMap::from([("input", "悲伤"), ("output", "😢 悲伤")]),
];

let example_prompt = PromptTemplate::new("输入: {input}\n输出: {output}");

let prompt = FewShotPromptTemplate::new(examples, example_prompt)
    .with_prefix("请将以下文本翻译为表情符号：")
    .with_suffix("输入: {input}\n输出:");

let values = HashMap::from([("input", "兴奋")]);
let result = prompt.format(&values).unwrap();
// 输出包含示例 + 输入
```

### ExampleSelector（示例选择器）

使用 `LengthBasedExampleSelector` 根据输入长度动态选择适量示例：

```rust
use langchainrust::prompts::{LengthBasedExampleSelector, FewShotPromptTemplate, PromptTemplate};
use std::collections::HashMap;

let examples = vec![
    HashMap::from([("input", "开心"), ("output", "😊")]),
    HashMap::from([("input", "悲伤"), ("output", "😢")]),
    HashMap::from([("input", "愤怒"), ("output", "😡")]),
];

let selector = LengthBasedExampleSelector::new(examples)
    .with_max_length(100);  // 最大文本长度

let example_prompt = PromptTemplate::new("{input} -> {output}");

let prompt = FewShotPromptTemplate::with_selector(selector, example_prompt)
    .with_prefix("表情符号翻译：")
    .with_suffix("输入: {input}\n输出:");

let values = HashMap::from([("input", "惊讶")]);
let result = prompt.format(&values).unwrap();
```

---

## Memory

LangChainRust 提供四种 Memory 类型，解决对话历史管理问题：

| Memory 类型 | 压缩方式 | Token 管理 | 适用场景 |
|-------------|----------|------------|----------|
| **BufferMemory** | 无压缩 | 无限制 | 短对话、需要完整历史 |
| **WindowMemory** | 窗口截断 | 硬性截断 | 简单控制、接受丢失 |
| **SummaryMemory** | LLM 摘要 | 智能压缩 | 长对话、节省 token |
| **SummaryBufferMemory** | 混合策略 | 动态压缩 | 平衡方案（推荐） |

### ConversationBufferMemory

保存全部对话历史，无压缩：

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

**特点**：所有对话都保存，token 会随对话增长无限增加。

### ConversationBufferWindowMemory

只保留最近 k 轮对话：

```rust
use langchainrust::ConversationBufferWindowMemory;

// k=2，保留最近 2 轮（4 条消息）
let mut memory = ConversationBufferWindowMemory::new(2);

// 添加多轮对话
for i in 1..=5 {
    let inputs = HashMap::from([("input".to_string(), format!("问题{}", i))]);
    let outputs = HashMap::from([("output".to_string(), format!("答案{}", i))]);
    memory.save_context(&inputs, &outputs).await?;
}

let vars = memory.load_memory_variables(&HashMap::new()).await?;
// 只返回最近 2 轮：问题4、问题5
// 问题1、2、3 被丢弃
```

**特点**：简单控制 token 数量，但早期对话会丢失。

### ConversationSummaryMemory

使用 LLM 自动摘要旧对话：

```rust
use langchainrust::{ConversationSummaryMemory, OpenAIChat};

let llm = OpenAIChat::new(config);
let mut memory = ConversationSummaryMemory::new(llm);

// 添加多轮对话，超过限制时自动摘要
for i in 1..=10 {
    memory.save_context(&inputs, &outputs).await?;
}

let buffer = memory.buffer().await;
// buffer = "用户张三喜欢编程，讨论了 Rust 语言..."
// 原始对话被压缩成摘要
```

**特点**：大幅压缩 token，保留关键信息。需要额外 LLM 调用。

### ConversationSummaryBufferMemory（推荐）

摘要 + 保留最近对话的混合策略：

```rust
use langchainrust::{ConversationSummaryBufferMemory, OpenAIChat};

let llm = OpenAIChat::new(config);

// max_token_limit = 100，超过时触发压缩
let mut memory = ConversationSummaryBufferMemory::new(llm, 100);

// 添加对话
for i in 1..=10 {
    let inputs = HashMap::from([("input".to_string(), format!("问题{}", i))]);
    let outputs = HashMap::from([("output".to_string(), format!("答案{}", i))]);
    memory.save_context(&inputs, &outputs).await?;
}

let vars = memory.load_memory_variables(&HashMap::new()).await?;
let history = vars.get("history").unwrap().as_str().unwrap();
// 输出: "摘要: 用户讨论了问题1-7...\n\nHuman: 问题8\nAI: 答案8\nHuman: 问题9\nAI: 答案9..."
```

**工作原理**：

```
┌─────────────────────────────────────────────────────────────┐
│  max_token_limit = 100                                       │
│                                                             │
│  total_tokens > 100 时触发压缩:                              │
│                                                             │
│  1. prune_messages() 从后往前保留到 limit                    │
│  2. 被裁掉的消息 → LLM 生成摘要 → buffer                      │
│  3. 清空 chat_memory，只保留 pruned 部分                      │
│                                                             │
│  最终:                                                       │
│  - buffer = "摘要内容"                                       │
│  - chat_memory = [最近消息]                                  │
│                                                             │
│  load_memory_variables:                                      │
│  - 返回 "摘要: xxx\n\nHuman: 最近消息\nAI: 回复"              │
└─────────────────────────────────────────────────────────────┘
```

### Memory 类型对比

| 维度 | BufferMemory | WindowMemory | SummaryMemory | SummaryBufferMemory |
|------|--------------|--------------|---------------|---------------------|
| **压缩方式** | 无 | 硬删除 | LLM摘要 | 摘要+保留 |
| **丢失信息** | 无 | 完全丢失 | 摘要保留关键 | 平衡 |
| **LLM调用** | 无 | 无 | 每轮摘要 | 触发时摘要 |
| **Token控制** | 无限 | 固定k轮 | 动态压缩 | 动态+保留最近 |

### return_messages 模式

控制输出格式：

```rust
// 默认：返回字符串
let memory = ConversationBufferMemory::new();
// history = "Human: 问题\nAI: 回答"

// 返回消息数组
let memory = ConversationBufferMemory::new()
    .with_return_messages(true);
// history = [{"type": "human", "content": "问题"}, {"type": "ai", "content": "回答"}]
```

### 自定义键名

```rust
let memory = ConversationBufferMemory::new()
    .with_input_key("question")    // 输入键
    .with_output_key("answer")     // 输出键
    .with_memory_key("context");   // 记忆键

let vars = memory.load_memory_variables(&HashMap::new()).await?;
let history = vars.get("context").unwrap();  // 使用自定义键
```

### ChatMessageHistory

底层消息存储容器：

```rust
use langchainrust::ChatMessageHistory;

let mut history = ChatMessageHistory::new();

history.add_user_message("你好");
history.add_ai_message("你好！有什么可以帮助你的？");

println!("消息数: {}", history.len());
println!("格式化: {}", history.to_string());
// 输出: "Human: 你好\nAI: 你好！有什么可以帮助你的？"
```

---

### VectorStoreRetrieverMemory（向量检索）

每轮对话嵌入后存入向量库，加载时按当前输入的语义相关性召回 top-k 历史。相比固定窗口的 buffer memory，长对话 / 跨会话能保留更多有效上下文。

```rust
use langchainrust::{VectorStoreRetrieverMemory, MockEmbeddings, BaseMemory};
use langchainrust::vector_stores::InMemoryVectorStore;
use std::collections::HashMap;

let mut memory = VectorStoreRetrieverMemory::new(
    InMemoryVectorStore::new(),
    MockEmbeddings::new(1536),
    4,  // 召回 4 条相关历史
);

memory.save_context(&inputs, &outputs).await?;
let vars = memory.load_memory_variables(&HashMap::new()).await?;
```

**特点**：语义召回，长对话不丢关键信息；依赖向量库与 embedding 模型，有额外开销。

### ContextWindow（长上下文管理）✨ v0.4.1

`ContextWindow` 管理超长对话的 token 预算，提供两种策略：截断(Truncate)和摘要(Summarize)。

```rust
use langchainrust::{ContextWindow, Message, OpenAIChat, Strategy};
use langchainrust::BaseChatModel;

// 策略1：截断 — 超 token 预算时丢弃最旧消息
let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096);
// 等价于 ContextWindow::new(4096).with_strategy(Strategy::Truncate)

// 策略2：摘要 — 超 token 预算时用 LLM 摘要压缩旧对话
let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096)
    .with_strategy(Strategy::Summarize)
    .with_llm(OpenAIChat::new(config));

// 添加消息，自动管理上下文窗口
cw.add_message(Message::human("你好")).await;
cw.add_message(Message::ai("你好！有什么可以帮你的？")).await;

// 获取窗口内的消息（已截断/已摘要）
let messages = cw.get_messages().await;
```

| 策略 | 行为 | 适用场景 |
|------|------|----------|
| `Truncate` | 超 token 预算直接丢弃最旧消息 | 简单场景、不需要保留旧信息 |
| `Summarize` | 超 token 预算用 LLM 压缩旧对话为摘要 | 长对话、需保留关键信息 |

## Chains

### SequentialChain（带 memory 注入）

与测试用例一致的"链式 + 记忆"例子可参考：[chain_test.rs](../tests/chain_test.rs)

核心思路：
- `SequentialChain` 串联多个 `PromptChain`
- `SimpleMemory` 会把历史写入，下一步将 `chat_history` 作为 system message 注入模板

### ConversationRetrievalChain（对话检索 Chain）

带记忆的检索增强对话 Chain，自动完成：加载对话历史 → 检索相关文档 → 组合上下文 → LLM 生成 → 保存记忆。

```rust
use langchainrust::{
    ConversationRetrievalChain, OpenAIChat, SimilarityRetriever,
    ConversationBufferMemory, BaseChain,
};
use std::sync::Arc;

let llm = OpenAIChat::from_env();
let retriever = Arc::new(SimilarityRetriever::new(store, embeddings));
let memory = ConversationBufferMemory::new()
    .with_return_messages(true);

let chain = ConversationRetrievalChain::new(llm, retriever, memory)
    .with_k(4)                          // 检索 4 个相关文档
    .with_verbose(true)                 // 开启执行日志
    .with_return_source_documents(true); // 返回来源文档

let result = chain.query("什么是 Rust 的所有权？").await?;
println!("回答: {}", result);
```

#### 配置方法

| 方法 | 说明 |
|------|------|
| `with_k(usize)` | 检索文档数量（默认 4） |
| `with_system_prompt(str)` | 自定义系统提示词 |
| `with_qa_prompt(str)` | 自定义 QA 模板 |
| `with_input_key(str)` | 输入键名（默认 query） |
| `with_output_key(str)` | 输出键名（默认 result） |
| `with_memory_key(str)` | 记忆键名（默认 history） |
| `with_return_source_documents(bool)` | 是否返回来源文档 |
| `with_verbose(bool)` | 是否打印执行日志 |

### Chain 流式输出 ✨ v0.4.1

`BaseChain::stream()` 提供逐 token 流式输出，`LLMChain` 和 `ConversationChain` 均已覆写。

```rust
use langchainrust::{LLMChain, BaseChain};
use futures_util::StreamExt;

let chain = LLMChain::new(llm, "你是一个有帮助的助手");
let mut stream = chain.stream(inputs).await?;

while let Some(token) = stream.next().await {
    match token {
        Ok(t) => print!("{}", t),  // 逐 token 输出（打字机效果）
        Err(e) => eprintln!("流式错误: {}", e),
    }
}
```

**特点**：与 `Runnable::stream()` 不同，Chain 流式会触发 `on_llm_new_token` 回调，可在 CallbackHandler 中监听。

---

## Agent

LangChainRust 提供两种 Agent：

| Agent | 工具调用方式 | 适用场景 |
|-------|-------------|----------|
| **FunctionCallingAgent** | 原生 Function Calling | 支持 FC 的模型（GPT-4、Claude、Gemini）**推荐** |
| **ReActAgent** | 文本解析（正则提取） | 不支持 FC 的模型、开源模型、本地部署模型 |

### FunctionCallingAgent（推荐）

使用原生 Function Calling，类型安全，更可靠。

#### 创建 FunctionCallingAgent

```rust
use langchainrust::{
    OpenAIChat, OpenAIConfig, BaseChatModel,
    FunctionCallingAgent, AgentExecutor, BaseAgent, BaseTool,
    Calculator, DateTimeTool,
};
use std::sync::Arc;

let config = OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY")?,
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-3.5-turbo".to_string(),
    ..Default::default()
};

let llm = OpenAIChat::new(config);

let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(DateTimeTool::new()),
];

// FunctionCallingAgent 自动绑定工具到 LLM
let agent = FunctionCallingAgent::new(llm, tools.clone(), None);

let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
    .with_max_iterations(5)
    .with_verbose(true);

let result = executor.invoke("计算 37 + 48".to_string()).await?;
println!("结果: {}", result);
```

#### 带自定义系统提示词

```rust
let agent = FunctionCallingAgent::new(
    llm,
    tools.clone(),
    Some("你是一个数学助手，专门帮助用户解决数学问题。".to_string()),
);
```

#### 执行日志

```
=== 迭代 1 ===
动作: calculator({"expression":"37 + 48"})
观察: 37 + 48 = 85

=== 迭代 2 ===
最终答案: {"output": "37 + 48 = 85"}
```

---

### ReActAgent（兼容旧模型）

使用文本解析方式，适用于不支持 Function Calling 的模型。

#### 创建 ReActAgent

```rust
use langchainrust::{
    OpenAIChat, OpenAIConfig, BaseChatModel,
    ReActAgent, AgentExecutor, BaseAgent, BaseTool,
    Calculator, DateTimeTool, SimpleMathTool,
};
use std::sync::Arc;

let llm = OpenAIChat::new(config);

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

---

### 两种 Agent 对比

| 维度 | FunctionCallingAgent | ReActAgent |
|------|---------------------|------------|
| **工具调用方式** | 原生 Function Calling | 文本解析（正则） |
| **可靠性** | 高（类型安全） | 较低（依赖 Prompt 格式） |
| **Token 消耗** | 低（不需要格式说明） | 高（Prompt 包含格式说明） |
| **参数处理** | JSON Schema 类型安全 | 文本提取，可能格式错误 |
| **历史传递** | Message 消息流 | scratchpad 文本 |
| **适用模型** | 支持 FC 的模型 | 所有模型 |

#### 选择建议

- 使用 OpenAI GPT-4/Claude/Gemini → **FunctionCallingAgent**
- 使用本地部署模型/开源模型 → **ReActAgent**

---

### ReActAgent（基础用法）

`ReActAgent` 是一个可以根据工具描述自动决定调用哪个工具的 Agent。

#### 创建 Agent

```rust
use langchainrust::agent::{AgentExecutor, ReActAgent};
use langchainrust::llms::{LLM, OpenAIConfig};
use langchainrust::memory::SimpleMemory;
use langchainrust::tools::{Calculator, Tool};
use std::sync::Arc;

let llm = LLM::new(OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY").unwrap(),
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-3.5-turbo".to_string(),
    streaming: false,
    factor: 3,
});

let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Calculator)];

// 方式1：基础创建
let agent = ReActAgent::new(llm, tools.clone(), None);

// 方式2：带 Memory
let agent = ReActAgent::new(llm, tools.clone(), Some(Box::new(SimpleMemory::default())));

// 方式3：带模板
use langchainrust::messages::Message;
use langchainrust::prompts::ChatPromptTemplate;

let template = ChatPromptTemplate::new(vec![
    Message::system("你是数学家{name}，回答风格是{style}。"),
    Message::human("请计算：{input}"),
]);

let agent = ReActAgent::with_template(
    llm,
    tools.clone(),
    Some(Box::new(SimpleMemory::default())),
    template,
);
```

#### 执行 Agent

```rust
let executor = AgentExecutor::new(Box::new(agent), tools).with_max_iterations(5);

// 基础调用
let result = executor.run("37 加 48 等于多少？").await.unwrap();

// 带变量调用
use std::collections::HashMap;
let vars = HashMap::from([
    ("name".to_string(), "小李".to_string()),
    ("style".to_string(), "简洁".to_string()),
]);
let result = executor.run_with_vars("1+3", vars).await.unwrap();
```

---

### 模型路由（智能选择模型）

`ReActAgent` 支持模型路由功能：根据问题难度自动选择最合适的模型。

#### 核心概念

- **系数（Factor）**：1-10 的等级，越贵/越强的模型系数越高
  - `gpt-3.5-turbo`: 1-3（便宜、快速）
  - `gpt-4`: 7-9（强大、昂贵）
  - `gpt-4-turbo`: 9-10（最强）
- **难度（Difficulty）**：传入的参数（1-10），未传入时默认为 1
- **路由逻辑**：选择系数 ≥ 难度 的最小系数模型（性价比最高）

#### 使用模型路由

```rust
use langchainrust::agent::{AgentExecutor, ReActAgent};
use langchainrust::llms::{LLM, OpenAIConfig, ModelConfig};
use langchainrust::tools::{Calculator, Tool};
use std::collections::HashMap;
use std::sync::Arc;

// 路由 LLM（用于决策选择哪个模型）
let router_llm = LLM::new(OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY").unwrap(),
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-3.5-turbo".to_string(),
    streaming: false,
    factor: 3,
});

// 候选模型列表
let models = vec![
    ModelConfig::OpenAI(OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY").unwrap(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        factor: 1,  // 便宜
    }),
    ModelConfig::OpenAI(OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY").unwrap(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-4".to_string(),
        streaming: false,
        factor: 8,  // 强大
    }),
];

let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Calculator)];

// 使用 with_models 创建带模型路由的 Agent
let agent = ReActAgent::with_models(
    router_llm,
    models,
    tools.clone(),
    None,           // memory
    None,           // template（可选）
);

let executor = AgentExecutor::new(Box::new(agent), tools).with_max_iterations(3);

// 传入难度参数
let vars = HashMap::from([("difficulty".to_string(), "8".to_string())]);
let result = executor
    .run_with_vars("用Python写一个快速排序算法", vars)
    .await
    .unwrap();
```

#### 混合多厂商模型

```rust
use langchainrust::llms::{ModelConfig, QwenConfig};

let models = vec![
    // OpenAI 模型
    ModelConfig::OpenAI(OpenAIConfig {
        api_key: "sk-xxx".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        factor: 2,
    }),
    ModelConfig::OpenAI(OpenAIConfig {
        api_key: "sk-xxx".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-4".to_string(),
        streaming: false,
        factor: 8,
    }),
    // Qwen 模型
    ModelConfig::Qwen(QwenConfig {
        api_key: "xxx".to_string(),
        base_url: "https://dashscope.aliyuncs.com/api/v1".to_string(),
        model: "qwen-plus".to_string(),
        factor: 5,
    }),
];
```

#### 路由选择逻辑

1. 路由 LLM 分析问题和候选模型列表
2. 根据 `difficulty` 参数（默认 1）选择匹配的模型
3. 优先选择系数 ≥ difficulty 的最小系数模型
4. 如果没有匹配的，选择系数最大的模型

#### 难度配置建议

| 使用场景 | difficulty 值 | 建议模型 |
|---------|---------------|----------|
| 简单问答、计算 | 1-3 | gpt-3.5-turbo |
| 代码生成 | 4-6 | gpt-4 |
| 复杂推理 | 7-9 | gpt-4-turbo |
| 多步规划 | 10 | 最强模型 |

---

### Tool 调用机制

`ReActAgent` 的工具调用是**可选的**：

1. **智能判断**：Agent 会根据问题内容自主判断是否需要使用工具
2. **工具调用格式**：如果需要使用工具，模型会输出 `[TOOL: 工具名 参数名=参数值]`
3. **直接回答**：如果不需要工具，模型会直接给出答案

```
# 使用工具的示例输出
[TOOL: calculator expression=37+48]

# 不使用工具的示例输出
Rust 是一种系统编程语言，具有内存安全、零成本抽象等特点...
```

### 判断是否使用了工具

使用 `run_with_details()` 方法获取执行详情：

```rust
use langchainrust::agent::{AgentExecutor, ReActAgent, ExecutionResult};

let executor = AgentExecutor::new(Box::new(agent), tools);

// 获取执行详情
let result: ExecutionResult = executor.run_with_details("计算 37+48").await?;

println!("最终答案: {}", result.answer);
println!("是否使用工具: {}", result.used_tools);
println!("调用的工具: {:?}", result.tool_calls);
println!("迭代次数: {}", result.iterations);

if result.used_tools {
    println!("✓ Agent 使用了工具来帮助回答");
} else {
    println!("✓ Agent 直接回答了问题");
}
```

### 执行日志

执行时会打印工具调用日志：

```
[工具调用] calculator {"expression": "37+48"}
[工具结果] 85
```

---

## Embeddings

**Embeddings** 将文本转换为向量，用于语义检索、相似度计算等场景。

### 支持的 Embeddings

| Provider | 类名 | 向量维度 | 特点 |
|----------|------|----------|------|
| **OpenAI** | `OpenAIEmbeddings` | 1536 | 高质量、稳定 |
| **DeepSeek** | `DeepSeekEmbeddings` | 1536 | 性价比高 |
| **Qwen** | `QwenEmbeddings` | 1536 | 阿里云、中文优化 |
| **Mock** | `MockEmbeddings` | 自定义 | 测试用 |

### OpenAI Embeddings

```rust
use langchainrust::{OpenAIEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(OpenAIEmbeddings::new(
    std::env::var("OPENAI_API_KEY")?
));

// 单文本嵌入
let vector = embeddings.embed("Rust 是一门系统编程语言").await?;
println!("向量维度: {}", vector.len());  // 1536

// 批量嵌入
let texts = vec![
    "Rust 是一门系统编程语言",
    "Python 是一门脚本语言",
];
let vectors = embeddings.embed_batch(texts).await?;
println!("批量向量数: {}", vectors.len());
```

### DeepSeek Embeddings

```rust
use langchainrust::{DeepSeekEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(DeepSeekEmbeddings::from_env());

let vector = embeddings.embed("深度学习的核心原理").await?;
```

### Qwen Embeddings

```rust
use langchainrust::{QwenEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(QwenEmbeddings::from_env());

let vector = embeddings.embed("通义千问的向量生成").await?;
```

### Mock Embeddings（测试用）

```rust
use langchainrust::{MockEmbeddings, Embeddings};
use std::sync::Arc;

// 指定向量维度
let embeddings = Arc::new(MockEmbeddings::new(128));

let vector = embeddings.embed("测试文本").await?;
println!("向量维度: {}", vector.len());  // 128
```

### LocalEmbeddings（离线）

纯 Rust 实现的轻量本地嵌入（词频 hash + L2 归一），不调用任何 API，适合离线、隐私、零成本场景的粗粒度检索。

```rust
use langchainrust::LocalEmbeddings;

// 指定维度，或用默认 256 维
let emb = LocalEmbeddings::default_dim();
let vec = emb.embed_query("你好世界").await?;
```

**限制**：词袋 hash，语义质量有限（无法识别同义 / 上下文）；需高质量语义嵌入请用 `OpenAIEmbeddings` 等，或等待 BGE / E5 集成。

### Embeddings 配置项

| 字段 | 说明 |
|------|------|
| `api_key` | API 密钥 |
| `base_url` | API 地址（支持代理） |
| `model` | 模型名称 |

---

## RAG

**RAG (Retrieval-Augmented Generation)** 先从知识库检索相关文档，再让 LLM 基于检索内容生成答案。

### 核心价值

让 LLM 访问私有知识库、最新数据，无需重新训练模型。

### 功能特性

| 特性 | 说明 |
|------|------|
| **文档加载** | 支持 PDF、CSV 等格式 |
| **智能分割** | RecursiveCharacterSplitter 递归分割 |
| **向量嵌入** | OpenAI Embeddings |
| **语义检索** | 基于向量相似度 |

### 完整 RAG 流程

```rust
use langchainrust::{
    Document, InMemoryVectorStore, MockEmbeddings,
    SimilarityRetriever, RetrieverTrait, RecursiveCharacterSplitter,
    TextSplitter, RetrievalQA, BaseChain,
};
use std::sync::Arc;

// 1. 准备文档
let docs = vec![
    Document::new("Rust 是一门系统编程语言..."),
    Document::new("Python 是一门脚本语言..."),
];

// 2. 分割文档
let splitter = RecursiveCharacterSplitter::new(200, 50);
let chunks = splitter.split_documents(&docs)?;

// 3. 创建检索器
let store = Arc::new(InMemoryVectorStore::new());
let embeddings = Arc::new(MockEmbeddings::new(128));
let retriever = SimilarityRetriever::new(store, embeddings);
retriever.add_documents(chunks).await?;

// 4. 创建 RetrievalQA
let qa_chain = RetrievalQA::new(llm, retriever, 3);

// 5. 执行问答
let result = qa_chain.invoke(HashMap::from([
    ("query".to_string(), Value::String("什么是 Rust？".to_string()))
])).await?;
```

### 文档分割配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `chunk_size` | 200-500 | 每个 chunk 的最大字符数 |
| `overlap` | 50-100 | 相邻 chunk 的重叠字符数 |

### SemanticSplitter（语义分块）

按语义相关性切分：先分句并嵌入，在相邻句向量相似度骤降处断块，相比字符级分割能更好保留语义完整性，提升检索质量。中英文分句（`。!?;` / `.!?\n`）。

```rust
use langchainrust::SemanticSplitter;
use langchainrust::OpenAIEmbeddings;

let splitter = SemanticSplitter::with_defaults(OpenAIEmbeddings::new(config));
// 或自定义：SemanticSplitter::new(emb, 0.5, 1000)  // 阈值 0.5，单块最大 1000 字符

let chunks = splitter.split_text(long_text).await;  // Vec<String>
```

**注意**：嵌入是异步操作，而 `TextSplitter` trait 是同步签名；为不破坏现有 trait，本分块器提供独立的异步接口 `split_text` / `split_document`，不实现同步 `TextSplitter`。

---

## BM25

**BM25** 是一种经典的 TF-IDF 加权检索算法，通过词频（TF）和逆文档频率（IDF）计算文档与查询的相关性评分。

### 核心价值

不依赖 Embedding，适合关键词精确匹配、长文档检索、专业术语搜索。

### 功能特性

| 特性 | 说明 |
|------|------|
| **关键词匹配** | 精确匹配查询关键词，无需语义理解 |
| **中英文分词** | 英文空格分割 + 中文单字/双字组合 |
| **停用词过滤** | 自动过滤常见停用词（中英文） |
| **参数可调** | k1 词频饱和参数、b 文档长度归一化 |

### BM25Retriever 基础用法

```rust
use langchainrust::{BM25Retriever, Document};

// 创建 BM25 检索器
let mut retriever = BM25Retriever::new();

// 添加文档
retriever.add_documents_sync(vec![
    Document::new("Rust 是一门系统编程语言"),
    Document::new("Python 是脚本语言"),
    Document::new("JavaScript 用于网页开发"),
]);

// 搜索关键词
let results = retriever.search("系统编程", 3);

for result in results {
    println!("文档: {}", result.document.content);
    println!("评分: {}", result.score);
}
```

### BM25 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k1` | 1.5 | 词频饱和参数，控制高频词的影响 |
| `b` | 0.75 | 文档长度归一化参数，控制长文档惩罚 |

```rust
// 自定义参数
let mut retriever = BM25Retriever::with_params(2.0, 0.5);
```

### ChunkedBM25Retriever（Parent-Child 结构）

支持 AutoMerging 模式，文档拆分为 Parent（大块）+ Leaf（小块）两层：

```rust
use langchainrust::{ChunkedBM25Retriever, AutoMergingConfig, ChunkedDocumentStore};
use std::sync::Arc;

// 配置 AutoMerging
let config = AutoMergingConfig::new()
    .with_leaf_size(400)     // Leaf chunk 大小
    .with_parent_size(2000)  // Parent chunk 大小
    .with_threshold(0.5);    // 合并阈值（50% Leaf 匹配时合并）

let store = Arc::new(ChunkedDocumentStore::new());
let mut retriever = ChunkedBM25Retriever::with_config(store, config);

// 添加文档（自动分割）
retriever.add_document(Document::new("长文档内容..."));

// 搜索（自动合并）
let results = retriever.search("关键词", 5);

for result in results {
    if result.is_merged() {
        println!("合并结果: {}", result.content());
    } else {
        println!("Leaf chunk: {}", result.content());
    }
}
```

---

## Hybrid（混合检索）

**混合检索** 结合 BM25 的关键词匹配能力和向量检索的语义理解能力，通过 RRF（Reciprocal Rank Fusion）算法融合结果。

### 核心价值

关键词精确匹配 + 语义理解，召回率更高，覆盖更全面。

### 功能特性

| 特性 | 说明 |
|------|------|
| **BM25 检索** | 关键词精确匹配，快速高效 |
| **向量检索** | 语义相似度匹配，理解意图 |
| **RRF 融合** | 排序融合算法，兼顾两种结果 |
| **统一索引** | 一次添加，双索引自动构建 |

### RRF 融合算法

**RRF（Reciprocal Rank Fusion）** 算法公式：

```
RRF_score(d) = Σ 1/(k + rank(d))
```

其中 k 通常为 60，rank(d) 是文档在各自检索结果中的排名。

### HybridRetriever 使用

```rust
use langchainrust::{HybridRetriever, BM25Retriever, Document};

// 创建混合检索器
let retriever = HybridRetriever::with_top_k(10, 10)  // BM25_k=10, Vector_k=10
    .with_rrf_k(60);

// 分别获取 BM25 和向量检索结果
let bm25_docs = bm25_retriever.search("查询词", 10);
let vector_docs = vector_store.search("查询词", 10);

// RRF 融合
let fused = retriever.retrieve(bm25_docs, vector_docs);

for doc in fused {
    println!("文档: {}", doc.document.content);
    println!("RRF 评分: {}", doc.score);
}
```

### UnifiedHybridIndex（统一混合索引）

最高级的混合检索接口，自动管理 BM25 和向量双索引：

```rust
use langchainrust::{UnifiedHybridIndex, HybridIndexConfig, OpenAIEmbeddings};
use std::sync::Arc;

// 配置混合索引
let config = HybridIndexConfig::new()
    .with_chunk_size(500)      // chunk 大小
    .with_top_k(10, 10)        // BM25_k, Vector_k
    .with_rrf_k(60)            // RRF 参数
    .with_merge_threshold(0.5); // AutoMerging 阈值

// 创建混合索引
let embeddings = Arc::new(OpenAIEmbeddings::new(api_key));
let index = UnifiedHybridIndex::with_config(embeddings, 1536, config);

// 添加文档（自动构建双索引）
index.add_document(Document::new("长文档内容...")).await?;

// 混合检索
let results = index.retrieve("查询词", 5).await?;

for result in results {
    println!("文档: {}", result.document.content);
    println!("RRF 评分: {}", result.score);
}
```

### BM25 vs 向量 vs Hybrid 对比

| 维度 | BM25 | 向量检索 | Hybrid |
|------|------|----------|--------|
| **匹配方式** | 关键词精确匹配 | 语义相似度 | 双路融合 |
| **召回率** | 中 | 中 | 高 |
| **精确度** | 高 | 中 | 高 |
| **依赖 Embedding** | 否 | 是 | 是 |
| **专业术语** | 优秀 | 一般 | 优秀 |
| **语义理解** | 差 | 优秀 | 优秀 |
| **推荐场景** | 关键词搜索 | 语义搜索 | 综合搜索 |

**选择建议**：
- 关键词精确匹配 → BM25
- 语义相似度搜索 → 向量检索
- 综合检索、提高召回率 → Hybrid（推荐）

### 三种检索模式架构对比

LangChainRust 提供三种检索模式，核心差异在于**内容存储位置**和**是否回表**：

| 模式 | 内容存储 | 回表 | 适用场景 |
|------|----------|------|----------|
| **SimpleVector** | 内容直接存 InMemoryVectorStore | ❌ 不回表 | 只用向量检索，简单快速 |
| **BM25 Only** | 内容存 ChunkedDocumentStore，BM25 只存词频索引 | ✅ 回表 | 只用关键词检索 |
| **Hybrid** | BM25 + 向量共用 ChunkedDocumentStore | ✅ 回表 | 混合检索，内容共享（推荐） |

#### 架构图解

```
┌─────────────────────────────────────────────────────────────────────┐
│  模式1: SimpleVector（不回表）                                        │
│  ┌─────────────────────┐                                            │
│  │  InMemoryVectorStore │                                            │
│  │  ┌─────────────────┐ │                                            │
│  │  │ Document        │ │  ← 内容直接存这里                           │
│  │  │ + Embedding     │ │                                            │
│  │  └─────────────────┘ │                                            │
│  └─────────────────────┘                                            │
│  搜索: 向量相似度 → 直接返回 Document（无需额外查询）                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  模式2/3: BM25/Hybrid（回表）                                         │
│  ┌─────────────────┐          ┌─────────────────────┐               │
│  │  BM25 Index     │          │  ChunkedDocumentStore│               │
│  │  (不存内容)      │          │  (存实际内容)         │               │
│  ├─────────────────┤          ├─────────────────────┤               │
│  │ chunk_id_list   │          │ parent_docs         │               │
│  │ term_freqs      │──────────│ chunks              │               │
│  │ term_index      │  回表    │                     │               │
│  └─────────────────┘          └─────────────────────┘               │
│  ┌─────────────────┐                                               │
│  │  Vector Index   │──────────回表──────────→ 同上                  │
│  │  (只存 embedding)│                                               │
│  └─────────────────┘                                               │
│  搜索: 索引返回 chunk_id → 回表查询 ChunkedDocumentStore 获取内容    │
└─────────────────────────────────────────────────────────────────────┘
```

#### 回表设计的优势

- **索引体积小**：BM25/向量索引只存统计信息和向量，不存内容
- **内容共享**：BM25 和向量共用同一份内容，避免重复存储
- **支持 AutoMerging**：Parent-Child 结构，匹配时自动合并

---

## Document Loaders

文档加载器用于从不同格式文件加载内容到 Document 对象。

### 支持的格式

| Loader | 格式 | 特性 |
|--------|------|------|
| **TextLoader** | .txt | 支持按行分割 |
| **JSONLoader** | .json | 支持指定 content_key |
| **MarkdownLoader** | .md | 支持按标题级别分割 |
| **PDFLoader** | .pdf | 提取 PDF 文本 |
| **CSVLoader** | .csv | 每行作为一个文档 |

### TextLoader

```rust
use langchainrust::{TextLoader, DocumentLoader};

// 加载整个文件作为一个文档
let loader = TextLoader::new("document.txt");
let docs = loader.load().await?;

// 按行分割
let loader = TextLoader::new_with_line_split("document.txt");
let docs = loader.load().await?;

// 每个文档包含 line_number 元数据
for doc in docs {
    println!("行 {}: {}", doc.metadata.get("line_number"), doc.content);
}
```

### JSONLoader

```rust
use langchainrust::{JSONLoader, DocumentLoader};

// 加载 JSON 数组
let loader = JSONLoader::new("data.json");
let docs = loader.load().await?;

// 指定内容字段
let loader = JSONLoader::new_with_content_key("data.json", "content");
let docs = loader.load().await?;

// 保留原始 JSON
let loader = JSONLoader::new("data.json")
    .with_preserve_raw(true);
let docs = loader.load().await?;
```

### MarkdownLoader

```rust
use langchainrust::{MarkdownLoader, DocumentLoader};

// 加载整个文件
let loader = MarkdownLoader::new("guide.md");
let docs = loader.load().await?;

// 按一级标题分割
let loader = MarkdownLoader::new_with_heading_split("guide.md", 1);
let docs = loader.load().await?;

// 每个文档包含 heading 元数据
for doc in docs {
    println!("标题: {}", doc.metadata.get("heading"));
}
```

### HTMLLoader

去除 `<script>`/`<style>`、剥离标签、解码常见 HTML 实体、压缩空白，从 HTML 字符串或 URL 提取纯文本。

```rust
use langchainrust::retrieval::HTMLLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

// 从 HTML 字符串
let loader = HTMLLoader::new("<p>Hello <b>world</b></p>");
let docs = loader.load().await?; // content: "Hello world"

// 从 URL（异步抓取后解析）
let loader = HTMLLoader::from_url("https://example.com");
let docs = loader.load().await?;

// 纯函数：直接提取文本
let text = HTMLLoader::extract_text("<script>x</script><p>a &amp; b</p>");
// -> "a & b"
```

### DocxLoader ✨ v0.4.1

解析 Word `.docx` 文件：ZIP 解压 + XML 提取 `<w:t>` 文本节点。

```rust
use langchainrust::retrieval::loaders::DocxLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

let loader = DocxLoader::new("document.docx");
let docs = loader.load().await?;
for doc in docs {
    println!("{}", doc.content);
}
```

### WebScraperLoader ✨ v0.4.1

网页爬取：提取页面文本，可递归跟踪同域链接。

```rust
use langchainrust::retrieval::loaders::WebScraperLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

let loader = WebScraperLoader::new("https://example.com")
    .with_max_depth(2)       // 递归深度
    .with_max_pages(10);     // 最大页面数
let docs = loader.load().await?;
```

### SitemapLoader ✨ v0.4.1

解析 `sitemap.xml`，批量爬取页面。

```rust
use langchainrust::retrieval::loaders::SitemapLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

let loader = SitemapLoader::new("https://example.com/sitemap.xml")
    .with_max_pages(50);     // 最大页面数
let docs = loader.load().await?;
```

---

## MultiQueryRetriever

多查询检索器，使用 LLM 生成多个查询变体，提高检索召回率。

### 工作原理

```
用户查询 → LLM 生成 N 个变体 → 分别检索 → 合并去重 → 返回结果

例如：
用户问："数据库超时怎么处理？"

LLM 生成变体：
- "数据库连接失败怎么解决"
- "数据库响应慢的排查方法"
- "MySQL PostgreSQL 连接问题"
```

### 使用方法

```rust
use langchainrust::{MultiQueryRetriever, SimilarityRetriever, InMemoryVectorStore, MockEmbeddings, OpenAIChat};
use std::sync::Arc;

let llm = OpenAIChat::new(config);
let retriever = Arc::new(SimilarityRetriever::new(store, embeddings));

let multi_query = MultiQueryRetriever::new(llm, retriever)
    .with_num_queries(3)        // 生成 3 个变体
    .with_k_per_query(5)        // 每个查询返回 5 个文档
    .with_final_k(10);          // 最终返回 10 个文档

let docs = multi_query.retrieve_multi("数据库超时").await?;
```

### StaticQueryGenerator（不依赖 LLM）

```rust
use langchainrust::StaticQueryGenerator;
use std::collections::HashMap;

let synonyms: HashMap<String, Vec<String>> = HashMap::from([
    ("数据库".to_string(), vec!["DB".to_string(), "存储".to_string()),
]);

let generator = StaticQueryGenerator::new()
    .with_synonym_expansion(synonyms)
    .with_prefix_expansion(vec!["如何".to_string(), "怎么".to_string()]);

let queries = generator.generate("数据库连接失败");
// 生成: ["DB连接失败", "存储连接失败", "如何 数据库连接失败", "怎么 数据库连接失败"]
```

---

## HyDE Retriever

**HyDE (Hypothetical Document Embedding)** 使用 LLM 生成假设文档，然后用假设文档检索真实文档。

### 工作原理

```
用户查询 → LLM 生成假设文档 → 用假设文档检索 → 返回真实文档

例如：
用户问："Rust 怎么处理并发？"

HyDE 生成假设文档：
"Rust 处理并发主要通过 async/await 和 tokio 库..."
（这是一个理想的答案）

然后用这个假设文档去检索，找到真实的类似文档
```

### 使用方法

```rust
use langchainrust::{HyDERetriever, SimilarityRetriever, OpenAIChat, OpenAIEmbeddings};
use std::sync::Arc;

let llm = OpenAIChat::new(config);
let embeddings = Arc::new(OpenAIEmbeddings::new(api_key));
let base_retriever = Arc::new(SimilarityRetriever::new(store, embeddings));

let hyde = HyDERetriever::new(llm, embeddings, base_retriever)
    .with_k(5)                      // 返回 5 个文档
    .with_include_original_query(true); // 同时用原始查询检索

let docs = hyde.retrieve("Rust 并发").await?;

// 获取生成的假设文档（调试用）
let hypothetical = hyde.get_hypothetical_document("Rust 并发").await?;
println!("假设文档: {}", hypothetical);
```

---

## Reranking

重排序对检索结果重新评分，提升精确度。

### 支持的 Reranker

| Reranker | 说明 |
|----------|------|
| **KeywordReranker** | 基于关键词匹配重排序 |
| **BM25Reranker** | BM25 公式重排序 |

### KeywordReranker

```rust
use langchainrust::{KeywordReranker, RerankingExecutor, SearchResult};
use std::collections::HashMap;

let reranker = Box::new(KeywordReranker::new());

// 可设置关键词权重
let weights: HashMap<String, f32> = HashMap::from([
    ("rust".to_string(), 2.0),
]);
let reranker = Box::new(KeywordReranker::new()
    .with_keyword_weights(weights));

let executor = RerankingExecutor::new(reranker)
    .with_top_n(5)          // 返回前 5 个
    .with_min_score(0.5);   // 最小分数阈值

let reranked = executor.rerank("Rust programming", search_results)?;
```

### BM25Reranker

```rust
use langchainrust::{BM25Reranker, RerankingExecutor};

let reranker = Box::new(BM25Reranker::new()
    .with_params(2.0, 0.5));  // k1=2.0, b=0.5

let executor = RerankingExecutor::new(reranker)
    .with_top_n(5);

let reranked = executor.rerank("Rust programming", search_results)?;
```

### 检索 + Reranking 流程

```rust
// 1. 检索
let results = retriever.retrieve_with_scores("Rust 并发", 20).await?;

// 2. Reranking
let reranker = Box::new(KeywordReranker::new());
let executor = RerankingExecutor::new(reranker).with_top_n(5);

let final_results = executor.rerank("Rust 并发", results)?;
```

---

## Retrieval + Agent (RAG)

将向量检索与 Agent 结合，实现 **RAG（检索增强生成）** 功能。

### 工作原理

```
┌─────────────┐      检索器      ┌─────────────┐
│   用户问题   │  ─────────────>  │  向量数据库  │
└─────────────┘                  └─────────────┘
       │                                │
       │                                ▼
       │                         ┌─────────────┐
       │                         │  相关文档   │
       │                         └─────────────┘
       │                                │
       ▼                                ▼
┌─────────────────────────────────────────────┐
│                   LLM                        │
│  (问题 + 检索到的文档作为上下文)              │
└─────────────────────────────────────────────┘
                      │
                      ▼
               ┌─────────────┐
               │   最终答案   │
               └─────────────┘
```

### 使用 with_retriever

通过 `ReActAgent::with_retriever` 创建带检索功能的 Agent：

```rust
use langchainrust::agent::{AgentExecutor, ReActAgent};
use langchainrust::llms::LLM;
use langchainrust::memory::SimpleMemory;
use langchainrust::retrieval::{
    Document, InMemoryVectorStore, MockEmbeddingModel,
    RecursiveCharacterSplitter, Retriever, SimilarityRetriever, TextSplitter,
};
use std::sync::Arc;

// 1. 准备文档
let docs = vec![
    Document::new("Rust是一种系统编程语言...".to_string()),
    Document::new("Python是一种脚本语言...".to_string()),
];

// 2. 分割文档
let splitter = RecursiveCharacterSplitter::new(100, 20);
let mut chunks = Vec::new();
for doc in docs {
    chunks.extend(splitter.split_document(&doc).unwrap());
}

// 3. 创建检索器
let embedding = Arc::new(MockEmbeddingModel::new(128));
let store = Box::new(InMemoryVectorStore::new());
let retriever = SimilarityRetriever::new(store, embedding);
retriever.add_documents(chunks).await?;

// 4. 创建 RAG Agent（传入 retriever）
let llm = LLM::new(config);
let agent = ReActAgent::with_retriever(
    llm,
    vec![],                              // 工具（可选）
    Some(Box::new(SimpleMemory::default())), // 记忆（可选）
    Arc::new(retriever) as Arc<dyn Retriever>,
    3,  // top_k
);

// 5. 执行查询
let executor = AgentExecutor::new(Box::new(agent), vec![]);
let result = executor.run_with_details("什么是Rust？").await?;
println!("回答: {}", result.answer);
```

### 带自定义模板

使用 `with_retriever_and_template`：

```rust
use langchainrust::messages::Message;
use langchainrust::prompts::ChatPromptTemplate;

let template = ChatPromptTemplate::new(vec![
    Message::system(
        "你是专业顾问。根据参考资料回答问题。\n\n参考资料：\n{context}",
    ),
    Message::human("{input}"),
]);

let agent = ReActAgent::with_retriever_and_template(
    llm,
    vec![],
    None,
    retriever,
    3,
    template,
);
```

### 执行日志

```
[检索] 正在从向量数据库检索相关文档...
[检索] 找到 3 个相关文档:
  [1] 相似度: 0.8542
  [2] 相似度: 0.7231
  [3] 相似度: 0.5123
```

### 不传 retriever

如果不传 retriever，Agent 会走普通逻辑（不检索）：

```rust
// 普通 Agent（无检索）
let agent = ReActAgent::new(llm, tools, memory);

// 带 retriever 的 Agent（会先检索文档）
let agent = ReActAgent::with_retriever(llm, tools, memory, retriever, 3);
```

---

## Plan-Execute Agent

Plan-Execute Agent 先用 LLM 规划任务步骤，逐步执行，失败时自动重规划，最后汇总结果。适合复杂、多步任务。

> 注：内部用 `FunctionCallingAgent` + 工具执行每一步，`llm` 当前需为 `OpenAIChat`。

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, PlanExecuteAgent, BaseTool};
use std::sync::Arc;

let llm = OpenAIChat::new(OpenAIConfig::default());
let tools: Vec<Arc<dyn BaseTool>> = vec![]; // 传入真实工具

let agent = PlanExecuteAgent::new(llm, tools)
    .with_max_replans(2); // 失败后最多重规划 2 次

let result = agent
    .run("调研 Rust 异步运行时，写示例代码，解释关键点")
    .await?;
println!("{}", result);
```

工作流程：规划 -> 逐步执行（FunctionCallingAgent + tools）-> 失败重规划 -> 汇总。

---

## Handoffs 多 Agent 交接

参考 OpenAI Agents SDK：主 Agent 可通过 `HandoffTool` 将任务委托给注册的专业 Agent。

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
let result = mgr.run("调研并写一篇文章".to_string()).await?;

// 为每个注册 Agent 生成 HandoffTool（名为 handoff_to_{agent}），供主 Agent 调用
let mgr = Arc::new(mgr);
let handoff_tools = mgr.handoff_tools();
let history = mgr.history(); // 交接历史
```

`handoff_tools()` 返回的工具名为 `handoff_to_{agent}`；也可用 `execute_handoff(Handoff)` 直接交接。

---

## Streaming Tool Calls

`StreamingFunctionCallingAgent` 流式输出 LLM 文本（逐 token），并通过事件流暴露工具调用状态。

```rust
use langchainrust::StreamingFunctionCallingAgent;
use langchainrust::agents::streaming::AgentStreamEvent;
use futures_util::StreamExt;

let agent = StreamingFunctionCallingAgent::new(llm);
let mut stream = agent.invoke_stream("用一句话介绍 Rust".to_string()).await;

while let Some(event) = stream.next().await {
    match event {
        AgentStreamEvent::Text { content } => print!("{}", content),
        AgentStreamEvent::ToolCall { state } => {
            // state: Started / ArgumentsStreaming / Completed / Failed ...
        }
        AgentStreamEvent::FinalAnswer { content } => println!("\n[完成] {}", content),
    }
}
```

事件见 `AgentStreamEvent`（`Text` / `ToolCall` / `FinalAnswer`）与 `ToolCallState`。

---

## Guardrails 安全护栏

输入/输出验证，防恶意输入、防敏感信息泄露。实现 `InputGuardrail` / `OutputGuardrail` trait，或用内置验证器，再用 `GuardedAgent` 包装 Agent。

```rust
use langchainrust::guardrails::{
    GuardrailsConfig, MaxLengthGuardrail, SensitiveInfoGuardrail, GuardedAgent,
};
use langchainrust::{BaseAgent, AgentExecutor, FunctionCallingAgent, OpenAIChat, OpenAIConfig};
use std::sync::Arc;

let config = GuardrailsConfig::new()
    .with_input(Arc::new(MaxLengthGuardrail::new(1000)))    // 输入限长
    .with_output(Arc::new(SensitiveInfoGuardrail::new()));  // 输出防敏感信息

let agent = FunctionCallingAgent::new(OpenAIChat::new(OpenAIConfig::default()), vec![], None);
let executor = Arc::new(AgentExecutor::new(
    Arc::new(agent) as Arc<dyn BaseAgent>,
    vec![],
));

let mut guarded = GuardedAgent::new(executor, config);
let result = guarded.invoke("帮我总结这段内容".to_string()).await?; // 验输入 -> Agent -> 验输出
let violations = guarded.violations();
```

内置验证器：`MaxLengthGuardrail`（输入限长）、`ForbiddenWordsGuardrail`（禁用词）、`SensitiveInfoGuardrail`（API key / 邮箱 / 信用卡 / 关键词，可 `with_keywords` 扩展）。也可用 `GuardrailRunner` 手动驱动验证流程。

---

## Token 计数器与成本估算

`TiktokenCounter` 用 cl100k_base（GPT-3.5/4/4o）计数；`TokenTrackingLLM` 包装 LLM 自动累计用量；`ModelPricing` 估算成本。

```rust
use langchainrust::{TokenTrackingLLM, ModelPricing, OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

let tracked = TokenTrackingLLM::for_openai(OpenAIChat::new(OpenAIConfig::default()))?;

let result = tracked.chat(vec![Message::human("你好")], None).await?;

let usage = tracked.get_usage();                               // prompt / completion / total tokens
let cost = tracked.estimate_cost(&ModelPricing::gpt4o_mini()); // 美元
```

`ModelPricing::gpt4o()` / `gpt4o_mini()` 为内置定价，也可 `ModelPricing::new(prompt_per_1k, completion_per_1k)` 自定义。

---

## Sessions 会话管理

`SessionManager` 管理多轮对话会话生命周期：创建/获取/归档，对话时自动维护历史，存储可插拔（`SessionStore` trait）。

```rust
use langchainrust::sessions::{SessionManager, MemorySessionStore};
use langchainrust::{OpenAIChat, OpenAIConfig};
use std::sync::Arc;

let manager = SessionManager::new(Arc::new(MemorySessionStore::new()));
let id = manager.create_session_for("user_1").await?;

let llm = OpenAIChat::new(OpenAIConfig::default());
let r1 = manager.chat(&id, &llm, "我叫小明".to_string()).await?;
let r2 = manager.chat(&id, &llm, "我叫什么？".to_string()).await?; // 记得上一轮

let history = manager.history(&id).await?;  // Vec<Message>
manager.clear(&id).await?;                   // 清空历史（保留会话）
manager.archive(&id).await?;                 // 归档
let sessions = manager.list_by_user("user_1").await?;
```

`SessionStore` trait 含 `create/get/update/delete/list_by_user`，可自行实现持久化后端（Redis/DB）；内置 `MemorySessionStore` 用于测试与单进程。

---

## MCP 协议

[MCP](https://modelcontextprotocol.io)（Model Context Protocol）是 Anthropic 推出的工具协议标准。`MCPClient` 连接任意 MCP Server 获取工具，并适配为 `BaseTool` 供 Agent 使用。

```rust
use langchainrust::mcp::{MCPClient, MCPConfig};
use langchainrust::{BaseAgent, AgentExecutor, FunctionCallingAgent, OpenAIChat, OpenAIConfig};
use std::sync::Arc;

// Stdio：启动 MCP Server 子进程
let config = MCPConfig::stdio(
    "npx",
    vec!["@anthropic/mcp-server-filesystem".to_string(), "/tmp".to_string()],
);
// 或 SSE：MCPConfig::sse("http://localhost:3001/sse")

let mut client = MCPClient::connect(config).await?;
let tools = client.list_tools().await?;           // tools/list
println!("MCP 工具数量: {}", tools.len());

// 适配为 BaseTool 列表，直接喂给 Agent
let mcp_tools = client.as_tools().await;
let agent = FunctionCallingAgent::new(
    OpenAIChat::new(OpenAIConfig::default()),
    mcp_tools,
    None,
);
let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, vec![]);
let result = executor.invoke("读取 /tmp/notes.txt".to_string()).await?;

client.close().await?;
```

`MCPConfig::stdio(command, args)` / `MCPConfig::sse(url)` / `.with_env(k, v)`；`client.call_tool(name, arguments)` 直接调用工具；`as_tools()` 把工具包装为 `MCPToolAdapter`（实现 `BaseTool`）。

---

### MCPServer（暴露本地工具）

`MCPServer` 与 `MCPClient` 对称：把本地 `BaseTool` 暴露为 MCP Server，供 Claude Desktop / Cursor 等 host 调用。支持 `initialize` 握手、`tools/list`、`tools/call`。

```rust
use langchainrust::{MCPServer, Calculator, BaseTool};
use std::sync::Arc;

let tool: Arc<dyn BaseTool> = Arc::new(Calculator::new());
let server = MCPServer::new()
    .with_tool(tool)
    .with_server_info("my-tools", "0.1.0");

// 监听 stdio，供 host 连接（如 Claude Desktop 配置该命令）
server.serve_stdio().await?;
```

也可用 `server.handle_request(req)` 单步处理 JSON-RPC 请求，自行管理 transport。

## Tools

### 内置工具

| 工具名 | 描述 | 参数 |
|--------|------|------|
| `Calculator` | 执行基本数学运算（加减乘除） | `expression`: 数学表达式 |
| `SimpleMathTool` | 高级数学运算（幂、开方、三角函数等） | `operation`: 操作类型, `value`: 数值 |
| `DateTimeTool` | 日期时间查询和计算 | `operation`: 操作类型, `datetime`: 日期时间 |
| `URLFetchTool` | 网页抓取和解析 | `operation`: 操作类型, `url`: 网址 |
| `WikipediaTool` | Wikipedia 百科搜索 | `query`: 搜索词, `lang`: 语言, `top_k`: 结果数 |
| `DuckDuckGoSearchTool` | 网页搜索引擎（无需 API Key） | `query`: 搜索词, `top_k`: 结果数 |
| `PythonREPLTool` | Python 代码执行 | `code`: Python 代码, `timeout_seconds`: 超时 |

### 使用工具

```rust
use langchainrust::{Calculator, DateTimeTool, SimpleMathTool, BaseTool};
use std::sync::Arc;

let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(DateTimeTool::new()),
    Arc::new(SimpleMathTool::new()),
];
```

### WikipediaTool

```rust
use langchainrust::tools::{WikipediaTool, Tool};
use langchainrust::tools::WikipediaInput;

let tool = WikipediaTool::new();
let result = tool.invoke(WikipediaInput {
    query: "Rust 编程语言".into(),
    top_k: Some(3),
    lang: Some("zh".into()),
    full_content: Some(false),
}).await?;

for r in &result.results {
    println!("标题: {}", r.title);
    println!("摘要: {}", r.snippet);
}
```

### DuckDuckGoSearchTool

无需 API Key 的网页搜索工具：

```rust
use langchainrust::tools::{DuckDuckGoSearchTool, Tool};
use langchainrust::tools::SearchInput;

let tool = DuckDuckGoSearchTool::new();
let result = tool.invoke(SearchInput {
    query: "Rust async/await 教程".into(),
    top_k: Some(5),
}).await?;

for r in &result.results {
    println!("{}: {}", r.title, r.url);
}
```

### PythonREPLTool

在本地 Python 环境执行代码，适合数学计算、数据处理场景：

```rust
use langchainrust::tools::{PythonREPLTool, Tool};
use langchainrust::tools::PythonREPLInput;

let tool = PythonREPLTool::new();  // 自动查找系统 Python

// 使用自定义 Python 路径
let tool = PythonREPLTool::with_python_path("/usr/bin/python3");

let result = tool.invoke(PythonREPLInput {
    code: r#"
import math
print(f"π = {math.pi}")
print(f"sin(π/2) = {math.sin(math.pi/2)}")
"#.into(),
    timeout_seconds: Some(30),
}).await?;

println!("{}", result.stdout);
```

### 直接调用工具

```rust
use langchainrust::{SimpleMathTool, BaseTool};

let math = SimpleMathTool::new();

let result = math.run(r#"{"operation": "power", "value": 2, "value2": 10}"#.to_string()).await?;
println!("2^10 = {}", result); // 输出: 2^10 = 1024
```

### to_tool_definition()

将 BaseTool 转换为 ToolDefinition，用于 Function Calling：

```rust
use langchainrust::{Calculator, BaseTool, to_tool_definition, ToolDefinition};

let calculator = Calculator::new();

// 自动从 args_schema() 生成 JSON Schema
let tool_def: ToolDefinition = to_tool_definition(&calculator);

// 用于 bind_tools()
let llm_with_tools = llm.bind_tools(vec![tool_def]);
```

### 自定义 Tool

实现 `BaseTool` trait 创建自定义工具：

```rust
use async_trait::async_trait;
use langchainrust::{BaseTool, ToolError};
use schemars::JsonSchema;
use serde::Deserialize;

// 定义输入类型（用于 JSON Schema）
#[derive(JsonSchema, Deserialize)]
struct EchoInput {
    text: String,
}

pub struct EchoTool;

#[async_trait]
impl BaseTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "原样返回输入的文本"
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        // 解析输入
        let args: EchoInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(e.to_string()))?;
        
        Ok(args.text)
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        // 自动生成 JSON Schema
        use schemars::schema_for;
        serde_json::to_value(schema_for!(EchoInput)).ok()
    }
}
```

使用自定义工具：

```rust
let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(EchoTool),
];

let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
```

### 扩展工具（HTTPTool / FileTool / SQLTool）

v0.3.0 新增三个面向生产环境的工具，均实现 `BaseTool`。

**HTTPTool** —— 发起 GET/POST 请求：

```rust
use langchainrust::HTTPTool;
use serde_json::json;

let http = HTTPTool::new();
let body = http.post("https://httpbin.org/post", json!({"k": "v"})).await?;
// 作为 BaseTool：输入 JSON {"url":"...","method":"get|post","body":{...}}
```

**FileTool** —— 沙箱文件读写（限制 `base_path`、扩展名白名单、大小上限、防路径越界）：

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

**SQLTool** —— 只读 SQL 查询（仅 SELECT，表白名单；需 `sqlite-storage` feature）：

```rust
use langchainrust::tools::extended::SQLTool;

let sql = SQLTool::new("data.db")?
    .with_allowed_tables(vec!["users".into()]);
let rows = sql.execute("SELECT id, name FROM users")?; // Vec<HashMap<String,String>>
// 非 SELECT（如 DROP/INSERT）会被拒绝
```

> `SQLTool` 在 `sqlite-storage` feature 下可用；`HTTPTool` / `FileTool` 默认可用。

---

## Callbacks

回调系统用于追踪、监控和记录 LLM 应用执行过程。

### CallbackManager

管理多个回调处理器：

```rust
use langchainrust::{CallbackManager, StdOutHandler, LangSmithHandler};
use std::sync::Arc;

let manager = CallbackManager::new()
    .add_handler(Arc::new(StdOutHandler::new()))
    .add_handler(Arc::new(LangSmithHandler::from_env()?));

// 使用回调
let llm = OpenAIChat::new(config);
// llm.with_callbacks(Arc::new(manager));
```

### StdOutHandler

输出到标准输出（调试用）：

```rust
use langchainrust::StdOutHandler;

let handler = StdOutHandler::new();
```

### FileCallbackHandler

输出到文件：

```rust
use langchainrust::{FileCallbackHandler, LogFormat};

// JSON 格式
let handler = FileCallbackHandler::new("trace.json", LogFormat::Json);

// 纯文本格式
let handler = FileCallbackHandler::new("trace.log", LogFormat::Text);
```

### LangSmith 追踪

LangSmith 是 LangChain 官方的追踪平台，用于监控和调试 LLM 应用。

#### 配置环境变量

```bash
export LANGSMITH_API_KEY="ls_xxxxx"       # API Key（必须）
export LANGSMITH_PROJECT="my-project"      # 项目名称
export LANGSMITH_TRACING="true"            # 启用追踪
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"  # API 端点
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
    workspace_id: None,  // 组织账号需要
};

let handler = LangSmithHandler::new(config);
```

#### RunTree

追踪执行树结构：

```rust
use langchainrust::{RunTree, RunType};

let run = RunTree::new(
    "my-run-id",
    RunType::LLM,      // 类型：LLM/Chain/Agent/Tool
    "GPT-4",           // 运行名称
);

// 更新运行状态
run.update_status("completed");
run.update_output("Response content...");

// 发送到 LangSmith
let client = LangSmithClient::new(config);
client.create_run(&run).await?;
```

#### LangSmith 功能

| 功能 | 说明 |
|------|------|
| **追踪** | 记录每次 LLM 调用 |
| **监控** | 查看 Token 使用、延迟 |
| **调试** | 对比不同版本输出 |
| **评估** | 测试集评估、对比 |
| **分享** | 分享追踪链接 |

---

### OtelHandler（OpenTelemetry）

把 LLM / Chain / Tool / Retriever 的开始 / 结束 / 错误转为 OpenTelemetry span，接入 OTel 链路追踪。需开启 `opentelemetry` feature，并先配置全局 tracer provider。

```toml
[dependencies]
langchainrust = { version = "0.4", features = ["opentelemetry"] }
```

```rust
use langchainrust::{CallbackManager, OtelHandler};
use std::sync::Arc;

// 需先 opentelemetry::global::set_tracer_provider(...) 配置 OTLP exporter
let manager = CallbackManager::new()
    .add_handler(Arc::new(OtelHandler::from_global("langchainrust")));
// llm.with_callbacks(Arc::new(manager));
```

执行事件会生成嵌套 span，可对接 Jaeger / Tempo / Grafana 等后端。

---

## Evaluation 评估

评估模块用于量化 LLM 输出质量：改了 prompt / 换了模型 / 接了 RAG，跑评测集看分数涨没涨。10 种评测器，分 5 类：

| 类别 | 评测器 | 说明 |
|------|--------|------|
| 字面 | `ExactMatch` / `StringDistance` | 完全相等 / Levenshtein 编辑距离归一 |
| 语义 | `EmbeddingSimilarity` / `LLMAsJudge` / `PairwiseJudge` | 向量余弦 / LLM 当裁判 / 成对二选一（交换 A/B 消位置偏差） |
| 规则 | `ContainsKeyword` / `RegexMatch` / `LengthCheck` | 关键词 / 正则 / 长度 |
| 经典 NLP | `Bleu` | n-gram 精度（字符级分词 + 平滑） |
| RAG | `Faithfulness` | 拆主张逐条验证，检测幻觉 |

### EvalRunner（批量评测）

`EvalRunner` 在 `Dataset` 上跑一组评测器，产出 `Report`（每条得分 + 各评测器均分）。

```rust
use langchainrust::evaluation::*;
use async_trait::async_trait;

// 1. 数据集：input + reference（参考答案）
let dataset = Dataset::new(vec![
    Example::new("2+2=?", "4"),
    Example::new("中国首都?", "北京"),
]);
// 或从 jsonl 加载：Dataset::from_jsonl("eval.jsonl")?

// 2. Predictor：被测对象（LLM / Chain / Agent），实现 predict
struct MyLLM;
#[async_trait]
impl Predictor for MyLLM {
    async fn predict(&self, input: &str) -> Result<String, EvalError> {
        Ok("4".to_string()) // 实际调用你的 LLM
    }
}

// 3. 评测器 + 运行
let runner = EvalRunner::new(vec![
    Box::new(ExactMatch),
    Box::new(StringDistance),
]);
let report = runner.run(&dataset, &MyLLM).await?;
println!("{:?}", report.summary);
// {"ExactMatch": 1.0, "StringDistance": 1.0}
```

### Faithfulness（RAG 幻觉检测）

`Faithfulness` 把预测拆成原子主张，逐条对照参考（上下文）验证，检测回答有没有编造。RAG 场景最实用。

```rust
use langchainrust::evaluation::{Faithfulness, Evaluator};
use langchainrust::OpenAIChat;

let judge = Faithfulness::new(OpenAIChat::new(config));
// reference 是上下文：「年假 15 天」
let ok = judge.eval("", "年假 15 天，可累积", "年假 15 天").await?;
assert_eq!(ok.value, 1.0); // 忠实

let halluc = judge.eval("", "年假 20 天", "年假 15 天").await?;
assert_eq!(halluc.value, 0.0); // 编造，抓到
```

`with_llm_split(true)` 用 LLM 拆主张（默认按句号），`with_empty_score(x)` 设无主张时的得分。验证并发执行（`join_all`）。

---

## 配置与安全

- **不要**把真实 API Key 提交到 Git 仓库
- 推荐通过环境变量读取（例如 `OPENAI_API_KEY`）
- `OpenAIConfig.streaming=true` 走流式 API；否则走非流式
- `OpenAIConfig.factor` 用于模型路由，值越大表示模型越强/越贵

---

## LangGraph

LangGraph 是图状工作流框架，用于构建复杂的 AI 应用流程。

### 核心概念

| 组件 | 说明 |
|------|------|
| **StateGraph** | 状态图构建器 |
| **GraphNode** | 节点抽象（SyncNode、AsyncNode） |
| **GraphEdge** | 边和条件路由 |
| **StateSchema** | 状态管理（AgentState） |
| **Reducer** | 状态更新策略（ReplaceReducer、AppendReducer） |
| **Checkpointer** | 执行状态持久化 |

### StateGraph 基础用法

```rust
use langchainrust::langgraph::{StateGraph, AgentState, START, END};
use std::collections::HashMap;

// 创建状态图
let mut graph = StateGraph::new();

// 添加节点
graph.add_node("analyze", |state: AgentState| {
    // 分析节点逻辑
    let mut new_state = state.clone();
    new_state.steps.push("已分析".to_string());
    new_state
});

graph.add_node("process", |state: AgentState| {
    // 处理节点逻辑
    let mut new_state = state.clone();
    new_state.steps.push("已处理".to_string());
    new_state
});

graph.add_node("output", |state: AgentState| {
    // 输出节点逻辑
    state
});

// 添加边
graph.add_edge(START, "analyze");
graph.add_edge("analyze", "process");
graph.add_edge("process", "output");
graph.add_edge("output", END);

// 编译图
let compiled = graph.compile();

// 执行
let initial_state = AgentState::new();
let result = compiled.invoke(initial_state).await?;
```

### 条件边路由

使用条件边根据状态决定下一步：

```rust
use langchainrust::langgraph::{ConditionalEdge, FunctionRouter};

// 条件路由函数
let router = FunctionRouter::new(|state: &AgentState| {
    if state.messages.len() > 5 {
        "summarize"
    } else {
        "continue"
    }
});

// 添加条件边
graph.add_conditional_edge(
    "analyze",
    ConditionalEdge::new(router, vec!["summarize", "continue"]),
);

graph.add_edge("summarize", END);
graph.add_edge("continue", "process");
```

### Human-in-the-loop / 中断恢复

在节点执行前/后中断，支持逐步执行和人工确认：

```rust
use langchainrust::langgraph::{CompiledGraph, GraphExecution, MemoryCheckpointer};

// 编译时设置中断点 + 检查点存储
let compiled = graph.compile()
    .map_err(|e| ...)?
    .with_checkpointer(MemoryCheckpointer::new())
    .with_interrupt_before(vec!["output", "analyze"]);  // 这些节点执行前暂停
    .with_interrupt_after(vec!["review"]);              // review节点执行后暂停

// 执行，到达中断点时返回 ExecutionInterrupted 错误
match compiled.invoke(initial_state).await {
    Ok(result) => { /* 全部完成 */ }
    Err(GraphError::ExecutionInterrupted(node)) => {
        println!("暂停在节点: {}", node);
        
        // 从检查点恢复执行上下文
        if let Some(execution) = compiled.create_resume_execution(&node).await {
            // 可以选择修改状态后继续
            let result = compiled.resume(execution).await?;
        }
    }
    Err(e) => { /* 其他错误 */ }
}
```

`last_checkpoint_state()` 可获取最后一个检查点的状态：

```rust
if let Some(state) = compiled.last_checkpoint_state().await {
    println!("检查点状态: {:?}", state);
}
```

### Subgraph 子图嵌套

将子图作为节点嵌入主图：

```rust
use langchainrust::langgraph::{SubgraphNode, StateMapper};

// 创建子图
let mut subgraph = StateGraph::new();
subgraph.add_node("sub_task_1", |state| state);
subgraph.add_node("sub_task_2", |state| state);
subgraph.add_edge(START, "sub_task_1");
subgraph.add_edge("sub_task_1", "sub_task_2");
subgraph.add_edge("sub_task_2", END);

let sub_compiled = subgraph.compile();

// 创建状态映射器（父子图状态转换）
let mapper = StateMapper::new(
    |parent: &AgentState| -> SubState {  // 父 → 子
        SubState::from_parent(parent)
    },
    |sub: &SubState, parent: &AgentState| -> AgentState {  // 子 → 父
        parent.merge_sub(sub)
    },
);

// 将子图作为节点添加到主图
graph.add_node("subgraph", SubgraphNode::new(sub_compiled, mapper));
```

### Parallel 并行执行

并行执行多个独立节点：

```rust
// 添加多个并行节点
graph.add_node("task_a", |state| { /* ... */ });
graph.add_node("task_b", |state| { /* ... */ });
graph.add_node("task_c", |state| { /* ... */ });

// Fan-Out: 从一个节点分发到多个并行节点
graph.add_edge(START, "dispatch");
graph.add_edge("dispatch", "task_a");
graph.add_edge("dispatch", "task_b");
graph.add_edge("dispatch", "task_c");

// Fan-In: 多个并行节点汇聚到一个节点
graph.add_edge("task_a", "merge");
graph.add_edge("task_b", "merge");
graph.add_edge("task_c", "merge");
graph.add_edge("merge", END);

// 编译时启用并行执行
let compiled = graph.compile()
    .with_parallel_execution(true);

// invoke_parallel() 并行执行
let result = compiled.invoke_parallel(initial_state).await?;
```

### Checkpointer 持久化

保存和恢复执行状态：

```rust
use langchainrust::langgraph::{
    MemoryCheckpointer, FileCheckpointer, Checkpointer,
};

// 内存 Checkpointer
let checkpointer = MemoryCheckpointer::new();

// 文件 Checkpointer
let file_checkpointer = FileCheckpointer::new("./checkpoints/");

// 编译时设置 Checkpointer
let compiled = graph.compile()
    .with_checkpointer(checkpointer);

// 执行时指定 thread_id（用于区分不同会话）
let thread_id = "conversation_123";
let result = compiled.invoke_with_thread(initial_state, thread_id).await?;

// 恢复之前的状态
let previous_state = compiled.get_state(thread_id).await?;
let resumed_result = compiled.invoke_with_thread(previous_state, thread_id).await?;
```

### 可视化输出

三种可视化格式：

```rust
// ASCII 图形（终端显示）
println!("{}", compiled.visualize_ascii());
// 输出:
//   START → analyze → process → output → END

// Mermaid 图表（Markdown 文档）
println!("{}", compiled.visualize_mermaid());
// 输出:
//   graph LR
//     START --> analyze
//     analyze --> process
//     process --> output
//     output --> END

// JSON 结构（程序处理）
let json = compiled.visualize_json();
// 输出:
//   {"nodes": ["analyze", "process", "output"], "edges": [...]}
```

### Graph 验证

自动检测常见问题：

```rust
// 编译时自动验证
let compiled = graph.compile();  // 会抛出验证错误

// 手动验证
graph.validate()?;  // 返回验证结果

// 检测项：
// - 死循环：validate_cycles()
// - 孤立节点：validate_unreachable_nodes()
// - 重复边：validate_duplicate_edges()
```

---

## Output Parsers

输出解析器将 LLM 的文本输出解析为结构化数据。

### StrOutputParser（字符串透传）

最简单的解析器，直接将 LLM 输出作为字符串返回：

```rust
use langchainrust::output_parsers::StrOutputParser;

let parser = StrOutputParser::new();
let result = parser.parse("Hello, world!").await?;
assert_eq!(result, "Hello, world!");
```

### CommaSeparatedListOutputParser（逗号分隔列表）

将逗号分隔的文本解析为列表，支持中英文逗号：

```rust
use langchainrust::output_parsers::CommaSeparatedListOutputParser;

let parser = CommaSeparatedListOutputParser::new();

// 英文逗号
let list = parser.parse("apple,banana,orange").await?;
assert_eq!(list, vec!["apple", "banana", "orange"]);

// 中文逗号
let list = parser.parse("苹果，香蕉，橙子").await?;
assert_eq!(list, vec!["苹果", "香蕉", "橙子"]);
```

### JsonOutputParser（JSON 解析）

从 LLM 输出中提取并解析 JSON，支持 Markdown 代码块包裹和部分解析：

```rust
use langchainrust::output_parsers::JsonOutputParser;
use serde_json::Value;

let parser = JsonOutputParser::new();

// 纯 JSON
let val: Value = parser.parse(r#"{"name": "Rust", "year": 2015}"#).await?;

// Markdown 代码块包裹
let md = "```json\n{\"name\": \"Rust\", \"year\": 2015}\n```";
let val: Value = parser.parse(md).await?;  // 自动提取

// 部分解析（流式场景）
let partial = parser.parse_partial(r#"{"name": "Rust"#).await?;
// 返回完成的部分
```

### StructuredOutputParser（键值对格式解析）

解析键值对格式的输出，每行一个字段：

```rust
use langchainrust::output_parsers::StructuredOutputParser;
use std::collections::HashMap;

let parser = StructuredOutputParser::new();

let output = "\
name: Rust
year: 2015
paradigm: 系统编程";

let result: HashMap<String, String> = parser.parse(output).await?;
assert_eq!(result.get("name").unwrap(), "Rust");
assert_eq!(result.get("year").unwrap(), "2015");
```

### TypedOutputParser\<T\>（类型化 JSON 反序列化）

将 JSON 输出直接反序列化为任意 `Deserialize` 类型：

```rust
use langchainrust::output_parsers::TypedOutputParser;
use serde::Deserialize;

#[derive(Deserialize)]
struct Language {
    name: String,
    year: u32,
    paradigm: String,
}

let parser = TypedOutputParser::<Language>::new();
let lang: Language = parser.parse(r#"{"name": "Rust", "year": 2015, "paradigm": "系统编程"}"#).await?;

println!("{} 诞生于 {}", lang.name, lang.year);
```

---

## Document Chains

文档处理 Chain 用于处理大量文档，提供不同的文档整合策略。

### StuffDocumentsChain（全文填充）

将所有文档拼接后一次性送入 LLM：

```rust
use langchainrust::{StuffDocumentsChain, OpenAIChat, Document};

let llm = OpenAIChat::from_env();
let chain = StuffDocumentsChain::new(llm)
    .with_max_doc_length(2000);  // 单文档最大字符数

let docs = vec![
    Document::new("Rust 是一门系统编程语言..."),
    Document::new("它的所有权机制保证了内存安全..."),
];

let answer = chain.invoke_with_documents(docs, "总结 Rust 的特点").await?;
```

适用场景：文档总量不超过 LLM 上下文窗口。

### RefineDocumentsChain（迭代优化）

逐个文档迭代，每次用前一轮输出 + 新文档联合生成：

```rust
use langchainrust::{RefineDocumentsChain, OpenAIChat, Document};

let llm = OpenAIChat::from_env();
let chain = RefineDocumentsChain::new(llm);

let docs = vec![
    Document::new("第一份文档内容..."),
    Document::new("第二份文档内容..."),
    Document::new("第三份文档内容..."),
];

let answer = chain.invoke_with_documents(docs, "综合分析这些问题").await?;
```

适用场景：文档总量超过上下文窗口，需要逐步精炼。

### MapReduceDocumentsChain（并行映射 + 合并）

先并行处理每个文档，再合并所有中间结果：

```rust
use langchainrust::{MapReduceDocumentsChain, OpenAIChat, Document};

let llm = OpenAIChat::from_env();
let chain = MapReduceDocumentsChain::new(llm);

let docs = vec![
    Document::new("Rust 的所有权系统..."),
    Document::new("Rust 的生命周期..."),
    Document::new("Rust 的 trait 系统..."),
];

let summary = chain.invoke_with_documents(docs, "用一句话总结").await?;
```

流程：并行映射（Map）→ 合并缩减（Reduce），适合大规模文档。

### MapRerankDocumentsChain（评分排序）

并行处理每个文档后按评分排序，返回最高分结果：

```rust
use langchainrust::{MapRerankDocumentsChain, OpenAIChat, Document};

let llm = OpenAIChat::from_env();
let chain = MapRerankDocumentsChain::new(llm);

let docs = vec![
    Document::new("方案 A：使用 async/await..."),
    Document::new("方案 B：使用线程..."),
    Document::new("方案 C：使用 Actor 模型..."),
];

let best_answer = chain.invoke_with_documents(docs, "哪个并发方案最适合？").await?;
// 返回评分最高的方案
```

---

## Vector Stores

### ChromaDB（轻量级向量数据库）

通过 ChromaDB HTTP API 进行向量存储和检索：

```rust
use langchainrust::{ChromaDBVectorStore, ChromaDBConfig, VectorStore, Document};
use std::sync::Arc;

// 配置连接（默认 http://localhost:8000）
let config = ChromaDBConfig::new(
    "http://localhost:8000",
    "my_collection",
    1536,  // 向量维度
);

let store = Arc::new(ChromaDBVectorStore::new(config).await?);

// 添加文档
store.add_documents(vec![
    Document::new("Rust 是一门系统编程语言"),
    Document::new("Python 是一门脚本语言"),
]).await?;

// 检索
let results = store.search("系统编程", 3).await?;
```

启动 ChromaDB：

```bash
docker run -p 8000:8000 chromadb/chroma
```

### PGVectorStore

PostgreSQL + pgvector 扩展向量库。需 `pgvector-storage` feature；因 `sqlx` / `pgvector` 依赖未在 crate 内启用，用户需自行在 `Cargo.toml` 添加 `sqlx` 与 `pgvector` 依赖。

```rust
use langchainrust::vector_stores::PGVectorStore;
use langchainrust::embeddings::Embeddings;

let store = PGVectorStore::new(
    "postgres://user:pass@localhost/db",
    "docs",
    1536, // 向量维度
).await?;
// embeddings: impl Embeddings（如 OpenAIEmbeddings）；docs: &[Document]
store.add_documents(&docs, &embeddings).await?;
let found = store.similarity_search("查询", 5, &embeddings).await?;
store.delete("doc-id").await?;
```

`PGVectorStore::new` 会自动 `CREATE EXTENSION IF NOT EXISTS vector` 并建表；`build_table_sql(table, dim)` 是可单独测试的建表 SQL 纯函数。

### PineconeStore

Pinecone 向量库（reqwest HTTP API，无需 feature，默认可用）。

```rust
use langchainrust::vector_stores::PineconeStore;
use langchainrust::embeddings::Embeddings;

// host 格式：https://{index-name}.svc.{environment}.pinecone.io
let store = PineconeStore::new("your-api-key", "https://my-index.svc.prod.pinecone.io");

// embeddings: impl Embeddings
store.upsert(&docs, &embeddings).await?;       // 自动 embed 文档
let qvec: Vec<f32> = embeddings.embed_query("查询").await?; // query 需传已 embed 的向量
let found = store.query(qvec, 5).await?;
store.delete(&["id1".to_string()]).await?;
```

`upsert` 自动调用 `embed_documents`；`query` 需传入已 embed 的向量（`embed_query` 的结果）。

---

## LLM Cache

内存缓存避免重复的 LLM 调用，减少延迟和费用。

### 基础用法

```rust
use langchainrust::core::cache::{LLMCache, CacheConfig};
use std::time::Duration;

// 默认配置：最多 1000 条目，1 小时 TTL
let cache = LLMCache::new();

// 缓存 LLM 调用
let key = "user_query_hash";
cache.put(key, llm_result).await;

// 获取缓存
if let Some(cached) = cache.get(key).await {
    println!("缓存命中: {}", cached.result.content);
}
```

### CacheConfig 配置

```rust
use langchainrust::core::cache::CacheConfig;
use std::time::Duration;

// 自定义配置：500 条缓存，5 分钟过期
let config = CacheConfig::new()
    .with_max_entries(500)
    .with_ttl(Duration::from_secs(300));

// 永不过期
let config = CacheConfig::new()
    .no_ttl();

// 禁用缓存
let config = CacheConfig::new()
    .disabled();

let cache = LLMCache::with_config(config);
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_entries` | 1000 | 最大缓存条目数 |
| `ttl` | Some(1 小时) | 过期时间，None 永不过期 |
| `enabled` | true | 是否启用 |

---

## 存储后端

### RedisDocumentStore

基于 Redis 的文档存储，支持持久化和分布式共享：

```rust
use langchainrust::{RedisDocumentStore, RedisStoreConfig};

let config = RedisStoreConfig::new("redis://127.0.0.1:6379")
    .with_prefix("myapp");

let store = RedisDocumentStore::new(config).await?;

store.add_document(doc).await?;
let doc = store.get_document("doc_id").await?;
```

需要启用 feature `redis-storage`：

```toml
[dependencies]
langchainrust = { features = ["redis-storage"] }
```

### SQLiteDocumentStore

基于 SQLite 的本地文档存储，零配置开箱即用：

```rust
use langchainrust::{SQLiteDocumentStore, SQLiteStoreConfig};

let config = SQLiteStoreConfig::new("data.db");
let store = SQLiteDocumentStore::new(config)?;

store.add_document(doc).await?;
let doc = store.get_document("doc_id").await?;
```

需要启用 feature `sqlite-storage`：

```toml
[dependencies]
langchainrust = { features = ["sqlite-storage"] }
```

---

## 测试

```bash
# 运行全部测试
cargo test

# 运行单元测试
cargo test --lib

# 运行 Function Calling Agent 测试
cargo test --test function_calling_agent -- --include-ignored --nocapture

# 运行 ReAct Agent 测试
cargo test --test integration_agent_react -- --include-ignored --nocapture

# 运行特定测试函数
cargo test test_fc_agent_with_calculator -- --include-ignored --nocapture
```

---

## 模块结构

```
src/
├── core/                # 核心抽象
│   ├── language_models/ # Base LLM traits
│   ├── runnables/       # Runnable trait
│   └── tools/           # Tool trait + ToolDefinition + to_tool_definition()
├── language_models/     # LLM 实现
│   └── openai/          # OpenAI 客户端（支持 bind_tools）
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
└── schema/              # 数据结构（Message、ToolCall）
```

---

## API 速查

### Agent 构造方法

| Agent | 方法 | 说明 |
|-------|------|------|
| **FunctionCallingAgent** | `new(llm, tools, system_prompt)` | 推荐，原生 FC |
| **ReActAgent** | `new(llm, tools, memory)` | 兼容旧模型 |

### 工具调用

| Agent | 工具调用方式 | 说明 |
|-------|-------------|------|
| FunctionCallingAgent | `tool_calls` JSON | 类型安全 |
| ReActAgent | 文本 `Action: xxx` | 正则提取 |

### Function Calling API

| 方法 | 说明 |
|------|------|
| `bind_tools(vec![ToolDefinition])` | 绑定工具到 LLM |
| `to_tool_definition(&tool)` | 将 BaseTool 转为 ToolDefinition |
| `ToolDefinition::from_type::<T>()` | 从类型自动生成 Schema |
| `ToolCall::parse_arguments::<T>()` | 解析工具参数 |

### RAG 执行流程

| 步骤 | 说明 |
|------|------|
| 1. 检索 | 从向量数据库检索 top_k 个相关文档 |
| 2. 构建上下文 | 将文档格式化为 prompt 上下文 |
| 3. 调用 LLM | 将问题和上下文一起发送给模型 |
| 4. 返回答案 | 模型基于上下文生成答案 |

---

## 最佳实践

使用 LangChainRust 的推荐做法：

### 01 Agent 选择

优先使用 FunctionCallingAgent（更可靠），本地模型用 ReActAgent。

```rust
// 支持 Function Calling 的模型（推荐）
let agent = FunctionCallingAgent::new(llm, tools.clone(), None);

// 不支持 Function Calling 的模型
let agent = ReActAgent::new(llm, tools.clone(), None);
```

### 02 模型选择

| 使用场景 | 推荐模型 |
|---------|----------|
| Agent 任务 | gpt-4 |
| 简单对话 | gpt-3.5-turbo |
| 代码生成 | gpt-4 |
| 复杂推理 | gpt-4-turbo |

### 03 Memory 选择

| 对话类型 | 推荐 Memory |
|---------|-------------|
| 长对话 | SummaryBufferMemory |
| 短对话 | BufferMemory |
| 固定窗口 | WindowMemory |

### 04 RAG 配置

推荐配置值：

| 参数 | 推荐值 |
|------|--------|
| chunk_size | 200-500 |
| overlap | 50-100 |
| top_k | 3-10 |

### 05 流式输出

交互场景使用 `stream_chat()`，降低感知延迟：

```rust
let mut stream = llm.stream_chat(messages, None).await?;

while let Some(chunk) = stream.next().await {
    if let Ok(token) = chunk {
        print!("{}", token);  // 实时打印（打字机效果）
    }
}
```

### 06 安全提示

API Key 使用环境变量，不要硬编码在代码中：

```rust
// 正确做法
let api_key = std::env::var("OPENAI_API_KEY")?;

// 错误做法（不要这样做）
let api_key = "sk-xxxxxxxx";  // ❌ 禁止硬编码
```

### 07 检索模式选择

| 场景 | 推荐模式 |
|------|----------|
| 关键词精确匹配 | BM25 |
| 语义相似度搜索 | 向量检索 |
| 综合搜索、提高召回率 | Hybrid（推荐） |

---

## A2A Agent 协议 ✨ v0.4.1

[A2A](https://github.com/google/A2A)（Agent-to-Agent）是 Google 提出的 Agent 间互调协议。LangChainRust 实现了完整的 A2A 支持：Server 暴露 agent、Client 调远程 agent，JSON-RPC 2.0 风格通信。

### A2AServer（暴露 agent）

`A2AServer` 不启动自己的 HTTP 服务器，而是提供 handler 函数，让你插入任意 HTTP 框架。

```rust
use langchainrust::a2a::{A2AServer, AgentCard};
use langchainrust::LLMChain;
use std::sync::Arc;

let chain = Arc::new(LLMChain::new(llm, "You are a helpful assistant"));
let server = A2AServer::new(chain)
    .with_card(AgentCard::new("my-agent", "A helpful agent", "http://localhost:8080"));

// 在 axum/actix 路由里调用:
// GET  /.well-known/agent.json → server.get_agent_card()
// POST /                       → server.handle_a2a_request(body).await
```

**Task Persistence**: `tasks/send` 创建的任务存入内存 `RwLock<HashMap>`，`tasks/get` 可查询，`tasks/cancel` 可状态转换。生产环境可替换为数据库后端。

### A2AClient（调用远程 agent）

```rust
use langchainrust::a2a::{A2AClient, A2AMessage};

let client = A2AClient::new("http://remote-agent:8080".to_string());

// 发现 agent
let card = client.get_agent_card().await?;
println!("Agent: {}", card.name);

// 发送任务
let task = client.send_task(A2AMessage::user("hello")).await?;
println!("Task ID: {}, Status: {}", task.id, task.status);

// 查询任务
let task = client.get_task(&task.id).await?;

// 取消任务
let task = client.cancel_task(&task.id).await?;
```

---

## with_structured_output ✨ v0.4.1

`StructuredOutputExt` trait 让 LLM 一行输出强类型结构，按 provider 能力走 function calling 或 JsonOutputParser 降级。

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
// answer.city = "北京", answer.population = 21540000
```

**降级策略**：若 provider 不支持原生 structured output，自动降级为 JsonOutputParser + JSON mode。

---

## FileVectorStore ✨ v0.4.1

JSON 持久化向量存储，填补 InMemory（不持久）与外部数据库（太重）之间的空缺。

```rust
use langchainrust::{FileVectorStore, VectorStore, Document, MockEmbeddings};
use std::path::PathBuf;

let path = PathBuf::from("./vectors.json");
let store = FileVectorStore::new(path, 4)?;  // 4 维

// 添加文档
let docs = vec![
    Document::new("Rust 注重安全和性能").with_id("rust"),
    Document::new("Python 适合快速开发").with_id("python"),
];
let embeddings = vec![
    vec![1.0, 0.0, 0.0, 0.0],
    vec![0.0, 1.0, 0.0, 0.0],
];
let ids = store.add_documents(docs, embeddings).await?;

// 语义搜索
let query = vec![0.9, 0.1, 0.0, 0.0];
let results = store.similarity_search(&query, 2).await?;

// 持久化：文件已自动写入磁盘，重启后 new(path, dim) 即加载
store.clear().await?;
```

**特性**：原子写入(tmp+rename)、维度校验、跨实例持久化。

---

## ComputerUseTool ✨ v0.4.1

对标 Anthropic computer use，提供截图、鼠标点击、键盘输入能力。

```rust
use langchainrust::ComputerUseTool;
use std::sync::Arc;

// Anthropic API 模式（默认）
let tool = ComputerUseTool::new();

// 或 Native 模式（需 feature computer-use-native）
// let tool = ComputerUseTool::new_native();

// 作为 BaseTool 使用
let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(tool)];
```

---

## 版本信息

LangChainRust v0.4.1 | MIT License

