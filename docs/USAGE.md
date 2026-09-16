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
  - Qwen3-Embedding 与 matryoshka 维度 ✨ v0.21.0
  - LocalEmbeddings
  - CandleEmbeddings（纯 Rust 本地推理） ✨ v0.21.0
  - Token 级嵌入（TokenLevelEmbeddings） ✨ v0.21.0
  - Late Chunking（后分块） ✨ v0.21.0
- [提示词](#prompts)
  - FewShotPrompt + ExampleSelectors
  - 版本化提示词注册表（PromptRegistry） ✨ v0.22.4
- [输出解析器](#output-parsers)
- [记忆](#memory)
  - VectorStoreRetrieverMemory
  - MongoPersistentMemory ✨ v0.15.0
  - ContextWindow（长上下文管理） ✨ v0.4.1
  - 两层语义记忆（TwoTierMemory） ✨ v0.22.4
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
  - Agent Web SSE（浏览器事件流） ✨ v0.22.4
  - AgentBuilder ✨ v0.14.0
  - Orchestrator ✨ v0.14.0
  - Supervisor 子 Agent 动态路由 ✨ v0.24.0
  - ApprovalGate 图路径审批门 ✨ v0.24.0
  - 并行工具调用（信号量限流） ✨ v0.24.0
  - ToolPolicy ✨ v0.14.0
  - 上下文压缩（CompactionConfig） ✨ v0.21.0
- [Plan-Execute 智能体](#plan-execute-agent)
- [Handoffs](#handoffs)
- [流式工具调用](#streaming-tool-calls)
- [护栏](#guardrails)
  - Guardable ✨ v0.15.0
  - 流式护栏 ✨ v0.15.0
  - PII 脱敏护栏（改写而非阻断） ✨ v0.22.4
  - Schema 输出护栏（边生成边验 JSON） ✨ v0.22.4
  - 审计持久化 ✨ v0.15.0
  - Retrieval Rail（检索护栏） ✨ v0.21.0
  - AI 透明披露（disclose） ✨ v0.21.0
- [Token 计数器](#token-counter)
  - 成本台账（PricingTable + CostTracker + 美元硬闸门） ✨ v0.22.4
- [会话](#sessions)
  - 事件溯源重写 ✨ v0.22.0（推荐路径）
  - 会话分叉（fork）
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
  - Contextual Retrieval ✨ v0.21.0
  - 语义缓存（SemanticCache） ✨ v0.21.0
- [BM25](#bm25)
  - 小到大检索（句子窗口 / 父文档） ✨ v0.24.0
- [混合检索](#hybrid-retrieval)
  - 原生混合搜索（Qdrant Query API） ✨ v0.21.0
  - Weighted 加权融合 + MMR 多样性 ✨ v0.23.0
  - Late Chunking 双腿注入 ✨ v0.24.0
- [文档加载器](#document-loaders)
  - HTMLLoader
  - DocxLoader ✨ v0.4.1
  - WebScraperLoader ✨ v0.4.1
  - SitemapLoader ✨ v0.4.1
- [MultiQueryRetriever](#multiqueryretriever)
- [HyDE 检索器](#hyde-retriever)
- [SelfQueryRetriever](#selfqueryretriever) ✨ v0.18.0
- [重排序](#reranking)
  - 神经重排序（Cohere / Jina 交叉编码器） ✨ v0.24.0
- [回调](#callbacks)
  - OtelHandler
- [评估](#evaluation)
  - 评估器（10 种类型）
  - EvalRunner
  - LLMAsJudge ✨ v0.15.0
  - PairwiseJudge ✨ v0.15.0
  - trace → golden → 回归门禁 ✨ v0.24.0
- [LangGraph](#langgraph)
  - Reducer ✨ v0.15.0
  - 边类型 ✨ v0.15.0
  - Checkpointer 家族 ✨ v0.15.0
  - 子图 / 动态规划 / 流式 ✨ v0.15.0
  - 节点内动态中断 / 恢复（interrupt + resume） ✨ v0.24.0
  - 审批 / 恢复收敛（ApprovalGate） ✨ v0.24.0
  - 状态历史与时间旅行（fork_from） ✨ v0.24.0
- [A2A 智能体协议](#a2a-agent-protocol) ✨ v0.4.1
  - v1.0.1：多传输声明（supportedInterfaces） ✨ v0.22.0
  - 卡片签名（JWS HS256） ✨ v0.22.0
- [with_structured_output](#with_structured_output) ✨ v0.4.1
  - 原生 JSON Schema 引擎约束 ✨ v0.21.0
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
langchainrust = "0.24.0"
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
| **任意 OpenAI 兼容端点** | `OpenAICompatibleChat` | Groq / OpenRouter / xAI 预设 + 自建 vLLM、LM Studio、SGLang、内网网关(v0.22.4) |

#### 统一客户端与自动发现 ✨ v0.15.0

`LLMClient::from_env()` 自动识别 11 家 Provider 的环境变量,零配置切换;`LLMClient::from_llm(provider)` 手动包装任意 `BaseChatModel`。`ChatModelWrapper` / `wrap_chat_model` 提供 trait 对象包装。

```rust
use langchainrust::LLMClient;
use langchainrust::language_models::ProviderError;

// 自动探测:哪个 Provider 配了环境变量就用哪个
let llm = LLMClient::from_env()?;
let response = llm.chat(vec![Message::human("Hello")], None).await?;

// 原生 Provider 也可以被包装
let client = LLMClient::from_llm(DeepSeekChat::from_env_result()?);
```

错误类型统一为 `ProviderError`,按供应商区分变体(OpenAI / Anthropic / Gemini / Azure / Cohere / Ollama / DeepSeek / Qwen / Moonshot / Zhipu / Mistral),`config.streaming` 决定 `chat()` 走流式还是普通路径。

#### DeepSeek（高性价比）

```rust
use langchainrust::{DeepSeekChat, DeepSeekConfig, BaseChatModel};
use langchainrust::schema::Message;

// 从环境变量读取
let llm = DeepSeekChat::from_env_result()?;

// 或手动指定模型(配置 builder + new)
let llm = DeepSeekChat::new(DeepSeekConfig::from_env_result()?.with_model("deepseek-chat"));

let response = llm.chat(vec![
    Message::human("Explain Rust ownership"),
], None).await?;
```

#### OpenAI 兼容端点:Groq / OpenRouter / xAI / 自建网关 ✨ v0.22.4

**解决什么问题**:OpenAI Chat Completions 协议已经成了行业事实标准——Groq、OpenRouter、xAI 以及自建的 vLLM、LM Studio、SGLang、Ollama 的 `/v1` 垫片、公司内网网关全都讲这套协议。旧做法是每接一家就复制一份约 300 行的委托代码(DeepSeek/Qwen/Moonshot 就是这么来的),厂商出新旗舰模型还得等框架升级。v0.22.4 新增 `OpenAICompatibleChat`:**一个传输层通吃所有兼容端点,厂商之间只差配置**(base URL、鉴权、默认模型、错误标签);流式、函数调用、结构化输出、重试和回调全部复用 `OpenAIChat` 的成熟实现。

托管路由用预设,`GroqChat` / `OpenRouterChat` / `XaiChat` 都只是 `OpenAICompatibleChat` 的类型别名:

```rust
use langchainrust::{OpenAICompatibleChat, BaseChatModel};
use langchainrust::schema::Message;

// 三种环境变量构造器:预设各自的 key / model / base_url 环境变量
let groq    = OpenAICompatibleChat::groq_from_env()?;        // GROQ_API_KEY
let router  = OpenAICompatibleChat::openrouter_from_env()?;  // OPENROUTER_API_KEY + OPENROUTER_MODEL
let xai     = OpenAICompatibleChat::xai_from_env()?;         // XAI_API_KEY
let reply = groq.chat(vec![Message::human("hi")], None).await?;
```

| 预设 | Base URL | 环境变量 | 默认模型 |
|---|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY`(必填)、`GROQ_MODEL`、`GROQ_BASE_URL`(可指向镜像) | `llama-3.3-70b-versatile` |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY`、`OPENROUTER_MODEL`(**必填**,无默认)、可选 `OPENROUTER_SITE_URL`+`OPENROUTER_SITE_NAME` 归因 | 无——模型必须是 `anthropic/claude-sonnet-4-5` 这样的厂商限定 id |
| xAI | `https://api.x.ai/v1` | `XAI_API_KEY`(必填)、`XAI_MODEL`、`XAI_BASE_URL` | `grok-4` |

另有常量 `GROQ_MODELS` / `XAI_MODELS` 只是**非穷举的提示清单**——model 字段接受端点能服务的任意字符串,新模型不用等框架更新;`DEFAULT_GROQ_MODEL` / `DEFAULT_XAI_MODEL` 是默认 id。OpenRouter 的归因头(`HTTP-Referer`、`X-Title`,显示在它的分析/排行榜上)可用 `config.with_openrouter_attribution(site_url, site_name)` 手工设置。

自建 / 私有端点走通用构造器,一个重要的默认行为是 **keyless**:

```rust
use langchainrust::OpenAICompatibleConfig;

// 本地 vLLM / LM Studio:默认不带 key,也根本不发 Authorization 头
//(不会发出 "Authorization: Bearer " 这种空头)
let local = OpenAICompatibleChat::new(
    OpenAICompatibleConfig::new("http://127.0.0.1:8000/v1/", "local-model"),
);

// 需要鉴权的私有网关:显式加 key 和任意请求头(租户头、网关注入等)
let gw = OpenAICompatibleChat::new(
    OpenAICompatibleConfig::new("https://gw.internal/v1", "qwen2.5-72b")
        .with_api_key("sk-xxx")
        .with_extra_header("X-Tenant", "acme"),
);
```

- base URL 结尾的 `/` 会被归一化去掉(避免拼出 `/v1//chat/completions`);通用 env 构造器读 `OPENAI_COMPATIBLE_BASE_URL`(必填)、`OPENAI_COMPATIBLE_MODEL`(必填)、`OPENAI_COMPATIBLE_API_KEY`(可选,不设即 keyless)。
- 出错时错误统一包成 `ProviderError::OpenAICompatible { provider, source }`,`provider` 是端点名(`"groq"` / `"openrouter"` / `"xai"` / 通用的 `"openai-compatible"`),可用 `provider_label()` 读取——一家端点故障不会和别的端点混淆。
- 能力面:`bind_tools(...)` / `with_tool_choice(...)` 函数调用;`with_structured_output::<T>()` 走 strict 工具绑定(任何兼容后端都能用);`with_json_schema_output::<T>()` 走服务端 `response_format: json_schema`(需要后端支持,不支持的模型回 4xx,显式报错不静默降级)。
- 安全细节:`Debug` 打印配置时 key 显示为 `***`,不会误打进日志。

**边界**:它只实现 Chat Completions 协议;厂商特有能力(OpenAI Responses API、Anthropic extended thinking 等)还得用各自的原生类型。它是完整的 `BaseChatModel`(trait 错误类型为 `ProviderError`),可以直接塞进 agent / 链。

#### Moonshot（长上下文）

```rust
use langchainrust::MoonshotChat;

let llm = MoonshotChat::with_model("moonshot-v1-128k")?;  // 128K 上下文,从环境变量读 key

let response = llm.chat(vec![
    Message::human("Analyze this long document..."),
], None).await?;
```

#### Qwen

```rust
use langchainrust::QwenChat;

let llm = QwenChat::from_env_result()?;  // 或 QwenChat::new(QwenConfig::from_env_result()?.with_model("qwen-plus"))

let response = llm.chat(vec![
    Message::human("Explain microservices in Chinese"),
], None).await?;
```

#### Zhipu（ChatGLM）

```rust
use langchainrust::ZhipuChat;

let llm = ZhipuChat::with_model("glm-4")?;  // 关联函数:从环境变量读 key 并指定模型;另有 from_env_result()

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

> **截断流不再被当成完整回答（✨ v0.24.0）**：Ollama 曾是唯一不检查终止标记的 chat provider——字节流提前 EOF（本地模型进程崩了 / 连接被切断）时会带着半截内容**成功返回**。v0.24.0 起解析器跟踪 `saw_terminal`（`[DONE]` 或 `choice.finish_reason`），EOF 前没见到终止标记就返回 `OllamaError::StreamInterrupted(String)`（`#[non_exhaustive]` 新变体，携带已收到的部分内容），与 OpenAI / Azure / Anthropic / Gemini 的契约对齐。升级时注意为这个新错误变体补一个 match 分支。

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
    "以下是反义词示例:",            // prefix:示例前的引导语
    "Input: {input}\nOutput:",      // suffix:真正的问题模板,必须用到 input_variables
    vec!["input".to_string()],      // input_variables:后缀中用到的变量
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

### 版本化提示词注册表(PromptRegistry) ✨ v0.22.4

**解决什么痛点:** prompt 是整个系统里调得最勤的"代码",但散落在各文件里的 `&str` 常量让三件事做不到——线上回答突然变差,查不出是哪次改 prompt 导致的;想回滚必须重新发版;多个组件共享一份字符串,谁都能悄悄改掉,正在跑的评测无法钉在"当时那一版"上。`PromptRegistry` 给提示词一个**只增不改、可寻址、可审计**的登记处。

**机制:命名空间 + 版本号 + 内容哈希,三位一体。**

- **命名空间**:斜杠分隔的路径(如 `"customer/support/zh"`),每段非空、字符限 `[A-Za-z0-9._-]`、总长 ≤128,非法直接 `PromptsError::InvalidNamespace`;
- **版本号**:同一命名空间下从 1 开始单调递增;
- **内容哈希**:每次注册对模板**原始字节**算 SHA-256,`RegisteredPrompt { namespace, version, hash, template, variables }` 里同时带回版本号与哈希,`variables` 在注册时就解析好;
- **不可变**:已存版本永不修改。回滚不是覆盖,而是把历史内容**作为新版本重新发布**——钉在旧版本号上的消费方不受影响,审计链不断;
- 线程安全(`RwLock`),所有操作都是 `&self`,多组件 `Arc` 共享一份即可。

```rust
use langchainrust::{PromptRegistry, VersionSpec};

let registry = PromptRegistry::new();
let v1 = registry.register("customer/support/zh", "你是客服。问题:{question}")?;
let v2 = registry.register("customer/support/zh", "你是资深客服,先共情再回答。问题:{question}")?;
assert_eq!(v1.version, 1);
assert_eq!(v1.hash.len(), 64);                 // SHA-256 十六进制
assert_eq!(v1.variables, vec!["question"]);   // 注册时解析占位符

// 三种定位方式:
let latest = registry.get("customer/support/zh")?;                 // 最新版
let pinned = registry.resolve("customer/support/zh", &1.into())?;  // 钉版本号(VersionSpec::Number)
let by_hash = registry.resolve("customer/support/zh", &v1.hash.as_str().into())?; // 按内容哈希
assert_eq!(pinned.template, v1.template);

// 文本引用:适合写在配置/数据库里,运行时再解析
let tpl = registry.fetch("customer/support/zh@3")?;          // 直接拿可运行的 PromptTemplate
let _  = registry.resolve_ref("customer/support/zh")?;       // 无 @ 等同 @latest
let _  = registry.resolve_ref("customer/support/zh@latest")?;
let _  = registry.resolve_ref("customer/support/zh@hash:9f2a1c0")?; // 或裸十六进制前缀
```

**寻址规则与失败方式(都是显式错误,不静默猜):**

| 选择器 | 行为 |
|---|---|
| `VersionSpec::Latest` | 当前最新版 |
| `VersionSpec::Number(n)` | 精确版本;不存在 → `VersionNotFound` |
| `VersionSpec::Hash(prefix)` | 全哈希或**唯一前缀**;前缀短于 7 字符或零命中 → `HashNotFound`;前缀命中多个版本 → `AmbiguousHash { prefix, versions }`;回滚重新发布的同内容哈希解析到**最新的那个同内容版本**(内容一致) |

**两个刻意的语义:**

1. **连续注册相同内容是空操作**——与当前 latest 文本一致时返回已有记录、不涨版本号(防止发布脚本每天 bump 一堆无意义版本);但历史内容在变更之后再次出现(即回滚)是合法的新版本。
2. **回滚即重新发布**:`let v3 = registry.rollback("customer/support/zh", 1)?;` 得到的是版本 3、内容同版本 1;若目标本来就是当前最新内容则原样返回。

配套盘点 API:`history(ns) -> Vec<PromptVersionInfo { version, hash }>`(按注册顺序,给审计/管理 UI 用)、`namespaces()`、`len()` / `is_empty()`;拿到的 `RegisteredPrompt::to_template()` 生成新的 `PromptTemplate`,直接 `.format(&vars)?` 或 pipe 进 LCEL 链。

**边界:** 注册表是**纯内存**的,进程退出即丢失——它解决的是单进程内的版本治理与寻址,不自带持久化、远程分发或鉴权;需要跨进程共享时,由应用侧把模板文本(及哈希)存数据库,启动时灌进注册表。

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

记忆给链或智能体加"上下文":让多轮对话记住前面说了什么,而不用每次把整段历史塞进提示词。内置的对话记忆有四种,另有向量检索记忆,各有取舍:

| 记忆 | 行为 | 适合 | 代价 |
|------|------|------|------|
| `ConversationBufferMemory` | 保留全部对话 | 短对话、信息不能丢 | token 随轮次线性增长 |
| `ConversationBufferWindowMemory` | 只留最近 k 轮 | 长对话、旧细节不重要 | 旧内容直接丢弃 |
| `ConversationSummaryMemory` | 全程只维护一份 LLM 摘要 | 长对话、只要梗概 | 每次写入多一次 LLM 调用,丢细节 |
| `ConversationSummaryBufferMemory`(推荐) | 旧消息摘要 + 近期原文 | 长对话且要近期细节 | 摘要消耗一次 LLM 调用 |
| `VectorStoreRetrieverMemory` | 按相似度检索记忆 | 知识型、联想式记忆 | 需要向量库 + 嵌入模型 |

- 想跨进程持久化 → 用 `MongoPersistentMemory`(见下方);
- 想限制单次上下文长度 → 用 `ContextWindow` 自动截断/摘要;
- 想跨会话记住"关于用户的事实"(偏好、身份、项目背景)而不是对话原文 → 用两层语义记忆 `TwoTierMemory`(v0.22.4,见本章末);
- 所有对话记忆实现统一 `BaseChatMemory` trait,可即插即换。

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

### 两层语义记忆:跨会话的"事实记忆" ✨ v0.22.4

上面的所有记忆管的都是**对话历史**——下一条 prompt 里要放哪些消息。v0.22.4 在 `lc-memory` 里新增了第二套抽象,管的是**知识**:从已完成的对话轮次中提炼出的、关于用户/任务的**持久事实**("用户偏好 Rust"、"用户在做 langchainrust 项目"),按命名空间隔离、按含义而不是按消息时间线召回。

**它解决的痛点**:对话窗口装不下跨会话事实。会话摘要记忆压缩的是"聊了什么",事实会随摘要轮次损耗;而且摘要只在本会话内有效。语义记忆把"值得长期记住的事实"单独存一份,新会话开始时按当前问题的语义召回相关条目注入 prompt——这正是生成式智能体(generative agents)类系统的记忆模型。

#### 统一接口 MemoryStore

三种存储(`ShortTermMemory` / `LongTermMemory` / `TwoTierMemory`)实现同一个 trait,都可以 `Arc` 共享给多个执行器:

```rust
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn put(&self, namespace: &str, item: MemoryItem) -> Result<(), MemoryError>;
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<MemoryItem>, MemoryError>;
    async fn search(&self, query: &MemoryQuery<'_>) -> Result<Vec<MemoryHit>, MemoryError>;
    async fn forget(&self, namespace: &str, key: &str) -> Result<bool, MemoryError>;
    async fn clear_namespace(&self, namespace: &str) -> Result<usize, MemoryError>;
    async fn len_namespace(&self, namespace: &str) -> Result<usize, MemoryError>;
}
```

- **命名空间是硬隔离边界**:通常一个用户(或一个会话)一个 namespace,读写和语义召回绝不跨命名空间;namespace、key、text 任一为空白都会显式报错。
- `MemoryItem::new(key, text)`:同 namespace 下 key 要稳定——重复 put 同一个 key 是**更新**而不是新增(刷新内容和访问时间)。默认 importance 0.5,可用 `.with_importance(0.0..=1.0)`,写入时会被 clamp 到 [0,1];还可带 `metadata: HashMap<String,String>`。
- `MemoryQuery::new(namespace, text)` 默认 `k = 5`、`min_score = 0.0`;链式 `.k(10)`(最小钳到 1)、`.min_score(0.3)` 可调。
- 返回的 `MemoryHit { key, text, score, importance, tier: MemoryTier, metadata }` 标明命中来自 `MemoryTier::Short` 还是 `Long`。
- `get` / 被 `search` 命中都算**一次访问**(刷新 `last_access_at`、`access_count +1`)——这是后面晋升机制的计数来源。

#### 相似度从哪来:SemanticScorer 与零依赖默认值

```rust
#[async_trait]
pub trait SemanticScorer: Send + Sync {
    async fn similarity(&self, query: &str, document: &str) -> f64; // [0,1]
}
```

默认实现 `LexicalScorer` **不调用任何模型**:对两段文本做 Unicode 感知分词、算词频向量的 cosine 相似度,完全离线、确定、零额外依赖。这意味着开箱即用的"语义"召回实质上是**词汇重合度**匹配——问"退款政策"能召回含"退款""政策"字样的事实,但召不回只写了"退货多久能到账"的近义条目。想要真正的向量语义,自己实现 `SemanticScorer`(内部调你的嵌入模型),通过 `ShortTermMemory::with_scorer(capacity, Arc::new(scorer))` 或 `LongTermMemory::with_config(weights, Arc::new(scorer))` 注入,存储层不用改一行。

#### 两层各是什么

| | 短期 `ShortTermMemory::new(capacity)` | 长期 `LongTermMemory::new()` |
|---|---|---|
| 容量 | 每个命名空间有界,满了按 **FIFO 淘汰最旧条目**(同 key 更新不占新槽) | 无界,读路径上**从不删除**,只是打分下沉 |
| 排序 | 纯相似度 | 三因子加权衰减分(见下) |
| 定位 | 本进程内的工作记忆 | 沉淀下来的持久事实 |

长期层的打分公式(`DecayWeights`,生成式智能体的经典做法):

```
score = 0.70 · 相似度 + 0.15 · 2^(−age / 半衰期 · ln2) + 0.15 · 重要度
```

- 默认权重 **相似度 0.7 / 近因 0.15 / 重要度 0.15**,近因半衰期 **7 天**(距上次访问每过 7 天,近因项减半);权重不必求和为 1,最终分会 clamp 到 [0,1]。
- 可调:`DecayWeights::new().with_weights(0.6, 0.2, 0.2).with_half_life(Duration::from_secs(3*24*3600))`。
- 效果:常被召回的事实持续"保鲜",高重要度的旧事实仍能被翻出来,又老又没人碰的事实自然沉底。

#### TwoTierMemory:组合两层 + 晋升

```rust
use langchainrust::memory::{TwoTierMemory, MemoryItem, MemoryQuery, PromotionPolicy};
use std::sync::Arc;

let mem = Arc::new(
    TwoTierMemory::new(64)  // 短期层每命名空间容量
        .with_policy(
            PromotionPolicy::new()              // 默认:重要度 ≥ 0.8 或被访问 ≥ 3 次
                .with_min_importance(0.7)
                .with_min_access_count(5),
        ),
);

// 写入永远先进短期层
mem.put("user-42", MemoryItem::new("lang_pref", "用户偏好 Rust 而非 Python").with_importance(0.9)).await?;

// 手动触发晋升(按命名空间或全部):达标的短期条目移入长期层并从短期删除
let promoted = mem.consolidate_namespace("user-42").await?;

// get:先查短期,没有再查长期;search:两层一起用同一把"长期公式"打分、按 key 去重(短期优先),统一排序
let hits = mem.search(&MemoryQuery::new("user-42", "他喜欢什么语言").k(5)).await?;
```

晋升是 **OR 条件**:重要度达标(写入时就被认定的高价值事实)**或**反复被访问(用出来的高价值事实),任一满足即晋升。重复晋升不会产生重复条目——长期层按 key 合并:保留最早创建时间、累加访问次数、重要度取最大、文本更新为最新、metadata 合并。

#### 与 AgentExecutor 接线:回忆在前、抽取在后

`lc-agents` 提供了 LLM 事实抽取器 `LlmMemoryExtractor`(facade 顶层直接导出),并在执行器上留了一个开关,**默认关闭**:

```rust
use langchainrust::{AgentExecutor, LlmMemoryExtractor};
use langchainrust::memory::TwoTierMemory;

let extractor = Arc::new(LlmMemoryExtractor::from_model(extraction_llm)); // 或 ::new(Arc::new(llm))
let executor = executor.with_semantic_memory(
    Arc::new(TwoTierMemory::new(64)), // 可被多个执行器共享的存储
    "user-42",                        // 命名空间,建议用用户 id
    extractor,
);
```

接线后每次运行的行为:

1. **规划前回忆(best-effort)**:用用户输入在该命名空间召回 top 5 事实,格式化成带分隔的清单注入运行 `inputs["semantic_memory"]`;想让事实出现在 prompt 里,提示词模板需要留 `{semantic_memory}` 占位符。注入文本自带 "treat as untrusted data" 提示——召回内容是数据不是指令。存储出错只 warn、降级为"无记忆",**绝不让记忆故障拖垮主流程**。
2. **回答后抽取(后台 detached task)**:`tokio::spawn` 一个脱离调用方的任务,让抽取模型把这轮 user/assistant 对话蒸馏成 JSON 事实数组 → put 进短期层 → 立即对该命名空间做一次晋升整理。调用方拿到答案后**不等抽取**,慢模型/网络抖动不会延迟回答;任务内任何失败只记日志。

`LlmMemoryExtractor` 的容错设计值得知道:模型被要求只回一个 JSON 数组(`{key, text, importance}`),但解析器容忍围栏代码块和前后散文(走 `parse_llm_json`),也接受 `{"memories":[...]}` / `{"items":[...]}` 信封;单条坏数据跳过不致命;缺 importance 默认 0.5;缺 key 时由事实文本归一化派生稳定 key(同一事实抽两次只更新一条);每轮最多收 10 条防失控;assistant 输出为空白时直接短路、不调模型。

#### 边界与注意

- **纯内存,没有内置持久化**:三个存储都是进程内结构(`Arc` 可跨任务共享),drop 即遗忘。跨进程/重启保留需要自己实现 `MemoryStore`(接 Mongo 等),或在应用层定期快照。
- **默认打分是词面匹配不是向量语义**;要嵌入语义请自备 `SemanticScorer`。
- 接执行器后**每轮多一次抽取模型调用**(费用/延迟),虽然在后台任务里不挡回答;不需要时别开 `with_semantic_memory`。
- 召回注入的是模型外部内容,已按不可信数据处理,但你的模板应避免把该块放进可执行指令位置。

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

// primary 失败时自动切换到 fallback(类型相同的 Runnable 直接传,内部自动抹类型)
let chain = primary.with_fallbacks(vec![fallback]);
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

> **边界说明**:其余 Provider(非 OpenAI/Qwen/DeepSeek)维持 `LLMClient` 收口;不做 Rust `|` 运算符重载(用 `.pipe()`);Retriever 的 `Runnable<String, Vec<Document>>` 适配器(`RetrieverRunnable`)已在 v0.17.0 补齐。

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

### Agent Web SSE：把 Agent 事件直接推到浏览器 ✨ v0.22.4

`AgentStreamEvent` 本来只活在进程内:要在网页上实时显示"打字机正文 + 工具调用进度",你得自己把事件流转成 `text/event-stream`——事件命名、单调 id、空闲心跳、断线重连去重、客户端断开后取消运行,每一样漏了都是坑。v0.22.4 把这套 SSE 桥做进了框架,分**无依赖的帧层**和 **feature 门控的 axum 服务层**两部分。

**第一层:框架中立的成帧器(零 HTTP 依赖,默认可用)**

`agent_sse_frames()` 吃进任意 `AgentStreamEvent` 流,吐出已经编号、带心跳、带终结帧的 `SseFrame` 流;`encode_sse_frame()` 负责渲染成精确的 SSE 线格式。actix / rocket / 手写 hyper service 都能直接拿来发:

```rust
use langchainrust::agents::{agent_sse_frames, encode_sse_frame, SseOptions};
use std::time::Duration;

// invoke_stream 返回 Pin<Box<dyn Stream<Item = AgentStreamEvent> + Send>>
let events = streaming_agent.invoke_stream(user_input).await;

let frames = agent_sse_frames(
    events,
    SseOptions::default()
        .with_heartbeat(Duration::from_secs(20)) // 空闲期发 ": keep-alive" 注释帧
        .with_retry(Duration::from_secs(3))     // 首帧带 retry: 重连提示(毫秒)
        .with_resume_from(last_event_id),       // 抑制 id ≤ 该值的帧(Last-Event-ID)
);
tokio::pin!(frames);
while let Some(frame) = frames.next().await {
    let bytes = encode_sse_frame(&frame); // 已是 "id:..\nevent:..\ndata:..\n\n"
    // 写入你自己的 HTTP 响应体
}
```

**线上事件契约**(`sse_event_name` / `sse_event_payload` 定义,字段稳定可依赖):

| SSE `event:` | data(JSON,紧凑单行) |
|---|---|
| `text` | `{"content": "..."}` 逐 token 正文 |
| `tool_call` | `{"state": "started"\|"arguments_streaming"\|"arguments_complete"\|"executing"\|"completed"\|"failed", "tool_name", "call_id", ...}`,不同 state 再带 `partial_args` / `args` / `result` / `error` |
| `tool_start` / `tool_end` | `{"name","input"}` / `{"name","output"}` |
| `pipeline_step` | `{"step","detail": string\|null}`(CRAG/AdaptiveRAG/DeepResearch 的进度) |
| `final_answer` | `{"content": "..."}` 完整答案 |
| `error` | `{"message": "..."}` 执行出错 |

每个内容帧带从 **1 开始的单调 `id:`**;运行结束必发一个**不带 id** 的 `done` 帧(`data:{"status":"done"}`)——出错也先发 `error` 再发 `done`,客户端只需认 `done` 收尾。`done` 故意无 id,避免重连时把"新一次运行"误判成旧运行的延续。

**第二层:`sse-server` feature 下的 axum 0.7 路由(开箱即服务)**

lc-agents 的 `sse-server` feature 提供现成路由器,`POST` 与 `GET` 同一条路径两种姿势都支持——浏览器原生 `EventSource` 只能 GET,程序化 `fetch()` 客户端适合 POST 长 prompt:

- `POST /agent/stream`:body 为 `{"input":"...", "last_event_id": 12?}`;
- `GET /agent/stream?input=...&last_event_id=...`:给 `new EventSource(url)` 用;
- 空白 input 返回 400,畸形 JSON 在开流之前就 4xx。

```rust
// 需要在 lc-agents 上启用 feature: sse-server(axum 0.7 + tower-http + tower)
use langchainrust::agents::{
    serve_agent_sse_on, agent_sse_router, AgentSseServerConfig,
    AgentStreamFactory, StreamingFunctionCallingAgent,
};
use std::{sync::Arc, time::Duration};

let agent = Arc::new(StreamingFunctionCallingAgent::new(chat_model));

// 每个 HTTP 请求开一条全新的事件流;Fn(String) -> impl Future 自动实现 AgentStreamFactory
let factory: Arc<dyn AgentStreamFactory> = Arc::new(move |input: String| {
    let agent = agent.clone();
    async move { agent.invoke_stream(input).await }
});

// 路由器可挂进你自己的 axum Router;心跳默认 20s(代理常 30–60s 切空闲连接)
let router = agent_sse_router(factory.clone());
// 或自定义:心跳间隔 / 关心跳(with ZERO)/ retry 提示
let router = agent_sse_router_with(factory.clone(), AgentSseServerConfig::default()
    .with_heartbeat(Duration::from_secs(15))
    .with_retry(Duration::from_secs(3)));

// 直接起服务:serve_agent_sse 绑 0.0.0.0:port;想绑回环/TLS/unix socket/随机测试端口,
// 自己 TcpListener::bind 后用 serve_agent_sse_on
serve_agent_sse_on(factory, listener).await?;
```

前端消费(GET 路由,原生 EventSource):

```javascript
const es = new EventSource(`/agent/stream?input=${encodeURIComponent(prompt)}`);
es.addEventListener('text', e => append(JSON.parse(e.data).content));
es.addEventListener('tool_call', e => renderTool(JSON.parse(e.data)));
es.addEventListener('final_answer', e => finish(JSON.parse(e.data).content));
es.addEventListener('error', e => showError(JSON.parse(e.data).message));
es.addEventListener('done', () => es.close());
```

**断线与重连的语义边界(设计上刻意保守):**

- **断开即取消**:HTTP 客户端消失 → 响应流被 drop → 成帧任务停止轮询 agent → agent 生产者下次发送即观察到通道关闭,运行被及时取消,不会"前台关了页面,后台还在烧 token"。
- **不做跨连接运行续传**:端点是无状态的,重连等于开一次**新运行**。客户端可带 `Last-Event-ID` 请求头(POST 也可放 body 的 `last_event_id` 字段,**头优先**),新运行里 id ≤ 该值的帧被抑制——这是给"会去重的客户端"用的,不承诺服务端能倒带重放一次已经结束的运行。

**安全边界:**

- 内置 CORS 层只放行 `http://localhost*` / `http://127.0.0.1*` 来源(方法 GET/POST/OPTIONS),其他浏览器来源拿不到 allow 头;正式部署请自行收紧或替换。
- **端点本身没有任何鉴权**。`serve_agent_sse` 默认绑 `0.0.0.0`,等于同网段可达者皆可用;要暴露到远端,请放到带鉴权的反向代理后面,或自己绑 listener(回环 / TLS)。
- 成帧器类型(`agent_sse_frames` / `SseFrame` / `SseOptions` 等)随 `lc-agents` 默认导出,`langchainrust::agents::*` 直接可见;**axum 服务件只在 `sse-server` feature 后**——facade crate 自己只在 dev-dependency 里开它(供示例编译),普通依赖不背 axum,下游需要服务件时在自己的 `lc-agents` 依赖上开 feature。
- 可运行的完整示例:`crates/lc/examples/agent_sse_server.rs`(`cargo run -p langchainrust --example agent_sse_server`,默认 `127.0.0.1:8090`,host/port/model 走 `AGENT_SSE_*` 环境变量)。

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

### Supervisor：LLM 动态路由子 Agent ✨ v0.24.0

**解决什么问题**：`SequentialPipeline` 的阶段表是写死的,`FanOutFanIn` 每轮都向全员广播——两者都不会根据任务内容决定"下一步该谁干"。多专才 Agent 场景(检索员 / 编码员 / 写手 / 审查员)需要的是一个**路由模型每轮决定**下一个子任务交给哪个具名 worker、或者判定工作已完成。v0.24.0 新增 `Supervisor`:每个 worker 是 `Orchestrator` trait 后面一个完整独立的 Agent(自己的执行器、预算、hooks),worker 的回答回填进路由模型的 scratchpad,下一轮路由基于前序 worker 的真实产出。

```rust
use langchainrust::{AgentTask, Orchestrator, RunContext, Supervisor, TaskAdapter};
use std::sync::Arc;

// 任意 String -> String 的 Orchestrator(AgentExecutor、其他 pipeline……)
// 经 TaskAdapter 适配成 AgentTask -> String 的 worker。
let researcher: Arc<dyn Orchestrator<Input = AgentTask, Output = String>> =
    Arc::new(TaskAdapter::new(Arc::new(research_executor)));
let writer = Arc::new(TaskAdapter::new(Arc::new(writer_executor)));

let supervisor = Supervisor::new(
    router_llm,                                         // String -> String 的路由模型
    vec![
        ("researcher".to_string(), researcher),
        ("writer".to_string(), writer),
    ],
    8,                                                  // 最大委派轮数(max_rounds)
);

let answer = supervisor
    .run_with_context(
        AgentTask::new("写一页 RAG 重排序技术简报"),
        &RunContext::new_random(),
    )
    .await?;
```

契约与边界:

- **只有一层**子 Agent 递归:worker 是叶子 orchestrator,不能再继续委派;
- 路由受 `max_rounds` 约束(构造后可用 `.with_max_rounds(n)` 调整,小于 1 自动夹到 1)。模型始终不发 `SUPERVISOR_FINISH` 时运行**报错而不是无限循环**;
- 决策协议:优先解析 JSON `{"next":"<worker>","task":"..."}` / `{"next":"FINISH","answer":"..."}`;弱模型输出不规整时回退 `<<<NEXT>>>` 分隔符格式——解析在 `parse_supervisor_decision`,提示词信封在 `supervisor_envelope(objective, worker_names, round, history)`,`SUPERVISOR_FINISH` 常量即 `"FINISH"`;
- 入口是 trait 方法 `run_with_context(input, &RunContext)`(不是固有方法 `.run`);`RunContext::new(id)` / `new_random()` 携带运行 id,`AgentTask::new(objective)` 还可链式挂期望产出与允许工具清单。

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
    max_cost_usd: Some(2.50),                          // v0.22.4:累计美元成本上限(按 CostTracker 实时台账计价)
    ..Default::default()
});
```

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

### 上下文压缩（CompactionConfig）✨ v0.21.0

长会话跑几十轮后，历史消息会把上下文窗口撑爆。上下文压缩让 executor 在**每轮 `plan()` 之前**检查触发条件，按 turn 边界裁剪历史（不会产生"孤儿 tool result"——工具输出和它的调用总是同进同出）。**默认不压缩**：不配置 `with_compaction` 就是零行为变化。

```rust
use langchainrust::agents::executor::{CompactionConfig, CompactionStrategy, CompactionTrigger};
use langchainrust::AgentExecutor;

let executor = AgentExecutor::new(agent, tools)
    .with_compaction(
        CompactionConfig::new(
            CompactionTrigger::TurnCount(20),              // 满 20 turn 触发
            CompactionStrategy::SlidingWindow { keep_recent_turns: 8 }, // 保留最近 8 turn
        )
        .with_min_recent_turns(2),                          // 压缩后至少保留 2 turn,默认 2
    );
```

**触发器**：`TurnCount(n)` / `TokenCount(n)` / `Any(a, b)` / `All(a, b)` 组合。
**策略**：`SlidingWindow { keep_recent_turns }`（保留最近 n turn）或 `TokenBudget { max_tokens, keep_recent_turns }`（超预算时从旧往新丢，直到塞进预算）。

**关键行为**：invoke 与 stream 双路径同语义；压缩次数计入 `AgentMetrics.compactions`（serde default,旧指标载荷兼容）；`min_recent_turns` 下限防止把会话压空。

## Plan-Execute Agent

**解决什么问题**：普通单循环 Agent（`FunctionCallingAgent` / `ReActAgent`）适合"一步能想清楚"的任务——想一步、干一步、再看结果。但像"先调研、再写代码、最后解释要点"这类复杂多步骤任务，模型一步想不出完整方案，直接动手又容易走偏。Plan-Execute Agent 把大任务拆成"先规划 → 逐步执行 → 失败重规划"的循环：先用 LLM 把任务拆成若干可执行步骤，每步交给一个单循环 Agent 执行，某一步失败就重新规划（而不是硬着头皮继续），全部完成后总结出最终结果。适用于复杂、多步骤、允许中途调整计划的任务。

> 注意:每个步骤通过 `FunctionCallingAgent` + 工具执行;`llm` 可以是任意 `BaseChatModel`(错误需能 `Into<ProviderError>`,v0.14 起执行器也可用 `agent_factory` 换成别的单循环 Agent,不再写死)。

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

### 并行工具调用（有界并发） ✨ v0.24.0

模型一轮返回多个 tool call 时,执行器**并发**执行它们,但并发度由信号量封顶(`AgentExecutor::with_max_concurrency`,默认 `DEFAULT_MAX_CONCURRENCY = 8`),而不是来多少起多少任务——对外部 API 配额、数据库连接和本机资源都可控:

```rust
let executor = AgentExecutor::new(agent, tools)
    .with_max_concurrency(4);   // 单轮最多 4 个工具调用同时在飞
```

工具观察值按**模型给出的调用顺序**(而非完成先后)zip 回 actions,因此多工具轮次的喂回序列是确定的;`invoke` 与流式两条路径给出同样的顺序保证。

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

`StreamingOutputGuardrail` trait 的 `validate_chunk(&ChunkContext) -> ChunkAction::{Pass, Replace, Block}`(v0.22.4 起入参收敛为携带分块与上下文的 `ChunkContext`)配合 `GuardedAgent::invoke_stream` 做两阶段检查:增量关键词检查 + 24 字符滑动窗口(防止跨块切断)+ 完整输出复查。有状态护栏(如扣留式脱敏)还可通过流末 flush 钩子吐出缓冲的尾部,避免尾段被丢弃。

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

### PII 脱敏护栏:改写而不是掐断 ✨ v0.22.4

`SensitiveInfoGuardrail` 的处置是 **Block**——输出里只要出现疑似 PII,整段回答被拦下。但客服工单摘要、对话转述这类场景要的是"内容照流、PII 消失":v0.22.4 新增的 `PiiRedactionGuardrail` 走输出侧 `Modify` 通道,把命中的标识符替换成固定标记后放行。

| `PiiKind` | 识别内容 | 替换为 | 误报防护 |
|---|---|---|---|
| `Email` | 邮箱 | `[REDACTED_EMAIL]` | 词边界 + 域名形态 |
| `Phone` | 中国大陆手机号 `1[3-9]` 开头 11 位,含 `+86`/`86`/连字符前缀 | `[REDACTED_PHONE]` | 词边界,长数字串中的 11 位子串不命中 |
| `NationalId` | 18 位身份证号(末位可为 X) | `[REDACTED_NATIONAL_ID]` | **GB 11643 加权校验码**,校验不过原样保留 |
| `CreditCard` | 15–19 位银行卡/信用卡(允许空格/连字符分隔) | `[REDACTED_CREDIT_CARD]` | **Luhn 校验**,不过校验不动 |
| `IpV4` | 点分 IPv4 | `[REDACTED_IP]` | 四段每段 ≤255,且排除邻接数字/点(版本号等) |

检测按"最长最特异优先"排序(卡→身份证→手机→邮箱→IP),避免短模式在长标识符内部先命中。这是**纯正则离线识别、零模型调用**,代价是覆盖面就是这五类——姓名、住址等语义 PII 不在其中。

```rust
use langchainrust::guardrails::{
    GuardrailsConfig, PiiRedactionGuardrail, PiiKind,
};
use std::sync::Arc;

// new() = 五类全开;only([...]) 收窄;disable(PiiKind::IpV4) 关单类
let pii = Arc::new(
    PiiRedactionGuardrail::new()
        .only([PiiKind::Email, PiiKind::Phone, PiiKind::CreditCard])
        .with_hold_back(48),                 // 流式扣留窗口,默认 48 字符
);

// 同一个类型同时实现 OutputGuardrail 与 StreamingOutputGuardrail:
let config = GuardrailsConfig::new()
    .with_output(pii.clone())              // 整段复查:redact() 后 Modify 放行
    .with_streaming(pii);                  // 流式增量:扣留式脱敏
// 也可以不当护栏用,直接拿到改写后的字符串:
let safe = PiiRedactionGuardrail::new().redact("联系我 a@b.com 或 13812345678");
```

**流式为什么需要"扣留"(hold-back):** 已经发给用户的字符收不回来,而标识符可能被模型切成两个块吐出(`"138"` + `"12345678"`),逐块正则永远慢半拍。护栏因此始终扣留末尾 48 个字符(覆盖最长的 19 位卡号带分隔符、18 位身份证),等后续文本证明"标识符不在继续写"才释放前面的安全部分;流末由 `flush()` 吐出缓冲区尾部。

- **`ChunkAction::Replace("")`(空串)= "先扣住这个 token"**,每个干净流的早期分块都会发生,所以**不计审计**;非空 Replace(真改写)才立即记一次干预。
- `flush()` 返回 `FlushOutput` 三态:`Empty`(无缓冲)/ `Release(text)`(原文释放,不记审计)/ `Rewritten(text)`(尾部含脱敏标记,记一次干预)。`GuardedAgent::invoke_stream` 在流末自动调 `flush_stream()` 并把尾部 delta 当普通分块发出,再跑整段输出复查。
- **一个护栏实例只能服务一条被守护的流**(缓冲状态存在实例内部的 `Mutex` 里);`GuardrailsConfig` 在 `GuardedAgent::new` 时被消费,并发流请各自建实例。上面 output/streaming 共用同一实例是可以的(整段 `validate` 不碰流式状态),但不要跨流共享。
- `with_hold_back(n)` 可调小以降低输出延迟,代价是标识符异常分散地跨块时可能漏放。

### Schema 输出护栏:边生成边验 JSON ✨ v0.22.4

结构化输出的老问题:模型把 JSON 写坏(类型错、缺必填、多吐字段),要等整段生成完、下游 `serde` 反序列化炸掉才发现——钱已经花了,流式 UI 也已经把坏数据亮给了用户。`SchemaOutputGuardrail` 把输出绑死在一份 JSON Schema 上(和 `StructuredOutput<T>` / `schemars::schema_for!` 同源),**流式期做宽松增量校验提前拦确定性错误,流末做严格终检**。

```rust
use langchainrust::guardrails::{GuardrailsConfig, SchemaOutputGuardrail};
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize, JsonSchema)]
struct Person { name: String, age: u32 }

// 方式一:让 schemars 从类型派生;方式二:SchemaOutputGuardrail::new(schema_value)
let rail = Arc::new(SchemaOutputGuardrail::for_type::<Person>());
let config = GuardrailsConfig::new()
    .with_streaming(rail.clone())   // 无状态、Clone,同一实例注册两次安全
    .with_output(rail);
// schemars 默认不输出 additionalProperties:false,护栏默认仍拒绝未声明字段;
// 想放开:SchemaOutputGuardrail::for_type::<Person>().allow_additional_properties()
```

**两阶段各自拦什么(刻意不同的严格度):**

| 检查项 | 流式增量(`validate_chunk` 看累计原文) | 流末终检(`validate_complete`) |
|---|---|---|
| 还不是合法 JSON / 括号没闭合 | 放行(文档还在写) | 阻断 |
| 已成形值类型错误(`age:"x"`) | **立即 Block** | 阻断 |
| 未声明属性(`strict_properties` 默认开) | **立即 Block** | 阻断 |
| 缺必填字段 | 容忍 | 阻断 |
| `enum`/`const` 不匹配、oneOf 计数 | 基本容忍(oneOf 零匹配会拦) | 阻断 |
| 长度/范围/条数边界 | 无意义("3"还可能变成"30") | 阻断 |
| ` ```json ` 围栏与前后散文 | — | 容忍,自动抽取 JSON 切片 |

增量解析走的是结构化输出章同款的 `PartialJsonParser`(容错、可修复半截 JSON);流式 Block 故意不带原因,精确的 `$.age: expected type integer, got string` 这类路径化报错由流末终检给出。终检支持的 Schema 子集覆盖普通 Rust 类型经 schemars 1.x 产出的全部常见结构:type、properties/required/additionalProperties、items、enum/const、oneOf/anyOf/allOf、本地 `$ref`/`$defs`、长度与数值边界。

它和 provider 侧 `response_format` / `with_json_schema_output` 是**两道独立的门**:后者是"请求模型按 schema 写"(不是所有后端支持),前者是"写完之后验证"(任何后端都能挂),两者可叠加;与 `with_structured_output`(工具绑定式结构化)也不冲突——护栏作用于最终文本,不挑生成路径。

### 审计持久化 ✨ v0.15.0

`AuditSink` trait + `FileAuditSink`(JSON Lines 追加式)把违规记录落盘,供事后分析:

```rust
use langchainrust::guardrails::audit::FileAuditSink;

let config = GuardrailsConfig::new()
    .with_output(Arc::new(SensitiveInfoGuardrail::new()))
    .with_audit_sink(Arc::new(FileAuditSink::new("guardrails.log")?));
```

`violations` 有界(`MAX_VIOLATIONS = 1000`),可 `clear_violations()` 清空。`SensitiveInfoGuardrail` 支持挂 LLM 裁判(`with_judge`,复用 `SensitiveJudge` / `LlmSensitiveJudge`)做上下文敏感检测,并对高误报词(`password`/`密码`/`token`/`secret`)降级为仅告警。

### Retrieval Rail(检索护栏)✨ v0.21.0

**解决什么问题**：输入输出护栏守住了用户消息和模型回复,但 RAG 检索回来的文档也是不可信输入——外部网页/PDF 里可能埋着 "ignore all previous instructions" 这类提示注入。RetrievalRail 对检索结果做**批量注入检测**(模式库与 `PromptInjectionHook` 共享),三种处置模式:

| 模式 | 行为 | 适用 |
|------|------|------|
| `Flag`(默认) | 保留文档,metadata 打标记 `retrieval_rail_flagged` | 保守,先观察 |
| `Redact` | 内容替换为 `[REDACTED by retrieval rail: ...]` | 保留条数、去掉内容 |
| `Drop` | 直接从结果集移除 | 强隔离 |

```rust
use std::sync::Arc;
use langchainrust::guardrails::{GuardedRetriever, RetrievalRail, RailAction};
use langchainrust::retrieval::{RetrieverTrait, SemanticCacheConfig, CachedRetriever};

// 直接扫描(不改变行为,拿到报告):
let mut results = /* Vec<SearchResult> */;
let report = RetrievalRail::default().scan(&mut results);
println!("flagged={} redacted={} dropped={}", report.flagged, report.redacted, report.dropped);

// 装饰任意 RetrieverTrait:
let guarded = GuardedRetriever::new(
    Arc::new(inner_retriever),            // 你的底层 retriever
    RetrievalRail::new(RailAction::Drop),
)
.with_audit_sink(audit_sink);             // 可选:命中写审计

// 与语义缓存叠加:护栏放缓存内侧(先过滤再入缓存,防污染)
let cached = CachedRetriever::new(
    Arc::new(guarded),
    Arc::new(embedder),
    SemanticCacheConfig::new(),
);
// cached 实现了 RetrieverTrait,可直接接入 RAGPipeline
```

**关键行为**:无命中时零改动;`with_patterns(vec![...])` 可追加自定义模式;词法/语义缓存条目永远是护栏过滤后的干净结果。

### AI 透明披露(disclose)✨ v0.21.0

对齐 **EU AI Act 第 50 条**:与 AI 系统交互时应告知用户。提供的是**能力**而非强制——是否调用由应用决定。

```rust
use std::sync::Arc;
use langchainrust::guardrails::{disclose, DisclosureConfig, AuditSink};

let audit: Arc<dyn AuditSink> = /* 你的 AuditSink(如 FileAuditSink) */;
let config = DisclosureConfig::new()
    .with_statement("{system} is an AI assistant. Responses may contain errors.") // 支持 {system} 占位符
    .with_session_disclosed(false);

// 会话开始时调用一次:渲染文案 + 写审计 + 返回渲染后文本
let text = disclose(&audit, &config, "support-bot", Some("trace-1")).await;
// text = "support-bot is an AI assistant. Responses may contain errors."
```

披露失败只记告警不抛错(fire-and-forget),不会阻断业务请求。

---

## Token Counter

**解决什么问题**：LLM 按 token 计费，但 token 数不等于字符数——一段中文、一段代码、一段英文的 token 密度各不相同。想回答"这一轮花了多少 token、多少钱"、发请求前估算 token 数（用来截断超长文本）、或自动累计每次调用的用量，都需要专门的工具。Token Counter 相关组件把"计数 → 追踪 → 计价"串成一条链。

```rust
use langchainrust::{TokenTrackingLLM, ModelPricing, OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

let tracked = TokenTrackingLLM::for_openai(OpenAIChat::new(OpenAIConfig::default()))?;

let result = tracked.chat(vec![Message::human("hi")], None).await?;

let usage = tracked.get_usage().await;                               // prompt / completion / total tokens
let cost = tracked.estimate_cost(&ModelPricing::gpt4o_mini()).await; // USD
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
- **算钱**：`get_usage().await` 拿累计的 prompt / completion / total tokens，`estimate_cost(&pricing).await` 按定价换算成 USD(这两个方法是 async)。

### 怎么选（人话）

- 只是想"每次调用记个数、最后报个总价"，直接用 `TokenTrackingLLM::for_openai(...)` 包装模型 + `estimate_cost`，一条链搞定。
- 要在发请求**前**估算一段文本的 token 数（比如判断要不要截断），用 `TiktokenCounter`（精确）或 `CharRatioCounter`（估算）。
- 中文内容占比高、且不追求精确时，`CharRatioCounter` 够用；正式计费请用精确分词。

### 成本台账:PricingTable + CostTracker + 美元硬闸门 ✨ v0.22.4

上面的 `ModelPricing` + `estimate_cost` 只能事后给**单个包装器**算一笔钱。生产场景缺三样东西:多个模型/多次运行的**累计与分模型汇总**、对接监控系统的**事件导出**、以及超预算时**真正停下来**的强制手段。v0.22.4 在 `lc-core::cost` 补齐(facade 根级直接导出)。

**三层分工,刻意把"测量"和"执法"分开:**

1. **`ModelPrice`——单价**。单位统一是 **USD / 1000 token**,输入输出分开报:
   ```rust
   use langchainrust::{ModelPrice, PricingTable, CostTracker};

   let p = ModelPrice::new(2.5, 10.0);          // $2.5/1K in, $10/1K out
   p.cost_of(1000, 500);                         // 2.5 + 5.0 = 7.5(纯函数)
   ModelPrice::free();                           // 本地/开源模型,0 也是合法价格
   p.blended_per_1k();                           // 路由比价用的单一数字:按 3:1 典型
                                                 // 输入输出比混合(0.75·in + 0.25·out)
   ```
2. **`PricingTable`——按 (provider, model) 查价的表**。provider 限定条目优先;查不到时回落到只按 model 注册的兜底条目(同名模型走不同网关时有用):
   ```rust
   let table = PricingTable::new()
       .with("openai", "gpt-4o", ModelPrice::new(2.5, 10.0))
       .with_model_only("some-self-hosted-model", ModelPrice::free());
   table.get(Some("openai"), "gpt-4o");  // 限定条目遮蔽同名兜底
   ```
   - `PricingTable::builtin()` 自带 12 条常见模型的价格快照(截至 2026-09:gpt-4o/gpt-4o-mini/gpt-4.1 系列/o4-mini、claude-3-5-sonnet/haiku-latest、gemini-1.5-pro/flash、groq llama 两款、deepseek-chat)。**这是方便起步的种子,不是持续维护的真相源**——厂商调价频繁,正式计费请走下面的注册表。
   - 生产路径:`ModelRegistry::fetch(url).await?` 从远端拉最新目录(`fetch_with_client` 可传自定义 reqwest client,也可 `from_json` 读本地 JSON、手工 `register`/`merge`),再 `PricingTable::from_registry(&registry)` 转成表;加载失败报 `CostError::Fetch/Payload`。
3. **`CostTracker`——线程安全的累计台账**。一个 run 建一个,或一个会话共享同一个 `Arc`;每次 LLM 调用记一笔:
   ```rust
   let tracker = Arc::new(
       CostTracker::with_builtin_prices()        // 或 CostTracker::new(Arc::new(table))
           .with_scope("session-user-42")         // 报表/事件里带的标签
           .with_metrics_sink(sink),              // 可选:每笔再导出一个 ObsEvent::Cost
   );

   let usd = tracker.record(Some("openai"), "gpt-4o", 1000, 500).await; // 返回本次费用
   tracker.record_usage(Some("openai"), "gpt-4o", &token_usage).await;  // 或直接吃 TokenUsage

   let total = tracker.total_cost_usd().await;   // 预算门读的就是这个数
   let report = tracker.report().await;          // CostReport:总调用/token/费用 +
                                                 // by_model 按 "openai/gpt-4o" 分模型汇总
   let each = tracker.records().await;           // 每笔明细 CostRecord(最旧在前)
   tracker.reset().await;                        // 同会话开启新 run 时清零
   ```

**两条关键的降级语义(接追踪器永远不会搞坏 agent 主循环):**

- 表里**查不到的模型照样计调用数和 token 数,只是价格按 0**——缺价格数据降级为"只计量",不报错;
- sink 导出失败只 `warn`,不传播给调用方。

**自动记账:把 tracker 挂到包装 LLM 上。** `TokenTrackingLLM` 提供 `.with_cost_tracker(arc)` 和 `.with_provider("openai")`:包装器自己也是 `BaseChatModel`(v0.20.1 起),直接塞进 agent,循环里每次 `chat`/流式/`bind_tools` 调用都自动计价累计;`with_temperature` / `with_max_tokens` / `bind_tools` 重建出的新包装器共享同一组 `Arc`,记账不断。模型 id 优先取响应里回传的 model,空了才退回 `model_name()`。

> 流式的已知边界:流式路径没有完整文本可估算,**只计 provider 在末块真实回报的 usage**;从不回报 usage 的 provider,流式下记账可能为 0。非流式 `chat` 在模型不报 usage 时会回退到 tiktoken 估算。

**美元硬闸门:测量之外的执法放在执行器。** tracker 本身只测量;真正的硬停止复用 §4.2 预算门:

```rust
use langchainrust::{AgentExecutor, CostTracker, BudgetConfig};

let tracker = Arc::new(CostTracker::with_builtin_prices());
let tracked_llm = TokenTrackingLLM::for_openai(llm)?
    .with_provider("openai")
    .with_cost_tracker(tracker.clone());   // 同一个 Arc:LLM 记账
let executor = AgentExecutor::new(agent, tools)
    .with_cost_tracker(tracker.clone())    // 同一个 Arc:执行器读表
    .with_budget(BudgetConfig {
        max_cost_usd: Some(1.0),           // 累计花满 $1 硬停
        max_tool_calls: Some(20),          // 预算门其余四项照旧可同用
        ..Default::default()
    });
```

执行器在**每次规划 LLM 调用之后**读 `total_cost_usd()`,达到/超过限额即返回 `AgentError::BudgetExceeded(BudgetExceeded::Cost { limit, actual })`——调用方能把"预算主动停止"和"模型没收敛"区分开;流式路径同样在每轮 LLM 后检查。两个不对称是有意为之:**只挂 tracker 不设限额 = 只测量不拦截;只设限额不挂 tracker = 永不触发**(读不到花费,读数恒为 0);记账必须靠 tracker 另一端连着 `TokenTrackingLLM`(或你自己在每次调用后 `record`)。

---

## Sessions

> **⚠️ v0.22.0 起推荐事件溯源路径**：本节前半部分介绍的 `SessionManager` 已标 `#[deprecated]`（原计划 0.23.0 移除；截至 0.24.0 仍随 crate 保留、可编译运行，仅发出 deprecation 警告）。新代码请直接用 [`EventSessionManager`](#事件溯源重写--v0220推荐路径)（见下方"事件溯源重写"节）；存量代码升级后仍可编译运行，只会出现 deprecation 警告。

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

### 事件溯源重写 ✨ v0.22.0（推荐路径）

**解决什么问题**：旧 `SessionManager` 是"改写式"——每次对话把整个 `Session` 读出来、改完写回去。三个硬伤：① 进程崩溃在"读-改-写"中间会**丢消息或丢整轮**；② 无法分叉——想试验一个分支对话只能复制整个会话；③ 没有"时间旅行"——无法回答"第 5 轮时的历史长什么样"。

0.22.0 起 lc-sessions 重写为**事件溯源（Event Sourcing）**：会话不再是可变对象，而是一条**只追加（append-only）的事件日志**。每次对话只追加事件，历史 = 从头投影（replay），崩溃安全、可分叉、可回放。

**核心四件套**：`SessionEvent`（事件）← `EventStore`（怎么存）← `EventSessionManager`（怎么用）← `project()`（怎么投影）。

```rust
use langchainrust::sessions::{EventSessionManager, MemoryEventStore, AutoCompaction};
use langchainrust::{OpenAIChat, OpenAIConfig};
use std::sync::Arc;

let manager = EventSessionManager::new(Arc::new(MemoryEventStore::new()))
    .with_max_context_turns(10)                                  // 轮窗（默认全量）
    .with_auto_compaction(AutoCompaction::new(20)?);             // 超 20 轮自动压快照

let id = manager.create_session().await?;                        // uuid v7

let llm = OpenAIChat::new(OpenAIConfig::default());
let r1 = manager.chat(&id, &llm, "My name is Tom".to_string()).await?;
let r2 = manager.chat(&id, &llm, "What is my name?".to_string()).await?;

let history = manager.history(&id).await?;   // 投影后的 Vec<Message>
// 清理/归档/删除:向 EventStore 追加 Metadata 事件(日志不可变,见下文"事件类型")
```

**旧 API → 新 API 对照**（旧 `SessionManager` / `SessionManagerRunnable` 自 0.22.0 起标 `#[deprecated]`，原计划 0.23.0 移除；截至 0.24.0 仍保留可用，`#[allow(deprecated)]` 可静默警告）：

| 旧（0.21.x） | 新（0.22.0） | 语义变化 |
|---|---|---|
| `SessionManager::new(Arc<dyn SessionStore>)` | `EventSessionManager::new(Arc<dyn EventStore>)` | 存储换为 append-only 事件日志；`MemoryEventStore` 对应 `MemorySessionStore` |
| `create_session() / create_session_for(user)` | `create_session()`（uuid v7） | 会话存在性 = 日志非空；用户归属走 `Metadata` 事件 |
| `chat(&id, &llm, msg)` | `chat(&id, &llm, msg)` | 同签名同语义，持久化从"改写"变为"追加"——崩溃安全 |
| `history(&id)` | `history(&id)` / `replay_session(&id)` | history = 投影后的消息；`replay_session` 给出旧式可变 `Session` 视图 |
| （无） | `fork_session(&id, branch, until)` | **新**：从主干复制前缀到新分支，主干不受影响 |
| `max_context_messages(n)`（消息计数窗） | `with_max_context_turns(n)`（轮窗） | n=1 含完整上一轮 + 当前消息；保证 user/ai 成对、无孤儿工具结果 |
| （无） | `with_auto_compaction(AutoCompaction)` | **新**：超 N 轮自动追加确定性 Snapshot 事件（无 LLM 调用） |
| `clear / archive / delete_session` | 直接向 `EventStore` 追加 `Metadata` 事件（截至 0.24.0 仍由调用方直接追加，封装方法尚未落地） | 事件日志不可变,"清理"变为追加状态事件 |
| `SessionStore`（自定义存储 trait） | `EventStore` | 四方法：`append / append_batch / read / fork`；append 幂等键 `(session, branch, id)` |

### 会话分叉（fork）

```rust
// 从主干复制"到某事件序号为止"的前缀到新分支,主干不受影响
// 返回的是一个绑定到新分支的 manager(同一 session_id,事件序号独立)
let mut experiment = manager.fork_session(&id, "experiment", None).await?;
// 继续在分支上 chat:写进 "experiment" 分支,主干看不到
experiment.chat(&id, &llm, "分支消息".to_string()).await?;
// 对同一 (session, branch) 重复 fork 是幂等的——前缀一致,不重复
```

`until_id: Option<u64>` 是事件序号(从 0 起):只复制到该序号为止——"回到第 5 轮再试另一种回答"就是 `fork_session(&id, "retry", Some(5))`。

### 崩溃安全与回放

- **幂等追加**：`append` 以 `(session_id, branch, id)` 为幂等键，同一事件重放不会重复写入——进程在"追加到一半"崩溃后，重启重放残缺批次是安全的。
- **投影（project）**：`project(&events)` / `to_session(&events)` 从事件流重建会话状态；孤儿工具结果（有 tool result 无对应 tool call）在投影时被检测并报错，保证喂给 LLM 的历史永远成对合法。
- **检查点占位**：`SessionCheckpoint` trait + `NoopCheckpoint` 已就位；持久化检查点（把投影落库）原计划 0.23.0 接入，截至 0.24.0 仍只有 no-op 实现，恢复时需全量重放事件日志。

### 事件类型

| `EventPayload` 变体 | 含义 |
|---|---|
| `SessionCreated` | 会话建立（uuid v7 时间有序） |
| `UserMessage` | 用户消息 |
| `AssistantMessage` | 助手回复（一轮一个 Turn） |
| `ToolCall` / `ToolResult` | 工具调用与结果（成对追加） |
| `Snapshot` | 确定性压缩快照（投影时用它替代重放前缀） |
| `Metadata` | 生命周期/归属等键值状态（clear / archive / delete / user） |

### 事件存储选型

测试、单进程用 `MemoryEventStore` 足够；多实例 / 跨进程 / 需要审计历史时，实现 `EventStore`（数据库 / Redis / Kafka 均可）——append-only + 幂等键的契约对后端非常友好，不存在"读改写竞态"。

---

## MCP

[MCP](https://modelcontextprotocol.io)（Model Context Protocol）是 Anthropic 推出的工具协议标准。0.22.4 的 lc-mcp 同时提供**三条客户端轨道**，按对端形态选用：

- **无状态轨道 `StatelessMcpClient`（v0.22.0 起）**：框架自研的 2026-07-28 单请求协议，每个请求都是自包含的 JSON-RPC HTTP POST（带 `Mcp-Method` / `Mcp-Name` 路由头与 `_meta`），**无握手、无会话**，直连 `MCPServer::serve_http` 起的服务；
- **官方 stdio 轨道 `StdioMcpClient`（v0.22.4 起）**：spawn 本地子进程，走完整 `initialize` 握手——对接 Claude Desktop / Cursor 等宿主生态的本地 stdio server；
- **官方 Streamable HTTP 轨道 `StreamableMcpClient`（v0.22.4 起）**：对接按官方 TS/Python SDK 部署的远端 Streamable HTTP server，走完整握手、由服务端分配 `Mcp-Session-Id`，JSON/SSE 内容协商，可挂 OAuth 2.1 令牌提供器。

三条轨道拿到的工具定义都走同一个 `MCPToolAdapter` 适配为 `BaseTool`。

> **历史脉络**：0.22.0 曾按"单轨"要求移除旧的握手式 `MCPClient`（SSE 传输 + 旧 stdio 客户端、事件流/partial-content 推送面）；0.22.4 以**官方协议兼容**的名义把 stdio 与 Streamable HTTP 两条官方轨道加了回来（与无状态轨道并存，不是 feature 门控的二选一）。已对官方 **TypeScript SDK 1.30.0** 与 **Python SDK** 做 4/4 互操作验证。

```rust
use langchainrust::mcp::{MCPToolAdapter, StatelessMcpClient};
use langchainrust::{BaseAgent, AgentExecutor, FunctionCallingAgent, OpenAIChat, OpenAIConfig};
use std::sync::Arc;

// 无状态连接：无握手，connect 只构造客户端（不会失败）
let client = StatelessMcpClient::connect("http://localhost:3001/mcp");

let tools = client.list_tools().await?;           // tools/list
println!("MCP tool count: {}", tools.len());

// 适配为 BaseTool 列表并交给 Agent
// 注意:适配器默认 fail-closed(RequireApproval)——不挂审批门/沙箱时每次调用
// 都在发出前被拒(ToolError::PermissionDenied)。完全信任的内网 server 显式
// allow_unattended_execution();生产场景见下方"工具执行审批与沙箱"。
let mcp_tools: Vec<Arc<dyn BaseTool>> = tools
    .into_iter()
    .map(|def| {
        Arc::new(
            MCPToolAdapter::new(client.clone(), def)
                .allow_unattended_execution(),
        ) as Arc<dyn BaseTool>
    })
    .collect();
let agent = FunctionCallingAgent::new(
    OpenAIChat::new(OpenAIConfig::default()),
    mcp_tools,
    None,
);
let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, vec![]);
let result = executor.invoke("Read /tmp/notes.txt".to_string()).await?;

client.close().await?;
```

`client.call_tool(name, arguments)` 直接调用工具；`MCPToolAdapter::new(client, def)` 将工具包装为实现 `BaseTool` 的适配器（`namespaced` 变体带 `server:tool` 前缀）；`.with_mrtr(...)` / `.with_answer_provider(...)` 处理 `input_required` 多轮请求；`.with_method_rate_limiter(...)` 做方法级限流。

**失败语义（按传输区分）**：无状态轨道没有连接生命周期——`connect()` 只构造客户端、不发请求，网络/服务端错误在**首个实际请求**处立刻暴露，没有"等待自动重连约 30s"的挂起；方法级限流命中时本地直接返回 `-32002`，不发网络请求；MRTR（多轮请求）未提供答案提供器时返回 `-32003`。

### 工具执行审批与沙箱（fail-closed）✨ v0.22.4（A16）

`MCPToolAdapter` 默认策略是 `ToolExecutionPolicy::RequireApproval`：没有任何放行配置时，工具调用在**发往 server 之前**就被拒绝（`ToolError::PermissionDenied`），远程工具永远不能在应用不知情的情况下执行。三种放行方式，按信任度递增/递减自选：

```rust
use langchainrust::mcp::{MCPToolAdapter, ServerSandbox, ToolCallApprover};

// 1) 完全信任(内网自建 server / 测试):显式接受无人值守执行
let trusted = MCPToolAdapter::new(client.clone(), def.clone())
    .allow_unattended_execution();

// 2) 沙箱:参数级白名单,由 ServerSandbox 的 ParamRule/EgressPolicy 判定
let sandboxed = MCPToolAdapter::from_client(std::sync::Arc::new(client_stdio.clone()), def.clone())
    .with_sandbox(std::sync::Arc::new(sandbox));

// 3) 运行时审批门:每次 run 都问 approver,人拒绝则 PermissionDenied、请求不外发
let guarded = MCPToolAdapter::new(client.clone(), def)
    .with_approver(std::sync::Arc::new(my_human_approver));
```

### 官方 stdio 轨道：`StdioMcpClient` ✨ v0.22.4

对接本地子进程形态的官方 MCP server（Claude Desktop、Cursor 生态里的 `npx`/`uvx` server）。构造时 spawn 子进程并完成标准 `initialize` 握手，拿到 server 声明的 capabilities/serverInfo：

```rust
use langchainrust::mcp::{StdioMcpClient, StdioCommand, MCPToolAdapter};
use std::sync::Arc;
use langchainrust::BaseTool;

// 等价于在命令行起: npx -y @modelcontextprotocol/server-everything
let command = StdioCommand::new("npx")
    .arg("-y")
    .arg("@modelcontextprotocol/server-everything");

// connect 是 Result:握手失败(子进程起不来/协议版本不支持)在此暴露
let client = StdioMcpClient::connect(command).await?;

// 也可显式指定版本协商策略与每请求超时(默认 Degrade + 60s)
// let client = StdioMcpClient::connect_with(
//     command, VersionPolicy::Reject, Duration::from_secs(30)).await?;

let tools = client.list_tools().await?;
let _ = client.ping().await?; // 官方 keep-alive 探活

// stdio 客户端同样实现 McpToolClient,走 from_client 适配(fail-closed 同上)
let adapter = MCPToolAdapter::from_client(
    Arc::new(client.clone()), tools.remove(0),
).allow_unattended_execution();

client.close().await?; // 关闭子进程;连接中途死亡不会自动重连(connection_lost)
```

### 官方 Streamable HTTP 轨道：`StreamableMcpClient` ✨ v0.22.4

对接按官方 SDK 部署的远端 Streamable HTTP server：`initialize` 握手后由服务端分配 `Mcp-Session-Id`，之后每个请求/通知都带会话 id；传输层同时支持 JSON 与 SSE 两种响应内容协商，服务端可通过 HTTP 202 异步推通知。断线可 `reconnect()` 重建会话。

```rust
use langchainrust::mcp::{StreamableMcpClient, StaticBearerToken};
use std::sync::Arc;

// 无鉴权端点:握手 + 版本协商(默认 Degrade,每请求 60s 超时)
let client = StreamableMcpClient::connect("http://127.0.0.1:8765/mcp").await?;

// 显式策略/超时:
// let client = StreamableMcpClient::connect_with(
//     url, VersionPolicy::Degrade, Duration::from_secs(60)).await?;

// 固定令牌(PAT/开发侧车):401 时自动失效重试一次
let client = StreamableMcpClient::connect_with_token_provider(
    "https://mcp.example.com/mcp",
    Arc::new(StaticBearerToken("pat-xxx".to_string())),
).await?;

// 完整 OAuth 2.1 刷新语义:自己实现 BearerTokenProvider(见下节)
let client = StreamableMcpClient::connect_token(
    url, Arc::new(oauth_provider), VersionPolicy::Degrade, Duration::from_secs(60),
).await?;

client.reconnect().await?; // 会话失效后重新握手,拿新的 Mcp-Session-Id
client.close().await?;
```

协议版本协商两条轨道共用 `VersionPolicy`：`Degrade`（默认，对端声明版本超出支持列表时降级到本库实现版本继续跑）与 `Reject`（严格模式，版本不符直接握手失败，错误码 `-32005`，stdio 下子进程一并关闭）。

### OAuth 2.1 受保护远端 server ✨ v0.22.4

官方 Streamable HTTP server 普遍以 OAuth 2.1 **资源服务器**（RFC 9728）形态鉴权。完整流程的机器部分由 lc-mcp 提供，需要人参与的部分（打开授权页、接收重定向回调）留给嵌入应用：

1. 未带令牌的请求收到 `401` + `WWW-Authenticate: Bearer resource_metadata="…"` 质询——`OAuthChallenge::parse(header, ..)` 解析出资源元数据地址、realm、scopes；
2. `discover_protected_resource(url)` / `discover_authorization_server(metadata_url)` 按 RFC 9728 / RFC 8414 发现授权服务器（元数据发现与令牌端点调用超时均为 10s）；
3. 应用在浏览器里跑 **授权码 + PKCE（常伴随 DCR 动态客户端注册）** 流程：`code_challenge` 由调用方自己持有 verifier 计算，库不引入新加密依赖；
4. `OAuthTokenClient::new(token_endpoint, client_id)` 做令牌交换：`exchange_authorization_code(code, redirect_uri, code_verifier)`、`refresh(..)`、`client_credentials(..)`；
5. 实现 `BearerTokenProvider`（`token()` 取当前访问令牌；401 时传输层调用 `invalidate(token)` **恰好一次**后用新令牌重试，再 401 才上抛），挂到 `connect_with_token_provider`。固定令牌用内置的 `StaticBearerToken` 即可。

### 互操作与服务端官方传输

- 0.22.4 补齐了官方 SDK keep-alive 依赖的 `ping` 处理，并实现完整 HTTP 状态矩阵（`400/404/405/406/415/401`）与 JSON/SSE 内容协商；对官方 **TypeScript SDK 1.30.0** 与 **Python SDK** 客户端/服务端四个方向互操作验证 4/4 通过。
- 服务端除无状态的 `serve_http` 外，新增**官方 Streamable HTTP** 出口 `MCPServer::serve_streamable_http(listener)`（`Arc<MCPServer>` 上调用，立即返回端点 URL，后台任务 accept）：握手、`Mcp-Session-Id` 分配、202 通知、Bearer 质询齐全，官方 SDK 客户端可直连。可运行的最小例程见 `crates/lc-mcp/examples/streamable_echo_server.rs`（`cargo run -p lc-mcp --example streamable_echo_server`，支持 `--port` / `--bearer`），stdio 侧对应 `stdio_echo_server.rs`。

---

### MCPServer

与 `StatelessMcpClient` 对称：将本地 `BaseTool` 暴露为 MCP Server。支持 `initialize` / `tools/list` / `tools/call`。

```rust
use langchainrust::{MCPServer, Calculator, BaseTool};
use std::sync::Arc;

let tool: Arc<dyn BaseTool> = Arc::new(Calculator::new());
let server = MCPServer::new()
    .with_tool(tool)
    .with_server_info("my-tools", "0.1.0");

server.serve_stdio().await?;
```

`server.handle_request(req)` 用于自定义传输层的单步 JSON-RPC 处理；`server.serve_http(listener)`（在 `Arc<MCPServer>` 上调用）把服务起成框架自研的无状态 HTTP 服务（`StatelessMcpClient::connect(url)` 直连，立即返回端点 URL），示例见 `crates/lc/examples/mcp_http_server.rs`；`server.serve_streamable_http(listener)` 起的是**官方 Streamable HTTP** 服务（官方 TS/Python SDK 可直连，见上方"互操作与服务端官方传输"），示例见 `crates/lc-mcp/examples/streamable_echo_server.rs`。

### ConnectionManager（连接池） ✨ v0.15.0

管理多个 `StatelessMcpClient` 的托管注册表，惰性构建、统一回收：

```rust
use langchainrust::mcp::{ConnectionManager, ServerSpec};

let manager = ConnectionManager::new();
manager.register(ServerSpec::new("files", "http://localhost:3001/mcp")).await?;
manager.register(ServerSpec::new("tools", "http://localhost:3002/mcp").keep_alive()).await?;

let client = manager.client("files").await?;  // 取某个 server 的客户端（惰性构建）
manager.reap_idle().await;                     // 回收空闲句柄
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
use langchainrust::mcp::{MCPGateway, GatewayServerSpec};

let gateway = MCPGateway::new();
gateway.register(GatewayServerSpec::new("files", "http://localhost:9001/mcp")).await?;
gateway.register(GatewayServerSpec::new("db", "http://localhost:9002/mcp")).await?;
gateway.sync_all().await?; // 拉取全部 server 的工具

let tools = gateway.as_base_tools().await?; // 自动加 server 前缀,互不冲突
```

配套能力:
- **`ServerSandbox`**:`ParamRule`(参数白名单/黑名单/类型校验)、`EgressPolicy`(出站策略,限制工具调用的网络/文件范围)
- **`TenantGateway`**:多租户隔离,每租户独立的工具命名空间 + 配额 + 访问控制
- **`ToolOrchestrator`**:工具 DAG 编排,声明依赖关系后自动排序/并行执行
- **`MethodRateLimiter`**:client 侧方法级限流(命中即返回 -32002,不发网络请求)
- **`VersionPolicy`**:协议版本协商策略,三条握手轨道共用——`Degrade`(默认,对端版本超出支持列表时降级到本库版本继续跑)/ `Reject`(严格,版本不符握手失败 `-32005`)

### MCP Server 原语接线 ✨ v0.18.0

client→server 原语(`resources/*` / `prompts/*` / `completion/complete`)为**注册制**:给 `MCPServer` 注册数据源后,对应方法返回真实数据;未注册仍返回 `method_not_found`(-32601,诚实边界)。`initialize` 握手时 `capabilities` 按实际注册项补齐(`tools` 恒声明)。

```rust
use langchainrust::mcp::{
    MCPError, MCPServer, Resource, ResourceContent, ResourceProvider,
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
| DuckDuckGoSearchTool | 网页搜索（免费、无需 key，适合低频/实验） | `query` |
| HostedSearchTool ✨ v0.22.4 | Tavily / Serper / Exa 三家托管搜索（无 feature 门控） | `query`, `top_k`, `include_answer` |
| CdpBrowserTool ✨ v0.22.4 | CDP 驱动 Chrome 抓 JS 渲染页（`browser-cdp` feature） | `operation`, `url`, `wait_ms` |
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

> **v0.22.4 加固（DNS 重绑定窗口闭合）**：旧实现在"检查时解析一次、真正建连时再解析一次"，两次解析之间存在 TOCTOU 窗口（检查看到公网 IP、建连被 rebinding 到内网）。现在**每一跳只解析一次**：取到 DNS 返回的**全部**地址（混合应答里只要有一个内网地址就拒绝），校验通过后用 reqwest 的 `resolve_to_addrs` 把本次连接**钉死在已校验的地址**上（URL 主机名不变，Host 头/TLS SNI 不受影响）；重定向在传输层禁用、手动逐跳跟随，每一跳重新解析/校验/钉 IP。JSON POST（`guarded_post_json`）同样钉 IP。网段表按 RFC 6890/5735/7913 补全（CGNAT `100.64.0.0/10`、基准测试 `198.18.0.0/15`、TEST-NET 等）。sitemap / web-scraper / HTML 三个 loader 与 provider 的消息媒体 URL、Whisper 音频抓取也统一走这条守卫路径；loader 响应体有 **1 MiB 硬上限**（逐 chunk 检查，超限报错而非截断）。

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

### 托管搜索与 CDP 浏览器 ✨ v0.22.4

**`HostedSearchTool`——三家托管搜索后端，无 feature 门控**：内置免费的 DuckDuckGo 抓页在生产环境不稳定、结果质量也有限；0.22.4 把三家商业搜索 API 做成同一个工具，参数化 `SearchBackend`，**不需要开 feature**：

| 构造器 | 后端 | 密钥环境变量 |
|------|------|------|
| `HostedSearchTool::tavily(key)` / `tavily_from_env()` | [Tavily](https://tavily.com)（agent 场景常用，带综合答案） | `TAVILY_API_KEY` |
| `HostedSearchTool::serper(key)` / `serper_from_env()` | [Serper](https://serper.dev)（Google 结果） | `SERPER_API_KEY` |
| `HostedSearchTool::exa(key)` / `exa_from_env()` | [Exa](https://exa.ai)（语义搜索） | `EXA_API_KEY` |

工具输入为 `HostedSearchInput { query, top_k: Option（默认 5、上限 20，超限 clamp 不报错）, include_answer: Option<bool>（默认 true）}`，输出含结果列表与（后端支持时的）综合答案；也可不进 Agent 直接 `.search(query, top_k: Option<usize>, include_answer: Option<bool>).await` 调用：

```rust
use langchainrust::HostedSearchTool;

let search = HostedSearchTool::tavily_from_env()?; // 缺 TAVILY_API_KEY 即 Err
let tool = std::sync::Arc::new(search) as std::sync::Arc<dyn langchainrust::BaseTool>;
// 交给 Agent 后,模型按 {"query": "...", "top_k": 5, "include_answer": true} 调用
```

**`CdpBrowserTool`——CDP 驱动本地 Chrome（`browser-cdp` feature）**：DuckDuckGo/`URLFetchTool` 只能拿静态 HTML，对 JS 渲染页面无能为力。`connect()` 收的是 Chrome **HTTP 调试基址**（`http://127.0.0.1:9222`，必须以 `http://`/`https://` 开头，传 `ws://` 直接 `InvalidInput`）：工具先 `PUT /json/new`（旧版 Chrome 回退 `GET /json` 附着现有页）拿到该标签页的 `webSocketDebuggerUrl`，再用 WebSocket（本机场景即明文 `ws://`）走 Chrome DevTools Protocol 执行四种操作：`navigate` / `extract_text`（渲染后正文）/ `extract_links` / `metadata`，输入 `BrowserInput { operation, url, wait_ms: Option（load 事件后额外等待，上限 10s）}`。SSRF 姿态与网络工具一致：**默认禁止内网/回环目标**，`with_allow_private_urls(true)` 显式放行。它是 opt-in feature（拉入 tokio-tungstenite）：

```toml
langchainrust = { version = "0.24", features = ["browser-cdp"] }
```

```rust
use langchainrust::CdpBrowserTool;

// Chrome 需以 --remote-debugging-port=9222 启动;connect 传 HTTP 调试基址(不是 ws://)
let browser = CdpBrowserTool::connect("http://127.0.0.1:9222").await?;
// 放行的是"要打开的页面 URL"(导航目标);内网后台系统显式放行,公网目标默认即可
let browser = browser.with_allow_private_urls(true);
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
| **Local** | `LocalEmbeddings` | 默认 | 纯 Rust,离线（`local-embeddings` feature) |
| **Candle** | `CandleEmbeddings` | 模型决定 | 纯 Rust 推理(`local-candle` feature)✨ v0.21.0 |
| **Cohere 视觉** ✨ v0.22.4 | `CohereVisionEmbeddings` | 1536 | 图文同一向量空间(Embed v4.0),无 feature 门控 |
| **Qwen 视觉** ✨ v0.22.4 | `QwenVisionEmbeddings` | 1024 | 图文同一向量空间(DashScope `multimodal-embedding-v1`),无 feature 门控 |

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

let embeddings = Arc::new(DeepSeekEmbeddings::from_env_result()?);

let vector = embeddings.embed("Deep learning fundamentals").await?;
```

### Qwen 嵌入

阿里云 Qwen 的嵌入模型，1536 维，中文效果更好。

```rust
use langchainrust::{QwenEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(QwenEmbeddings::from_env_result()?);

let vector = embeddings.embed("Qwen vector generation").await?;
```

### Qwen3-Embedding 与 matryoshka 维度 ✨ v0.21.0

Qwen3 系列嵌入模型(`qwen3-embedding-0.6b` / `4b` / `8b`)已在 `QwenEmbeddingsConfig` 中注册维度映射,并支持 **matryoshka 输出**:通过 `with_dimensions` 指定 32~4096 之间的任意输出维度,DashScope 会按需截断向量——存储成本与检索精度可以按需权衡。

```rust
use langchainrust::{QwenEmbeddings, QwenEmbeddingsConfig};
use std::sync::Arc;

// Qwen3-Embedding + 自定义输出维度(注意 with_dimensions 返回 Result,0.6B 模型默认 1024 维)
let config = QwenEmbeddingsConfig::new("sk-...")
    .with_model("qwen3-embedding-0.6b")
    .with_dimensions(512)?;
let embeddings = Arc::new(QwenEmbeddings::new(config)?);
assert_eq!(embeddings.dimension(), 512);

// 不设 dimensions 时用模型默认维度:0.6B=1024 / 4B=2560 / 8B=4096
let cfg8b = QwenEmbeddingsConfig::new("sk-...").with_model("qwen3-embedding-8b");
let e8b = QwenEmbeddings::new(cfg8b)?;
assert_eq!(e8b.dimension(), 4096);
```

**关键行为**:

- 默认模型仍是 `text-embedding-v1`(1536 维),兼容旧代码;Qwen3 需显式 `with_model`
- `dimensions` 范围校验 32~4096,越界报错;只有 Qwen3 系列支持 matryoshka 截断
- 请求体快照测试保证 `dimensions` 只在设置时透传,不影响 DeepSeek 等其他 provider

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

内建请求韧性:对 429 / 5xx 自动重试(默认 3 次),带 **jitter 抖动**(避免重试风暴)并优先遵循服务端 `Retry-After` 响应头 ✨ v0.21.0;批量嵌入并发度 8、批次上限 2048;向量统一 L2 归一化,便于余弦相似度比对。

```rust
use langchainrust::{OpenAIEmbeddings, Embeddings, retrieval::graph_rag::EmbeddingMatcher};

let emb = Arc::new(OpenAIEmbeddings::new("sk-..."));
let docs = vec!["Rust ownership".into(), "Borrow checker".into()];
let matcher = EmbeddingMatcher::new(emb, docs);
let top = matcher.query("memory safety in Rust", 2).await?; // 语义最相近的 2 篇
```

### CandleEmbeddings(纯 Rust 本地推理)✨ v0.21.0

基于 [Candle](https://github.com/huggingface/candle) 的 CPU-only 本地嵌入后端,无需 ONNX Runtime。启用 `local-candle` feature 后可用,支持 BERT 家族模型(mask 加权 mean-pooling,L2 归一化,批大小 16)。

```toml
# Cargo.toml
langchainrust = { version = "0.24.0", features = ["local-candle"] }
```

```rust
use langchainrust::embeddings::CandleEmbeddings;
use langchainrust::Embeddings;

// 方式一:从 Hugging Face Hub 下载(config.json + tokenizer.json + model.safetensors,带本地缓存)
let embedder = CandleEmbeddings::from_hf_hub("BAAI/bge-small-en-v1.5")?;

// 方式二:从本地模型目录加载
// let embedder = CandleEmbeddings::from_dir("./models/bge-small")?;

let vec = embedder.embed_query("Rust is a systems programming language.").await?;
println!("dim = {}", embedder.dimension()); // 384(config.hidden_size)
```

**适用场景**:已有 safetensors 权重、想完全离线运行;与 fastembed(ONNX)互为替代后端。

### Token 级嵌入(TokenLevelEmbeddings)✨ v0.21.0

可选能力 trait:实现方为每个 token 返回 `(字节偏移 span, L2 归一化向量)`。它**不在** `Embeddings` 主 trait 里——实现了才有(能力探测),避免给不支持 token 输出的 provider 强加契约。

```rust
use langchainrust::embeddings::token_level::{TokenEmbedding, TokenLevelEmbeddings, TokenSpan};

// trait 使用原生 async fn(trait_variant 生成 Send 变体),泛型静态分发,不支持 dyn
pub trait TokenLevelEmbeddings {
    async fn embed_tokens(&self, text: &str)
        -> Result<Vec<TokenEmbedding>, langchainrust::embeddings::EmbeddingError>;
}

// TokenSpan 是字节偏移(多字节安全):
// "你好 world" → tokens[1] = TokenSpan { start: 7, end: 12 } ("world")
```

本地 fastembed(ONNX)路径已实现该能力(与常规 embed 共享推理管线,span 与 [CLS]/[SEP]/padding 对齐)。

### Late Chunking(后分块)✨ v0.21.0

传统做法"先切块、再逐块嵌入"会让每块丢失全文上下文。late chunking 反过来:**整篇文本先过一次 token 级嵌入,再按块边界对 token 向量做 mean-pooling**,每块都携带全文上下文。长文档检索质量通常更好。

```rust
use langchainrust::retrieval::late_chunking::{late_chunk, LateChunkConfig};

let config = LateChunkConfig::new()
    .with_chunk_size(1024)    // 字节,默认 1024
    .with_chunk_overlap(128); // 字节,默认 128,须满足 0 < overlap < size
config.validate()?;

// embedder 需实现 TokenLevelEmbeddings(如本地 ONNX 路径)
let chunks = late_chunk(&embedder, "整篇长文档……", &config).await?;
for c in &chunks {
    // c.range = (start, end) 字节区间;c.text = 块文本;c.vector = 池化后 L2 归一化向量
}
```

**注意**:late chunking 不是银弹——块内自洽的短文档收益为零;需要模型支持 token 级输出,否则用不了。

### 视觉嵌入(图文同一向量空间)✨ v0.22.4

普通 `Embeddings` 只把文本映射成向量;**多模态嵌入**把**图片和文本映射进同一个向量空间**——这才让"以文搜图/以图搜文"成为可能:把商品照片入库建索引,直接用一句"红色运动鞋"检索。0.22.4 新增(均**无 feature 门控**):

| 类型 | 后端 | 维度 | 密钥 |
|------|------|------|------|
| `CohereVisionEmbeddings` | Cohere Embed v4.0(图片必须内联字节) | 1536 | `COHERE_API_KEY` |
| `QwenVisionEmbeddings` | DashScope `multimodal-embedding-v1`(可传公网 URL,服务端抓取) | 1024 | `QWEN_API_KEY` |
| `MockVisionEmbeddings` | 确定性离线后端 | 自定义 | 无(测试用) |

图片用供应商中立的 `ImageInput` 表达:`ImageInput::from_url(url)` / `from_data_uri("data:image/png;base64,…")` / `from_base64(raw_base64, "image/png")`。Cohere 后端要求内联字节,纯 `Url` 会被显式拒绝(库不会替你去抓调用方给的 URL——那会绕过 SSRF 防护把图片流量引到嵌入服务方的内网);需要 URL 入库就自己抓取后走 `Base64`,或改用支持 URL 引用的 DashScope 后端。

```rust
use langchainrust::embeddings::{
    ImageInput, QwenVisionEmbeddings, VisionEmbeddings,
};

let vision = QwenVisionEmbeddings::from_env_result()?; // QWEN_API_KEY
let img_vec = vision
    .embed_image(&ImageInput::from_url("https://example.com/shoe.jpg"))
    .await?;
let text_vec = vision.embed_text("红色运动鞋").await?;
// 两个向量同空间、均 L2 归一化,直接算余弦即跨模态相似度
let _ = vision.embed_images(&[img1, img2]).await?; // 批量,HTTP 后端走单次请求
assert_eq!(vision.dimension(), 1024);
```

关键约定:文本查询必须用**同一个视觉模型**的 `embed_text`,不能拿纯文本 `Embeddings` 的向量去查图片库(两个模型家族的向量空间不一致);后端保证返回向量 L2 归一化、批量请求条数对齐(不一致直接报错,不静默)。多模态 RAG 的切块/检索封装见 [Multimodal RAG](#多模态-rag-视觉)。

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
langchainrust = { version = "0.24.0", features = ["chromadb"] }
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

### 多模态 RAG(图文混合检索)✨ v0.22.4

纯文本 RAG 里图片是"二等公民"——最多靠 alt 文本被搜到,图片本身的视觉内容进不了索引。多模态 RAG 用 [`VisionEmbeddings`](#视觉嵌入图文同一向量空间-v0224) 把**图文都嵌入同一个向量空间**,让"图片里的内容"可被问答。两个封装(在 `lc-rag`,无 feature 门控):

**`MultimodalChunker`——把有序的混排内容切成带模态标记的 `Document`**:

```rust
use langchainrust::retrieval::{
    ImageAsset, MediaBlock, MultimodalChunkConfig, MultimodalChunker,
};

let chunker = MultimodalChunker::with_config(
    MultimodalChunkConfig::default()
        .with_chunk_chars(800)                 // 文本块最大字符数(UTF-8 边界安全);None=整块一篇
        .with_metadata("source", "product-manual-v3"), // 公共元数据,每篇继承
);
let blocks = vec![
    MediaBlock::Text("产品概述:这双跑鞋采用……".into()),
    MediaBlock::Image(
        ImageAsset::new("https://cdn.example.com/shoe-red.jpg")
            .with_caption("红色运动鞋外观图")
            .with_mime("image/jpeg"),
    ),
    MediaBlock::Text("尺码表如下……".into()),
];
let docs = chunker.chunk(&blocks); // 保持输入顺序;空白文本块跳过;每张图独立成篇
```

图片文档的正文是 caption(可空),模态信息进保留元数据键:`mm_kind`(`text`/`image`)、`mm_url`、`mm_caption`、`mm_mime`——`mm_*` 为保留前缀,会覆盖公共元数据里的同名键。

**`MultimodalRetriever`——同一视觉模型嵌入两种模态并检索**:

```rust
use std::sync::Arc;
use langchainrust::embeddings::{ImageInput, QwenVisionEmbeddings};
use langchainrust::retrieval::{ModalityFilter, MultimodalRetriever};
use langchainrust::InMemoryVectorStore; // 或任意 VectorStore(Qdrant/Pinecone/PG…)

let vision = Arc::new(QwenVisionEmbeddings::from_env_result()?);
let store = Arc::new(InMemoryVectorStore::new()); // 维度随首篇入库向量确定
let retriever = MultimodalRetriever::new(store.clone(), vision.clone());

// 入库:图片走批量 embed_image、文本逐块 embed_text,自动恢复原始顺序后写库
retriever.add_documents(docs).await?;

// 文本查询 → 图文混合命中(默认 Any);可限定只要图片/只要文本
let hits = retriever.retrieve_modality("红色运动鞋", 5, ModalityFilter::Images).await?;
let scored = retriever.retrieve_with_scores_modality("尺码", 5, ModalityFilter::Any).await?;

// 反向:以图搜图 / 以图搜文(拍照找同款、找配图段落)
let more = retriever
    .retrieve_by_image(&ImageInput::from_url("https://cdn.example.com/query.jpg"), 5, ModalityFilter::Any)
    .await?;
```

**关键行为与边界**:① 查询向量必须来自同一个视觉模型(文本走它的 `embed_text`,不是纯文本嵌入模型);② 模态过滤下推给存储后端(`mm_kind` 元数据过滤),另加一层结果侧兜底过滤——即使自定义后端静默忽略过滤条件,也不会把错误模态漏进窄查询;③ 图片文档的 `mm_url` 支持 http(s) URL 或完整 `data:` URI(内联 base64);④ Cohere 后端不能服务端抓 URL,入库前需自行把图片解析成 data URI 或用 DashScope;⑤ 检索返回的是 `Document`,生成侧怎么用图片(多模态 LLM / caption 回退)由应用决定。

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

### 小到大检索：句子窗口 / 父文档 ✨ v0.24.0

`ChunkedBM25Retriever` 的父文档合并是**阈值门控**的(同一父文档命中比例过线才合并)。v0.24.0 补了两个"无条件小到大"的检索器,思路同 LlamaIndex 的 sentence-window / parent-document:**用最小的语义单元保证命中精度,喂给 LLM 的却是它周围的完整上下文**。

**句子窗口**——以单句建索引,命中后返回该句前后各 N 句的窗口(默认 `window=2`、`top_k=3`,同一来源文档去重后只给一个窗口):

```rust
use langchainrust::retrieval::RetrieverTrait;
use langchainrust::SentenceWindowRetriever;

let retriever = SentenceWindowRetriever::from_documents(docs)
    .with_window(2)    // 命中句两侧各取 2 句
    .with_top_k(3);

let windows = retriever.retrieve("Rust 异步运行时", 3).await?;
```

**父文档**——叶子小块只负责匹配,任意叶子命中都返回**整个父文档**(无阈值、不按比例):

```rust
use langchainrust::{InMemoryChunkedDocumentStore, ParentDocumentRetriever};

let store = Arc::new(InMemoryChunkedDocumentStore::new());
// (parent_id, chunk_ids):chunk id 形如 {parent_id}::{segment},重复写入同一父文档是幂等替换
let (_parent_id, _chunk_ids) = store
    .add_parent_with_chunks(parent_doc, vec!["叶子 1".into(), "叶子 2".into()])
    .await?;

let retriever = ParentDocumentRetriever::new(store);
// 也可用 with_config(store, AutoMergingConfig) 调叶子切分;.inner() 借到底层 BM25
let parents = retriever.retrieve("查询", 4).await?;
```

| 检索器 | 匹配单元 | 返回单元 | 合并条件 |
|---|---|---|---|
| `ChunkedBM25Retriever` | 叶子块 | 叶子或父文档 | 命中率过 `AutoMergingConfig` 阈值 |
| `SentenceWindowRetriever` | 单句 | ±N 句窗口 | 无条件(按来源去重) |
| `ParentDocumentRetriever` | 叶子块 | 整个父文档 | 无条件,任一叶子命中即返回父文档 |

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

### Weighted 加权融合与 MMR 多样性 ✨ v0.23.0

RRF 只看排名,两路权重不可调,而且同一个窄主题可能占满 top-k。v0.23.0 给 `UnifiedHybridIndex` 补齐两种正式旋钮:

**加权线性融合**——`FusionMode::Rrf`(默认)之外可用 `FusionMode::Weighted { bm25_weight, vector_weight }`,在候选池内先做 min-max 归一化再加权:

```rust
use langchainrust::{FusionMode, HybridIndexConfig, UnifiedHybridIndex};

let config = HybridIndexConfig::new()
    .with_fusion(FusionMode::Weighted { bm25_weight: 0.3, vector_weight: 0.7 });
let index = UnifiedHybridIndex::with_config(embeddings, vector_store, 1536, config);
```

**MMR(Maximal Marginal Relevance)**——贪心选取"相关性 − 与已选结果相似度"最高的候选,λ 越大越偏向相关性(`λ=1` 退化为纯相关,`λ=0` 最大化多样性,常用 0.5–0.7):

```rust
use langchainrust::mmr;

// 索引方法:先取 20 个融合候选、对这 20 个多做一次嵌入,返回多样化后的 5 个
let picks = index.retrieve_mmr("查询", 20, 5, 0.6).await?;

// 也可对自带的 (id, 相关性, 向量) 三元组直接跑纯算法
let ids: Vec<String> = mmr(&candidates, 0.6_f32, 5);
```

池内归一化保证 λ 的权衡在 RRF 与 Weighted 两种融合下含义一致。

### Late Chunking 双腿注入 ✨ v0.24.0

[Late Chunking（后分块）](#late-chunking后分块-v0210)解决了"分块嵌入丢全文上下文",但 v0.21 的 `late_chunk` 只产出 `LateChunk`,落库要自己接。v0.24.0 把它接进混合索引的**两条腿**:池化向量进向量索引,同文本同步进 BM25/父文档 store——一次调用完成双索引注册:

```rust
use langchainrust::{late_chunk, LateChunkConfig, UnifiedHybridIndex};

let config = LateChunkConfig::new().with_chunk_size(1024).with_chunk_overlap(128);
let chunks = late_chunk(&token_embedder, &document.content, &config).await?;

// BM25 与向量两条腿用同一套确定性 id({parent_id}::{segment}),
// 融合时解析回同一个父文档;重复注册同一父文档为幂等替换。
let parent_id = index.add_late_chunked_document(document, &chunks).await?;
```

只需要纯向量单腿时,可用 v0.23.0 起的自由函数 `late_index_in(&vector_store, &embedder, parent_key, text, &config)` 一次完成"token 级嵌入 → 池化 → 入库"(后端无关,id 形如 `{parent_key}:{i}`)。

### 检索模式对比

| 模式 | 内容存储 | 查找 | 使用场景 |
|------|------------------|--------|----------|
| SimpleVector | InMemoryVectorStore | 无查找 | 纯向量，简单场景 |
| BM25 Only | ChunkedDocumentStore | 查找 | 纯关键词 |
| Hybrid | ChunkedDocumentStore（共享） | 查找 | 组合检索（推荐） |

### 原生混合搜索（Qdrant Query API）✨ v0.21.0

`UnifiedHybridIndex` 的 RRF 融合发生在客户端,需要先把两路候选拉回内存。`QdrantVectorStore`(≥ 1.10)支持把**多路向量召回 + 融合**下推到服务端 Query API,一次网络往返完成。能力通过 `NativeHybridSearch` trait 探测——不支持的 store 显式报错并指向客户端 RRF,绝不静默降级。

```toml
langchainrust = { version = "0.24.0", features = ["qdrant-integration"] }
```

```rust
use langchainrust::vector_stores::{
    FusionMethod, NativeHybridQuery, NativeHybridSearch, QdrantConfig, QdrantVectorStore,
};

let store = QdrantVectorStore::new(
    QdrantConfig::new("http://localhost:6334", "my-collection").with_vector_size(1536),
).await?;

assert!(store.supports_native_hybrid());

let query = NativeHybridQuery::new(
    vec![dense_vec, sparse_vec],   // ≥ 2 路查询向量,同维度
    10,                            // 最终返回 top-10
)?
.with_fusion(FusionMethod::Rrf)    // 或 FusionMethod::Dbsf
.with_prefetch_limit(50);          // 每路召回数,默认 max(limit*2, 10)

let results = store.native_hybrid_search(&query).await?; // 服务端融合
```

**关键行为**:`NativeHybridQuery::new` 校验向量数 ≥ 2 且维度一致;`FusionMethod` 直接映射 Qdrant 的 `rrf` / `dbsf` 融合;不支持原生能力的 store 调用会返回明确错误(提示改用客户端 RRF 兜底)。

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

上面的静态中断只能停在**节点边界**(节点前/后),中断点必须在编译期声明。v0.24.0 新增**节点内动态中断**:节点函数自己决定"在什么数据条件下暂停、把什么问题抛给外部",恢复时**重新进入同一个节点**并拿到人工答复,副作用不必重放。

### 节点内动态中断 / 恢复（interrupt + resume） ✨ v0.24.0

```rust
use langchainrust::langgraph::{
    GraphError, InterruptibleNode, StateUpdate, ThreadSafeMemoryCheckpointer,
};
use serde_json::json;

// 闭包签名:(&S, Option<&serde_json::Value>) -> boxed future
// resume=None 表示首次进入;Some(value) 表示带着人工决策重新进入。
let charge = InterruptibleNode::new("charge", |state, resume| {
    let command = state.output.clone().unwrap_or_default();
    Box::pin(async move {
        match resume {
            None => Err(GraphError::InterruptRequest {
                // 抛给外部世界的暂停载荷(要问的问题、上下文……)
                payload: json!({ "kind": "tool_approval", "command": command }),
            }),
            Some(decision) => {
                // 恢复路径:副作用在这里只发生一次
                let mut next = state.clone();
                next.set_output(format!("decision={decision}, charged"));
                Ok(StateUpdate::full(next))
            }
        }
    })
});

let compiled = graph.compile()?
    // 动态中断必须挂 checkpointer——暂停当下先落检查点,进程重启也能恢复
    .with_checkpointer(ThreadSafeMemoryCheckpointer::new());

// 首次执行:节点内挂起,调用方拿到 DynamicInterrupt { node, payload }
let err = compiled.invoke(AgentState::new("charge $99")).await.unwrap_err();
match err {
    GraphError::DynamicInterrupt { node, payload } => {
        assert_eq!(node, "charge");
        // 把 payload 发给审批 UI / 另一进程 / 邮件工单……
        // 人工决策以任意 JSON 回灌,节点带着它重新进入:
        let invocation = compiled.resume_with_value(&node, json!({"approved": true})).await?;
        assert!(invocation.final_state.output.unwrap().contains("charged"));
    }
    other => return Err(other),
}
```

语义要点:

- **一套持久化**:暂停时检查点已写入 checkpointer(内存 / 文件 / SQLite / Postgres / Redis),恢复值放在 `NodeConfig.metadata` 的 `INTERRUPT_RESUME_KEY`(常量 `"__lc_interrupt_resume"`)里注入,不再有第二套 resume 存储;
- **副作用只发生一次**:节点重新进入,闭包自己按 `resume` 分支把副作用放在恢复路径——首次进入在挂起前不应落副作用;
- **可级联**:恢复后另一个节点再次挂起会得到新的 `DynamicInterrupt`,可多轮中断→恢复;递归预算沿用挂起前已消耗的部分,重复中断不能绕过 `recursion_limit`;
- **流式可见**:`StreamEvent::NodeInterrupt(node, payload)` / `Resumed(node, value)` 在流式执行中实时上报暂停与恢复。

### 审批 / 恢复收敛（ApprovalGate） ✨ v0.24.0

工具审批不再需要独立的审批存储:`lc_agents::graph_approval::ApprovalGate` 就是一个图节点,把"高风险工具调用"直接表达为一次节点内中断。

```rust,ignore
use langchainrust::{ApprovalDecision, ApprovalGate};

let gate = ApprovalGate::new("charge_card", |command: &str| {
    // 真正的副作用只在 Allow / Modify 的恢复路径执行
    charge_payment(command); "receipt-123".to_string()
});
graph.add_node(gate);
```

首次进入中断载荷固定为 `{"kind":"tool_approval","tool":<name>,"command":<state.output>}`;恢复时回灌序列化后的 `ApprovalDecision`:`Allow`(执行一次工具并继续)、`Deny { reason }`(工具整体跳过、零副作用、图继续走到 END)、`Modify { arguments, note }`(携带改写参数与备注)。图路径上**唯一**持久化是 checkpointer(跨进程审批用 `FileCheckpointer` 等);旧执行器的 `ResumeStore` / `FileResumeStore` 只在非图的 executor 路径(`with_resume_store` / `pending_approval()` / `executor.resume(decision)`)继续服役。

### 状态历史与时间旅行（fork_from） ✨ v0.24.0

挂上 checkpointer 后,每个检查点都是可回访的状态快照:

```rust
use langchainrust::langgraph::CheckpointInfo;

// 按时间顺序读出全部快照:id / timestamp / seq / recursion_count / state
let history: Vec<CheckpointInfo<AgentState>> = compiled.get_state_history()?;

// 从任意旧快照"开一条新时间线":以该快照状态为基底,从指定节点开始只向前跑
let forked = compiled
    .fork_from(&history[2].id, "analyze", Some(corrected_state))
    .await?;
```

- fork 是**新血统**:从旧快照另存一份新检查点,原时间线不动;
- **只向前、不回放**:从 `continue_at_node` 开始执行,fork 点之前的副作用不会重跑——"换个分支再试一遍"是安全的;
- `override_state: None` 用快照原状态;`Some(s)` 可人工修正状态后再继续;
- fork 与中断可组合:fork 出的执行同样可以在节点内挂起、再 `resume_with_value`。

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

**内存 / 文件后端**:

- `MemoryCheckpointer` —— 进程内（单线程）
- `ThreadSafeMemoryCheckpointer` —— 并发安全
- `FileCheckpointer::new(path)` —— 落盘持久化（不实现 `Default`，必须显式给路径，失败可传播；原子写：先写临时文件再 rename，崩溃不会留下半截检查点）

配合 `with_checkpointer` + `with_interrupt_before` 实现「暂停 → 恢复」工作流。

### 持久化 Checkpointer(SQLite / Postgres / Redis)✨ v0.22.4

内存/文件检查点随进程消失,多实例部署也无法共享。0.22.4 新增三个生产级后端,挂法与内存版完全一致(`.with_checkpointer(...)`),各走独立 feature:`checkpoint-sqlite` / `checkpoint-postgres` / `checkpoint-redis`。

```rust
// SQLite —— rusqlite bundled(从源码编译 SQLite amalgamation,不依赖系统 libsqlite3),
// WAL 日志 + busy timeout,文件可被另一个进程重新打开(写者瞬时交接)
let cp = SqliteCheckpointer::<MyState>::new("./data/graph.db", "thread-42")?;

// Postgres —— tokio-postgres;条件 UPDATE 做存储侧 OCC
let cp = PostgresCheckpointer::<MyState>::connect(
    "host=127.0.0.1 user=app dbname=lg password=…", "thread-42").await?;
// 或复用已有连接:PostgresCheckpointer::with_client(client, "thread-42").await

// Redis —— Lua 脚本 CAS,单条脚本内完成版本比较与写入
let cp = RedisCheckpointer::<MyState>::connect("redis://127.0.0.1/", "thread-42").await?;
// 或复用已有 MultiplexedConnection:RedisCheckpointer::with_connection(conn, "thread-42").await

let compiled = graph.compile()?.with_checkpointer(std::sync::Arc::new(cp));
```

**乐观并发控制(OCC)**:每个检查点带版本号,`update_state` 要求"我基于版本 N 修改";存储侧发现当前版本已被别的写者推进时,本次写入**整体失败**而不是覆盖对方的编辑,错误为 `GraphError::CheckpointVersionConflict { checkpoint_id, expected, actual }`——调用方据此重读最新状态、合并后重试(后写胜出由调用方决定怎么合并,库不静默丢更新)。Postgres 的版本判定在单条条件 UPDATE 内完成(跨进程串行化),Redis 用 Lua 脚本原子 CAS,SQLite 靠文件写锁 + 版本复查。

**怎么选**:单进程/边缘部署用 SQLite(零运维);已有 Postgres 基础设施、要多实例共享 + 连接池生态选 Postgres;超低延迟、检查点可接受 Redis 持久化语义、已有 Redis 运维选 Redis。三个后端都实现同一个 Checkpointer trait,换后端不动图代码。

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

- **白名单防乱用字段(v0.18.1 收紧)**：`allowed_attributes` 是唯一允许出现在 filter 里的字段集合；LLM 构造了白名单外的字段时**直接返回 `RetrieverError::InvalidFilter`**,不会丢弃过滤条件、悄悄退化成搜全库(那会让"我的退款单"越过租户/范围边界)。唯一例外:白名单显式传空 `Vec` 表示"本检索器禁用过滤",此时才警告并忽略 filter。
- **结构化优先、文本回落**：拆解走 `structured_call`（与 Guardrails / Evaluation 同源）拿结构化参数；模型不支持结构化输出时，把整段文本当查询词回落。
- **可进 LCEL**：实现 `RetrieverTrait`，用 `RetrieverRunnable` 包进链中与其他检索器一样组合。

对比：MultiQuery / HyDE 解决的是**召回不足**（问法变体、假设文档），SelfQuery 解决的是**查询里隐含的过滤意图**——语料带结构化元数据、用户查询里带筛选条件时用它。

---

<a id="contextual-retrieval"></a>
## Contextual Retrieval ✨ v0.21.0

Anthropic 提出的索引期增强：每个 chunk 前面只有自己，缺少"它在整篇文档里处于什么位置"的上下文，导致很多语义相关但措辞不同的内容检索不到。`ContextualEnhancer` 在**索引期**用一个小 LLM 为每块生成 1-2 句上下文说明，拼在原文前面入库——检索命中率显著提升。

```rust
use langchainrust::retrieval::contextual::{ContextualConfig, ContextualEnhancer};
use langchainrust::retrieval::Document;

let enhancer = ContextualEnhancer::new(llm)   // 任意 BaseChatModel(建议小模型控制成本)
    .with_config(ContextualConfig::new().with_max_concurrency(4)); // 并发上限,默认 4

let docs = vec![Document::new("Revenue grew 3% year over year.")];
let enhanced = enhancer.enhance_documents(&docs).await;

// enhanced[0].content = "<LLM 生成的上下文>\nRevenue grew 3% ..."
// enhanced[0].metadata["contextual_context"] = 上下文原文(可检索、可审计)
// 之后照常 index_documents(enhanced) 即可
```

**关键行为**：

- **幂等**：已带 `contextual_context` metadata 的文档自动跳过，重复调用不会叠加
- **fail-open**：单块 LLM 调用失败只记 warning、用原文入库，不阻塞索引
- **成本提示**：每块一次 LLM 调用，大语料先评估费用；建议用便宜的小模型

---

<a id="semantic-cache"></a>
## 语义缓存（SemanticCache）✨ v0.21.0

相同/相似的问题每次都重新检索一遍向量库，纯浪费。`CachedRetriever` 包装任意 `RetrieverTrait`：**词法命中**（查询字符串完全相同）直接返回缓存且跳过 embedding；否则对缓存条目算**余弦相似度**，超阈值即视为同义查询命中。

```rust
use std::sync::Arc;
use langchainrust::retrieval::{CachedRetriever, SemanticCacheConfig, RetrieverTrait};
use langchainrust::Embeddings;

let cached = CachedRetriever::new(
    Arc::new(inner_retriever),   // 任意 RetrieverTrait
    Arc::new(embeddings),        // 用于给查询算向量
    SemanticCacheConfig::new()
        .with_threshold(0.95)    // 语义命中阈值,默认 0.95
        .with_max_entries(256)   // FIFO 容量,默认 256
        .with_ttl(Some(std::time::Duration::from_secs(600))), // 可选 TTL
);

let docs = cached.retrieve("apple", 3).await?;   // miss → 检索并缓存
let fast = cached.retrieve("apple", 3).await?;   // 词法命中,零成本

cached.cache().invalidate();                      // 语料更新后手动失效
```

**关键行为**：`k` 参与缓存身份（同查询不同 k 各自缓存）；embedding 失败向上传播、不缓存；`add_documents` 自动 `invalidate()`。配合护栏使用时把 `GuardedRetriever` 放内侧（见 Retrieval Rail 一节）。

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

### 神经重排序：Cohere / Jina 交叉编码器 ✨ v0.24.0

词法重排器零成本、可离线;召回质量优先时改用托管**交叉编码器**:查询与每个候选拼接后联合打分,精度显著高于向量点积。v0.24.0 把两家服务收敛在同一个异步 trait 后面,换厂商不改调用点。

```rust
use langchainrust::retrieval::RetrieverTrait;
use langchainrust::{rerank_async, CohereRerank, SearchResult};

// 传空字符串则回退读 COHERE_API_KEY;默认模型 rerank-multilingual-v3.0
let cohere = CohereRerank::new("")
    .with_model("rerank-v3.5")          // 可选
    // .with_base_url("https://api.cohere.com")  // 测试时可指向 mock
    .no_proxy();                        // 绕过环境代理(服务端到服务端直连场景)

// 先粗召回多捞一些,再让交叉编码器精选
let pool: Vec<SearchResult> = retriever.retrieve_with_scores("查询", 20).await?;
let top: Vec<SearchResult> = rerank_async(&cohere, "查询", pool, 5).await?;
```

- `AsyncReranker` 只有一个方法:`score_async(&self, query: &str, documents: &[Document]) -> Result<Vec<f32>, RerankingError>`;`JinaRerank::new("")` 读 `JINA_API_KEY`,默认模型 `jina-reranker-v2-base-multilingual`,同样提供 `.with_base_url()` / `.with_model()` / `.no_proxy()`;
- `rerank_async` 把 API 返回的 `results[].index` **重映射回输入位置**(服务端不保证按序返回),按分数降序后截断到 `top_n`;空候选直接返回空 `Vec`;
- 响应体畸形是**显式错误**,不会静默地把未排序输入当结果返回。

> 多样性重排（MMR）与 Weighted 加权融合见上文[混合检索](#hybrid-retrieval)一节的 `retrieve_mmr` / `mmr`。

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

> **v0.22.4**：Agent 执行器在**规划阶段**（`BaseAgent::plan` / `plan_stream`，含 ReAct 与 Function-Calling 每一轮规划）也会带上当前 Run 的回调配置——此前规划 LLM 调用对回调/追踪不可见，现在 `on_llm_*` 钩子、LangSmith 与 OTel 都能收到这些调用。注意这是 `BaseAgent::plan`/`plan_stream` trait 方法签名的破坏性变更：末尾新增 `config: Option<&RunnableConfig>` 参数，自定义实现需同步更新（执行器内部已自动传入，直接使用内置 Agent 不受影响）。

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
langchainrust = { version = "0.24.0", features = ["opentelemetry"] }
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

**OTLP 一键导出管道 ✨ v0.22.4**：`opentelemetry` feature 只带 API 依赖（进程内记录，不实际导出）；要把 span 发到 OpenTelemetry Collector,开 `otlp` feature 用电池齐全的管道——OTLP over **HTTP/JSON**(reqwest,不需要原生 TLS 工具链)、Tokio 运行时上的批量处理器、全局安装 W3C trace context 传播、`service.name` 资源:

```toml
langchainrust = { version = "0.24.0", features = ["otlp"] }
```

```rust
// 读环境变量自建 tracer provider,返回的 guard 活到进程结束(drop 时 flush)
let _otlp = langchainrust::callbacks::otlp::install_otlp_pipeline()?;
// 环境变量:OTEL_EXPORTER_OTLP_ENDPOINT(默认 http://localhost:4318,自动拼 /v1/traces)
// OTEL_EXPORTER_OTLP_HEADERS(k1=v1,k2=v2)、OTEL_EXPORTER_OTLP_TIMEOUT(默认 10s)
// OTEL_SERVICE_NAME(默认 langchainrust);自定义参数用 install_otlp_pipeline_with(OtlpConfig)
```

最小可运行例程:`crates/lc/examples/otel/otlp_tracing.rs`(`--features otlp`)。

**gen_ai 语义约定对齐 ✨ v0.21.0**：span 属性对齐 OpenTelemetry GenAI 语义约定（development 状态）——`gen_ai.system`、`gen_ai.request.model`、`gen_ai.request.max_tokens` / `gen_ai.request.temperature`、`gen_ai.response.finish_reason`、`gen_ai.response.model`、token 用量 `gen_ai.client.token.usage.prompt_tokens` / `completion_tokens`，以及扩展属性 `gen_ai.usage.cache_read.input_tokens` / `gen_ai.usage.reasoning.output_tokens`（provider 有上报才填）。LLM 调用 span 的 `gen_ai.operation.name = "chat"`，检索 span 为 `"retrieve"`，工具 span 携带 `gen_ai.tool.name`（`gen_ai.tool.*` 为扩展命名空间，标准稳定后迁移）。Langfuse / Grafana 等消费方可直接按 `gen_ai.*` 属性过滤聚合。

---

<a id="evaluation"></a>
## 评估

量化 LLM 输出质量：在更改提示词 / 模型 / 添加 RAG 之后，运行评估集并查看分数是否提升。5 个类别共 13 个评估器，覆盖从字面匹配到 RAG 幻觉检测：

| 类别 | 评估器 | 描述 |
|----------|-----------|-------------|
| 字面匹配 | `ExactMatch` / `StringDistance` | 精确相等 / 归一化 Levenshtein 距离 |
| 语义 | `EmbeddingSimilarity` / `LLMAsJudge` / `PairwiseJudge` | 余弦相似度 / LLM 评判 / 成对比较（交换 A/B 以消除位置偏差） |
| 规则 | `ContainsKeyword` / `RegexMatch` / `LengthCheck` | 关键词 / 正则 / 长度 |
| 经典 NLP | `Bleu` | n-gram 精确率（字符级 + 平滑） |
| RAG | `Faithfulness` | 拆分声明，逐一验证，检测幻觉 |
| RAG（RAGAS 三指标）✨ v0.22.4 | `ContextPrecision` / `ContextRecall` / `AnswerRelevancy` | 上下文精确率 / 上下文召回率 / 答案相关性,见下方专节 |

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

### RAGAS 三指标（RAG 专用评估器）✨ v0.22.4

RAG 系统的错误形态和普通问答不同:答得"像模像样"但检索没喂对料、召回的料里根本没有答案所需的信息、答案跑题。RAGAS 把这三种情况拆成三个独立指标,全部由 LLM 裁判驱动(与 Faithfulness 同源:优先 `bind_tools` 结构化调用、文本回落,单次评估内裁判并发上限 4,避免触发限流):

| 评估器 | 衡量什么 | 算法要点 | 输入 |
|--------|---------|---------|------|
| `ContextPrecision` | 召回的上下文**排序质量**——有用的块是否排在前面 | 裁判逐块判定与问题是否相关,按排名加权精确率 | 问题 + 有序 contexts |
| `ContextRecall` | 标准答案的信息有多少能在召回上下文里找到 | 拆参考答案为声明,逐句归因到上下文,算可归因比例 | 参考答案 + contexts |
| `AnswerRelevancy` | 答案是否切题(不看参考回答,防答非所问) | LLM 从答案反生成 N 个问题(默认 3),与真实问题算嵌入余弦相似度后取均值 | 问题 + 答案(+ 嵌入模型) |

`ContextPrecision` / `ContextRecall` 只实现 `RagEvaluator`(普通 `Evaluator` 没有 contexts 槽位);`AnswerRelevancy` 两个 trait 都实现,也能在非 RAG runner 里打分。

```rust
use langchainrust::evaluation::{
    AnswerRelevancy, ContextPrecision, ContextRecall, Dataset, EvalRunner, Example,
};

let dataset = Dataset::new(vec![
    // RAG 样例用 with_contexts,contexts 按检索排名顺序传入
    Example::with_contexts(
        "年假多少天?",
        "15 天",
        vec!["员工年假为 15 个工作日……".to_string(), "报销需附发票".to_string()],
    ),
]);

let judge = OpenAIChat::new(config);                 // 任意 BaseChatModel
let runner = EvalRunner::new(vec![])
    .with_rag_evaluators(vec![
        Box::new(ContextPrecision::new(judge.clone())),       // 排名第 1 的块无关 → 低分
        Box::new(ContextRecall::new(judge.clone())),
        Box::new(AnswerRelevancy::new(judge, embeddings)),    // 额外需要 Embeddings
    ]);

let report = runner.run(&dataset, &predictor).await?;
// 上下文超长时:with_max_context_chars(默认每块/拼接上限 2000 字符)
// 空 contexts 等无分可打的情况:with_empty_score(x) 显式指定默认分
// AnswerRelevancy::with_n_questions(5) 调整反生成问题数
```

旧 JSONL 数据集没有 contexts 列时反序列化为空 vec(`#[serde(default)]`),普通评估器照跑、RAG 评估器按 `with_empty_score` 处理。

> 底层复用 `core::judge::structured_call` 的结构化判定路径（强制 LLM 输出 JSON 后解析，错误统一为 `StructuredJudgeError`），保证裁判结果可机读。

### trace → golden → 回归门禁 ✨ v0.24.0

离线手写数据集会过时。v0.24.0 把"**线上 trace → 入库 golden 集 → CI 回归门禁**"闭环补齐:生产录播经 lc-testkit 转成 golden JSONL(转换规则见[测试章"录播 → golden 数据集"](#录播--golden-数据集评分桥--v0240)),新 prompt / 新模型在同一份 JSONL 上跑分,报告与基线报告逐指标对比,退步超阈值就让 CI 红。

```rust
use langchainrust::evaluation::{compare_reports, Dataset, EvalRunner, Report};

// 基线报告来自已知良好版本(Report 支持 JSON round-trip,可落盘 / 传 CI artifact)
let baseline: Report = serde_json::from_str(&std::fs::read_to_string("eval/baseline.json")?)?;

// 候选跑:加载入库 JSONL,predictor 包住新 prompt / 新模型
let dataset = Dataset::from_jsonl("eval/golden.jsonl")?;
let candidate: Report = EvalRunner::new(evaluators).run(&dataset, &new_predictor).await?;

// tolerance = 可接受的均值波动;delta 严格小于 -tolerance 才算退步(边界 -0.02 仍通过)
let cmp = compare_reports(&baseline, &candidate, 0.02);
assert!(!cmp.is_regressed(), "检测到指标回归:\n{}", cmp.to_table());
```

`ReportComparison` 给出:

| 字段 | 内容 |
|---|---|
| `deltas: Vec<MetricDelta>` | 每个共有评估器的 `baseline_mean` / `candidate_mean` / `delta` / 双方计分数 |
| `regressions: Vec<Regression>` | 仅含 `delta < -tolerance`(严格不等式,边界值不算回归)的指标 |
| `added: Vec<String>` / `dropped: Vec<String>` | 候选比基线**新增 / 丢失**的评估器名——候选偷偷少跑了一个评估器在同一张表里可见 |

`cmp.to_table()` 输出可直接贴进 CI 日志的对比表。基线怎么来:首次在已知良好版本上跑一遍,把 `Report` 序列化入库即可。

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
langchainrust = { version = "0.24.0", features = ["mongodb-persistence"] }
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
langchainrust = { version = "0.24.0", features = ["redis-storage"] }
```

或

```toml
[dependencies]
langchainrust = { version = "0.24.0", features = ["sqlite-storage"] }
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

在 23 个 crate 组成的 workspace 里，测试是保证每个模块行为正确的最后防线——解析器、缓存、状态机这类纯逻辑一旦写错，会悄悄影响所有上层功能。用 `cargo test` 从 workspace 根目录跑一遍，能把各 crate 的单元测试、集成测试与文档测试一起执行，尽早发现问题。

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
lc-testkit = "0.24.0"
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

### 录播 → golden 数据集（评分桥） ✨ v0.24.0

v0.24.0 起 lc-testkit 依赖 lc-evaluation,录播文件可以直接转成**评测数据集**:一条 `RecordedExchange` 映射成一条 `Example`,规则固定——请求历史里**最后一条 human 消息** → `input`,录到的 assistant 文本 → `reference`,非空工具结果按排名顺序 → `contexts`(让 RAGAS 类评估器也能给检索质量打分)。Agent 一轮用户提问会产生多条 exchange(包括纯工具调用、文本为空的),`is_scoring_candidate` 先过滤;`golden_dataset` 对漏网的坏行显式报错:

| 错误 | 含义 |
|---|---|
| `GoldenError::NoUserMessage(i)` | 第 i 条交换里没有 human 消息 |
| `GoldenError::EmptyReference(i)` | assistant 文本为空(纯工具轮次) |

```rust,ignore
use lc_testkit::{
    golden_dataset, is_scoring_candidate, read_exchanges, write_golden_jsonl,
};

// 录播文件 → 过滤 → golden 数据集 → 入库的 eval/golden.jsonl
let exchanges = read_exchanges("fixtures/recorded.jsonl")?;
let scored: Vec<_> = exchanges.iter().filter(|e| is_scoring_candidate(e)).cloned().collect();
let dataset = golden_dataset(&scored)?;
write_golden_jsonl(&dataset, "eval/golden.jsonl")?;

// 或一步到位读文件成数据集
let dataset = lc_testkit::golden_dataset_from_file("fixtures/recorded.jsonl")?;
```

零网络、无需 API key 的**离线冒烟跑**:用同一录播文件构造 FIFO `ReplayPredictor`,每条 reference 都会原样复现:

```rust,ignore
use lc_evaluation::{EvalRunner, ExactMatch};
use lc_testkit::{replay_golden_from_file, ReplayStrategy};

let (dataset, replay) = replay_golden_from_file("fixtures/recorded.jsonl", ReplayStrategy::Fifo)?;
let report = EvalRunner::new(vec![Box::new(ExactMatch)])
    .run(&dataset, &replay)
    .await?;
assert!(replay.remaining() == 0);   // 每条样例都消费到了
```

`lc-testkit` 不是 facade 依赖,需要直接引(上面已把 dev-dependency 版本钉到 0.24.0);真模型的回归跑法见[评估章的 trace → golden → 回归门禁](#trace--golden-回归门禁--v0240)。

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

### 协议流程与任务生命周期

典型的 A2A 调用流程（发现 → 派活 → 跟进 → 收尾）：

| 步骤 | JSON-RPC 方法 / 端点 | 说明 |
|---|---|---|
| 1. 发现 | `GET /.well-known/agent-card.json` | 获取远程智能体的 Agent Card |
| 2. 派活 | `tasks/send` | 提交一个任务（`A2AMessage`），服务端立即回 `submitted`，链在后台执行 |
| 3a. 轮询 | `tasks/get` | 按任务 ID 查询状态、结果与错误 |
| 3b. 推送 | `GET /events`（SSE） | 订阅任务状态推送，免轮询（需服务端 `with_streaming`） |
| 3c. 追问 | `tasks/send` 带 `taskId` | 对 `input-required` 任务补充信息、续谈多轮 |
| 4a. 取消 | `tasks/cancel` | 取消未完成的任务 |
| 4b. 列举 | `tasks/list` | 按 `TaskFilter` 列举任务 |
| 4c. 编排 | `tasks/runWorkflow` | 一次提交有序的多步骤工作流（上限 50 步，后台执行） |

任务状态机共 9 个状态，每次流转都做合法性校验（取消一个正在运行的任务后，后台链不可能再把它写回活状态）：

```
submitted ──▶ working ──▶ completed
                │  ├──▶ failed
                │  ├──▶ input-required ──(带 taskId 续谈)──▶ working
                │  ├──▶ cancelled
                └──▶ rejected / auth-required / expired
```

Server 侧是**三个端点**（而非只有两个）：

- `GET /.well-known/agent-card.json` → 返回 Agent Card（智能体的自我描述，供发现）
- `POST /` → 接收并处理 JSON-RPC 请求（`handle_a2a_request` / `handle_a2a_request_authenticated`）
- `GET /events` → SSE 任务推送流（仅 `with_streaming` 启用后有事件）

### A2AServer（暴露你的智能体）

`A2AServer` 默认提供可插入任何 HTTP 框架（axum、actix、warp）的处理函数；启用 `lc-a2a` 的 `axum` feature 后还能直接 `serve` 起一个完整 HTTP 服务（见下文）。

```rust
use langchainrust::a2a::{A2AServer, AgentCard};
use langchainrust::LLMChain;
use std::sync::Arc;

let chain = Arc::new(LLMChain::new(llm, "You are a helpful assistant"));
let server = A2AServer::new(chain)
    .with_card(AgentCard::new("my-agent", "A helpful agent", "http://localhost:8080"));

// 在你的 HTTP 处理函数中：
// GET  /.well-known/agent-card.json → server.get_agent_card()
// POST /                            → server.handle_a2a_request(body).await
```

`A2AServer::new` 接收任何 `Arc<dyn BaseChain>`；如果后端是一个带记忆的 Agent，可以用 `A2AServer::from_agent(Arc<AgentExecutor>)` 直接包装（内部走 `AgentExecutorChain` 适配器），续谈同一任务时能获得真正的多轮上下文连续性。

服务端以 builder 方式装配生产能力（均为 v0.13.0 起的能力，除特别注明）：

| 方法 | 作用 |
|---|---|
| `with_auth_token(token)` | 要求每个请求带 `Authorization: Bearer <token>`，卡片自动声明 `bearer`；常量时间比较防计时侧漏；`POST /` 与 `GET /events` **同口径**校验（0.20.0 修复了 SSE 端点曾完全不鉴权的漏洞） |
| `with_store(Arc<dyn TaskStore>)` | 替换默认内存任务存储，接入自己的数据库后端 |
| `with_max_tasks(n)` | 换一个容量为 n 的内存存储（LRU 淘汰最久未更新的任务），默认 10,000 |
| `with_task_ttl(Some(d))` | 终态任务 TTL（默认 24 小时，读取路径惰性清理；`None` 关闭过期） |
| `with_background_cleanup(interval)` | 额外起一个后台定时清扫任务 |
| `with_streaming(capacity)` | 开启 SSE 推送总线，卡片自动声明 `{"sse": true}`；`subscribe()` 拿接收端 |
| `with_skill_router` / `with_skill_map` | 按请求里的 `skillId` 把任务派发给不同的链（`SkillMapRouter::new().with_skill("translate", chain)`） |
| `with_rate_limiter(Arc<RateLimiter>)` | 每请求并发数 + 每分钟请求数双限流，超限回 HTTP 429 |

**任务持久化**：任务通过 `TaskStore` trait 存取（`upsert` / `get` / `list` / `delete` / `compare_and_update`），默认实现是带 LRU 淘汰的 `InMemoryTaskStore`，进程重启即丢失。trait 自带基于状态机的 `compare_and_update`（CAS），有条件地替换任务状态——自行实现数据库后端时应覆盖为真正的原子条件写，避免"取消"与"链完成"竞争时互相覆盖终态。

**幂等与归属**：请求 metadata 里带 `message_id` 的 `tasks/send` 是幂等的——重试同一个 id 只会取回已创建的任务，不会把链跑两遍（在途 id 有原子预占，服务端 0.20.0 还修了竞争失败者不释放预占、把 id 永久"毒化"的 bug）；任务可带 `owner`，非属主调用 `tasks/get`、`tasks/cancel` 会被拒（`-32003`）。

### 开箱即用的 axum 服务（`axum` feature）

不想自己接 HTTP 框架时，给 `lc-a2a` 开 `axum` feature，直接起服务：

```toml
[dependencies]
langchainrust = "0.24"
lc-a2a = { version = "0.24.0", features = ["axum"] }
```

```rust
// 路由:GET /.well-known/agent-card.json、POST /、GET /events(SSE)
server.serve(8080).await?;           // 绑 0.0.0.0:8080
// 或 server.serve_on(listener).await —— 自定义地址 / TLS / 临时端口
```

CORS 默认只放行 `http://localhost` / `http://127.0.0.1` 来源（0.22.0 审计前曾是任意来源）；面向公网部署时应自行收紧来源白名单，并在前置网关补 TLS 与速率限制。完整可运行示例见 `crates/lc/examples/a2a_http_server.rs`。

### A2AClient（调用远程智能体）

```rust
use langchainrust::a2a::{A2AClient, A2AMessage};

let client = A2AClient::new("http://remote-agent:8080")?;  // 非 HTTPS 仅警告;30s 请求 / 10s 连接超时

// 发现智能体(卡片带签名且配置了校验密钥时,此处会硬校验,见后文)
let card = client.get_agent_card().await?;

// 发送任务(立即返回 submitted,链在远端后台跑)
let task = client.send_task(A2AMessage::user("hello")).await?;

// 查询(只取任务) / 查询详情(任务 + result + error)
let task = client.get_task(&task.id).await?;
let details = client.get_task_details(&task.id).await?;

// 取消任务
let task = client.cancel_task(&task.id).await?;
```

完整配置走 builder——Bearer 鉴权、强制 HTTPS、超时、W3C `traceparent` 链路追踪、卡片签名校验都在这里：

```rust
let client = A2AClient::builder("https://agent.example.com")
    .bearer_token("s3cr3t")                 // 每个请求(含 SSE)带 Authorization 头
    .enforce_https(true)                    // 非 HTTPS 直接 build 失败(默认只警告)
    .timeout(Duration::from_secs(30))
    .connect_timeout(Duration::from_secs(10))
    .with_traceparent(TraceContext::new("trace-id", "parent-id"))  // 同时写入 metadata trace_id
    .card_verification_secret(b"shared-secret".to_vec())           // 校验卡片 JWS/hex 签名
    .require_card_signature(true)           // 卡片带签名却验不了时硬失败,而不是只警告
    .build()?;
```

**同步等结果**：`send_task_and_wait(message, timeout)` 内部按 1s 间隔轮询 `tasks/get`（单次 GET 抖动会小退避重试 3 次），到终态返回 `A2ATaskResult`；远端追问时返回 `A2AError::InputRequired { task_id, prompt }`，用 `resume_task(&task_id, message)` 补答即可，不会傻轮询到超时。要"重试也绝不重复执行"，用 `send_task_with_message_id` / `send_task_and_wait_with_message_id` 两个幂等变体。

**SSE 流式跟进**（替代轮询）：

```rust
// 先订阅再派活,不丢早期事件;SSE 走独立的无总超时 HTTP 客户端,长连接不会被 30s 切断
let mut stream = client
    .send_task_streaming("http://remote-agent:8080/events", A2AMessage::user("做个季度汇总"))
    .await?;
while let Some(event) = stream.next().await {
    let note: TaskPushNotification = event?;   // 事件带 task.id,多任务可按 id 过滤
}
// 已有任务只想订阅:client.connect_sse(url).await?
```

**工作流**：`tasks/runWorkflow` 一次提交有序步骤，服务端后台执行、步骤数硬上限 50：

```rust
use langchainrust::a2a::{A2ARequest, A2AWorkflow, WorkflowStep};

let wf = A2AWorkflow::new(vec![
    WorkflowStep::new("draft", "先写一版提纲"),
    WorkflowStep::with_skill("review", "审一遍并给出修改意见", "critic"),  // 按技能派发(三参关联函数,非 builder)
])
.with_workflow_id("wf-001");
let resp = client.post_request(A2ARequest::run_workflow(1, &wf)).await?;
```

部署示例见仓库 `crates/lc/examples/a2a_http_server.rs`（axum HTTP 封装）。

### v1.0.1：多传输声明（supportedInterfaces） ✨ v0.22.0

**解决什么问题**：v0.3 的 Agent Card 只能声明"一种协议版本 + 一种传输"——同时服务 JSON-RPC 和 HTTP+JSON 两类客户端时只能起两张卡。A2A v1.0.1 把单字段重构为 `supportedInterfaces[]`：一张卡声明多个 `(protocolVersion, transport, url)` 组合，每个接口还可以带企业租户标签。

```rust
use langchainrust::a2a::protocol::{AgentCard, AgentInterface, A2ATransport, A2A_VERSION_V101};

let card = AgentCard::new("agent-a", "A helpful agent", "https://a.example")
    .with_supported_interface(AgentInterface::new(
        A2A_VERSION_V101,           // "1.0.1",每个接口独立 protocolVersion
        A2ATransport::HttpJson,     // JsonRpc | HttpJson | Grpc
        "https://a.example/a2a",
    ))
    .with_supported_interface(
        AgentInterface::new("1.0.1", A2ATransport::JsonRpc, "https://a.example/a2a/jsonrpc")
            .with_tenant("acme"),   // 企业多租户(可选)
    );

assert!(card.is_v101());        // 卡片是否声明了 v1.0.1 接口
```

旧字段保留（v0.3 读者兼容）；新卡应填 `supportedInterfaces`，旧字段视为后备。

**协商（negotiate）**：客户端拿到卡片后，用"我要什么传输 + 我支持哪些版本"挑一个双方都匹配的接口，匹配不到返回清晰错误而不是猜：

```rust
// client_versions 为客户端支持的协议版本列表;命中即返回该 AgentInterface
let picked = card.negotiate(A2ATransport::HttpJson, &["1.0.1", "1.0"])?;
```

### 卡片签名（JWS HS256） ✨ v0.22.0

**解决什么问题**：Agent Card 是发现协议的核心——客户端拿它决定把任务发给谁、用什么协议。卡片被中间人篡改（换端点、降协议版本）就是一次供应链攻击。v1.0.1 提供卡片签名：签名前先把卡片 JSON 按 RFC 8785-lite 规范化（递归键排序），再打 HMAC-SHA256 紧凑型 JWS。

```rust
use langchainrust::a2a::client::{sign_agent_card, verify_card_signature};
use langchainrust::a2a::protocol::AgentCard;

let secret = b"shared-secret-between-registrar-and-clients";

// 发布方(注册中心/Agent 自己):就地计算规范 JSON 签名,hex 写入 card.signature 字段
let mut card = AgentCard::new("agent-a", "A helpful agent", "https://a.example");
sign_agent_card(&mut card, secret)?;

// 消费方(客户端):验证。任何字段被篡改(端点/版本/能力)都会验证失败
verify_card_signature(&card, secret)?;

// 也可以直接操作紧凑型 JWS token(不落卡片字段,适合放响应头):
use langchainrust::a2a::client::{sign_card_jws, verify_card_jws};
let jws = sign_card_jws(&card, secret)?;
verify_card_jws(&card, &jws, secret)?;

// 防算法混淆:token 里 alg 不是 HS256、或卡与 token 不匹配,一律拒绝
```

边界（诚实声明，截至 v0.22.4）：
- **HS256 对称密钥**：注册方与客户端共享 secret；ES256 非对称签名在 0.22.0 时计划挪到 0.22.1，但 0.22.1–0.22.4 实际未落地，当前仍只有 HS256；
- **JSON 规范化为 RFC 8785-lite**（递归键排序），非完整 JCS（数字/unicode 边角与完整 JCS 有差异）；
- **有效期由卡片自身的 `expiresAt` 字段承载**，签名不含时间戳声明；
- `A2ATransport::Grpc` 目前只是卡片上可声明的枚举值，crate 内没有 gRPC 线协议绑定（tonic codegen 依赖 protoc、非 hermetic，自 0.22.0 推迟至今未做）；HTTP+JSON 与 JSON-RPC 两条绑定可独立使用。

### 企业级扩展积木 ✨ v0.13.0

基础三件套（Card / send / get）之外，`lc-a2a` 还带一整套面向"成百上千个 Agent 互联"的积木。它们都是独立类型、按需取用，不影响最简用法：

| 模块（`langchainrust::a2a::`） | 入口类型 | 解决什么问题 |
|---|---|---|
| 注册发现 | `AgentRegistry` / `RegistryClient` | 进程内注册表（按技能关键词 / 数据分级检索卡片）与注册中心 HTTP 客户端 |
| 跨组织联邦 | `FederationGateway` + `CallPolicy` / `DataContract` | 出站调用前做组织白名单、技能白名单、payload 大小与数据密级校验，可最小化外发 metadata，并逐站验证下游卡片 |
| 技能路由 | `SkillRouter` trait / `SkillMapRouter` | 服务端按 `skillId` 把任务派给不同的链 |
| 容错客户端 | `ResilientA2AClient` + `ResilienceConfig` | 四层容错：传输重试（指数退避）→ SSE 断线重连 → 等待总时限（超时可凭 task_id 续等）→ 备援 Agent 链 |
| 限流 | `RateLimiter` | 并发许可 + 每分钟请求数双闸门，超限回 429 |
| 信任与沙箱 | `TrustRegistry` / `TrustedAgent` / `TrustConfig` / `SandboxConfig` | 防冒充 / 防篡改 / 委托跳数信任衰减；读写路径、域名、payload 的沙箱策略 |
| 水平扩展 | `SkillIndex` / `TaskSharder` / `StickyRouter` / `CircuitBreaker` / `HierarchyPolicy` / `DelegationGuard` / `TaskGraph` | 技能倒排索引、任务一致性分片、粘性路由、熔断、上下级委派策略与跳数上限、无环任务图 |

最常用的是容错客户端——主 Agent 抖动时自动退避重试，不可达时按序切到备援 Agent：

```rust
use langchainrust::a2a::{A2AClient, A2AMessage, ResilientA2AClient, ResilienceConfig};

let primary = A2AClient::new("https://agent-a.internal")?;
let backup  = A2AClient::builder("https://agent-b.internal").bearer_token(token).build()?;
let resilient = ResilientA2AClient::new(primary, ResilienceConfig::default())
    .with_fallback(backup);

let task = resilient.send_task(A2AMessage::user("hello")).await?;
// 跨重试要恰好一次语义,改用 send_task_with_message_id
```

### 关键行为与边界

| 能力 | 状态 |
|---|---|
| `tasks/send`（后台执行、9 态状态机、`message_id` 幂等、owner 隔离） | 已实现 |
| `tasks/get` / `tasks/cancel` / `tasks/list` | 已实现 |
| 多轮续谈（`input-required` → 带 `taskId` 续谈） | 已实现（`resume_task`，并发续谈有在途互斥） |
| `tasks/runWorkflow` 有序工作流 | 已实现（50 步硬上限，0.20.0 起后台执行） |
| SSE 任务推送（`with_streaming` + `GET /events`） | 已实现（与 POST 同口径鉴权） |
| Static Bearer 鉴权（常量时间比较，卡片声明 bearer） | 已实现（`with_auth_token`） |
| 可插拔 `TaskStore`（默认内存 LRU 10,000 + TTL） | trait 已实现；crate 只提供内存后端，数据库后端自备 |
| 开箱 axum 服务 | 已实现（`lc-a2a` 的 `axum` feature，`serve` / `serve_on`） |
| 企业积木：注册发现 / 联邦网关 / 容错 / 限流 / 信任 / 扩展 | 已实现 ✨ v0.13.0 |
| Agent Card 发现 + v1.0.1 `supportedInterfaces` 多传输声明与协商 | 已实现 ✨ v0.22.0 |
| 卡片签名 JWS HS256（RFC 8785-lite 规范化） | 已实现 ✨ v0.22.0 |
| 分布式追踪 | metadata `trace_id` + W3C `traceparent` 头（builder 配置） |
| gRPC 绑定 | ⛔ 仅有卡片枚举值，无实现（0.22.0 推迟至今） |
| ES256 非对称签名 / 完整 JCS | ⛔ 0.22.0 计划于 0.22.1，截至 0.22.4 未落地 |
| OAuth / OIDC | ⛔ 内置仅 Static Bearer；联邦场景需在前置网关自接 |
| TLS 终结 | ⛔ 应用层不提供，生产部署放在反向代理 / 网关层 |

### 生产注意点

- 默认任务存储是内存级，进程重启即丢失；生产环境实现 `TaskStore` trait 接数据库（记得把 `compare_and_update` 覆盖成数据库的原子条件更新）。
- 鉴权用 `with_auth_token` 或前置网关注入；别让任意任务 ID 可被查询 / 取消——多租户场景给请求 metadata 带 `owner`。
- 客户端在非 HTTPS 地址上只会警告，跨网部署请用 `builder().enforce_https(true)` 或直接上 TLS 网关。
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

### 工作机制：两套同名 API，别混用

框架里有**两个** `with_structured_output`，名字相同、形状不同：

1. **Provider 自带方法（工具调用约束，推荐）**：`OpenAIChat`（以及 `OpenAICompatibleChat`、`OllamaChat` 等）上的 `chat.with_structured_output::<T>()`——**同步**返回一个 `StructuredOutputMethod<T>`，它把 schema 作为一个 strict 工具声明绑给模型，再用消息调用 `invoke`，从结构化工具参数里直接解析出 `T`：

   ```rust
   use schemars::JsonSchema;
   use serde::Deserialize;
   use langchainrust::language_models::openai::OpenAIChat;
   use langchainrust::schema::Message;

   #[derive(JsonSchema, Deserialize)]
   struct Answer {
       city: String,
       population: u64,
   }

   let chat = OpenAIChat::new(config);
   let method = chat.with_structured_output::<Answer>();        // 注意:不 async,返回的是"方法对象"
   let answer: Answer = method
       .invoke(vec![Message::human("日本人口最多的城市是哪座?人口多少?")])
       .await?;
   ```

2. **泛用 trait 方法（任意 chat 模型，提示注入 + 容错解析）**：`StructuredOutputExt` 对所有 `BaseChatModel` 自动实现，入参是**手写 JSON Schema + 提示词**，把 schema 注入 system 消息，输出经 `JsonOutputParser` 自动剥 markdown 代码块、容错修复后反序列化：

   ```rust
   use langchainrust::StructuredOutputExt;

   let schema = serde_json::json!({
       "type": "object",
       "properties": { "city": {"type":"string"}, "population": {"type":"integer"} },
       "required": ["city", "population"]
   });
   let answer: Answer = chat
       .with_structured_output::<Answer>(schema, "日本人口最多的城市是哪座?人口多少?")
       .await?;
   ```

选型很简单：OpenAI 及其兼容后端用第 1 种（走原生工具调用，不依赖模型自觉遵守文本格式）；模型不支持工具调用、或你只拿到一个泛型 `BaseChatModel` 时用第 2 种。

### 流式版本 stream_structured_output

`StreamingStructuredOutputExt` trait（同样对所有 `BaseChatModel` 自动实现）提供流式版：JSON 一边生成一边经 `PartialJsonParser` 增量解析，**字段一出来就能用**——每成功解析一次就 yield 一个 `T`（已生成的字段填值，其余为默认），流末给最终完整对象。适合前端边收边渲染，比如"书名先出来、作者后出来、年份最后"。

```rust
use langchainrust::StreamingStructuredOutputExt;
use futures_util::StreamExt;

let mut stream = chat
    .stream_structured_output::<Answer>(schema, "列出三本 Rust 经典书")
    .await?;
while let Some(item) = stream.next().await {
    let partial: Answer = item?;   // 累积中的部分结果
}
```

> 注意 trait 约束：流式版的 `T` 除 `Deserialize + Serialize` 外还需 `Clone + PartialEq + Unpin`（相邻解析结果去重）。

### 关键行为

| 行为 | 说明 |
|---|---|
| schema 定义 | 用 Rust 结构体 + `JsonSchema` / `Deserialize` 派生声明输出结构（provider 方法自动 `schema_for!`） |
| 工具调用约束 | `OpenAIChat` 等的方法把 schema 绑成 strict 工具，从 tool 参数直接解析，不靠文本格式 |
| 泛用回落 | `StructuredOutputExt` 对任意模型注入 schema 提示 + `JsonOutputParser` 容错解析 |
| 类型安全 | 返回编译期确定的强类型，不手写解析 |
| 流式版本 | `stream_structured_output` + `PartialJsonParser` 边生成边解析，产出部分结果流 |

### 怎么选

- OpenAI / 兼容后端、要"一次拿完整结果、类型确定"→ `chat.with_structured_output::<T>()` 再 `invoke(messages)`。
- 要"边生成边用、逐字段渲染"→ `stream_structured_output(schema, prompt)`。
- 要引擎在解码层就**保证**合法 JSON（而不是事后解析）→ 下一节 `with_json_schema_output`。
- 只是想解析模型已有的输出（不是让模型按 schema 返回）→ 直接用 `JsonOutputParser` 或 `TypedOutputParser` 等解析器。

### 原生 JSON Schema 引擎约束（with_json_schema_output）✨ v0.21.0

上面两条路径都是**拿到输出后再解析**——模型仍可能给出不合 schema 的内容。`with_json_schema_output` 把 schema 直接下发给 OpenAI（`response_format: {type: "json_schema", strict: true}`），由**引擎在解码时约束**每个 token 只能生成合乎 schema 的 JSON，从源头消灭格式错误。

```rust
use schemars::JsonSchema;
use serde::Deserialize;
use langchainrust::language_models::openai::OpenAIChat;
use langchainrust::schema::Message;

#[derive(Debug, Deserialize, JsonSchema)]
struct Person {
    name: String,
    age: u32,
}

let chat = OpenAIChat::new(config);
let method = chat.with_json_schema_output::<Person>(); // 自动生成 strict json_schema
let person: Person = method.invoke(vec![Message::human("Introduce Alice, 30.")]).await?;
```

**关键行为**：

- schema 由 **schemars 1.x** 从 Rust 类型生成;`make_strict_schema` 自动补 `additionalProperties: false` + 全字段 `required`(strict 模式要求)
- 提供该方法的只有 `OpenAIChat` 与 v0.22.4 起的 `OpenAICompatibleChat`(后者原样转发 `response_format`,Groq/OpenRouter/xAI 等后端在底层模型支持时才生效);不支持时服务端会显式回 4xx,不静默降级。`OllamaChat` 等其他 provider 只有工具调用版 `with_structured_output`
- `AnthropicChat` / `GeminiChat` 各有自己的结构化输出方法类型(`AnthropicStructuredOutputMethod` / `GeminiStructuredOutputMethod`)
- `PartialJsonParser` 流式增量解析保留为泛用 fallback 与流式路径

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

### 概念引入：对齐 Anthropic Computer Use beta

普通工具让 Agent "调用 API、查数据"；Anthropic 的 computer use 让模型通过"截图 → 坐标点击 → 键盘输入"的循环操作 GUI。`ComputerUseTool` 是这一协议的**客户端封装**：把六个标准动作翻译成 `computer_20250124` 工具请求，POST 到 Anthropic `/v1/messages`（beta 头 `computer-use-2025-01-24`），再把响应原样返回。

> **边界（先读再用）**：这个工具**不操控运行程序的这台机器**——它不调用任何系统鼠标/键盘 API、不自带虚拟机/浏览器，真实的 GUI 自动化发生在 Anthropic beta API 背后的托管环境里。它更像一个"动作 → computer-use 请求"的转发器：`ComputerUseOutput.result` 是 Anthropic 的原始响应文本；仅 `screenshot` 动作会额外从响应里抽出 base64 截图。请求体里模型固定 `claude-sonnet-4-20250514`、`max_tokens=4096`（当前版本不可配置）。

六个动作（与 Anthropic 协议同名）：

| 动作 | 必需参数 | 说明 |
|---|---|---|
| `screenshot` | — | 截一张当前屏幕，响应中抽 base64 图像 |
| `click` | `coordinate:[x,y]`（必填） | 鼠标点击；`text:"right"` 表示右键，默认左键 |
| `type` | `text`（必填） | 输入文本 |
| `scroll` | `coordinate` + `direction`（up/down/left/right，均必填） | 滚动，`amount` 默认 3 |
| `key_press` | `keys:["ctrl","a"]`（必填） | 组合键，内部拼成 `"ctrl+a"` |
| `wait` | — | `duration_ms` 默认 1000ms |

坐标会做构造时给定的宽高边界校验；动作名/必需参数缺失都在发请求前返回 `ToolError::InvalidInput`。

### 构造与接入

```rust
use langchainrust::ComputerUseTool;
use std::sync::Arc;

// 没有 new()；唯一真实构造器是 new_anthropic(key, 宽, 高)
let tool = ComputerUseTool::new_anthropic("sk-ant-...", 1024, 768)
    .with_base_url("https://api.anthropic.com")  // 可选：默认官方地址
    .with_timeout(Duration::from_secs(60));      // 可选：HTTP 超时，默认 60s

// 作为 BaseTool 使用
let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(tool)];
```

### 使用注意

- `Default::default()` 虽然存在，但用的是**空 API key、1024×768**，第一次调用就会报 "API key is required"——没有 `new()`，请走 `new_anthropic`
- 目前只有 `AnthropicApi` 一种 `ComputerMode`（枚举里唯一的变体），没有本地 OS 自动化后端
- 副作用发生在 Anthropic 侧环境，费用与安全策略以 Anthropic beta 条款为准；请求失败、非 2xx 都会带响应体转成 `ToolError::ExecutionFailed`

---

<a id="v050-new-features"></a>
## v0.5.0 新特性 ✨ v0.5.0

### RouterLLM（模型路由 + 回退）

`RouterLLM` 本身实现 `BaseChatModel`：在异构模型池中按策略挑选主模型，失败时按注册顺序逐个回退，对调用方就是一个普通 chat 模型，即插即用。

**六种路由策略（前五种 v0.5.0，`LatencyWeighted` ✨ v0.22.4）：**

| 策略 | 行为 | 使用场景 |
|----------|----------|----------|
| `Fallback` | 永远按注册顺序先试主模型，失败再试下一个 | 生产环境容错 |
| `RoundRobin` | 每次调用轮换起始下标，流量均摊 | 负载均衡、摊平速率限制 |
| `LeastLatency` | 选近期 EMA 延迟最低的模型（每次调用后更新指数移动平均） | 延迟敏感、只想"永远打最快的那个" |
| `LatencyWeighted(beta)` ✨ | 按 `(1/延迟)^beta` 加权抽签决定主模型；没试过的模型乐观地共享最佳延迟（保留探索流量），抽签为确定性 SplitMix64、无新依赖 | 既要低延迟、又不想把新模型"饿死" |
| `LowestCost` | 选最便宜的模型 | 成本优化 |
| `InputDirected(Arc<Fn(&str)->usize>)` | 按输入文本闭包选主模型，其余按顺序兜底 | 按查询复杂度/语种路由 |

```rust
use langchainrust::{RouterLLM, RoutingStrategy, BaseChatModel};

// 1. Fallback:主模型 + 备用模型(便捷构造器)
let router = RouterLLM::with_fallbacks(gpt4, vec![claude, local_model]);
let result = router.chat(messages, None).await?;

// 2. 最低成本路由——定价有三种给法:
//    a) with_cost:一个相对成本数,最简单
let router = RouterLLM::new(RoutingStrategy::LowestCost)
    .with_cost(cheap_model, 0.01)
    .with_cost(powerful_model, 0.03);
//    b) with_priced_model + ModelPrice:真实美元单价(每 1K token 输入/输出价),
//       路由权重取 0.75 输入 + 0.25 输出的混合价,同时给预算闸门记账
//    c) with_registry(ModelRegistry) + with_model_as(model, "provider/model-id"):
//       价格来自可 JSON 拉取的模型目录(ModelRegistry::fetch),目录查不到的 key 排最后

// 3. 输入定向路由(闭包返回主模型下标)
let router = RouterLLM::new(RoutingStrategy::InputDirected(Arc::new(|input| {
    if input.contains("code") { 1 } else { 0 }
})))
.with_model(general_model)
.with_model(code_model);

// 4. 延迟加权路由(beta=1.0:100ms 模型被抽中为主的概率约为 1000ms 的 10 倍)
let router = RouterLLM::new(RoutingStrategy::LatencyWeighted(1.0))
    .with_model(fast_model)
    .with_model(slow_but_smart_model);

// 作为普通 BaseChatModel 使用——即插即用替换
let result = router.chat(messages, None).await?;
let stream = router.stream_chat(messages, None).await?;
```

**每模型限流（`ModelRateLimit`，✨ v0.22.4）**：给单个槽位挂准入闸门——滑窗限 RPS/RPM、并发上限、FIFO 等待队列（`with_max_queue` 满了快速拒绝、`with_wait_timeout` 等不到就让路给下一个候选模型），流式调用期间许可持有到整个响应体读完：

```rust
use langchainrust::ModelRateLimit;
let limit = ModelRateLimit::per_minute(600)
    .with_max_concurrent(20)
    .with_max_queue(32)
    .with_wait_timeout(Duration::from_secs(2));
let router = RouterLLM::new(RoutingStrategy::Fallback)
    .with_model_rate_limited(primary, limit)
    .with_model(backup);      // 主模型限流饱和时自动切到备援
```

**共享预算闸门（`RouterBudget`，✨ v0.22.4）**：多个路由器可共享同一个美元/token 预算；每次调用前预扣预估费用，闸门跳开后**付费槽位直接跳过、免费/无价槽位仍可兜底**，调用后按模型上报的真实用量记账。超限错误沿现有回退链传递（facade 里命名为 `RouterBudgetExceeded`，因为 `BudgetExceeded` 一名已被 Agent 执行器占用）：

```rust
use langchainrust::RouterBudget;
let budget = RouterBudget::with_cost_and_token_limits(5.0, 2_000_000); // 5 美元 / 200 万 token
let router = RouterLLM::new(RoutingStrategy::LowestCost)
    .with_priced_model(paid, price)
    .with_model(free_local)
    .with_budget(budget.clone());
let _ = (budget.spent_usd(), budget.is_tripped(), budget.trips());  // 可观测读数
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

固定的"先检索再生成"对两类查询都不划算：寒暄、常识题白付一次向量检索；复杂比较题一次 top-k 又覆盖不全。AdaptiveRAG 在入口加一次 LLM 路由，把每个查询分进三档：

- **NoRetrieval**：模型凭参数知识直接答，完全不碰检索器（`sources` 为空）
- **SingleSearch**：常规路径，检索一次（默认取 4 篇）再生成
- **MultiQuery**：先生成多个改写查询（默认 3 个）逐个检索、合并后再生成

```rust
use langchainrust::agents::adaptive_rag::{AdaptiveRAG, RagDecision};

// 结构体叫 AdaptiveRAG（不是 AdaptiveRAGAgent）；new(llm, retriever)
let agent = AdaptiveRAG::new(llm, retriever)
    .with_retrieve_k(4)          // 每条查询的检索篇数，默认 4
    .with_multi_query_count(3);  // MultiQuery 档的改写查询数，默认 3

// 复杂问题 -> LLM 选择 MultiQuery，生成多个查询角度
let result = agent.invoke("Compare tokio vs async-std scheduling").await?;
assert_eq!(result.decision, RagDecision::MultiQuery);
println!("{} (来源 {} 篇)", result.answer, result.sources.len());

// 简单问候 -> LLM 选择 NoRetrieval，完全跳过检索
let result = agent.invoke("Hello").await?;
assert!(result.sources.is_empty());
```

返回 `AdaptiveRAGResult { answer, decision, sources }`；路由解析失败是显式错误 `AdaptiveRAGError::DecisionParse`，不会静默退化成某一档。与 CRAG 一样提供 `stream()`：先发 `PipelineStep`（路由决策）事件，最后发 `FinalAnswer`，可用于在 UI 上展示"本次为什么没检索/检索了几轮"。

---

### GraphRAG（知识图谱 RAG）

向量搜索会遗漏关系。GraphRAG 提取实体 + 关系 → 构建图 → **分层 Leiden 社区发现** → 逐层社区摘要 → 按社区/邻居查询。

> v0.22.4 变更（B10）：社区算法从早期的标签传播/"tier"近似，换成了**确定性的加权模块度 Leiden**（fast-local 局部移动 + 保证精化 + 聚合，节点遍历顺序用带种子的 SplitMix64 打乱，同输入必得同结果），并支持递归层次——粗层把社区内部边折成自环再检测。旧的 `community_size_tiers` 配置已删除。

```rust
use langchainrust::{GraphRAG, GraphRAGConfig, GraphQueryMode, GraphGlobalLevel};

// 配置是可选的;以下均为新默认值
let config = GraphRAGConfig::default()
    .with_leiden_resolution(1.0)    // 模块度分辨率:越大社区越碎
    .with_leiden_seed(42)           // 确定性遍历种子
    .with_max_community_levels(3);  // 递归社区层次上限
let graph_rag = GraphRAG::new(llm).with_config(config);

graph_rag.add_documents(&documents).await?;   // 只做 LLM 实体/关系抽取并入图
graph_rag.build_communities().await?;         // 必须显式调用:分层 Leiden + 逐层 LLM 摘要

// 全局查询:搜索最粗层社区摘要(宏观问题;每个实体恰好被一个根社区覆盖)
let result = graph_rag.query("overall tech stack architecture", GraphQueryMode::Global).await?;

// 局部查询:取相关实体的邻域子图(具体问题)
let result = graph_rag.query("Alice's advisor's students", GraphQueryMode::Local).await?;

// 混合:社区摘要 + 局部邻域
let result = graph_rag.query("...", GraphQueryMode::Hybrid).await?;

// 层次感知:指定用哪一层的社区摘要答全局/混合问题
let _ = graph_rag.query("...", GraphQueryMode::GlobalAt(GraphGlobalLevel::Level(0))).await?; // 0 = 基础 Leiden 分区
let _ = graph_rag.query("...", GraphQueryMode::GlobalAt(GraphGlobalLevel::All)).await?;      // 所有层一起(更费 token)
let _ = graph_rag.query("...", GraphQueryMode::HybridAt(GraphGlobalLevel::Coarsest)).await?;
```

**流水线：** 文档 → LLM 实体+关系抽取（按文档设上限、同名去重）→ 图构建 → `build_communities()`：分层 Leiden → L0 用实体内边摘要、上层用子社区摘要 + 跨社区关系 ROLLUP → 查询五模式（`Global` / `Local` / `Hybrid` / `GlobalAt(level)` / `HybridAt(level)`，层次选择 `Coarsest`（默认）/ `Level(n)` / `All`）。无外部图库依赖，图存在进程内；另有 `entity_count` / `relation_count` / `community_count` / `community_summaries` 可观测方法。

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

MCP 规范定义了 6 类原语。LangChainRust 中 **已实现调用逻辑** 的是：`initialize`（握手）、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list`、`prompts/get`、`completion/complete`。通知（无 id 的消息）也不再被静默丢弃：`notifications/cancelled`（请求取消某次调用）、`notifications/progress`（进度）、`notifications/roots/list_changed`、`notifications/initialized` 都有显式分发——但截至 v0.22.4 实现体只记录日志、预留扩展点，尚未接真正的取消/进度回调；**规范中的流式工具结果通知并未实现**，不要据此假设工具支持增量输出。client→server 原语均为**注册制**：注册数据源后才返回真实数据，未注册仍返回 `method_not_found`。server→host 方向的 `sampling/createMessage` / `elicitation/create` 由 `MCPServer` 发起，需注入回调。

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

> **安全前提（务必先读）**：`SandboxTool` 默认**禁用**——不调用 `.with_dangerously_allow(true)`，任何执行请求都直接返回错误（0.20.0 安全加固）。内置的 `LocalSandbox` 只是"子进程 + 超时"，**没有任何 OS 级隔离**；Python 那个危险导入黑名单在源码注释里明确写着是 *noise filter*（`__import__("o"+"s")` 之类的写法即可绕过），**不是安全边界**。不可信代码必须放进容器 / VM / WASM 后再经自定义 `CodeSandbox` 实现接入。曾经存在的 `WasmSandbox` / `E2BSandbox` 只有接口、函数体永远 "not implemented"，0.20.0 审查时连同 `sandbox-wasm` / `sandbox-e2b` feature 一起删除——承诺但不能交付的后端比没有更糟。

```rust
use langchainrust::tools::sandbox::{LocalSandbox, CodeSandbox, SandboxTool, Language};

// 直接使用后端：run(code, language, timeout_ms) -> Result<RunResult, SandboxError>
let sandbox = LocalSandbox::new()
    .with_python_path("python3")   // 可选，默认自动探测 python3 / python
    .with_node_path("node");       // 可选，JavaScript 后端，默认 "node"

let result = sandbox.run("print(2 + 2)", Language::Python, 30_000).await?;
assert_eq!(result.stdout.trim(), "4");
// RunResult { stdout, stderr, exit_code, execution_time_ms }

// 包装为 BaseTool 供智能体使用 —— 必须显式打开执行开关
let tool = SandboxTool::new(LocalSandbox::new(), Language::Python)
    .with_timeout(30_000)                 // 默认 30 秒；超时映射为 ToolError::Timeout(秒)
    .with_dangerously_allow(true);        // 不开这一行，工具调用必失败
```

后端能力边界（v0.22.4）：

- **Python**：`python -c` 子进程执行；运行前做危险导入子串检查（os/subprocess/sys/socket/pickle 等 20 个模块，命中即报错），再次强调可绕过，仅用于挡住随手写出的危险代码
- **JavaScript**：`node -e` 子进程执行，无任何额外检查
- **Rust**：`Language::Rust` 枚举存在（序列化/模式匹配完整），但 `LocalSandbox` 直接返回 `SandboxError::UnsupportedLanguage`——需要编译工具链，刻意未实现
- 自定义后端：实现 `async fn run(&self, code: &str, language: Language, timeout_ms: u64) -> Result<RunResult, SandboxError>` 的 `CodeSandbox` trait，即可塞进同一个 `SandboxTool<S>`

---

### OpenAI Responses API

连接 `/v1/responses`（默认 base_url `https://api.openai.com/v1`，默认模型 `gpt-4o`），由 **OpenAI 服务端托管**内置工具——一次请求内模型自行决定检索/执行，框架不参与工具循环。

```rust
use langchainrust::language_models::openai::responses::{ResponsesModel, ResponsesConfig, BuiltinTool};

let config = ResponsesConfig::new("your-api-key")
    .with_model("gpt-4o")                              // 默认即 gpt-4o
    .with_temperature(0.2)
    .with_max_tokens(2000)
    .with_builtin_tool(BuiltinTool::WebSearch)
    .with_builtin_tool(BuiltinTool::CodeInterpreter);

let model = ResponsesModel::new(config);
// 也可 ResponsesModel::from_env()? 读 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL

let result = model.chat(messages, None).await?;
// result.content  = 服务端跑完工具后的最终文本（拒绝项渲染为 [Refusal: ...]）
// result.tool_calls = 服务端执行轨迹（web_search / file_search /
//                     code_interpreter / computer_use），仅记录，不需要你回填
```

边界与现状（v0.22.4，使用前请知悉）：

- **外部可直接构造的只有 `WebSearch` / `CodeInterpreter` 两个无字段变体**。`FileSearch { vector_store_ids }` 与 `ComputerUse { display_width, display_height }` 的字段不是 `pub`，枚举外部无法构造（crate 内部测试在用）——在字段开放前，这两个变体对框架使用者相当于摆设
- 与普通 chat 的消息差异：工具消息会被转成 `function_call_output`（`call_id` + output），人类消息支持图片（`input_image`，多模态直连）
- 工具在 OpenAI 服务器上执行，`tool_calls` 只是把 output item 映射出来的**执行轨迹**；这里没有本地 ReAct 循环，框架不会替你运行代码或回填结果
- 支持 `stream_chat`：`output_text.delta` 逐 token 发射，`response.completed` 事件携带真实 input/output token 用量（`StreamChunk.token_usage`）

---

### Anthropic Extended Thinking

配置思考预算让 Claude 在正文前先输出 thinking 块。非流式路径思考文本经 `LLMResult.thinking_content: Option<String>` 暴露（只有 thinking 块、没有正文时 `content` 保持空串，思考内容绝不会混进答案）；流式路径思考增量**不进数据流**——`stream_chat` 会把 thinking token 从输出流里丢弃，只能通过回调处理器的 `on_llm_thinking(run, chunk)` 观察。

```rust
use langchainrust::{AnthropicChat, AnthropicConfig};

let config = AnthropicConfig::new("your-api-key")
    // 0.22.4 模型别名已刷新到 4.x：claude-opus-4-1 / claude-sonnet-4-5 / claude-haiku-4-5
    // （另含两个 3.5 legacy 别名）；env 默认仍指向 claude-3-5-sonnet-20241022
    .with_model("claude-sonnet-4-5")
    // Anthropic 硬规则：max_tokens 必须大于思考预算（预算本身计入输出上限）
    .with_max_tokens(12_000);
let model = AnthropicChat::new(config)
    .with_thinking(10_000); // 预算 token 数；API 要求 ≥ 1024

let result = model.chat(messages, None).await?;
println!("Thinking: {:?}", result.thinking_content); // Some("...")
println!("Answer: {}", result.content);
```

使用边界（v0.22.4）：

- 框架**不校验**预算合法性：低于 1024 或忘记把 `max_tokens` 调到预算之上，错误来自 Anthropic API 而不是编译期
- 开启 thinking 时 temperature 等采样参数受 Anthropic 侧约束（思考模式只允许少数固定取值）
- thinking 块的 `signature` 与 `redacted_thinking` 未做处理：多轮工具调用场景下不会把上一轮思考块回传，长链路 agentic 用例需自行评估
- 另一处开关在 config 层：`AnthropicConfig::with_thinking(ThinkingConfig::enabled(n))`，效果等同 model 层快捷方法；另有 `with_prompt_caching(true)` 透传 `cache_control` 断点（0.22.4）

---

### 流式结构化输出

`PartialJsonParser` 增量地把流式 JSON 解析为部分结构体——每收到一批 token 就尝试一次反序列化，UI 可以逐字段渲染而不必等整段回答结束。这是泛用 blanket 路径（任何 `BaseChatModel` 自动获得），底层走"schema 系统提示 + 逐块解析"，与 provider 原生 strict-tool 结构化是两套机制，区别见[结构化输出](#结构化输出)章。

```rust
use langchainrust::core::structured_output::StreamingStructuredOutputExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// T 的完整约束：DeserializeOwned + Serialize + Clone + PartialEq + Unpin + Send + Sync
// （PartialEq 用于相邻去重；Serialize 是硬约束，少了编译不过）
#[derive(JsonSchema, Serialize, Deserialize, Clone, PartialEq, Default)]
struct UserInfo {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    age: Option<u32>,
    #[serde(default)]
    email: Option<String>,
}

let schema = serde_json::to_value(schemars::schema_for!(UserInfo)).unwrap();
// 参数是 (schema, prompt)：内部组装 system(schema 指令) + human(prompt) 两条消息
let stream = model.stream_structured_output::<UserInfo>(schema, "Tell me about Alice, age 30").await?;
pin_mut!(stream);
while let Some(result) = stream.next().await {
    let partial = result?;
    if let Some(name) = &partial.name {
        println!("Got name: {}", name); // 在所有字段到达之前即可获取
    }
}
```

行为细节：字段必须能"缺省反序列化"（`Option` 或 `#[serde(default)]`），否则中途的半成品 JSON 反序列化失败、该次部分结果被跳过，只在最终完整时产出一个值；处理器用 `last_value` 对相邻的相同结果去重，不会重复发射未变化的结构体。

---

### Batch API

`BatchClient` 用同一套接口屏蔽两家的批量通道：**OpenAI**（JSONL 上传 + `/v1/batches`）与 **Anthropic**（`/v1/messages/batches`，内联请求数组）。批量通道价格约为实时的一半，但结果有分钟到 24 小时级延迟，只适合离线评测、翻译、语料生成这类不急的任务。

```rust
// 注意路径：facade 在 core::batch 下（同时也顶层再导出了一份）
use langchainrust::core::batch::{BatchClient, BatchProvider, BatchRequest};

let client = BatchClient::new(BatchProvider::OpenAI, "your-api-key");
// BatchProvider::Anthropic 走同一套 API

let requests = vec![
    BatchRequest {
        custom_id: "req-1".to_string(),   // 自定义关联 ID，结果原样带回
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

// 一体化：提交 → 每 5s 轮询 → 最长等 300s，超时返回 BatchError::Timeout
let results = client.submit_and_wait(requests, 5_000, 300_000).await?;
for result in results {
    // 单条失败不影响其他条目：每条自带 Result<LLMResult, BatchError>
    match result.result {
        Ok(llm) => println!("{}: {}", result.custom_id, llm.content),
        Err(e)  => eprintln!("{} failed: {e}", result.custom_id),
    }
}
```

也可以拆开控制：`submit() -> BatchId`、`poll(&id) -> BatchStatus`（`InProgress` / `Completed` / `Failed` / `Expired` / `Cancelled`）、`results(&id)`、`cancel(&id)`——适合把批次 ID 落库、跨进程恢复轮询的场景。

---

### 追踪（分布式追踪）

`Tracer` + `SpanGuard`（RAII）手工埋点：guard drop 即结算 span（持续时间、父子关系自动写回后端），也可提前 `.end()`。三种后端：`InMemoryTracingBackend`（测试断言用）、`ConsoleTracingBackend`（打印）、`OtelTracingBackend`（接 OpenTelemetry 全局 tracer，在 `opentelemetry` feature 后，导出器由使用方自行装配，`OtelTracingBackend::from_global("service-name")`）。

```rust
use langchainrust::callbacks::tracing::{
    Tracer, ConsoleTracingBackend, SpanKind, SpanTokenUsage,
};
use std::sync::Arc;

// SpanKind 只有这 5 + 1 种：
// Llm / Chain / Tool / Retriever / Agent / Custom(String)
// —— 不存在 "Internal" 之类的通用变体
let tracer = Tracer::new(Arc::new(ConsoleTracingBackend));
let span = tracer.start("agent_run", SpanKind::Agent);
{
    // 父子关系靠 task-local span 栈自动维系；栈上没有活动 span 时，start_child() 降级为根 span
    let _retrieve = tracer.start_child("retrieve", SpanKind::Retriever);
    let docs = retriever.retrieve(&query).await?;
} // _retrieve drop -> 子 span 自动记录结束时间
{
    let mut generate = tracer.start_child("generate", SpanKind::Llm);
    let answer = llm.chat(messages, None).await?;
    // 结束前补挂属性：token / 成本 / GenAI 语义约定 / 元数据
    let _ = generate
        .with_tokens(SpanTokenUsage { prompt_tokens: 120, completion_tokens: 80, total_tokens: 200 })
        .with_cost(0.0021)
        .with_gen_ai_request_model("gpt-4o");
}
span.end(); // 不手动 end 也会在 drop 时结算
```

- 并发任务（`tokio::spawn`）里要用 `start_child` 前先 `init_task_span_stack()`——span 栈是 task-local 的，不会跨任务继承；显式传父 ID 可用 `start_child_with_parent(name, kind, parent_id)`
- span 上可挂 `with_tokens(SpanTokenUsage { .. })` / `with_cost(f64)` / `with_gen_ai_request_model(..)` / `with_gen_ai_response_model(..)` / `with_metadata(key, serde_json::Value)`，失败路径调 `set_error(msg)`（`&mut self`），后端会带上错误标记
- 这是与回调体系（`on_llm_start` 等）平行的**手工**埋点 API，适合给检索器、业务步骤等非模型阶段补 span

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

`InMemoryDocumentStore` 和 `InMemoryChunkedDocumentStore` 之前使用 `tokio::sync::RwLock` 的 `blocking_read()`/`blocking_write()`，在异步上下文中会因 "Cannot block the current thread from within a runtime" 而 panic。当时切换为 `std::sync::RwLock`。（**v0.22.4 现状已继续演化**：`InMemoryDocumentStore` 改回 `tokio::sync::RwLock` 但全程 `.await`、不再有 blocking 调用；`InMemoryChunkedDocumentStore` 则刻意保留 `std::sync::RwLock`，因为它有同步调用方，并对 poison 做了恢复处理——读源码时以现状为准。）

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

- **Feature gate 声明**：`sandbox-e2b` 和 `sandbox-wasm` feature 在代码中被引用但未在 `Cargo.toml` `[features]` 中声明——当时补声明；**但这两个后端始终只有接口、函数体永远 "not implemented"，已在 v0.20.0 连同 feature 一起删除**（见上文[代码解释器沙箱](#代码解释器沙箱)），v0.22.4 里只剩 `LocalSandbox`
- **Clippy 零警告**：所有 clippy 警告已解决

---

## 更多资源

| 资源 | 内容 |
|----------|---------|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | 贡献指南 |
| [API Docs](https://docs.rs/langchainrust) | Rust API reference |