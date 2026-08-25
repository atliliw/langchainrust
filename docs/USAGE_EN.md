# Usage Guide

This document provides detailed usage instructions. For a quick overview, see [README.md](../README.md).

---

## Table of Contents

- [LLM](#llm)
  - Multi-Provider Support
  - Unified Client & Auto-Discovery ✨ v0.15.0
  - OpenAI Chat
  - Streaming
  - Function Calling
  - Ollama (Local LLM)
  - Google Gemini
  - Multimodal Vision
  - Message Structure ✨ v0.15.0
  - MultimodalModel ✨ v0.15.0
  - OpenAI Assistants API
- [Embeddings](#embeddings)
  - OpenAI Embeddings
  - DeepSeek Embeddings
  - Qwen Embeddings
  - LocalEmbeddings
- [Prompts](#prompts)
  - FewShotPrompt + ExampleSelectors
- [Output Parsers](#output-parsers)
- [Memory](#memory)
  - VectorStoreRetrieverMemory
  - MongoPersistentMemory ✨ v0.15.0
  - ContextWindow (Long Context Management) ✨ v0.4.1
- [LLM Cache](#llm-cache)
- [Chains](#chains)
  - ConversationRetrievalChain
  - RouterChain ✨ v0.14.0
  - Chain Streaming ✨ v0.4.1
  - ConversationChain ✨ v0.13.0
  - invoke_with_config ✨ v0.15.0
- [LCEL (LangChain Expression Language)](#lcel-langchain-expression-language-) ✨ v0.9.0
  - RunnableWithFallbacks ✨ v0.10.0
  - RunnableAssign ✨ v0.10.0
  - RunnableRetry ✨ v0.11.0
  - CancellationToken ✨ v0.11.0
  - Adapters (AgentEventRunnable / OrchestratorRunnable) ✨ v0.13.0
  - Unified Composition (v0.15.0)
- [Document Chains](#document-chains)
- [Agents](#agents)
  - Agent Hooks ✨ v0.11.0
  - Agent Streaming ✨ v0.12.0
  - AgentBuilder ✨ v0.14.0
  - Orchestrator ✨ v0.14.0
  - ToolPolicy ✨ v0.14.0
- [Plan-Execute Agent](#plan-execute-agent)
- [Handoffs](#handoffs)
- [Streaming Tool Calls](#streaming-tool-calls)
- [Guardrails](#guardrails)
  - Guardable ✨ v0.15.0
  - Streaming Guardrails ✨ v0.15.0
  - Audit Persistence ✨ v0.15.0
- [Token Counter](#token-counter)
- [Sessions](#sessions)
  - Session Lifecycle ✨ v0.15.0
  - Attaching Memory ✨ v0.15.0
- [MCP](#mcp)
  - MCPServer
  - ConnectionManager ✨ v0.15.0
  - SamplingGuard ✨ v0.15.0
  - MCPGateway ✨ v0.15.0
- [Tools](#tools)
  - WikipediaTool
  - DuckDuckGoSearchTool
  - PythonREPLTool
  - Extended Tools (HTTPTool / FileTool / SQLTool)
  - `#[tool]` Procedural Macro ✨ v0.10.0
  - ToolRegistry ✨ v0.15.0
  - StructuredTool ✨ v0.15.0
  - SSRF Protection ✨ v0.15.0
- [RAG](#rag)
  - RAGPipeline ✨ v0.15.0
  - ChromaDB
  - PGVectorStore
  - PineconeStore
  - SemanticSplitter
  - Unified VectorStore Trait ✨ v0.15.0
- [BM25](#bm25)
- [Hybrid Retrieval](#hybrid-retrieval)
- [Document Loaders](#document-loaders)
  - HTMLLoader
  - DocxLoader ✨ v0.4.1
  - WebScraperLoader ✨ v0.4.1
  - SitemapLoader ✨ v0.4.1
- [MultiQueryRetriever](#multiqueryretriever)
- [HyDE Retriever](#hyde-retriever)
- [Reranking](#reranking)
- [Callbacks](#callbacks)
  - OtelHandler
- [Evaluation](#evaluation)
  - Evaluators (10 types)
  - EvalRunner
  - LLMAsJudge ✨ v0.15.0
  - PairwiseJudge ✨ v0.15.0
- [LangGraph](#langgraph)
  - Reducers ✨ v0.15.0
  - Edge Types ✨ v0.15.0
  - Checkpointer Family ✨ v0.15.0
  - Subgraph / Dynamic Planning / Streaming ✨ v0.15.0
- [A2A Agent Protocol](#a2a-agent-protocol) ✨ v0.4.1
- [with_structured_output](#with_structured_output) ✨ v0.4.1
- [FileVectorStore](#filevectorstore) ✨ v0.4.1
- [ComputerUseTool](#computerusetool) ✨ v0.4.1
- [v0.5.0 New Features](#v050-new-features) ✨ v0.5.0
  - RouterLLM (Model Routing + Fallback)
  - CorrectiveRAG
  - AdaptiveRAG
  - GraphRAG (Knowledge Graph RAG)
  - Deep Research Agent
  - MCP Protocol Primitives
  - Code Interpreter Sandbox
  - OpenAI Responses API
  - Anthropic Extended Thinking
  - Streaming Structured Output
  - Batch API
  - Tracing (Distributed Tracing)
  - v0.5.0 Quality Hardening (176 Fixes)
- [v0.5.2 Fixes](#v052-fixes) ✨ v0.5.2
- [Testing](#testing)
- [MongoDB Storage](#mongodb-storage)
- [Redis / SQLite Storage](#redis--sqlite-storage)

---

## Quickstart

> This tutorial builds a complete LLM application from scratch: first **chat**, then **remember context**, then **retrieve documents**, finally **call tools**. It is one continuous program — each section adds one capability on top of the previous one, so just read straight down.

### 1. Installation & environment

Add to `Cargo.toml`:

```toml
[dependencies]
langchainrust = "0.15"
tokio = { version = "1", features = ["full"] }
```

Set environment variables (OpenAI shown; switching providers only changes the env vars, see [LLM](#llm)):

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional, defaults to the official endpoint
```

### 2. First chat

The most direct usage: build an LLM, pass in a system prompt and a user message, get a reply.

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
            Message::system("You are a concise Rust assistant."),
            Message::human("Explain in one sentence what Rust is."),
        ],
        None,
    ).await?;

    println!("{}", response.content);
    Ok(())
}
```

Key points:
- All 11 providers implement the same `BaseChatModel` trait — switching providers changes only the `XxxConfig`;
- `Message::system` / `Message::human` build messages; `chat(Vec<Message>, None)` returns a full `LLMResult`;
- Want tokens as they're generated? Use `stream_chat()` or `config.streaming = true`, see [Streaming](#streaming).

### 3. Prompt templates

Template the system/user messages and fill variables at runtime instead of hard-coding strings:

```rust
use langchainrust::{ChatPromptTemplate, Message, Runnable};
use std::collections::HashMap;

// reuse `llm` from section 2
let prompt = ChatPromptTemplate::from_messages([
    Message::system("You are a translation assistant."),
    Message::human("Translate \"{text}\" into English."),
]);

let mut vars = HashMap::new();
vars.insert("text".to_string(), "你好,世界".to_string());

// ChatPromptTemplate is itself a Runnable — run it standalone, output is Vec<Message>
let messages = prompt.invoke(vars, None).await?;
```

Key points:
- Messages use `{variable}` placeholders, filled by a `HashMap<String, String>` at `invoke` time;
- A missing variable fails **loudly** — it never silently renders a broken prompt;
- Full guide: [Prompts](#prompts).

### 4. Compose a chain with LCEL

`Runnable`s compose with `.pipe()`. `prompt.pipe(llm).pipe(parser)` is a complete chain — one call returns the final answer:

```rust
use langchainrust::{ChatPromptTemplate, Message, OpenAIChat, Runnable, StrOutputParser};
use std::collections::HashMap;

// reuse `llm` from section 2
let prompt = ChatPromptTemplate::from_messages([
    Message::system("You are a concise Rust assistant."),
    Message::human("{question}"),
]);

let chain = prompt.pipe(llm).pipe(StrOutputParser::new());

let mut vars = HashMap::new();
vars.insert("question".to_string(), "What is the ownership system?".to_string());
let answer: String = chain.invoke(vars, None).await?;
println!("{answer}");
```

Key points:
- `StrOutputParser` extracts `content` from `LLMResult`, so the chain's output type becomes `String`;
- Four unified base operations: `invoke` / `batch` / `stream` / `transform`;
- All LCEL operators: [LCEL section](#lcel-langchain-expression-language-).

### 5. Add memory: make it remember you

`RunnableWithMessageHistory` wraps "read memory → assemble input → LLM → write back" into a single Runnable, so multi-turn chats need no manual history assembly:

```rust
use langchainrust::{
    ConversationBufferMemory, OpenAIChat, Runnable, RunnableWithMessageHistory, StrOutputParser,
};

// reuse `llm` from section 2
let memory = ConversationBufferMemory::new().with_return_messages(true);

let chat = RunnableWithMessageHistory::new(llm, memory).pipe(StrOutputParser::new());

let r1: String = chat.invoke("I'm Xiaoming, please remember me.".to_string(), None).await?;
let r2: String = chat.invoke("What's my name?".to_string(), None).await?;
// r2 answers "Xiaoming"
```

Key points:
- Input is a plain `String` — memory reads/writes are encapsulated inside the Runnable;
- The four memories trade off differently (full / sliding window / summary / summary+raw), see [Memory](#memory);
- Cross-process persistence via `MongoPersistentMemory`, see [MongoPersistentMemory](#mongopersistentmemory).

### 6. Hook up retrieval: RAG

`RAGPipelineBuilder` assembles "retrieve + generate", and `RagRunnable` makes it one link of a chain. This uses **BM25 local keyword search** — no vector database needed:

```rust
use langchainrust::{BM25Retriever, Document, OpenAIChat, RAGPipelineBuilder, RagRunnable, Runnable};
use std::sync::Arc;

// reuse `llm` from section 2
let mut retriever = BM25Retriever::new();
retriever.add_documents_sync(vec![
    Document::new("Rust is a systems programming language focused on safety and performance.").with_id("rust_intro"),
    Document::new("Rust's core features include ownership, borrow checking, and zero-cost abstractions.").with_id("rust_features"),
]);

let pipeline = RAGPipelineBuilder::new()
    .llm(llm)
    .retriever(retriever)
    .retrieve_k(2)
    .build()?;

let rag_chain = RagRunnable::new(Arc::new(pipeline));

let answer: String = rag_chain.invoke("What are Rust's core features?".to_string(), None).await?;
```

Key points:
- Only the answer generation calls the LLM; retrieval is fully local — runs with zero vector stores;
- Need citations? `RAGPipeline::query_with_sources()` returns every source, see [End-to-End RAGPipeline](#end-to-end-ragpipeline);
- Vector / BM25 / hybrid — which to pick: [Retrieval Mode Comparison](#retrieval-mode-comparison).

### 7. Give it tools: make it an agent

Make the app not just answer but **decide which tool to call**. `FunctionCallingAgent` reads the model's `tool_calls`; `AgentExecutor` executes them:

```rust
use langchainrust::tools::Calculator;
use langchainrust::{AgentExecutor, BaseAgent, BaseTool, FunctionCallingAgent, OpenAIChat};
use std::sync::Arc;

// reuse `llm` from section 2
let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
let agent = FunctionCallingAgent::new(llm, tools.clone(), None);

let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
    .with_max_iterations(3)
    .with_verbose(true);

let result = executor.invoke("What is 25 + 17?".to_string()).await?;
```

Key points:
- `FunctionCallingAgent` is the recommended path (native tool_calls); models without tool calling use `ReActAgent`;
- `max_iterations` caps the loop; tool timeouts and LLM retries are handled by the Executor, see [Agents](#agents).

### Next steps

| Want to do | Read |
|------------|------|
| Different models / streaming / structured output | [LLM](#llm) |
| Multi-turn memory & persistence | [Memory](#memory) |
| Full RAG & retrieval strategies | [RAG](#rag) · [BM25](#bm25) · [Hybrid Retrieval](#hybrid-retrieval) |
| MCP tool ecosystem | [MCP](#mcp) |
| Production guardrails / evaluation / tracing | [Guardrails](#guardrails) · [Evaluation](#evaluation) · [Callbacks](#callbacks) |

---

## LLM

This section covers how to connect to LLMs: instantiate any provider, streaming, function calling, and multimodal. All providers implement the same `BaseChatModel` trait with identical APIs — get your flow working with one provider first, then swap to another anytime without touching business code. New here? See [Quickstart](#quickstart) section 2.

### Multi-Provider Support

LangChainRust supports multiple LLM providers with unified API:

| Provider | Class | Features |
|----------|-------|----------|
| **OpenAI** | `OpenAIChat` | GPT-4, GPT-3.5-turbo |
| **DeepSeek** | `DeepSeekChat` | DeepSeek-V3, cost-effective |
| **Moonshot** | `MoonshotChat` | Kimi, long context |
| **Qwen** | `QwenChat` | Alibaba Cloud |
| **Zhipu** | `ZhipuChat` | ChatGLM |
| **Anthropic** | `AnthropicChat` | Claude, safety-focused |
| **Ollama** | `OllamaChat` | Local deployment |
| **Gemini** | `GeminiChat` | Google Gemini, multimodal |
| **Azure** | `AzureChat` | Azure OpenAI, enterprise compliance |
| **Cohere** | `CohereChat` | Command R+, RAG scenarios |
| **Mistral** | `MistralChat` | Mistral Large/Medium |

#### Unified Client & Auto-Discovery ✨ v0.15.0

`LLMClient::from_env()` auto-detects env vars for all 11 providers — zero-config switching; `LLMClient::from_llm(provider)` wraps any `BaseChatModel` manually. `ChatModelWrapper` / `wrap_chat_model` provide trait-object wrapping.

```rust
use langchainrust::LLMClient;
use langchainrust::language_models::ProviderError;

// Auto-detect: whichever provider has env vars configured wins
let llm = LLMClient::from_env()?;
let response = llm.chat(vec![Message::human("Hello")], None).await?;

// Any native provider can be wrapped too
let client = LLMClient::from_llm(DeepSeekChat::from_env());
```

Errors are unified as `ProviderError`, with per-vendor variants (OpenAI / Anthropic / Gemini / Azure / Cohere / Ollama / DeepSeek / Qwen / Moonshot / Zhipu / Mistral); `config.streaming` decides whether `chat()` takes the streaming or plain path.

#### DeepSeek (Cost-Effective)

```rust
use langchainrust::{DeepSeekChat, BaseChatModel};
use langchainrust::schema::Message;

// From environment
let llm = DeepSeekChat::from_env();

// Or manual config
let llm = DeepSeekChat::with_model("deepseek-chat");

let response = llm.chat(vec![
    Message::human("Explain Rust ownership"),
], None).await?;
```

#### Moonshot (Long Context)

```rust
use langchainrust::MoonshotChat;

let llm = MoonshotChat::with_model("moonshot-v1-128k");  // 128K context

let response = llm.chat(vec![
    Message::human("Analyze this long document..."),
], None).await?;
```

#### Qwen

```rust
use langchainrust::QwenChat;

let llm = QwenChat::from_env();  // Or QwenChat::with_model("qwen-plus")

let response = llm.chat(vec![
    Message::human("Explain microservices in Chinese"),
], None).await?;
```

#### Zhipu (ChatGLM)

```rust
use langchainrust::ZhipuChat;

let llm = ZhipuChat::from_env();  // Or ZhipuChat::with_model("glm-4")

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

Use OpenAI GPT series models. Supports custom `base_url` (compatible with all OpenAI API format services), `temperature` controls randomness.

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

### Streaming

LLMs generate text token by token. Streaming lets you see each token in real time instead of waiting for the complete response. Ideal for chat interfaces and real-time display.

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
        print!("{}", token);  // Real-time output
    }
}
```

### Function Calling

Let the LLM decide when to call tools. `bind_tools` attaches tool definitions to the LLM, which returns `tool_calls` instead of plain text. The framework handles parsing arguments, calling tools, and returning results.

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

### Ollama (Local LLM)

Ollama lets you run open-source models (Llama, Mistral, etc.) locally — no API key needed, data never leaves your machine. Good for privacy-sensitive scenarios or offline use.

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

### Multimodal Vision

`ImageContent` represents an image (URL or base64 data URI). Build image-bearing messages with `Message::human_with_image`; both `OpenAIChat` and `OllamaChat` serialize them to their native multimodal formats automatically.

```rust
use langchainrust::schema::{ImageContent, Message};
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};

let msg = Message::human_with_image("Describe this image", "https://example.com/cat.jpg");
// Or multiple images:
// let msg = Message::human_with_images("Compare these two", vec![
//     ImageContent::from_url("https://example.com/a.jpg"),
//     ImageContent::from_base64_with_mime(base64_str, "image/png"),
// ]);

let llm = OpenAIChat::new(OpenAIConfig::default());
let resp = llm.chat(vec![msg], None).await?;
println!("{}", resp.content);
```

`ImageContent::from_url(url)` / `from_base64(data)` / `from_base64_with_mime(data, mime)`; you can also chain with `Message::human(text).with_image(ImageContent)`. Same for `OllamaChat`.

---

### Message Structure ✨ v0.15.0

`Message` is the unified conversation-message struct — besides the textual `content` it carries multimodal attachments and tool calls:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `String` | Text content |
| `images` / `audio` / `files` | `Vec<...>` | Image / audio / file attachments |
| `message_type` | `MessageType` | System / Human / Ai |
| `tool_calls` | `Option<Vec<ToolCall>>` | Tool calls to execute, carried on AI messages |
| `name` / `id` / `additional_kwargs` | — | Role name / message ID / extra fields |

```rust
use langchainrust::schema::{Message, AudioContent, FileContent, ToolCall};

// Message with audio / file attachment
let msg = Message::human_with_audio("Transcribe this audio", AudioContent::from_base64(data));
let msg = Message::human_with_file("Read this file", FileContent::from_url("file:///tmp/doc.pdf"));

// AI initiates tool calls
let msg = Message::ai_with_tool_calls("", vec![
    ToolCall::builder("call_1")
        .name("calculator")
        .arguments(r#"{"expression":"25+17"}"#)
        .build(),
]);
```

Constructors cover the common combinations: `Message::system/human/ai`, `human_with_image(s)`, `human_with_audio`, `human_with_file`, `ai_with_tool_calls`; serde stays backward-compatible (attachment fields carry `#[serde(default)]`, so old data still deserializes).

### MultimodalModel ✨ v0.15.0

The `MultimodalModel` trait (extension of `BaseChatModel`) declares three capability interfaces: speech recognition / speech synthesis / text-to-image. **The default implementation returns `MultimodalError::Unsupported`** — only providers that genuinely support a capability override it, avoiding fake "looks supported, actually errors" multimodal behavior. The OpenAI family is implemented; other providers get an explicit Unsupported error on those methods instead of a silent failure.

```rust
use langchainrust::MultimodalModel;

let text = llm.transcribe(AudioContent::from_base64(data)).await?; // supported providers only
// let audio = llm.generate_speech("hello").await?;
// let img = llm.generate_image("a cat").await?; → Err(Unsupported) when unsupported
```

---

### OpenAI Assistants API

`OpenAIAssistant` wraps the official OpenAI Assistants API (Assistants / Threads / Run) with server-side session state, suited for multi-turn complex tasks. Requires the OpenAI official endpoint; some compatible-mode endpoints may not support it.

```rust
use langchainrust::{OpenAIAssistant, OpenAIConfig};

let config = OpenAIConfig::default();
let assistant = OpenAIAssistant::create(&config, "gpt-4o", "You are a translator").await?;
// or reuse: OpenAIAssistant::from_id(config, "asst_xxx")

let answer = assistant.run_once("Translate: Hello").await?;
```

**Limitation**: Run with tool calls (`requires_action`) is not implemented; returns `AssistantError::RequiresAction`. Use `FunctionCallingAgent` for tool calls.

## Prompts

Prompt templates replace `{variable}` placeholders with actual values, avoiding manual string concatenation. The framework provides three templates covering all scenarios from simple to complex.

### PromptTemplate

The most basic template — a single text string with `{variable}` placeholders. Use when you don't need role distinction, just need to compose a prompt.

```rust
use langchainrust::prompts::PromptTemplate;
use std::collections::HashMap;

let template = PromptTemplate::new("Hello, {name}! Today is {day}.");

let vars = HashMap::from([
    ("name", "Alice"),
    ("day", "Monday"),
]);

let prompt = template.format(&vars)?;
// Output: "Hello, Alice! Today is Monday."
```

### ChatPromptTemplate

Multi-message template — each message has a role (system/human/ai), with variables replaced in the text. Use when you need to set a system role or distinguish conversation turns. This is the most commonly used template in Agents and Chains.

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

Few-shot template — inserts "input→output" examples into the prompt to teach the LLM a specific format. Use when you need to guide the output format (e.g., translation, sentiment analysis, format conversion). The LLM sees the examples and mimics their format.

**How it works**: Concatenates prefix + each example (formatted via `example_prompt`) + suffix into a complete prompt. The LLM sees a full text containing examples.

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

When you have many examples, you don't need to send them all to the LLM — selectors pick the most relevant ones by strategy, saving tokens and improving quality.

```rust
use langchainrust::prompts::LengthBasedExampleSelector;

// Length-based: selects examples up to max length
let selector = LengthBasedExampleSelector::new(examples).with_max_length(50);
```

---

## Output Parsers

LLMs return plain text strings. Output parsers convert them into structured data. Choose based on what format you need:

| Parser | Input | Output | Use Case |
|--------|-------|--------|----------|
| `StrOutputParser` | Any text | String as-is | Just need text, no conversion |
| `CommaSeparatedListOutputParser` | Comma-separated text | `Vec<String>` | LLM outputs a list |
| `JsonOutputParser` | JSON text | `serde_json::Value` | Need flexible JSON structure |
| `StructuredOutputParser` | `key: value` text | `HashMap<String, String>` | Simple key-value pairs, no JSON needed |
| `TypedOutputParser<T>` | JSON text | Strongly typed `T` | Need type-safe structured output |

> **Tip**: If the LLM supports Function Calling, prefer `with_structured_output()` — it's more reliable than parsers.

### StrOutputParser

The simplest parser — returns text as-is. Typically used as the last step in an LCEL pipeline to ensure the output type is `String`.

```rust
use langchainrust::output_parsers::{StrOutputParser, BaseOutputParser};

let parser = StrOutputParser::new();
let result = parser.parse("Hello world")?;
```

### CommaSeparatedListOutputParser

Parses comma-separated text into a string list. Use when you want the LLM to enumerate items, tags, keywords, etc.

```rust
use langchainrust::output_parsers::CommaSeparatedListOutputParser;

let parser = CommaSeparatedListOutputParser::new();
let result = parser.parse("apple, banana, cherry")?;
```

### JsonOutputParser

Extracts JSON from LLM output. Supports both complete JSON and partial extraction from markdown code blocks (LLMs often wrap JSON in ` ```json ``` `).

```rust
use langchainrust::output_parsers::JsonOutputParser;
use serde_json::Value;

// Full JSON parsing
let parser = JsonOutputParser::<Value>::new();
let result: Value = parser.parse(r#"{"name": "Rust"}"#)?;

// Partial parsing (extract JSON from markdown)
let partial = parser.parse_partial("Here is the JSON:\n```json\n{\"name\": \"Rust\"\n}")?;
```

### StructuredOutputParser

Parses `key: value` format text into a HashMap. More lenient than JsonOutputParser — the LLM doesn't need to output strict JSON, just write `key: value` per line.

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

Deserializes JSON text into a strongly typed struct. Requires `T` to implement `Deserialize`. Safer than `JsonOutputParser<Value>` — field types are checked at compile time.

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
## Memory

Memory gives chains and agents "context": letting multi-turn conversations remember what was said before, without stuffing the whole history into the prompt each time. The four built-in memories each trade something off:

| Memory | Behavior | Best for | Cost |
|--------|----------|----------|------|
| `ConversationBufferMemory` | Keeps all conversation | Short conversations, no detail may be lost | Token usage grows linearly with turns |
| `ConversationBufferWindowMemory` | Keeps only the last k turns | Long conversations, old detail unimportant | Old content is dropped |
| `ConversationSummaryBufferMemory` (recommended) | Summarizes old messages + recent verbatim | Long conversations that still need recent detail | Summarizing costs one LLM call |
| `VectorStoreRetrieverMemory` | Retrieves memories by similarity | Knowledge-type, associative memory | Requires a vector store |

- Need cross-process persistence → use `MongoPersistentMemory` (below);
- Need to cap per-call context length → use `ContextWindow` for automatic truncation/summarization;
- All memories implement the unified `BaseChatMemory` trait — plug-and-play interchangeable.

### ConversationBufferMemory

Keeps all conversation history:

```rust
use langchainrust::{ConversationBufferMemory, BaseMemory};

let mut memory = ConversationBufferMemory::new();

memory.save_context(
    HashMap::from([("input", "My name is Alice")]),
    HashMap::from([("output", "Hello Alice!")]),
).await?;

let vars = memory.load_memory_variables(&HashMap::new()).await?;
// Output: "Human: My name is Alice\nAI: Hello Alice!"
```

### ConversationBufferWindowMemory

Keeps only the last k conversation turns. Use when the conversation is long and you don't need the full history, to avoid exceeding token limits.

```rust
use langchainrust::ConversationBufferWindowMemory;

// k=2, keep last 2 turns (4 messages)
let mut memory = ConversationBufferWindowMemory::new(2);

for i in 1..=5 {
    memory.save_context(
        HashMap::from([("input", format!("Question {}", i))]),
        HashMap::from([("output", format!("Answer {}", i))]),
    ).await?;
}

// Only returns last 2 turns, Q1-Q3 are dropped
let vars = memory.load_memory_variables(&HashMap::new()).await?;
```

### ConversationSummaryBufferMemory (Recommended)

Summarizes old messages while keeping recent ones in full. Combines the benefits of BufferMemory (preserving recent details) and SummaryMemory (compressing old content). The best choice for long conversations.

```rust
use langchainrust::ConversationSummaryBufferMemory;

let llm = OpenAIChat::new(config);

// max_token_limit = 100, triggers compression when exceeded
let mut memory = ConversationSummaryBufferMemory::new(llm, 100);

for i in 1..=10 {
    memory.save_context(&inputs, &outputs).await?;
}

// Returns: "Summary: User discussed...\n\nHuman: Recent\nAI: Response"
let vars = memory.load_memory_variables(&HashMap::new()).await?;
```

| Memory Type | Compression | Token Control | Use Case |
|-------------|-------------|---------------|----------|
| BufferMemory | None | Unlimited | Short conversations |
| WindowMemory | Hard delete | Fixed k | Simple control |
| SummaryMemory | LLM summary | Dynamic | Long conversations |
| SummaryBufferMemory | Hybrid | Dynamic + keep recent | Balanced (recommended) |

---

### VectorStoreRetrieverMemory

Embeds each turn into a vector store and recalls top-k relevant history by semantic similarity to the current input. Compared to fixed-window buffer memory, it preserves more useful context in long / cross-session conversations.

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

**Trade-off**: semantic recall keeps key info in long chats; depends on a vector store + embedding model (extra cost).

### Unified `BaseChatMemory` Trait ✨ v0.15.0

All conversation memories implement the unified `BaseChatMemory` trait (`save_context` / `load_memory_variables` / `clear`), so they are interchangeable and can enter the LCEL pipeline directly (`RunnableWithMessageHistory::new(llm, memory)` accepts any `BaseMemory`).

<a id="mongopersistentmemory"></a>
### MongoPersistentMemory (Cross-Process Persistence) ✨ v0.15.0

Persists conversation history to MongoDB — survives server restarts, and multiple instances share the same memory. Internally composes `ConversationSummaryBufferMemory` with a built-in token budget; concurrent writes use an optimistic lock to prevent lost updates.

```rust
use langchainrust::memory::MongoPersistentMemory;

let mut memory = MongoPersistentMemory::new(
    "mongodb://localhost:27017",
    "chatdb",
    "sessions",
    llm,        // any BaseChatModel, generic M
    2000,       // token limit
).await?;

memory.set_session_id_async("user-123".to_string()).await;  // bind session
memory.save_context(&inputs, &outputs).await?;
```

### Summary-Failure Visibility

`ConversationSummaryMemory` / `ConversationSummaryBufferMemory` expose `last_summary_error() -> Option<&str>`: an LLM summarization failure is never swallowed — callers can read the last failure reason and decide a degradation strategy.

### ContextWindow (Long Context Management) ✨ v0.4.1

`ContextWindow` manages token budget for long conversations with two strategies: Truncate and Summarize.

```rust
use langchainrust::{ContextWindow, Message, OpenAIChat, Strategy};

// Strategy 1: Truncate — discard oldest messages when over token budget
let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096)?;
let fitted = cw.fit(messages).await?;

// Strategy 2: Summarize — use LLM to compress old conversation when over budget
let cw = ContextWindow::with_strategy(4096, Strategy::summarize(llm))?;
let fitted = cw.fit(messages).await?;
```

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `Truncate` | Discard oldest messages over budget | Simple scenarios |
| `Summarize` | LLM compresses old conversation into summary | Long conversations needing key info |

## LLM Cache

### Concept: Why Cache

LLM calls are the slowest and most expensive part of your application — a single request crosses the network, waits in a queue, and generates tokens, costing both time and money. When users keep asking the same question, or a batch job produces many duplicate requests, calling the real API every time is slow and drains your budget. The idea behind caching is simple: **reuse the previous result for the same input** instead of making a real LLM call. The cache returns previous results for identical inputs, supporting TTL expiration and capacity limits.

When to use:

- High-frequency repeated queries (e.g. the same question sent to many users, repeatedly summarizing the same batch of documents)
- Batch processing / evaluation with many similar or identical inputs
- When you need deterministic results and can tolerate occasional staleness

Workflow (2 steps):

1. Build a `CacheConfig`, declaring the TTL (time-to-live) and the capacity limit
2. Create an `LLMCache` and read/write it yourself with `build_key` + `get`/`put` (`LLMCache` is a standalone component — it does not auto-attach to a model)

### How It Works

#### Cache Key Logic

The cache keys on the call input: when the input is exactly the same (e.g. the same message sequence), it hits and reuses the previous result; a different input is treated as a new entry. The key determines whether the cache "recognizes" a request — for the cache to work, repeated calls must keep their input consistent.

#### TTL Expiration

Each record carries a TTL (time-to-live) and automatically expires once it is exceeded. `with_ttl` sets the global expiration time. Expired entries are cleaned up on access / eviction, so "stale answers don't hold their slot forever."

#### Capacity Limit + LRU Eviction

`with_max_entries` declares the maximum number of entries to cache. When the cache is full, old entries must be evicted — it uses **LRU (least recently used) eviction**: evict the entry "not accessed for the longest time" rather than the one "inserted first." This way, hot entries that are frequently hit won't be squeezed out by new entries, and the hit rate stays stable. Since v0.14.0, a cache hit refreshes the entry's "last-used time," so eviction semantics are true LRU.

#### Hit Refresh

Each cache hit (get) refreshes the entry's "last-used time," keeping LRU semantics correct — an entry just used "becomes younger" and won't be evicted by the next insert.

### In-Memory Cache with TTL

```rust
use langchainrust::core::cache::{CacheConfig, LLMCache};
use std::time::Duration;

let config = CacheConfig::new()
    .with_ttl(Duration::from_secs(3600))  // 1 hour
    .with_max_entries(1000);              // 1000 entries

let cache = LLMCache::with_config(config);

// LLMCache is standalone — wire it into your call path manually:
// build_key to key, get/put to read and write
let messages = vec![Message::human("Hello")];
let key = LLMCache::build_key(&messages, "gpt-4o")?;

if let Some(hit) = cache.get(&key).await {
    // Cache hit, use the cached result
    let result = hit.result;
} else {
    // Miss, make the real call and store the result
    let result = llm.chat(messages, None).await?;
    cache.put(key, result).await;
}
```

### Key Behaviors at a Glance

| Behavior | Description |
|---|---|
| Cache key | Same input reuses the previous result, no real call |
| TTL expiration | Automatically expires after the time-to-live |
| Capacity limit | `with_max_entries` caps the number of cached entries; eviction triggers when full |
| LRU eviction | Evicts the "least recently used" entry; hot entries stay |
| Hit refresh | A get hit refreshes the last-used time, keeping LRU semantics correct |

### How to Choose / Usage Tips

- The more stable the result and the more repeated the requests, the bigger the cache payoff; a longer `with_ttl` covers more repeated requests.
- Don't cache data that is sensitive to freshness (prices, inventory, latest status), or shorten the TTL.
- This is an **in-memory** cache — it is cleared when the process restarts; for cross-process / restart persistence, put "key → result" into external storage.

---

<a id="lcel-langchain-expression-language-"></a>
## LCEL (LangChain Expression Language) ✨ v0.9.0

LCEL provides Python LangChain-style pipe composition: chain `Runnable` components into pipelines with `.pipe()`. Unlike handwritten glue code that takes a result and passes it to the next step, the **whole chain you compose is itself a Runnable** — you can keep composing it, batch it, stream it, and add automatic retry / fallback on top.

Think of each LLM call as a work station, and LCEL as the **conveyor belt** — `.pipe()` connects prompt templates, models, parsers, memory and retrievers so data flows through automatically; each segment you add gives the chain one more capability.

### Why LCEL

Before v0.14.0, whether a component "could enter a chain" was inconsistent, forcing users to write `.content` extraction, hand-assemble messages, and wrap things in boilerplate:

| Component | Before v0.14.0 | Symptom |
|---|---|---|
| Parser | `Runnable<String, String>` | Can't take `LLMResult`; `llm.pipe(parser)` doesn't compile |
| Prompt | no Runnable | `ChatPromptTemplate` can only be `format`ed by hand, can't enter a chain |
| Memory | no Runnable | Handwritten "read memory → call model → write back" glue |
| Native provider | errors not in `LcelError` | Only `LLMClient`-wrapped providers can `pipe` |

Since v0.15.0, prompts, memory, native providers, parsers and RAG are all `Runnable` — one chain runs the whole flow, no glue code. See [Unified Composition (v0.15.0)](#unified-lcel).

### Runnable: the unified "executable unit"

Everything in the framework — prompt templates, chat models, output parsers, agents — used to have different calling conventions. The `Runnable` trait gives them one interface, so any component executes the same way and can be composed with any other.

**Four base actions**:

| Action | What it does | Example |
|---|---|---|
| `invoke` | Single run: one input → one output | Ask the model once, get one answer |
| `batch` | Batch run: N inputs → N outputs | Tag 100 comments in one go |
| `stream` | Streaming: outputs arrive one by one | Model generates text as it renders |
| `transform` | Stream-to-stream: input stream → output stream | Chunk-process a long document without reading it all |

Key behaviors:
- **Every component gets all four for free** — even one that only implements `invoke` has default implementations for the other three (by default, one `invoke` call wrapped as a stream / repeated in a loop).
- `batch` supports **concurrency**: control how many run at once via `RunnableConfig.max_concurrency`; unset means sequential.
- True token-by-token streaming (first-token latency) requires the component to override `stream` — language models do; plain functions don't.

### RunnableConfig: the "settings sheet" for one run

Each run can carry extra info — which business it belongs to, whether callbacks are attached, whether it can be cancelled. Instead of changing code, pass a `RunnableConfig`.

| Field | What it does |
|---|---|
| `tags` | Tag the run for filtering/tracing (e.g. `"user-123"`, `"rag-flow"`) |
| `metadata` | Arbitrary key-value pairs for business info |
| `max_concurrency` | Parallelism for batch runs |
| `run_id` / `run_name` | Unique ID and name for this run, for tracing |
| `callbacks` | Attach a callback manager to listen to run events |
| `cancellation_token` | Cancellation token; long tasks can be stopped externally |

Config merge rules (parent + child):
- **Tags**: merged in order, de-duplicated (parent order kept, duplicates removed, not sorted).
- **Other fields**: child value **overrides** parent.
- **Callbacks**: parent and child **both** fire.

### Composition operators: build pipelines from Runnables

Real applications aren't "call the model once" — they're "retrieve → build prompt → call model → parse result". These operators are the building blocks: compose simple units into complex flows, and the composed thing is itself a Runnable you can keep composing. Composition is **type-safe** — the compiler enforces "previous output type == next input type", so a mismatch fails to compile.

| Operator | What it does | Example |
|---|---|---|
| `pipe` (chain) | Feed A's output into B, step by step | prompt → model → parser in one line |
| `RunnableLambda` (wrap fn) | Put a plain function into the pipeline | custom logic like text cleaning |
| `RunnablePassthrough` (pass-through) | Pass input through unchanged | keep the "question" alive all the way through RAG |
| `RunnableParallel` (parallel) | Run the same data down several lines, merge into one map | one line retrieves docs, another keeps the question |
| `RunnableBranch` (branch) | Pick a branch by input content, default otherwise | refunds → support branch, technical → tech branch |
| `RunnableBinding` (bind) | Bind fixed config to a unit | the whole chain uses one model config |
| `RunnableAssign` (assign) | Append new fields to the data | attach an uppercased copy to the question |
| `RunnableWithFallbacks` (fallback) | Try backup units when the primary fails | auto-switch to a backup model when the main one is down |
| `with_retry` (retry) | Auto-retry on failure with backoff | network jitter / transient rate limits don't kill the flow |

### Basic Pipe

```rust
use langchainrust::{
    RunnableExt, RunnableLambda, RunnablePassthrough,
};

// Simple pipeline: input -> double -> format as string
let doubler = RunnableLambda::new_sync(|x: i32| x * 2);
let formatter = RunnableLambda::new_sync(|x: i32| format!("Result: {}", x));

let chain = doubler.pipe(formatter);
let result = chain.invoke(5, None).await?;
// result = "Result: 10"
```

### Three-Step Pipeline (Prompt | LLM | Parser)

```rust
use langchainrust::{RunnableExt, RunnableLambda, StrOutputParser};

let prompt = RunnableLambda::new_sync(|query: String| {
    format!("Answer the following question: {}", query)
});
let parser = RunnableLambda::new_sync(|output: String| {
    output.trim().to_string()
});

// prompt.pipe(llm).pipe(parser) — LLM step requires a real API
let chain = prompt.pipe(parser);
let result = chain.invoke("What is Rust?".to_string(), None).await?;
```

### RunnableLambda (Wrap a Function)

Plain functions can enter the pipeline too: `new_sync` wraps a sync closure, `new_sync_fallible` a fallible one, `new_async` an async one. So any custom logic — text cleaning, string assembly, an HTTP fetch — becomes a segment of the chain, with the return value automatically wrapped as `Result<_, LcelError>`.

```rust
use langchainrust::{LcelError, RunnableExt, RunnableLambda};

// new_sync: sync closure, output auto-wrapped in Ok
let clean = RunnableLambda::new_sync(|s: String| s.trim().to_string());

// new_async: async closure returning Result<O, LcelError>
let fetch = RunnableLambda::new_async(|url: String| async move {
    Ok(format!("fetched {}", url.trim()))
});

let chain = clean.pipe(fetch);
let result = chain.invoke("  https://example.com  ".to_string(), None).await?;
// result = "fetched https://example.com"
```

### RunnablePassthrough (Pass-through)

```rust
use langchainrust::RunnablePassthrough;

// Passthrough passes input through unchanged
let passthrough = RunnablePassthrough::<String>::new();
let result = passthrough.invoke("hello".to_string(), None).await?;
// result = "hello"

// True streaming: transform passes input stream through without buffering
let stream = passthrough.transform(input_stream, None).await;
```

### RunnableParallel (Fan-out / Fan-in)

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

### RunnableBranch (Conditional Routing)

```rust
use langchainrust::{RunnableExt, RunnableLambda, RunnableBranch};

let short_handler = RunnableLambda::new_sync(|s: String| format!("Short: {}", s));
let long_handler = RunnableLambda::new_sync(|s: String| format!("Long: {}", s));
let default_handler = RunnableLambda::new_sync(|s: String| format!("Default: {}", s));

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
// result = "Short: hi"
```

### RunnableBinding (Config Binding)

```rust
use langchainrust::{RunnableBinding, RunnableConfig};

// Pre-bind config and kwargs
let bound = runnable
    .bind("temperature", serde_json::json!(0.7))
    .with_config(RunnableConfig::new().with_tag("production"));
let result = bound.invoke(input, None).await?;
```

### Batch Execution

```rust
let results = chain.batch(vec![1, 2, 3], None).await?;
// results = ["Result: 2", "Result: 4", "Result: 6"]
```

### Stream Execution

```rust
use futures_util::StreamExt;

let mut stream = chain.stream("hello".to_string(), None).await?;
while let Some(item) = stream.next().await {
    println!("Token: {}", item?);
}
```

### RunnableWithFallbacks (Fallback on Failure) ✨ v0.10.0

```rust
use langchainrust::{RunnableExt, RunnableLambda};

let primary = RunnableLambda::new_sync(|x: i32| -> i32 {
    if x < 0 { panic!("negative") } else { x * 2 }
});
let fallback = RunnableLambda::new_sync(|x: i32| x.abs() * 2);

// Automatically falls back when primary fails
let chain = primary.with_fallbacks(vec![fallback.into_runnable_any()]);
let result = chain.invoke(-5, None).await?;
// result = 10 (fallback executed)
```

### RunnableAssign (Field Injection) ✨ v0.10.0

```rust
use langchainrust::{
    RunnableExt, RunnableLambda, RunnableParallel, RunnablePassthrough,
    core::runnables::RunnableAssign,
};
use std::collections::HashMap;
use serde_json::Value;

// RunnableParallel.assign() — inject new fields into the parallel output HashMap
let parallel = RunnableParallel::new()
    .with("question", RunnablePassthrough::<String>::new())
    .with("context", RunnableLambda::new_sync(|_: String| "some context".to_string()));

// assign appends a field after the parallel output
let chain = parallel.assign("answer", RunnableLambda::new_sync(|map: HashMap<String, Value>| {
    let ctx = map.get("context").unwrap().as_str().unwrap();
    format!("Based on: {}", ctx)
}));

let result = chain.invoke("What is Rust?".to_string(), None).await?;
// result = {"question": "What is Rust?", "context": "some context", "answer": "Based on: some context"}
```

### RunnableRetry (Automatic Retry) ✨ v0.11.0

`with_retry(RetryConfig)` wraps any Runnable and retries automatically with exponential backoff on failure.

```rust
use langchainrust::{
    RunnableExt, RunnableLambda, core::runnables::{RetryConfig, RetryOn},
};
use std::time::Duration;

let flaky = RunnableLambda::new_sync(|x: i32| {
    if rand::random::<f32>() < 0.3 { panic!("transient") } else { x }
});

// Default: max 3 retries, exponential backoff 0.5s→10s, transient errors only
let chain = flaky.with_retry(RetryConfig::default());

// Custom: max 5 retries, initial 100ms, multiplier 2.0, all errors
let config = RetryConfig::new(5)
    .with_initial_delay(Duration::from_millis(100))
    .with_max_delay(Duration::from_secs(5))
    .with_backoff_multiplier(2.0)
    .with_retry_on(RetryOn::AllErrors);
let chain = flaky.with_retry(config);
```

- `RetryOn::TransientErrors` (default) — retry transient errors only: HTTP 429 / 500 / 502 / 503 / 504, plus rate limit, timeout, connection reset, etc.
- `RetryOn::AllErrors` — retry on every error
- `RetryOn::Custom(predicate)` — custom decision

### CancellationToken ✨ v0.11.0

A cross-task shared cancellation flag: after `cancel()`, all clones become cancelled simultaneously; long-running tasks poll `is_cancelled()` to exit gracefully.

```rust
use langchainrust::core::runnables::CancellationToken;

let token = CancellationToken::new();
let cloned = token.clone();

// Auto-cancel after a timeout
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(30)).await;
    cloned.cancel();
});

// Inject into the Runnable config
let config = RunnableConfig::default().with_cancellation_token(token.clone());
let result = chain.invoke(input, Some(config)).await?;

// Poll explicitly inside a loop
if token.is_cancelled() {
    return Ok("stopped by cancellation".to_string());
}
```

`await token.cancelled()` suspends until cancellation fires (lightweight spin, does not block the thread).

### Adapters (Bridge Existing Components to LCEL)

```rust
use langchainrust::{ChainRunnable, AgentRunnable, RagRunnable};

// Chain adapter
let chain_runnable = ChainRunnable::new(arc_chain);
let result = chain_runnable.invoke(input_map, None).await?;

// Agent adapter
let agent_runnable = AgentRunnable::new(arc_agent_executor);
let result = agent_runnable.invoke("query".to_string(), None).await?;

// RAG adapter
let rag_runnable = RagRunnable::new(arc_rag_pipeline);
let result = rag_runnable.invoke("query".to_string(), None).await?;
```

**AgentEventRunnable** ✨ v0.13.0

Unlike `AgentRunnable`, `AgentEventRunnable::stream()` preserves **all** `AgentStreamEvent` variants (`Text` / `ToolCall` / `ToolStart` / `ToolEnd` / `PipelineStep` / `FinalAnswer` / `Error`) instead of filtering down to the final answer; the non-streaming `invoke()` returns a single `FinalAnswer` event.

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

// stream preserves all event variants
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

// invoke returns a single FinalAnswer
if let AgentStreamEvent::FinalAnswer { content } =
    agent.invoke("What is Rust?".to_string(), None).await?
{
    println!("{}", content);
}
```

**OrchestratorRunnable** ✨ v0.13.0

Wraps high-level orchestrators (`PlanExecuteAgent` / `AdaptiveRAG` / `CorrectiveRAG` / `DeepResearch` / `FanOutFanIn` / `SequentialPipeline` / `TaskAdapter` / `ReviewOrchestrator`) as a `Runnable`, letting them enter LCEL pipelines. `config.metadata["trace_id"]` is propagated through to the orchestrator's `RunContext`.

```rust
use langchainrust::{BaseTool, OrchestratorRunnable, PlanExecuteAgent, Runnable, RunnableConfig};
use std::sync::Arc;

let tools: Vec<Arc<dyn BaseTool>> = vec![];
let plan_exec = PlanExecuteAgent::new(llm, tools);
let runnable = OrchestratorRunnable::new(plan_exec);

// trace_id propagates to RunContext
let config = RunnableConfig::new()
    .with_metadata("trace_id".to_string(), serde_json::json!("trace-001"));
let result: String = runnable.invoke("Research Rust async runtimes".to_string(), Some(config)).await?;
```

<a id="unified-lcel"></a>
### Unified Composition (v0.15.0) — prompt / memory / LLM / parser / RAG all pipeable

v0.15.0 unifies the framework's core capabilities into `Runnable` — one chain can run "prompt + memory + LLM + parser + RAG" end to end, with no handwritten glue code. Four changes:

1. **5 output parsers now take `LLMResult`** — `StrOutputParser` / `JsonOutputParser` / `CommaSeparatedListOutputParser` / `StructuredOutputParser` / `TypedOutputParser`'s invoke reads `input.content` and runs the original `parse`, so `llm.pipe(parser)` compiles.
2. **`ChatPromptTemplate` implements `Runnable`** — as the first chain segment: input variables map → output `Vec<Message>`.
3. **`RunnableWithMessageHistory`** — wraps "LLM + memory" into a single `Runnable<String, LLMResult>` that automatically reads history → builds input → calls LLM → writes back.
4. **Native provider errors unified** — `OpenAIChat` / `QwenChat` / `DeepSeekChat` errors merge into `LcelError`, so `pipe` works directly without wrapping in `LLMClient`.

**What can be piped now**:

| Component | Runnable form | Position in the chain |
|---|---|---|
| Prompt | `ChatPromptTemplate` | First segment: variables map → message list |
| Memory | `RunnableWithMessageHistory` | Wraps "LLM + memory"; auto reads history → builds input → calls LLM → writes back |
| LLM | native `OpenAIChat` / `QwenChat` / `DeepSeekChat` | Middle segment; errors unified into `LcelError`, no `LLMClient` wrapper needed |
| Parser | Str / Json / List / Structured / Typed | Last segment; takes `LLMResult` directly, auto-extracts `content` |
| RAG | `RagRunnable` | A whole segment: question in, answer out |
| Error | `LcelError` | Unified across the chain; parser / provider / chain errors converge into one type |

**Typical composition shapes**:

- **Plain QA chain**: prompt → LLM → parser. Variables map in, string answer out.
- **Multi-turn chat chain**: memory → LLM → parser. User message in, answer out, history read/written automatically.
- **RAG chain**: retrieve → generate. Question in, answer grounded in retrieved material out.
- **Combined**: all of the above coexist in one program, sharing one LLM instance, to form a full conversational RAG assistant.

**P0 core chain: prompt + LLM + parser**

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
    Message::system("You are a concise Rust assistant. Output conclusions only."),
    Message::human("{question}"),
]);
let chain = prompt.pipe(llm).pipe(StrOutputParser::new());

let mut vars = HashMap::new();
vars.insert("question".to_string(), "Explain what Rust is in one sentence".to_string());
let answer = chain.invoke(vars, None).await?;
```

**Multi-turn chain: memory + LLM + parser**

```rust
use langchainrust::{
    ConversationBufferMemory, RunnableExt, RunnableWithMessageHistory, StrOutputParser,
};

let memory = ConversationBufferMemory::new().with_return_messages(true);
let chat_chain = RunnableWithMessageHistory::new(llm.clone(), memory)
    .pipe(StrOutputParser::new());

let r1 = chat_chain.invoke("My name is Alice, please remember me.".to_string(), None).await?;
let r2 = chat_chain.invoke("What is my name?".to_string(), None).await?; // remembers last turn
```

**RAG chain: local BM25 retrieval + LLM generation**

```rust
use langchainrust::{
    BM25Retriever, Document, RAGPipelineBuilder, RagRunnable, Runnable, RunnableExt,
};
use std::sync::Arc;

let retriever = BM25Retriever::new();
retriever.add_documents_sync(vec![
    Document::new("Rust is a systems programming language created by Mozilla, focused on safety and performance.").with_id("rust_intro"),
    Document::new("Rust's core features include the ownership system, borrow checking, and zero-cost abstractions.").with_id("rust_features"),
]);

let pipeline = RAGPipelineBuilder::new()
    .llm(llm)
    .retriever(retriever)
    .retrieve_k(2)
    .build()?;
let rag_chain = RagRunnable::new(Arc::new(pipeline));

let answer = rag_chain.invoke("What are Rust's core features?".to_string(), None).await?;
```

**Full five-segment composition (prompt + memory + LLM + parser + RAG in one chain)**

Five capabilities in one runnable program, sharing one LLM instance to form a complete conversational RAG assistant — this is the full content of `crates/lc/examples/lcel/lcel_compose.rs`:

```rust
use langchainrust::{
    BM25Retriever, ChatPromptTemplate, ConversationBufferMemory, Document, Message, OpenAIChat,
    OpenAIConfig, RAGPipelineBuilder, RagRunnable, Runnable, RunnableExt, RunnableWithMessageHistory,
    StrOutputParser,
};
use std::collections::HashMap;
use std::sync::Arc;

let api_key = std::env::var("OPENAI_API_KEY").expect("please set the OPENAI_API_KEY env var");
let llm = OpenAIChat::new(OpenAIConfig {
    api_key,
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-4o-mini".to_string(),
    ..Default::default()
});

// 1. prompt + LLM + parser —— Runnable<HashMap<String, String>, String>
let prompt = ChatPromptTemplate::from_messages([
    Message::system("You are a concise Rust assistant. Output conclusions only, no extra text."),
    Message::human("{question}"),
]);
let qa_chain = prompt.pipe(llm.clone()).pipe(StrOutputParser::new());
let answer = qa_chain.invoke(HashMap::from([(
    "question".to_string(),
    "Explain what Rust is in one sentence".to_string(),
)]), None).await?;

// 2. memory + LLM + parser —— Runnable<String, String>, history read/written automatically
let memory = ConversationBufferMemory::new().with_return_messages(true);
let chat_chain = RunnableWithMessageHistory::new(llm.clone(), memory)
    .pipe(StrOutputParser::new());
let r1 = chat_chain.invoke("My name is Alice, please remember me.".to_string(), None).await?;
let r2 = chat_chain.invoke("What is my name?".to_string(), None).await?; // remembers last turn

// 3. RAG chain: local BM25 retrieval + LLM generation —— Runnable<String, String>
let retriever = BM25Retriever::new();
retriever.add_documents_sync(vec![
    Document::new("Rust is a systems programming language created by Mozilla, focused on safety and performance.").with_id("rust_intro"),
    Document::new("Rust's core features include the ownership system, borrow checking, and zero-cost abstractions.").with_id("rust_features"),
]);
let pipeline = RAGPipelineBuilder::new()
    .llm(llm)
    .retriever(retriever)
    .retrieve_k(2)
    .build()?;
let rag_chain = RagRunnable::new(Arc::new(pipeline));
let answer = rag_chain.invoke("What are Rust's core features?".to_string(), None).await?;
```

Run it with `cargo run --example lcel_compose` (env `OPENAI_API_KEY` required; `OPENAI_BASE_URL` / `TEST_CHAT_MODEL` optional).

**Unified error type `LcelError`**

Errors along the whole chain converge into `LcelError`: parser errors implement `From<OutputParserError>`, native OpenAI errors implement `From<OpenAIError>`, so `prompt.pipe(llm).pipe(parser)` returns `Result<T, LcelError>` — one `?` handles the whole chain, no per-segment `match` needed.

> **Boundary note**: other providers (non OpenAI/Qwen/DeepSeek) stay wrapped behind `LLMClient`; no Rust `|` operator overloading (use `.pipe()`); the `Runnable<String, Vec<Document>>` adapter for Retriever is left for v0.16.

---

## Chains

Chains combine LLMs with prompts, memory, retrieval, and other components into reusable pipelines. Each Chain receives input, executes a series of steps, and returns output.

### LLMChain

The most basic chain — a prompt template + an LLM. Input variables are substituted into the template, sent to the LLM, and the result is returned. It's the building block for more complex chains.

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

Chains multiple Chains in sequence — the output of one Chain becomes the input of the next. Use for multi-step tasks like "analyze first, then summarize".

```rust
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

Retrieval-Augmented Question Answering — first retrieves relevant documents from a vector store, then sends the documents and question together to the LLM. The simplest form of RAG.

```rust
use langchainrust::{RetrievalQA, SimilarityRetriever};

let retriever = SimilarityRetriever::new(store, embeddings);
let qa = RetrievalQA::new(llm, retriever, 3);

let answer = qa.invoke(HashMap::from([
    ("query", "What is BM25?"),
])).await?;
```

**Return sources**: `.with_return_source_documents(true)` returns the retrieved raw documents alongside the answer, handy for showing evidence / auditing:

```rust
let qa = RetrievalQA::new(llm, retriever, 3).with_return_source_documents(true);
let result = qa.invoke(HashMap::from([("query", "What is BM25?")])).await?;
// result.source_documents carries the hit Document list
```

### RouterChain (Routing Chain) ✨ v0.14.0

Dispatches different inputs to different sub-chains by rules. `RouterChain` uses keyword matching; `LLMRouterChain` lets the LLM decide.

```rust
use langchainrust::chains::RouterChain;
use std::sync::Arc;

let router = RouterChain::new()
    .add_route_with_keywords("math", "math operations", Arc::new(math_chain), vec!["add", "minus", "multiply"])
    .add_route("general", "general Q&A", Arc::new(general_chain))
    .with_default(Arc::new(fallback_chain));

let answer = router.invoke(HashMap::from([("input", "what is 3 plus 5")])).await?;
```

```rust
use langchainrust::chains::LLMRouterChain;

// LLM variant: model decides the routing target from descriptions
let router = LLMRouterChain::new(llm)
    .add_route("translation", "translation requests", Arc::new(trans_chain))
    .add_route("code", "programming questions", Arc::new(code_chain))
    .with_default(Arc::new(general_chain));
let answer = router.invoke(HashMap::from([("input", "write a bubble sort in Rust")])).await?;
```

> `add_route_with_keywords` attaches per-route keywords for fast matching; unmatched input falls back to `with_default`.

### ConversationRetrievalChain

Retrieval-augmented conversation with memory: each question automatically retrieves relevant documents and loads conversation history, so the LLM can reference both the knowledge base and previous turns.

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

A conversational chain with pluggable memory — `from_memory` accepts any `BaseMemory` implementation (window / summary / vector-store / persistent), or assemble one with `ConversationChainBuilder` and customize the system prompt and keys.

```rust
use langchainrust::{
    ConversationChain, ConversationChainBuilder, ConversationBufferWindowMemory,
    OpenAIChat, OpenAIConfig,
};
use std::sync::Arc;
use tokio::sync::Mutex;

let llm = OpenAIChat::new(OpenAIConfig::default());

// Option 1: from_memory with any BaseMemory
let memory = Arc::new(Mutex::new(ConversationBufferWindowMemory::new(4)));
let chain = ConversationChain::from_memory(llm.clone(), memory);
let answer = chain.predict("Hello!").await?;

// Option 2: Builder (also pluggable + custom system prompt / keys)
let chain = ConversationChainBuilder::new(llm)
    .memory(ConversationBufferWindowMemory::new(6))
    .system_prompt("You are a helpful assistant.")
    .build();
let answer = chain.predict("What is Rust?").await?;
```

---

## Document Chains

When you have too many documents to fit in a single prompt, Document Chains provide different strategies for processing multiple documents:

| Chain | Strategy | Use Case |
|-------|----------|----------|
| **StuffDocumentsChain** | Stuff all docs into one prompt | Few docs, total length within token limit |
| **RefineDocumentsChain** | Iterate over docs, refining the answer | Need incremental refinement, docs have dependencies |
| **MapReduceDocumentsChain** | Process each doc independently, then combine | Many docs, can be processed in parallel |
| **MapRerankDocumentsChain** | Score each doc independently, pick the best | Need to select the most relevant doc |

### StuffDocumentsChain

Combine all documents with a prompt and send to the LLM in one call. Simplest approach, but total document length must fit within the LLM's token limit.

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

Iteratively refine: generate an initial answer from the first document, then progressively refine it with subsequent documents. Good for synthesizing information across documents, but cannot run in parallel.

```rust
use langchainrust::chains::RefineDocumentsChain;

let initial_llm = Arc::new(LLMChain::new(llm.clone(), "Summarize: {text}"));
let refine_llm = Arc::new(LLMChain::new(llm, "Refine summary with: {text}"));

let chain = RefineDocumentsChain::new(initial_llm, refine_llm);
let result = chain.invoke(documents).await?;
```

### MapReduceDocumentsChain

Map phase processes each document independently (can run in parallel), Reduce phase combines all results. Best for large document sets where each doc can be processed separately.

```rust
use langchainrust::chains::MapReduceDocumentsChain;

let map_chain = Arc::new(LLMChain::new(llm.clone(), "Summarize: {text}"));
let reduce_chain = Arc::new(LLMChain::new(llm, "Combine: {summaries}"));

let chain = MapReduceDocumentsChain::new(map_chain, reduce_chain);
let result = chain.invoke(documents).await?;
```

### MapRerankDocumentsChain

Score each document independently, then pick the best by score. Use when you need to select the most relevant document from multiple candidates.

```rust
use langchainrust::chains::MapRerankDocumentsChain;

let map_chain = Arc::new(LLMChain::new(llm, "{text}\nScore (0-10):"));

let chain = MapRerankDocumentsChain::new(map_chain);
let (best_doc, score) = chain.invoke(documents).await?;
```

---

### Chain Streaming ✨ v0.4.1

`BaseChain::stream()` provides token-by-token streaming output. `LLMChain` and `ConversationChain` have overridden implementations.

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

### invoke_with_config (Callback Passthrough) ✨ v0.15.0

`BaseChain::invoke_with_config(inputs, config)` injects a `RunnableConfig` (callbacks / metadata) at call time. Composite chains (`SequentialChain` / `RouterChain`) **forward the config to sub-chains** instead of silently dropping it — callbacks stay consistent across the whole pipeline.

```rust
use langchainrust::{CallbackManager, StdOutHandler, ChainResult};

let config = RunnableConfig::new()
    .with_callbacks(Arc::new(CallbackManager::new().add_handler(Arc::new(StdOutHandler::new()))));
let result: ChainResult = chain.invoke_with_config(inputs, config).await?;
```

---

## Agents

Agents are LLM applications that can autonomously call tools and perform multi-step reasoning. Unlike Chains with fixed flows, Agents dynamically decide which tools to call and how many steps to execute based on the input.

**When to use an Agent, and when to use a Chain / RAGPipeline?**

| Need | Use |
|------|-----|
| Fixed flow: prompt → model → parse | Chain / LCEL |
| Answer questions from private documents | RAGPipeline |
| Decide which tool to call, multi-step reasoning | Agent |
| Uncertain retrieval quality, needs self-correction / deep research | `CorrectiveRAGAgent` / `DeepResearchAgent` |

**Which of the three base Agents should you pick?**

| Agent | Mechanism | Models | Scenario |
|-------|-----------|--------|----------|
| `FunctionCallingAgent` (recommended) | Native tool_calls | GPT-4 / Claude / Gemini etc. | Most scenarios |
| `ReActAgent` | Text "think/act" regex parsing | Models without function calling | Compatibility with older models |
| `PlanExecuteAgent` | Plan first, then execute step-by-step, re-plan on failure | Any | Breaking down complex tasks |

### FunctionCallingAgent (Recommended)

Uses the LLM's native Function Calling capability to invoke tools. Type-safe and highly reliable — the preferred choice for models that support FC (GPT-4, Claude, Gemini).

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

### ReActAgent (Legacy)

Uses the ReAct (Reasoning + Acting) pattern: the LLM outputs "thought→action→observation" text, and the framework parses and calls tools. Good compatibility, but relies on text parsing so less reliable than FunctionCallingAgent. Use with models that don't support FC.

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

| Agent | Tool Calling | Reliability | Use Case |
|-------|--------------|-------------|----------|
| FunctionCallingAgent | Native FC | High (type-safe) | GPT-4, Claude, Gemini |
| ReActAgent | Text parsing | Medium | Models without FC support |

### Agent Streaming ✨ v0.12.0

CRAG, AdaptiveRAG, and DeepResearch support `stream()` for step-by-step pipeline events, enabling real-time progress display.

**CRAG Streaming:**

```rust
use langchainrust::agents::crag::CorrectiveRAGAgent;

let agent = CorrectiveRAGAgent::new(llm, retriever);
let stream = agent.stream("What is Rust ownership?").await?;

// Step-by-step events:
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

**AdaptiveRAG Streaming:**

```rust
use langchainrust::agents::adaptive_rag::AdaptiveRAG;

let agent = AdaptiveRAG::new(llm, retriever);
let stream = agent.stream("Compare tokio vs async-std").await?;

// Event flow:
// PipelineStep { step: "routing", detail: "Deciding retrieval strategy..." }
// PipelineStep { step: "routed", detail: "Decision: MultiQuery" }
// PipelineStep { step: "retrieving", ... }
// PipelineStep { step: "generating", ... }
// FinalAnswer { content: "..." }
```

**DeepResearch Streaming:**

```rust
use langchainrust::agents::deep_research::DeepResearchAgent;

let agent = DeepResearchAgent::new(llm)
    .with_searcher(Box::new(DuckDuckGoSearchTool::new()));

let stream = agent.stream_research("Rust async runtimes comparison").await?;

// Event flow (multi-round search):
// PipelineStep { step: "planning", detail: "Decomposing topic into subtopics..." }
// PipelineStep { step: "searching", detail: "Round 1/3: Searching 3 subtopics..." }
// PipelineStep { step: "searched", detail: "Found 12 results" }
// PipelineStep { step: "synthesizing", detail: "Synthesizing findings..." }
// PipelineStep { step: "gaps_found", detail: "Found 2 knowledge gaps" }
// PipelineStep { step: "searching", detail: "Round 2/3: Searching gaps..." }
// PipelineStep { step: "completed", detail: "Research completed in 2 rounds" }
// FinalAnswer { content: "..." }
```

### AgentBuilder (Chained Construction) ✨ v0.14.0

`AgentBuilder` offers chained construction to assemble the LLM, tools and execution parameters in one go; `max_iterations` is clamped to `[1, 100]` to avoid infinite loops.

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

`build()` returns an `AgentExecutor`. Robustness fallbacks are built in: tool-execution timeout (`tool_timeout`), exponential-backoff LLM retry, and a `Semaphore`-limited concurrent Actions pool.

### Orchestrator ✨ v0.14.0

The `Orchestrator` trait organizes multiple agents into workflows:

- **FanOutFanIn** — fans out to multiple sub-agents running in parallel, then merges results with a custom aggregator (voting / concatenation)
- **SequentialPipeline** — runs serially, feeding each step's output into the next

```rust
use langchainrust::agents::{FanOutFanIn, SequentialPipeline};

// Serial: two agents run in sequence
let pipeline = SequentialPipeline::new()
    .add(researcher_agent)
    .add(writer_agent);
let result = pipeline.run("Rust async runtimes".to_string()).await?;
```

`OrchestratorRunnable` wraps them as LCEL `Runnable`s so they can enter a `pipe()` pipeline (see the LCEL Adapters section).

### Agent Hooks (Five Safety Controls) ✨ v0.11.0

Hooks inject safety controls into the agent execution lifecycle:

```rust
use langchainrust::agents::hooks::{AgentHook, PromptInjectionHook, TokenBudgetHook, ContentFilterHook};

let hook = AgentHook::new()
    .on_before_tool_call(approval_callback)   // allow / deny / skip
    .with_hook(Arc::new(PromptInjectionHook::new())) // injection detection
    .with_hook(Arc::new(TokenBudgetHook::new(100_000))) // budget cap
    .with_hook(Arc::new(ContentFilterHook::new()));    // content filtering
```

### ToolPolicy (Tool Risk Tiers) ✨ v0.14.0

`ToolPolicy` + `ToolRisk` tier tools: high-risk tools require a stricter approval path to prevent privilege escalation.

```rust
use langchainrust::agents::policy::{ToolPolicy, ToolRisk};

let mut policy = ToolPolicy::new();
policy.set_risk("delete_file", ToolRisk::High);
// High-risk tool calls go through approval instead of executing directly
```

## Plan-Execute Agent

A plain single-loop Agent (`FunctionCallingAgent` / `ReActAgent`) suits tasks that can be thought through in one step — think, act, observe the result. But for complex multi-step tasks like "research first, then write code, then explain key points", the model cannot produce a complete plan in a single step, and diving straight in tends to go off track. The Plan-Execute Agent breaks a big task into a "plan first → execute step by step → re-plan on failure" loop: it first uses an LLM to decompose the task into executable steps, hands each step to a single-loop Agent, re-plans when a step fails (instead of stubbornly continuing), and finally summarizes the result once all steps complete. Suited for complex, multi-step tasks that allow mid-course plan adjustments.

> Note: each step runs via `FunctionCallingAgent` + tools; `llm` must currently be `OpenAIChat`.

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, PlanExecuteAgent, BaseTool};
use std::sync::Arc;

let llm = OpenAIChat::new(OpenAIConfig::default());
let tools: Vec<Arc<dyn BaseTool>> = vec![]; // pass real tools

let agent = PlanExecuteAgent::new(llm, tools)
    .with_max_replans(2); // re-plan at most 2 times on failure

let result = agent
    .run("Research Rust async runtimes, write example code, explain key points")
    .await?;
println!("{}", result);
```

### Workflow

| Phase | What it does |
|---|---|
| Plan | Uses the LLM to break the task into several executable steps |
| Execute | Hands each step to a `FunctionCallingAgent` + tools and collects the step result |
| Re-plan | Re-plans when a step fails; the re-plan count is capped by `with_max_replans`, avoiding idle spinning |
| Answer | Summarizes the final response after all steps complete |

### Difference from FunctionCallingAgent / ReActAgent

| | FunctionCallingAgent / ReActAgent | PlanExecuteAgent |
|---|---|---|
| Role | Single-loop executor (implements `BaseAgent`, can go inside an `AgentExecutor`) | High-level orchestrator (does not implement `BaseAgent`; has its own `run()`) |
| Task shape | Single-step "think → act → observe" until the model says it is done | Decomposes a complex task into steps first, then drives executors step by step |
| Failure handling | Feeds tool failure results back to the LLM to think again | A step failure triggers re-planning |
| Best for | Single tasks with clear decisions | Complex multi-step tasks that need decomposition first |

**How to choose (in plain words)**: use `FunctionCallingAgent` / `ReActAgent` directly for simple tasks; reach for Plan-Execute when the task is too big to see what to do first at a glance and needs to be broken into steps. It is the "boss": it does not do the work itself — it delegates to single-loop Agents.

### Key behaviors and notes

- **Each step runs independently (cold start)**: every step is executed by a fresh `FunctionCallingAgent` + `Executor` and discarded when done; by default steps do not share context. For tasks that depend strongly on step results (step 2 needs step 1's output), write the previous step's result into the next step's description.
- **Re-planning is capped**: `with_max_replans` limits the number of re-plans, preventing infinite re-planning after failures.
- **Configurable executor**: since v0.14, the Agent used to execute steps can be configured via `agent_factory` instead of being hard-coded to `FunctionCallingAgent`.
- **PlanExecute vs DeepResearch**: PlanExecute suits tasks whose steps are fairly independent (search → plan an itinerary → write a report); for research-style tasks (research → research more → synthesize) that need context chained across steps and intermediate conclusions preserved, `DeepResearch` is a better fit.

---

## Handoffs

Making one Agent do everything — both research and writing — strains the model and hurts reusability. Handoffs let the primary Agent transfer control to another specialist Agent when it realizes mid-task that "this part should be done by that expert". Inspired by the OpenAI Agents SDK: the primary Agent delegates tasks to registered specialist Agents via `HandoffTool`. Suited for a "team of specialist Agents with clear divisions of labor" — the primary Agent only decides who does the work, the concrete job goes to the matching specialist, and the specialist continues from the handoff point.

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

// Run the primary agent
let result = mgr.run("Research and write an article".to_string()).await?;

// Generate a HandoffTool for each registered agent (named handoff_to_{agent})
let mgr = Arc::new(mgr);
let handoff_tools = mgr.handoff_tools();
let history = mgr.history(); // handoff history
```

### How to use

1. **Register specialist Agents**: `register_agent("writer", writer)` names and registers each specialist.
2. **Set the primary Agent**: `set_primary("researcher")` designates the entry Agent; the task starts from it.
3. **Run the primary Agent**: `mgr.run(...)` executes the task.
4. **Generate handoff tools**: `handoff_tools()` produces one tool named `handoff_to_{agent}` per registered Agent; once these tools are bound to an Agent, the model can proactively choose "who to hand off to". You can also skip binding tools and hand off directly in code with `execute_handoff(Handoff)`.

### Key behaviors

| Behavior | Description |
|---|---|
| `handoff_tools()` | Returns a set of `handoff_to_{agent}` tools, named one-to-one with the registered names |
| `execute_handoff(Handoff)` | Hands off directly in code without going through tools |
| `history()` | The handoff history, traceable back to who handed off to whom |
| `max_handoff_depth` | Maximum handoff depth (default 10), preventing Agent A↔B from looping infinitely between each other; when the limit is reached, the handoff terminates and returns a clear error |

### When to use / when not to

- **Use it**: when task boundaries are clear and each subdomain has a dedicated Agent (e.g. writer / researcher / coder), with the primary Agent only doing dispatch.
- **Do not use it**: when you need to "assign work and collect results" (distributing a task to several Agents in parallel and aggregating the results) — Handoffs is a **single transfer of control**, not fan-out aggregation; in that scenario `FanOutFanIn` is more appropriate.

---

## Streaming Tool Calls

Regular Agents wait until the entire execution completes before returning, so the user can only wait, unable to tell whether it is stuck or working. Streaming tool calls let `StreamingFunctionCallingAgent` stream LLM text token by token and expose tool-call state through the event stream — users can watch the Agent's full journey of "thinking, calling a tool, which stage a tool call has reached, and finally giving the answer" in real time. Suited for chat-style experiences (typewriter effect), long-task progress display, and debugging Agent behavior.

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

### Event stream

`invoke_stream` returns an async Stream that yields `AgentStreamEvent`s one at a time:

| Event | Meaning |
|---|---|
| `Text { content }` | LLM text, token by token; printing it directly gives the typewriter effect |
| `ToolCall { state }` | A tool-call state change (see below) |
| `FinalAnswer { content }` | The final answer, usually near the end of the stream |

`ToolCallState` covers the tool-call lifecycle:

| State | Meaning |
|---|---|
| `Started` | The tool call starts |
| `ArgumentsStreaming` | Tool arguments are being generated piece by piece |
| `Completed` | The tool call completes |
| `Failed` | The tool call fails |

### When to use streaming

| Scenario | Why streaming |
|---|---|
| Chat-style UI | Text appears word by word, so waiting is less agonizing |
| Long / multi-step tasks | Shows "what is being done" in real time, so users know it is not stuck |
| Debugging Agents | Observe the reasoning text + tool-call timing directly to pinpoint issues quickly |

### Notes

- The event stream delivers **state changes** of the LLM text and tool calls; a `ToolCall` event describes "which stage the call has reached". The tool execution result does flow back to the LLM for its next decision, but the result body itself does not appear in the event stream.
- This Agent's streaming surface focuses on LLM output + tool-call state; if you only need fine-grained events at the tool-execution level, look at the `Executor::stream` capability instead — the two are different vantage points.

---

## Guardrails

Input/output validation to block malicious input and sensitive-information leakage. Implement `InputGuardrail` / `OutputGuardrail`, or use built-in validators, then wrap an agent with `GuardedAgent`.

```rust
use langchainrust::guardrails::{
    GuardrailsConfig, MaxLengthGuardrail, SensitiveInfoGuardrail, GuardedAgent,
};
use langchainrust::{BaseAgent, AgentExecutor, FunctionCallingAgent, OpenAIChat, OpenAIConfig};
use std::sync::Arc;

let config = GuardrailsConfig::new()
    .with_input(Arc::new(MaxLengthGuardrail::new(1000)))    // limit input length
    .with_output(Arc::new(SensitiveInfoGuardrail::new()));  // block sensitive output

let agent = FunctionCallingAgent::new(OpenAIChat::new(OpenAIConfig::default()), vec![], None);
let executor = Arc::new(AgentExecutor::new(
    Arc::new(agent) as Arc<dyn BaseAgent>,
    vec![],
));

let mut guarded = GuardedAgent::new(executor, config);
let result = guarded.invoke("Summarize this content".to_string()).await?; // validate input -> agent -> validate output
let violations = guarded.violations();
```

Built-in validators: `MaxLengthGuardrail` (input length), `ForbiddenWordsGuardrail` (banned words), `SensitiveInfoGuardrail` (API keys / emails / credit cards / keywords, extend with `with_keywords`). You can also drive validation manually with `GuardrailRunner`.

### Type-Separated Guardrail Results ✨ v0.15.0

Guardrail results are split into two types by input/output, so the type system enforces the safety rules:

- **`InputGuardrailResult`** — only `Pass` / `Block` (no `Modify` on the input side)
- **`OutputGuardrailResult`** — `Pass` / `Block` / `Modify`

"Modify only applies to output" is guaranteed by the compiler, not a runtime convention. `GuardrailError::Blocked` carries `reason` / `partial` / `suggestion`, so a failure can include partial content for graceful degradation.

### Guardable (Decoupled Wrap Target) ✨ v0.15.0

`GuardedAgent` no longer only accepts `AgentExecutor`. The `Guardable` trait (`invoke_str` / `stream_str`) lets **any executable unit** be wrapped in guardrails:

- `AgentExecutor` implements `Guardable` directly
- Any `BaseChain` adapts via `ChainGuardable`
- `GuardedAgent::from_chain` provides a chain-entry point

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

### Streaming Guardrails ✨ v0.15.0

The `StreamingOutputGuardrail` trait (`validate_chunk -> ChunkAction::{Pass, Replace, Block}`) works with `GuardedAgent::invoke_stream` for two-phase checking: incremental keyword checks + a 24-char sliding window (so cross-chunk cuts can't slip through) + a full-output recheck.

```rust
use futures_util::StreamExt;

let mut stream = guarded.invoke_stream("Write a long summary".to_string()).await?;
while let Some(chunk) = stream.next().await {
    let chunk = chunk?;   // GuardableChunk { token, is_final }
    print!("{}", chunk.token);
    if chunk.is_final {
        break;            // last chunk, end streaming
    }
}
```

### Audit Persistence ✨ v0.15.0

The `AuditSink` trait + `FileAuditSink` (JSON Lines, append-only) persist violation records to disk for post-hoc analysis:

```rust
use langchainrust::guardrails::audit::FileAuditSink;

let config = GuardrailsConfig::new()
    .with_output(Arc::new(SensitiveInfoGuardrail::new()))
    .with_audit_sink(Arc::new(FileAuditSink::new("guardrails.log")?));
```

`violations` is bounded (`MAX_VIOLATIONS = 1000`) and can be cleared with `clear_violations()`. `SensitiveInfoGuardrail` supports attaching an LLM judge (`with_judge`, reusing `SensitiveJudge` / `LlmSensitiveJudge`) for context-sensitive detection, and downgrades high-false-positive words (`password` / `token` / `secret`) to warning-only.

---

## Token Counter

LLMs bill by token, but token count ≠ character count — a block of Chinese, a block of code, and a block of English each have different token densities. Answering "how many tokens and how much money did this round cost", estimating the token count before a request (to truncate overly long text), or automatically accumulating usage per call all need dedicated tooling. The Token Counter components chain "count → track → price" together.

```rust
use langchainrust::{TokenTrackingLLM, ModelPricing, OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

let tracked = TokenTrackingLLM::for_openai(OpenAIChat::new(OpenAIConfig::default()))?;

let result = tracked.chat(vec![Message::human("hi")], None).await?;

let usage = tracked.get_usage();                               // prompt / completion / total tokens
let cost = tracked.estimate_cost(&ModelPricing::gpt4o_mini()); // USD
```

### What the four components do

| Component | What it does | When to use |
|---|---|---|
| `TiktokenCounter` | Precise counting with the same tokenizer OpenAI uses (`cl100k_base`) | When you need exact token counts (billing, aligning the context window) |
| `CharRatioCounter` | A rough estimate when tiktoken is unavailable — extrapolates by character-count ratio (common for Chinese text) | Quick estimates, fallback when no precise tokenizer is available |
| `TokenTrackingLLM` | Wraps any model and automatically records and accumulates token usage per call | When you want automatic cumulative usage without manual math |
| `ModelPricing` | Prices each model (per-1k-token price) and computes cost from accumulated usage | Estimating cost / charging by usage |

> Import paths: `TiktokenCounter` / `TokenTrackingLLM` / `ModelPricing` are at the crate root; `CharRatioCounter` must be imported from `langchainrust::core::token_counter::CharRatioCounter`.

### Precise counting vs estimation

| Scenario | Which to pick |
|---|---|
| Billing, limits, aligning the window | `TiktokenCounter` for precise counts |
| Quick estimates, truncating long Chinese text | `CharRatioCounter` for rough estimates |
| Automatically recording usage per call | `TokenTrackingLLM` (wrap the model) |
| Computing USD cost from usage | `ModelPricing` + `estimate_cost` |

### Key behaviors

- **Real usage first**: `TokenTrackingLLM` prefers the real usage stats returned by the model API; it falls back to estimation only when the model does not return them.
- **Non-invasive to the original model**: the tracked object is the wrapper itself; the wrapped model's behavior is unchanged — wrap it once and usage accumulates automatically.
- **Built-in pricing**: `ModelPricing::gpt4o()` / `gpt4o_mini()` are built in; use `ModelPricing::new(prompt_per_1k, completion_per_1k)` to price other models.
- **Computing cost**: `get_usage()` returns the accumulated prompt / completion / total tokens, and `estimate_cost(&pricing)` converts them to USD using the pricing.

### How to choose (in plain words)

- Just want to "record a count per call and report a total at the end"? Wrap the model with `TokenTrackingLLM::for_openai(...)` + `estimate_cost`, and the whole chain is done.
- To estimate a text's token count **before** sending a request (e.g. deciding whether to truncate), use `TiktokenCounter` (precise) or `CharRatioCounter` (estimate).
- When Chinese content dominates and precision is not a priority, `CharRatioCounter` is enough; for real billing, use precise tokenization.

---

## Sessions

Multi-turn conversation must remember context — what the user said last turn and how the assistant replied. But "where to store it, how to persist it, how to retrieve it" is boilerplate every app re-implements. `SessionManager` abstracts conversations into lifecycle management: create a session → write conversation into it → pull history anytime → archive/clean up. It also natively supports **multi-session isolation**: each session has its own id and owning user, so conversations from different users or different topics never interfere with each other.

**Core trio**: `Session` (the session itself) ← `SessionStore` (how it is stored) ← `SessionManager` (how it is used).

```rust
use langchainrust::sessions::{SessionManager, MemorySessionStore};
use langchainrust::{OpenAIChat, OpenAIConfig};
use std::sync::Arc;

let manager = SessionManager::new(Arc::new(MemorySessionStore::new()));
let id = manager.create_session_for("user_1").await?;

let llm = OpenAIChat::new(OpenAIConfig::default());
let r1 = manager.chat(&id, &llm, "My name is Tom".to_string()).await?;
let r2 = manager.chat(&id, &llm, "What is my name?".to_string()).await?; // remembers the previous turn

let history = manager.history(&id).await?;  // Vec<Message>
manager.clear(&id).await?;                   // clear history (keep session)
manager.archive(&id).await?;                 // archive
let sessions = manager.list_by_user("user_1").await?;
```

### Session model

| Field | What it is |
|---|---|
| `id` | The session's unique identifier |
| `user_id` | The owning user (nullable, supports anonymous) |
| `messages` | The conversation message list (`Vec<Message>`), growing by appending |
| `status` | Lifecycle state (`Active` / `Archived` / `Deleted`) |
| `metadata` | Free key-value extension attributes |

### SessionManager methods

| Method | What it does |
|---|---|
| `create_session_for(user)` | Creates a new session and returns the session id |
| `chat(&id, &llm, msg)` | The core: appends the user message → feeds history to the LLM → appends the reply back into the session (history maintained automatically) |
| `history(&id)` | Returns the full conversation history (`Vec<Message>`) |
| `clear(&id)` | Clears history (keeps the session) |
| `archive(&id)` | Archives (no longer active, but preserved) |
| `list_by_user(user)` | Lists all sessions belonging to a user |

`chat()` is the core — the caller only passes "session id + LLM + user message"; history reads and writes are handled entirely by `SessionManager`, so there is no need to maintain a `Vec<Message>` by hand.

### Session-level history management

- Each session's history is maintained independently; every `chat()` feeds the LLM based on that session's existing history, which is why the second utterance in the same session can "remember the previous turn".
- By default an internal buffer maintains the full history directly — simple and straightforward; long sessions grow linearly in tokens with each turn, so attach a memory component when you need to control cost (see below).
- The `SessionStore` trait covers `create/get/update/delete/list_by_user`; `MemorySessionStore` is the built-in implementation (in-process memory + tokio lock), suitable for tests and single-process use; you can implement your own backend (Redis / database).

### Session Lifecycle ✨ v0.15.0

`SessionStatus` forms a closed state machine: `Active` → `Archived` → `Deleted`. Deletion is **soft delete**: the record stays (auditable / recoverable) but no longer appears in the user's session list.

### Attaching the Memory System ✨ v0.15.0

`SessionManager` keeps history in an internal buffer by default; `with_memory` can attach any `BaseMemory` (e.g. `ConversationSummaryBufferMemory` / `MongoPersistentMemory`), routing session history through summary compression or cross-process persistence:

```rust
let mut manager = SessionManager::new(Arc::new(MemorySessionStore::new()));
manager = manager.with_memory(Arc::new(Mutex::new(
    ConversationSummaryBufferMemory::new(llm, 2000),
)));
let r = manager.chat(&id, &llm, "question".to_string()).await?;
```

**How to choose memory (in plain words)**: skip it to keep the full history — simplest semantics, but long sessions bloat; attach `ConversationSummaryBufferMemory` to compress long sessions into "summary + recent window" and control token cost; attach persistent memory for history that survives across processes and is shared between instances. Once attached, the conversation history is handled by the memory component rather than passed through in full.

### Notes

- **Concurrent writes to the same session**: `chat()` internally does "read → append → write back" in three steps; concurrent writes to the same session must be serialized by you (e.g. a per-session lock), or messages may be lost.
- **Session vs long-term memory**: lc-sessions manages "the process record of one conversation"; cross-session long-term memory (persona, preferences) goes to lc-memory; a session is conversation context organized by time.
- **Session vs checkpoints**: a session stores "what was said"; a Checkpointer stores "how far graph execution got" (lc-langgraph). Both involve persistence, but their semantics differ.
- **Storage choice**: `MemorySessionStore` is enough for tests and single-process scenarios; for multiple instances / cross-process shared history, switch to a database or Redis backend implementing `SessionStore`.

---

## MCP

[MCP](https://modelcontextprotocol.io) (Model Context Protocol) is the tool protocol standard introduced by Anthropic. `MCPClient` connects to any MCP Server to obtain tools and adapts them as `BaseTool` for agents.

```rust
use langchainrust::mcp::{MCPClient, MCPConfig};
use langchainrust::{BaseAgent, AgentExecutor, FunctionCallingAgent, OpenAIChat, OpenAIConfig};
use std::sync::Arc;

// Stdio: spawn an MCP Server subprocess
let config = MCPConfig::stdio(
    "npx",
    vec!["@anthropic/mcp-server-filesystem".to_string(), "/tmp".to_string()],
);
// Or SSE: MCPConfig::sse("http://localhost:3001/sse");

let mut client = MCPClient::connect(config).await?;
let tools = client.list_tools().await?;           // tools/list
println!("MCP tool count: {}", tools.len());

// Adapt to a BaseTool list and hand it to an agent
// P0-3: as_tools auto-discovers tools, no need to call list_tools first
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

`MCPConfig::stdio(command, args)` / `MCPConfig::sse(url)` / `.with_env(k, v)`; `client.call_tool(name, arguments)` calls a tool directly; `as_tools()` wraps tools as `MCPToolAdapter` (implements `BaseTool`).

---

### MCPServer

Symmetric to `MCPClient`: expose local `BaseTool`s as an MCP Server for Claude Desktop / Cursor hosts. Supports `initialize` / `tools/list` / `tools/call`.

```rust
use langchainrust::{MCPServer, Calculator, BaseTool};
use std::sync::Arc;

let tool: Arc<dyn BaseTool> = Arc::new(Calculator::new());
let server = MCPServer::new()
    .with_tool(tool)
    .with_server_info("my-tools", "0.1.0");

server.serve_stdio().await?;
```

`server.handle_request(req)` for single-step JSON-RPC handling with custom transport.

### Transport Resilience ✨ v0.15.0

MCP connections auto-reconnect: after a disconnect they retry with exponential backoff (starting at 0.5s, capped at 30s) and recover automatically once the Server restarts. `MCPServer` has the same hot-restart capability — a reconnecting host resumes the session.

### ConnectionManager (Connection Pool) ✨ v0.15.0

Manages the lifecycle of multiple `MCPClient`s — auto-reconnect and unified shutdown:

```rust
use langchainrust::mcp::{ConnectionManager, ServerSpec};

let manager = ConnectionManager::new();
manager.register(ServerSpec::new("files", MCPConfig::sse("http://localhost:3001/sse"))).await?;
manager.register(ServerSpec::new("tools", MCPConfig::stdio("npx", vec!["...".into()]))).await?;

let client = manager.client("files").await?;  // grab one server's connection
manager.reap_idle().await;                     // reap idle connections
// manager.shutdown().await;                     // shut everything down
```

### Tool Namespacing / Discovery / Timeout ✨ v0.15.0

- **`ToolNamespace`**: tool names are prefixed `server:tool` automatically, so same-named tools across servers don't collide; `register(server, tools, conflict)` returns namespaced results you can build `MCPToolAdapter::namespaced(...)` from
- **`ToolDiscovery`**: batch discovery + health check, filtering out tools from unavailable servers
- **`ToolSpec`**: `timeout` (per-call timeout), `max_retries` and other execution policy; a timeout hits the circuit breaker

### ServerHealth / CircuitBreaker ✨ v0.15.0

Each server has a health status (`HealthStatus`) and circuit breaker:

```rust
use langchainrust::mcp::{CircuitBreaker, HealthStatus};

let breaker = CircuitBreaker::new(5); // 5 consecutive failures -> open
if !breaker.allow_request() {
    // breaker open: short-circuit immediately, no backend call
} else {
    match call_tool().await {
        Ok(v) => breaker.record_success(), // success, counter auto-resets
        Err(_) => breaker.record_failure(),
    }
}
```

`ServerHealth` records latency, error rate, and last probe time for upstream routing decisions.

### SamplingGuard (Sampling Protection) ✨ v0.15.0

Recursion protection for server-side sampling requests (`resources/sampling/createMessage`): limits nested depth, the whole chain's token budget, and total duration, so a model can't recursively sample itself into exhausting resources:

```rust
use langchainrust::mcp::SamplingGuard;

let guard = SamplingGuard::new(5, 100_000) // max nested depth 5, whole-chain token budget 100k
    .with_timeout(std::time::Duration::from_secs(60)); // whole-chain total duration cap
let lease = guard.enter(4000)?; // enter one sampling; returns a SamplingLease that frees depth on drop
```

### MCPGateway (Gateway) ✨ v0.15.0

Aggregates multiple MCP servers behind one entry point, routing by the `server` parameter:

```rust
use langchainrust::mcp::{MCPGateway, GatewayServerSpec, MCPConfig};

let gateway = MCPGateway::new();
gateway.register(GatewayServerSpec::new("files", MCPConfig::stdio("npx", vec!["filesystem".into(), "/tmp".into()]))).await?;
gateway.register(GatewayServerSpec::new("db", MCPConfig::sse("http://localhost:9000/sse"))).await?;
gateway.sync_all().await?; // pull tools from all servers

let tools = gateway.as_base_tools().await?; // auto-prefixed by server, no collisions
```

Companion capabilities:
- **`ServerSandbox`**: `ParamRule` (parameter allow/deny lists, type checks), `EgressPolicy` (outbound policy restricting a tool call's network / file scope)
- **`PartialContent`**: streaming tool results returned in chunks; `stream_tool_call` pushes while executing
- **`TenantGateway`**: multi-tenant isolation — per-tenant tool namespaces + quotas + access control
- **`ToolOrchestrator`**: tool DAG orchestration — auto-orders / parallelizes once dependencies are declared
- **`VersionPolicy`**: multi-version MCP protocol negotiation (`VersionPolicy::Latest` / `Pin("2024-11-05")`)

> **Note**: `Resources` / `Prompts` / `Completion` / `Elicitation` types are defined, but the corresponding primitive call logic is not yet implemented (calls return `method_not_found`). The currently available primitives are `initialize` / `tools/list` / `tools/call`, plus streaming tool results (`PartialContent`, via `notifications/tool_partial` notification + `subscribe_tool_stream` subscription) and `notifications/cancelled` cancellation.

## Tools

Tools are the "hands" of Agents — they let LLMs perform calculations, search, read/write files, and more. Each tool implements the `BaseTool` trait, defining its name, description, parameter schema, and execution logic.

### Built-in Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| Calculator | Math operations | `expression` |
| DateTimeTool | Date/time queries | `operation`, `datetime` |
| SimpleMathTool | Power, sqrt, trig | `operation`, `value` |
| URLFetchTool | Fetch URLs | `url` |
| WikipediaTool | Wikipedia search | `query` |
| DuckDuckGoSearchTool | Web search | `query` |
| PythonREPLTool | Execute Python | `code` |

### Custom Tool

When built-in tools aren't enough, implement the `BaseTool` trait to create your own. You need to define an input struct (`JsonSchema` + `Deserialize`) and a `run` method.

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

### `#[tool]` Procedural Macro ✨ v0.10.0

Auto-generate `BaseTool` + `Tool` implementations with the `#[tool]` macro — no boilerplate needed:

```rust
use langchainrust::{BaseTool, Tool, ToolError, tools::tool};

// One macro = ~20 lines of hand-written code above
#[tool(description = "Echo the input text back")]
fn echo(
    #[param(desc = "The text to echo back")]
    text: String,
) -> Result<String, ToolError> {
    Ok(text)
}

// Auto-generates:
// - EchoTool struct (BaseTool + Tool impl)
// - EchoInput struct (Deserialize + JsonSchema)
// - args_schema() from JsonSchema automatically

// Usage is identical to hand-written Tools
let tool = EchoTool::new();
let schema = BaseTool::args_schema(&tool);  // JSON Schema
let result = tool.run(r#"{"text":"hello"}"#.to_string()).await?;
// result = "\"hello\""

// Supports Option<T> for optional parameters
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

### ToolRegistry (Tool Registry) ✨ v0.15.0

A registry managing a set of tools by name: register, look up, remove, and describe in bulk — can be fed directly to an LLM to show available tools.

```rust
use langchainrust::ToolRegistry;
use std::sync::Arc;

let mut registry = ToolRegistry::new();
registry.register(Arc::new(Calculator::new()));
registry.register(Arc::new(DateTimeTool::new()));

registry.get("calculator");                 // Option<&Arc<dyn BaseTool>>
registry.contains("datetime_tool");
registry.tool_names();                       // Vec<&str>
let description = registry.describe_tools(); // LLM-readable tool manifest
registry.remove("calculator");
```

### StructuredTool (Structured Wrapper) ✨ v0.15.0

Wraps a generic tool implementing the `Tool` trait into a `BaseTool`, auto-handling JSON input parsing and output serialization:

```rust
use langchainrust::{Tool, core::tools::StructuredTool};

let tool = StructuredTool::new(my_tool, Some("my_tool"), Some("description"));
let result = tool.run(r#"{"k": "v"}"#.to_string()).await?; // parses/serializes internally
```

### SSRF Protection (Network Tools) ✨ v0.15.0

`URLFetchTool` / `HTTPTool` **enable SSRF protection by default**: every hop — the initial request and each redirect — is checked for private/loopback targets; a hit is rejected with a hint to `.with_allow_private_ips(true)` to explicitly allow.

```rust
let tool = URLFetchTool::new();                 // private networks blocked by default
let tool = URLFetchTool::new().with_allow_private_ips(true); // explicitly allow
```

Implementation details: `is_private_ip` is the crate-wide single implementation (no duplicated logic), covering 127.0.0.0/8, 10/8, 172.16/12, 192.168/16, 169.254.169.254, IPv6 private ranges and IPv4-mapped IPv6 (`::ffff:127.0.0.1`); automatic redirects are disabled in favor of `guarded_get`, which re-checks each hop — closing the "public first hop, redirect into private network" bypass.

### WikipediaTool

Search Wikipedia article summaries. Use when an Agent needs to look up encyclopedic knowledge.

```rust
use langchainrust::WikipediaTool;

let tool = WikipediaTool::new();
let result = tool.run(r#"{"query": "Rust programming"}"#).await?;
```

### DuckDuckGoSearchTool

Search the web using DuckDuckGo. No API key required. Use when an Agent needs real-time web information.

```rust
use langchainrust::DuckDuckGoSearchTool;

let tool = DuckDuckGoSearchTool::new();
let result = tool.run(r#"{"query": "langchain rust"}"#).await?;
```

### PythonREPLTool

Execute Python code in a subprocess and return the output. Use for dynamic computation, data processing, or scientific calculations. Note: code runs locally, ensure your environment is secure.

```rust
use langchainrust::PythonREPLTool;

let tool = PythonREPLTool::new();
let result = tool.run(r#"{"code": "print(sum(range(10)))"}"#).await?;
```

> **Security boundary**: the built-in "dangerous import blacklist" (`os` / `sys` / `subprocess` / `__import__` / `eval` / `exec`, etc.) is **noise filtering, not a security boundary** — encoding-based bypasses like `__import__`, `"o"+"s"` concatenation, `().__class__` reflection, and unicode obfuscation slip through, and it also false-positives on string literals. Real isolation must go through the [code interpreter sandbox](#v050-new-features) (`LocalSandbox` subprocess + timeout); the blacklist only reduces the noise that reaches the sandbox. Do not rely on `PythonREPLTool` for isolation over untrusted input.

### Extended Tools (HTTPTool / FileTool / SQLTool)

Three production-oriented tools added in v0.3.0, all implementing `BaseTool`.

**HTTPTool** -- issue GET/POST requests:

```rust
use langchainrust::HTTPTool;
use serde_json::json;

let http = HTTPTool::new();
let body = http.post("https://httpbin.org/post", json!({"k": "v"})).await?;
// As BaseTool: input JSON {"url":"...","method":"get|post","body":{...}}
```

**FileTool** -- sandboxed file read/write (confined to `base_path`, extension allowlist, size cap, path-traversal protection):

```rust
use langchainrust::FileTool;
use std::path::PathBuf;

let file = FileTool::new(PathBuf::from("./workspace"))
    .with_allowed_extensions(vec!["txt".into(), "md".into(), "json".into()])
    .with_max_size(10 * 1024 * 1024);
let content = file.read("notes.txt").await?;
file.write("out.txt", "hello").await?;
// As BaseTool: input JSON {"op":"read|write|list","path":"...","content":"..."}
```

**SQLTool** -- read-only SQL queries (SELECT only, table allowlist; requires `sqlite-storage` feature):

```rust
use langchainrust::tools::extended::SQLTool;

let sql = SQLTool::new("data.db")?
    .with_allowed_tables(vec!["users".into()]);
let rows = sql.execute("SELECT id, name FROM users")?; // Vec<HashMap<String,String>>
// Non-SELECT (e.g. DROP/INSERT) is rejected

// Parameterized queries (recommended, prevents SQL injection)
let rows = sql.execute_parameterized("SELECT * FROM users WHERE name = ?", &["Alice".into()])?;
```

As a tool call it prefers the parameterized `{"sql": "...", "params": [...]}` form.

> `SQLTool` is available under the `sqlite-storage` feature; `HTTPTool` / `FileTool` are available by default.

---

## Embeddings

**Embeddings** convert text into fixed-dimension floating-point vectors, where semantically similar texts are closer in vector space. The foundation for semantic retrieval, similarity calculation, and RAG.

### Supported Embeddings

| Provider | Class | Dimension | Features |
|----------|-------|-----------|----------|
| **OpenAI** | `OpenAIEmbeddings` | 1536 | High quality |
| **DeepSeek** | `DeepSeekEmbeddings` | 1536 | Cost-effective |
| **Qwen** | `QwenEmbeddings` | 1536 | Chinese optimized |
| **Cohere** | `CohereEmbeddings` | Custom | RAG scenarios, multilingual |
| **FastEmbed** | `FastEmbedEmbeddings` | 384 | Local ONNX acceleration |
| **BagOfWords** | `BagOfWordsEmbeddings` | Custom | Pure local bag-of-words |
| **Mock** | `MockEmbeddings` | Custom | Testing |
| **Local** | `LocalEmbeddings` | Default | Pure Rust, offline |

### OpenAI Embeddings

Uses OpenAI's text-embedding-ada-002 model, 1536 dimensions. Highest quality but requires API calls.

```rust
use langchainrust::{OpenAIEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(OpenAIEmbeddings::new(
    std::env::var("OPENAI_API_KEY")?
));

// Single text embedding
let vector = embeddings.embed("Rust is a systems language").await?;
println!("Dimension: {}", vector.len());  // 1536

// Batch embedding
let texts = vec![
    "Rust is a systems language",
    "Python is a scripting language",
];
let vectors = embeddings.embed_batch(texts).await?;
```

### DeepSeek Embeddings

DeepSeek's embedding model, 1536 dimensions. Lower cost than OpenAI.

```rust
use langchainrust::{DeepSeekEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(DeepSeekEmbeddings::from_env());

let vector = embeddings.embed("Deep learning fundamentals").await?;
```

### Qwen Embeddings

Alibaba Cloud Qwen's embedding model, 1536 dimensions. Better performance for Chinese text.

```rust
use langchainrust::{QwenEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(QwenEmbeddings::from_env());

let vector = embeddings.embed("Qwen vector generation").await?;
```

### Mock Embeddings (Testing)

Generates fixed-dimension random vectors without any API calls. For testing and development only, not for production.

```rust
use langchainrust::{MockEmbeddings, Embeddings};
use std::sync::Arc;

// Custom dimension
let embeddings = Arc::new(MockEmbeddings::new(128));

let vector = embeddings.embed("Test text").await?;
println!("Dimension: {}", vector.len());  // 128
```

---

### LocalEmbeddings

Lightweight local embeddings in pure Rust (word-frequency hash + L2 normalization), no API calls. For offline / privacy / zero-cost coarse retrieval.

```rust
use langchainrust::LocalEmbeddings;

let emb = LocalEmbeddings::default_dim();
let vec = emb.embed_query("hello world").await?;
```

**Limitation**: bag-of-words hash, limited semantic quality. Use `OpenAIEmbeddings` etc. for high-quality embeddings.

### Unified `Embeddings` Trait ✨ v0.15.0

All embedding providers implement the unified `Embeddings` trait (`embed` / `embed_batch` / `embed_query`), so they are drop-in replaceable and can be composed into an `EmbeddingMatcher` for similarity retrieval. Unified error semantics:

- `EmptyInput` — empty text
- `EmptyVectorInBatch` — one item in the batch returned an empty vector
- `BatchMismatch` — input count does not match returned vector count

Errors are not silently swallowed: any failed embedding returns an explicit error, no silent degradation.

### Retry & Concurrency ✨ v0.15.0

Built-in request resilience: automatic retry on 429 / 5xx (default 3 times); batch embedding concurrency 8, batch cap 2048; vectors are L2-normalized uniformly for cosine-similarity comparison.

```rust
use langchainrust::{OpenAIEmbeddings, Embeddings, retrieval::graph_rag::EmbeddingMatcher};

let emb = Arc::new(OpenAIEmbeddings::new("sk-..."));
let docs = vec!["Rust ownership".into(), "Borrow checker".into()];
let matcher = EmbeddingMatcher::new(emb, docs);
let top = matcher.query("memory safety in Rust", 2).await?; // 2 most semantically similar
```

## RAG

RAG (Retrieval-Augmented Generation) lets LLMs answer questions based on your private data, not just training knowledge. Flow: documents → split → embed → store in vector DB → retrieve relevant docs → send to LLM with the question.

**Which of the three implementation paths should you pick?**

| Path | Approach | Best for |
|------|----------|----------|
| `RAGPipeline` | All-in-one "retrieve + generate" wrapper, built via builder | Quick start, out-of-the-box |
| LCEL manual chain | Pipe `retriever + prompt \| llm` yourself | Fine control over the prompt and intermediate steps |
| RAG agent | `CorrectiveRAGAgent` / `AdaptiveRAG` | Uncertain retrieval quality, needs self-correction |

**Which retrieval method?** Exact keyword matches → BM25, semantic similarity → vector retrieval, both → hybrid retrieval. Compare in [Hybrid Retrieval](#hybrid-retrieval).

<a id="end-to-end-ragpipeline"></a>
### End-to-End RAGPipeline ✨ v0.15.0

`RAGPipeline` wraps "retrieve + generate" into a ready-to-use pipeline. `RAGPipelineBuilder` offers chained construction: LLM, retriever (or embeddings + vector store combination), recall count `retrieve_k`, system prompt `system`.

```rust
use langchainrust::{
    BM25Retriever, Document, RAGPipelineBuilder, RetrieverTrait,
};

let retriever = BM25Retriever::new();
retriever.add_documents_sync(vec![
    Document::new("Rust is a systems programming language focused on safety and performance.").with_id("intro"),
    Document::new("The ownership system and borrow checking are core to Rust.").with_id("ownership"),
]);

// Retriever approach (zero dependencies, local)
let pipeline = RAGPipelineBuilder::new()
    .llm(llm)
    .retriever(retriever)
    .retrieve_k(2)
    .system("Answer based on the provided context. Do not make things up.")
    .build()?;

// Or embeddings + vector store approach (semantic retrieval)
let pipeline = RAGPipelineBuilder::new()
    .llm(llm)
    .embeddings(OpenAIEmbeddings::new(api_key))
    .vector_store(ChromaDBVectorStore::new(
        ChromaDBConfig::new("http://localhost:8000", "docs", 1536),
    ).await?)
    .retrieve_k(3)
    .build()?;
```

Three call styles:

```rust
// 1. Answer only
let answer: String = pipeline.query("What are Rust's core features?").await?;

// 2. With sources (auditing / evidence display)
let answer_with_sources = pipeline.query_with_sources("What are Rust's core features?").await?;
println!("{}", answer_with_sources.answer);
for src in &answer_with_sources.sources { /* each source Document and similarity */ }

// 3. Into an LCEL pipeline (RagRunnable wrapper)
let rag_chain = RagRunnable::new(Arc::new(pipeline));
let answer = rag_chain.invoke("What are Rust's core features?".to_string(), None).await?;
```

> **Design point**: `RetrieverTrait` unifies the three retriever families — Similarity / BM25 / UnifiedHybrid — and `RAGPipeline` depends on the trait rather than a concrete implementation, so swapping retrieval strategy does not touch business code.

### Document Splitting

Long documents must be split into smaller chunks for effective retrieval. `RecursiveCharacterSplitter` splits by character count, preferring to break at paragraph/sentence boundaries to maintain semantic integrity.

```rust
use langchainrust::{RecursiveCharacterSplitter, TextSplitter};

let splitter = RecursiveCharacterSplitter::new(200, 50);

let chunks = splitter.split_document(&Document::new(
    "Long text to split..."
))?;
```

### SemanticSplitter

Splits by semantic relevance: sentence-tokenize + embed, break where adjacent similarity drops sharply. Better semantic integrity than character-level splitting. Chinese/English sentence boundaries (`。!?;` / `.!?\n`).

```rust
use langchainrust::SemanticSplitter;
use langchainrust::OpenAIEmbeddings;

let splitter = SemanticSplitter::with_defaults(OpenAIEmbeddings::new(config));
// or: SemanticSplitter::new(emb, 0.5, 1000)

let chunks = splitter.split_text(long_text).await;  // Vec<String>
```

**Note**: embedding is async while `TextSplitter` is sync; to avoid breaking the sync trait, this splitter exposes async `split_text` / `split_document` and does not implement sync `TextSplitter`.

### Vector Store

Store embedded documents for similarity retrieval. `InMemoryVectorStore` is for development and testing; use ChromaDB, Qdrant, PGVector, etc. for production persistence.

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

Persistent vector store using Chroma. Requires a running Chroma service (default port 8000). Good for production-grade persistence and retrieval.

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

PostgreSQL + pgvector extension vector store. Good when you already have PostgreSQL infrastructure and want relational DB + vector search in one. Requires the `pgvector-storage` feature; since `sqlx` / `pgvector` deps are not enabled inside the crate, add `sqlx` and `pgvector` to your `Cargo.toml` yourself.

```rust
use langchainrust::vector_stores::PGVectorStore;
use langchainrust::embeddings::Embeddings;

let store = PGVectorStore::new(
    "postgres://user:pass@localhost/db",
    "docs",
    1536, // vector dimension
).await?;
// embeddings: impl Embeddings (e.g. OpenAIEmbeddings); docs: &[Document]
store.add_documents(&docs, &embeddings).await?;
let found = store.similarity_search("query", 5, &embeddings).await?;
store.delete("doc-id").await?;
```

`PGVectorStore::new` runs `CREATE EXTENSION IF NOT EXISTS vector` and creates the table; `build_table_sql(table, dim)` is a pure function for the table DDL.

### PineconeStore

Pinecone cloud vector store (reqwest HTTP API, no feature required, available by default). Good when you want a managed vector service without self-hosting a database.

```rust
use langchainrust::vector_stores::PineconeStore;
use langchainrust::embeddings::Embeddings;

// host format: https://{index-name}.svc.{environment}.pinecone.io
let store = PineconeStore::new("your-api-key", "https://my-index.svc.prod.pinecone.io");

// embeddings: impl Embeddings
store.upsert(&docs, &embeddings).await?;       // auto-embeds documents
let qvec: Vec<f32> = embeddings.embed_query("query").await?; // query takes an embedded vector
let found = store.query(qvec, 5).await?;
store.delete(&["id1".to_string()]).await?;
```

`upsert` calls `embed_documents` automatically; `query` takes an already-embedded vector (result of `embed_query`).

### Unified `VectorStore` Trait ✨ v0.15.0

All backends implement the same 10-method `VectorStore` trait — consistent interface, plug-and-play:

| Method | Description |
|--------|-------------|
| `add_documents` | Batch write (document + vector) |
| `similarity_search` | Vector similarity retrieval (descending) |
| `embed_query` / `similarity_search_text` | Backends with a built-in embedder can take text directly |
| `similarity_search_with_min_score` | With a minimum-score threshold |
| `get_document` / `get_embedding` | Read by ID |
| `delete_document` / `count` / `clear` | Management |

```rust
use langchainrust::vector_stores::{VectorStore, VectorStoreBuilder};

// Unified factory: switch backends under the same trait
let store: Arc<dyn VectorStore> = VectorStoreBuilder::in_memory().build().await?;
let store = VectorStoreBuilder::file_backed("kb.bin", 384).build().await?;
let store = VectorStoreBuilder::qdrant("http://localhost:6334", "kb").build().await?;
```

**Error type** `VectorStoreError` has four variants: `DocumentNotFound` / `EmbeddingError` / `StorageError` / `ConnectionError`.

**Backend list**: `InMemoryVectorStore`, `ChromaDBVectorStore`, `PGVectorStore`, `PineconeStore`, `LanceDBVectorStore`, `Neo4jVectorStore`, `QdrantVectorStore`, `FileVectorStore`, `ChunkedVectorStore`, plus the `DocumentStore` family (`InMemoryDocumentStore` / `MongoChunkedDocumentStore` / `RedisDocumentStore` / `SQLiteDocumentStore`).

> **Honest errors, no silent degradation**: backends requiring a feature (e.g. Qdrant) return an explicit error when the feature is off (pointing to `qdrant-integration`), and do **not** silently fall back to in-memory storage — otherwise production code would think it's writing persistent data and lose everything on restart.

---

## BM25

BM25 is a classic keyword retrieval algorithm that scores relevance based on term frequency and document length. Unlike vector retrieval (semantic similarity), BM25 excels at exact keyword matching — searching "Rust ownership" prioritizes documents containing those exact words. No embedding model needed, zero cost, fast.

### BM25Retriever (Keyword Search)

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

### BM25 Parameters

k1 controls term frequency saturation (higher = more weight on frequent terms), b controls document length normalization (higher = more penalty for long docs). Defaults k1=1.5, b=0.75 work well for most cases.

| Parameter | Default | Description |
|-----------|---------|-------------|
| k1 | 1.5 | Term frequency saturation |
| b | 0.75 | Document length normalization |

```rust
let retriever = BM25Retriever::with_params(2.0, 0.5);
```

### ChunkedBM25Retriever (Parent-Child)

Solves the "small chunk matches but loses context" problem: documents are split into leaf chunks for the BM25 index, and when multiple leaf chunks from the same parent match, they're automatically merged into the full parent document.

```rust
use langchainrust::{ChunkedBM25Retriever, AutoMergingConfig, ChunkedDocumentStore};

let config = AutoMergingConfig::new()
    .with_leaf_size(400)      // Leaf chunk size
    .with_threshold(0.5);     // Merge when 50%+ leaves match

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

## Hybrid Retrieval

Vector retrieval excels at semantic similarity, BM25 excels at keyword matching — they complement each other. Hybrid retrieval uses both and merges results with RRF (Reciprocal Rank Fusion), achieving higher recall than either method alone.

### RRF Fusion Algorithm

```
RRF_score(d) = Σ 1/(k + rank(d))
```

Where k=60, rank(d) is document rank in each result list.

### UnifiedHybridIndex

All-in-one hybrid retrieval: internally maintains both BM25 and vector indexes, auto-dual-indexes when adding documents, auto-dual-retrieves + RRF merges on query. No need to manually manage two indexes.

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

// Auto-build dual index
index.add_document(Document::new("Document content")).await?;

// Hybrid search
let results = index.retrieve("query", 5).await?;

for result in results {
    println!("Content: {}", result.document.content);
    println!("RRF Score: {}", result.score);
}
```

### Retrieval Mode Comparison

| Mode | Content Storage | Lookup | Use Case |
|------|------------------|--------|----------|
| SimpleVector | InMemoryVectorStore | No lookup | Vector-only, simple |
| BM25 Only | ChunkedDocumentStore | Lookup | Keyword-only |
| Hybrid | ChunkedDocumentStore (shared) | Lookup | Combined (recommended) |

---

## LangGraph

LangGraph defines complex workflows as directed graphs: each node is a processing step, edges define execution order. More flexible than Chains — supports conditional branching, loops, human-in-the-loop, and subgraphs. Use when you need fine-grained control over execution flow.

### StateGraph

The most basic graph — define nodes and edges, state passes between nodes. `AgentState` is the built-in state struct with `messages`, `steps`, and other fields.

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

let result = compiled.invoke(AgentState::new("user question".to_string())).await?;
```

### Conditional Edge

Dynamically choose the next node based on current state. `FunctionRouter` takes a closure that returns the target node name. Good for "summarize if many messages, otherwise continue" branching logic.

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
    None, // default target; used when the router returns a value not in targets
);
```

### Human-in-the-loop / Interrupt & Resume

Pause execution before critical nodes, wait for human confirmation, then continue. `with_interrupt_before` specifies which nodes to interrupt at; `MemoryCheckpointer` saves execution state for cross-session resume.

```rust
use langchainrust::langgraph::{GraphError, MemoryCheckpointer};

let compiled = graph.compile()
    .map_err(|e| ...)?
    .with_checkpointer(MemoryCheckpointer::new())
    .with_interrupt_before(vec!["output", "analyze"]);

match compiled.invoke(state).await {
    Ok(result) => { /* complete */ }
    Err(GraphError::ExecutionInterrupted(node)) => {
        println!("Paused at: {}", node);
        if let Some(exec) = compiled.create_resume_execution(&node).await {
            let result = compiled.resume(exec).await?;
        }
    }
    Err(e) => { /* error */ }
}
```

### Reducers (State Merge Rules) ✨ v0.15.0

How a child node's returned state merges into the shared state is decided by the Reducer:

- `ReplaceReducer` — overwrite the field directly (default)
- `AppendReducer` — append (the `messages` array uses it to accumulate across steps)

```rust
use langchainrust::langgraph::{StateGraph, AppendMessagesReducer};

let mut graph = StateGraph::new();
graph.set_reducer("messages", std::sync::Arc::new(AppendMessagesReducer));
```

### Edge Types ✨ v0.15.0

`GraphEdge` has four edge kinds:

| Edge | Semantics |
|------|-----------|
| `Fixed` | Fixed jump `source → target` |
| `Conditional` | Dynamic selection via a routing function |
| `FanOut` | One node fans out to multiple targets in parallel |
| `FanIn` | Multiple nodes converge into one merge point |

```rust
graph.add_fan_out("query", vec!["crag".to_string(), "graph".to_string(), "vector".to_string()]);
graph.add_fan_in(vec!["crag".to_string(), "graph".to_string(), "vector".to_string()], "merge");
```

### Checkpointer Family ✨ v0.15.0

- `MemoryCheckpointer` — in-process (single-threaded)
- `ThreadSafeMemoryCheckpointer` — concurrency-safe
- `FileCheckpointer::new(path)` — persisted to disk (does **not** implement `Default`; the path must be explicit, failures propagate)

Pair with `with_checkpointer` + `with_interrupt_before` for "pause → resume" workflows.

### Graph Definition Persistence ✨ v0.15.0

The `GraphPersistence` trait stores graph definitions (nodes / edges / reducers) for reuse: `MemoryPersistence` / `FilePersistence` / `MongoPersistence`.

### Subgraph / Dynamic Planning / Streaming ✨ v0.15.0

- `SubgraphNode` — graph-in-graph, wrapping a sub-flow as a reusable node
- `DynamicPlanner` / `DynamicInjection` / `DynamicTask` — construct tasks at runtime, inject parallel branches
- `compiled.stream_collected(input)` — returns `Vec<StreamEvent<S>>` to observe node execution progress step by step

---

## Document Loaders

Load documents from various file formats, converting them into a unified `Document` structure (`content` + `metadata`) for downstream splitting and retrieval.

### Document Family ✨ v0.15.0

Unified data structures run through the whole chain — load → split → store → retrieve:

| Type | Purpose |
|------|---------|
| `Document` | Raw document: `content` + `metadata` (chainable `with_id` / `with_metadata`) |
| `VectorDocument` | Document carrying a vector (internal vector-store storage) |
| `SearchResult` | Retrieval result: `document` + `score` |
| `ChunkDocument` | Leaf block in a parent-child structure, holding a parent-document reference |

`RecursiveCharacterSplitter` picks delimiters by priority: **paragraph → line → sentence → character**, only degrading to the next level when the previous one still exceeds the limit — keeping semantics as intact as possible.

### Supported Formats

| Loader | Format | Features |
|--------|--------|----------|
| **TextLoader** | .txt | Line-by-line splitting |
| **JSONLoader** | .json | Specify content_key |
| **MarkdownLoader** | .md | Split by heading level |
| **PDFLoader** | .pdf | Extract PDF text |
| **CSVLoader** | .csv | Each row as document |

### TextLoader

Load plain text files. Supports whole-file loading and line-by-line splitting.

```rust
use langchainrust::{TextLoader, DocumentLoader};

let loader = TextLoader::new("document.txt");
let docs = loader.load().await?;

// Split by lines
let loader = TextLoader::new_with_line_split("document.txt");
let docs = loader.load().await?;
```

### JSONLoader

Load JSON files. By default extracts the entire JSON string as content; specify `content_key` to extract only a specific field's value.

```rust
use langchainrust::{JSONLoader, DocumentLoader};

let loader = JSONLoader::new("data.json");
let docs = loader.load().await?;

// Specify content field
let loader = JSONLoader::new_with_content_key("data.json", "content");
let docs = loader.load().await?;
```

### MarkdownLoader

Load Markdown files. Supports splitting by heading level — content under each heading becomes a separate document, maintaining section-level semantic integrity.

```rust
use langchainrust::{MarkdownLoader, DocumentLoader};

// Split by heading level
let loader = MarkdownLoader::new_with_heading_split("guide.md", 1);
let docs = loader.load().await?;
```

### HTMLLoader

Strips `<script>`/`<style>`, removes tags, decodes common HTML entities, and collapses whitespace to extract plain text from an HTML string or URL.

```rust
use langchainrust::retrieval::HTMLLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

// From an HTML string
let loader = HTMLLoader::new("<p>Hello <b>world</b></p>");
let docs = loader.load().await?; // content: "Hello world"

// From a URL (fetched asynchronously, then parsed)
let loader = HTMLLoader::from_url("https://example.com");
let docs = loader.load().await?;

// Pure function: extract text directly
let text = HTMLLoader::extract_text("<script>x</script><p>a &amp; b</p>");
// -> "a & b"
```

### DocxLoader ✨ v0.4.1

Parse Word `.docx` files: ZIP extraction + XML `<w:t>` text node parsing.

```rust
use langchainrust::retrieval::loaders::DocxLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

let loader = DocxLoader::new("document.docx");
let docs = loader.load().await?;
```

### WebScraperLoader ✨ v0.4.1

Web page scraping: extract page text, with recursive same-domain link following.

```rust
use langchainrust::retrieval::loaders::WebScraperLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

let loader = WebScraperLoader::new("https://example.com")
    .with_max_depth(2)
    .with_max_pages(10);
let docs = loader.load().await?;
```

### SitemapLoader ✨ v0.4.1

Parse `sitemap.xml` and batch-crawl pages.

```rust
use langchainrust::retrieval::loaders::SitemapLoader;
use langchainrust::retrieval::loaders::DocumentLoader;

let loader = SitemapLoader::new("https://example.com/sitemap.xml")
    .with_max_pages(50);
let docs = loader.load().await?;
```

---

## MultiQueryRetriever

A user's query may not match document wording, causing retrieval misses. MultiQueryRetriever uses an LLM to rewrite one query into multiple variations, retrieves for each, then merges and deduplicates — improving recall.

It's positioned as a "query expansion" enhanced retriever: it turns one retrieval into many, using different phrasings to fish the same corpus, specifically addressing **insufficient recall**. The typical scenario is free-form user questions and inconsistent document terminology — the document says "DB connection timeout", the user asks "database timeout", a single retrieval's keywords don't line up, and the relevant passage never makes it into the Top-K. Retrieving multiple variations in parallel is like opening the book with several phrasings at once, so misses become much less likely. It suits recall-sensitive scenarios where you'd rather return more and let downstream precision-ranking pick the best.

### Use Cases

| Scenario | Symptom | Suggestion |
|---|---|---|
| User query wording doesn't match document terminology | Low relevance, missed recalls | Use MultiQueryRetriever, multiple variations cover different phrasings |
| The same concept has several names in documents | Low hit rate for synonyms and aliases | Use MultiQueryRetriever, LLM rewriting produces synonymous expressions |
| Query is short and vague, intent not expanded | Retrieval results scattered, unfocused | Use MultiQueryRetriever, multi-variant rewriting splits the intent |
| Don't want extra LLM calls, limited budget | Retrieval is good enough, no extra generation wanted | Use StaticQueryGenerator or a plain retriever |
| Retrieval must be precise, small return set | Precision matters more than recall | Multi-variant recall, then reranking |

### How It Works

```
User query → LLM generates N variations → Retrieve each → Merge & dedupe → Return results
```

Key behaviors:

- **Query rewriting**: The LLM rewrites the original query into N variations, the count controlled by `with_num_queries`. Rewriting isn't just rephrasing — it also breaks the intent down from different angles, covering synonyms, abbreviations, and colloquial expressions, so each variant can hit a different kind of document.
- **Parallel retrieval**: Each variation calls the underlying retriever separately, and each path returns `k_per_query` results. Any retriever implementing `RetrieverTrait` works as the base — `SimilarityRetriever`, `BM25Retriever`, and `UnifiedHybridIndex` all fit, not limited to vector retrieval.
- **Merge & dedupe**: Aggregate the N result sets; if the same document is found by multiple variants, keep only one copy.
- **Truncated return**: The merged results are truncated to `final_k`, returning the final Top-K. Multi-variant recall amplifies the return volume, and `final_k` is the final gate controlling how many results feed downstream.

### Usage

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

Parameters:

| Parameter | Purpose | Example |
|---|---|---|
| `with_num_queries` | Number of query variations the LLM generates; more variations mean wider coverage but more expensive LLM calls | `3` |
| `with_k_per_query` | Number of results each variation retrieves; determines the recall depth per variant | `5` |
| `with_final_k` | Number of results finally returned after merge & dedupe; the final count fed downstream | `10` |

Notes:

- **LLM is not limited to OpenAI**: MultiQueryRetriever internally holds a chat model implementing `BaseChatModel` (as a trait object). The `OpenAIChat` in the example is just one of them — any provider works.
- **Parsing LLM output is the fragile point**: Query variations come from the LLM's free-text output, parsed by splitting lines. If the model output carries numbering ("1. xxx"), quotes, or extra explanation, that dirty text may be treated as a query and cause one variant to recall odd results. In production, give the model explicit output-format requirements.
- **An enhancer is not a retriever**: MultiQueryRetriever consumes `Arc<dyn RetrieverTrait>` but doesn't implement the trait itself, so it can't be wrapped by another layer of enhanced retriever.

### StaticQueryGenerator (No LLM)

Query generator that doesn't need an LLM — expands queries via a synonym table. Use when you don't want extra LLM calls, or when query patterns are predictable.

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

Key behaviors:

- **Word-level expansion**: `generate` looks up query terms in the synonym table and replaces or expands matched words into multiple variations. No LLM involved — zero extra calls, zero latency.
- **Trade-off vs MultiQueryRetriever**: StaticQueryGenerator is "dictionary-style" expansion — it only handles pre-registered synonyms and won't generate brand-new natural-language phrasings. MultiQueryRetriever is "generative" expansion — more flexible variants but more expensive. Use the former when synonyms are clear and query patterns are predictable; use the latter when corpus terminology is complex and generative rewriting is needed.
- **Returns a query list**: `generate` returns the expanded query list, and you decide how to hand them to a retriever.

---

## HyDE Retriever

**HyDE (Hypothetical Document Embeddings)** solves the "query too short, doesn't match documents" problem: first use an LLM to generate a hypothetical answer (which may be inaccurate), then use the hypothetical answer's embedding to retrieve real documents. The hypothetical answer's wording is closer to real documents, so retrieval works better.

The idea is "write the answer first, then look it up": a short query is an old problem in vector retrieval. A phrase like "Rust concurrency" only encodes the semantics of a few keywords in its embedding, far from the expanded phrasings in documents — "async/await, thread safety, data races" — so similarity scores come out low. HyDE has the LLM write a hypothetical answer document for the query; that text's wording, sentence length, and information density are much closer to real documents, so retrieving with its embedding naturally hits more. Note that the hypothetical answer itself can be wrong — it's only used to "align the wording"; what's actually returned is the retrieved real document. Suitable for queries that are too short, too colloquial, or too far in style from documents' long-form writing.

### Use Cases

| Scenario | Symptom | Suggestion |
|---|---|---|
| Query too short (a few keywords) | Embedding captures only keywords, low similarity to long documents | Use HyDE, generate a hypothetical document first, then retrieve |
| Query colloquial, documents written long-form | Wording style mismatch, poor retrieval | Use HyDE, the hypothetical document converts colloquial to written long-form style |
| User question far from document phrasing | Vector similarity unreliable | Use HyDE to improve recall |
| Worried the hypothetical answer biases retrieval | Generated content quality is unstable | Turn on `with_include_original_query` to fold the original query into retrieval |
| Recall sufficient, only precision wanted | Retrieved results are relevant enough | No HyDE needed, retrieve directly + rerank |

### How It Works

```
User query → LLM generates hypothetical document → Retrieve using hypothetical doc → Return real docs
```

Key behaviors:

- **Generate a hypothetical document**: The LLM writes a plausible-looking answer (the hypothetical answer) for the query. The value of this step isn't "answering correctly" but "writing like a document" — fleshing a short query into long-form text in the same style as real documents.
- **Retrieve with the hypothetical document**: Hand the hypothetical document to the underlying retriever. In the example the base is `SimilarityRetriever`, which first embeds the hypothetical document internally, then computes similarity against the documents in the store. So HyDE itself doesn't need a separate embedding-model parameter — vectorizing the hypothetical document is handled by the underlying retriever.
- **Return real documents**: Both hitting and ranking happen between "hypothetical document ↔ real documents", and what's finally returned is the real documents, not the hypothetical one. The hypothetical document only exists at the retrieval moment and is discarded after use.

### Usage

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

Parameters:

| Parameter | Purpose |
|---|---|
| `with_k(5)` | Final number of results returned; returns the top k real documents to downstream |
| `with_include_original_query(true)` | Whether to use the original query together with the hypothetical document as retrieval entry points. When on, it's effectively dual-path retrieval — "original phrasing + hypothetical answer" — reducing the risk of the hypothetical answer biasing results |

Notes:

- **LLM is not limited to OpenAI**: HyDERetriever holds a chat model implementing `BaseChatModel` (trait object); the `OpenAIChat` in the example is just one of them.
- **The underlying retriever is an interface**: HyDE consumes `Arc<dyn RetrieverTrait>`; `SimilarityRetriever`, `BM25Retriever`, and `UnifiedHybridIndex` can all serve as the base. For keyword retrieval too, a hypothetical document contains more complete keywords than a short query, so it helps there as well.
- **An enhancer is not a retriever**: Like MultiQueryRetriever, HyDERetriever doesn't implement `RetrieverTrait` itself and can't be wrapped by another enhancement layer.

---

## Reranking

Initial retrieval may return less-relevant results. Rerankers re-score retrieval results, pushing the most relevant to the top for better precision.

It's positioned as a precision-ranking step "after recall, before feeding the model". The first retrieval (recall) pursues "don't miss anything" and would rather return more; reranking does one more strict scoring pass over the recalled results, pushing irrelevant ones down and the most relevant to the front, then keeps `top_n` for downstream. The problem it solves is **insufficient precision** — recalled results are mixed with irrelevant passages, and feeding all of them to the LLM dilutes attention and wastes context. It pairs well with recall-expanding enhancers like MultiQueryRetriever and HyDE: the enhancer casts a wider net, reranking curates the catch.

### Use Cases

| Scenario | Symptom | Suggestion |
|---|---|---|
| Many recall results with uneven relevance | Relevant documents buried among irrelevant ones | Use reranking, push the most relevant to the front |
| Pairing with MultiQuery/HyDE | Multi-variant recall amplifies volume and adds noise | Rerank, keep only `top_n` for downstream |
| Single retrieval, results already precise | The first few results are what you want | No reranking needed, save one scoring pass |
| Want a controllable return count | Per-path recall volume is uncontrollable | Use `with_top_n` to fix the final count |

### Supported Rerankers

| Reranker | Description |
|----------|-------------|
| **KeywordReranker** | Keyword matching reranking |
| **BM25Reranker** | BM25 formula reranking |

Neither needs an extra model call — both score the passed-in retrieval results directly, fast and cheap. The difference lies in how refined the scoring formula is; how to choose is shown below:

| How to choose | KeywordReranker | BM25Reranker |
|---|---|---|
| Scoring basis | Position and count of query keywords in documents | BM25 formula: term frequency + rarity + document-length normalization |
| Complexity | Simple, keyword hits score high | More precise, better discrimination |
| Embedding model needed? | No | No |
| Adjustable parameters | None | `with_params(k1, b)`, controls term-frequency saturation and length-penalty strength |
| Best for | Fast, rough, small result sets | Large result sets, need finer discrimination |

### KeywordReranker

Rerank based on keyword matching — more query keywords appearing in a document and earlier in the text means a higher score. Simple and fast, no embedding model needed.

```rust
use langchainrust::{KeywordReranker, RerankingExecutor};

let reranker = Box::new(KeywordReranker::new());

let executor = RerankingExecutor::new(reranker)
    .with_top_n(5)
    .with_min_score(0.5);

let reranked = executor.rerank("Rust programming", search_results)?;
```

Key behaviors:

- **Scoring mechanism**: Counts the occurrences and positions of query keywords in each retrieved result — more occurrences and earlier positions mean higher scores. This is "keyword-hit" scoring and doesn't involve semantics.
- **Keep top_n**: `with_top_n(5)` means only the top 5 are kept after reranking and the rest discarded; `rerank` returns exactly those 5, so downstream gets a deterministic count.
- **Minimum-score filtering**: `with_min_score(0.5)` sets a score floor; results below it are filtered out, used to drop clearly irrelevant results; if unset, nothing is filtered.
- **Decoupled from retrieval**: `rerank` accepts the passed-in `search_results` list and doesn't care which retriever produced them, so it can follow any retrieval results — `SimilarityRetriever`, MultiQueryRetriever, HyDE, and so on.

### BM25Reranker

Rerank using the BM25 formula — more precise than KeywordReranker, considering term frequency saturation and document length normalization. Adjustable k1/b parameters.

```rust
use langchainrust::{BM25Reranker, RerankingExecutor};

let reranker = Box::new(BM25Reranker::new()
    .with_params(2.0, 0.5));

let executor = RerankingExecutor::new(reranker).with_top_n(5);

let reranked = executor.rerank("Rust programming", results)?;
```

Key behaviors:

- **Scoring mechanism**: Scores with the BM25 formula, adding three considerations over keyword hits — term-frequency saturation (diminishing marginal returns once a term appears enough), document-length normalization (one more occurrence in a long document isn't remarkable), and inverse document frequency (rarer terms matter more).
- **Adjustable parameters**: The two parameters of `with_params(k1, b)` control the strength of term-frequency saturation and length normalization respectively; the example `(2.0, 0.5)` is a common starting point and can be fine-tuned on real data.
- **Keep top_n**: `with_top_n(5)` determines the final count kept — only the top 5 are returned after reranking. The example doesn't set `with_min_score`, so nothing is filtered by score by default.
- **Also model-free**: Like KeywordReranker, no embedding model is needed — it scores already-retrieved results directly, keeping cost controlled.

---

## Callbacks

The callback system lets you insert custom logic at key points in LLM calls (start, end, error, streaming token) for logging, tracing, and monitoring. `CallbackManager` manages multiple handlers and triggers them in order.

### CallbackManager

Manage multiple callback handlers, supporting composition (e.g., output to both console and LangSmith simultaneously):

```rust
use langchainrust::{CallbackManager, StdOutHandler, LangSmithHandler};
use std::sync::Arc;

let manager = CallbackManager::new()
    .add_handler(Arc::new(StdOutHandler::new()))
    .add_handler(Arc::new(LangSmithHandler::from_env()?));
```

### StdOutHandler

Output to stdout (for debugging). The simplest callback — directly prints LLM input and output.

```rust
use langchainrust::StdOutHandler;

let handler = StdOutHandler::new();
```

### FileCallbackHandler

Output to file. Supports JSON format (for programmatic parsing) and text format (for human reading).

```rust
use langchainrust::{FileCallbackHandler, LogFormat};

// JSON format
let handler = FileCallbackHandler::new("trace.json", LogFormat::Json);

// Text format
let handler = FileCallbackHandler::new("trace.log", LogFormat::Text);
```

### CallbackHandler Lifecycle ✨ v0.15.0

Implementing `CallbackHandler` plugs you into the callback system. Every Run has a three-phase lifecycle: `on_run_start` → `on_run_end` / `on_run_error`; component-level hooks (`on_llm_start/end/new_token/thinking/error`, `on_chain_*`, `on_tool_*`, `on_retriever_*`) are optional to override, with no-op defaults. `StdOutHandler`'s `verbose` flag controls whether component-level detail is printed.

### LangSmith Tracing

LangSmith is LangChain's official tracing platform for monitoring and debugging LLM applications.

#### Environment Variables

```bash
export LANGSMITH_API_KEY="ls_xxxxx"       # Required
export LANGSMITH_PROJECT="my-project"      # Project name
export LANGSMITH_TRACING="true"            # Enable tracing
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
```

#### Use LangSmithHandler

```rust
use langchainrust::{CallbackManager, LangSmithHandler, StdOutHandler};
use std::sync::Arc;

// Auto-configure from environment
let langsmith = LangSmithHandler::from_env()?;

let manager = CallbackManager::new()
    .add_handler(Arc::new(StdOutHandler::new()))
    .add_handler(Arc::new(langsmith));
```

#### Manual Configuration

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

#### LangSmith Features

| Feature | Description |
|---------|-------------|
| **Tracing** | Record every LLM call |
| **Monitoring** | View token usage, latency |
| **Debugging** | Compare different version outputs |
| **Evaluation** | Test set evaluation |
| **Sharing** | Share trace links |

---

### OtelHandler

Converts LLM / Chain / Tool / Retriever start / end / error events into OpenTelemetry spans. Requires the `opentelemetry` feature and a configured global tracer provider.

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

Nested spans; export to Jaeger / Tempo / Grafana.

---

## Evaluation

Quantify LLM output quality: after changing prompts / models / adding RAG, run an eval set and see if scores improved. 10 evaluators in 5 categories, covering everything from literal matching to RAG hallucination detection:

| Category | Evaluators | Description |
|----------|-----------|-------------|
| Literal | `ExactMatch` / `StringDistance` | exact equal / Levenshtein distance normalized |
| Semantic | `EmbeddingSimilarity` / `LLMAsJudge` / `PairwiseJudge` | cosine / LLM judge / pairwise (swap A/B to remove position bias) |
| Rule | `ContainsKeyword` / `RegexMatch` / `LengthCheck` | keyword / regex / length |
| Classic NLP | `Bleu` | n-gram precision (char-level + smoothing) |
| RAG | `Faithfulness` | split claims, verify each, detect hallucination |

### EvalRunner

Run a set of evaluators over a `Dataset`, produce a `Report` (per-example scores + per-evaluator averages). Supports loading eval sets from JSONL files.

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

Splits the prediction into atomic claims and verifies each against the reference (context), detecting fabrication. Most useful for RAG.

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

`with_llm_split(true)` uses LLM to split claims (default: by period); `with_empty_score(x)` sets the score when no claims. Verification runs concurrently (`join_all`).

### LLMAsJudge ✨ v0.15.0

Uses an LLM to score on a 0-10 scale, with a customizable rubric (`with_rubric`) and max score (`with_max_score`).

```rust
use langchainrust::evaluation::LLMAsJudge;

let judge = LLMAsJudge::new(OpenAIChat::new(config))
    .with_rubric("Score on correctness, completeness, and clarity")
    .with_max_score(10);
let score = judge.eval(input, output, reference).await?; // 0.0 ~ 10.0
```

### PairwiseJudge (Pairwise Comparison) ✨ v0.15.0

Arena mode: the LLM judge picks between two answers, returning `Verdict::{AWins, BWins, Tie}`.

```rust
use langchainrust::evaluation::{PairwiseJudge, Verdict};

let judge = PairwiseJudge::new(OpenAIChat::new(config));
match judge.compare("the question?", &answer_a, &answer_b).await? {
    Verdict::AWins => { /* A is better */ }
    Verdict::BWins => { /* B is better */ }
    Verdict::Tie    => { /* tie */ }
}
```

> **Position-bias mitigation**: A/B order is swapped automatically and the comparison runs twice; only when both runs pick the same winner does it count as a real win, otherwise it's a tie. The two calls fire concurrently, adding no serial round trips.

### Report Fault Tolerance ✨ v0.15.0

`EvalRunner.run` is fault-tolerant per example: a single `predict` failure or one evaluator's scoring failure only lands in `Report::failures` (with `index` and `stage`); the rest of the examples score normally — one bad data point never sinks the whole evaluation.

```rust
let report = runner.run(&dataset, &MyLLM).await?;
if !report.failures.is_empty() {
    eprintln!("{} failures", report.failures.len());
}
```

> Underneath, they reuse `core::judge::structured_call`'s structured-decision path (forcing the LLM to emit JSON before parsing, with errors unified as `StructuredJudgeError`), so judge results stay machine-readable.

---

## MongoDB Storage

MongoDB storage solves two kinds of problems: first, persisting the **document store** to MongoDB so the "parent document + child chunks" relationship of long documents is **shared across processes and survives restarts**; second, persisting **conversation memory** to MongoDB so multi-turn memory is **shared across multiple instances**. Default in-memory stores (like `InMemoryChunkedDocumentStore`) are gone the moment the process exits, while dedicated vector databases are too heavy; when an app runs multiple instances or needs real persistence, MongoDB is a production-grade middle option.

This section introduces two kinds of objects with different purposes — don't confuse them:

| Object | Family | What it stores | In one sentence |
|---|---|---|---|
| `MongoChunkedDocumentStore` | Document store (DocumentStore family) | Document text + parent/child chunk relationships | After long documents are chunked, the text lands in MongoDB |
| `MongoPersistentMemory` | Persistent memory (memory family) | Conversation history / compressed summaries | Multiple instances share the same conversation memory |

Workflow (using the document store as an example): first `create_indexes()` builds the query indexes → `add_parent_document(doc, 500)` chunks the long document by chunk size and stores the child chunks → retrieve all child chunks by parent document ID with `get_chunks_for_parent`. After a search hits a small chunk, use the chunk ID to go back to the document store for the parent chunk's text — this is the standard back-to-source path for chunked retrieval.

### Use Cases (When to Choose MongoDB)

| Scenario | Recommended? | Reason |
|---|---|---|
| Multi-instance deployment sharing the same documents/memory | ✅ Recommended | All instances connect to the same MongoDB and read the same data |
| Data must survive service restarts | ✅ Recommended | Data is persisted, not dependent on process memory |
| Single-machine demo, small data, zero dependencies wanted | ⚠️ Can swap to a lighter backend | SQLite / file storage can be used locally instead |
| Process-restart data loss is acceptable | ❌ Not necessary | In-memory implementations are simpler, no service needed |

### Enable Feature

```toml
[dependencies]
langchainrust = { version = "0.8", features = ["mongodb-persistence"] }
```

### Usage

```rust
use langchainrust::{MongoChunkedDocumentStore, MongoStoreConfig, ChunkedDocumentStoreTrait};

let config = MongoStoreConfig::new(
    "mongodb://localhost:27017",
    "langchainrust_db"
);

let store = MongoChunkedDocumentStore::new(config).await?;
store.create_indexes().await?;

// Same interface as InMemory
let (parent_id, chunk_ids) = store.add_parent_document(doc, 500).await?;
let chunks = store.get_chunks_for_parent(&parent_id).await?;
```

`MongoStoreConfig::new` takes two arguments:

| Parameter | Meaning | Example |
|---|---|---|
| First parameter | MongoDB connection string | `mongodb://localhost:27017` |
| Second parameter | Database name | `langchainrust_db` |

### MongoPersistentMemory (Persisting Conversation Memory)

`MongoChunkedDocumentStore` manages "document text", `MongoPersistentMemory` manages "conversation memory" — the two are persistence at different layers. `MongoPersistentMemory` (see [MongoPersistentMemory](#mongopersistentmemory) for details) internally composes `ConversationSummaryBufferMemory` with its own token budget, writing "history + summary" into MongoDB; multiple instances connected to the same database can share the same memory.

| Behavior | Description |
|---|---|
| Persistence | Memory is stored in MongoDB, survives service restarts |
| Multi-instance sharing | Same database, same collection, multiple instances read/write the same memory |
| Token budget | Internally a summary buffer, auto-compresses when over budget |
| Optimistic locking | Concurrent writes don't overwrite each other, preventing "last write clobbers first write" |
| Session binding | `set_session_id_async` binds the current session |

### Key Behaviors

- `create_indexes()` builds indexes: for first-time database setup, call it first to prepare for subsequent queries by parent ID / chunk ID.
- Parent/child chunk relationship is persisted: deleting a parent document also deletes all its child chunks.
- Interface is identical to the InMemory implementation: the same `ChunkedDocumentStoreTrait`, switching backends only changes the constructor line.
- Stores text, not vectors: vectors are indexed by a companion vector store (like `ChunkedVectorStore`); `MongoChunkedDocumentStore` only handles "text + chunk relationships".
- Provides `_blocking` synchronous methods for synchronous retrieval paths like BM25.

### How to Choose

When should you use the MongoDB document store? In one sentence: when you need **multiple processes/instances to share the same document text**, or need **real persistence**. If you're single-machine with small data, the SQLite document store is lighter (see next section); if you have large data and want professional vector retrieval, pair it with a vector store — vectors go in the vector store, text goes here.

---

<a id="redis--sqlite-storage"></a>
## Redis / SQLite Storage

Both are lightweight implementations of `ChunkedDocumentStoreTrait` that manage "document text + parent/child chunks" — doing the same job as the MongoDB document store, but with the opposite trade-offs: **Redis goes distributed/shared, SQLite goes local/zero-dependency**. Which one you pick depends on what infrastructure you already have and whether the data needs to be shared across instances.

The workflow is exactly the same as the MongoDB document store: `add_parent_document(doc, 500)` chunks and stores → `get_chunks_for_parent` retrieves child chunks by parent ID. The interface is unified; switching backends only changes the constructor line.

### Use Cases and Trade-offs

| Backend | Where data lives | Needs external service | Data lifecycle | Use case |
|---|---|---|---|---|
| `RedisDocumentStore` | Redis server memory | Yes, start Redis first | Resides in Redis; whether it survives depends on Redis's own persistence config (RDB/AOF), not managed by this library | Multi-instance sharing, existing Redis infrastructure, need cross-process consistency |
| `SQLiteDocumentStore` | Local `.db` file | No, zero dependency | Written directly to a local file, data survives process exit | Single machine, local development, no service |

A one-line memory aid: Redis is "a shared warehouse used by many people", SQLite is "this machine's own drawer".

### Enable Feature

```toml
[dependencies]
langchainrust = { version = "0.8", features = ["redis-storage"] }
```

or

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

Note the differing semantics of the two constructor arguments:

| Backend | Constructor argument | Meaning |
|---|---|---|
| `RedisDocumentStore::new(uri)` | `redis://127.0.0.1:6379` | Redis connection string, pointing at an already-running service |
| `SQLiteDocumentStore::new(path)` | `langchain.db` | Local file path; the file is auto-created if it doesn't exist |

### Key Behaviors

- Both implement `ChunkedDocumentStoreTrait` and are **document stores** — they store document text and parent/child chunk relationships, **not vectors**; vectors go in a vector store.
- `add_parent_document` auto-chunks and returns `(parent_id, chunk_ids)`; `get_chunks_for_parent` retrieves all child chunks by parent ID.
- Deleting a parent document also deletes all its child chunks.
- Provides `_blocking` synchronous methods for synchronous retrieval paths like BM25.
- Interface is identical to the InMemory implementation; switching backends only changes the constructor line.
- `RedisDocumentStore` data visibility depends on whether all instances connect to the same Redis; `SQLiteDocumentStore` data lives in the local file, most natural for single-machine use.

### How to Choose

| Your situation | Recommendation |
|---|---|
| Single machine / local development / don't want to install any service | `SQLiteDocumentStore` |
| Multi-instance deployment / team already uses Redis infrastructure | `RedisDocumentStore` |
| Large data, need production-grade reliability and more complex queries | `MongoChunkedDocumentStore` (previous section) |

### Feature Gating

| Feature Flag | Storage Backend | Dependencies |
|-------------|-----------------|--------------|
| `redis-storage` | Redis | redis crate |
| `sqlite-storage` | SQLite | rusqlite crate |
| `mongodb-persistence` | MongoDB | mongodb crate |

---

## Testing

### Concept: Why Test

In a workspace of 21 crates, tests are the last line of defense ensuring each module behaves correctly — once pure logic like parsers, caches, and state machines goes wrong, it silently affects every layer above. Running `cargo test` from the workspace root executes each crate's unit tests, integration tests, and doc tests together, catching problems early.

```bash
cargo test
```

### Test Coverage Scope

| Level | Location | Covered By |
|---|---|---|
| Unit tests | Inside each crate | Pure logic: parsers, caches, eviction policies, state transitions, etc. |
| Integration tests | The facade crate (`langchainrust`)'s tests directory | Cross-module assembly, external behavior |
| Doc tests (doctest) | Code examples in public API docs | Examples are runnable, APIs really exist |

### How to Run

- `cargo test` at the root: runs across all crates
- `cargo test --workspace`: explicitly runs the full workspace
- Run a single crate: `cargo test -p langchainrust` (the facade crate's package name), or `cd` into the crate directory and run it there

### How to Write Tests for the Lib

**Unit tests**: wrap test code in `#[cfg(test)]` modules inside the implementation file, testing only the module's internal logic — no network, no real models. Behaviors like cache keys, parsing, and LRU eviction are ideal for covering both positive and negative cases here.

**Integration tests**: treat the lib as an external user, call its public API, and verify that cross-module assembly is correct. For cases that touch real APIs, prefer **mock implementations** — they're fast, cost nothing, and let you assert results deterministically. For example, a behavior like "sensitive output is actually blocked" should be verified with explicit assertions, not just "runs without panicking."

### Key Points

- Cover pure logic with unit tests first — fast and easy to localize; cover positive and negative cases, don't only test the happy path.
- Cover external composition behavior with integration tests; integration tests need explicit assertions, not just "no panic."
- When writing doc examples, make them runnable as doctests — examples are tests, and the APIs shown in examples must really exist.

---

## A2A Agent Protocol ✨ v0.4.1

### Concept: What Is A2A

[A2A](https://github.com/google/A2A) (Agent-to-Agent) is Google's inter-agent interoperability protocol, solving "how agents developed by different teams and vendors call each other." LangChainRust provides full A2A support: Server to expose agents, Client to call remote agents, using JSON-RPC 2.0 style messaging.

When to use:

- To open your own Agent to external (cross-organization / cross-service) calls
- To orchestrate remote Agents rather than local function calls
- When you need a standardized "discover → dispatch → check progress → cancel" protocol instead of rolling your own

Layering: **A2A handles "Agent ↔ Agent communication"; MCP handles "Agent ↔ tools / data sources."** The two often work together — MCP supplies tools, Agents collaborate over A2A, and responsibilities stay clearer.

### Protocol Flow

A typical A2A call flow (4 steps):

| Step | Operation | Role |
|---|---|---|
| 1. Discover | `get_agent_card` fetches the remote agent's Agent Card | Client |
| 2. Dispatch | `send_task` submits a task (`A2AMessage`) | Client → Server |
| 3. Check progress | `get_task` queries status and result by task ID | Client |
| 4. Cancel | `cancel_task` cancels an unfinished task | Client |

On the Server side, there are two corresponding endpoints:

- `GET /.well-known/agent-card.json` → returns the Agent Card (the agent's self-description, for discovery)
- `POST /` → receives and processes JSON-RPC requests (`handle_a2a_request`)

<a id="a2a-agent-protocol"></a>
### A2AServer (Expose Your Agent)

`A2AServer` provides handler functions that you plug into any HTTP framework (axum, actix, warp) — it does NOT start its own HTTP listener.

```rust
use langchainrust::a2a::{A2AServer, AgentCard};
use langchainrust::LLMChain;
use std::sync::Arc;

let chain = Arc::new(LLMChain::new(llm, "You are a helpful assistant"));
let server = A2AServer::new(chain)
    .with_card(AgentCard::new("my-agent", "A helpful agent", "http://localhost:8080"));

// In your HTTP handler:
// GET  /.well-known/agent-card.json → server.get_agent_card()
// POST /                       → server.handle_a2a_request(body).await
```

**Task Persistence**: Tasks from `tasks/send` are stored in an in-memory `RwLock<HashMap>`. `tasks/get` retrieves them, `tasks/cancel` transitions their status. For production, wrap with your own database-backed store.

### A2AClient (Call Remote Agent)

```rust
use langchainrust::a2a::{A2AClient, A2AMessage};

let client = A2AClient::new("http://remote-agent:8080".to_string()).unwrap();

// Discover agent
let card = client.get_agent_card().await?;

// Send task
let task = client.send_task(A2AMessage::user("hello")).await?;

// Get task
let task = client.get_task(&task.id).await?;

// Cancel task
let task = client.cancel_task(&task.id).await?;
```

**Current boundary**: `tasks/send` / `tasks/get` / `tasks/cancel` and `AgentCard` are implemented; task state-machine extensions, auth (token), and streaming push are planned (⏳). A deployment example lives in the repo at `crates/lc/examples/a2a_http_server.rs` (axum HTTP wrapper).

### Key Behaviors and Boundaries

| Capability | Status |
|---|---|
| `tasks/send` | Implemented (tasks stored in in-memory `RwLock<HashMap>`) |
| `tasks/get` | Implemented (retrieves by task ID) |
| `tasks/cancel` | Implemented (transitions task status) |
| Agent Card discovery | Implemented |
| Task state-machine extensions (submitted → working → terminal) | ⏳ Planned |
| Auth (token) | ⏳ Planned |
| Streaming push | ⏳ Planned |
| TLS / rate limiting | ⏳ Planned; add in the HTTP layer before production deployment |

### Production Notes

- Task storage is in-memory and lost on process restart; for production, switch to a database-backed store (transactional, recoverable).
- The spec requires auth; add token validation in your own HTTP layer so arbitrary task IDs can't be queried / canceled.
- A deployment example lives in the repo at `crates/lc/examples/a2a_http_server.rs` (axum HTTP wrapper).

### How to Choose

- Use A2A for cross-organization / cross-service interoperability over a standard protocol.
- For just coordinating multiple Agents within one process, prefer local orchestration (hand-off, dispatch and collect results) — don't introduce a network protocol.
- When Agents need to do work (call tools / data sources), combine with MCP layering — one manages tools, the other manages Agent-to-Agent communication.

---

<a id="with_structured_output"></a>
## with_structured_output ✨ v0.4.1

### Concept: Give It a Schema, Get a Strongly-Typed Object in One Step

`StructuredOutputExt` trait lets you get strongly-typed output from an LLM in one call — it uses function calling when available and falls back to `JsonOutputParser`. The traditional approach — "prompt the model for JSON → parse it by hand → tolerate errors → convert to a type" — is tedious and fragile: model output is often wrapped in a JSON code block, padded with extra prose, and occasionally not even standard. `with_structured_output` makes the framework handle the whole flow for you: **give it a schema, get a strongly-typed object in one step.**

When to use:

- To parse model output into a struct your program can use directly
- To skip hand-writing the "prompt JSON → parse → tolerate → convert" glue
- When you need strong typing for output fields and want type safety at compile time

### How It Works

The `StructuredOutputExt` trait provides `with_structured_output`, built on a "two-level priority":

1. **Function calling first**: when the model supports function calling, the framework binds the schema to the model as a tool declaration; the model returns tool_calls, and the framework parses the target type directly from the structured arguments — done in one step, independent of text formatting.
2. **Fall back to JsonOutputParser**: when the model doesn't support function calling, or returns no tool calls, it automatically falls back to `JsonOutputParser` — parsing the model's textual JSON (auto-stripping markdown code blocks, tolerantly repairing it) into the target type.

One call, two paths switched automatically, fully transparent to the caller.

### Streaming Version: stream_structured_output

`with_structured_output` gives you the "complete result"; `stream_structured_output` is the streaming version — JSON is parsed as it's generated, built on the `PartialJsonParser` incremental parser, so **a field is usable as soon as it comes out**. It suits frontend render-as-you-receive scenarios, e.g. "title comes first, author next, year last."

### Code Example

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

### Key Behaviors

| Behavior | Description |
|---|---|
| Schema definition | Declare the output structure with a Rust struct + `JsonSchema` / `Deserialize` derives |
| Function calling first | Uses native tool_calls when supported, parses structured arguments directly |
| Automatic fallback | Falls back to `JsonOutputParser` when function calling isn't supported / no tool_calls |
| Type safety | Returns a compile-time determined strong type, no manual parsing |
| Streaming version | `stream_structured_output` + `PartialJsonParser` parse as they generate |

### How to Choose

- "Get the complete result at once, with a determined type" → use `with_structured_output`.
- "Use it as it's generated, render field by field" → use `stream_structured_output`.
- Just want to parse output the model already produced (not make it return per a schema) → use a parser like `JsonOutputParser` or `TypedOutputParser` directly.

---

<a id="filevectorstore"></a>
## FileVectorStore ✨ v0.4.1

JSON-persisted vector store. Bridges the gap between InMemory (not persistent) and external databases (too heavy).

It serializes **vectors + documents** to JSON on a disk file: data survives process exit, and after a restart you can load the previous data back with the same `new(path, dim)`; no database to install and no network dependency. It suits demos, offline use, and small-scale local knowledge bases — the "need persistence but don't want to run a service" cases.

Workflow: `FileVectorStore::new(path, 4).await` specifies the on-disk path and vector dimension (**creation is async, needs `.await`**) → `add_documents(docs, embeddings)` stores a batch of "documents + vectors" → `similarity_search(&query, k)` returns the top-k for a query vector. Write operations are persisted automatically, no manual saving needed.

### Create and Load

| Parameter | Meaning |
|---|---|
| `path` | JSON file path (e.g. `./vectors.json`) |
| `dim` | Vector dimension (e.g. 4), fixed when the store is created |

Loading: call `FileVectorStore::new(path, dim).await` again with the same `path` + `dim` to read back the previously persisted data. The dimension is fixed at creation; later `add_documents` calls with mismatched-dimension vectors error out directly, preventing contamination of the same index.

### Usage

```rust
use langchainrust::{FileVectorStore, VectorStore, Document, MockEmbeddings};
use std::path::PathBuf;

let path = PathBuf::from("./vectors.json");
let store = FileVectorStore::new(path, 4).await?;  // 4 dimensions (async creation)

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

// Persistence: file is automatically written; load with new(path, dim) on restart
store.clear().await?;
```

**Features**: Atomic write (tmp+rename), dimension validation, cross-instance persistence.

### Key Behaviors

| Behavior | Description |
|---|---|
| JSON persistence | Vectors + documents are written to a JSON file, survive restarts |
| Atomic write | Writes a temp file first, then renames; power loss/crash won't corrupt the existing file |
| Dimension validation | Dimension fixed at creation; wrong-dimension vectors error out |
| Honest deletion | Deleting a non-existent document returns `DocumentNotFound`, doesn't pretend success |
| Cross-instance persistence | Any instance can read the file at the same path (shared disk / demo scenarios) |
| Pure storage | Vectors are generated by the caller and passed in; `similarity_search` also directly takes a query vector |

### Use Cases

- Demo / prototype: don't want to install a database service just for a demo.
- Offline small-scale knowledge base: small local data, a JSON file is enough.
- No external service: no network dependency, works out of the box.
- Small data but must survive restart: memory loses it, files don't.

### How to Choose

| Your situation | Recommendation |
|---|---|
| Data can be lost, in-process is fine | `InMemoryVectorStore` |
| Need persistence, small data, no service | `FileVectorStore` |
| Long documents need chunked retrieval | `ChunkedVectorStore` + document store |
| Large data / high concurrency / production | Professional vector database (e.g. Chroma / Qdrant / Pinecone) |

---

<a id="computerusetool"></a>
## ComputerUseTool ✨ v0.4.1

### Concept: Let an Agent Operate the Browser / Desktop

Ordinary tools let an Agent "call APIs, query data"; `ComputerUseTool` lets an Agent **operate the screen like a human** — screenshot the UI, move the mouse and click, type with the keyboard. It aligns with Anthropic's computer use API and suits automation tasks where "there's no ready-made interface, only UI operations." It provides screenshot, mouse click, and keyboard input capabilities.

When to use:

- Web / desktop apps have no usable API; you can only operate through the UI
- Automating GUI flows like "fill form → click button → read result"
- Having the Agent complete data entry, UI inspection, etc., through "look at the screen + operate"

### Capabilities

| Capability | Purpose |
|---|---|
| Screenshot | Agent first "sees" the current screen and knows what the UI looks like |
| Mouse click | Clicks a target position / element to select, submit, etc. |
| Keyboard input | Fills text boxes, presses shortcuts |

### Hook It Up as a Tool

`ComputerUseTool` implements `BaseTool`, so it can go into an Agent's tool list. The Agent calls it on demand in its "think → act → observe → think again" loop: screenshot to observe first → decide where to click → perform the click → screenshot again to confirm the result.

```rust
use langchainrust::ComputerUseTool;
use std::sync::Arc;

// Anthropic API mode (default)
let tool = ComputerUseTool::new();

// Use as BaseTool
let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(tool)];
```

### Usage Notes

- This is a tool with real side effects — it operates a real screen. Verify it first in a controlled environment (test page / isolated desktop) before hooking it up, and mind permission and security boundaries.
- The Agent "sees" the UI through screenshots; screenshot quality directly affects how accurately it selects points and types.
- The default is Anthropic API mode (see the code comment); whether other backends can be swapped in depends on what this tool version supports.

---

## v0.5.0 New Features ✨ v0.5.0

### RouterLLM (Model Routing + Fallback)

`RouterLLM` implements `BaseChatModel`, routing calls across a pool of heterogeneous models and falling back on failure.

**Five routing strategies:**

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `Fallback` | Primary fails → try next | Production fault tolerance |
| `RoundRobin` | Rotate across models | Load balancing, rate-limit avoidance |
| `LeastLatency` | Pick fastest recent model | Latency-sensitive |
| `LowestCost` | Pick cheapest model | Cost optimization |
| `InputDirected` | Custom closure over input text | Route by query complexity |

```rust
use langchainrust::{RouterLLM, RoutingStrategy, BaseChatModel};

// 1. Fallback: primary + backups
let router = RouterLLM::with_fallbacks(gpt4, vec![claude, local_model]);
let result = router.chat(messages, None).await?;

// 2. Lowest cost routing
let router = RouterLLM::new(RoutingStrategy::LowestCost)
    .with_cost(cheap_model, 0.01)
    .with_cost(powerful_model, 0.03);

// 3. Input-directed routing
let router = RouterLLM::new(RoutingStrategy::InputDirected(Arc::new(|input| {
    if input.contains("code") { 1 } else { 0 }
})))
.with_model(general_model)
.with_model(code_model);

// 4. Least latency routing
let router = RouterLLM::new(RoutingStrategy::LeastLatency)
    .with_model(fast_model)
    .with_model(slow_but_smart_model);

// Works as a normal BaseChatModel — drop-in replacement
let result = router.chat(messages, None).await?;
let stream = router.stream_chat(messages, None).await?;
```

---

### CorrectiveRAG

Standard RAG retrieves documents that may be irrelevant, yet the LLM still hallucinates a plausible answer. CorrectiveRAG adds three gates: grade documents -> rewrite query or supplement with web search -> hallucination check.

```rust
use langchainrust::agents::crag::CorrectiveRAGAgent;

let agent = CorrectiveRAGAgent::new(llm, retriever)
    .with_web_fallback(Box::new(web_tool))  // optional: web search fallback
    .with_hallucination_check(true)       // optional: hallucination detection (default: true)
    .with_grade_threshold(0.6)            // optional: relevance threshold (default: 0.6)
    .with_retrieve_k(4)                   // optional: number of docs to retrieve (default: 4)
    .with_grader_llm(grader_llm)          // optional: separate LLM for grading (avoids self-verification bias)
    .with_max_context_tokens(4000);       // optional: truncate low-scoring docs to fit token budget

let answer = agent.invoke("What is Rust ownership?").await?;
```

**Flow:** query -> retrieve -> grade -> [irrelevant? -> rewrite/web search -> re-retrieve] -> generate -> hallucination check -> output

**Builder methods:**

| Method | Default | Description |
|-------|---------|-------------|
| `with_web_fallback(tool)` | None | Web search tool (`Box<dyn BaseTool>`) for supplementing poor retrieval |
| `with_hallucination_check(bool)` | `true` | Enable/disable hallucination detection |
| `with_grade_threshold(f64)` | `0.6` | Average relevance score below this triggers corrective path (clamped to 0.0-1.0) |
| `with_retrieve_k(usize)` | `4` | Number of documents to retrieve |
| `with_grader_llm(llm)` | None | Separate LLM for hallucination checking; avoids self-verification bias where a model tends to endorse its own output |
| `with_max_context_tokens(usize)` | None | Truncate lowest-scoring documents to fit within this token budget |

---

### AdaptiveRAG

LLM decides retrieval strategy per query: NoRetrieval (skip retrieval), SingleSearch (one query), MultiQuery (multiple angles).

```rust
use langchainrust::agents::adaptive_rag::AdaptiveRAG;

let agent = AdaptiveRAG::new(llm, retriever);

// Complex question -> LLM picks MultiQuery, generates multiple queries
let answer = agent.invoke("Compare tokio vs async-std scheduling").await?;

// Simple greeting -> LLM picks NoRetrieval, skips retrieval entirely
let answer = agent.invoke("Hello").await?;
```

---

### GraphRAG (Knowledge Graph RAG)

Vector search misses relationships. GraphRAG extracts entities + relations -> builds graph -> Label Propagation community detection -> community summaries -> query by community.

```rust
use langchainrust::{GraphRAG, GraphQueryMode};

let mut graph_rag = GraphRAG::new(llm);
graph_rag.add_documents(&documents).await?;

// Global query: search community summaries (macro questions)
let result = graph_rag.query("overall tech stack architecture", GraphQueryMode::Global).await?;

// Local query: search entity neighbors (specific questions)
let result = graph_rag.query("Alice's advisor's students", GraphQueryMode::Local).await?;

// Hybrid: combine both
let result = graph_rag.query("...", GraphQueryMode::Hybrid).await?;
```

**Pipeline:** documents -> LLM entity+relation extraction -> graph building -> Label Propagation community detection -> LLM community summaries -> query (Global/Local/Hybrid). No external graph library dependency.

---

### Deep Research Agent

Multi-round deep research: decompose topic into sub-topics -> parallel search across multiple tools -> deduplicate -> synthesize -> discover gaps -> re-search -> cited report.

```rust
use langchainrust::agents::deep_research::DeepResearchAgent;

let agent = DeepResearchAgent::new(llm)
    .with_searcher(Box::new(DuckDuckGoSearchTool::new()))  // add search tools (at least one required)
    .with_max_rounds(3)           // max research rounds (default: 2)
    .with_max_subtopics(5)        // max sub-topics to decompose (default: 5)
    .with_max_source_tokens(8000);// optional: truncate source snippets to fit token budget

let report = agent.research("Compare Rust async runtimes: tokio vs async-std vs smol").await?;
println!("{}", report.markdown);           // full markdown report with inline citations
println!("Rounds: {}", report.rounds_completed);
for citation in &report.citations {
    println!("[{}] {} - {}", citation.index, citation.source, citation.snippet);
}
```

**Builder methods:**

| Method | Default | Description |
|-------|---------|-------------|
| `with_searcher(tool)` | None (required) | Add a search tool; multiple tools are queried in parallel |
| `with_max_rounds(n)` | `2` | Maximum search-synthesize iterations |
| `with_max_subtopics(n)` | `5` | Maximum sub-topics for decomposition |
| `with_max_source_tokens(n)` | None | Truncate source snippets to fit within this token budget |

**ResearchReport fields:** `markdown` (full report with inline `[1]` citations), `citations` (ordered list with `index`/`source`/`url`/`snippet`), `subtopics` (investigated sub-topics), `rounds_completed`.

---

### MCP Protocol Primitives

The MCP spec defines 6 categories of primitives. In LangChainRust the primitives **with implemented call logic** are `initialize` (handshake), `tools/list`, `tools/call`, plus streaming tool results (`notifications/tool_partial`) and cancellation (`notifications/cancelled`). The remaining primitives have their message types defined (serializable/deserializable, usable for type design), but their **call logic is not yet implemented** — direct calls return `method_not_found`:

| Primitive | Status | Description |
|-----------|--------|-------------|
| **Resources** | ⏳ types defined | browse/read server resources |
| **Prompts** | ⏳ types defined | fetch predefined prompt templates |
| **Completion** | ⏳ types defined | parameter auto-completion suggestions |
| **Elicitation** | ⏳ types defined | interactive prompts to the user |
| **Roots** | ⏳ types defined | discover client root directories |
| **Sampling** | ✅ sampling guard | server-side `SamplingGuard` protecting `sampling/createMessage` |

> Server-side sampling has its own `SamplingGuard` (depth / token budget / timeout triple protection), see the [MCP](#mcp) section. If you need Resources/Prompts primitives you can extend on top of the type layer.

---

### Code Interpreter Sandbox

Safe code execution with `LocalSandbox` (subprocess + timeout).

```rust
use langchainrust::tools::sandbox::{LocalSandbox, CodeSandbox, SandboxTool, Language};

// Direct sandbox usage
let sandbox = LocalSandbox::new()
    .with_python_path("python3");  // optional: custom interpreter path

let result = sandbox.run("print(2 + 2)", Language::Python, 30_000).await?;
assert_eq!(result.stdout.trim(), "4");

// Or wrap as a BaseTool for agent use
let tool = SandboxTool::new(LocalSandbox::new(), Language::Python)
    .with_timeout(30_000);  // 30 second timeout
```

- **LocalSandbox**: subprocess execution, auto-kill on timeout, captures stdout/stderr, dangerous import check for Python (the only built-in backend)

---

### OpenAI Responses API

Connect to `/v1/responses` with built-in tools: WebSearch, FileSearch, CodeInterpreter, ComputerUse -- one request, model handles tool calls automatically.

```rust
use langchainrust::language_models::openai::responses::{ResponsesModel, ResponsesConfig, BuiltinTool};

let config = ResponsesConfig::new("your-api-key")
    .with_model("gpt-4o")
    .with_builtin_tool(BuiltinTool::WebSearch)
    .with_builtin_tool(BuiltinTool::CodeInterpreter);

let model = ResponsesModel::new(config);

let result = model.chat(messages, None).await?;
// result.content includes the final answer after tool execution
```

---

### Anthropic Extended Thinking

Configure `budget_tokens` to let Claude think before answering. Thinking block exposed via `thinking_content` in `LLMResult`; streaming via `on_llm_thinking` callback.

```rust
use langchainrust::{AnthropicChat, AnthropicConfig};

let config = AnthropicConfig::new("your-api-key")
    .with_model("claude-sonnet-5");
let model = AnthropicChat::new(config)
    .with_thinking(10000); // up to 10000 thinking tokens

let result = model.chat(messages, None).await?;
println!("Thinking: {:?}", result.thinking_content);
println!("Answer: {}", result.content);
```

---

### Streaming Structured Output

`PartialJsonParser` incrementally parses streaming JSON into partial structs -- no need to wait for all tokens.

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
        println!("Got name: {}", name); // available before all fields arrive
    }
}
```

---

### Batch API

`BatchClient` unifies OpenAI and Anthropic batch workflows: submit → poll → results, at 50% cost.

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

### Tracing (Distributed Tracing)

`Tracer` + `SpanGuard` (RAII) auto-manages parent-child spans. Backends: InMemory / Console / OTel.

```rust
use langchainrust::callbacks::tracing::{Tracer, ConsoleTracingBackend, SpanKind};
use std::sync::Arc;

let tracer = Tracer::new(Arc::new(ConsoleTracingBackend));
let span = tracer.start("agent_run", SpanKind::Internal);
{
    let _retrieve = tracer.start_child("retrieve", SpanKind::Internal);
    let docs = retriever.retrieve(&query).await?;
} // _retrieve drop -> child span auto-records end time
{
    let _generate = tracer.start_child("generate", SpanKind::Internal);
    let answer = llm.chat(messages, None).await?;
}
span.end(); // span auto-records duration, token count, etc.
```

---

<a id="v050-new-features"></a>
### v0.5.0 Quality Hardening (176 Fixes)

After implementing 12 new features, a two-pass full-codebase review of 223 files found and fixed 176 issues (23 CRITICAL / 63 HIGH / 75 MEDIUM / 15 LOW).

**Key fixes:**

- **Security**: PythonREPL dangerous import check, HTTPTool/URLFetchTool SSRF protection (private IP + DNS rebinding), SQLTool injection prevention, Gemini API key moved to header
- **Multi-turn Function Calling**: Anthropic/Gemini/Ollama tool message mapping errors causing multi-turn FC to break — all corrected
- **Streaming**: Ollama/Anthropic/Gemini SSE cross-chunk token loss fixed; `Runnable::stream()` changed from fake streaming to real streaming (per-token emission)
- **Concurrency**: `std::sync::Mutex` in async contexts replaced with `tokio::sync::Mutex`; MCP Transport request-level mutex; HandoffManager lock merging
- **Panic fixes**: `choices[0]` out-of-bounds → `.first().ok_or()`; `from_env()` returns `Result`; Regex → LazyLock; Mutex poison → `into_inner()` recovery
- **Data correctness**: UTF-8 char-boundary slicing; RRF document ID uses content hash; error propagation replaces silent swallowing

**Verification:** 826 unit tests passing · clippy zero warnings · cargo fmt clean

---

## v0.5.2 Fixes ✨ v0.5.2

v0.5.2 is a stability and correctness release with critical bug fixes for several v0.5.0 features.

### GraphRAG Community Summary Fix

Community summaries were concatenating raw entity IDs (`e_xxx`) instead of entity names, producing meaningless summaries that degraded Global/Hybrid query quality. Fixed by looking up entity names via `store.get_entity()`.

### Deep Research Report Format Fix

The synthesizer asked the LLM to output a full markdown report as a JSON string field, causing frequent `serde_json` parse failures due to unescaped `\n`, `"`, `\` in markdown. Replaced with a delimiter-based format:

```
<<<REPORT>>>
...markdown report...
<<<END_REPORT>>>
<<<GAPS>>>
[...gap descriptions...]
<<<END_GAPS>>>
```

The report portion is now raw text with no escaping needed. The old JSON format is kept as a fallback for backward compatibility.

### DocumentStore Async Panic Fix

`InMemoryDocumentStore` and `InMemoryChunkedDocumentStore` used `tokio::sync::RwLock` with `blocking_read()`/`blocking_write()`, which panics inside async contexts with "Cannot block the current thread from within a runtime". Switched to `std::sync::RwLock` which works in both sync and async contexts.

### CRAG Grading Improvements

**Threshold fix**: Default `grade_threshold` changed from `0.5` to `0.6`. The old threshold sat in the zone where LLM grading is least stable, and the ambiguous parse default (`0.5`) was exactly equal to the threshold — making correction triggering nearly random. Now the ambiguous default is `0.4`, well below the `0.6` threshold.

**Hallucination detection bias fix**: Added `with_grader_llm()` builder to inject a separate LLM for hallucination detection, preventing the model from endorsing its own output:

```rust
use langchainrust::agents::crag::CorrectiveRAGAgent;

let agent = CorrectiveRAGAgent::new(llm.clone(), retriever)
    .with_grader_llm(claude_llm)  // Use a different LLM for grading
    .with_grade_threshold(0.6);    // New default: 0.6 (was 0.5)
```

Additional improvements:
- `GradeResult` now has an `is_ambiguous` field indicating whether the score came from fuzzy parsing
- Hallucination detection prompt now includes adversarial framing ("Be skeptical")
- Hallucination check LLM failure degrades gracefully (returns `grounded: false`) instead of aborting

<a id="v052-fixes"></a>
### Other v0.5.2 Changes

- **Feature gate declarations**: `sandbox-e2b` and `sandbox-wasm` features were referenced in code but not declared in `Cargo.toml` `[features]` — now properly declared
- **Clippy zero warnings**: All clippy warnings resolved

---

## More Resources

| Resource | Content |
|----------|---------|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guide |
| [API Docs](https://docs.rs/langchainrust) | Rust API reference |