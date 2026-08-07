# Usage Guide

This document provides detailed usage instructions. For a quick overview, see [README.md](../README.md).

---

## Table of Contents

- [LLM](#llm)
  - Multi-Provider Support
  - OpenAI Chat
  - Streaming
  - Function Calling
  - Ollama (Local LLM)
  - Google Gemini
  - Multimodal Vision
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
  - ContextWindow (Long Context Management) ✨ v0.4.1
- [LLM Cache](#llm-cache)
- [Chains](#chains)
  - ConversationRetrievalChain
  - Chain Streaming ✨ v0.4.1
- [LCEL (LangChain Expression Language)](#lcel-langchain-expression-language-) ✨ v0.9.0
  - RunnableWithFallbacks ✨ v0.10.0
  - RunnableAssign ✨ v0.10.0
  - RunnableRetry ✨ v0.11.0
  - CancellationToken ✨ v0.11.0
- [Document Chains](#document-chains)
- [Agents](#agents)
  - Agent Hooks ✨ v0.11.0
  - Agent Streaming ✨ v0.12.0
- [Plan-Execute Agent](#plan-execute-agent)
- [Handoffs](#handoffs)
- [Streaming Tool Calls](#streaming-tool-calls)
- [Guardrails](#guardrails)
- [Token Counter](#token-counter)
- [Sessions](#sessions)
- [MCP](#mcp)
  - MCPServer
- [Tools](#tools)
  - WikipediaTool
  - DuckDuckGoSearchTool
  - PythonREPLTool
  - Extended Tools (HTTPTool / FileTool / SQLTool)
  - `#[tool]` Procedural Macro ✨ v0.10.0
- [RAG](#rag)
  - ChromaDB
  - PGVectorStore
  - PineconeStore
  - SemanticSplitter
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
- [LangGraph](#langgraph)
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
  - MCP Full Protocol (6 Primitives)
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

## LLM

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

### Ollama (Local LLM)

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

### PromptTemplate

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

```rust
use langchainrust::prompts::{LengthBasedExampleSelector, SemanticExampleSelector};

// Length-based: selects examples up to max length
let selector = LengthBasedExampleSelector::new(examples, example_prompt, 50);

// Semantic: selects most similar examples via embeddings
let selector = SemanticExampleSelector::new(embeddings, examples, 2);
```

---

## Output Parsers

### StrOutputParser

```rust
use langchainrust::output_parsers::{StrOutputParser, BaseOutputParser};

let parser = StrOutputParser::new();
let result = parser.parse("Hello world")?;
```

### CommaSeparatedListOutputParser

```rust
use langchainrust::output_parsers::CommaSeparatedListOutputParser;

let parser = CommaSeparatedListOutputParser::new();
let result = parser.parse("apple, banana, cherry")?;
```

### JsonOutputParser

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

## Memory

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

Keeps only last k turns:

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

Summarizes old messages, keeps recent ones:

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

### ContextWindow (Long Context Management) ✨ v0.4.1

`ContextWindow` manages token budget for long conversations with two strategies: Truncate and Summarize.

```rust
use langchainrust::{ContextWindow, Message, OpenAIChat, Strategy};
use langchainrust::BaseChatModel;

// Strategy 1: Truncate — discard oldest messages when over token budget
let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096);

// Strategy 2: Summarize — use LLM to compress old conversation when over budget
let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096)
    .with_strategy(Strategy::Summarize)
    .with_llm(OpenAIChat::new(config));

cw.add_message(Message::human("hello")).await;
cw.add_message(Message::ai("Hi! How can I help?")).await;

let messages = cw.get_messages().await;
```

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `Truncate` | Discard oldest messages over budget | Simple scenarios |
| `Summarize` | LLM compresses old conversation into summary | Long conversations needing key info |

## LLM Cache

### In-Memory Cache with TTL

```rust
use langchainrust::cache::{LLMCache, CacheConfig};
use std::time::Duration;

let config = CacheConfig::new()
    .with_ttl(Duration::from_secs(3600))  // 1 hour
    .with_max_size(1000);                 // 1000 entries

let cache = LLMCache::new(config);

// Use with LLM
let llm = OpenAIChat::new(config)
    .with_cache(cache);

// Subsequent identical calls return cached result
let r1 = llm.chat(vec![Message::human("Hello")], None).await?;
let r2 = llm.chat(vec![Message::human("Hello")], None).await?;  // cache hit
```

---

## LCEL (LangChain Expression Language) ✨ v0.9.0

LCEL provides Python LangChain-style pipe composition, chaining `Runnable` components via `pipe()` into pipelines.

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
    RunnableExt, RunnableLambda, RunnableParallel, RunnableAssign,
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

---

## Chains

### LLMChain

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

```rust
use langchainrust::{RetrievalQA, SimilarityRetriever};

let retriever = SimilarityRetriever::new(store, embeddings);
let qa = RetrievalQA::new(llm, retriever, 3);

let answer = qa.invoke(HashMap::from([
    ("query", "What is BM25?"),
])).await?;
```

### ConversationRetrievalChain

Retrieval-augmented conversation with memory:

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

### StuffDocumentsChain

Combine all documents with a prompt:

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

Iteratively refine by processing one document at a time:

```rust
use langchainrust::chains::RefineDocumentsChain;

let initial_llm = Arc::new(LLMChain::new(llm.clone(), "Summarize: {text}"));
let refine_llm = Arc::new(LLMChain::new(llm, "Refine summary with: {text}"));

let chain = RefineDocumentsChain::new(initial_llm, refine_llm);
let result = chain.invoke(documents).await?;
```

### MapReduceDocumentsChain

Map each document then reduce:

```rust
use langchainrust::chains::MapReduceDocumentsChain;

let map_chain = Arc::new(LLMChain::new(llm.clone(), "Summarize: {text}"));
let reduce_chain = Arc::new(LLMChain::new(llm, "Combine: {summaries}"));

let chain = MapReduceDocumentsChain::new(map_chain, reduce_chain);
let result = chain.invoke(documents).await?;
```

### MapRerankDocumentsChain

Map and rerank by score:

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

---

## Agents

### FunctionCallingAgent (Recommended)

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

## Plan-Execute Agent

The Plan-Execute Agent first plans task steps with an LLM, executes them step by step, re-plans on failure, and finally summarizes. Suited for complex, multi-step tasks.

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

Flow: plan -> execute each step (FunctionCallingAgent + tools) -> re-plan on failure -> summarize.

---

## Handoffs

Inspired by the OpenAI Agents SDK: a primary agent can delegate tasks to registered specialist agents via `HandoffTool`.

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

`handoff_tools()` returns tools named `handoff_to_{agent}`; you can also hand off directly with `execute_handoff(Handoff)`.

---

## Streaming Tool Calls

`StreamingFunctionCallingAgent` streams LLM text token by token and exposes tool-call state through the event stream.

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

Events: `AgentStreamEvent` (`Text` / `ToolCall` / `FinalAnswer`) and `ToolCallState`.

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

---

## Token Counter

`TiktokenCounter` counts with cl100k_base (GPT-3.5/4/4o); `TokenTrackingLLM` wraps an LLM to accumulate usage; `ModelPricing` estimates cost.

```rust
use langchainrust::{TokenTrackingLLM, ModelPricing, OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

let tracked = TokenTrackingLLM::for_openai(OpenAIChat::new(OpenAIConfig::default()))?;

let result = tracked.chat(vec![Message::human("hi")], None).await?;

let usage = tracked.get_usage();                               // prompt / completion / total tokens
let cost = tracked.estimate_cost(&ModelPricing::gpt4o_mini()); // USD
```

`ModelPricing::gpt4o()` / `gpt4o_mini()` are built-in; use `ModelPricing::new(prompt_per_1k, completion_per_1k)` for custom pricing.

---

## Sessions

`SessionManager` manages the lifecycle of multi-turn conversation sessions: create/get/archive, auto-maintain history on each chat, with pluggable storage (`SessionStore` trait).

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

The `SessionStore` trait has `create/get/update/delete/list_by_user`; implement your own backend (Redis/DB). `MemorySessionStore` is built-in for tests and single-process use.

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

## Tools

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
use langchainrust::{tool, BaseTool, Tool, ToolError};

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

### WikipediaTool

```rust
use langchainrust::WikipediaTool;

let tool = WikipediaTool::new();
let result = tool.run(r#"{"query": "Rust programming"}"#).await?;
```

### DuckDuckGoSearchTool

```rust
use langchainrust::DuckDuckGoSearchTool;

let tool = DuckDuckGoSearchTool::new();
let result = tool.run(r#"{"query": "langchain rust"}"#).await?;
```

### PythonREPLTool

```rust
use langchainrust::PythonREPLTool;

let tool = PythonREPLTool::new();
let result = tool.run(r#"{"code": "print(sum(range(10)))"}"#).await?;
```

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
```

> `SQLTool` is available under the `sqlite-storage` feature; `HTTPTool` / `FileTool` are available by default.

---

## Embeddings

**Embeddings** convert text to vectors for semantic retrieval and similarity calculation.

### Supported Embeddings

| Provider | Class | Dimension | Features |
|----------|-------|-----------|----------|
| **OpenAI** | `OpenAIEmbeddings` | 1536 | High quality |
| **DeepSeek** | `DeepSeekEmbeddings` | 1536 | Cost-effective |
| **Qwen** | `QwenEmbeddings` | 1536 | Chinese optimized |
| **Mock** | `MockEmbeddings` | Custom | Testing |

### OpenAI Embeddings

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

```rust
use langchainrust::{DeepSeekEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(DeepSeekEmbeddings::from_env());

let vector = embeddings.embed("Deep learning fundamentals").await?;
```

### Qwen Embeddings

```rust
use langchainrust::{QwenEmbeddings, Embeddings};
use std::sync::Arc;

let embeddings = Arc::new(QwenEmbeddings::from_env());

let vector = embeddings.embed("Qwen vector generation").await?;
```

### Mock Embeddings (Testing)

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

## RAG

### Document Splitting

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

Persistent vector store using Chroma:

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

PostgreSQL + pgvector extension vector store. Requires the `pgvector-storage` feature; since `sqlx` / `pgvector` deps are not enabled inside the crate, add `sqlx` and `pgvector` to your `Cargo.toml` yourself.

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

Pinecone vector store (reqwest HTTP API, no feature required, available by default).

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

---

## BM25

### BM25Retriever (Keyword Search)

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

### BM25 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| k1 | 1.5 | Term frequency saturation |
| b | 0.75 | Document length normalization |

```rust
let retriever = BM25Retriever::with_params(2.0, 0.5);
```

### ChunkedBM25Retriever (Parent-Child)

AutoMerging: When multiple leaf chunks match, merge to parent:

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

### RRF Fusion Algorithm

```
RRF_score(d) = Σ 1/(k + rank(d))
```

Where k=60, rank(d) is document rank in each result list.

### UnifiedHybridIndex

One interface for BM25 + Vector dual retrieval:

```rust
use langchainrust::{UnifiedHybridIndex, HybridIndexConfig, OpenAIEmbeddings};

let config = HybridIndexConfig::new()
    .with_chunk_size(500)
    .with_top_k(10, 10)        // BM25_k, Vector_k
    .with_rrf_k(60);

let embeddings = Arc::new(OpenAIEmbeddings::new(api_key));
let index = UnifiedHybridIndex::with_config(embeddings, 1536, config);

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

### StateGraph

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

### Conditional Edge

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

### Human-in-the-loop / Interrupt & Resume

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

---

## Document Loaders

Load documents from various file formats.

### Supported Formats

| Loader | Format | Features |
|--------|--------|----------|
| **TextLoader** | .txt | Line-by-line splitting |
| **JSONLoader** | .json | Specify content_key |
| **MarkdownLoader** | .md | Split by heading level |
| **PDFLoader** | .pdf | Extract PDF text |
| **CSVLoader** | .csv | Each row as document |

### TextLoader

```rust
use langchainrust::{TextLoader, DocumentLoader};

let loader = TextLoader::new("document.txt");
let docs = loader.load().await?;

// Split by lines
let loader = TextLoader::new_with_line_split("document.txt");
let docs = loader.load().await?;
```

### JSONLoader

```rust
use langchainrust::{JSONLoader, DocumentLoader};

let loader = JSONLoader::new("data.json");
let docs = loader.load().await?;

// Specify content field
let loader = JSONLoader::new_with_content_key("data.json", "content");
let docs = loader.load().await?;
```

### MarkdownLoader

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

Generate multiple query variations using LLM to improve retrieval recall.

### How It Works

```
User query → LLM generates N variations → Retrieve each → Merge & dedupe → Return results
```

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

### StaticQueryGenerator (No LLM)

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

## HyDE Retriever

**HyDE (Hypothetical Document Embedding)** generates a hypothetical document using LLM, then retrieves real documents similar to it.

### How It Works

```
User query → LLM generates hypothetical document → Retrieve using hypothetical doc → Return real docs
```

### Usage

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

## Reranking

Re-score retrieval results to improve precision.

### Supported Rerankers

| Reranker | Description |
|----------|-------------|
| **KeywordReranker** | Keyword matching reranking |
| **BM25Reranker** | BM25 formula reranking |

### KeywordReranker

```rust
use langchainrust::{KeywordReranker, RerankingExecutor};

let reranker = Box::new(KeywordReranker::new());

let executor = RerankingExecutor::new(reranker)
    .with_top_n(5)
    .with_min_score(0.5);

let reranked = executor.rerank("Rust programming", search_results)?;
```

### BM25Reranker

```rust
use langchainrust::{BM25Reranker, RerankingExecutor};

let reranker = Box::new(BM25Reranker::new()
    .with_params(2.0, 0.5));

let executor = RerankingExecutor::new(reranker).with_top_n(5);

let reranked = executor.rerank("Rust programming", results)?;
```

---

## Callbacks

Callback system for tracing, monitoring, and logging LLM application execution.

### CallbackManager

Manage multiple callback handlers:

```rust
use langchainrust::{CallbackManager, StdOutHandler, LangSmithHandler};
use std::sync::Arc;

let manager = CallbackManager::new()
    .add_handler(Arc::new(StdOutHandler::new()))
    .add_handler(Arc::new(LangSmithHandler::from_env()?));
```

### StdOutHandler

Output to stdout (for debugging):

```rust
use langchainrust::StdOutHandler;

let handler = StdOutHandler::new();
```

### FileCallbackHandler

Output to file:

```rust
use langchainrust::{FileCallbackHandler, LogFormat};

// JSON format
let handler = FileCallbackHandler::new("trace.json", LogFormat::Json);

// Text format
let handler = FileCallbackHandler::new("trace.log", LogFormat::Text);
```

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

Quantify LLM output quality: after changing prompts / models / adding RAG, run an eval set and see if scores improved. 10 evaluators in 5 categories:

| Category | Evaluators | Description |
|----------|-----------|-------------|
| Literal | `ExactMatch` / `StringDistance` | exact equal / Levenshtein distance normalized |
| Semantic | `EmbeddingSimilarity` / `LLMAsJudge` / `PairwiseJudge` | cosine / LLM judge / pairwise (swap A/B to remove position bias) |
| Rule | `ContainsKeyword` / `RegexMatch` / `LengthCheck` | keyword / regex / length |
| Classic NLP | `Bleu` | n-gram precision (char-level + smoothing) |
| RAG | `Faithfulness` | split claims, verify each, detect hallucination |

### EvalRunner

Run a set of evaluators over a `Dataset`, produce a `Report` (per-example scores + per-evaluator averages).

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

---

## MongoDB Storage

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

---

## Redis / SQLite Storage

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

### Feature Gating

| Feature Flag | Storage Backend | Dependencies |
|-------------|-----------------|--------------|
| `redis-storage` | Redis | redis crate |
| `sqlite-storage` | SQLite | rusqlite crate |
| `mongodb-persistence` | MongoDB | mongodb crate |

---

## Testing

```bash
cargo test
```

---

## A2A Agent Protocol ✨ v0.4.1

[A2A](https://github.com/google/A2A) (Agent-to-Agent) is Google's protocol for inter-agent communication. LangChainRust provides full A2A support: Server to expose agents, Client to call remote agents, using JSON-RPC 2.0 style messaging.

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
// GET  /.well-known/agent.json → server.get_agent_card()
// POST /                       → server.handle_a2a_request(body).await
```

**Task Persistence**: Tasks from `tasks/send` are stored in an in-memory `RwLock<HashMap>`. `tasks/get` retrieves them, `tasks/cancel` transitions their status. For production, wrap with your own database-backed store.

### A2AClient (Call Remote Agent)

```rust
use langchainrust::a2a::{A2AClient, A2AMessage};

let client = A2AClient::new("http://remote-agent:8080".to_string());

// Discover agent
let card = client.get_agent_card().await?;

// Send task
let task = client.send_task(A2AMessage::user("hello")).await?;

// Get task
let task = client.get_task(&task.id).await?;

// Cancel task
let task = client.cancel_task(&task.id).await?;
```

---

## with_structured_output ✨ v0.4.1

`StructuredOutputExt` trait lets you get strongly-typed output from an LLM in one call. Uses function calling when available, falls back to JsonOutputParser.

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

JSON-persisted vector store. Bridges the gap between InMemory (not persistent) and external databases (too heavy).

```rust
use langchainrust::{FileVectorStore, VectorStore, Document, MockEmbeddings};
use std::path::PathBuf;

let path = PathBuf::from("./vectors.json");
let store = FileVectorStore::new(path, 4)?;  // 4 dimensions

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

---

## ComputerUseTool ✨ v0.4.1

Computer use tool aligned with Anthropic's computer use API. Provides screenshot, mouse click, and keyboard input capabilities.

```rust
use langchainrust::ComputerUseTool;
use std::sync::Arc;

// Anthropic API mode (default)
let tool = ComputerUseTool::new();

// Or Native mode (requires feature computer-use-native)
// let tool = ComputerUseTool::new_native();

// Use as BaseTool
let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(tool)];
```

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
use langchainrust::retrieval::graph_rag::{GraphRAG, GraphQueryMode};

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

### MCP Full Protocol (6 Primitives)

v0.5.0 completes the MCP spec with all 6 primitives, both Client and Server:

| Primitive | Purpose | Typical Use |
|-----------|---------|-------------|
| **Resources** | Browse/read server resources | Claude Desktop reading local files |
| **Prompts** | Get predefined prompt templates | Standardized prompt management |
| **Completion** | Auto-complete suggestions | Parameter auto-completion |
| **Elicitation** | Interactive prompts to user | User confirmation needed |
| **Roots** | Discover client root directories | Server needs to know accessible paths |
| **Sampling** | Server proxies LLM request via client | Server needs LLM capability |

```rust
use langchainrust::mcp::MCPClient;

// Client: browse resources
let resources = client.list_resources().await?;
let content = client.read_resource("file:///data/report.pdf").await?;

// Get prompt templates
let prompts = client.list_prompts().await?;
let prompt = client.get_prompt("code_review", arguments).await?;

// Completion suggestions
let completions = client.complete("file:///src/", "main").await?;
```

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

- **LocalSandbox**: subprocess execution, auto-kill on timeout, captures stdout/stderr, dangerous import check for Python
- **E2B cloud sandbox** (feature gate `sandbox-e2b`): remote micro-VM, full isolation
- **WASM sandbox** (feature gate `sandbox-wasm`): browser-grade sandbox, zero network

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

### Other v0.5.2 Changes

- **Feature gate declarations**: `sandbox-e2b` and `sandbox-wasm` features were referenced in code but not declared in `Cargo.toml` `[features]` — now properly declared
- **Clippy zero warnings**: All clippy warnings resolved

---

## More Resources

| Resource | Content |
|----------|---------|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guide |
| [API Docs](https://docs.rs/langchainrust) | Rust API reference |