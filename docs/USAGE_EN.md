# Usage Guide

This document provides detailed usage instructions. For a quick overview, see [README.md](../README.md).

---

## Table of Contents

- [LLM](#llm)
- [Prompts](#prompts)
- [Memory](#memory)
- [Chains](#chains)
- [Agents](#agents)
- [Tools](#tools)
- [RAG](#rag)
- [BM25](#bm25)
- [Hybrid Retrieval](#hybrid-retrieval)
- [Document Loaders](#document-loaders)
- [MultiQueryRetriever](#multiqueryretriever)
- [HyDE Retriever](#hyde-retriever)
- [Reranking](#reranking)
- [LangGraph](#langgraph)
- [MongoDB Storage](#mongodb-storage)

---

## LLM

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

---

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

---

## Tools

### Built-in Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| Calculator | Math operations | `expression` |
| DateTimeTool | Date/time queries | `operation`, `datetime` |
| SimpleMathTool | Power, sqrt, trig | `operation`, `value` |
| URLFetchTool | Fetch URLs | `url` |

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

---

## RAG

### Document Splitting

```rust
use langchainrust::{RecursiveCharacterSplitter, TextSplitter};

let splitter = RecursiveCharacterSplitter::new(200, 50);

let chunks = splitter.split_document(&Document::new(
    "Long text to split..."
))?;
```

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

### Human-in-the-loop

```rust
let compiled = graph.compile()
    .with_interrupt_before(vec!["output"]);

let execution = compiled.invoke_with_execution(state).await?;

if execution.is_interrupted() {
    // Review state
    println!("Paused at: {}", execution.current_node);
    
    // Resume after approval
    let result = compiled.resume(execution).await?;
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

## MongoDB Storage

### Enable Feature

```toml
[dependencies]
langchainrust = { version = "0.2.6", features = ["mongodb-persistence"] }
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

## Testing

```bash
cargo test
```

---

## More Resources

| Resource | Content |
|----------|---------|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guide |
| [API Docs](https://docs.rs/langchainrust) | Rust API reference |