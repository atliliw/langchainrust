# Chains

Chains compose operations into pipelines. LangChainRust provides LLMChain, SequentialChain, ConversationChain, RouterChain, RetrievalQA, and document chains.

## Chain Types

| Chain | Struct | Description |
|-------|--------|-------------|
| LLMChain | `LLMChain<M>` | Prompt + LLM (simplest chain) |
| SequentialChain | `SequentialChain` | Execute chains in sequence |
| ConversationChain | `ConversationChain<M>` | LLM with conversation memory |
| RouterChain | `RouterChain` | Keyword-based routing |
| LLMRouterChain | `LLMRouterChain<M>` | LLM-based routing |
| RetrievalQA | `RetrievalQA<M>` | RAG question answering |
| ConversationRetrievalChain | `ConversationRetrievalChain<M>` | RAG with memory |
| StuffDocumentsChain | `StuffDocumentsChain<M>` | All docs in one prompt |
| RefineDocumentsChain | `RefineDocumentsChain<M>` | Iterative refinement |
| MapReduceDocumentsChain | `MapReduceDocumentsChain<M>` | Parallel map + reduce |
| MapRerankDocumentsChain | `MapRerankDocumentsChain<M>` | Parallel map + rank |

## LLMChain

```rust
use langchainrust::{LLMChain, LLMChainBuilder};
use std::collections::HashMap;
use serde_json::Value;

let chain = LLMChain::new(llm, "Explain this topic: {topic}")
    .with_input_key("topic")
    .with_output_key("text");

let mut inputs = HashMap::new();
inputs.insert("topic".to_string(), Value::String("Rust ownership".to_string()));
let result = chain.invoke(inputs).await?;

// Builder pattern
let chain = LLMChainBuilder::new(llm, "Summarize: {text}")
    .input_key("text")
    .output_key("summary")
    .build();
```

## SequentialChain

```rust
use langchainrust::SequentialChain;

let pipeline = SequentialChain::new()
    .add_chain(Arc::new(chain1), vec!["topic"], vec!["features"])
    .add_chain(Arc::new(chain2), vec!["features"], vec!["summary"]);

let result = pipeline.invoke(inputs).await?;
```

## ConversationChain

```rust
use langchainrust::{ConversationChain, ConversationBufferMemory};

let chain = ConversationChain::builder(llm)
    .memory(ConversationBufferMemory::new())
    .system_prompt("You are a helpful assistant.")
    .build();

let reply = chain.predict("What is Rust?").await?;
let reply2 = chain.predict("Tell me more about ownership.").await?; // remembers context
```

## RetrievalQA

```rust
use langchainrust::RetrievalQA;

let qa = RetrievalQA::new(llm, retriever)
    .with_k(4)
    .with_return_source_documents(true);

let answer = qa.query("What is LangChain?").await?;
let (answer, sources) = qa.query_with_sources("What is LangChain?").await?;
```

## Document Chains

```rust
use langchainrust::{StuffDocumentsChain, RefineDocumentsChain, MapReduceDocumentsChain};

// Stuff -- all documents in one prompt
let chain = StuffDocumentsChain::new(llm)
    .with_prompt_template("Summarize: {context}\nQuestion: {input}");
let result = chain.invoke_with_documents(docs, "What is the main idea?").await?;

// Refine -- iterative refinement
let chain = RefineDocumentsChain::new(llm)
    .with_initial_prompt("Answer based on: {context}\nQuestion: {input}")
    .with_refine_prompt("Refine this answer: {existing_answer}\nNew context: {context}");

// MapReduce -- parallel map then reduce
let chain = MapReduceDocumentsChain::new(llm)
    .with_map_prompt("Summarize: {context}")
    .with_reduce_prompt("Combine summaries: {context}");
```
