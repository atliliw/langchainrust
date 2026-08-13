# Memory

LangChainRust provides conversation memory management with multiple strategies: buffer, window, summary, summary-buffer, persistent, and context window.

## Memory Types

| Type | Struct | Description | Use Case |
|------|--------|-------------|----------|
| Buffer | `ConversationBufferMemory` | Stores all messages | Short conversations |
| Window | `ConversationBufferWindowMemory` | Keeps last k rounds | Bounded history |
| Summary | `ConversationSummaryMemory` | LLM-compressed summary | Long conversations |
| SummaryBuffer | `ConversationSummaryBufferMemory` | Summary + recent messages | Balanced approach |
| Persistent | `PersistentMemory` trait | Load/save to storage | Cross-session |
| ContextWindow | `ContextWindow` | Token-budget fitting | LLM context limits |
| VectorStore | `VectorStoreRetrieverMemory` | Semantic retrieval | Large history |

## Buffer & Window Memory

```rust
use langchainrust::{ConversationBufferMemory, ConversationBufferWindowMemory};
use std::collections::HashMap;

// Buffer -- stores everything
let mut memory = ConversationBufferMemory::new()
    .with_input_key("input".to_string())
    .with_output_key("output".to_string());

memory.save_context(
    &HashMap::from([("input".into(), "What is Rust?".into())]),
    &HashMap::from([("output".into(), "A systems language.".into())]),
).await?;

let loaded = memory.load_memory_variables(&HashMap::new()).await?;

// Window -- keeps last k rounds
let mut memory = ConversationBufferWindowMemory::new(5); // last 5 rounds
```

## Summary & SummaryBuffer Memory

```rust
use langchainrust::{ConversationSummaryMemory, ConversationSummaryBufferMemory, OpenAIChat};

// Summary -- LLM compresses history
let mut memory = ConversationSummaryMemory::new(llm)
    .with_max_recent_turns(2);

// SummaryBuffer -- hybrid with token limit
let mut memory = ConversationSummaryBufferMemory::new(llm, 4000)
    .with_memory_key("history");
```

## ContextWindow

```rust
use langchainrust::{ContextWindow, Strategy};

// Truncate strategy (default) -- drops oldest, preserves system messages
let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096)?;
let fitted = cw.fit(messages).await?;

// Summarize strategy -- LLM compresses older messages
let cw = ContextWindow::with_strategy(4096, Strategy::summarize(llm))?;
let fitted = cw.fit(messages).await?;
```

> **Budget semantics (Truncate)**: System 消息恒保留且不占预算;若 System 自身超预算,原样返回(可能超预算)。

## Persistent Memory

```rust
use langchainrust::{PersistentMemory, PersistenceConfig};

let config = PersistenceConfig::new()
    .with_auto_save(true)
    .with_token_limit(4000);

// Implement PersistentMemory for your storage backend
// Methods: load_from_store, save_to_store, delete_session, session_exists
```

## ChatMessageHistory

```rust
use langchainrust::ChatMessageHistory;

let mut history = ChatMessageHistory::new();
history.add_user_message("Hello");
history.add_ai_message("Hi there!");
history.add_system_message("You are helpful.");
let messages = history.messages();
```
