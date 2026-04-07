# LangChain Rust Examples

Complete examples demonstrating LangChain Rust features, organized by difficulty level.

## Directory Structure

```
examples/
├── basic/           # Basic examples - getting started
│   ├── hello_llm.rs
│   ├── streaming.rs
│   ├── prompt_template.rs
│   └── tools.rs
├── intermediate/    # Intermediate - Agents, Memory, Chains
│   ├── agent_with_tools.rs
│   ├── memory_conversation.rs
│   └── chain_pipeline.rs
└── advanced/        # Advanced - RAG, full pipelines
    ├── rag_demo.rs
    ├── multi_tool_agent.rs
    └── full_pipeline.rs
```

## Prerequisites

Set environment variables:

```bash
# Bash/Zsh
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional: custom endpoint
```

```powershell
# PowerShell
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

---

## Basic Examples

### 1. hello_llm - Basic LLM Chat
```bash
cargo run --example hello_llm
```
Demonstrates creating an OpenAI client and basic conversation.

**Key concepts:**
- OpenAIChat initialization
- Message types (System, Human, AI)
- Basic chat invocation

### 2. streaming - Streaming Output
```bash
cargo run --example streaming
```
Shows how to receive streaming responses token-by-token.

**Key concepts:**
- Streaming configuration
- Async stream handling
- Real-time output display

### 3. prompt_template - Prompt Templates
```bash
cargo run --example prompt_template
```
Demonstrates PromptTemplate and ChatPromptTemplate usage.

**No API key required.**

**Key concepts:**
- Variable substitution with `{variable}`
- ChatPromptTemplate for multi-message templates
- Template reuse with different variables

### 4. tools - Built-in Tools
```bash
cargo run --example tools
```
Shows how to directly use built-in tools: Calculator, DateTimeTool, SimpleMathTool, URLFetchTool.

**No API key required.**

**Key concepts:**
- Tool initialization
- Direct tool invocation
- JSON input/output

---

## Intermediate Examples

### 1. agent_with_tools - Agent with Tools
```bash
cargo run --example agent_with_tools
```
Demonstrates ReActAgent using tools to answer questions.

**Key concepts:**
- ReActAgent - Reasoning + Acting
- AgentExecutor - Execution and iteration control
- Automatic tool selection

### 2. memory_conversation - Multi-turn Conversations
```bash
cargo run --example memory_conversation
```
Shows how to implement multi-turn conversations with memory.

**Key concepts:**
- ChatMessageHistory
- Conversation history management
- Context persistence

### 3. chain_pipeline - Chain Workflows
```bash
cargo run --example chain_pipeline
```
Demonstrates LLMChain and SequentialChain.

**Key concepts:**
- LLMChain - Single-step chain
- SequentialChain - Multi-step pipeline
- Chain composition

---

## Advanced Examples

### 1. rag_demo - RAG Pipeline
```bash
cargo run --example rag_demo
```
Complete RAG (Retrieval-Augmented Generation) pipeline.

**Pipeline:**
1. Prepare knowledge documents
2. Split documents into chunks
3. Generate embeddings
4. Store in vector database
5. Retrieve relevant documents
6. Generate answer with context

**Key concepts:**
- Document splitting
- Embedding generation
- Vector storage
- Similarity retrieval

### 2. multi_tool_agent - Multi-Tool Agent
```bash
cargo run --example multi_tool_agent
```
Shows Agent automatically selecting and using multiple tools.

**Key concepts:**
- Multi-tool composition
- Automatic tool selection
- Complex task decomposition

### 3. full_pipeline - Complete AI Application
```bash
cargo run --example full_pipeline
```
Complete AI application combining all components.

**Features:**
- LLM integration
- Agent + Tools
- Conversation memory
- RAG knowledge retrieval
- Intelligent Q&A system

---

## Quick Reference

### Run Without API Key
```bash
cargo run --example prompt_template
cargo run --example tools
```

### Run With API Key
```bash
export OPENAI_API_KEY="your-key"
cargo run --example hello_llm
cargo run --example streaming
cargo run --example agent_with_tools
cargo run --example memory_conversation
cargo run --example chain_pipeline
cargo run --example rag_demo
cargo run --example multi_tool_agent
cargo run --example full_pipeline
```

## Difficulty Guide

| Level | Audience | Prerequisites |
|-------|----------|---------------|
| Basic | Beginners | Rust basics, async concepts |
| Intermediate | Intermediate developers | Basic level + Agent/Memory concepts |
| Advanced | Advanced developers | Intermediate level + RAG/vector search |

## Troubleshooting

### Error: "Invalid API Key"
Ensure `OPENAI_API_KEY` is correctly set.

### Error: "Connection refused"
Check network connection and `OPENAI_BASE_URL` setting.

### Slow first run
Initial compilation takes time. Subsequent runs are faster.

## Notes

- **Security**: Never commit API keys to version control
- **Cost**: Some examples call real APIs and may incur charges
- **Network**: Some examples require network access