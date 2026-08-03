# Contributing to langchainrust

Thank you for your interest in contributing to langchainrust! We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code.

## Table of Contents

- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Good First Issues](#good-first-issues)
- [License](#license)

## Reporting Bugs

If you find a bug, please [open an issue](https://github.com/atliliw/langchainrust/issues/new?template=bug_report.md) and include:

1. A clear title and description
2. Steps to reproduce
3. Expected vs. actual behavior
4. Environment info (Rust version, OS, enabled features)
5. A minimal reproducible example if possible

## Requesting Features

Feature requests are welcome! Please [open an issue](https://github.com/atliliw/langchainrust/issues/new?template=feature_request.md) and describe:

1. What you want
2. Why it's useful
3. A proposed API or implementation idea (optional)

## Development Setup

```bash
# Clone the repository
git clone https://github.com/atliliw/langchainrust.git
cd langchainrust

# Build
cargo build

# Run tests
cargo test

# Run clippy
cargo clippy

# Check formatting
cargo fmt --check
```

### Prerequisites

- Rust 1.82+ (see `rust-toolchain.toml`)
- For integration tests: set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variables

## Code Standards

### Formatting

Use `cargo fmt` to auto-format your code. CI will reject PRs that are not formatted.

### Linting

Ensure `cargo clippy` passes with no warnings:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Testing

All new features and bug fixes must include tests. See [Running Tests](#running-tests) for details.

### Documentation

Public API items must have doc comments (`///`):

```rust
/// Creates a new ReActAgent.
///
/// # Arguments
/// * `llm` - The LLM client
/// * `tools` - Available tools
/// * `system_prompt` - Optional custom system prompt
///
/// # Example
/// ```
/// use langchainrust::{ReActAgent, OpenAIChat, OpenAIConfig};
///
/// let llm = OpenAIChat::new(config);
/// let agent = ReActAgent::new(llm, tools, None);
/// ```
pub fn new(llm: OpenAIChat, tools: Vec<Arc<dyn BaseTool>>, system_prompt: Option<String>) -> Self {
    // ...
}
```

### Error Handling

Use `Result<T, E>` with custom error types. Avoid `unwrap()` in library code:

```rust
// Good: explicit error handling
let value = some_option.ok_or(MyError::InvalidInput("missing value".into()))?;

// Bad: panics in production
let value = some_option.unwrap();
```

### Naming Conventions

- **Struct/Enum**: PascalCase (`OpenAIChat`, `AgentError`)
- **Functions/methods**: snake_case (`embed_query`, `add_documents`)
- **Variables**: snake_case (`api_key`, `max_tokens`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_ITERATIONS`)

## Commit Messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation update
- `style`: Code formatting (no logic changes)
- `refactor`: Code refactoring
- `test`: Test additions or updates
- `chore`: Build/tooling changes
- `perf`: Performance improvement

Example:
```
feat: add Qwen LLM support

- Implement QwenChat client
- Add QwenConfig for configuration
- Include streaming support

Closes #123
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit with a descriptive message (`git commit -m 'feat: add amazing feature'`)
5. Push to your branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### PR Checklist

Before submitting, ensure:

- [ ] Code passes `cargo fmt --check`
- [ ] Code passes `cargo clippy` (no warnings)
- [ ] All tests pass with `cargo test`
- [ ] New code includes tests
- [ ] Public API has documentation comments
- [ ] No hardcoded secrets or API keys

## Project Structure

```
src/
├── core/            # Core abstractions (Runnable, Tool, RouterLLM, Batch, StructuredOutput)
├── schema/          # Data structures (Message, Document)
├── language_models/ # LLM implementations (OpenAI, Anthropic, Ollama, Gemini, etc.)
├── tools/           # Built-in tools (Calculator, HTTP, SQL, Sandbox, etc.)
├── agents/          # Agent implementations (ReAct, FunctionCalling, CRAG, DeepResearch, etc.)
├── memory/          # Memory implementations (Buffer, Window, Summary, ContextWindow)
├── chains/          # Chain implementations (LLMChain, Sequential, RetrievalQA, DocumentChains)
├── embeddings/      # Embedding models (OpenAI, DeepSeek, Qwen, Local ONNX)
├── vector_stores/   # Vector stores (InMemory, Qdrant, MongoDB, ChromaDB, Redis, etc.)
├── retrieval/       # Retrieval components (BM25, Hybrid, HyDE, MultiQuery, GraphRAG)
├── langgraph/       # LangGraph workflow engine (StateGraph, Subgraph, Parallel, Checkpointer)
├── mcp/             # MCP protocol (Client + Server, 6 primitives)
├── a2a/             # A2A protocol (Agent-to-Agent)
├── callbacks/       # Callback system (StdOut, LangSmith, OTel, Tracing)
├── evaluation/      # Evaluation framework (10 evaluators)
├── guardrails/      # Safety guardrails (Input/Output)
├── sessions/        # Session management
├── prompts/         # Prompt templates
├── output_parsers/  # Output parsers
└── error.rs         # Unified error types

examples/            # Runnable examples
tests/               # Integration and unit tests
docs/                # Documentation
```

## Running Tests

```bash
# Run all unit tests (no API key needed)
cargo test --lib

# Run specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture

# Run integration tests (requires API key)
export OPENAI_API_KEY="your-key"
cargo test --test integration_llm_chat

# Run tests with all features
cargo test --all-features

# Run ignored tests (requires real API keys)
cargo test -- --ignored
```

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/atliliw/langchainrust/labels/good%20first%20issue) — these are specifically chosen to be approachable for new contributors. Typical examples include:

- Adding a new document loader
- Implementing a new output parser
- Writing additional tests
- Fixing documentation typos

If you're unsure where to start, feel free to ask in the issue comments or open a discussion.

## License

By contributing to langchainrust, you agree that your contributions will be licensed under the MIT or Apache-2.0 license, at the option of the project users.
