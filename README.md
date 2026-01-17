# langchainrust

一个受 LangChain 启发的 Rust 框架，用于构建基于大模型（LLM）的应用：提示词模板（Prompt Template）、链式调用（Chains）、工具调用（Tools）、Agent、Memory、检索（Retrieval）等。

详细使用文档：docs/USAGE.md

## 特性

- LLM
  - OpenAI 兼容接口（支持流式与非流式）
  - Qwen（基础实现）
- Prompts
  - `PromptTemplate`：字符串模板 `{var}` 替换
  - `ChatPromptTemplate`：多角色消息模板（system/user/assistant）
- Memory
  - `SimpleMemory`：简易历史记录（可用于注入对话上下文）
- Chains
  - `PromptChain`：使用 `ChatPromptTemplate` + LLM 生成输出
  - `SequentialChain`：串联多个 Chain，并可将 memory 作为 chat_history 注入
- Tools
  - `Tool` trait、`Calculator`、`WeatherTool`
- Agent
  - `ReActAgent` + `AgentExecutor`
  - 支持 memory 注入（chat_history 系统消息）
  - 支持提示词模板（可外部传入 `ChatPromptTemplate`）
  - 支持运行时变量注入（一次调用传入多组模板变量）
- Retrieval（实验性）
  - 文档切分、向量存储、检索器等基础组件

## 安装

```toml
cargo add langchainrust
```

## 快速开始

## 配置与安全

- 请不要把真实 API Key 提交到 Git 仓库
- 推荐通过环境变量读取（例如 `OPENAI_API_KEY`）
- `OpenAIConfig.streaming=true` 会走流式 API；否则走非流式

## 运行测试

```bash
cargo test
```

更多用法与示例：

- docs/USAGE.md
- tests/chain_test.rs
- tests/agent_chain_like_test.rs

## License

MIT OR Apache-2.0
