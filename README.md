# langchainrust

一个受 LangChain 启发的 Rust 框架，用于构建基于大模型（LLM）的应用：提示词模板（Prompt Template）、链式调用（Chains）、工具调用（Tools）、Agent、Memory、检索（Retrieval）等。

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
cargo add langchainrust


```toml
[dependencies]
langchainrust = { git = "https://github.com/atliliw/langchainrust" }
tokio = { version = "1", features = ["full"] }
```

## 快速开始

### 1) 直接调用 LLM（纯文本）

```rust
use langchainrust::llms::{LLM, OpenAIConfig};

#[tokio::main]
async fn main() {
    let llm = LLM::new(OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY").unwrap(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
    });

    let out = llm.invoke("用一句话解释 Rust 的所有权").await.unwrap();
    println!("{}", out);
}
```

### 1.1) 流式输出（streaming）

```rust
use futures_util::StreamExt;
use langchainrust::llms::{LLM, OpenAIConfig};

#[tokio::main]
async fn main() {
    let llm = LLM::new(OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY").unwrap(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: true,
    });

    let mut stream = llm.stream_generate("生成一段 100 字的短文").await.unwrap();
    while let Some(tok) = stream.next().await {
        print!("{}", tok.unwrap());
    }
}
```

### 2) PromptTemplate（字符串模板）

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

### 3) ChatPromptTemplate（多消息模板）

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

### 4) Chains：SequentialChain（带 memory 注入）

与测试用例一致的“链式 + 记忆”例子可参考：

- [chain_test.rs](tests/chain_test.rs)

核心思路：
- `SequentialChain` 串联多个 `PromptChain`
- `SimpleMemory` 会把历史写入，下一步将 `chat_history` 作为 system message 注入模板

### 5) Agent：ReActAgent（带 memory + 提示词模板 + 多变量）

与测试用例一致的“像 chain 一样的 agent 用法”可参考：

- [agent_chain_like_test.rs](tests/agent_chain_like_test.rs)

这个例子体现了：
- `ChatPromptTemplate` 作为 agent 的提示词模板
- 运行时通过 `run_with_vars` 传入多个模板变量（例如 `{name}` `{style}` `{multiplier}`）
- `SimpleMemory` 以 chat_history（system message）方式注入到下一轮推理里

注意：当前 `ReActAgent` 的工具调用解析规则是从模型输出中查找以 `行为：` 开头的行，并解析为 `tool_name key=value ...` 的形式；否则视为最终答案。

示例（简化版）：

```rust
use langchainrust::agent::{AgentExecutor, ReActAgent};
use langchainrust::llms::{LLM, OpenAIConfig};
use langchainrust::memory::SimpleMemory;
use langchainrust::messages::Message;
use langchainrust::prompts::ChatPromptTemplate;
use langchainrust::tools::{Calculator, Tool};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let llm = LLM::new(OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY").unwrap(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
    });

    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Calculator)];

    let template = ChatPromptTemplate::new(vec![
        Message::system("你是数学家{name}，回答风格是{style}。"),
        Message::human("请详细计算：{input}。"),
        Message::human("请把结果乘{multiplier}后给出最终答案。"),
        Message::human("回答时附带你的名字：{name}。"),
    ]);

    let agent = ReActAgent::with_template(
        llm,
        tools.clone(),
        Some(Box::new(SimpleMemory::default())),
        template,
    );

    let executor = AgentExecutor::new(Box::new(agent), tools).with_max_iterations(3);

    let vars = HashMap::from([
        ("name".to_string(), "小李".to_string()),
        ("style".to_string(), "简洁".to_string()),
        ("multiplier".to_string(), "100".to_string()),
    ]);

    let out = executor.run_with_vars("1+3", vars).await.unwrap();
    println!("{}", out);
}
```

### 6) 自定义 Tool

内置工具在 [tools.rs](src/tools/tools.rs) 中提供了 `Calculator` / `WeatherTool`。你也可以按 `Tool` trait 自定义：

```rust
use async_trait::async_trait;
use langchainrust::tools::{Tool, ToolInput, ToolOutput};

pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "原样返回输入的 text"
    }

    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, Box<dyn std::error::Error>> {
        let text = input.parameters.get("text").ok_or("缺少 text 参数")?;
        Ok(ToolOutput { success: true, result: text.clone() })
    }

    fn parameters(&self) -> Vec<(&str, &str)> {
        vec![("text", "要回显的文本")]
    }

    fn return_direct(&self) -> bool {
        false
    }
}
```

## 配置与安全

- 请不要把真实 API Key 提交到 Git 仓库
- 推荐通过环境变量读取（例如 `OPENAI_API_KEY`）
- `OpenAIConfig.streaming=true` 会走流式 API；否则走非流式

## 运行测试

```bash
cargo test
```

单独运行某个测试文件：

```bash
cargo test --test chain_test
cargo test --test agent_chain_like_test
```

## 模块结构

- `src/llms`：LLM 实现（OpenAI/Qwen）
- `src/prompts`：提示词模板
- `src/messages`：消息结构（system/user/assistant）
- `src/memory`：记忆接口与实现
- `src/chains`：链式组合
- `src/tools`：工具接口与内置工具
- `src/agent`：Agent 与执行器
- `src/retrieval`：检索组件（实验性）

## License

MIT OR Apache-2.0
