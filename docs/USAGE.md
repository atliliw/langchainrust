# 使用文档

这份文档承载 README 的详细内容；GitHub 首页 README 会保持简短。

## 目录

- [LLM](#llm)
  - 直接调用（纯文本）
  - 流式输出（streaming）
- [Prompts](#prompts)
  - PromptTemplate
  - ChatPromptTemplate
- [Memory](#memory)
- [Chains](#chains)
- [Agent](#agent)
  - ReActAgent（基础用法）
  - 模型路由（智能选择模型）
  - Tool 调用输出格式约定
- [Tools](#tools)
  - 自定义 Tool
- [配置与安全](#配置与安全)
- [测试](#测试)
- [模块结构](#模块结构)

---

## LLM

### 直接调用 LLM（纯文本）

```rust
use langchainrust::llms::{LLM, OpenAIConfig};

#[tokio::main]
async fn main() {
    let llm = LLM::new(OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY").unwrap(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        factor: 3,
    });

    let out = llm.invoke("用一句话解释 Rust 的所有权").await.unwrap();
    println!("{}", out);
}
```

### 流式输出（streaming）

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
        factor: 3,
    });

    let mut stream = llm.stream_generate("生成一段 100 字的短文").await.unwrap();
    while let Some(tok) = stream.next().await {
        print!("{}", tok.unwrap());
    }
}
```

### OpenAIConfig 配置项

| 字段 | 类型 | 说明 |
|------|------|------|
| `api_key` | String | API 密钥 |
| `base_url` | String | API 基础 URL |
| `model` | String | 模型名称 |
| `streaming` | bool | 是否启用流式输出 |
| `factor` | u8 | 模型系数（1-10），用于模型路由 |

---

## Prompts

### PromptTemplate（字符串模板）

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

### ChatPromptTemplate（多消息模板）

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

---

## Memory

### SimpleMemory

`SimpleMemory` 会保存一定数量的历史输出（默认 5 条），可以用于构造对话上下文。

```rust
use langchainrust::memory::SimpleMemory;

let memory = SimpleMemory::default();
memory.add("用户问题", "AI回答");
let context = memory.context();  // 获取上下文
let history = memory.history();  // 获取历史记录列表
```

---

## Chains

### SequentialChain（带 memory 注入）

与测试用例一致的"链式 + 记忆"例子可参考：[chain_test.rs](../tests/chain_test.rs)

核心思路：
- `SequentialChain` 串联多个 `PromptChain`
- `SimpleMemory` 会把历史写入，下一步将 `chat_history` 作为 system message 注入模板

---

## Agent

### ReActAgent（基础用法）

`ReActAgent` 是一个可以根据工具描述自动决定调用哪个工具的 Agent。

#### 创建 Agent

```rust
use langchainrust::agent::{AgentExecutor, ReActAgent};
use langchainrust::llms::{LLM, OpenAIConfig};
use langchainrust::memory::SimpleMemory;
use langchainrust::tools::{Calculator, Tool};
use std::sync::Arc;

let llm = LLM::new(OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY").unwrap(),
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-3.5-turbo".to_string(),
    streaming: false,
    factor: 3,
});

let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Calculator)];

// 方式1：基础创建
let agent = ReActAgent::new(llm, tools.clone(), None);

// 方式2：带 Memory
let agent = ReActAgent::new(llm, tools.clone(), Some(Box::new(SimpleMemory::default())));

// 方式3：带模板
use langchainrust::messages::Message;
use langchainrust::prompts::ChatPromptTemplate;

let template = ChatPromptTemplate::new(vec![
    Message::system("你是数学家{name}，回答风格是{style}。"),
    Message::human("请计算：{input}"),
]);

let agent = ReActAgent::with_template(
    llm,
    tools.clone(),
    Some(Box::new(SimpleMemory::default())),
    template,
);
```

#### 执行 Agent

```rust
let executor = AgentExecutor::new(Box::new(agent), tools).with_max_iterations(5);

// 基础调用
let result = executor.run("37 加 48 等于多少？").await.unwrap();

// 带变量调用
use std::collections::HashMap;
let vars = HashMap::from([
    ("name".to_string(), "小李".to_string()),
    ("style".to_string(), "简洁".to_string()),
]);
let result = executor.run_with_vars("1+3", vars).await.unwrap();
```

---

### 模型路由（智能选择模型）

`ReActAgent` 支持模型路由功能：根据问题难度自动选择最合适的模型。

#### 核心概念

- **系数（Factor）**：1-10 的等级，越贵/越强的模型系数越高
  - `gpt-3.5-turbo`: 1-3（便宜、快速）
  - `gpt-4`: 7-9（强大、昂贵）
  - `gpt-4-turbo`: 9-10（最强）
- **难度（Difficulty）**：传入的参数（1-10），未传入时默认为 1
- **路由逻辑**：选择系数 ≥ 难度 的最小系数模型（性价比最高）

#### 使用模型路由

```rust
use langchainrust::agent::{AgentExecutor, ReActAgent};
use langchainrust::llms::{LLM, OpenAIConfig, ModelConfig};
use langchainrust::tools::{Calculator, Tool};
use std::collections::HashMap;
use std::sync::Arc;

// 路由 LLM（用于决策选择哪个模型）
let router_llm = LLM::new(OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY").unwrap(),
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-3.5-turbo".to_string(),
    streaming: false,
    factor: 3,
});

// 候选模型列表
let models = vec![
    ModelConfig::OpenAI(OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY").unwrap(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        factor: 1,  // 便宜
    }),
    ModelConfig::OpenAI(OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY").unwrap(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-4".to_string(),
        streaming: false,
        factor: 8,  // 强大
    }),
];

let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Calculator)];

// 使用 with_models 创建带模型路由的 Agent
let agent = ReActAgent::with_models(
    router_llm,
    models,
    tools.clone(),
    None,           // memory
    None,           // template（可选）
);

let executor = AgentExecutor::new(Box::new(agent), tools).with_max_iterations(3);

// 传入难度参数
let vars = HashMap::from([("difficulty".to_string(), "8".to_string())]);
let result = executor
    .run_with_vars("用Python写一个快速排序算法", vars)
    .await
    .unwrap();
```

#### 混合多厂商模型

```rust
use langchainrust::llms::{ModelConfig, QwenConfig};

let models = vec![
    // OpenAI 模型
    ModelConfig::OpenAI(OpenAIConfig {
        api_key: "sk-xxx".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        factor: 2,
    }),
    ModelConfig::OpenAI(OpenAIConfig {
        api_key: "sk-xxx".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-4".to_string(),
        streaming: false,
        factor: 8,
    }),
    // Qwen 模型
    ModelConfig::Qwen(QwenConfig {
        api_key: "xxx".to_string(),
        base_url: "https://dashscope.aliyuncs.com/api/v1".to_string(),
        model: "qwen-plus".to_string(),
        factor: 5,
    }),
];
```

#### 路由选择逻辑

1. 路由 LLM 分析问题和候选模型列表
2. 根据 `difficulty` 参数（默认 1）选择匹配的模型
3. 优先选择系数 ≥ difficulty 的最小系数模型
4. 如果没有匹配的，选择系数最大的模型

#### 难度配置建议

| 使用场景 | difficulty 值 | 建议模型 |
|---------|---------------|----------|
| 简单问答、计算 | 1-3 | gpt-3.5-turbo |
| 代码生成 | 4-6 | gpt-4 |
| 复杂推理 | 7-9 | gpt-4-turbo |
| 多步规划 | 10 | 最强模型 |

---

### Tool 调用输出格式约定

当前 `ReActAgent` 的工具调用解析规则是从模型输出中查找以 `行为：` 开头的行，并解析为 `tool_name key=value ...` 的形式；否则视为最终答案。

---

## Tools

### 自定义 Tool

内置工具在 [tools.rs](../src/tools/tools.rs) 中提供了 `Calculator` / `WeatherTool`。你也可以按 `Tool` trait 自定义：

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

---

## 配置与安全

- **不要**把真实 API Key 提交到 Git 仓库
- 推荐通过环境变量读取（例如 `OPENAI_API_KEY`）
- `OpenAIConfig.streaming=true` 走流式 API；否则走非流式
- `OpenAIConfig.factor` 用于模型路由，值越大表示模型越强/越贵

---

## 测试

```bash
# 运行全部测试
cargo test

# 运行单个测试文件
cargo test --test chain_test
cargo test --test agent_chain_like_test

# 运行模型路由测试
cargo test --test model_factor_test -- --nocapture

# 运行特定测试函数
cargo test test_model_routing_with_real_question -- --nocapture
```

---

## 模块结构

```
src/
├── llms/           # LLM 实现
│   ├── mod.rs      # 导出 LLM、OpenAIConfig、ModelConfig、QwenConfig
│   ├── openai.rs   # OpenAI 兼容接口
│   └── qwen.rs     # Qwen 接口
├── prompts/        # 提示词模板
├── messages/       # 消息结构（system/user/assistant）
├── memory/         # 记忆接口与实现
├── chains/         # 链式组合
├── tools/          # 工具接口与内置工具
├── agent/          # Agent 与执行器
│   ├── mod.rs      # Agent trait、AgentAction、AgentError
│   ├── agent.rs    # ReActAgent（含模型路由）
│   └── executor.rs # AgentExecutor
└── retrieval/      # 检索组件（实验性）
```

---

## API 速查

### ReActAgent 构造方法

| 方法 | 参数 | 说明 |
|------|------|------|
| `new(llm, tools, memory)` | LLM、工具列表、可选内存 | 基础 Agent |
| `with_template(llm, tools, memory, template)` | 增加模板参数 | 带模板的 Agent |
| `with_models(llm, models, tools, memory, template)` | 增加模型列表参数 | 带模型路由的 Agent |

### 模型路由参数

| 参数名 | 别名 | 默认值 | 说明 |
|--------|------|--------|------|
| `difficulty` | `难度`、`level` | 1 | 问题难度（1-10） |
