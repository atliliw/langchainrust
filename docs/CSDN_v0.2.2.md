# langchainrust v0.2.2 发布：回调系统、Tool Calling 与 LangSmith 集成

## 前言

今天发布了 langchainrust v0.2.2 版本，这是一个重要的功能更新版本。主要新增了完整的**回调系统（Callback System）**和**Tool Calling 增强**，让 Rust 开发者也能享受到 LangChain Python 版的追踪和调试体验。

## 版本更新概览

| 功能 | 说明 |
|------|------|
| **CallbackHandler trait** | 定义 LLM/Chain/Tool/Retriever 回调接口 |
| **CallbackManager** | 多处理器管理和分发 |
| **StdOutHandler** | 控制台日志输出 |
| **LangSmithHandler** | LangSmith 平台追踪集成 |
| **bind_tools()** | LLM 绑定工具定义 |
| **ToolDefinition** | 工具定义结构 |
| **with_structured_output<T>()** | 结构化输出方法 |
| **Runnable trait** | LCEL 基础执行接口 |

---

## 一、回调系统（Callback System）

### 为什么需要回调系统？

在开发 LLM 应用时，我们经常遇到这些问题：
- **调试困难**：不知道 Agent 执行了哪些工具、传入了什么参数
- **性能分析**：不清楚 LLM 调用耗时多久、消耗了多少 tokens
- **生产监控**：无法追踪完整的执行链路

LangChain Python 的回调系统很好地解决了这些问题，现在 Rust 版本也拥有了同样的能力。

### 核心设计

```rust
// 回调处理器 trait
#[async_trait]
pub trait CallbackHandler: Send + Sync {
    // LLM 回调
    async fn on_llm_start(&self, run: &RunTree, messages: &[Message]);
    async fn on_llm_end(&self, run: &RunTree, response: &str);
    async fn on_llm_new_token(&self, run: &RunTree, token: &str);
    async fn on_llm_error(&self, run: &RunTree, error: &str);
    
    // 工具回调
    async fn on_tool_start(&self, run: &RunTree, tool_name: &str, input: &str);
    async fn on_tool_end(&self, run: &RunTree, output: &str);
    async fn on_tool_error(&self, run: &RunTree, error: &str);
    
    // Chain 回调
    async fn on_chain_start(&self, run: &RunTree, inputs: &Value);
    async fn on_chain_end(&self, run: &RunTree, outputs: &Value);
}

// 回调管理器
pub struct CallbackManager {
    handlers: Vec<Arc<dyn CallbackHandler>>,
}
```

### 使用示例：控制台日志

```rust
use langchainrust::callbacks::{CallbackManager, StdOutHandler};

// 创建回调管理器
let callbacks = Arc::new(
    CallbackManager::new()
        .add_handler(Arc::new(StdOutHandler::new()))
);

// 配置 RunnableConfig
let config = RunnableConfig::new()
    .with_callbacks(callbacks)
    .with_run_name("my_agent");

// 执行时自动触发回调
let result = agent.invoke("计算 15 + 27".into(), Some(config)).await?;
```

**输出效果：**

```
🤖 [LLM] START: my_agent
   ID: 019d7bfe-2c92-7871-ad4d-fb294a67d442
   Messages: 1 message(s)
   
🔧 [TOOL] START: calculator
   Input: {"expression": "15 + 27"}
   
🔧 [TOOL] END: calculator
   Output: 42
   
🤖 [LLM] END: my_agent (941ms)
   Tokens: prompt=20, completion=7, total=27
```

### LangSmith 集成

LangSmith 是 LangChain 官方的追踪平台，可以可视化查看完整的执行链路：

```rust
use langchainrust::callbacks::LangSmithHandler;

let langsmith = Arc::new(
    LangSmithHandler::new("your-langsmith-api-key")
        .with_project("my-project")
);

let callbacks = Arc::new(
    CallbackManager::new()
        .add_handler(Arc::new(StdOutHandler::new()))
        .add_handler(langsmith)
);
```

---

## 二、Tool Calling 增强

### bind_tools() API

```rust
use langchainrust::{OpenAIChat, ToolDefinition, Runnable};

let llm = OpenAIChat::from_env();

// 定义工具
let tools = vec![
    ToolDefinition::new("calculator", "计算数学表达式")
        .with_parameters(json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string" }
            }
        })),
];

// 绑定工具到 LLM
let llm_with_tools = llm.bind_tools(tools);

// 调用
let response = llm_with_tools.chat(
    vec![Message::human("计算 2 + 3")],
    None
).await?;

// 解析工具调用
if let Some(tool_calls) = response.tool_calls {
    for call in tool_calls {
        println!("工具: {}", call.function.name);
        println!("参数: {}", call.function.arguments);
    }
}
```

### 结构化输出

```rust
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherOutput {
    city: String,
    temperature: f32,
    condition: String,
}

// 获取结构化输出
let llm = OpenAIChat::from_env();
let structured = llm.with_structured_output::<WeatherOutput>();

let result: WeatherOutput = structured
    .invoke("北京今天的天气".into(), None)
    .await?;
    
println!("城市: {}, 温度: {}°C, 天气: {}", 
    result.city, result.temperature, result.condition);
```

---

## 三、Runnable 接口

为后续的 LCEL（LangChain Expression Language）做准备，新增了统一的执行接口：

```rust
#[async_trait]
pub trait Runnable<Input: Send + Sync + 'static, Output: Send + Sync + 'static>: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn invoke(&self, input: Input, config: Option<RunnableConfig>) -> Result<Output, Self::Error>;
    
    async fn batch(&self, inputs: Vec<Input>, config: Option<RunnableConfig>) -> Result<Vec<Output>, Self::Error>;
    
    async fn stream(&self, input: Input, config: Option<RunnableConfig>) 
        -> Result<Pin<Box<dyn Stream<Item = Result<Output, Self::Error>> + Send>>, Self::Error>;
}
```

---

## 四、安装和使用

### 安装

```toml
[dependencies]
langchainrust = "0.2.2"
tokio = { version = "1.0", features = ["full"] }
```

### 基础使用

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;
use langchainrust::callbacks::{CallbackManager, StdOutHandler};
use langchainrust::RunnableConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建 LLM
    let llm = OpenAIChat::new(OpenAIConfig::from_env());
    
    // 添加回调
    let callbacks = Arc::new(
        CallbackManager::new().add_handler(Arc::new(StdOutHandler::new()))
    );
    let config = RunnableConfig::new().with_callbacks(callbacks);
    
    // 调用
    let response = llm.chat(
        vec![Message::human("你好，介绍一下 Rust 语言")],
        Some(config)
    ).await?;
    
    println!("{}", response.content);
    Ok(())
}
```

---

## 五、完整示例：Agent + Tools + Callbacks

```rust
use langchainrust::{
    agents::{ReActAgent, AgentExecutor},
    tools::{Calculator, SimpleMathTool},
    callbacks::{CallbackManager, StdOutHandler},
    RunnableConfig,
    BaseTool,
};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // 创建工具
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
        Arc::new(SimpleMathTool::new()),
    ];
    
    // 创建 Agent
    let llm = OpenAIChat::from_env();
    let agent = Arc::new(ReActAgent::new(llm, tools.clone(), None));
    
    // 创建 Executor（带回调）
    let callbacks = Arc::new(
        CallbackManager::new().add_handler(Arc::new(StdOutHandler::new()))
    );
    
    let executor = AgentExecutor::new(agent, tools)
        .with_verbose(true)
        .with_callbacks(callbacks);
    
    // 执行
    let result = executor.invoke("计算 5 的阶乘".into()).await.unwrap();
    println!("结果: {}", result);
}
```

**输出：**

```
=== 迭代 1 ===
Thought: 需要计算阶乘，使用数学工具
Action: math
Action Input: {"operation": "factorial", "value": 5}
Observation: 120

最终答案: 5的阶乘是120
结果: 5的阶乘是120
```

---

## 六、Roadmap

后续版本计划：

| 功能 | 优先级 | 说明 |
|------|--------|------|
| Ollama 支持 | P1 | 本地模型推理，OpenAI API 兼容 |
| Runnable::stream() | P0 | 完善流式输出 trait |
| OpenAIFunctionsAgent | P1 | 使用 function calling 的 Agent |
| LCEL 组合操作符 | P2 | 简化链式组合 |
| 更多向量数据库 | P2 | Milvus、Pinecone、ChromaDB |
| LangGraph | P3 | 状态机 Agent |

---

## 七、GitHub 地址

项目地址：https://github.com/atliliw/langchainrust

欢迎大家 Star、提 Issue、贡献代码！

---

## 总结

langchainrust v0.2.2 版本为 Rust LLM 开发带来了：

1. **完整的回调系统** - 追踪、调试、监控一体化
2. **LangSmith 集成** - 可视化执行链路
3. **Tool Calling 增强** - bind_tools()、结构化输出
4. **Runnable 接口** - 为 LCEL 打基础

Rust 开发者现在可以像 Python 版一样方便地调试和监控 LLM 应用了！