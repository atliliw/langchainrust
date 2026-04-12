# 使用文档

这份文档承载 README 的详细内容；GitHub 首页 README 会保持简短。

## 目录

- [LLM](#llm)
  - 直接调用（纯文本）
  - 流式输出（streaming）
  - Function Calling（bind_tools）
- [Prompts](#prompts)
  - PromptTemplate
  - ChatPromptTemplate
- [Memory](#memory)
- [Chains](#chains)
- [Agent](#agent)
  - FunctionCallingAgent（推荐）
  - ReActAgent（兼容旧模型）
  - 两种 Agent 对比
  - 模型路由（智能选择模型）
  - Tool 调用输出格式约定
- [Tools](#tools)
  - 内置工具
  - 自定义 Tool
  - to_tool_definition()
- [配置与安全](#配置与安全)
- [测试](#测试)
- [模块结构](#模块结构)

---

## LLM

### 直接调用 LLM（纯文本）

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")?,
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        ..Default::default()
    };
    
    let llm = OpenAIChat::new(config);
    
    let messages = vec![
        Message::system("你是一个友好的助手。"),
        Message::human("什么是 Rust 语言？"),
    ];
    
    let response = llm.chat(messages, None).await?;
    println!("{}", response.content);
    
    Ok(())
}
```

### 流式输出（streaming）

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, BaseChatModel};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")?,
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        streaming: true,
        ..Default::default()
    };
    
    let llm = OpenAIChat::new(config);
    
    let messages = vec![Message::human("写一段 100 字的短文")];
    
    let stream = llm.stream_chat(messages, None).await?;
    // 使用 futures_util::StreamExt 处理流
    // while let Some(tok) = stream.next().await { print!("{}", tok?); }
    
    Ok(())
}
```

### Function Calling（bind_tools）

通过 `bind_tools()` 将工具绑定到 LLM，让模型能够调用外部函数：

```rust
use langchainrust::{OpenAIChat, OpenAIConfig, ToolDefinition, BaseChatModel};
use langchainrust::schema::Message;
use schemars::JsonSchema;
use serde::Deserialize;

// 定义工具输入类型
#[derive(JsonSchema, Deserialize)]
struct CalculatorInput {
    expression: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")?,
        base_url: "https://api.openai.com/v1".to_string(),
        model: "gpt-3.5-turbo".to_string(),
        ..Default::default()
    };
    
    let llm = OpenAIChat::new(config);
    
    // 创建工具定义（自动生成 JSON Schema）
    let tool = ToolDefinition::from_type::<CalculatorInput>(
        "calculator",
        "计算数学表达式"
    );
    
    // 绑定工具到 LLM
    let llm_with_tools = llm.bind_tools(vec![tool]);
    
    let messages = vec![Message::human("计算 25 + 17")];
    let response = llm_with_tools.chat(messages, None).await?;
    
    // 检查是否有工具调用
    if let Some(tool_calls) = response.tool_calls {
        let call = &tool_calls[0];
        println!("工具: {}", call.function.name);
        println!("参数: {}", call.function.arguments);
        
        // 解析参数
        let args: CalculatorInput = call.parse_arguments()?;
        println!("表达式: {}", args.expression);
    }
    
    Ok(())
}
```

### OpenAIConfig 配置项

| 字段 | 类型 | 说明 |
|------|------|------|
| `api_key` | String | API 密钥 |
| `base_url` | String | API 基础 URL |
| `model` | String | 模型名称 |
| `streaming` | bool | 是否启用流式输出 |
| `temperature` | Option<f32> | 采样温度 (0.0-2.0) |
| `max_tokens` | Option<usize> | 最大生成 token 数 |
| `tools` | Option<Vec<ToolDefinition>> | 绑定的工具定义 |

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

LangChainRust 提供两种 Agent：

| Agent | 工具调用方式 | 适用场景 |
|-------|-------------|----------|
| **FunctionCallingAgent** | 原生 Function Calling | 支持 FC 的模型（GPT-4、Claude、Gemini）**推荐** |
| **ReActAgent** | 文本解析（正则提取） | 不支持 FC 的模型、开源模型、本地部署模型 |

### FunctionCallingAgent（推荐）

使用原生 Function Calling，类型安全，更可靠。

#### 创建 FunctionCallingAgent

```rust
use langchainrust::{
    OpenAIChat, OpenAIConfig, BaseChatModel,
    FunctionCallingAgent, AgentExecutor, BaseAgent, BaseTool,
    Calculator, DateTimeTool,
};
use std::sync::Arc;

let config = OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY")?,
    base_url: "https://api.openai.com/v1".to_string(),
    model: "gpt-3.5-turbo".to_string(),
    ..Default::default()
};

let llm = OpenAIChat::new(config);

let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(DateTimeTool::new()),
];

// FunctionCallingAgent 自动绑定工具到 LLM
let agent = FunctionCallingAgent::new(llm, tools.clone(), None);

let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
    .with_max_iterations(5)
    .with_verbose(true);

let result = executor.invoke("计算 37 + 48".to_string()).await?;
println!("结果: {}", result);
```

#### 带自定义系统提示词

```rust
let agent = FunctionCallingAgent::new(
    llm,
    tools.clone(),
    Some("你是一个数学助手，专门帮助用户解决数学问题。".to_string()),
);
```

#### 执行日志

```
=== 迭代 1 ===
动作: calculator({"expression":"37 + 48"})
观察: 37 + 48 = 85

=== 迭代 2 ===
最终答案: {"output": "37 + 48 = 85"}
```

---

### ReActAgent（兼容旧模型）

使用文本解析方式，适用于不支持 Function Calling 的模型。

#### 创建 ReActAgent

```rust
use langchainrust::{
    OpenAIChat, OpenAIConfig, BaseChatModel,
    ReActAgent, AgentExecutor, BaseAgent, BaseTool,
    Calculator, DateTimeTool, SimpleMathTool,
};
use std::sync::Arc;

let llm = OpenAIChat::new(config);

let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(DateTimeTool::new()),
    Arc::new(SimpleMathTool::new()),
];

let agent = ReActAgent::new(llm, tools.clone(), None);

let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
    .with_max_iterations(5);

let result = executor.invoke("计算 37 + 48".to_string()).await?;
println!("结果: {}", result);
```

---

### 两种 Agent 对比

| 维度 | FunctionCallingAgent | ReActAgent |
|------|---------------------|------------|
| **工具调用方式** | 原生 Function Calling | 文本解析（正则） |
| **可靠性** | 高（类型安全） | 较低（依赖 Prompt 格式） |
| **Token 消耗** | 低（不需要格式说明） | 高（Prompt 包含格式说明） |
| **参数处理** | JSON Schema 类型安全 | 文本提取，可能格式错误 |
| **历史传递** | Message 消息流 | scratchpad 文本 |
| **适用模型** | 支持 FC 的模型 | 所有模型 |

#### 选择建议

- 使用 OpenAI GPT-4/Claude/Gemini → **FunctionCallingAgent**
- 使用本地部署模型/开源模型 → **ReActAgent**

---

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

### Tool 调用机制

`ReActAgent` 的工具调用是**可选的**：

1. **智能判断**：Agent 会根据问题内容自主判断是否需要使用工具
2. **工具调用格式**：如果需要使用工具，模型会输出 `[TOOL: 工具名 参数名=参数值]`
3. **直接回答**：如果不需要工具，模型会直接给出答案

```
# 使用工具的示例输出
[TOOL: calculator expression=37+48]

# 不使用工具的示例输出
Rust 是一种系统编程语言，具有内存安全、零成本抽象等特点...
```

### 判断是否使用了工具

使用 `run_with_details()` 方法获取执行详情：

```rust
use langchainrust::agent::{AgentExecutor, ReActAgent, ExecutionResult};

let executor = AgentExecutor::new(Box::new(agent), tools);

// 获取执行详情
let result: ExecutionResult = executor.run_with_details("计算 37+48").await?;

println!("最终答案: {}", result.answer);
println!("是否使用工具: {}", result.used_tools);
println!("调用的工具: {:?}", result.tool_calls);
println!("迭代次数: {}", result.iterations);

if result.used_tools {
    println!("✓ Agent 使用了工具来帮助回答");
} else {
    println!("✓ Agent 直接回答了问题");
}
```

### 执行日志

执行时会打印工具调用日志：

```
[工具调用] calculator {"expression": "37+48"}
[工具结果] 85
```

---

## Retrieval + Agent (RAG)

将向量检索与 Agent 结合，实现 **RAG（检索增强生成）** 功能。

### 工作原理

```
┌─────────────┐      检索器      ┌─────────────┐
│   用户问题   │  ─────────────>  │  向量数据库  │
└─────────────┘                  └─────────────┘
       │                                │
       │                                ▼
       │                         ┌─────────────┐
       │                         │  相关文档   │
       │                         └─────────────┘
       │                                │
       ▼                                ▼
┌─────────────────────────────────────────────┐
│                   LLM                        │
│  (问题 + 检索到的文档作为上下文)              │
└─────────────────────────────────────────────┘
                      │
                      ▼
               ┌─────────────┐
               │   最终答案   │
               └─────────────┘
```

### 使用 with_retriever

通过 `ReActAgent::with_retriever` 创建带检索功能的 Agent：

```rust
use langchainrust::agent::{AgentExecutor, ReActAgent};
use langchainrust::llms::LLM;
use langchainrust::memory::SimpleMemory;
use langchainrust::retrieval::{
    Document, InMemoryVectorStore, MockEmbeddingModel,
    RecursiveCharacterSplitter, Retriever, SimilarityRetriever, TextSplitter,
};
use std::sync::Arc;

// 1. 准备文档
let docs = vec![
    Document::new("Rust是一种系统编程语言...".to_string()),
    Document::new("Python是一种脚本语言...".to_string()),
];

// 2. 分割文档
let splitter = RecursiveCharacterSplitter::new(100, 20);
let mut chunks = Vec::new();
for doc in docs {
    chunks.extend(splitter.split_document(&doc).unwrap());
}

// 3. 创建检索器
let embedding = Arc::new(MockEmbeddingModel::new(128));
let store = Box::new(InMemoryVectorStore::new());
let retriever = SimilarityRetriever::new(store, embedding);
retriever.add_documents(chunks).await?;

// 4. 创建 RAG Agent（传入 retriever）
let llm = LLM::new(config);
let agent = ReActAgent::with_retriever(
    llm,
    vec![],                              // 工具（可选）
    Some(Box::new(SimpleMemory::default())), // 记忆（可选）
    Arc::new(retriever) as Arc<dyn Retriever>,
    3,  // top_k
);

// 5. 执行查询
let executor = AgentExecutor::new(Box::new(agent), vec![]);
let result = executor.run_with_details("什么是Rust？").await?;
println!("回答: {}", result.answer);
```

### 带自定义模板

使用 `with_retriever_and_template`：

```rust
use langchainrust::messages::Message;
use langchainrust::prompts::ChatPromptTemplate;

let template = ChatPromptTemplate::new(vec![
    Message::system(
        "你是专业顾问。根据参考资料回答问题。\n\n参考资料：\n{context}",
    ),
    Message::human("{input}"),
]);

let agent = ReActAgent::with_retriever_and_template(
    llm,
    vec![],
    None,
    retriever,
    3,
    template,
);
```

### 执行日志

```
[检索] 正在从向量数据库检索相关文档...
[检索] 找到 3 个相关文档:
  [1] 相似度: 0.8542
  [2] 相似度: 0.7231
  [3] 相似度: 0.5123
```

### 不传 retriever

如果不传 retriever，Agent 会走普通逻辑（不检索）：

```rust
// 普通 Agent（无检索）
let agent = ReActAgent::new(llm, tools, memory);

// 带 retriever 的 Agent（会先检索文档）
let agent = ReActAgent::with_retriever(llm, tools, memory, retriever, 3);
```

---

## 任务规划（Task Planning）

将复杂任务自动分解为子任务，依次执行，最后汇总结果。

### 工作原理

```
┌─────────────────┐
│   复杂问题       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TaskPlanner    │  ← 分解任务
│  分解为子任务    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  子任务 1       │ → 结果 1
├─────────────────┤
│  子任务 2       │ → 结果 2
├─────────────────┤
│  子任务 3       │ → 结果 3
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    汇总结果      │
└─────────────────┘
```

### TaskPlanner（任务规划器）

```rust
use langchainrust::agent::TaskPlanner;
use langchainrust::llms::LLM;

let llm = LLM::new(config);
let planner = TaskPlanner::new(llm)
    .with_max_sub_tasks(5)
    .with_verbose(true);  // 开启日志输出

// 分解任务
let plan = planner.plan("分析项目代码，写测试用例，运行测试").await?;

println!("分解为 {} 个子任务:", plan.sub_tasks.len());
for task in &plan.sub_tasks {
    println!("  [{}] {}", task.id, task.description);
}
```

### PlannedExecutor（自动规划执行器）

自动完成：规划 → 执行 → 汇总

```rust
use langchainrust::agent::{PlannedExecutor, ReActAgent};
use langchainrust::llms::LLM;
use langchainrust::tools::Tool;
use std::sync::Arc;

let llm = LLM::new(config);
let tools: Vec<Arc<dyn Tool>> = vec![];

let planned_executor = PlannedExecutor::new(
    llm,
    Box::new(ReActAgent::new(LLM::new(config), tools.clone(), None)),
    tools,
)
.with_max_sub_tasks(3)      // 最多 3 个子任务
.with_max_iterations(2)    // 每个子任务最多 2 次迭代
.with_verbose(true);       // 开启日志输出

// 执行复杂任务
let result = planned_executor
    .run("调研 Rust 异步编程最佳实践，写示例代码，解释关键点")
    .await?;

println!("{}", result);
```

### 日志控制

默认不打印工作过程日志，使用 `with_verbose(true)` 开启：

```rust
// 不打印日志（默认）
let executor = PlannedExecutor::new(llm, agent, tools);

// 打印详细日志
let executor = PlannedExecutor::new(llm, agent, tools)
    .with_verbose(true);
```

开启日志后的输出示例：

```
[规划] 正在分析任务...
[规划] 任务已分解为 3 个子任务:
  [1] 分析项目结构
  [2] 提取核心功能
  [3] 生成总结报告

[执行] 任务 1/3: 分析项目结构
[完成] 任务 1 执行成功
...

[汇总] 正在汇总所有任务结果...
```

### 获取详细执行结果

```rust
// 返回规划详情和每个子任务的结果
let (plan, results) = planned_executor
    .run_with_plan("复杂问题")
    .await?;

for result in &results {
    println!("任务 {}: {} - {}",
        result.id,
        result.description,
        if result.success { "成功" } else { "失败" }
    );
}
```

### 执行日志

```
[规划] 正在分析任务...
[规划] 任务已分解为 3 个子任务:
  [1] 分析项目结构
  [2] 提取核心功能
  [3] 生成总结报告

[执行] 任务 1/3: 分析项目结构
[完成] 任务 1 执行成功
[执行] 任务 2/3: 提取核心功能
[完成] 任务 2 执行成功
[执行] 任务 3/3: 生成总结报告
[完成] 任务 3 执行成功

[汇总] 正在汇总所有任务结果...
```

### SimplePlannedExecutor（简化版）

只规划不执行：

```rust
use langchainrust::agent::SimplePlannedExecutor;

let executor = SimplePlannedExecutor::new(llm);

// 只获取任务规划
let plan = executor.plan("复杂问题").await?;

// 手动执行每个子任务...

// 汇总结果
let summary = executor.summarize("原始问题", &results).await?;
```

### 子任务依赖关系

```rust
pub struct SubTask {
    pub id: usize,
    pub description: String,
    pub depends_on_previous: bool,  // 是否依赖前一个任务
}
```

当 `depends_on_previous = true` 时，执行器会自动将上一个任务的结果附加到当前任务的输入中。

---

## Tools

### 内置工具

| 工具名 | 描述 | 参数 |
|--------|------|------|
| `Calculator` | 执行基本数学运算（加减乘除） | `expression`: 数学表达式 |
| `SimpleMathTool` | 高级数学运算（幂、开方、三角函数等） | `operation`: 操作类型, `value`: 数值 |
| `DateTimeTool` | 日期时间查询和计算 | `operation`: 操作类型, `datetime`: 日期时间 |
| `URLFetchTool` | 网页抓取和解析 | `operation`: 操作类型, `url`: 网址 |

### 使用工具

```rust
use langchainrust::{Calculator, DateTimeTool, SimpleMathTool, BaseTool};
use std::sync::Arc;

let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(DateTimeTool::new()),
    Arc::new(SimpleMathTool::new()),
];
```

### 直接调用工具

```rust
use langchainrust::{SimpleMathTool, BaseTool};

let math = SimpleMathTool::new();

let result = math.run(r#"{"operation": "power", "value": 2, "value2": 10}"#.to_string()).await?;
println!("2^10 = {}", result); // 输出: 2^10 = 1024
```

### to_tool_definition()

将 BaseTool 转换为 ToolDefinition，用于 Function Calling：

```rust
use langchainrust::{Calculator, BaseTool, to_tool_definition, ToolDefinition};

let calculator = Calculator::new();

// 自动从 args_schema() 生成 JSON Schema
let tool_def: ToolDefinition = to_tool_definition(&calculator);

// 用于 bind_tools()
let llm_with_tools = llm.bind_tools(vec![tool_def]);
```

### 自定义 Tool

实现 `BaseTool` trait 创建自定义工具：

```rust
use async_trait::async_trait;
use langchainrust::{BaseTool, ToolError};
use schemars::JsonSchema;
use serde::Deserialize;

// 定义输入类型（用于 JSON Schema）
#[derive(JsonSchema, Deserialize)]
struct EchoInput {
    text: String,
}

pub struct EchoTool;

#[async_trait]
impl BaseTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "原样返回输入的文本"
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        // 解析输入
        let args: EchoInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(e.to_string()))?;
        
        Ok(args.text)
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        // 自动生成 JSON Schema
        use schemars::schema_for;
        serde_json::to_value(schema_for!(EchoInput)).ok()
    }
}
```

使用自定义工具：

```rust
let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(EchoTool),
];

let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
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

# 运行单元测试
cargo test --lib

# 运行 Function Calling Agent 测试
cargo test --test function_calling_agent -- --include-ignored --nocapture

# 运行 ReAct Agent 测试
cargo test --test integration_agent_react -- --include-ignored --nocapture

# 运行特定测试函数
cargo test test_fc_agent_with_calculator -- --include-ignored --nocapture
```

---

## 模块结构

```
src/
├── core/                # 核心抽象
│   ├── language_models/ # Base LLM traits
│   ├── runnables/       # Runnable trait
│   └── tools/           # Tool trait + ToolDefinition + to_tool_definition()
├── language_models/     # LLM 实现
│   └── openai/          # OpenAI 客户端（支持 bind_tools）
├── agents/              # Agent 框架
│   ├── react/           # ReActAgent（文本解析）
│   └── function_calling/ # FunctionCallingAgent（原生 FC）
├── prompts/             # 提示词模板
├── memory/              # 记忆管理
├── chains/              # 链式调用
├── retrieval/           # RAG 组件
├── embeddings/          # 文本嵌入
├── vector_stores/       # 向量存储
├── tools/               # 内置工具
└── schema/              # 数据结构（Message、ToolCall）
```

---

## API 速查

### Agent 构造方法

| Agent | 方法 | 说明 |
|-------|------|------|
| **FunctionCallingAgent** | `new(llm, tools, system_prompt)` | 推荐，原生 FC |
| **ReActAgent** | `new(llm, tools, memory)` | 兼容旧模型 |

### 工具调用

| Agent | 工具调用方式 | 说明 |
|-------|-------------|------|
| FunctionCallingAgent | `tool_calls` JSON | 类型安全 |
| ReActAgent | 文本 `Action: xxx` | 正则提取 |

### Function Calling API

| 方法 | 说明 |
|------|------|
| `bind_tools(vec![ToolDefinition])` | 绑定工具到 LLM |
| `to_tool_definition(&tool)` | 将 BaseTool 转为 ToolDefinition |
| `ToolDefinition::from_type::<T>()` | 从类型自动生成 Schema |
| `ToolCall::parse_arguments::<T>()` | 解析工具参数 |

### RAG 执行流程

| 步骤 | 说明 |
|------|------|
| 1. 检索 | 从向量数据库检索 top_k 个相关文档 |
| 2. 构建上下文 | 将文档格式化为 prompt 上下文 |
| 3. 调用 LLM | 将问题和上下文一起发送给模型 |
| 4. 返回答案 | 模型基于上下文生成答案 |

