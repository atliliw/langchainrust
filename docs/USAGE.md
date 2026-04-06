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
| `SimpleMathTool` | 高级数学运算（幂、开方、三角函数等） | `operation`: 操作类型, `value`: 数值, `value2`: 第二个数值（可选） |
| `DateTimeTool` | 日期时间查询和计算 | `operation`: 操作类型, `datetime`: 日期时间（可选）, `value`: 数值（可选）, `unit`: 单位（可选） |
| `URLFetchTool` | 网页抓取和解析 | `operation`: 操作类型, `url`: 网址, `max_length`: 最大长度（可选） |
| `WeatherTool` | 获取城市天气（模拟） | `city`: 城市名称 |
| `TextTool` | 文本处理工具 | `operation`: 操作类型, `text`: 文本内容 |
| `WebSearchTool` | 网络搜索（模拟） | `query`: 搜索关键词 |
| `JsonTool` | JSON 处理工具 | `operation`: 操作类型, `data`: JSON字符串 |

### SimpleMathTool 详细用法

支持的操作类型：

| 操作 | 说明 | 示例 |
|------|------|------|
| `power` | 幂运算 | `{"operation": "power", "value": 2, "value2": 10}` → 1024 |
| `sqrt` | 平方根 | `{"operation": "sqrt", "value": 16}` → 4 |
| `log` | 对数（可指定底数） | `{"operation": "log", "value": 100, "base": 10}` → 2 |
| `ln` | 自然对数 | `{"operation": "ln", "value": 2.718}` → ~1 |
| `sin` | 正弦函数（弧度） | `{"operation": "sin", "value": 1.57}` → ~1 |
| `cos` | 余弦函数（弧度） | `{"operation": "cos", "value": 0}` → 1 |
| `tan` | 正切函数（弧度） | `{"operation": "tan", "value": 0.785}` → ~1 |
| `abs` | 绝对值 | `{"operation": "abs", "value": -5}` → 5 |
| `factorial` | 阶乘（最大20） | `{"operation": "factorial", "value": 5}` → 120 |
| `mod` | 取模运算 | `{"operation": "mod", "value": 17, "value2": 5}` → 2 |
| `gcd` | 最大公约数 | `{"operation": "gcd", "value": 12, "value2": 18}` → 6 |
| `lcm` | 最小公倍数 | `{"operation": "lcm", "value": 4, "value2": 6}` → 12 |
| `pi` | 圆周率 | `{"operation": "pi"}` → 3.14159... |
| `e` | 自然常数 | `{"operation": "e"}` → 2.71828... |

### DateTimeTool 详细用法

支持的操作类型：

| 操作 | 说明 | 示例 |
|------|------|------|
| `now` | 获取当前时间 | `{"operation": "now"}` |
| `format` | 格式化日期 | `{"operation": "format", "datetime": "2024-01-15"}` |
| `add` | 添加时间 | `{"operation": "add", "datetime": "2024-01-15", "value": 3, "unit": "days"}` |
| `subtract` | 减去时间 | `{"operation": "subtract", "datetime": "2024-01-15", "value": 1, "unit": "weeks"}` |
| `weekday` | 获取星期几 | `{"operation": "weekday", "datetime": "2024-01-15"}` → 星期一 |
| `diff` | 计算时间差 | `{"operation": "diff", "datetime": "2024-01-01", "target": "2024-01-15"}` → 14天 |

时间单位支持：`seconds`, `minutes`, `hours`, `days`, `weeks`, `months`, `years`

### URLFetchTool 详细用法

支持的操作类型：

| 操作 | 说明 | 示例 |
|------|------|------|
| `fetch` | 抓取完整网页 | `{"operation": "fetch", "url": "https://example.com"}` |
| `extract_text` | 提取纯文本 | `{"operation": "extract_text", "url": "https://example.com"}` |
| `extract_links` | 提取所有链接 | `{"operation": "extract_links", "url": "https://example.com"}` |
| `extract_images` | 提取图片链接 | `{"operation": "extract_images", "url": "https://example.com"}` |
| `metadata` | 提取元数据 | `{"operation": "metadata", "url": "https://example.com"}` |

### 使用工具

```rust
use langchainrust::tools::{Calculator, DateTimeTool, SimpleMathTool, URLFetchTool, BaseTool};
use std::sync::Arc;

// 创建工具列表
let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(Calculator::new()),
    Arc::new(DateTimeTool::new()),
    Arc::new(SimpleMathTool::new()),
    Arc::new(URLFetchTool::new()),
];

// 不传工具也可以工作（Agent 会直接回答）
let tools: Vec<Arc<dyn BaseTool>> = vec![];
```

### 直接调用工具

```rust
use langchainrust::tools::{SimpleMathTool, BaseTool};

let math = SimpleMathTool::new();

// 直接调用工具
let result = math.run(r#"{"operation": "power", "value": 2, "value2": 10}"#.to_string()).await?;
println!("2^10 = {}", result); // 输出: 2^10 = 1024
```

### 自定义 Tool

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

# 运行 retrieval 测试
cargo test --test retrieval_test -- --nocapture

# 运行任务规划测试
cargo test --test planner_test -- --nocapture

# 运行 RAG Agent 测试
cargo test --test retrieval_agent_test -- --nocapture

# 运行特定测试函数
cargo test test_full_planning_workflow -- --nocapture
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
│   ├── mod.rs      # 导出 Tool trait 和所有工具
│   ├── tool.rs     # Tool trait 定义
│   └── tools.rs    # 内置工具实现
├── agent/          # Agent 与执行器
│   ├── mod.rs      # Agent trait、AgentAction、AgentError
│   ├── executor.rs # AgentExecutor
│   ├── react/      # ReActAgent 模块
│   │   ├── mod.rs      # ReActAgent 结构体
│   │   ├── types.rs    # AnyLLM、RoutingState
│   │   ├── routing.rs  # 模型路由
│   │   ├── retrieval.rs # RAG 检索
│   │   ├── parser.rs   # 响应解析
│   │   └── agent_impl.rs # Agent trait 实现
│   └── planner/    # 任务规划模块
│       ├── mod.rs      # 模块导出
│       ├── types.rs    # Plan、SubTask、TaskResult
│       ├── planner.rs  # TaskPlanner
│       └── executor.rs # PlannedExecutor
└── retrieval/      # 检索组件
    ├── mod.rs      # 导出所有 retrieval 组件
    ├── traits.rs   # 核心 trait 定义
    ├── document.rs # 文档结构
    ├── text_splitters.rs # 文本分割器
    ├── embeddings.rs     # 嵌入模型
    ├── vector_stores.rs  # 向量存储
    └── retrievers.rs     # 检索器
```

---

## API 速查

### ReActAgent 构造方法

| 方法 | 参数 | 说明 |
|------|------|------|
| `new(llm, tools, memory)` | LLM、工具列表、可选内存 | 基础 Agent |
| `with_template(llm, tools, memory, template)` | 增加模板参数 | 带模板的 Agent |
| `with_models(llm, models, tools, memory, template)` | 增加模型列表参数 | 带模型路由的 Agent |
| `with_retriever(llm, tools, memory, retriever, top_k)` | 增加检索器 | 带 RAG 的 Agent |
| `with_retriever_and_template(...)` | 增加检索器和模板 | RAG + 自定义模板 |
| `with_all(...)` | 所有参数 | 完整功能 Agent |

### 工具调用

| 行为 | 说明 |
|------|------|
| 使用工具 | 模型输出 `[TOOL: 工具名 参数=值]` |
| 直接回答 | 模型直接输出答案文本 |

### 模型路由参数

| 参数名 | 别名 | 默认值 | 说明 |
|--------|------|--------|------|
| `difficulty` | `难度`、`level` | 1 | 问题难度（1-10） |

### RAG 执行流程

| 步骤 | 说明 |
|------|------|
| 1. 检索 | 从向量数据库检索 top_k 个相关文档 |
| 2. 构建上下文 | 将文档格式化为 prompt 上下文 |
| 3. 调用 LLM | 将问题和上下文一起发送给模型 |
| 4. 返回答案 | 模型基于上下文生成答案 |

### 任务规划 API

| 类型 | 方法/字段 | 说明 |
|------|----------|------|
| `TaskPlanner::new(llm)` | 构造 | 创建任务规划器 |
| `.with_max_sub_tasks(n)` | 配置 | 设置最大子任务数 |
| `.with_verbose(bool)` | 配置 | 是否打印日志（默认 false） |
| `.plan(question)` | 方法 | 分解任务，返回 Plan |
| `.summarize(question, results)` | 方法 | 汇总执行结果 |
| `PlannedExecutor::new(llm, agent, tools)` | 构造 | 创建规划执行器 |
| `.with_max_sub_tasks(n)` | 配置 | 设置最大子任务数 |
| `.with_max_iterations(n)` | 配置 | 设置每个子任务最大迭代次数 |
| `.with_memory(memory)` | 配置 | 设置记忆模块 |
| `.with_verbose(bool)` | 配置 | 是否打印日志（默认 false） |
| `.run(question)` | 方法 | 执行复杂任务 |
| `.run_with_plan(question)` | 方法 | 返回规划和详细结果 |
| `SimplePlannedExecutor::new(llm)` | 构造 | 创建简化版执行器 |
| `.with_verbose(bool)` | 配置 | 是否打印日志（默认 false） |

### 任务规划流程

| 步骤 | 说明 |
|------|------|
| 1. 规划 | LLM 将问题分解为子任务 |
| 2. 执行 | 依次执行每个子任务 |
| 3. 汇总 | 将所有结果汇总为最终答案 |

