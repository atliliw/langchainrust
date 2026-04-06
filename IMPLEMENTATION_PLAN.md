# LangChain Rust 实施计划

## 目标
从 LangChain Python 版本学习，用 Rust 完全重写，实现工业级 LLM 框架。

---

## Python 源代码分析

### 核心模块结构 (langchain/libs/core/langchain_core/)
```
核心层 (176 个 Python 文件):
├── runnables/        # LCEL 核心 - 最高优先级
├── language_models/  # LLM 抽象 - 最高优先级  
├── tools/            # 工具系统 - 高优先级
├── messages/         # 消息类型 - 高优先级
├── prompts/          # 提示词 - 中优先级
├── output_parsers/   # 输出解析 - 中优先级
├── callbacks/        # 回调系统 - 中优先级
├── documents/        # 文档抽象 - 低优先级
├── embeddings/       # 向量嵌入 - 低优先级
└── vectorstores/     # 向量存储 - 低优先级
```

---

## 实施策略：模块逐个攻克

### Phase 1: 核心抽象层 (Week 1-2)

#### 模块 1: runnables/ (LCEL 核心)
**学习资料**:
- `langchain/libs/core/langchain_core/runnables/base.py` (6261 行)
- `langchain/libs/core/langchain_core/runnables/config.py`
- `langchain/libs/core/langchain_core/runnables/utils.py`

**关键接口**:
```python
# Python 版本
class Runnable(Generic[Input, Output], ABC):
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output
    def batch(self, inputs: List[Input], config: Optional[RunnableConfig] = None) -> List[Output]
    def stream(self, input: Input, config: Optional[RunnableConfig] = None) -> Iterator[Output]
    def pipe(self, other: Runnable) -> RunnableSequence
```

**Rust 实现计划**:
- [ ] 创建 `src/core/runnables/mod.rs`
- [ ] 实现 `Runnable<Input, Output>` trait
- [ ] 实现 `RunnableConfig` struct
- [ ] 实现 `RunnableSequence` (链式调用)
- [ ] 编写测试用例

**工作量**: 2-3 天

---

#### 模块 2: messages/ (消息类型)
**学习资料**:
- `langchain/libs/core/langchain_core/messages/__init__.py`
- `langchain/libs/core/langchain_core/messages/human.py`
- `langchain/libs/core/langchain_core/messages/ai.py`
- `langchain/libs/core/langchain_core/messages/system.py`
- `langchain/libs/core/langchain_core/messages/tool.py`

**关键接口**:
```python
# Python 版本
class BaseMessage(Serializable):
    content: str | list[str | dict]
    type: Literal["system", "human", "ai", "tool"]
    additional_kwargs: dict[str, Any]

class HumanMessage(BaseMessage): ...
class AIMessage(BaseMessage): ...
class SystemMessage(BaseMessage): ...
class ToolMessage(BaseMessage):
    tool_call_id: str
```

**Rust 实现计划**:
- [ ] 创建 `src/schema/messages/mod.rs`
- [ ] 实现 `Message` enum (System/Human/AI/Tool)
- [ ] 实现 `MessageContent` (Text/MultiPart)
- [ ] 实现 `MessagePart` (Text/Image/ToolCall)
- [ ] 编写测试用例

**工作量**: 1-2 天

---

#### 模块 3: language_models/ (LLM 抽象)
**学习资料**:
- `langchain/libs/core/langchain_core/language_models/base.py` (391 行)
- `langchain/libs/core/langchain_core/language_models/chat_models.py`
- `langchain/libs/core/langchain_core/language_models/llms.py`

**关键接口**:
```python
# Python 版本
class BaseLanguageModel(Runnable[LanguageModelInput, LanguageModelOutput]):
    def get_num_tokens(self, text: str) -> int
    @property
    def model_name(self) -> str

class BaseChatModel(BaseLanguageModel):
    def generate(self, messages: List[BaseMessage]) -> LLMResult
    def stream(self, messages: List[BaseMessage]) -> Iterator[ChatGenerationChunk]

class BaseLLM(BaseLanguageModel):
    def generate(self, prompts: List[str]) -> LLMResult
    def stream(self, prompt: str) -> Iterator[str]
```

**Rust 实现计划**:
- [ ] 创建 `src/core/language_models/mod.rs`
- [ ] 实现 `BaseLanguageModel` trait
- [ ] 实现 `BaseChatModel` trait
- [ ] 实现 `BaseLLM` trait
- [ ] 编写测试用例

**工作量**: 2-3 天

---

#### 模块 4: tools/ (工具系统)
**学习资料**:
- `langchain/libs/core/langchain_core/tools/__init__.py`
- `langchain/libs/core/langchain_core/tools/base.py`
- `langchain/libs/core/langchain_core/tools/structured.py`

**关键接口**:
```python
# Python 版本
class BaseTool(Runnable[ToolInput, ToolOutput]):
    name: str
    description: str
    args_schema: Type[BaseModel]  # Pydantic model
    
    def invoke(self, input: dict) -> Any
    def _run(self, *args, **kwargs) -> Any
```

**Rust 实现计划**:
- [ ] 创建 `src/core/tools/mod.rs`
- [ ] 实现 `Tool` trait with JSON Schema
- [ ] 使用 `schemars` crate 生成 schema
- [ ] 实现工具注册表
- [ ] 编写测试用例

**工作量**: 2 天

---

### Phase 2: LLM 实现 (Week 3-4)

#### 模块 5: OpenAI 实现
**学习资料**:
- `langchain/libs/partners/openai/langchain_openai/chat_models/base.py`
- `langchain/libs/partners/openai/langchain_openai/llms/base.py`

**关键功能**:
- API 调用 (非流式)
- 流式响应 (SSE)
- Token 计数
- Function Calling

**Rust 实现计划**:
- [ ] 创建 `src/language_models/openai/mod.rs`
- [ ] 实现 `OpenAIChat` struct
- [ ] 实现 HTTP 客户端 (reqwest)
- [ ] 实现 SSE 流式解析
- [ ] 编写集成测试

**工作量**: 3-4 天

---

### Phase 3: Agents (Week 5-6)

#### 模块 6: agents/ (Agent 系统)
**学习资料**:
- `langchain/libs/langchain/langchain/agents/agent.py`
- `langchain/libs/langchain/langchain/agents/openai_functions_agent/base.py`

**Rust 实现计划**:
- [ ] 实现 `AgentExecutor`
- [ ] 实现 `ReActAgent`
- [ ] 实现工具调用逻辑
- [ ] 编写测试用例

**工作量**: 3-4 天

---

## 执行流程

### 对于每个模块：

1. **学习阶段** (1-2 小时)
   - 阅读对应的 Python 代码
   - 理解核心接口和数据结构
   - 标注关键功能点

2. **设计阶段** (1-2 小时)
   - 设计 Rust 等价接口
   - 考虑 Rust 类型系统的约束
   - 绘制模块依赖图

3. **实现阶段** (数小时到数天)
   - 编写 Rust 代码
   - 处理编译错误
   - 实现核心功能

4. **测试阶段** (1-2 小时)
   - 编写单元测试
   - 编写集成测试
   - 验证与 Python 版本的兼容性

---

## 每日工作流程

### 第一步：选择模块
选择当前优先级最高的模块。

### 第二步：学习 Python 代码
```bash
# 阅读对应的 Python 文件
cat langchain/libs/core/langchain_core/runnables/base.py | less
```

### 第三步：提取关键信息
- 接口定义
- 数据结构
- 核心方法
- 错误处理

### 第四步：Rust 实现
创建对应的 Rust 文件并实现。

### 第五步：测试验证
```bash
cargo test
```

---

## 下一步行动

1. ✅ 清理现有代码
2. 🔄 分析 Python `runnables/base.py`
3. ⏳ 实现 Rust `Runnable` trait
4. ⏳ 测试验证
5. ⏳ 进入下一个模块

---

## 成功标准

每个模块完成后必须满足：
- [ ] 编译通过 (cargo check)
- [ ] 测试通过 (cargo test)
- [ ] 文档完整 (cargo doc)
- [ ] 功能与 Python 版本对等

