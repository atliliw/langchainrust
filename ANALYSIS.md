# Python Runnable 核心接口分析

## 从 Python 学习的关键点

### 1. Runnable 核心方法 (runnables/base.py)

```python
class Runnable(Generic[Input, Output], ABC):
    # 核心方法
    def invoke(input: Input, config: Optional[RunnableConfig] = None) -> Output
    def batch(inputs: List[Input], config: Optional[RunnableConfig] = None) -> List[Output]
    def stream(input: Input, config: Optional[RunnableConfig] = None) -> Iterator[Output]
    def pipe(other: Runnable) -> RunnableSequence  # 链式调用
```

### 2. RunnableConfig 结构 (runnables/config.py)

```python
class RunnableConfig(TypedDict, total=False):
    tags: list[str]              # 标签 (用于过滤)
    metadata: dict[str, Any]     # 元数据 (JSON 序列化)
    callbacks: Callbacks         # 回调系统
    run_name: str                # 运行名称
    max_concurrency: int | None  # 最大并发数
    recursion_limit: int         # 递归限制 (默认 25)
    configurable: dict[str, Any] # 可配置参数
    run_id: uuid.UUID | None     # 运行 ID
```

### 3. 关键设计原则

1. **泛型设计**: `Runnable[Input, Output]` 支持任意类型
2. **配置传递**: config 通过参数传递，支持继承和合并
3. **异步支持**: 所有方法都有 async 版本
4. **链式调用**: `pipe()` 方法实现组合

---

## Rust 实现策略

### 简化原则

由于 Python 版本非常复杂 (6261 行)，我们需要：

1. **保留核心功能**: invoke/batch/stream/pipe
2. **简化配置**: 只保留最常用的配置项
3. **类型安全**: 利用 Rust 类型系统
4. **异步优先**: Rust 的异步是原生的

### Rust 类型约束

- Rust 的 trait 不能像 Python 的 ABC 那样灵活
- 需要明确的生命周期和所有权
- async trait 需要使用 `async_trait` crate
- 泛型参数需要 Send + Sync 约束

---

## 实施步骤

1. 创建简化的 `RunnableConfig`
2. 创建核心 `Runnable` trait
3. 实现基本的 `pipe` 功能
4. 编写测试验证