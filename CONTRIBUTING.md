# 贡献指南

感谢您有兴趣为 LangChain Rust 做贡献！

## 如何贡献

### 报告 Bug

如果您发现了 bug，请在 [Issues](https://github.com/atliliw/langchainrust/issues) 中创建一个新 issue，包含：

1. 清晰的标题和描述
2. 重现步骤
3. 预期行为和实际行为
4. 环境信息（Rust 版本、操作系统等）

### 提交功能请求

欢迎提交功能请求！请在 issue 中描述：

1. 您想要的功能
2. 为什么这个功能有用
3. 可能的实现方式（如果有想法）

### 提交代码

#### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/atliliw/langchainrust.git
cd langchainrust

# 安装依赖并构建
cargo build

# 运行测试
cargo test

# 运行 clippy
cargo clippy

# 检查格式
cargo fmt --check
```

#### 代码规范

1. **格式化**: 使用 `cargo fmt` 自动格式化代码
2. **Lint**: 确保通过 `cargo clippy` 检查（无警告）
3. **测试**: 所有新功能必须包含测试
4. **文档**: 公共 API 必须有文档注释

#### 提交规范

使用清晰的提交信息：

```
<type>: <description>

[optional body]

[optional footer]
```

类型：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

示例：
```
feat: add Qwen LLM support

- Implement QwenChat client
- Add QwenConfig for configuration
- Include streaming support

Closes #123
```

#### Pull Request 流程

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

#### PR 检查清单

- [ ] 代码通过 `cargo fmt --check`
- [ ] 代码通过 `cargo clippy`（无警告）
- [ ] 所有测试通过 `cargo test`
- [ ] 添加了必要的文档
- [ ] 添加了必要的测试

### 代码风格

#### 命名规范

- **Struct/Enum**: PascalCase (如 `OpenAIChat`, `AgentError`)
- **函数/方法**: snake_case (如 `embed_query`, `add_documents`)
- **变量**: snake_case (如 `api_key`, `max_tokens`)
- **常量**: SCREAMING_SNAKE_CASE (如 `MAX_ITERATIONS`)

#### 文档注释

使用 `///` 进行文档注释：

```rust
/// 创建新的 ReActAgent
///
/// # 参数
/// * `llm` - LLM 客户端
/// * `tools` - 可用工具列表
/// * `system_prompt` - 自定义系统提示词
///
/// # 示例
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

#### 错误处理

使用 `Result<T, Box<dyn Error>>` 或自定义错误类型：

```rust
// 推荐：自定义错误类型
#[derive(Debug)]
pub enum MyError {
    InvalidInput(String),
    ExecutionFailed(String),
}

// 不推荐：过度使用 unwrap()
let value = some_option.unwrap(); // ❌

// 推荐：使用 ? 操作符
let value = some_option.ok_or(MyError::InvalidInput("missing value".into()))?; // ✅
```

### 项目结构

```
src/
├── core/           # 核心抽象（Runnable, Tool 等）
├── schema/         # 数据结构（Message 等）
├── language_models/ # LLM 实现（OpenAI 等）
├── tools/          # 内置工具
├── agents/         # Agent 实现
├── memory/         # Memory 实现
├── chains/         # Chain 实现
├── embeddings/     # Embedding 模型
├── vector_stores/  # 向量存储
└── retrieval/      # 检索组件

examples/           # 示例代码
tests/              # 集成测试
docs/               # 文档
```

### 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_name

# 运行带输出的测试
cargo test -- --nocapture

# 运行 ignored 测试（需要 API Key）
cargo test -- --ignored
```

### 获取帮助

如果您有任何问题，可以：

1. 查看 [文档](./docs/USAGE.md)
2. 查看 [示例](./examples/)
3. 在 [Discussions](https://github.com/atliliw/langchainrust/discussions) 中提问
4. 创建 [Issue](https://github.com/atliliw/langchainrust/issues)

## 许可证

通过贡献代码，您同意您的代码将在 MIT 或 Apache-2.0 双许可证下发布。