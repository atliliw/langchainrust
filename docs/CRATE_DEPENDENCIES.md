# Crate 依赖关系与发布流程

> 每次发版前查阅此文件，按依赖层级从底到顶依次 `cargo publish`。

---

## 一、发版前检查清单

发版前必须按顺序完成以下步骤，**全部通过后才能执行发布**。

### 1. 编译 & 测试

```bash
# 全 workspace 编译
cargo check --workspace

# 全 workspace 测试
cargo test --workspace

# 如果有集成测试（如 #[tool] 宏）
cargo test -p lc-tools --test tool_macro
```

**通过标准**：0 error，0 failed test。

### 2. Clippy 检查

```bash
cargo clippy --workspace -- -D warnings
```

**通过标准**：0 warning。如果有 warning，先修再发。

### 3. 版本号统一更新

所有 crate 的 `version` 必须统一更新为新版本号。包括：
- 每个 crate 的 `[package] version`
- 每个 crate 内部依赖的 `lc-* = { version = "x.y.z" }` 引用

```bash
# 示例：从 0.9.0 升到 0.10.0
# 一条命令搞定（package version 和依赖 version 格式一致，一次替换即可）
find . -name "Cargo.toml" -exec sed -i 's/version = "0.9.0"/version = "0.10.0"/g' {} +
```

> ⚠️ 替换后务必 `cargo check --workspace` 确认没有遗漏。

### 4. 文档更新

发版前必须更新以下文档：

| 文档 | 更新内容 | 何时更新 |
|------|---------|---------|
| `CHANGELOG.md` | 新增版本条目（Added / Changed / Fixed / Deprecated） | **每次发版必改** |
| `README.md` | Core Features 表、架构图、安装版本号 | 有新 feature 时 |
| `docs/USAGE.md` | 新功能用法示例、目录新增条目 | 有新 feature 时 |
| `docs/USAGE_EN.md` | 同上英文版 | 有新 feature 时 |
| `docs/CRATE_DEPENDENCIES.md` | 新增/删除 crate、依赖关系变化 | crate 结构变化时 |
| `docs/internal/vX.Y.Z/EXECUTION_PLAN.md` | 标记已完成项、记录推迟项 | 版本开发过程中 |

**检查方法**：

```bash
# 确认 CHANGELOG 有新版本条目
head -20 CHANGELOG.md

# 确认 README 安装版本号已更新
grep 'langchainrust = "' README.md

# 确认 USAGE 目录包含新功能
grep -n "v0.10.0" docs/USAGE.md docs/USAGE_EN.md
```

### 5. Git 提交 & 合并

```bash
# 确认所有改动已提交
git status

# 合并到 main（如果还在 feature 分支）
git checkout main
git merge feat/vX.Y.Z-xxx

# 推送到远程
git push origin main
```

### 6. 打 Tag

```bash
git tag v0.10.0
git push origin v0.10.0
```

---

## 二、依赖关系图

```
lc-shared ─────────────────────────────────────────────────────────────────────┐
lc-schema ──(lc-shared)───────────────────────────────────────────────────────┤
lc-callbacks ──(lc-schema)────────────────────────────────────────────────────┤
lc-core ──(lc-shared, lc-schema, lc-callbacks)───────────────────────────────┤
lc-prompts ──(lc-schema)──────────────────────────────────────────────────────┤
lc-embeddings ──(lc-core)─────────────────────────────────────────────────────┤
lc-vector-stores ──(lc-shared, lc-core, lc-embeddings)────────────────────────┤
lc-providers ──(lc-core, lc-schema, lc-callbacks, lc-shared)─────────────────┤
lc-tools-derive ──(无内部依赖)──────────────────────────────────────────────────┤
lc-tools ──(lc-core, lc-tools-derive)──────────────────────────────────────────┤
lc-rag ──(lc-shared, lc-core, lc-schema, lc-embeddings, lc-vector-stores, lc-prompts, lc-providers)──┤
lc-memory ──(lc-shared, lc-schema, lc-core, lc-prompts, [lc-embeddings], [lc-vector-stores], [lc-providers])──┤
lc-langgraph ──(无内部依赖)───────────────────────────────────────────────────┤
lc-sessions ──(lc-core, lc-schema)────────────────────────────────────────────┤
lc-mcp ──(lc-core)────────────────────────────────────────────────────────────┤
lc-evaluation ──(lc-core, lc-schema, lc-embeddings, lc-providers)─────────────┤
lc-chains ──(lc-core, lc-schema, lc-shared, lc-memory, lc-rag)───────────────┤
lc-agents ──(lc-core, lc-schema, lc-shared, lc-callbacks, lc-memory, lc-rag, lc-providers, lc-tools, lc-embeddings, lc-vector-stores, lc-prompts)──┤
lc-guardrails ──(lc-agents, lc-core)──────────────────────────────────────────┤
lc-a2a ──(lc-chains)──────────────────────────────────────────────────────────┤
langchainrust ──(全部 20 个 lc-* crate)──────────────────────────────────────┘
```

## 逐 Crate 依赖明细

| Crate | 内部依赖 | 路径 |
|-------|---------|------|
| **lc-shared** | *(无)* | `crates/lc-shared` |
| **lc-schema** | lc-shared | `crates/lc-schema` |
| **lc-callbacks** | lc-schema | `crates/lc-callbacks` |
| **lc-core** | lc-shared, lc-schema, lc-callbacks | `crates/lc-core` |
| **lc-prompts** | lc-schema | `crates/lc-prompts` |
| **lc-embeddings** | lc-core | `crates/lc-embeddings` |
| **lc-vector-stores** | lc-shared, lc-core, lc-embeddings | `crates/lc-vector-stores` |
| **lc-providers** | lc-core, lc-schema, lc-callbacks, lc-shared | `crates/lc-providers` |
| **lc-tools** | lc-core, lc-tools-derive | `crates/lc-tools` |
| **lc-tools-derive** | *(无)* | `crates/lc-tools-derive` |
| **lc-rag** | lc-shared, lc-core, lc-schema, lc-embeddings, lc-vector-stores, lc-prompts, lc-providers | `crates/lc-rag` |
| **lc-memory** | lc-shared, lc-schema, lc-core, lc-prompts, *(可选: lc-embeddings, lc-vector-stores, lc-providers)* | `crates/lc-memory` |
| **lc-langgraph** | *(无内部依赖)* | `crates/lc-langgraph` |
| **lc-sessions** | lc-core, lc-schema | `crates/lc-sessions` |
| **lc-mcp** | lc-core | `crates/lc-mcp` |
| **lc-evaluation** | lc-core, lc-schema, lc-embeddings, lc-providers | `crates/lc-evaluation` |
| **lc-chains** | lc-core, lc-schema, lc-shared, lc-memory, lc-rag | `crates/lc-chains` |
| **lc-agents** | lc-core, lc-schema, lc-shared, lc-callbacks, lc-memory, lc-rag, lc-providers, lc-tools, lc-embeddings, lc-vector-stores, lc-prompts | `crates/lc-agents` |
| **lc-guardrails** | lc-agents, lc-core | `crates/lc-guardrails` |
| **lc-a2a** | lc-chains | `crates/lc-a2a` |
| **langchainrust** | 全部 20 个 lc-* crate | `crates/lc` |

---

## 三、发布顺序（拓扑排序）

按依赖层级从底到顶，同一层内可并行发布：

### 第 1 层 — 无内部依赖
```
cargo publish -p lc-shared
cargo publish -p lc-langgraph
```

### 第 2 层 — 仅依赖第 1 层
```
cargo publish -p lc-schema
```

### 第 3 层 — 依赖第 1-2 层
```
cargo publish -p lc-callbacks
cargo publish -p lc-prompts
```

### 第 4 层 — 依赖第 1-3 层
```
cargo publish -p lc-core
```

### 第 5 层 — 依赖第 1-4 层
```
cargo publish -p lc-embeddings
cargo publish -p lc-tools-derive
cargo publish -p lc-tools
cargo publish -p lc-sessions
cargo publish -p lc-mcp
```

### 第 6 层 — 依赖第 1-5 层
```
cargo publish -p lc-vector-stores
cargo publish -p lc-providers
```

### 第 7 层 — 依赖第 1-6 层
```
cargo publish -p lc-rag
cargo publish -p lc-memory
cargo publish -p lc-evaluation
```

### 第 8 层 — 依赖第 1-7 层
```
cargo publish -p lc-chains
```

### 第 9 层 — 依赖第 1-8 层
```
cargo publish -p lc-agents
```

### 第 10 层 — 依赖第 1-9 层
```
cargo publish -p lc-guardrails
cargo publish -p lc-a2a
```

### 第 11 层 — Facade（依赖全部）
```
cargo publish -p langchainrust
```

---

## 四、一键发布脚本

> ⚠️ 使用 USTC 镜像时需加 `--registry crates-io`，或临时关闭镜像（见下方说明）。

```bash
# 按顺序逐个发布，失败自动重试
for crate in \
  lc-shared lc-langgraph \
  lc-schema \
  lc-callbacks lc-prompts \
  lc-core \
  lc-embeddings lc-tools-derive lc-tools lc-sessions lc-mcp \
  lc-vector-stores lc-providers \
  lc-rag lc-memory lc-evaluation \
  lc-chains \
  lc-agents \
  lc-guardrails lc-a2a \
  langchainrust; do
  echo "=== Publishing $crate ==="
  cargo publish -p "$crate" --registry crates-io
  if [ $? -ne 0 ]; then
    echo "!!! $crate failed, waiting 30s and retrying..."
    sleep 30
    cargo publish -p "$crate" --registry crates-io
  fi
  sleep 5
done
echo "=== All crates published! ==="
```

---

## 五、镜像说明

当前 `~/.cargo/config.toml` 配置了 USTC 镜像：

```toml
[source.crates-io]
replace-with = "ustc"

[source.ustc]
registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"
```

**发布时注意：**
- 镜像同步有延迟，刚发布的新版本镜像可能还没同步
- 方案 A：加 `--registry crates-io` 绕过镜像直接访问 crates.io
- 方案 B：临时注释掉 `replace-with = "ustc"`，发布完再恢复
- 发布完成后记得恢复镜像配置

---

## 六、完整发版流程（Step by Step）

以下是从开发完成到发布上线的完整流程，按顺序执行：

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: 代码验证                                        │
│  □ cargo check --workspace                               │
│  □ cargo test --workspace                                │
│  □ cargo clippy --workspace -- -D warnings               │
└──────────────────────┬──────────────────────────────────┘
                       │ 全部通过
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: 版本号更新                                       │
│  □ find . -name "Cargo.toml" -exec sed -i ...           │
│  □ cargo check --workspace (确认版本号替换无误)            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: 文档更新                                        │
│  □ CHANGELOG.md — 新版本条目                              │
│  □ README.md — Features 表 + 架构图 + 安装版本号          │
│  □ docs/USAGE.md — 新功能示例 + 目录                      │
│  □ docs/USAGE_EN.md — 同上英文版                          │
│  □ docs/CRATE_DEPENDENCIES.md — crate 结构变化时          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: Git 提交 & 合并                                  │
│  □ git add -A && git commit                              │
│  □ git checkout main && git merge feat/vX.Y.Z-xxx       │
│  □ git push origin main                                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5: 打 Tag                                          │
│  □ git tag vX.Y.Z                                        │
│  □ git push origin vX.Y.Z                                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Step 6: 发布到 crates.io                                │
│  □ 按拓扑顺序逐个 cargo publish（见第三节）                │
│  □ 或使用一键发布脚本（见第四节）                           │
│  □ 发布后验证: docs.rs 上能搜到新版本                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Step 7: 发布后验证                                       │
│  □ cargo install langchainrust --version X.Y.Z           │
│  □ 新项目 cargo add langchainrust 编译通过                 │
│  □ docs.rs 文档已更新                                     │
│  □ crates.io 页面已更新                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 七、常见问题

### Q: cargo publish 报 "already exists"
该版本已发布过，需要升版本号。crates.io 不允许覆盖已发布版本。

### Q: cargo publish 报依赖找不到
上游 crate 还没发布到 crates.io。检查是否按拓扑顺序发布，确保依赖的 crate 先发布。

### Q: 镜像同步延迟导致下游 crate 找不到上游
等 1-2 分钟后重试，或加 `--registry crates-io` 绕过镜像。

### Q: 版本号替换后 cargo check 报错
可能是 sed 替换了不该替换的内容（如依赖的第三方 crate 版本号恰好相同）。检查 `git diff` 确认替换范围正确。
