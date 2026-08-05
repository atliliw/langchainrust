# Crate 依赖关系与发布顺序

> 每次发版前查阅此文件，按依赖层级从底到顶依次 `cargo publish`。

## 依赖关系图

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
| **langchainrust** | 全部 19 个 lc-* crate | `crates/lc` |

## 发布顺序（拓扑排序）

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

## 一键发布脚本

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

## 镜像说明

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

## 版本号更新

发版前需统一更新所有 crate 的 `version` 字段：

```bash
# 示例：从 0.9.0 升到 0.10.0
find . -name "Cargo.toml" -exec sed -i 's/version = "0.9.0"/version = "0.10.0"/g' {} +
```

同时更新各 Cargo.toml 中的内部依赖版本号：
```bash
# 更新 lc-* 依赖的 version 引用
find . -name "Cargo.toml" -exec sed -i 's/version = "0.9.0"/version = "0.10.0"/g' {} +
```

> 两条 sed 其实是同一条命令，因为内部依赖的 `version = "0.9.0"` 和 package 的 `version = "0.9.0"` 格式一致，一次替换即可。
