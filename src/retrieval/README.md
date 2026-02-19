# Retrieval 模块文档

本文档详细介绍了 `langchainrust` 中 Retrieval（检索）模块的设计原理和使用方法。

## 目录

- [概述](#概述)
- [核心概念](#核心概念)
- [架构设计](#架构设计)
- [组件详解](#组件详解)
  - [Document 和 DocumentChunk](#document-和-documentchunk)
  - [TextSplitter（文本分割器）](#textsplitter文本分割器)
  - [EmbeddingModel（嵌入模型）](#embeddingmodel嵌入模型)
  - [VectorStore（向量存储）](#vectorstore向量存储)
  - [Retriever（检索器）](#retriever检索器)
- [工作流程](#工作流程)
- [使用示例](#使用示例)
- [实现原理](#实现原理)

---

## 概述

Retrieval 模块实现了 **RAG（Retrieval-Augmented Generation，检索增强生成）** 的核心功能。它的主要作用是：

1. **文档处理**：将长文档分割成适合检索的小块
2. **向量化**：将文本转换为向量表示
3. **存储检索**：基于语义相似度检索相关文档

这使得 LLM 能够利用外部知识库，突破上下文长度限制，提供更准确、更有依据的回答。

---

## 核心概念

### 什么是向量检索？

```
┌─────────────┐      嵌入模型      ┌─────────────┐
│   查询文本   │  ───────────────>  │  查询向量   │
└─────────────┘                    └─────────────┘
                                          │
                                          ▼ 相似度计算
┌─────────────┐                    ┌─────────────┐
│  文档块1    │  ───────────────>  │  向量1      │
├─────────────┤                    ├─────────────┤
│  文档块2    │  ───────────────>  │  向量2      │ 
├─────────────┤                    ├─────────────┤
│  文档块3    │  ───────────────>  │  向量3      │
└─────────────┘                    └─────────────┘
     存储                              向量数据库
```

向量检索的核心思想：
1. 将文本转换为高维向量（嵌入）
2. 相似的内容在向量空间中距离相近
3. 通过计算向量相似度找到最相关的文档

### 余弦相似度

我们使用余弦相似度来衡量两个向量的相似程度：

```
cos(A, B) = (A · B) / (||A|| * ||B||)

其中：
- A · B 是向量点积
- ||A|| 是向量的模（长度）
- 结果范围：[-1, 1]，1表示完全相同，0表示正交（无关），-1表示完全相反
```

---

## 架构设计

```
┌────────────────────────────────────────────────────────────────┐
│                        Retrieval 模块                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   Document   │───>│ TextSplitter │───>│   Chunk      │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                │               │
│                                                ▼               │
│  ┌──────────────┐                      ┌──────────────┐       │
│  │EmbeddingModel│<─────────────────────│VectorStore   │       │
│  └──────────────┘                      └──────────────┘       │
│         │                                     │               │
│         │                                     ▼               │
│         │                             ┌──────────────┐       │
│         └────────────────────────────>│  Retriever   │       │
│                                       └──────────────┘       │
│                                             │                 │
└─────────────────────────────────────────────┼─────────────────┘
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │ SearchResult │
                                       └──────────────┘
```

### 模块职责

| 组件 | 职责 | 接口 |
|------|------|------|
| `Document` | 原始文档容器 | 结构体 |
| `DocumentChunk` | 文档分块 | 结构体 |
| `TextSplitter` | 文本分割 | `split_document`, `split_text` |
| `EmbeddingModel` | 文本向量化 | `embed`, `embed_batch` |
| `VectorStore` | 向量存储与搜索 | `add_documents`, `similarity_search` |
| `Retriever` | 检索封装 | `retrieve`, `retrieve_with_filter` |

---

## 组件详解

### Document 和 DocumentChunk

```rust
// 原始文档
pub struct Document {
    pub content: String,                    // 文档内容
    pub metadata: HashMap<String, String>,  // 元数据（来源、作者等）
}

// 文档块（分割后的）
pub struct DocumentChunk {
    pub content: String,                    // 块内容
    pub metadata: HashMap<String, String>,  // 继承的元数据
    pub chunk_index: usize,                 // 块索引
    pub document_id: Option<String>,        // 原文档ID
}
```

**使用示例：**

```rust
use langchainrust::retrieval::{Document, DocumentChunk};

// 创建文档
let doc = Document::new("这是一段很长的文档...".to_string())
    .with_metadata("source".to_string(), "wiki".to_string())
    .with_metadata("author".to_string(), "张三".to_string());

// 创建文档块
let chunk = DocumentChunk::new("分块内容".to_string(), 0)
    .with_metadata("source".to_string(), "wiki".to_string())
    .with_document_id("doc_001".to_string());
```

### TextSplitter（文本分割器）

文本分割器负责将长文档分割成适合检索的小块。分割策略对检索质量有重要影响。

#### 三种分割器

| 分割器 | 策略 | 适用场景 |
|--------|------|----------|
| `FixedSizeSplitter` | 按字符数分割 | 简单场景、固定长度需求 |
| `RecursiveCharacterSplitter` | 递归按分隔符分割 | 通用场景、保持语义 |
| `RegexSplitter` | 正则表达式分割 | 自定义分割规则 |

**分割原理：**

```
原始文档：
"这是第一段。\n\n这是第二段，比较长。\n\n这是第三段。"

RecursiveCharacterSplitter（chunk_size=50）：
1. 首先尝试按 "\n\n" 分割
2. 如果单段超过 50 字符，继续按 "\n" 分割
3. 如果还超，按 ". " 分割
4. 最后按字符分割

结果：
- "这是第一段。"
- "这是第二段，比较长。"
- "这是第三段。"
```

**使用示例：**

```rust
use langchainrust::retrieval::{RecursiveCharacterSplitter, FixedSizeSplitter, TextSplitter, Document};

// 递归字符分割器
let splitter = RecursiveCharacterSplitter::new(100, 20);  // chunk_size=100, overlap=20
let doc = Document::new("很长的文档...".to_string());
let chunks = splitter.split_document(&doc)?;

// 固定大小分割器
let fixed_splitter = FixedSizeSplitter::new(50, 10);
let text_chunks = fixed_splitter.split_text("文本内容")?;
```

#### 重叠（Overlap）的作用

```
无重叠：
["这是第一块内容", "第二块内容继续", "第三块内容"]

有重叠（overlap=5）：
["这是第一块内容", "内容第二块继续", "继续第三块内容"]

重叠的好处：
- 避免关键信息被截断
- 保持上下文连贯性
- 提高检索召回率
```

### EmbeddingModel（嵌入模型）

嵌入模型将文本转换为高维向量，是语义检索的核心。

**接口定义：**

```rust
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>>;
    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, Box<dyn Error>>;
    fn embedding_dimension(&self) -> usize;
}
```

**MockEmbeddingModel 实现原理：**

当前实现的 `MockEmbeddingModel` 使用哈希函数生成确定性向量：

```
1. 对文本进行哈希
2. 使用线性同余生成器产生伪随机数
3. 归一化向量（单位长度）

注意：这是测试用的 Mock 实现，
实际应用应使用 OpenAI Embeddings、BGE 等真实模型。
```

### VectorStore（向量存储）

向量存储负责存储文档向量并执行相似度搜索。

**接口定义：**

```rust
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add_documents(&mut self, documents: Vec<(DocumentChunk, Vec<f32>)>) -> Result<(), Box<dyn Error>>;
    async fn similarity_search(&self, query: Vec<f32>, k: usize) -> Result<Vec<(DocumentChunk, f32)>, Box<dyn Error>>;
    async fn delete_documents(&mut self, ids: Vec<String>) -> Result<(), Box<dyn Error>>;
}
```

**InMemoryVectorStore 实现：**

```rust
pub struct InMemoryVectorStore {
    vectors: HashMap<String, (DocumentChunk, Vec<f32>)>,
}

// 相似度计算
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}
```

**搜索流程：**

```
1. 接收查询向量
2. 计算与所有存储向量的余弦相似度
3. 按相似度降序排序
4. 返回 Top-K 个结果
```

### Retriever（检索器）

检索器是高层封装，整合嵌入模型和向量存储。

**工作流程：**

```
┌─────────────┐
│   查询文本   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ EmbeddingModel  │  文本 -> 向量
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  VectorStore    │  相似度搜索
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ SearchResult    │  返回结果
└─────────────────┘
```

---

## 工作流程

### 完整的 RAG 检索流程

```rust
use langchainrust::retrieval::{
    Document, RecursiveCharacterSplitter, MockEmbeddingModel,
    InMemoryVectorStore, SimilarityRetriever, Retriever, TextSplitter,
};
use std::sync::Arc;

// 1. 准备文档
let doc = Document::new("很长的文档内容...".to_string());

// 2. 分割文档
let splitter = RecursiveCharacterSplitter::new(100, 20);
let chunks = splitter.split_document(&doc)?;

// 3. 创建嵌入模型和向量存储
let embedding = Arc::new(MockEmbeddingModel::new(128));
let store = Box::new(InMemoryVectorStore::new());

// 4. 创建检索器并添加文档
let retriever = SimilarityRetriever::new(store, embedding);
retriever.add_documents(chunks).await?;

// 5. 检索
let results = retriever.retrieve("查询内容", 5).await?;
for result in results {
    println!("内容: {}", result.chunk.content);
    println!("相似度: {}", result.score);
}
```

---

## 使用示例

### 示例1：文档分割

```rust
use langchainrust::retrieval::{Document, RecursiveCharacterSplitter, TextSplitter};

let doc = Document::new(
    "Rust是一种系统编程语言。\n\n\
     它注重内存安全。\n\n\
     Python是一种脚本语言。".to_string()
);

let splitter = RecursiveCharacterSplitter::new(30, 5);
let chunks = splitter.split_document(&doc)?;

for chunk in chunks {
    println!("块 {}: {}", chunk.chunk_index, chunk.content);
}
```

### 示例2：向量存储

```rust
use langchainrust::retrieval::{DocumentChunk, InMemoryVectorStore, VectorStore};

let mut store = InMemoryVectorStore::new();

// 添加文档
let chunks = vec![
    (DocumentChunk::new("文档1".to_string(), 0), vec![0.1; 64]),
    (DocumentChunk::new("文档2".to_string(), 1), vec![0.2; 64]),
];
store.add_documents(chunks).await?;

// 搜索
let query = vec![0.15; 64];
let results = store.similarity_search(query, 2).await?;
```

### 示例3：带过滤的检索

```rust
use langchainrust::retrieval::{DocumentChunk, SimilarityRetriever, Retriever};
use std::collections::HashMap;

// 添加带元数据的文档
let chunks = vec![
    DocumentChunk::new("Rust教程".to_string(), 0)
        .with_metadata("category".to_string(), "programming".to_string()),
    DocumentChunk::new("苹果介绍".to_string(), 1)
        .with_metadata("category".to_string(), "fruit".to_string()),
];

// 只检索编程类文档
let mut filter = HashMap::new();
filter.insert("category".to_string(), "programming".to_string());

let results = retriever.retrieve_with_filter("教程", 5, filter).await?;
```

---

## 实现原理

### 为什么使用 Trait？

```rust
// 使用 Trait 可以灵活切换实现
pub trait EmbeddingModel: Send + Sync { ... }
pub trait VectorStore: Send + Sync { ... }
pub trait Retriever: Send + Sync { ... }
```

好处：
- **可替换性**：可以轻松切换不同的嵌入模型（OpenAI、BGE、本地模型）
- **可测试性**：使用 Mock 实现进行单元测试
- **可扩展性**：添加新实现不影响现有代码

### 向量维度选择

| 模型 | 维度 | 说明 |
|------|------|------|
| OpenAI text-embedding-ada-002 | 1536 | 高质量，但维度较大 |
| BGE-small | 384 | 轻量级中文模型 |
| MockEmbeddingModel | 可配置 | 测试用 |

### 性能考虑

1. **批量嵌入**：使用 `embed_batch` 批量处理，减少 API 调用
2. **异步设计**：所有 I/O 操作都是异步的
3. **内存存储**：`InMemoryVectorStore` 适合小规模数据，大规模应使用专业向量数据库

### 扩展建议

```rust
// 1. 实现真实嵌入模型
struct OpenAIEmbedding { ... }
impl EmbeddingModel for OpenAIEmbedding { ... }

// 2. 实现持久化向量存储
struct QdrantStore { ... }
impl VectorStore for QdrantStore { ... }

// 3. 实现重排序器
struct CrossEncoderReranker { ... }
impl Reranker for CrossEncoderReranker { ... }
```

---

## 总结

Retrieval 模块提供了 RAG 应用所需的核心组件：

1. **文档处理**：`Document`, `DocumentChunk`, `TextSplitter`
2. **向量化**：`EmbeddingModel` trait
3. **存储检索**：`VectorStore`, `Retriever`

通过 Trait 抽象，可以灵活替换底层实现，适应不同场景需求。
