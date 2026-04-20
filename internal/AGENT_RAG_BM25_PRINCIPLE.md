# LangChainRust: Agent + RAG + BM25 高效数据检索原理

> 本文档详细讲解 LangChainRust 中 Agent 如何结合 RAG 和 BM25 实现高效数据查找

---

## 目录

1. [概述](#概述)
2. [BM25 关键词检索原理](#bm25-关键词检索原理)
3. [RAG 语义检索原理](#rag-语义检索原理)
4. [回表机制详解](#回表机制详解)
5. [Hybrid 混合检索与 RRF 融合](#hybrid-混合检索与-rrf-融合)
6. [Agent + RAG + BM25 工作流程](#agent--rag--bm25-工作流程)
7. [性能优化建议](#性能优化建议)

---

## 概述

### 为什么需要混合检索？

| 检索方式 | 优势 | 劣势 |
|---------|------|------|
| **BM25** | 关键词精确匹配、专业术语搜索、无需 Embedding | 无语义理解、同义词无法匹配 |
| **向量检索** | 语义相似度、同义词匹配、意图理解 | 专业术语效果差、依赖 Embedding 质量 |
| **Hybrid** | 结合两者优势、召回率更高 | 计算复杂度增加 |

**核心结论**：Hybrid 检索召回率比单一检索高 20-30%。

---

## BM25 关键词检索原理

### BM25 公式

BM25 是 TF-IDF 的改进版本，核心公式：

```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))
```

### 参数详解

| 参数 | 默认值 | 作用 |
|------|--------|------|
| **k1** | 1.5 | 词频饱和参数，控制高频词的影响上限 |
| **b** | 0.75 | 文档长度归一化，惩罚过长/过短文档 |

### IDF 计算

```rust
// src/retrieval/bm25/algorithm.rs
pub fn compute_idf(n: usize, total_docs: usize) -> f64 {
    if n == 0 || total_docs == 0 {
        return 0.0;
    }
    
    let numerator = total_docs as f64 - n as f64 + 0.5;
    let denominator = n as f64 + 0.5;
    
    (numerator / denominator + 1.0).ln()
}
```

**IDF 含义**：
- 出现在所有文档的词（如"是"、"的"）→ IDF 低 → 对评分贡献小
- 只出现在少数文档的词（如"Rust"、"BM25"）→ IDF 高 → 对评分贡献大

### BM25 评分计算

```rust
// src/retrieval/bm25/algorithm.rs
pub fn bm25_score(
    query_terms: &[String],
    doc_term_freqs: &HashMap<String, usize>,
    doc_length: usize,
    avgdl: f64,
    idf_values: &HashMap<String, f64>,
    params: &BM25Params,
) -> f64 {
    let mut score = 0.0;
    
    for term in query_terms {
        let idf = idf_values.get(term).copied().unwrap_or(0.0);
        let tf = doc_term_freqs.get(term).copied().unwrap_or(0);
        
        // TF 归一化部分（核心创新）
        let dl_ratio = doc_length as f64 / avgdl;
        let tf_component = (tf as f64 * (params.k1 + 1.0))
            / (tf as f64 + params.k1 * (1.0 - params.b + params.b * dl_ratio));
        
        score += idf * tf_component;
    }
    
    score
}
```

### k1 和 b 的调优建议

| 场景 | k1 | b | 原因 |
|------|-----|-----|------|
| **短文档搜索** | 1.2 | 0.5 | 减少长度惩罚 |
| **长文档搜索** | 2.0 | 0.85 | 增加长度惩罚 |
| **专业术语搜索** | 1.5 | 0.75 | 默认值即可 |

---

## RAG 语义检索原理

### 向量相似度计算

```rust
// src/retrieval/unified_hybrid.rs
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
```

### 向量检索流程

```
用户查询 → Embedding 模型 → 查询向量 → 计算所有文档相似度 → Top-K 结果
```

**关键代码**：

```rust
// src/retrieval/unified_hybrid.rs
async fn vector_search(&self, query: &str) -> Result<Vec<Document>, VectorStoreError> {
    // 1. 查询向量化
    let query_embedding = self.embeddings.embed_query(query).await?;
    
    // 2. 计算相似度
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
            let score = Self::cosine_similarity(&query_embedding, &entry.embedding);
            (idx, score)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect();
    
    // 3. 排序取 Top-K
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k_indices: Vec<(usize, f32)> = scored.into_iter().take(self.config.vector_k).collect();
    
    // 4. 回表获取内容
    let mut docs = Vec::new();
    for (idx, _score) in top_k_indices {
        let entry = &vectors[idx];
        if let Some(chunk) = self.document_store.get_chunk(&entry.chunk_id).await? {
            docs.push(Document::new(chunk.content));
        }
    }
    
    Ok(docs)
}
```

---

## 回表机制详解

### 什么是回表？

**回表** = 索引只存储 ID/统计信息，实际内容需要额外查询获取。

### 为什么需要回表？

| 不回表 | 回表 |
|--------|------|
| 索引存内容 → 索引体积大 | 索引只存 ID → 索引体积小 |
| 内容重复存储 | 内容共享（BM25 + 向量共用） |
| 无法 AutoMerging | 支持 Parent-Child 合并 |

### 回表架构图

```
┌─────────────────┐          ┌─────────────────────┐
│  BM25 Index     │          │  ChunkedDocumentStore│
│  (不存内容)      │          │  (存实际内容)         │
├─────────────────┤          ├─────────────────────┤
│ chunk_id_list   │          │ parent_docs         │
│ term_freqs      │──────────│ chunks              │
│ term_index      │  回表    │ parent_to_chunks    │
│ (倒排索引)       │          │                     │
└─────────────────┘          └─────────────────────┘

┌─────────────────┐
│  Vector Index   │──────────回表──────────→ 同上
│  (只存 embedding│
│   + chunk_id)   │
└─────────────────┘
```

### 回表代码实现

```rust
// src/retrieval/bm25/chunked.rs
// BM25 搜索后回表
let leaf_chunks: Vec<ChunkDocument> = matched_leaves
    .iter()
    .filter_map(|(idx, _)| {
        let chunk_id = self.index.get_chunk_id(*idx)?;      // 从索引获取 ID
        let chunk = self.index.store().get_chunk(&chunk_id)  // 回表获取内容
            .ok()
            .flatten()?;
        Some(chunk)
    })
    .collect();

// src/retrieval/unified_hybrid.rs
// 向量搜索后回表
for (idx, _score) in top_k_indices {
    let entry = &vectors[idx];
    if let Some(chunk) = self.document_store.get_chunk(&entry.chunk_id).await? {
        docs.push(Document::new(chunk.content));
    }
}
```

### AutoMerging 合并逻辑

当同一 Parent 的多个 Leaf Chunk 匹配时，自动合并为完整 Parent：

```rust
// src/retrieval/bm25/chunked.rs
fn auto_merge(&self, scored_chunks: Vec<(usize, f64)>, k: usize) -> Vec<ChunkedSearchResult> {
    let threshold = self.index.config.merge_threshold;
    
    for (parent_id, matched_leaves) in parent_stats {
        let ratio = matched_leaves.len() as f32 / leaves_per_parent as f32;
        
        if ratio >= threshold {
            // 合并：返回完整 Parent 文档
            let parent_doc = self.index.store()
                .get_parent_document(&parent_id)?;
            
            results.push(ChunkedSearchResult {
                merged_parent: parent_doc,
                leaf_chunks: Vec::new(),  // 不返回 Leaf
                score: avg_score,
                parent_id,
            });
        } else {
            // 不合并：返回匹配的 Leaf Chunks
            let leaf_chunks = matched_leaves
                .iter()
                .filter_map(|(idx, _)| {
                    let chunk_id = self.index.get_chunk_id(*idx)?;
                    self.index.store().get_chunk(&chunk_id).ok().flatten()
                })
                .collect();
            
            results.push(ChunkedSearchResult {
                merged_parent: None,
                leaf_chunks,
                score: avg_score,
                parent_id,
            });
        }
    }
}
```

---

## Hybrid 混合检索与 RRF 融合

### RRF 融合算法

**RRF（Reciprocal Rank Fusion）** 是一种排序融合算法，不依赖原始分数，只依赖排名。

**公式**：

```
RRF_score(d) = Σ 1/(k + rank(d))
```

- **k**：平滑参数，默认 60
- **rank(d)**：文档在各检索结果中的排名（从 1 开始）

### RRF 实现

```rust
// src/retrieval/hybrid.rs
pub fn reciprocal_rank_fusion(
    bm25_results: Vec<Document>,
    vector_results: Vec<Document>,
    k: usize,
) -> Vec<RetrievedDocument> {
    let mut rrf_scores: HashMap<String, (f64, Document)> = HashMap::new();
    
    // BM25 结果处理
    for (rank, doc) in bm25_results.iter().enumerate() {
        let doc_id = doc.id.clone().unwrap_or_default();
        let rrf_contribution = 1.0 / (k as f64 + (rank + 1) as f64);
        
        rrf_scores
            .entry(doc_id.clone())
            .and_modify(|(score, _)| *score += rrf_contribution)
            .or_insert((rrf_contribution, doc.clone()));
    }
    
    // 向量结果处理
    for (rank, doc) in vector_results.iter().enumerate() {
        let doc_id = doc.id.clone().unwrap_or_default();
        let rrf_contribution = 1.0 / (k as f64 + (rank + 1) as f64);
        
        rrf_scores
            .entry(doc_id.clone())
            .and_modify(|(score, _)| *score += rrf_contribution)
            .or_insert((rrf_contribution, doc.clone()));
    }
    
    // 按 RRF 分数排序
    let mut results = rrf_scores.into_iter()
        .map(|(_, (score, doc))| RetrievedDocument {
            document: doc,
            score,
            source: RetrievalSource::Hybrid,
        })
        .collect();
    
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    
    results
}
```

### RRF 优势

| 特性 | 说明 |
|------|------|
| **不依赖原始分数** | BM25 和向量分数量纲不同，直接加权困难 |
| **排名融合** | 只看排名位置，公平对待两种检索 |
| **重叠加权** | 两种检索都返回的文档得分更高 |
| **鲁棒性** | 对异常分数不敏感 |

### 计算示例

假设 k=60：

| 文档 | BM25 排名 | Vector 排名 | RRF 分数 |
|------|----------|------------|---------|
| doc1 | 1 | 2 | 1/(60+1) + 1/(60+2) = 0.032 |
| doc2 | 2 | 1 | 1/(60+2) + 1/(60+1) = 0.032 |
| doc3 | 3 | 5 | 1/(60+3) + 1/(60+5) = 0.030 |
| doc4 | 10 | 未出现 | 1/(60+10) = 0.014 |

**doc1 和 doc2 在两种检索中都排名靠前，得分最高。**

---

## Agent + RAG + BM25 工作流程

### 完整流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户提问                                    │
│              "Rust 语言的内存安全是如何实现的？"                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    UnifiedHybridIndex                               │
│  ┌───────────────────┐        ┌───────────────────┐                │
│  │   BM25 检索        │        │   向量检索         │                │
│  │  "内存安全" → Top-K│        │  Embedding → Top-K │                │
│  └───────────────────┘        └───────────────────┘                │
│              ↓                          ↓                          │
│         回表获取内容              回表获取内容                       │
│              ↓                          ↓                          │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    RRF 融合                                      ││
│  │   合并两种结果 → 按 RRF 分数排序 → Top-N                          ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       检索结果                                      │
│  1. "Rust 的所有权机制保证内存安全..."  (RRF=0.032)                │
│  2. "Rust 编译期检查防止内存泄漏..."    (RRF=0.028)                │
│  3. "Rust 的借用规则确保..."           (RRF=0.025)                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent                                        │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  FunctionCallingAgent / ReActAgent                               ││
│  │  1. 接收检索结果作为 Context                                      ││
│  │  2. 构建 Prompt: "基于以下资料回答问题..."                        ││
│  │  3. 调用 LLM 生成回答                                             ││
│  │  4. 可选：调用工具（Calculator、URLFetch）补充信息                ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        最终回答                                     │
│  "Rust 的内存安全主要通过以下机制实现：                              │
│   1. 所有权系统：每个值有唯一所有者...                               │
│   2. 借用检查：编译期验证引用有效性...                               │
│   3. 生命周期：确保引用不会超出数据范围..."                          │
└─────────────────────────────────────────────────────────────────────┘
```

### RetrievalQA 实现

```rust
// Agent + RAG 结合的核心实现
pub struct RetrievalQA {
    llm: Arc<dyn BaseChatModel>,
    retriever: Arc<dyn RetrieverTrait>,
    top_k: usize,
}

impl RetrievalQA {
    pub async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<Value, ChainError> {
        let query = inputs.get("query").unwrap().as_str().unwrap();
        
        // 1. 混合检索
        let docs = self.retriever.retrieve(query, self.top_k).await?;
        
        // 2. 构建 Prompt
        let context = docs.iter()
            .map(|d| d.content.clone())
            .join("\n\n");
        
        let prompt = format!(
            "基于以下资料回答问题：\n\n资料：\n{}\n\n问题：{}\n\n回答：",
            context, query
        );
        
        // 3. 调用 LLM
        let response = self.llm.chat(vec![
            Message::human(prompt),
        ], None).await?;
        
        Ok(Value::String(response.content))
    }
}
```

### Agent 工具调用增强

```rust
// Agent 可以调用工具补充检索信息
let tools: Vec<Arc<dyn BaseTool>> = vec![
    Arc::new(URLFetchTool::new()),  // 获取最新在线资料
    Arc::new(Calculator::new()),     // 计算统计数据
];

let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
let executor = AgentExecutor::new(Arc::new(agent), tools);

// 检索结果 + 工具调用 = 更完整的回答
let result = executor.invoke(query).await?;
```

---

## 性能优化建议

### 1. 索引优化

| 优化点 | 建议 |
|------|------|
| BM25 索引大小 | 只存词频统计，不存内容 |
| 向量索引 | 使用 Qdrant/Milvus 替代内存索引 |
| 文档分割 | chunk_size=500，overlap=50 |

### 2. 检索参数调优

```rust
let config = HybridIndexConfig::new()
    .with_chunk_size(500)       // chunk 大小
    .with_top_k(10, 10)         // BM25_k=10, Vector_k=10
    .with_rrf_k(60)             // RRF 参数
    .with_merge_threshold(0.5); // AutoMerging 阈值
```

### 3. 存储后端选择

| 场景 | 推荐 |
|------|------|
| 开发/测试 | InMemoryChunkedDocumentStore |
| 生产环境 | MongoChunkedDocumentStore |
| 高频访问 | Redis 缓存层 + MongoDB 持久化 |

### 4. 并行执行

```rust
// BM25 和向量检索并行执行
let bm25_future = async { bm25_retriever.search_async(query, k).await };
let vector_future = async { vector_store.search(query, k).await };

let (bm25_docs, vector_docs) = tokio::join!(bm25_future, vector_future);

// RRF 融合
let results = reciprocal_rank_fusion(bm25_docs, vector_docs, 60);
```

---

## 总结

| 核心原理 | 关键点 |
|---------|--------|
| **BM25** | IDF 衡量词重要性，TF 归一化避免高频词主导 |
| **向量检索** | Embedding 语义编码，cosine 相似度匹配 |
| **回表** | 索引存 ID，内容共享，支持 AutoMerging |
| **RRF 融合** | 排名融合，不依赖分数量纲，重叠加权 |
| **Agent + RAG** | 检索结果作为 Context，LLM 生成回答 |

**最佳实践**：Hybrid 检索 + MongoDB 存储 + Agent 工具调用 = 生产级 RAG 系统。

---

## 相关文件

| 文件 | 内容 |
|------|------|
| `src/retrieval/bm25/algorithm.rs` | BM25 核心算法 |
| `src/retrieval/bm25/chunked.rs` | ChunkedBM25 + AutoMerging |
| `src/retrieval/hybrid.rs` | RRF 融合算法 |
| `src/retrieval/unified_hybrid.rs` | 统一混合索引 |
| `src/vector_stores/document_store.rs` | 回表存储实现 |
| `src/vector_stores/mongo_document_store.rs` | MongoDB 存储 |
| `docs/USAGE.md` | 使用指南 |
| `docs/USAGE_EN.md` | 英文使用指南 |