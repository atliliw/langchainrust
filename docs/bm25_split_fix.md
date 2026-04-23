# BM25 分割算法修复说明

## 问题

BM25 的 Parent-Child 分割算法使用了简单字符切分，导致：
- 句子中间被切断
- 单词中间被切断
- 语义边界被破坏

## 修复前 vs 修复后

### 修复前（简单字符切分）

```
原始文档："人工智能在医疗领域的应用越来越广泛，包括疾病诊断、药物研发等。"

BM25 分割结果：
    chunk_0: "人工智能在医疗领域的应用越来越广泛，包括疾病诊断、药物研"  ← 切到"药物研"中间
    chunk_1: "发等。"
```

代码：
```rust
// document_store.rs 第 302-304 行
let end = std::cmp::min(start + chunk_size, total_len);
let chunk_content: String = chars[start..end].iter().collect();
// 直接按字符索引切分，不考虑语义边界
```

### 修复后（语义边界分割）

```
原始文档："人工智能在医疗领域的应用越来越广泛，包括疾病诊断、药物研发等。"

BM25 分割结果：
    chunk_0: "人工智能在医疗领域的应用越来越广泛，"  ← 在逗号处分割
    chunk_1: "包括疾病诊断、药物研发等。"         ← 在句号处分割
```

代码：
```rust
// 使用 RecursiveCharacterSplitter
let splitter = RecursiveCharacterSplitter::new(chunk_size, chunk_size / 10);
let chunks = splitter.split_text(content);
// 按分隔符优先级分割：段落 > 行 > 句号 > 空格 > 字符
```

## RecursiveCharacterSplitter 分隔符优先级

```
优先级 1: "\n\n" (段落分隔)
优先级 2: "\n"   (行分隔)
优先级 3: "。"   (中文句号)
优先级 4: "."    (英文句号)
优先级 5: " "    (空格)
优先级 6: ""     (字符，最后手段)
```

算法会优先在高优先级分隔符处切分，只有当无法满足 chunk_size 要求时才使用低优先级分隔符。

## 改动文件

| 文件 | 改动 |
|------|------|
| `src/vector_stores/document_store.rs` | 引入 `RecursiveCharacterSplitter`，修改 `split_and_store_chunks_blocking` 和 `split_and_store_chunks_async` |
| `src/vector_stores/mongo_document_store.rs` | 同样修改，MongoDB 存储也使用语义分割 |

## 影响

- BM25 搜索结果更精准（完整的句子/段落）
- 搜索质量提升（不会因为单词被切断而漏匹配）
- 与向量存储分割逻辑一致（统一使用 `RecursiveCharacterSplitter`）

## 配置参数

```rust
// chunk_size: 每个 chunk 的目标大小
// chunk_overlap: chunk 之间的重叠大小（默认 chunk_size / 10）
let splitter = RecursiveCharacterSplitter::new(chunk_size, chunk_size / 10);
```

重叠确保相邻 chunk 有一定内容重复，避免边界信息丢失。