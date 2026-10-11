# lc-embeddings` 审阅报告

本报告为只读审计(未改动任何代码),覆盖 `crates/lc-embeddings/src/` 全部 `.rs` 文件(含子目录)。每条发现均标注 `文件:行号` + 触发条件 + 后果。

## 1. 概览

`lc-embeddings` 提供向量嵌入能力,抽象为 `Embeddings` trait(P2-8:返回向量统一 L2 归一化)与多维度的 `VisionEmbeddings` trait(图/文同空间,B7)。三套后端:

- **在线 HTTP**:OpenAI / Cohere / DeepSeek / Qwen(OpenAI 兼容协议,`openai_compat.rs` 抽公共实现 + `CompatSpec` 扩展点)、DashScope 原生多模态、Cohere Embed v4 视觉。
- **本地推理**:Candle(嵌入模型,含 FP16/Int8 量化)、ONNX Runtime(`local/nn`,`SessionPool` 有界并发)、token_level(晚切分)、fastembed(文本嵌入)。
- **离线测试**:MockEmbeddings / MockVisionEmbeddings / `test_support.rs`(嵌入式 HTTP stub)。

所有在线 provider 共用 `retry.rs` 的 `post_json_with_retry`(指数退避 + Retry-After + 有界抖动)与 180s 请求超时。总体质量高:**无 panic/unwrap 泄漏、错误体不吞、key 全脱敏、构造期 fail-fast、批对齐显式报错**。真正的风险集中在"返回向量宽度不做校验""重复 index 掩蔽错位""fastembed 全局锁串行"三处。

## 2. 遍历的文件清单

```
crates/lc-embeddings/src/lib.rs
crates/lc-embeddings/src/retry.rs
crates/lc-embeddings/src/openai.rs
crates/lc-embeddings/src/openai_compat.rs
crates/lc-embeddings/src/deepseek.rs
crates/lc-embeddings/src/qwen.rs
crates/lc-embeddings/src/cohere.rs
crates/lc-embeddings/src/vision.rs
crates/lc-embeddings/src/vision_qwen.rs
crates/lc-embeddings/src/vision_cohere.rs
crates/lc-embeddings/src/mock.rs
crates/lc-embeddings/src/candle_emb.rs
crates/lc-embeddings/src/fastembed_emb.rs
crates/lc-embeddings/src/token_level.rs
crates/lc-embeddings/src/local/mod.rs
crates/lc-embeddings/src/local/tests.rs
crates/lc-embeddings/src/local/nn/mod.rs
crates/lc-embeddings/src/local/nn/nn_tests.rs
crates/lc-embeddings/src/test_support.rs
```

## 3. 发现的 bug / 高风险点

### 3.1. [MEDIUM-HIGH] 返回嵌入的**宽度从不校验**等于 `dimension()`

- 位置:`openai.rs:182-190`(embed_query 直接 first()/clone,无 len 检查)、`284-298`;`openai_compat.rs:164-172 / 250-258`;`cohere.rs:273-281`(只校验**数量**,不校验每个向量宽度);`vision_qwen.rs:238-242`;`vision_cohere.rs:256-262`。每个 provider 的 `dimension()` 都是静态查表(`openai.rs:104-115`、`cohere.rs:210-218`、`qwen.rs:149-159`、`deepseek.rs:105-113`),网络返回的 `Vec<f32>` 长度从未与 `self.dimension` 比较。
- 触发:服务端模型变更、matryoshka `dimensions` 越界、截断,或单文本响应返回空/窄向量。
- 后果:不同宽度向量被**静默接受、L2 归一化、原样返回**给信任 `dimension()` 的调用方(retrieval 距离、向量库建索引都对宽度敏感)。无任何断言或报错。这是"静默错误向量宽度"类,缺陷集中在"不校验"而不在"会崩溃"。

### 3.2. [MEDIUM] 重复 / 过满的 provider `index` 掩蔽批错位

- 位置:`openai.rs:268-287`、`openai_compat.rs:227-246`
- 触发:provider 对同一向量返回两个相同 index,或返回数量多于槽位但恰好盖满全部槽位。
- 后果:`all_results[global_index] = Some(item.embedding)` 是覆盖式;守卫只拦"越界 index"(→`BatchMismatch`)和"最终空槽"(→`EmptyVectorInBatch`)。若 provider 返回 `[0,1,1]`(2 文本),槽位 1 被覆盖为重复向量,第 3 个多余条目被忽略,**整批返回 `Ok`**——错位恰好填满槽位时完全隐形。批对齐保护只覆盖"少了/越界",不覆盖"重复/多了但盖满"。

### 3.3. [MEDIUM] retry.rs:429/5xx 无条件重放,无 at-most-once

- 位置:`retry.rs:48-85`(esp. 57-63, 66-83);`is_transient`(121-123 = `429 || >=500`);被每个 `post_json_with_retry` 调用方继承:`openai.rs:155-163, 233-241`、`openai_compat.rs:137-145, 200-208`、`vision_qwen.rs:199-207`、`vision_cohere.rs:195-203`。
- 触发:首个 POST 返回 429 或任意 5xx。
- 后果:退避后**原样重发同一 body**(57-63 行),最多 3 次。5xx(500/502/504)是歧义的——provider 可能已收到并嵌入了文本后才报错(双重计费);只有 429 是真正安全可重试。对比 lc-providers 已用 `SAFE_RETRY`(at-most-once)钉住同一命题,此处无对应配置。**注意**:transport 错误(`send().await?`,63 行)**不会**重试——不会在歧义网络失败上双重计费,这是加分项。`cohere.rs` 走统一 `lc_core::HttpClient`(携带 pre-dispatch 安全默认),避开了这一类。

### 3.4. [MEDIUM] fastembed:单个全局 `Mutex<TextEmbedding>` 串行化全部并发推理

- 位置:`fastembed_emb.rs`(全局 `Arc<Mutex<TextEmbedding>>`)
- 触发:多任务并发调用 `embed_documents`。
- 后果:所有并发请求在全局锁上排队,推理完全串行,吞吐被锁死为核心问题(高并发批嵌入时显著低于可并行实现)。属可观测的性能/可扩展性缺口。

### 3.5. [LOW-MED] vision_cohere 整图像批不分块,一次 POST

- 位置:`vision_cohere.rs:233-256`(embed_images 一次性把所有图塞进 `images` 数组)
- 触发:图像批超过 provider 单请求图像上限。
- 后果:相对文本路径(`cohere.rs:316`,COHERE_MAX_BATCH=96 分块)与 vision_qwen(`vision_qwen.rs:270`,`MAX_CONTENTS_PER_REQUEST=10`),此处**不分块**。超上限批 → 一个超大 POST → 400/413 整批失败;不静默丢,但缺少与兄弟实现一致的健壮性。

### 3.6. [LOW] candle_emb:FP16/Int8 量化"量化后又反量化回 FP32",内存零节省

- 位置:`candle_emb.rs`(F16/Int8 → 量化 → 反量化回 FP32)
- 触发:启用 INT8/FP16 量化。
- 后果:最终权重仍是 FP32 占用,没有实际内存收益;`quantize_int8(...).unwrap_or(t)` 在量化失败时**静默回退原张量**,失败不可见。文档已诚实说明零节省,属"功能名不副实 + 失败被吞"。

### 3.7. [LOW/观察] 无最大输入 token 感知/截断

- 位置:`openai.rs:149-152, 228-231`、`openai_compat.rs:130-134, 193-197`、`cohere.rs:252-257`
- 触发:单条超长文档。
- 后果:`input` 原样发送;分块只按条目数(`batch_size`/96),不按 token 长度。一条超长文本可耗尽模型 token 预算导致整块 400,同时 provider 可能静默截断超长输入,嵌入的是截断后形式。观察项,非硬故障。

## 4. 不合理 / 可改进

- **错误体读取**(P1-4)在所有非 2xx 路径都读了 `response.text()` 并在读取失败时报错,未被吞——正确。`openai.rs:168-174, 246-255`、`openai_compat.rs:150-156, 213-219`、`cohere.rs:236-242+267-271`、`vision_qwen.rs:211-216+220-234`、`vision_cohere.rs:207-214` 均确认。
- **生产代码无 unwrap/expect/panic**:全部集中在 `#[cfg(test)]` 或 `unwrap_or_default`(env 读取主)。retry 数学用 `checked_shl(...).unwrap_or` 作用于本地计数器(`retry.rs:101`),非外部数据。
- **`dimension` 静态表 + 构造期 P1-2 校验**是好的 fail-fast,但**返回侧不校验**(3.1)是同一能力的不对称应用。
- **`openai_compat.rs` 的 `CompatSpec::validate` 钩子**(qwen 用它拒 `dimensions` 越界/非 qwen3)设计良好,可作为"返回宽度校验"的落点。

## 5. 可加的新功能(2026 主流 LLM SDK 角度)

1. **批对齐计数改进**:对重复 index 与"多了但盖满"也报错(`BatchMismatch`),把 3.2 的静默错位变成显式失败。
2. **返回宽度断言**:解析后 `assert_eq!(v.len(), self.dimension)`,不匹配即报错——成本一行,堵住 3.1 整类。
3. **嵌入缓存 / 去重**:同一文本哈希去重,降低成本与锁竞争,主流向量栈标配。
4. **模型 == 检索的持久化编目**:内置常见模型的维度/成本/限流表(OpenAI/Cohere/Qwen/Mistral),构造期 fail-fast + 文档生成。
5. **批量并发令牌桶限流**:为多 chunk 并发嵌入(已有 `MAX_CONCURRENT_CHUNKS`)配 provider 速率预算,替代"一次性并发 8 块"的静态上限。
6. **本地张量到向量库的零拷贝接入 + FP8 量化**:当前量化零收益(3.6),补真正的低精度驻留。

## 6. 对标主流框架

- **强项**:统一 L2 归一化契约、批对齐显式报错(P0-1/P0-2)、构造期维度校验、SSE 有界、`Retry-After` + 有界抖动重试、Cohere/vision URL → 内联(SSRF 侧防)。这些一致性指标胜过多数社区 embeddings 实现。
- **差距**:返回宽度不校验(多数 SDK 靠 provider 保证)与重复 index 掩蔽(较少见但更隐蔽);fastembed 全局锁串行与主流"每线程池"方案有差距。
- **优势**:`local/nn` 的 `SessionPool`(condvar 有界并发)、token_level 晚切分、`CompatSpec` 扩展点是优于需求平均水平的架构。

## 7. 一句总评

质量高、一致性极好,没有 panic/泄漏/错误吞并级 bug;真正值得修的是"返回宽度不校验 + 重复 index 掩蔽 + 无条件 5xx 重放(双计费 + fastembed 串行)";这三处修复成本低、收益直接。