# lc-shared 代码审查报告

## 1. 概览

`lc-shared` 是工作区多个 crate 共享的基础类型库,用于打破循环依赖。提供四类能力:

- **工具类型**(`tool_types.rs`):`ToolCall` / `FunctionCall` / `ToolCallResult`,以及构建器 `ToolCallBuilder`。`ToolCall::parse_arguments` 走高容错 JSON 解析,为 lc-schema(Message 用 ToolCall)与 lc-core(工具定义)共用。
- **文档类型**(`document_types.rs`):`Document` / `VectorDocument` / `SearchResult` / `ChunkDocument`,供 lc-vector-stores 与 lc-rag 共用。
- **切分器类型**(`splitter_types.rs`):`TextSplitter` trait + `RecursiveCharacterSplitter`,供 lc-vector-stores 与 lc-rag 共用。
- **容错 JSON 修复**(`json_repair.rs`):LLM 产出的 JSON 修复管线(代码围栏剥离、括号配对提取、非转义引号修复、末尾逗号移除、尾随垃圾截断),本 crate 与 lc-core 收敛到同一实现。

crate 很小且以纯数据 + 无副作用为主,整体质量良好。底层已有大量对历史 stack-overflow / 引号损坏的针对性修复(见文件注释中的 K1/0.20.0 等记录)。

## 2. 遍历的文件清单(功能源码)

- `lib.rs`(模块声明与 flat re-export)
- `document_types.rs`
- `splitter_types.rs`
- `tool_types.rs`
- `json_repair.rs`

(3 个文件的 `#[cfg(test)]` 测试模块按约束只略读,不计入缺陷。)

## 3. 发现的 bug / 高风险点

### [MEDIUM] 重叠区取自"已重叠"的前一 chunk,内容会跨 3 个 chunk 复读

- 位置:`splitter_types.rs:188-205`(特别是 `let prev = &overlapped[i - 1];` 取的是 `overlapped` 而非原始 `chunks[i-1]`)
- 触发条件:chunk_overlap > 0,且连续 ≥3 个原始分片都较短(接近边界)、`overlapped[i-1]` 本身已被前面的重叠填充过。
- 后果:重叠区从"已经带重叠的上一 chunk"取尾巴,其尾部已包含更早 chunk[i-2] 的部分内容,导致 chunk[i-2] 的文本在 overlapped[i] 中被再次带上。连续多段会形成内容反复/放大,送去给 LLM 的上下文出现重复文本(轻微质量与 token 浪费)。修复思路是取原始 `chunks[i-1]` 的尾部作为 overlap 基准。
- 证据:第 187 行 `let prev = &overlapped[i - 1];`(overlapped 已被写入本轮循环);第 188-189 行从 `prev` 算 `overlap_chars` 并切片。

### [LOW] 新增 chunk 重叠预算截断方向的边界语义

- 位置:`splitter_types.rs:192-202`
- 说明:当 `budget = chunk_size - chunk.len()` 小于 overlap 长度时,`overlap` 被截断成 `ov_chars[ov_chars.len() - budget..]`(保留最后 budget 个字符,即最贴近当前 chunk 的那段)。语义保留"离边界最近的重叠",是合理设计;但当 `budget == 0` 时重叠被完全清空,等于该 chunk 无重叠——可接受但文档未说明。非 bug,仅为可改进点。

### [LOW] `fix_unescaped_quotes` 启发式的理论误判面

- 位置:`json_repair.rs:149-197`
- 触发:字符串值本身以结构字符结尾的内容(如值内紧邻 `:`、`,`、`]`、`}` 的引号)。
- 后果:引号是否"闭合"依赖后一非空白字符的启发式判断,存在把内容里的合法引号误判/漏判的可能。已声明为 best-effort,且有转义位处理(`escape_next`),实测数组/嵌套等常见场景可靠。不足以算 bug,列为已知近似面。

## 4. 不合理 / 可改进

1. **`Document` / `ChunkDocument` 的 metadata 用 `HashMap`**:序列化顺序不确定。跨进程稳定哈希或改用有序容器(若丰网某些依赖重排)有助于可复现输出。非功能性缺陷,提醒而已。
2. **`RecursiveCharacterSplitter` 的 `chunk_overlap` 无上界校验**(`splitter_types.rs:82-85`):用户传 `chunk_overlap > chunk_size` 时,叠加预算会饱和,出现部分双倍。建议在构建时 clamp 到 `chunk_size`。
3. **`with_separators` 允许空 separator 列表**(`splitter_types.rs:82`):当前靠"字符级兜底 + 硬停"已避免死循环(第 147-150 行的守卫),行为安全,但 API 文档未提示空 / 退化配置的语义。
4. `ToolCall::parse_arguments`(`tool_types.rs:60`)正确把容错解析失败上游化,没有 panic,良好。

## 5. 可加的新功能

- `TextSplitter` 增加 `split_text_with_indices` / 元数据继承回调,便于下游溯源 chunk→原文偏移。
- `ChunkDocument` 加 `parent_id` → chunk_id 的反向索引辅助函数,便于向量库组接父子文档。
- JSON 修复管线补充可配置的"修复步骤开关",类似 LangChain 的 `parse_json_markdown` 的参数化版本。

## 6. 对标主流(LangChain Core)

- LangChain 的 `TextSplitter` 走 Character/Recursive 两套并支持长度函数与 metadata 合并;本 crate 的 `RecursiveCharacterSplitter` 覆盖了核心能力,但缺"按 token 数切分"(当前按字符计)与"规则/句边界切分"两类常用变体。
- 容错 JSON 方面,langchain-community 用 `json_repair` crate;本仓库自研了具备围栏剥离/引号修复/尾随逗号/截断的管线,定制度更强,但与上游 `json_repair` 相比缺少括号自动补全。整体"够用、更可控"。
- 缺乏字符串长度函数与 tokenizer 抽象,是相对 LangChain 的主要差距。

## 7. 一句总评

类型设计干净、防御充分(已多为历史坑打过补丁),唯一值得修的是重叠区从已重叠 chunk 取值造成的跨 chunk 内容复读。