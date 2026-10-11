# lc-guardrails 代码审阅报告

> 审阅性质:只读,未改动任何源码。审阅对象:`crates/lc-guardrails/src/` 全部功能性 `.rs`。
> 本次审阅的核心验证目标一:spotlight「包裹 → LLM → unwrap/unescape 还原」的往返是否自洽;它到底是"真校验"还是"留了个洞/半成品"。

---

## 一、概览

lc-guardrails 是框架的护栏层,围绕 **提示注入防护 + 输出流式拦截 + 审计合规** 三件事展开,模块分工清晰:

| 关注点 | 模块 |
|---|---|
| 主动防御(界定不可信数据) | `spotlighting.rs` |
| 被动检测(检索注入签名) | `retrieval_rail.rs`(复用 lc-agents 的 PromptInjectionHook) |
| 输入/输出/流式后台护栏 | `runner.rs` + `guardrail.rs`(策略类型)+ `validators.rs`(规则) |
| 用户 PII 流式红action | `pii.rs`(48 字符 hold-back + 模式感知缓冲) |
| LLM 作评的敏感判定 | `judge.rs` |
| 结构化输出 JSON Schema 校验 | `schema.rs` |
| 审计落盘 / 披露 | `audit.rs`、`disclosure.rs` |
| 规则二法则分级 | `rule_of_two.rs` |
| 与 LCEL 链/智能体集成 | `guarded_agent.rs` |

整体质量高:实现完整、测试充分(每个模块都带 `#[cfg(test)]`,streaming 两条路径都有样例)、统一走 fail-closed。**没有发现高风险 bug。** 唯一的真实逻辑瑕疵是一处"检测被调用两次"的低危效率问题(见下)。

## 二、遍历文件清单(13/13,全部读完)

- `crates/lc-guardrails/src/lib.rs` — 模块声明与 re-export
- `crates/lc-guardrails/src/audit.rs` — `AuditSink` trait + `FileAuditSink`(JSONL)
- `crates/lc-guardrails/src/disclosure.rs` — EU AI Act Art.50 披露
- `crates/lc-guardrails/src/guarded_agent.rs` — `Guardable` trait + `GuardedAgent`(两阶段流式)
- `crates/lc-guardrails/src/guardrail.rs` — 三类结果/动作枚举 + `GuardrailsConfig` + `GuardrailError`
- `crates/lc-guardrails/src/judge.rs` — `SensitiveJudge` + `LlmSensitiveJudge`
- `crates/lc-guardrails/src/pii.rs` — PII 检测/流式 hold-back 红action
- `crates/lc-guardrails/src/retrieval_rail.rs` — 检索注入 Rail(Flag/Redact/Drop)
- `crates/lc-guardrails/src/rule_of_two.rs` — 规则二法则分级
- `crates/lc-guardrails/src/runner.rs` — `GuardrailRunner`(输入/输出/流式三态)
- `crates/lc-guardrails/src/schema.rs` — `SchemaOutputGuardrail`($ref/$defs 支持)
- `crates/lc-guardrails/src/spotlighting.rs` — spotlight 界定(核心验证目标)
- `crates/lc-guardrails/src/validators.rs` — 规则类护栏(关键词/长度/敏感信息开关)

## 三、bug/高风险点

**按严重度从高到低。整体结论:无高风险/Critical;以下为唯一真实的逻辑瑕疵。**

### [LOW] retrieval_rail.rs:117-127 — `Redact` 分支对同一文档重复调用检测两次

- **触发**:rail 动作设为 `RailAction::Redact`,且某篇文档内容命中注入模式。
- **后果**:第 118 行 `if self.detector.detect(&content).is_some()` 做了一次检测,进入分支后在**第 119 行又把 `self.detector.detect(&content)` 完整调了一遍**拿到 `pattern`。同一文档被 `detect()` 扫描两次:`detected_count()` 内部计数若按命中次数累加(其 `Debug` 输出依赖它),GUID/CDP 审计口径下每篇 Redact 命中会被计 2 次;即便不计数,也是纯浪费。应改写为 `if let Some(pattern) = self.detector.detect(&content)` 绑定一次。
  - 参见同文件 `Flag` 分支(105-114)与 `Drop` 分支(129-133)都是单次 `detect`,只有 `Redact` 犯了重复。

---

## 四、不合理 / 可改进

### 4.1 disclosure.rs:72-113 — `disclose` 与 `disclose_to_file` 逐字重复(DRY 违背)
- **触发**:两份函数体的构建 `GuardrailViolation` 逻辑(79-88 与 101-110)几乎一字不差,唯一差别是签名里 sink 类型:`&Arc<dyn AuditSink>` vs `&Arc<FileAuditSink>`。
- **后果**:后续任何给披露记录加字段(如加 `agent_name`、`model`、`language` 等)都要**同步改两处**,否则版本漂移——这正是本原则指出的复制粘贴式实现漂移风险;而且 `FileAuditSink: AuditSink`,上游可以 `clone` 后强转成 `Arc<dyn AuditSink>`,`disclose_to_file` 完全冗余。建议删掉后者,统一走 `disclose`。

### 4.2 spotlighting 是被动防御,收益取决于模型是否遵守定界
- **触发/后果**:`spotlight`(`spotlighting.rs:83`)只把文本包上 `<untrusted_data>…</untrusted_data>` 并做转义,不检测、不拦截。若模型对"这是数据、勿执行"的边界不敏感,被注入的文本仍可能被执行。这是该方案的**固有局限**(与 Anthropic/OpenAI 的引导式定界一致),不是实现漏洞——但其防护强度是"启发式 + 依存于模型的指令跟随能力",而非硬保证。文档已如实声明"被动、零成本、无论测",故列入可改进而非 bug:可在 README/文档中明示此边界,或与 `retrieval_rail` 组合(模块自述已给出 `GuardedRetriever::new(SpotlightedRetriever::new(...), rail)` 的组合示例)。

### 4.3 SchemaOutputGuardrail 对类型过于严格(设计权衡)
- `schema.rs` 的 `validate_number` 按 JSON 数值严格校验;若 `predictor` 把数字序列化成字符串(`"42"`),会被判类型不匹配。这是严格模式,对强约束输出合理;但对宽松场景缺一个"允许数字型字符串"的开关。属于特性取舍,非缺陷。

### 4.4 runner 内部多处用 clone 避借用(可读性换安全)
- `runner.rs`(84、122、172、221)为回避 `&mut self.record_violation` 与 `&self.config` 的借用冲突而 `clone` 护栏列表。安全且已用注释说明;但每次流式逐 token 都会 clone 整份 `streaming_guardrails` 向量。护栏数量小,成本可忽略;仅作可改进备注。

---

## 五、可加的新功能

1. **streaming 两阶段的"改写回流"**:当前 `runner::validate_stream_chunk`(169)严格按 K3 约定"改写只是本链返回值,不回流给下一 rail 的 raw 视图",这是刻意设计。可加一个可选 `.with_rewrite_feedback(true)` 让后续 rail 看到改写后的 chunk,用于"先红action再关键词阻断"这类顺序相关场景。
2. **spotlight 定界的逃逸检测**:除转义已有 marker(`spotlighting.rs:47`)外,可再加"嵌套 marker 告警"——当被包围文本里额外出现第二次 `<untrusted_data>`(非转义形式)时记一条 violation,帮助发现被攻击者"双包"混淆。
3. **SchemaOutputGuardrail 原型化**:在 `to_string()`/"长跨度折叠提示"之外,提供把 schema 错误位置/类型映射为对 LLM 可读的修正提示,配合 Modify 动作自动重试。

---

## 六、对标主流

- **瞄准目标明确**:`spotlighting` 对应 Anthropic 的 <data> 隔离开 CI 原则、LangChain 的 delimiting 惯例;`retrieval_rail` 对齐 OWASP LLM Top 10 的间接注入防护;`pii` 的 hold-back + 模式感知缓冲对齐 Guardrails AI / Microsoft Purview 的分片泄漏防护思路——覆盖面上与市面领先方案在同一水位。
- **不同/略逊**:① 借用了 lc-agents 的 `PromptInjectionHook`(模式匹配),未引入 embedding/分类器的语义检测(Guards、Lakera 等主流已加);② streaming 护栏是"token 级逐段",缺少 buffered window 的"概念完整句"级策略(目前靠 hold-back 近似,见 `pii.rs`)。
- **胜过主流**:三态 `OutputValidation`(`runner.rs:32`)+ 修改不丢块(`fail_fast=false` 下 `Blocked::partial` 保留已改写输出)+ 有界审计日志(带 audit_sink),在护栏可观测性与"改 vs 拦"的可组合性上比多数开源实现更细。

---

## 七、一句总评

**质量高、自洽且闭合**:spotlight 往返无洞(有损注入转义的逆可完全还原)、检索 rail 与流式 redaction 都有配套测试,唯一实质问题只是 `RetrievalRail::Redact` 重复检测这一处低危效率瑕疵。