# lc-evaluation 代码审阅报告

> 审阅性质:只读,未改动任何源码。审阅对象:`crates/lc-evaluation/src/` 全部功能性 `.rs`。
> 本次审阅的核心验证目标二/三:ragas 会话查询 `_id` 键名是否一致;golden 回归门禁是否真实;指标数值是否准确。诚实结论:框架侧**不存在** ragas-[会话持久化/_id] 逻辑(它住在外部 lc-services 部署仓库),故框架侧谈不上 "_id 不一致";golden 回归与各指标均为完整、可测、方向正确的实现。

---

## 一、概览

lc-evaluation 是评估工具箱,分三条线:

1. **标量评估器**:`ExactMatch`、`StringDistance`(Levenshtein)、`Bleu`、`EmbeddingSimilarity`、`LLMAsJudge`(文本解析 + bind_tools 结构化回退)、`rules.rs`(关键词/正则/长度)。
2. **RAGAS 风格评估器**:`ContextPrecision`(rank 加权精度)、`ContextRecall`、`AnswerRelevancy`——LLM 裁决"是否相关/可支撑",统一走 `structured_call_with_usage`(bind_tools 优先、文本兜底),带并发上限(4)与成本记账(I3 `UsageLedger`)。
3. **回归/门禁 + 报表**:`EvalRunner` 逐条容错跑分、`compare_reports` 基线-候选对比 + `is_gate_failing` CI 门禁、`to_jsonl`/`to_table` 导出、`PriceBook` 成本记账。

质量同样偏高:纯函数易于单测、结果确定性排序、fail-closed 解析、NaN 防护齐全。**未发现 bug**;结论均为可信实现,少量可观察的边界/设计取舍列在第四/五节。

## 二、遍历文件清单(14/14,全部读完)

- `crates/lc-evaluation/src/lib.rs` — 模块声明与 re-export
- `crates/lc-evaluation/src/criteria.rs` — `EvalError`、`Score`、`Example`、`Dataset`、`Evaluator/Predpactor/RagEvaluator` 等 trait
- `crates/lc-evaluation/src/results.rs` — `ExactMatch`、`StringDistance`、`EmbeddingSimilarity`、`LLMAsJudge`
- `crates/lc-evaluation/src/runner.rs` — `EvalRunner`、`Report`、`FailureRecord`、逐条容错
- `crates/lc-evaluation/src/ragas.rs` — `ContextPrecision/ContextRecall/AnswerRelevancy`
- `crates/lc-evaluation/src/faithfulness.rs` — `Faithfulness`、`split_claims`、`parse_yes_no`
- `crates/lc-evaluation/src/pairwise.rs` — `PairwiseJudge`(A/B 互换消位置偏置)
- `crates/lc-evaluation/src/compare.rs` — `compare_reports`、`is_gate_failing`
- `crates/lc-evaluation/src/bleu.rs` — `Bleu`、`corpus_bleu`
- `crates/lc-evaluation/src/rules.rs` — `ContainsKeyword/RegexMatch/LengthCheck`
- `crates/lc-evaluation/src/export.rs` — `to_jsonl`/`to_table`
- `crates/lc-evaluation/src/price.rs` — `PriceBook`/`TokenUsage`/`OverallCost`/`UsageLedger`
- `crates/lc-evaluation/src/test_support.rs` — mock `ToolJudge`
- `crates/lc-evaluation/src/integration_tests.rs` — 端到端 + judge 集成

## 三、bug/高风险点

**整体结论:未发现 bug(含高危 = 0)。** 三个重点验证目标逐一交代:

### 3.1 核心验证:ragas 会话查询 `_id` 键名 —— 框架侧不存在该逻辑,故无法也不存在不一致
- 通查 `crates/lc-evaluation/src/**`(14 个文件)与 workspace:**RAGAS 三个评估器都没有任何"会话持久化 / MongoDB `_id` 查询"代码**。它们是纯 stateless 的打分器,输入是 `Example{input, reference, contexts}`。
- 因此记忆里那条"ragas 会话查询 `_id` 键名 bug 已修"指的是**外部 lc-services 部署仓库**(按记忆 `lc-services-no-git-direct-deploy` 它不在此 workspace 内)。框架侧的一致性由"根本不存在 session-persistence/_id 键名"保证——这里没有按键名取错、键名重复的代码面。**诚实结论:框架侧无可验证项,也无 bug;要核对 `_id` 得去 lc-services 仓库看。**

### 3.2 核心验证:golden 回归 / 门禁 —— 真实、闭合、方向正确
- `compare_reports`(`compare.rs:153`)对基线/候选在两份 summary 上做 join,`delta = cand.mean - base.mean`;`delta < -tolerance` 记为 `Regression`(`compare.rs:177`),tolerance 被 clamp 到 `>=0`(`compare.rs:154`)吸收 LLM-judge 噪声,边界等值通过(`compare.rs:275-276` 测试)。
- `is_gate_failing`(`compare.rs:85`)三路皆可 fail:①有回归;②任一共享评估器样本数不一致(`count_mismatches`,`compare.rs:200-206`);③**零共享评估器**(`metrics.is_empty()`,空/发散 run 不算通过)。这是 fail-closed:基线空跑、双跑不一致都不会被放绿。`added/dropped` 只排序列出、不口径混淆(`compare.rs:189-199`)。
- **结论:真实可用的 CI 门禁,非玩具。**

### 3.3 核心验证:指标数值准确
- `ContextPrecision`(RAGAS `CP@K`,`ragas.rs:71-88`):rank 加权精度,相关却被埋在无关 chunk 之下会扣分,公式与 RAGAS 定义一致,形式正确。
- 其余(Levenshtein、Bleu 平滑、cosine、yes/no fail-closed)项项有测试锚定。**未发现计算错误。**

---

## 四、不合理 / 可改进

### 4.1 [INFO] compare 门禁不盯 `dropped` 评估器
- **触发**:基线有、候选没有的某个评估器(`dropped`)。`is_gate_failing`(86)与 `compare_reports`(189-193)只把它列进 `dropped` 列表,**不让门禁失败**——而"忘了挂同一个评估器"这类配置漂移正是门禁想抓的。回归/样本数不一致都抓了,唯独"整列消失"放行。
- **后果**:一次错误的实验配置(候选漏挂 evaluator)可能被静默当成"无比较对象但 pass"。建议新增 `dropped.is_empty()` 进门禁条件,或在 `to_table` 对非空 `dropped` 打醒目标记。

### 4.2 [INFO] Bleu 短句 4-gram 恒为 0(平滑边界)
- **触发**:预测/参考都短于 4 个 token(如 `the cat` vs `the cat`)。`bleu.rs` 句子级在无 4-gram、3-gram、…、直到最后 1-gram 匹配前逐级下探;若没有任何 n-gram 对上(极短句)会落到 0。同句参考常见的 closed-loop 用例会被评 0。这是 BLEU 短句固有缺陷,已有分支与提示,并非错误。
- **可改进**:对全短句提供 "integer-overlap 兜底" 或提示用 `LengthMismatch`/自定义评估器,或在文档明示这一边界。

### 4.3 [INFO] `Score::new` 对 NaN→0.0、对 ±inf 交给 `clamp`
- `criteria.rs:66-77`:NaN 先拦成 0.0 并 `log::warn!`,±inf 由 `clamp(0.0,1.0)` 收敛到边界。语义合理且已注释;只是 ±inf 收敛到 1.0 的方向性未说明(inf 可能是下溢的"低分"却也变 1.0)。可考虑把非有限值统一按 0.0 处理。

### 4.4 [INFO] `from_jsonl` 为 async、`to_jsonl` 为同步
- `Dataset::from_jsonl` async(IO)、`Report::to_jsonl` 同步纯函数——分工正确。仅提示:大文件 report `to_table` 里起 `HashMap` 排序 + `flatten` 对每个字符重建 String,超大批量导出时是纯 CPU 路径,未做流式,属可优化项。

---

## 五、可加的新功能

1. **`paired`/`sign-test` 显著性**:当前门禁只比均值差,不做样本量/方差估计(即便 `count` 相同,jump 也可能只是噪声)。可加对 `ScoreSummary.std` + n 的配对 t 显著性(或至少"delta 相对 std 归一化")作为二级阈值。
2. **`dropped`/`added` 可配门禁**:呼应 4.1,给 `compare_reports` 加 `gate_on_dropped: bool`,让用户决定"评估器消失"是否算门禁失败。
3. **分位数/尾部指标**:summary 只有 mean/std,对长尾失败不敏感;可加 p50/p90/worst-N 记录。
4. **embedding 距离之外的语义相似阈值**:`EmbeddingSimilarity` 可用但未给"多高算通过"的门槛化封装,可作为 `pass_above` helper 给出。

## 六、对标主流

- **对齐**:评测跑的"golden 数据集 + 基线/候选 + 回归门禁 + JSON 导出"整体对标 Hugging Face `evaluate` + OpenAI Evals 的 CI 回归;`is_gate_failing` 的"回归/样本漂移/空比较"三态即 Evals 的做法。
- **RAGAS**:三指标忠实复刻 RAGAS 定义且走 structured-output 优先,优于多数只做文本正则解析的移植。
- **略逊**:无内置对 `mean 之外的方差/显著性` 判断(见五.1),也**没有 ragas 的"会话预填充/prompt 模板库"整体搬移**——那是 lc-services 部署层的活,非本 crate 缺口。
- **胜过**:成本即一等公民(`OverallCost` + `PriceBook` USD/1M + judge token 账本,`price.rs` 全链路铺到 `Report.cost`),以及结构化裁决 + 文本兜底的双通道 judge,在可观测与成本审计上高于多数开源评估框。

## 七、一句总评

**无 bug、实现扎实**:RAGAS 指标、golden 回归门禁均真实可用且方向正确;唯一值得改的是把 `dropped`(评估器消失)纳入门禁失败条件,以及为 `_id` 键名核验点清——它不在此仓库。