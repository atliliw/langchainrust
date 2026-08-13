# Evaluation

LangChainRust provides 10 built-in evaluators and an LLM-as-judge for quantifying prompt and model changes.

## Evaluators

| Evaluator | Type | Description |
|-----------|------|-------------|
| `ExactMatch` | String | Returns 1.0 for exact match, 0.0 otherwise |
| `StringDistance` | String | Levenshtein distance normalized to 0.0-1.0 |
| `ContainsKeyword` | Rule | Checks for keyword presence (any or all) |
| `RegexMatch` | Rule | Matches against a regex pattern |
| `LengthCheck` | Rule | Validates output length within min/max bounds |
| `Bleu` | NLP | BLEU score with character-level Chinese support |
| `EmbeddingSimilarity` | Semantic | Cosine similarity of embeddings mapped to 0.0-1.0 |
| `LLMAsJudge` | LLM | LLM scores predictions on a configurable rubric |
| `Faithfulness` | LLM | Verifies claims against reference context |
| `PairwiseJudge` | LLM | Compares two predictions with position-bias mitigation |

## Core Types

```rust
pub struct Score { pub value: f64, pub label: Option<String> }  // 0.0-1.0
pub struct Example { pub input: String, pub reference: String }
pub struct Dataset { pub examples: Vec<Example> }
// Report 携带原文 + 汇总(均值/标准差)+ 失败清单,可反序列化(落盘二次分析)
pub struct Report {
    pub per_example: Vec<ExampleReport>,          // 每条含 input/reference/prediction 原文
    pub summary: HashMap<String, ScoreSummary>,   // 每个评测器: mean / std / count
    pub failures: Vec<FailureRecord>,             // 逐条容错记录(为空 = 全部成功)
}
pub struct ExampleReport {
    pub index: usize,
    pub input: String,
    pub reference: String,
    pub prediction: String,
    pub scores: HashMap<String, Score>,
}
pub struct ScoreSummary { pub mean: f64, pub std: f64, pub count: usize }
```

## Batch Evaluation

```rust
use langchainrust::{EvalRunner, ExactMatch, StringDistance, Dataset, Example, Evaluator};

let dataset = Dataset::new(vec![
    Example::new("2+2?", "4"),
    Example::new("Capital of France?", "Paris"),
]);

let runner = EvalRunner::new(vec![
    Box::new(ExactMatch),
    Box::new(StringDistance),
]);

let report = runner.run(&dataset, &predictor).await?;
// report.summary: {"exact_match": ScoreSummary { mean: 0.5, .. }, "string_distance": { .. }}
// 单条 predict / 打分失败不中断整批,记入 report.failures(P1-3)
```

## Individual Evaluators

```rust
use langchainrust::{ContainsKeyword, RegexMatch, LengthCheck, Bleu, LLMAsJudge, Faithfulness};

// Rule-based
let eval = ContainsKeyword::new(vec!["Rust".into()]).all_required(true);
let score = eval.eval("What language?", "Rust is great", "").await?;

let eval = RegexMatch::new(r"\d+")?;
let score = eval.eval("Calculate", "The answer is 42", "").await?;

let eval = LengthCheck::new().min(10).max(100);

// BLEU
let eval = Bleu::new().with_char_level(true); // for Chinese

// LLM-as-judge
let eval = LLMAsJudge::new(judge_llm)
    .with_rubric("Score accuracy on a scale of 0-10")
    .with_max_score(10);

// Faithfulness (hallucination detection)
let eval = Faithfulness::new(judge_llm)
    .with_llm_split(true)  // LLM-based claim splitting
    .with_empty_score(0.0);
```

## Pairwise Comparison

```rust
use langchainrust::{PairwiseJudge, Verdict};

let judge = PairwiseJudge::new(llm).with_rubric("Which answer is more accurate?");
let verdict = judge.compare("What is X?", "Answer A", "Answer B").await?;
// Verdict::AWins, BWins, or Tie
```

`PairwiseJudge` 也实现了 `PairwiseEvaluator` trait(P1-1),可以直接进 `EvalRunner`
统一报告:成对评测器以 `(prediction, reference)` 作为 A/B 两个候选(竞技场用法把待比
答案放进 reference 槽),得分 1.0 = prediction 优、0.5 = 平局、0.0 = reference 优:

```rust
let runner = EvalRunner::new(vec![/* 单点评测器 */])
    .with_pairwise(vec![Box::new(judge)]);
let report = runner.run(&dataset, &predictor).await?;
```
