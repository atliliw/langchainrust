//! Real LLM integration tests (ignored by default, need network + API key).

use super::*;
// P2-1: 无 `local-embeddings` feature 时 `LocalEmbeddings` 是已弃用的
// BagOfWordsEmbeddings 别名;此处有意使用廉价本地嵌入,豁免降级警告。
#[allow(deprecated)]
use lc_embeddings::LocalEmbeddings;
use lc_providers::openai::{OpenAIChat, OpenAIConfig};

#[tokio::test]
#[ignore = "需要真实 API key + 网络"]
async fn test_judge_real_llm() {
    let config = OpenAIConfig::default();
    let judge = LLMAsJudge::new(OpenAIChat::new(config));
    let cases: &[(&str, &str, &str, &str)] = &[
        ("正确", "法国首都是哪座城市?", "巴黎", "巴黎"),
        ("错误", "法国首都是哪座城市?", "巴黎", "伦敦"),
        (
            "改写",
            "法国首都是哪座城市?",
            "巴黎",
            "法国的首都是巴黎,位于法国北部。",
        ),
    ];
    let mut results: Vec<(&str, f64)> = Vec::new();
    for (name, input, reference, prediction) in cases {
        match judge.eval(input, prediction, reference).await {
            Ok(s) => {
                println!("[{}] value={} label={:?}", name, s.value, s.label);
                results.push((name, s.value));
            }
            Err(e) => {
                println!("[{}] ERROR: {}", name, e);
                results.push((name, -1.0));
            }
        }
    }
    for (name, v) in &results {
        assert!(*v >= 0.0, "{} illegal: {}", name, v);
        assert!(*v <= 1.0, "{} out of range: {}", name, v);
    }
    assert!(
        results[0].1 >= results[1].1,
        "correct({}) < wrong({})",
        results[0].1,
        results[1].1
    );
}

#[tokio::test]
#[ignore = "需要真实 API key + 网络(LLMAsJudge)"]
async fn test_all_four_evaluators_real() {
    let config = OpenAIConfig::default();
    let judge_model = OpenAIChat::new(config);
    let evaluators: Vec<Box<dyn Evaluator>> = vec![
        Box::new(ExactMatch),
        Box::new(StringDistance),
        Box::new(EmbeddingSimilarity::new(LocalEmbeddings::default_dim())),
        Box::new(LLMAsJudge::new(judge_model)),
    ];
    let cases: &[(&str, &str, &str, &str)] = &[
        ("正确", "法国首都是哪座城市?", "巴黎", "巴黎"),
        ("错误", "法国首都是哪座城市?", "巴黎", "伦敦"),
        (
            "改写",
            "法国首都是哪座城市?",
            "巴黎",
            "法国的首都是巴黎,位于法国北部。",
        ),
    ];
    println!(
        "\n{:<6} | {:<12} | {:<14} | {:<14} | {:<10}",
        "样例", "ExactMatch", "StringDistance", "EmbeddingSim", "LLMAsJudge"
    );
    println!("{}", "-".repeat(70));
    for (label, input, reference, prediction) in cases {
        let mut row = format!("{:<6}", label);
        for ev in &evaluators {
            let s = ev.eval(input, prediction, reference).await.unwrap();
            assert!(
                s.value >= 0.0 && s.value <= 1.0,
                "{}/{} out of range: {}",
                label,
                ev.name(),
                s.value
            );
            row.push_str(&format!(" | {:<12.4}", s.value));
        }
        println!("{}", row);
    }
    println!();
}

#[tokio::test]
#[ignore = "需要真实 API key + 网络(LLMAsJudge)"]
async fn test_four_scenarios_real() {
    let config = OpenAIConfig::default();
    let judge_model = OpenAIChat::new(config);
    let evaluators: Vec<Box<dyn Evaluator>> = vec![
        Box::new(ExactMatch),
        Box::new(StringDistance),
        Box::new(EmbeddingSimilarity::new(LocalEmbeddings::default_dim())),
        Box::new(LLMAsJudge::new(judge_model)),
    ];
    let article = "光合作用是植物、藻类和某些细菌利用阳光将二氧化碳和水转化为葡萄糖和氧气的过程。它主要在叶绿体中进行,依赖叶绿素吸收光能。光合作用分为光反应和暗反应两个阶段:光反应在类囊体膜上发生,产生 ATP 和 NADPH;暗反应在基质中进行,利用这些产物固定二氧化碳。光合作用是地球上大多数生命的能量来源,也是大气中氧气的主要来源。";
    #[allow(clippy::type_complexity)]
    let scenarios: &[(&str, &str, &str, &[(&str, &str)])] = &[
        ("RAG幻觉", "公司年假多少天?", "年假 15 天", &[("A忠实", "员工年假为 15 天"), ("B幻觉", "员工年假为 20 天,可累积")]),
        ("翻译", "It's raining cats and dogs.", "倾盆大雨", &[("A意译", "大雨滂沱"), ("B直译", "正在下猫和狗")]),
        ("代码", "写一个反转字符串的Python函数", "s[::-1]", &[("A切片", "return s[::-1]"), ("B循环", "for i in range(len(s)-1,-1,-1): result += s[i]"), ("C错", "s.reverse()")]),
        ("摘要", "请总结以下文章的要点", article, &[("A好摘要", "光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖和氧气的过程,在叶绿体中进行,分光反应和暗反应两阶段,是地球生命的主要能量来源。"), ("B差摘要", "光合作用是动物利用阳光制造食物的过程,只发生在根部。")]),
    ];
    for (sname, input, reference, preds) in scenarios {
        println!("\n=== 场景:{} ===", sname);
        for (plabel, pred) in *preds {
            let mut row = format!("{:<10}", plabel);
            for ev in &evaluators {
                let s = ev.eval(input, pred, reference).await.unwrap();
                row.push_str(&format!(" | {:<12.4}", s.value));
            }
            println!("{}", row);
        }
    }
    println!();
}

#[tokio::test]
#[ignore = "需要真实 API key + 网络"]
async fn test_pairwise_and_faithfulness_real() {
    let model = OpenAIChat::new(OpenAIConfig::default());
    let pairwise = PairwiseJudge::new(model.clone());
    let v = pairwise
        .compare(
            "公司年假多少天?",
            "员工年假为 15 天",
            "员工年假为 20 天,可累积",
        )
        .await
        .unwrap();
    println!("Pairwise(忠实 vs 幻觉): {:?}", v);
    assert_ne!(v, Verdict::BWins, "幻觉回答不应胜出");
    let faith = Faithfulness::new(model);
    let s_hallucinated = faith
        .eval(
            "公司年假多少天?",
            "员工年假为 20 天,可累积。",
            "员工年假为 15 天。",
        )
        .await
        .unwrap();
    let s_faithful = faith
        .eval(
            "公司年假多少天?",
            "员工年假为 15 天。",
            "员工年假为 15 天。",
        )
        .await
        .unwrap();
    println!(
        "Faithfulness(幻觉): {} / 忠实: {}",
        s_hallucinated.value, s_faithful.value
    );
    assert!(
        s_faithful.value >= s_hallucinated.value,
        "忠实({}) < 幻觉({})",
        s_faithful.value,
        s_hallucinated.value
    );
}
