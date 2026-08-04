//! Evaluation 模块示例
//!
//! 展示如何使用评测器评估 LLM 输出质量。
//!
//! # 运行
//! ```bash
//! cargo run --example evaluation
//! ```

use langchainrust::evaluation::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. ExactMatch 评测器
    let evaluator = ExactMatch;
    let result = evaluator.eval("What language?", "Rust", "Rust").await?;
    println!("ExactMatch (相同): score = {}", result.value);
    let result = evaluator.eval("What language?", "Rust", "Python").await?;
    println!("ExactMatch (不同): score = {}", result.value);

    // 2. ContainsKeyword 评测器
    let evaluator = ContainsKeyword::new(vec!["Rust".to_string()]);
    let result = evaluator
        .eval(
            "What language?",
            "Rust is a systems programming language",
            "",
        )
        .await?;
    println!("ContainsKeyword: score = {}", result.value);

    // 3. RegexMatch 评测器
    let evaluator = RegexMatch::new(r"\d+")?;
    let result = evaluator.eval("Calculate", "The answer is 42", "").await?;
    println!("RegexMatch (含数字): score = {}", result.value);
    let result = evaluator.eval("Calculate", "No numbers here", "").await?;
    println!("RegexMatch (无数字): score = {}", result.value);

    // 4. LengthCheck 评测器
    let evaluator = LengthCheck::new().min(10).max(100);
    let result = evaluator
        .eval("Explain", "This is a medium length response", "")
        .await?;
    println!("LengthCheck: score = {}", result.value);

    // 5. StringDistance 评测器
    let evaluator = StringDistance;
    let result = evaluator
        .eval("Greet", "hello world", "hello world")
        .await?;
    println!("StringDistance (相同): score = {}", result.value);

    // 6. Bleu 评测器
    let evaluator = Bleu::new();
    let result = evaluator
        .eval(
            "Translate",
            "the cat is on the mat",
            "the cat sat on the mat",
        )
        .await?;
    println!("Bleu: score = {}", result.value);

    println!("\n所有评测器演示完成!");
    Ok(())
}
