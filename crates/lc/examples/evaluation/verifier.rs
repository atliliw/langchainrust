//! What: logprob-based Verifier and best-of-N candidate selection.
//! Run: cargo run -p langchainrust --example evaluation_verifier
//! Env: OPENAI_API_KEY (logprobs-capable model; optional: OPENAI_MODEL)

use langchainrust::evaluation::{best_of_n, LogprobVerifier};
use langchainrust::{OpenAIChat, OpenAIConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = OpenAIChat::new(OpenAIConfig::from_env_result()?);
    let verifier = LogprobVerifier::new(llm)
        .with_max_score(9)
        .with_top_logprobs(5);

    let task = "Explain in one sentence why Rust prevents data races.";
    let candidates = vec![
        "Rust prevents data races at compile time through ownership and borrowing rules."
            .to_string(),
        "Rust is fast.".to_string(),
        "The borrow checker rejects aliasing mutable references, so shared mutation is \
         impossible without synchronization."
            .to_string(),
    ];

    let result = best_of_n(&verifier, task, &candidates).await?;
    println!("best index: {}", result.best_index);
    println!("best: {}", result.best);
    for (index, score) in result.scores.iter().enumerate() {
        println!(
            "candidate {index}: score={:.2}/{} confidence={:.2} source={:?}",
            score.score, score.max_score, score.confidence, score.source
        );
    }
    Ok(())
}
