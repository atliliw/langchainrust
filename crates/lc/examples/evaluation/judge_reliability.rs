//! What: position-bias and chance-corrected accounting for LLM judges.
//! Run: cargo run -p langchainrust --example evaluation_judge_reliability
//! Env: none (pure functions)

use langchainrust::evaluation::{
    cohen_kappa, order_swap_consistent, reliability_report, same_family, Verdict,
};

fn main() {
    // A winner counts only if it survives both A/B and B/A.
    let original = [
        Verdict::AWins,
        Verdict::BWins,
        Verdict::AWins,
        Verdict::Tie,
    ];
    let swapped = [
        Verdict::BWins,
        Verdict::AWins,
        Verdict::AWins,
        Verdict::Tie,
    ];
    for (a, b) in original.iter().zip(swapped.iter()) {
        println!("{a:?} + {b:?} -> {:?}", order_swap_consistent(*a, *b));
    }

    // Chance-corrected agreement.
    let judge_a = [true, true, false, false, true];
    let judge_b = [true, false, false, true, true];
    println!("cohen_kappa: {:?}", cohen_kappa(&judge_a, &judge_b));

    // Same-family self-preference risk.
    println!("gpt-5 vs gpt-4o: {}", same_family("gpt-5", "gpt-4o"));
    println!("gpt-5 vs claude-3: {}", same_family("gpt-5", "claude-3"));

    println!(
        "reliability report: {:?}",
        reliability_report(&original, &swapped)
    );
}
