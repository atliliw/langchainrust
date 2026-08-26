//! Guardrails example (runnable for real, needs an API key)
//!
//! Shows `GuardedAgent` protecting LLM input/output safety (rewritten in P2-5, replacing the
//! previous doc example that only used println):
//! - A real agent (`FunctionCallingAgent` + `AgentExecutor`) wrapped by `GuardedAgent`;
//!   the input passes through `MaxLengthGuardrail`, the output through `SensitiveInfoGuardrail`;
//! - Direct verification of `SensitiveInfoGuardrail`: context-sensitive detection (P2-1, plain
//!   mentions pass) and false-positive grading (P2-2, concrete patterns are blocked);
//! - LLM judge (P2-3): `LlmSensitiveJudge` only blocks "assignment-style mentions" when a real
//!   leak is confirmed.
//!
//! # Run
//! ```bash
//! OPENAI_API_KEY=sk-xxx cargo run --example guardrails
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)
//! - `OPENAI_BASE_URL`: API base URL (optional, defaults to the official endpoint)

use langchainrust::guardrails::{
    GuardableChunk, GuardedAgent, GuardrailsConfig, LlmSensitiveJudge, MaxLengthGuardrail,
    SensitiveInfoGuardrail,
};
use langchainrust::guardrails::{GuardrailError, OutputGuardrail};
use langchainrust::tools::Calculator;
use langchainrust::{
    AgentExecutor, BaseAgent, BaseTool, FunctionCallingAgent, OpenAIChat, OpenAIConfig,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("please set the OPENAI_API_KEY environment variable");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let llm = OpenAIChat::new(OpenAIConfig {
        api_key,
        base_url,
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    });

    println!("=== GuardedAgent end-to-end (input length limit + output sensitive detection) ===\n");
    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
    let agent = FunctionCallingAgent::new(llm.clone(), tools.clone(), None);
    let executor = Arc::new(
        AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools).with_max_iterations(3),
    );
    let config = GuardrailsConfig::new()
        .with_input(Arc::new(MaxLengthGuardrail::new(1000)))
        .with_output(Arc::new(SensitiveInfoGuardrail::new()));
    let mut guarded = GuardedAgent::new(executor, config);

    let result = guarded.invoke("What is 2 + 2?".to_string()).await?;
    println!("Agent output: {result}");
    if guarded.violations().is_empty() {
        println!("→ no guardrail triggered ✓\n");
    }

    println!("=== SensitiveInfoGuardrail direct verification (deterministic, no LLM call) ===\n");
    let g = SensitiveInfoGuardrail::new();
    let demos = [
        (
            "How to store passwords safely",
            "Pass (P2-1 plain mention not blocked)",
        ),
        (
            "You should move the password field to environment variables",
            "Pass (P2-1 plain mention not blocked)",
        ),
        (
            "Please contact user@example.com",
            "Block (concrete email pattern)",
        ),
        (
            "secret key sk-abcdefghijklmnopqrstuvwxyz123456",
            "Block (API key pattern)",
        ),
    ];
    for (text, expect) in demos {
        let outcome = g.validate(text).await;
        println!("  {text:?} → {outcome:?} (expected: {expect})");
    }

    println!("\n=== P2-3 LLM judge: only block assignment-style mentions when a real leak is confirmed ===\n");
    let judged =
        SensitiveInfoGuardrail::new().with_judge(Arc::new(LlmSensitiveJudge::new(llm.clone())));
    for text in ["password is abc123", "password=hunter2"] {
        let outcome = judged.validate(text).await;
        println!(
            "  {text:?} → {outcome:?} (the LLM decides whether a real leak happened; see the judge interface at lc_guardrails::judge)"
        );
    }

    println!("\n=== Streaming output guardrails (two-stage, P1-4) ===\n");
    let tools2: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
    let agent2 = FunctionCallingAgent::new(llm, tools2.clone(), None);
    let executor2 = Arc::new(AgentExecutor::new(
        Arc::new(agent2) as Arc<dyn BaseAgent>,
        tools2,
    ));
    let mut guarded_stream = GuardedAgent::new(
        executor2,
        GuardrailsConfig::new().with_output(Arc::new(SensitiveInfoGuardrail::new())),
    );

    use futures_util::StreamExt;
    match guarded_stream
        .invoke_stream("What is 6 * 7?".to_string())
        .await
    {
        Ok(mut stream) => {
            let mut full = String::new();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(GuardableChunk { token, .. }) => {
                        print!("{token}");
                        full.push_str(&token);
                    }
                    Err(GuardrailError::Blocked { reason, .. }) => {
                        println!("\n[streaming output blocked] {reason}");
                    }
                    Err(e) => println!("\n[streaming error] {e}"),
                }
            }
            println!("\nstreaming final output: {full}");
        }
        Err(e) => println!("[streaming unavailable] {e}"),
    }

    Ok(())
}
