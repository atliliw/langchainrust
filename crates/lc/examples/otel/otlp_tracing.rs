//! OTLP tracing example (B9, v0.22.4; requires the `otlp` feature)
//!
//! Installs the OTLP/HTTP-JSON export pipeline and drives an `OtelHandler`
//! over a chain → LLM run tree. The exported spans follow the stabilized
//! OTel GenAI semantic conventions:
//!
//! - span names `chat {model}` / `execute_tool {tool}`
//! - `gen_ai.provider.name`, `gen_ai.operation.name`, `gen_ai.request.model`,
//!   `gen_ai.response.model`
//! - `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`
//! - `gen_ai.response.finish_reasons` (string array)
//! - input messages as `gen_ai.user.message` / `gen_ai.system.message`
//!   events and the answer as a `gen_ai.choice.message` event
//! - errors set the span status plus `error.type`
//! - every span carries `langchainrust.run_id` / `langchainrust.trace_id`,
//!   the join keys against `lc_evaluation` reports
//!
//! # Collector
//!
//! ```bash
//! docker run --rm -p 4318:4318 otel/opentelemetry-collector-contrib:latest
//! ```
//!
//! # Run
//! ```bash
//! cargo run -p langchainrust --example otlp_tracing --features otlp
//! # optional: OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_HEADERS,
//! #           OTEL_SERVICE_NAME
//! ```

use langchainrust::schema::Message;
use langchainrust::{otlp::install_otlp_pipeline, CallbackHandler, OtelHandler, RunTree, RunType};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OTLP GenAI tracing example ===");

    // 1. Install the exporter. Keep the guard alive for the whole process —
    //    dropping it flushes batched spans and shuts the provider down.
    let _otlp = install_otlp_pipeline()?;
    let handler = OtelHandler::from_global("langchainrust");

    // 2. Emit a chain → LLM run tree (a real OpenAIChat/Agent produces exactly
    //    these callbacks; we drive them directly so the example needs no key).
    let mut chain = RunTree::new("demo-chain", RunType::Chain, json!({}));
    chain.trace_id = Some(chain.id);
    handler.on_chain_start(&chain, &json!({})).await;

    let mut llm = RunTree::new(
        "gpt-4o-mini:chat",
        RunType::Llm,
        json!({"model": "gpt-4o-mini", "messages": ["What is 2+2?"]}),
    );
    // Parent linkage → one trace in the collector.
    llm.parent_run_id = Some(chain.id);
    llm.trace_id = Some(chain.id);
    llm.metadata.insert("temperature".to_string(), json!(0.0));

    let messages = vec![
        Message::system("Answer with a single word."),
        Message::human("What is 2+2?"),
    ];
    handler.on_llm_start(&llm, &messages).await;

    // Providers record response model + token usage on outputs; the handler
    // resolves the semconv attributes from there.
    llm.end(json!({
        "content": "Four",
        "model": "gpt-4o-mini-2024-07-18",
        "finish_reason": "stop",
        "token_usage": {
            "prompt_tokens": 18u64,
            "completion_tokens": 2u64,
            "total_tokens": 20u64
        },
    }));
    handler.on_llm_end(&llm, "Four").await;
    handler
        .on_chain_end(&chain, &json!({"answer": "Four"}))
        .await;

    println!(
        "emitted 2 spans (chain + chat gpt-4o-mini); run_id={}",
        chain.id
    );
    println!("they are exported on shutdown of the OTLP guard");
    Ok(())
}
