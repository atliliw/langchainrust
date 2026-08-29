//! OtelHandler example
//!
//! Shows how to use OpenTelemetry callbacks to trace LLM calls.
//!
//! # Run
//! ```bash
//! cargo run --example otel_tracing
//! ```
//!
//! Note: requires the `opentelemetry` feature.

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OpenTelemetry tracing example ===\n");

    // OtelHandler can be injected as a callback into LLM calls
    // to trace per-call token usage, latency, and more,
    // and export them to an OpenTelemetry Collector (Jaeger/Zipkin, etc.)

    println!("OtelHandler features:");
    println!("1. Traces LLM call latency and token usage");
    println!("2. Exports spans to an OpenTelemetry Collector");
    println!("3. Supports backends such as Jaeger/Zipkin/Prometheus");
    println!("\nUsage:");
    println!("  let handler = OtelHandler::from_global(\"langchainrust\");");
    println!("  llm.with_callback(handler).chat(messages, None).await?;");

    #[cfg(feature = "opentelemetry")]
    {
        use langchainrust::OtelHandler;
        let _handler = OtelHandler::from_global("langchainrust");
        println!("\n✅ opentelemetry feature is enabled, OtelHandler is available");
    }

    #[cfg(not(feature = "opentelemetry"))]
    {
        println!("\n⚠️ opentelemetry feature is not enabled");
        println!("run with: cargo run --example otel_tracing --features opentelemetry");
    }

    Ok(())
}
