//! OtelHandler 示例
//!
//! 展示如何使用 OpenTelemetry callback 追踪 LLM 调用。
//!
//! # 运行
//! ```bash
//! cargo run --example otel_tracing
//! ```
//!
//! 注意: 需要 `opentelemetry` feature。

use langchainrust::{BaseChatModel, OpenAIChat, OpenAIConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OpenTelemetry 追踪示例 ===\n");

    // OtelHandler 可以作为 callback 注入到 LLM 调用中
    // 追踪每次调用的 token 用量、延迟等信息
    // 并导出到 OpenTelemetry Collector(Jaeger/Zipkin 等)

    println!("OtelHandler 功能:");
    println!("1. 追踪 LLM 调用延迟和 token 用量");
    println!("2. 导出 span 到 OpenTelemetry Collector");
    println!("3. 支持 Jaeger/Zipkin/Prometheus 等后端");
    println!("\n使用方式:");
    println!("  let handler = OtelHandler::new();");
    println!("  llm.with_callback(handler).chat(messages, None).await?;");

    #[cfg(feature = "opentelemetry")]
    {
        use langchainrust::OtelHandler;
        let _handler = OtelHandler::new();
        println!("\n✅ opentelemetry feature 已启用,OtelHandler 可用");
    }

    #[cfg(not(feature = "opentelemetry"))]
    {
        println!("\n⚠️ opentelemetry feature 未启用");
        println!("请使用 cargo run --example otel_tracing --features opentelemetry 运行");
    }

    Ok(())
}
