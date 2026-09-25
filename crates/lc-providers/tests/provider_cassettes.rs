// lc-providers/tests/provider_cassettes.rs
//! T4 (v0.23.0) offline cassette-playback tests.
//!
//! lc-providers' live tests were historically `#[ignore]`d with zero fixtures,
//! so response→`LLMResult` field mapping (usage capture, thinking/tool-call
//! parsing, finish_reason, streaming aggregation, error-status mapping) only
//! surfaced from user reports. These tests replay recorded responses through a
//! loopback HTTP/1.1 server (see `common/mod.rs`) so provider behaviour is
//! asserted **offline**, without an API key or network.
//!
//! Routes are keyed on the request path only; the server ignores request
//! bodies, which is all the provider fixtures need.
//!
//! T5 (v0.23.0): the *capture* server additionally records the last request
//! body, so the `_in_request` tests below assert what the provider **sends**
//! (cache-TTL breakpoints, strict `tool_choice`, OpenAI `response_format`) —
//! not just how it maps responses.

mod common;

use common::{routes, MockResponse, Routes};
use futures_util::StreamExt;
use lc_core::language_models::{BaseChatModel, TokenUsage};
use lc_core::tools::ToolDefinition;
use lc_providers::openai::{response_format::ResponseFormat, OpenAIError};
use lc_providers::{AnthropicChat, AnthropicConfig, AnthropicError, OpenAIChat, OpenAIConfig};
use lc_schema::Message;
use serde_json::json;

/// Drives a provider's chat over a mock server for the given request path.
///
/// Builds a route map from (path → canned response) and returns the base_url a
/// provider config should use so its request lands on `route_path`.
async fn mock_base_url(route_path: &str, resp: MockResponse) -> String {
    let routes: Routes = routes::build(vec![(route_path.to_string(), resp)]);
    let addr = common::spawn_mock_server(routes).await;
    format!("http://127.0.0.1:{}/v1", addr.port())
}

// ─────────────────────── Anthropic (non-streaming) ───────────────────────

fn anthropic_chat(base_url: &str) -> AnthropicChat {
    AnthropicChat::new(
        AnthropicConfig::new("test-key-sk-ant")
            .with_base_url(base_url)
            .with_model("claude-sonnet-4-5"),
    )
}

#[tokio::test]
async fn anthropic_nonstream_text_usage_mapping() {
    let base_url = mock_base_url(
        "/v1/messages",
        MockResponse::json(
            200,
            r#"{
                "id":"msg_01","type":"message","role":"assistant",
                "model":"claude-sonnet-4-5",
                "content":[{"type":"text","text":"Hello from cassette."}],
                "stop_reason":"end_turn",
                "usage":{"input_tokens":120,"output_tokens":45,
                         "cache_creation_input_tokens":0,"cache_read_input_tokens":0}
            }"#,
        ),
    )
    .await;
    let model = anthropic_chat(&base_url);
    let result = model
        .chat(vec![Message::human("hi")], None)
        .await
        .expect("non-stream chat should succeed offline");

    assert_eq!(result.content, "Hello from cassette.");
    assert_eq!(result.model, "claude-sonnet-4-5");
    assert!(result.tool_calls.is_none());
    let usage = result.token_usage.expect("usage captured");
    assert_eq!(usage.prompt_tokens, 120);
    assert_eq!(usage.completion_tokens, 45);
    assert_eq!(usage.total_tokens, 165);
}

#[tokio::test]
async fn anthropic_nonstream_thinking_tool_call_mapping() {
    let base_url = mock_base_url(
        "/v1/messages",
        MockResponse::json(
            200,
            r#"{
                "id":"msg_02","type":"message","role":"assistant",
                "model":"claude-sonnet-4-5",
                "content":[
                    {"type":"thinking","thinking":"Let me work through this.","signature":"s1"},
                    {"type":"tool_use","id":"toolu_01","name":"get_weather",
                     "input":{"city":"Beijing"}},
                    {"type":"text","text":"Checking the weather now."}
                ],
                "stop_reason":"tool_use",
                "usage":{"input_tokens":90,"output_tokens":38,
                         "cache_creation_input_tokens":0,"cache_read_input_tokens":0}
            }"#,
        ),
    )
    .await;
    let model = anthropic_chat(&base_url);
    let result = model
        .chat(vec![Message::human("weather in Beijing")], None)
        .await
        .expect("thinking + tool_use chat succeeds");

    assert_eq!(result.content, "Checking the weather now.");
    assert_eq!(
        result.thinking_content.as_deref(),
        Some("Let me work through this.")
    );
    let calls = result.tool_calls.expect("tool call captured");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name(), "get_weather");
    assert!(calls[0].arguments().contains("city"));
    assert_eq!(result.token_usage.unwrap().total_tokens, 128);
}

#[tokio::test]
async fn anthropic_nonstream_cache_usage_totals() {
    let base_url = mock_base_url(
        "/v1/messages",
        MockResponse::json(
            200,
            r#"{
                "id":"msg_03","type":"message","role":"assistant",
                "model":"claude-sonnet-4-5",
                "content":[{"type":"text","text":"cached reply"}],
                "stop_reason":"end_turn",
                "usage":{"input_tokens":10,"output_tokens":5,
                         "cache_creation_input_tokens":500,"cache_read_input_tokens":120}
            }"#,
        ),
    )
    .await;
    let model = anthropic_chat(&base_url);
    let result = model
        .chat(vec![Message::human("again")], None)
        .await
        .expect("cache-tagged usage chat succeeds");

    // total_tokens is input+output only; cache_creation/read are surfaced as
    // input in signature but must not inflate the terminal total they contribute.
    let usage = result.token_usage.unwrap();
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
}

#[tokio::test]
async fn anthropic_error_status_mapping() {
    // 400 is non-retryable for DEFAULT_RETRY, so the offline test is deterministic.
    let base_url = mock_base_url(
        "/v1/messages",
        MockResponse::json(
            400,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad input"}}"#,
        ),
    )
    .await;
    let model = anthropic_chat(&base_url);
    let err = model
        .chat(vec![Message::human("boom")], None)
        .await
        .expect_err("400 must surface an error");
    assert!(
        matches!(err, AnthropicError::Api(ref s) if s.contains("HTTP 400")),
        "expected HTTP 400 Api error, got {err:?}"
    );
}

// ─────────────────────── Anthropic (streaming) ───────────────────────

#[tokio::test]
async fn anthropic_stream_aggregates_text_and_usage() {
    let sse = concat!(
        r#"data: {"type":"message_start","message":{"id":"msg_s1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"stop_reason":null,"usage":{"input_tokens":8,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}"#,
        "\n\n",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        "\n\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"stream"}}"#,
        "\n\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ing"}}"#,
        "\n\n",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "\n\n",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":8,"output_tokens":3,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}"#,
        "\n\n",
        r#"data: {"type":"message_stop"}"#,
        "\n\n",
    );
    let base_url = mock_base_url("/v1/messages", MockResponse::sse_stream(sse)).await;
    let model = anthropic_chat(&base_url);
    let mut stream = model
        .stream_chat(vec![Message::human("go")], None)
        .await
        .expect("stream setup succeeds");

    let mut text = String::new();
    let mut usage: Option<TokenUsage> = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("stream chunk decodes");
        text.push_str(&chunk.text);
        if let Some(u) = chunk.token_usage {
            usage = Some(u);
        }
    }
    assert_eq!(text, "streaming");
    let usage = usage.expect("terminal usage chunk present");
    assert_eq!(usage.prompt_tokens, 8);
    assert_eq!(usage.completion_tokens, 3);
}

#[tokio::test]
async fn anthropic_stream_tool_call_aggregation() {
    let sse = concat!(
        r#"data: {"type":"message_start","message":{"id":"msg_s2","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}"#,
        "\n\n",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_s1","name":"get_weather"}}"#,
        "\n\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#,
        "\n\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Beijing\"}"}}"#,
        "\n\n",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "\n\n",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"input_tokens":5,"output_tokens":4,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}"#,
        "\n\n",
        r#"data: {"type":"message_stop"}"#,
        "\n\n",
    );
    let base_url = mock_base_url("/v1/messages", MockResponse::sse_stream(sse)).await;
    let model = anthropic_chat(&base_url);
    let mut stream = model
        .stream_chat(
            vec![Message::human("weather"), Message::human("beijing")],
            None,
        )
        .await
        .expect("stream setup succeeds");

    let mut tool_call = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("stream chunk decodes");
        if let Some(tc) = chunk.tool_calls {
            tool_call = Some(tc);
        }
    }
    let calls = tool_call.expect("stream tool call captured");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name(), "get_weather");
}

#[tokio::test]
async fn anthropic_stream_early_message_stop_surfaces_error() {
    // Stream cuts off with only content_block_delta — no message_stop, no usage.
    // The provider must treat the missing terminal marker as an error, not
    // silently return a truncated answer (A12 regression guard).
    let sse = concat!(
        r#"data: {"type":"message_start","message":{"id":"msg_s3","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}"#,
        "\n\n",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        "\n\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}"#,
        "\n\n",
    );
    let base_url = mock_base_url("/v1/messages", MockResponse::sse_stream(sse)).await;
    let model = anthropic_chat(&base_url);
    let mut stream = model
        .stream_chat(vec![Message::human("go")], None)
        .await
        .expect("stream setup succeeds");

    let mut saw_error = false;
    while let Some(chunk) = stream.next().await {
        if chunk.is_err() {
            saw_error = true;
        }
    }
    assert!(
        saw_error,
        "truncated stream without message_stop must error"
    );
}

// ─────────────────────── OpenAI (non-streaming) ───────────────────────

#[tokio::test]
async fn openai_nonstream_usage_and_tool_call_mapping() {
    let base_url = mock_base_url(
        "/v1/chat/completions",
        MockResponse::json(
            200,
            r#"{
                "id":"chatcmpl_01","object":"chat.completion","created":1,
                "model":"gpt-4o",
                "choices":[{"index":0,"message":{
                    "role":"assistant","content":null,
                    "tool_calls":[{"id":"call_01","type":"function",
                        "function":{"name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}}]
                },"finish_reason":"tool_calls"}],
                "usage":{"prompt_tokens":25,"completion_tokens":12,"total_tokens":37}
            }"#,
        ),
    )
    .await;
    let model = OpenAIChat::new(
        OpenAIConfig::new("test-key-sk-open")
            .with_base_url(base_url)
            .with_model("gpt-4o"),
    );
    let result = model
        .chat(vec![Message::human("weather")], None)
        .await
        .expect("openai non-stream succeeds");

    // Reasoning models keep `content` empty on a tool-call-only turn.
    assert!(result.content.is_empty());
    let calls = result.tool_calls.expect("tool call captured");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name(), "get_weather");
    assert!(calls[0].arguments().contains("Beijing"));
    let usage = result.token_usage.expect("usage captured");
    assert_eq!(
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        ),
        (25, 12, 37)
    );
}

#[tokio::test]
async fn openai_nonstream_reasoning_not_leaked_to_content() {
    // Q3: reasoning-only OpenAI-compatible response — `reasoning_content` may be
    // populated by thinking models while `content` stays empty. Guard that the
    // reasoning does not fall through into `content`.
    let base_url = mock_base_url(
        "/v1/chat/completions",
        MockResponse::json(
            200,
            r#"{
                "id":"chatcmpl_02","object":"chat.completion","created":1,
                "model":"deepseek-reasoner",
                "choices":[{"index":0,"message":{
                    "role":"assistant","content":"",
                    "reasoning_content":"chain of thought summary",
                    "tool_calls":null
                },"finish_reason":"stop"}],
                "usage":{"prompt_tokens":40,"completion_tokens":30,"total_tokens":70}
            }"#,
        ),
    )
    .await;
    let model = OpenAIChat::new(
        OpenAIConfig::new("test-key-sk-open")
            .with_base_url(base_url)
            .with_model("deepseek-reasoner"),
    );
    let result = model
        .chat(vec![Message::human("reason")], None)
        .await
        .expect("openai reasoning chat succeeds");

    assert_eq!(result.content, "", "reasoning must NOT leak into content");
    assert_eq!(
        result.thinking_content.as_deref(),
        Some("chain of thought summary")
    );
}

#[tokio::test]
async fn openai_error_status_mapping() {
    let base_url = mock_base_url(
        "/v1/chat/completions",
        MockResponse::json(
            404,
            r#"{"error":{"message":"model not found","type":"invalid_request_error"}}"#,
        ),
    )
    .await;
    let model = OpenAIChat::new(
        OpenAIConfig::new("test-key-sk-open")
            .with_base_url(base_url)
            .with_model("missing-model"),
    );
    let err = model
        .chat(vec![Message::human("x")], None)
        .await
        .expect_err("404 must surface an error");
    assert!(
        matches!(err, OpenAIError::Api(ref s) if s.contains("HTTP 404")),
        "expected HTTP 404 Api error, got {err:?}"
    );
}

// ────────────────── T5 (v0.23.0) request-body capture ──────────────────

/// Helper: every `captured_request` test needs to *drive* a chat call against
/// the loopback server (mere model construction never opens a socket). Each
/// caller supplies an async drive closure that builds the provider, calls
/// `chat(..)`, and `.await`s the request to completion; this helper awaits it,
/// then returns the request body the server observed, parsed as JSON.
async fn captured_request(
    route_path: &str,
    resp: MockResponse,
    drive: impl FnOnce(String) -> futures_util::future::BoxFuture<'static, ()>,
) -> serde_json::Value {
    let routes: Routes = routes::build(vec![(route_path.to_string(), resp)]);
    let server = common::spawn_mock_server_capture(routes).await;
    let base_url = format!("http://127.0.0.1:{}/v1", server.addr.port());
    drive(base_url).await;
    let body = server
        .last_body
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .expect("capture server observed a request body");
    serde_json::from_str(&body).expect("request body is valid JSON")
}

fn k_hello() -> MockResponse {
    MockResponse::json(
        200,
        r#"{
            "id":"msg_ttl","type":"message","role":"assistant",
            "model":"claude-sonnet-4-5",
            "content":[{"type":"text","text":"ok"}],
            "stop_reason":"end_turn",
            "usage":{"input_tokens":5,"output_tokens":1,
                     "cache_creation_input_tokens":0,"cache_read_input_tokens":0}
        }"#,
    )
}

fn two_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new("tool_a", "tool a")
            .with_parameters(json!({"type":"object","properties":{"x":{"type":"string"}}})),
        ToolDefinition::new("tool_b", "tool b")
            .with_parameters(json!({"type":"object","properties":{"y":{"type":"string"}}})),
    ]
}

#[tokio::test]
async fn anthropic_cache_ttl_breakpoints_in_request() {
    // with_prompt_cache_ttl("1h") must emit `cache_control: {"type":"ttl","ttl":"1h"}`
    // on the system block AND the last tool definition (not tool_a).
    let body = captured_request("/v1/messages", k_hello(), |base_url| {
        Box::pin(async move {
            let model = AnthropicChat::new(
                AnthropicConfig::new("test-key-sk-ant")
                    .with_base_url(base_url)
                    .with_model("claude-sonnet-4-5")
                    .with_system_prompt("You are terse.")
                    .with_prompt_cache_ttl("1h"),
            )
            .bind_tools(two_tools());
            model
                .chat(vec![Message::human("hi")], None)
                .await
                .expect("cached chat delivered offline");
        })
    })
    .await;

    let system = body["system"].as_array().expect("system present");
    assert_eq!(
        system[0]["cache_control"]["type"], "ttl",
        "system block uses long-lived TTL marker"
    );
    assert_eq!(system[0]["cache_control"]["ttl"], "1h");

    let tools = body["tools"].as_array().expect("tools present");
    assert_eq!(tools.len(), 2);
    assert!(
        tools[0].get("cache_control").is_none(),
        "only the last tool is cached (stable prefix)"
    );
    assert_eq!(tools[1]["cache_control"]["type"], "ttl");
    assert_eq!(tools[1]["cache_control"]["ttl"], "1h");
}

#[tokio::test]
async fn anthropic_strict_tool_choice_literal_in_request() {
    // Reserved literals (auto/any/required/none) map to {type: literal}; a
    // named tool maps to {type: tool, name}. T5 closed the gap where `required`
    // used to fall through to the `tool` branch.
    let body = captured_request("/v1/messages", k_hello(), |base_url| {
        Box::pin(async move {
            let model = AnthropicChat::new(
                AnthropicConfig::new("test-key-sk-ant")
                    .with_base_url(base_url)
                    .with_model("claude-sonnet-4-5"),
            )
            .bind_tools(two_tools())
            .with_tool_choice("required");
            model
                .chat(vec![Message::human("pick a tool")], None)
                .await
                .expect("required-choice chat delivered offline");
        })
    })
    .await;
    assert_eq!(
        body["tool_choice"],
        json!({"type": "required"}),
        "'required' is a reserved literal, not a tool name"
    );

    let body = captured_request("/v1/messages", k_hello(), |base_url| {
        Box::pin(async move {
            let model = AnthropicChat::new(
                AnthropicConfig::new("test-key-sk-ant")
                    .with_base_url(base_url)
                    .with_model("claude-sonnet-4-5"),
            )
            .bind_tools(two_tools())
            .with_tool_choice("tool_a");
            model
                .chat(vec![Message::human("use tool_a")], None)
                .await
                .expect("named-choice chat delivered offline");
        })
    })
    .await;
    assert_eq!(
        body["tool_choice"],
        json!({"type": "tool", "name": "tool_a"}),
        "a named tool choice uses the {{type, name}} form"
    );
}

#[tokio::test]
async fn openai_response_format_in_request() {
    // engine-side structured output: response_format must ride on the outward
    // request, not just shape the response parser.
    let body = captured_request(
        "/v1/chat/completions",
        MockResponse::json(
            200,
            r#"{
                "id":"chatcmpl_rf","object":"chat.completion","created":1,
                "model":"gpt-4o",
                "choices":[{"index":0,"message":{"role":"assistant","content":"{\"a\":1}"},
                            "finish_reason":"stop"}],
                "usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}
            }"#,
        ),
        |base_url| {
            Box::pin(async move {
                let model = OpenAIChat::new(
                    OpenAIConfig::new("test-key-sk-open")
                        .with_base_url(base_url)
                        .with_model("gpt-4o")
                        .with_response_format(ResponseFormat::JsonObject),
                );
                model
                    .chat(vec![Message::human("json please")], None)
                    .await
                    .expect("response_format chat delivered offline");
            })
        },
    )
    .await;
    assert_eq!(
        body["response_format"]["type"], "json_object",
        "request carries response_format for structured output"
    );
}

// ===========================================================================
// T1 (v0.25.0): live wire capture — the tape the USER records, CI never runs.
// ===========================================================================
//
// The loopback recorder (`common::spawn_live_recorder`) relays one request to a
// real upstream, captures the true wire response (status/headers/body), strips
// credential headers + redacts key-shaped tokens, and writes a fixture envelope
// to `tests/fixtures/<provider>/<case>.json`. Point it at a real provider with a
// real key to produce committed replay data. Replay itself is hermetic: the
// offline cassette tests above consume only committed fixtures.
//
// How to record (from the crate root):
//   RECORD_UPSTREAM=https://api.openai.com/v1                                \
//   OPENAI_API_KEY=sk-...                                                    \
//   RECORD_PROVIDER=openai RECORD_CASE=chat_completion                       \
//   cargo test -p lc-providers --test provider_cassettes record_chat_ -- --ignored --nocapture
//
// The checklist of providers / cases / env vars lives in
// docs/internal/v0.25.0/T1_FIXTURE_RECORDING.md.

#[tokio::test]
#[ignore = "records a live fixture; set RECORD_UPSTREAM + a real key to run"]
async fn record_chat_completion_fixture() {
    let Ok(upstream) = std::env::var("RECORD_UPSTREAM") else {
        eprintln!("RECORD_UPSTREAM not set; skipping live capture");
        return;
    };
    let provider = std::env::var("RECORD_PROVIDER").unwrap_or_else(|_| "openai".into());
    let case = std::env::var("RECORD_CASE").unwrap_or_else(|_| "chat_completion".into());

    // Start clean: a failed prior run may have left a non-200 fixture behind.
    let fixture_path = std::path::Path::new("tests/fixtures")
        .join(&provider)
        .join(format!("{case}.json"));
    if fixture_path.exists() {
        std::fs::remove_file(&fixture_path).ok();
    }

    let recorder = common::spawn_live_recorder(&upstream, &provider, &case).await;

    // The provider key overrides RECORD_API_KEY, else falls back to OPENAI_API_KEY.
    let key = std::env::var("RECORD_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .expect("set RECORD_API_KEY or OPENAI_API_KEY to record a fixture");

    let body = r#"{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":8}"#;
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let resp = client
        .post(format!(
            "http://127.0.0.1:{}/v1/chat/completions",
            recorder.addr.port()
        ))
        .header("content-type", "application/json")
        .bearer_auth(&key)
        .body(body)
        .send()
        .await
        .expect("relay response from loopback recorder");
    assert!(
        resp.status().is_success(),
        "upstream for {upstream} returned {}; pick a model the key can call",
        resp.status()
    );

    // The recorder wrote a sanitized fixture; load it back both ways.
    let loaded = common::load_fixture_opt(&provider, &case)
        .unwrap_or_else(|| panic!("recorder did not write {fixture_path:?}"));
    assert_eq!(loaded.status, 200);
    // Strict loader agrees the fixture now exists on disk.
    let _strict = common::load_fixture(&provider, &case);

    eprintln!(
        "\n[recorder] wrote fixture:\n  {}\n[recorder] sanitized body:\n{}",
        fixture_path.display(),
        loaded.body
    );
}
