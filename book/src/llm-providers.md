# LLM Providers

LangChainRust supports 10+ LLM providers through a unified `BaseChatModel` trait.

## Supported Providers

| Provider | Config | Auth |
|----------|--------|------|
| OpenAI | `OpenAIChat` | `OPENAI_API_KEY` |
| Anthropic | `AnthropicChat` | `ANTHROPIC_API_KEY` |
| Gemini | `GeminiChat` | `GEMINI_API_KEY` |
| Ollama | `OllamaChat` | Local (no key) |
| DeepSeek | `DeepSeekChat` | `DEEPSEEK_API_KEY` |
| Qwen | `QwenChat` | `QWEN_API_KEY` |
| Moonshot | `MoonshotChat` | `MOONSHOT_API_KEY` |
| Zhipu | `ZhipuChat` | `ZHIPU_API_KEY` |
| Mistral | `MistralChat` | `MISTRAL_API_KEY` |
| Azure OpenAI | `AzureOpenAIChat` | `AZURE_OPENAI_API_KEY` |
| Cohere | `CohereChat` | `COHERE_API_KEY` |

## Auto-Detection

`LLMClient::from_env()` auto-detects the provider from environment variables:

```rust
use langchainrust::LLMClient;

let llm = LLMClient::from_env().expect("Set an API key");
```

Priority: OpenAI → Anthropic → Mistral → Ollama (local fallback).

## OpenAI-Compatible Wrappers

DeepSeek, Qwen, Moonshot, Zhipu, and Mistral all wrap `OpenAIChat` internally
via `into_openai_config()` — they use the same chat format but with different
base URLs and default models.

## Native Providers

Azure OpenAI and Cohere have native implementations because their URL/auth
format differs from OpenAI's.

## Streaming

All providers support streaming via `stream_chat()`:

```rust
let stream = llm.stream_chat(messages, None).await?;
while let Some(token) = stream.next().await {
    print!("{}", token?.content);
}
```

## Multimodal

OpenAI supports Whisper (STT), TTS, and DALL-E via the `MultimodalModel` trait:

```rust
use langchainrust::MultimodalModel;

let transcript = llm.transcribe(audio).await?;
let speech = llm.generate_speech("Hello!").await?;
let image = llm.generate_image("A cat wearing a hat").await?;
```

## Retry & Fallback

```rust
use langchainrust::runnables::{RetryConfig, RunnableExt};

let llm_with_retry = llm.with_retry(RetryConfig::new(3));
```
