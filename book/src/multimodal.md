# Multimodal

LangChainRust supports multimodal interactions including audio transcription (Whisper), text-to-speech (TTS), image generation (DALL-E), and vision with images.

## MultimodalModel Trait

```rust
#[async_trait]
pub trait MultimodalModel: BaseChatModel + Send + Sync {
    async fn transcribe(&self, audio: AudioContent) -> Result<String, MultimodalError>;
    async fn generate_speech(&self, text: &str) -> Result<Vec<u8>, MultimodalError>;
    async fn generate_image(&self, prompt: &str) -> Result<ImageContent, MultimodalError>;
}
```

All methods default to returning `MultimodalError::Unsupported`. Only OpenAI implements them currently.

## Content Types

| Type | Construction | Description |
|------|-------------|-------------|
| `AudioContent` | `from_url()`, `from_base64()` | Audio input for transcription |
| `ImageContent` | `from_url()`, `from_base64()` | Image URL or base64 data |
| `FileContent` | `from_url()`, `from_base64()`, `with_name()` | Generic file with MIME type |

```rust
use langchainrust::{AudioContent, ImageContent, FileContent};

let audio = AudioContent::from_url("https://example.com/audio.wav");
let audio = AudioContent::from_base64("base64data...");

let image = ImageContent::from_url("https://example.com/photo.jpg");
let image = ImageContent::from_base64("iVBORw0KGgo...");

let file = FileContent::from_url("https://example.com/doc.pdf")
    .with_name("report.pdf");
```

## Whisper (Speech-to-Text)

```rust
use langchainrust::MultimodalModel;

let audio = AudioContent::from_base64(audio_base64);
let transcript = llm.transcribe(audio).await?;
```

OpenAI-specific with voice options:

```rust
use langchainrust::openai::OpenAIChat;

// Provider-specific method with more control
let transcript = llm.whisper_transcribe(audio).await?;
```

## TTS (Text-to-Speech)

```rust
use langchainrust::MultimodalModel;

let speech = llm.generate_speech("Hello, world!").await?;

// Provider-specific with voice selection
use langchainrust::openai::TtsVoice;
let speech = llm.tts_generate("Hello!", TtsVoice::Alloy).await?;
// Voices: Alloy, Echo, Fable, Onyx, Nova, Shimmer
```

## DALL-E (Image Generation)

```rust
use langchainrust::MultimodalModel;

let image = llm.generate_image("A cat wearing a hat").await?;

// Provider-specific with size control
use langchainrust::openai::DallEImageSize;
let image = llm.dalle_generate("A cat wearing a hat", DallEImageSize::S1024).await?;
// Sizes: S256, S512, S1024, S1792x1024, S1024x1792
```

## Vision (Image Understanding)

```rust
use langchainrust::Message;

let messages = vec![
    Message::system("Describe what you see."),
    Message::human_with_image("What is in this image?", "https://example.com/cat.jpg"),
];
let response = llm.chat(messages, None).await?;
```

## Anthropic Images

Anthropic supports image inputs via the same `Message::human_with_image` pattern. The image content is formatted according to the Anthropic API specification.
