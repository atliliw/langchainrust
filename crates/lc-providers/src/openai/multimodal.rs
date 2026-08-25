// lc-providers/src/openai/multimodal.rs
//! OpenAI multimodal capabilities: Whisper (STT), TTS, DALL-E.
//!
//! Implements `MultimodalModel` for `OpenAIChat`, providing:
//! - `transcribe()` — Whisper speech-to-text
//! - `generate_speech()` — TTS text-to-speech
//! - `generate_image()` — DALL-E image generation

use async_trait::async_trait;
use lc_core::language_models::{MultimodalError, MultimodalModel};
use lc_schema::{AudioContent, ImageContent};
use serde::Deserialize;

use super::chat::OpenAIChat;

/// Whisper API response.
#[derive(Debug, Deserialize)]
struct WhisperResponse {
    text: String,
}

// TTS is returned as raw bytes — no structured response needed.

/// DALL-E API response.
#[derive(Debug, Deserialize)]
struct DallEImage {
    url: Option<String>,
    b64_json: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DallEResponse {
    data: Vec<DallEImage>,
}

/// TTS voice options.
#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsVoice {
    /// The "alloy" voice.
    Alloy,
    /// The "echo" voice.
    Echo,
    /// The "fable" voice.
    Fable,
    /// The "onyx" voice.
    Onyx,
    /// The "nova" voice.
    Nova,
    /// The "shimmer" voice.
    Shimmer,
}

impl TtsVoice {
    /// Returns the API string for this voice.
    pub fn as_str(&self) -> &'static str {
        match self {
            TtsVoice::Alloy => "alloy",
            TtsVoice::Echo => "echo",
            TtsVoice::Fable => "fable",
            TtsVoice::Onyx => "onyx",
            TtsVoice::Nova => "nova",
            TtsVoice::Shimmer => "shimmer",
        }
    }
}

/// DALL-E image size options.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum DallEImageSize {
    /// 256x256 image size.
    #[serde(rename = "256x256")]
    S256,
    /// 512x512 image size.
    #[serde(rename = "512x512")]
    S512,
    /// 1024x1024 image size.
    #[serde(rename = "1024x1024")]
    S1024,
    /// 1792x1024 image size.
    #[serde(rename = "1792x1024")]
    S1792x1024,
    /// 1024x1792 image size.
    #[serde(rename = "1024x1792")]
    S1024x1792,
}

impl DallEImageSize {
    /// Returns the API string for this size.
    pub fn as_str(&self) -> &'static str {
        match self {
            DallEImageSize::S256 => "256x256",
            DallEImageSize::S512 => "512x512",
            DallEImageSize::S1024 => "1024x1024",
            DallEImageSize::S1792x1024 => "1792x1024",
            DallEImageSize::S1024x1792 => "1024x1792",
        }
    }
}

/// OpenAI multimodal extensions.
///
/// These methods are on `OpenAIChat` directly (not through the trait)
/// because they require provider-specific parameters (voice, size, etc.).
impl OpenAIChat {
    /// Transcribes audio using Whisper.
    ///
    /// Sends audio to the `/v1/audio/transcriptions` endpoint.
    pub async fn whisper_transcribe(&self, audio: AudioContent) -> Result<String, MultimodalError> {
        let url = format!("{}/audio/transcriptions", self.config.base_url);

        let audio_data = if audio.is_base64() {
            // Decode base64 data
            let b64 = audio.base64_data().unwrap_or("");
            base64_decode(b64)?
        } else {
            // Fetch from URL
            let response = self
                .client
                .get(&audio.url)
                .send()
                .await
                .map_err(|e| MultimodalError::HttpError(e.to_string()))?;
            response
                .bytes()
                .await
                .map_err(|e| MultimodalError::HttpError(e.to_string()))?
                .to_vec()
        };

        let part = reqwest::multipart::Part::bytes(audio_data)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| MultimodalError::HttpError(e.to_string()))?;

        let form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", "whisper-1");

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| MultimodalError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MultimodalError::ApiError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let whisper_response: WhisperResponse = response
            .json()
            .await
            .map_err(|e| MultimodalError::ParseError(e.to_string()))?;

        Ok(whisper_response.text)
    }

    /// Generates speech using OpenAI TTS.
    ///
    /// Sends text to the `/v1/audio/speech` endpoint and returns raw audio bytes.
    pub async fn tts_generate(
        &self,
        text: &str,
        voice: TtsVoice,
    ) -> Result<Vec<u8>, MultimodalError> {
        let url = format!("{}/audio/speech", self.config.base_url);

        let body = serde_json::json!({
            "model": "tts-1",
            "input": text,
            "voice": voice.as_str(),
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| MultimodalError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MultimodalError::ApiError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| MultimodalError::HttpError(e.to_string()))?;

        Ok(bytes.to_vec())
    }

    /// Generates an image using DALL-E.
    ///
    /// Sends a prompt to the `/v1/images/generations` endpoint.
    pub async fn dalle_generate(
        &self,
        prompt: &str,
        size: DallEImageSize,
    ) -> Result<ImageContent, MultimodalError> {
        let url = format!("{}/images/generations", self.config.base_url);

        let body = serde_json::json!({
            "model": "dall-e-3",
            "prompt": prompt,
            "n": 1,
            "size": size.as_str(),
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| MultimodalError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MultimodalError::ApiError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let dalle_response: DallEResponse = response
            .json()
            .await
            .map_err(|e| MultimodalError::ParseError(e.to_string()))?;

        let image = dalle_response
            .data
            .first()
            .ok_or_else(|| MultimodalError::ApiError("No image in response".to_string()))?;

        if let Some(url) = &image.url {
            Ok(ImageContent::from_url(url))
        } else if let Some(b64) = &image.b64_json {
            Ok(ImageContent::from_base64(b64))
        } else {
            Err(MultimodalError::ApiError(
                "No image URL or base64 data in response".to_string(),
            ))
        }
    }
}

/// Implement MultimodalModel trait for OpenAIChat.
#[async_trait]
impl MultimodalModel for OpenAIChat {
    async fn transcribe(&self, audio: AudioContent) -> Result<String, MultimodalError> {
        self.whisper_transcribe(audio).await
    }

    async fn generate_speech(&self, text: &str) -> Result<Vec<u8>, MultimodalError> {
        self.tts_generate(text, TtsVoice::Alloy).await
    }

    async fn generate_image(&self, prompt: &str) -> Result<ImageContent, MultimodalError> {
        self.dalle_generate(prompt, DallEImageSize::S1024).await
    }
}

/// Helper to decode base64 string.
fn base64_decode(input: &str) -> Result<Vec<u8>, MultimodalError> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(input)
        .map_err(|e| MultimodalError::ParseError(format!("Base64 decode error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_voice_str() {
        assert_eq!(TtsVoice::Alloy.as_str(), "alloy");
        assert_eq!(TtsVoice::Shimmer.as_str(), "shimmer");
    }

    #[test]
    fn test_dalle_size_str() {
        assert_eq!(DallEImageSize::S256.as_str(), "256x256");
        assert_eq!(DallEImageSize::S1024.as_str(), "1024x1024");
        assert_eq!(DallEImageSize::S1792x1024.as_str(), "1792x1024");
    }

    #[test]
    fn test_base64_decode_valid() {
        let decoded = base64_decode("aGVsbG8=").unwrap();
        assert_eq!(String::from_utf8_lossy(&decoded), "hello");
    }

    #[test]
    fn test_base64_decode_invalid() {
        let result = base64_decode("!!!invalid!!!");
        assert!(result.is_err());
    }
}
