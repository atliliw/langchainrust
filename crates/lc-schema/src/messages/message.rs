//! Message data structures for chat models.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use lc_shared::tools::ToolCall;

use super::audio::AudioContent;
use super::file::FileContent;
use super::image::ImageContent;
use super::video::VideoContent;

/// Modality of a media attachment on a multimodal message.
///
/// B7 (v0.22.4): the common vocabulary used by the provider-neutral
/// [`MediaPart`] view; each chat backend maps these onto its own wire blocks
/// (`image_url` / `input_audio` / `input_video` / `file`, Anthropic
/// `image`/`document`, Gemini `inline_data`/`file_data`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Still image (vision input).
    Image,
    /// Audio clip (e.g. voice input / transcription inside a chat turn).
    Audio,
    /// Video clip.
    Video,
    /// Attached file/document (e.g. PDF).
    File,
}

/// One media attachment of a [`Message`], exposed by [`Message::media_parts`].
///
/// This is the provider-neutral unified view (B7, v0.22.4) over the parallel
/// `images` / `audio` / `videos` / `files` vectors: provider request builders
/// iterate [`Message::media_parts`] once instead of each growing their own
/// modality-specific special cases.
#[derive(Debug, Clone, PartialEq)]
pub enum MediaPart<'a> {
    /// An image attachment.
    Image(&'a ImageContent),
    /// An audio attachment.
    Audio(&'a AudioContent),
    /// A video attachment.
    Video(&'a VideoContent),
    /// A file/document attachment.
    File(&'a FileContent),
}

impl<'a> MediaPart<'a> {
    /// Returns the attachment's modality.
    pub fn modality(&self) -> Modality {
        match self {
            MediaPart::Image(_) => Modality::Image,
            MediaPart::Audio(_) => Modality::Audio,
            MediaPart::Video(_) => Modality::Video,
            MediaPart::File(_) => Modality::File,
        }
    }

    /// Returns the raw value (https URL, `gs://` URI, or `data:` URI).
    pub fn url(&self) -> &str {
        match self {
            MediaPart::Image(m) => &m.url,
            MediaPart::Audio(m) => &m.url,
            MediaPart::Video(m) => &m.url,
            MediaPart::File(m) => &m.url,
        }
    }

    /// Returns the explicit MIME type if the attachment carries one.
    ///
    /// [`FileContent`] can store an out-of-band MIME type; for data URIs the
    /// type embedded in the `data:` prefix is parsed instead.
    pub fn mime_type(&self) -> Option<&str> {
        match self {
            MediaPart::File(f) => f.mime_type.as_deref().or_else(|| data_uri_mime(&f.url)),
            MediaPart::Image(i) => data_uri_mime(&i.url),
            MediaPart::Audio(a) => data_uri_mime(&a.url),
            MediaPart::Video(v) => data_uri_mime(&v.url),
        }
    }

    /// Returns the optional filename (file attachments only).
    pub fn name(&self) -> Option<&str> {
        match self {
            MediaPart::File(f) => f.name.as_deref(),
            _ => None,
        }
    }

    /// Returns the raw base64 payload if the value is a base64 data URI.
    pub fn base64_data(&self) -> Option<&str> {
        let url = self.url();
        url.split_once(',')
            .filter(|(prefix, _)| prefix.contains("base64"))
            .map(|(_, data)| data)
    }

    /// Returns whether the value is a `data:` URI.
    pub fn is_data_uri(&self) -> bool {
        self.url().starts_with("data:")
    }
}

/// Extracts the MIME segment from a data URI (`data:<mime>;base64,...`).
pub(crate) fn data_uri_mime(url: &str) -> Option<&str> {
    url.strip_prefix("data:")?
        .split([';', ','])
        .next()
        .filter(|mime| !mime.is_empty())
}

/// Message type classification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    /// System message
    System,
    /// Human (user) message
    Human,
    /// AI (assistant) message
    AI,
    /// Tool result message, carrying the matching tool_call_id
    Tool {
        /// Associated tool call ID
        tool_call_id: String,
    },
}

/// Complete message structure for chat interactions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    /// Message text content
    pub content: String,

    /// Image content (multimodal vision)
    #[serde(default)]
    pub images: Vec<ImageContent>,

    /// Audio content (multimodal audio)
    #[serde(default)]
    pub audio: Vec<AudioContent>,

    /// Video content (multimodal video) — B7 (v0.22.4)
    #[serde(default)]
    pub videos: Vec<VideoContent>,

    /// File content (multimodal document)
    #[serde(default)]
    pub files: Vec<FileContent>,

    /// Message type
    #[serde(rename = "type")]
    pub message_type: MessageType,

    /// Message name (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Additional keyword arguments
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_kwargs: HashMap<String, Value>,

    /// Message ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Tool call list (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// Creates a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            audio: Vec::new(),
            videos: Vec::new(),
            files: Vec::new(),
            message_type: MessageType::System,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a human (user) message.
    pub fn human(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            audio: Vec::new(),
            videos: Vec::new(),
            files: Vec::new(),
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a human message with an image (vision).
    pub fn human_with_image(content: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: vec![ImageContent::from_url(image_url)],
            audio: Vec::new(),
            videos: Vec::new(),
            files: Vec::new(),
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a human message with multiple images.
    pub fn human_with_images(content: impl Into<String>, images: Vec<ImageContent>) -> Self {
        Self {
            content: content.into(),
            images,
            audio: Vec::new(),
            videos: Vec::new(),
            files: Vec::new(),
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a human message with audio content.
    pub fn human_with_audio(content: impl Into<String>, audio: AudioContent) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            audio: vec![audio],
            videos: Vec::new(),
            files: Vec::new(),
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a human message with video content.
    pub fn human_with_video(content: impl Into<String>, video: VideoContent) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            audio: Vec::new(),
            videos: vec![video],
            files: Vec::new(),
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a human message with file content.
    pub fn human_with_file(content: impl Into<String>, file: FileContent) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            audio: Vec::new(),
            videos: Vec::new(),
            files: vec![file],
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates an AI (assistant) message.
    pub fn ai(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            audio: Vec::new(),
            videos: Vec::new(),
            files: Vec::new(),
            message_type: MessageType::AI,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates an AI message with tool calls.
    pub fn ai_with_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            audio: Vec::new(),
            videos: Vec::new(),
            files: Vec::new(),
            message_type: MessageType::AI,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: Some(tool_calls),
        }
    }

    /// Creates a tool result message.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            audio: Vec::new(),
            videos: Vec::new(),
            files: Vec::new(),
            message_type: MessageType::Tool {
                tool_call_id: tool_call_id.into(),
            },
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Sets the message name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the message ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Adds an additional keyword argument.
    pub fn with_additional_kwarg(mut self, key: impl Into<String>, value: Value) -> Self {
        self.additional_kwargs.insert(key.into(), value);
        self
    }

    /// Adds an image to the message (vision).
    pub fn with_image(mut self, image: ImageContent) -> Self {
        self.images.push(image);
        self
    }

    /// Adds audio content to the message.
    pub fn with_audio(mut self, audio: AudioContent) -> Self {
        self.audio.push(audio);
        self
    }

    /// Adds video content to the message.
    pub fn with_video(mut self, video: VideoContent) -> Self {
        self.videos.push(video);
        self
    }

    /// Adds file content to the message.
    pub fn with_file(mut self, file: FileContent) -> Self {
        self.files.push(file);
        self
    }

    /// Returns whether the message has images.
    pub fn has_images(&self) -> bool {
        !self.images.is_empty()
    }

    /// Returns whether the message has audio content.
    pub fn has_audio(&self) -> bool {
        !self.audio.is_empty()
    }

    /// Returns whether the message has video content.
    pub fn has_videos(&self) -> bool {
        !self.videos.is_empty()
    }

    /// Returns whether the message has file content.
    pub fn has_files(&self) -> bool {
        !self.files.is_empty()
    }

    /// Returns whether the message has any multimodal content (images, audio,
    /// video, or files).
    pub fn is_multimodal(&self) -> bool {
        self.has_images() || self.has_audio() || self.has_videos() || self.has_files()
    }

    /// Provider-neutral view over every attachment, in canonical order
    /// (images → audio → video → files, each in insertion order).
    ///
    /// B7 (v0.22.4): chat backends map this single stream onto their wire
    /// blocks instead of special-casing the parallel content vectors.
    pub fn media_parts(&self) -> Vec<MediaPart<'_>> {
        let mut parts: Vec<MediaPart<'_>> = Vec::with_capacity(
            self.images.len() + self.audio.len() + self.videos.len() + self.files.len(),
        );
        parts.extend(self.images.iter().map(MediaPart::Image));
        parts.extend(self.audio.iter().map(MediaPart::Audio));
        parts.extend(self.videos.iter().map(MediaPart::Video));
        parts.extend(self.files.iter().map(MediaPart::File));
        parts
    }

    /// Returns the message type as a string.
    ///
    /// Tool messages include their `tool_call_id` (e.g. `"tool:call_123"`) so
    /// the type string is unambiguous about which tool result the message holds.
    pub fn type_str(&self) -> String {
        match &self.message_type {
            MessageType::System => "system".to_string(),
            MessageType::Human => "human".to_string(),
            MessageType::AI => "ai".to_string(),
            MessageType::Tool { tool_call_id } => format!("tool:{tool_call_id}"),
        }
    }

    /// Returns whether the message has tool calls.
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_deref().is_some_and(|t| !t.is_empty())
    }

    /// Returns the tool calls if present.
    pub fn get_tool_calls(&self) -> Option<&[ToolCall]> {
        self.tool_calls.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_human_with_image() {
        let msg = Message::human_with_image("描述这张图", "https://example.com/img.jpg");
        assert_eq!(msg.content, "描述这张图");
        assert_eq!(msg.images.len(), 1);
        assert_eq!(msg.images[0].url, "https://example.com/img.jpg");
        assert!(msg.has_images());
    }

    #[test]
    fn test_human_no_images_by_default() {
        let msg = Message::human("纯文本");
        assert!(msg.images.is_empty());
        assert!(!msg.has_images());
    }

    #[test]
    fn test_with_image_builder() {
        let msg = Message::human("看图")
            .with_image(ImageContent::from_url("https://example.com/a.png"))
            .with_image(ImageContent::from_base64("abc"));
        assert_eq!(msg.images.len(), 2);
    }

    #[test]
    fn test_message_deserialize_without_images_field() {
        // The old format (no images field) must still deserialize (#[serde(default)])
        let json = r#"{"content":"hi","type":"human"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content, "hi");
        assert!(msg.images.is_empty());
    }

    #[test]
    fn test_human_with_images_multiple() {
        let msg = Message::human_with_images(
            "多图",
            vec![
                ImageContent::from_url("https://example.com/1.jpg"),
                ImageContent::from_url("https://example.com/2.jpg"),
            ],
        );
        assert_eq!(msg.images.len(), 2);
    }

    #[test]
    fn test_system_ai_no_images() {
        assert!(Message::system("s").images.is_empty());
        assert!(Message::ai("a").images.is_empty());
        assert!(Message::tool("id", "c").images.is_empty());
    }

    #[test]
    fn test_type_str_includes_tool_call_id() {
        assert_eq!(Message::system("s").type_str(), "system");
        assert_eq!(Message::human("h").type_str(), "human");
        assert_eq!(Message::ai("a").type_str(), "ai");
        assert_eq!(
            Message::tool("call_123", "result").type_str(),
            "tool:call_123"
        );
    }

    #[test]
    fn test_has_tool_calls_empty_and_present() {
        let with_calls = Message::ai_with_tool_calls(
            "call tool",
            vec![ToolCall::builder("call_1")
                .name("weather")
                .arguments(r#"{"city":"beijing"}"#)
                .build()],
        );
        assert!(with_calls.has_tool_calls());
        assert_eq!(with_calls.get_tool_calls().unwrap().len(), 1);

        // No panic on None or on an empty vec
        assert!(!Message::ai("plain").has_tool_calls());
        let empty = Message::ai_with_tool_calls("no calls", vec![]);
        assert!(!empty.has_tool_calls());
    }

    // --- B7: video + unified MediaPart view ---

    #[test]
    fn test_human_with_video() {
        let msg = Message::human_with_video(
            "看视频",
            VideoContent::from_url("https://example.com/clip.mp4"),
        );
        assert!(msg.has_videos());
        assert!(msg.is_multimodal());
        assert_eq!(msg.videos.len(), 1);
        assert_eq!(msg.videos[0].url, "https://example.com/clip.mp4");
    }

    #[test]
    fn test_with_video_builder() {
        let msg = Message::human("视频")
            .with_video(VideoContent::from_url("https://example.com/a.mp4"))
            .with_video(VideoContent::from_base64("abc"));
        assert_eq!(msg.videos.len(), 2);
    }

    #[test]
    fn test_video_field_defaults_on_old_json() {
        // Messages serialized before the videos field existed must still deserialize.
        let json = r#"{"content":"hi","type":"human","images":[],"audio":[],"files":[]}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert!(msg.videos.is_empty());
        assert!(!msg.has_videos());
    }

    #[test]
    fn test_media_parts_canonical_order_and_modality() {
        let msg = Message::human("mixed")
            .with_image(ImageContent::from_url("https://e.com/a.png"))
            .with_audio(AudioContent::from_url("https://e.com/a.mp3"))
            .with_video(VideoContent::from_url("https://e.com/a.mp4"))
            .with_file(FileContent::from_base64("abc", "application/pdf"));

        let parts = msg.media_parts();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0].modality(), Modality::Image);
        assert_eq!(parts[1].modality(), Modality::Audio);
        assert_eq!(parts[2].modality(), Modality::Video);
        assert_eq!(parts[3].modality(), Modality::File);
        assert_eq!(parts[2].url(), "https://e.com/a.mp4");
        assert_eq!(parts[3].name(), None);
        assert_eq!(parts[3].mime_type(), Some("application/pdf"));
        assert_eq!(parts[3].base64_data(), Some("abc"));
    }

    #[test]
    fn test_media_part_data_uri_mime() {
        let img = ImageContent::from_base64_with_mime("zzz", "image/webp");
        let msg = Message::human("h").with_image(img);
        let part = &msg.media_parts()[0];
        assert!(part.is_data_uri());
        assert_eq!(part.mime_type(), Some("image/webp"));
        assert_eq!(part.base64_data(), Some("zzz"));

        // Plain URL: no embedded MIME, no base64.
        let plain = Message::human_with_image("h", "https://example.com/x.jpg");
        let p = &plain.media_parts()[0];
        assert!(!p.is_data_uri());
        assert_eq!(p.mime_type(), None);
    }

    #[test]
    fn test_file_part_explicit_mime_and_name() {
        let file = FileContent::from_url_with_mime("https://e.com/d.pdf", "application/pdf")
            .with_name("d.pdf");
        let msg = Message::human_with_file("读文件", file);
        let part = &msg.media_parts()[0];
        assert_eq!(part.mime_type(), Some("application/pdf"));
        assert_eq!(part.name(), Some("d.pdf"));
    }
}
