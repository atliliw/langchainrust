use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A single content element of an A2A message or task artifact (A2A v0.3
/// `parts` model).
///
/// A2A v0.3 replaces a bare string message body with an array of heterogeneous
/// `Part`s, each internally tagged by a `type` discriminator (`text` / `file`
/// / `data`). Every variant optionally carries a `metadata` object for
/// transport-neutral annotations.
///
/// Wire shape:
///
/// ```json
/// { "type": "text", "text": "hello", "metadata": { "lang": "en" } }
/// { "type": "file", "file": { "name": "a.txt", "mimeType": "text/plain", "uri": "..." } }
/// { "type": "data", "data": { "count": 3 } }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Part {
    /// Plain-text content.
    Text {
        /// The text payload.
        text: String,
        /// Optional annotations (kept on the wire only when present).
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },
    /// A file reference or inline file content.
    File {
        /// The file descriptor (name, mime type, URI / bytes).
        file: FilePart,
        /// Optional annotations (kept on the wire only when present).
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },
    /// Structured (JSON) data.
    Data {
        /// The structured payload.
        data: Value,
        /// Optional annotations (kept on the wire only when present).
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },
}

impl Part {
    /// A text part without metadata.
    pub fn text(text: impl Into<String>) -> Self {
        Part::Text {
            text: text.into(),
            metadata: None,
        }
    }

    /// A text part carrying metadata.
    pub fn text_with_metadata(text: impl Into<String>, metadata: Value) -> Self {
        Part::Text {
            text: text.into(),
            metadata: Some(metadata),
        }
    }

    /// A file part without metadata.
    pub fn file(file: FilePart) -> Self {
        Part::File {
            file,
            metadata: None,
        }
    }

    /// A structured-data part without metadata.
    pub fn data(data: Value) -> Self {
        Part::Data {
            data,
            metadata: None,
        }
    }

    /// A structured-data part carrying metadata.
    pub fn data_with_metadata(data: Value, metadata: Value) -> Self {
        Part::Data {
            data,
            metadata: Some(metadata),
        }
    }

    /// The wire `type` discriminator of this part (`text`, `file`, or `data`).
    pub fn kind(&self) -> &'static str {
        match self {
            Part::Text { .. } => "text",
            Part::File { .. } => "file",
            Part::Data { .. } => "data",
        }
    }

    /// The text content, when this is a text part.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Part::Text { text, .. } => Some(text.as_str()),
            _ => None,
        }
    }

    /// The file descriptor, when this is a file part.
    pub fn as_file(&self) -> Option<&FilePart> {
        match self {
            Part::File { file, .. } => Some(file),
            _ => None,
        }
    }

    /// The structured payload, when this is a data part.
    pub fn as_data(&self) -> Option<&Value> {
        match self {
            Part::Data { data, .. } => Some(data),
            _ => None,
        }
    }
}

/// A file descriptor carried by [`Part::File`] (A2A v0.3 `FilePart`).
///
/// Either a `uri` reference or the raw `bytes` (base64-encoded) may be
/// populated; both are optional to support servers that return one or the
/// other. `size` is the byte length when known.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilePart {
    /// File name.
    pub name: String,
    /// MIME type of the file (serialized as `mimeType`).
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    /// Remote URI the file can be downloaded from (when not inlined).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// Inline file content, base64-encoded (when not referenced by URI).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<String>,
    /// File size in bytes, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Optional annotations (kept on the wire only when present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl FilePart {
    /// Create a file descriptor from its required name and MIME type.
    pub fn new(name: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            mime_type: mime_type.into(),
            uri: None,
            bytes: None,
            size: None,
            metadata: None,
        }
    }

    /// Attach a remote URI for downloading the file.
    pub fn with_uri(mut self, uri: impl Into<String>) -> Self {
        self.uri = Some(uri.into());
        self
    }

    /// Attach inline base64 content.
    pub fn with_bytes(mut self, bytes: impl Into<String>) -> Self {
        self.bytes = Some(bytes.into());
        self
    }

    /// Attach the file size in bytes.
    pub fn with_size(mut self, size: u64) -> Self {
        self.size = Some(size);
        self
    }

    /// Attach metadata annotations.
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// A task artifact: the output content of a completed task (A2A v0.3
/// `TaskArtifact`).
///
/// Unlike the legacy single-string [`super::model::A2ATaskResult`], an artifact
/// carries a heterogeneous array of [`Part`]s, so a task may return text,
/// files, and structured data together. Used as the payload of push
/// [`crate::protocol::TaskPushNotification::ArtifactUpdate`] events and available on the pull
/// path for clients that stream results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct A2AArtifact {
    /// Unique artifact identifier (serialized as `artifactId`).
    #[serde(rename = "artifactId")]
    artifact_id: String,
    /// The heterogeneous output content.
    pub parts: Vec<Part>,
    /// Optional annotations (kept on the wire only when present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl A2AArtifact {
    /// Create an artifact from its parts, with an empty artifact id until
    /// [`A2AArtifact::with_artifact_id`] is applied.
    pub fn new(parts: Vec<Part>) -> Self {
        Self {
            artifact_id: String::new(),
            parts,
            metadata: None,
        }
    }

    /// Create an artifact from a single text part.
    pub fn text(artifact_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            artifact_id: artifact_id.into(),
            parts: vec![Part::text(text)],
            metadata: None,
        }
    }

    /// Set the artifact identifier.
    pub fn with_artifact_id(mut self, artifact_id: impl Into<String>) -> Self {
        self.artifact_id = artifact_id.into();
        self
    }

    /// Attach metadata annotations.
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// The artifact identifier.
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    /// The heterogeneous output parts.
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// Concatenate all text-part contents (the "plain text" reading of an
    /// artifact), returning `None` when no text part is present.
    pub fn text_content(&self) -> Option<String> {
        let texts: Vec<&str> = self.parts.iter().filter_map(Part::as_text).collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join("\n"))
        }
    }
}
