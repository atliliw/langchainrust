use std::collections::HashMap;
#[derive(Debug, Clone)]
pub struct Document {
    pub content: String,
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(content: String) -> Self {
        Self {
            content,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub chunk_index: usize,
    pub document_id: Option<String>,
}

impl DocumentChunk {
    pub fn new(content: String, chunk_index: usize) -> Self {
        Self {
            content,
            metadata: HashMap::new(),
            chunk_index,
            document_id: None,
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn with_document_id(mut self, document_id: String) -> Self {
        self.document_id = Some(document_id);
        self
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: DocumentChunk,
    pub score: f32,
}

impl SearchResult {
    pub fn new(chunk: DocumentChunk, score: f32) -> Self {
        Self { chunk, score }
    }
}