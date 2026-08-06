# Embeddings

LangChainRust provides a unified `Embeddings` trait with multiple backends for generating text embedding vectors.

## Supported Providers

| Provider | Struct | Default Model | Dimension | Feature Gate |
|----------|--------|---------------|-----------|-------------|
| OpenAI | `OpenAIEmbeddings` | `text-embedding-ada-002` | 1536 | always |
| DeepSeek | `DeepSeekEmbeddings` | `deepseek-embedding` | 1536 | always |
| Qwen | `QwenEmbeddings` | `text-embedding-v1` | 1536 | always |
| Cohere | `CohereEmbeddings` | `embed-english-v3.0` | 1024 | always |
| FastEmbed | `FastEmbedEmbeddings` | `BGESmallENV15` | 384 | `fastembed` |
| BagOfWords | `BagOfWordsEmbeddings` | `local-bow` | configurable | always |
| Mock | `MockEmbeddings` | `mock-embeddings` | configurable | always |

## Embeddings Trait

```rust
#[async_trait]
pub trait Embeddings: Send + Sync {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
}
```

## OpenAI Embeddings

```rust
use langchainrust::{OpenAIEmbeddings, OpenAIEmbeddingsConfig};

// From environment (OPENAI_API_KEY)
let embedder = OpenAIEmbeddings::from_env_result()?;

// Explicit config
let embedder = OpenAIEmbeddings::new(
    OpenAIEmbeddingsConfig::new("sk-...")
        .with_model("text-embedding-3-small")
        .with_base_url("https://api.openai.com/v1"),
);

let vec = embedder.embed_query("What is Rust?").await?;
let vecs = embedder.embed_documents(&["doc1 text", "doc2 text"]).await?;
```

## FastEmbed (Local ONNX)

```rust
use langchainrust::FastEmbedEmbeddings;
use fastembed::EmbeddingModel;

// Default model (BGESmallENV15, 384-dim)
let embedder = FastEmbedEmbeddings::default_model()?;

// Specific model
let embedder = FastEmbedEmbeddings::with_model(EmbeddingModel::BGESmallENV15)?;
let vec = embedder.embed_query("hello world").await?;
```

## BagOfWords & Mock

```rust
use langchainrust::{BagOfWordsEmbeddings, MockEmbeddings};

// BagOfWords (no API needed, FNV-1a hash based)
let embedder = BagOfWordsEmbeddings::default_dim(); // 256-dim
let embedder = BagOfWordsEmbeddings::new(128);

// Mock (deterministic pseudo-random for testing)
let embedder = MockEmbeddings::new(1536);
```

## Cosine Similarity

```rust
use langchainrust::cosine_similarity;

let sim = cosine_similarity(&vec_a, &vec_b)?;
// Returns f64 in range [-1.0, 1.0]
```
