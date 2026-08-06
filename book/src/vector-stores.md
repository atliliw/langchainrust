# Vector Stores

LangChainRust provides a unified `VectorStore` trait with 10+ backends for storing and retrieving document embeddings.

## Supported Backends

| Store | Struct | Feature Gate | Auth |
|-------|--------|-------------|------|
| InMemory | `InMemoryVectorStore` | always | None |
| File | `FileVectorStore` | always | None |
| ChromaDB | `ChromaDBVectorStore` | always | `http://localhost:8000` |
| Pinecone | `PineconeStore` | always | API key + host |
| LanceDB | `LanceDBVectorStore` | always | Optional API key |
| Neo4j | `Neo4jVectorStore` | always | bolt:// + password |
| Qdrant | `QdrantVectorStore` | `qdrant-integration` | `http://localhost:6334` |
| PGVector | `pgvector` (helpers) | `pgvector-storage` | sqlx connection |
| Redis | `RedisDocumentStore` | `redis-storage` | Redis URL |
| SQLite | `SQLiteDocumentStore` | `sqlite-storage` | File path |
| MongoDB | `MongoChunkedDocumentStore` | `mongodb-persistence` | MongoDB URI |

## VectorStore Trait

```rust
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add_documents(&self, documents: Vec<Document>, embeddings: Vec<Vec<f32>>) -> Result<Vec<String>, VectorStoreError>;
    async fn similarity_search(&self, query_embedding: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorStoreError>;
    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError>;
    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError>;
    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError>;
    async fn count(&self) -> usize;
    async fn clear(&self) -> Result<(), VectorStoreError>;
}
```

## InMemory & File

```rust
use langchainrust::{InMemoryVectorStore, FileVectorStore, Document};

// In-memory (no persistence)
let store = InMemoryVectorStore::new();
let ids = store.add_documents(docs, embeddings).await?;
let results = store.similarity_search(&query_vec, 5).await?;

// File-backed (JSON persistence)
let store = FileVectorStore::new(PathBuf::from("./vectors.json"), 1536)?;
```

## ChromaDB & Qdrant

```rust
use langchainrust::{ChromaDBVectorStore, ChromaDBConfig};

// ChromaDB
let store = ChromaDBVectorStore::new(ChromaDBConfig::new(
    "http://localhost:8000", "my_collection", 1536,
))?;

// Qdrant (feature: qdrant-integration)
use langchainrust::{QdrantVectorStore, QdrantConfig, QdrantDistance};
let store = QdrantVectorStore::new(QdrantConfig::new("http://localhost:6334", "my_collection")
    .with_vector_size(1536)
    .with_distance(QdrantDistance::Cosine))?;
```

## Document & Builder Pattern

```rust
use langchainrust::{Document, VectorStoreBuilder, VectorStoreType};

let doc = Document::new("Hello, world!")
    .with_metadata("source", "test")
    .with_id("doc-1");

// Builder pattern
let store = VectorStoreBuilder::in_memory().build().await?;
let store = VectorStoreBuilder::file_backed("./vectors.json", 1536).build().await?;
```
