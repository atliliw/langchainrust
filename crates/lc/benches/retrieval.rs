// benches/retrieval.rs
//! Benchmarks for BM25 retrieval and vector search operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use langchainrust::{
    BM25Retriever, Document, Embeddings, InMemoryVectorStore, MockEmbeddings, Tokenizer,
    VectorStore,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helper: generate a set of realistic English documents
// ---------------------------------------------------------------------------

fn generate_english_documents(count: usize) -> Vec<Document> {
    let topics = [
        "Rust is a systems programming language focused on safety and performance",
        "Python is a high-level general-purpose programming language widely used in data science",
        "JavaScript is a dynamic language primarily used for web development and server-side applications",
        "Go is a statically typed compiled language designed at Google for simplicity and efficiency",
        "C++ is a powerful general-purpose language with object-oriented and generic programming features",
        "Java is a class-based object-oriented language designed for minimal implementation dependencies",
        "TypeScript is a strongly typed programming language that builds on JavaScript",
        "Kotlin is a cross-platform statically typed language with type inference and interoperability with Java",
        "Swift is a general-purpose programming language built using a modern approach to safety and performance",
        "Ruby is an interpreted high-level language supporting multiple programming paradigms",
        "Haskell is a purely functional programming language known for its strong static type system",
        "Scala combines object-oriented and functional programming in one concise high-level language",
        "R is a programming language for statistical computing and graphics supported by the R Core Team",
        "Perl is a family of high-level general-purpose interpreted dynamic programming languages",
        "Lua is a lightweight high-level multi-paradigm programming language designed for embedded use",
        "Dart is a client-optimized language for fast apps on any platform with a mature ecosystem",
        "Elixir is a functional concurrent programming language running on the Erlang virtual machine",
        "Clojure is a dynamic dialect of Lisp that runs on the Java virtual machine",
        "Erlang is a general-purpose concurrent garbage-collected programming language and runtime system",
        "F Sharp is a functional-first programming language that also supports object-oriented and imperative programming",
    ];

    (0..count)
        .map(|i| {
            let base = topics[i % topics.len()];
            Document::new(format!("{} [doc {}]", base, i))
        })
        .collect()
}

fn generate_chinese_documents(count: usize) -> Vec<Document> {
    let topics = [
        "Rust 是一门系统编程语言，注重安全性和性能表现",
        "Python 是一种高级通用编程语言，广泛应用于数据科学和人工智能",
        "JavaScript 是一种动态语言，主要用于网页开发和服务器端应用",
        "Go 是一种静态类型编译语言，由谷歌设计，追求简洁和高效",
        "C++ 是一种强大的通用语言，支持面向对象和泛型编程特性",
        "Java 是一种基于类的面向对象语言，设计目标是减少实现依赖",
        "TypeScript 是一种强类型编程语言，构建在 JavaScript 之上",
        "Kotlin 是一种跨平台静态类型语言，与 Java 完全互操作",
        "Swift 是一种通用编程语言，采用现代方法确保安全和性能",
        "Ruby 是一种解释型高级语言，支持多种编程范式",
    ];

    (0..count)
        .map(|i| {
            let base = topics[i % topics.len()];
            Document::new(format!("{} [文档 {}]", base, i))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// BM25 benchmarks
// ---------------------------------------------------------------------------

fn bench_bm25_index_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_index_building");

    for size in [50, 200, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let retriever = BM25Retriever::new();
                let docs = generate_english_documents(size);
                retriever.add_documents_sync(black_box(docs));
            });
        });
    }

    group.finish();
}

fn bench_bm25_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_search");

    for size in [50, 200, 1000] {
        let retriever = BM25Retriever::new();
        retriever.add_documents_sync(generate_english_documents(size));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| {
                let _results = retriever.search(black_box("programming language"), 10);
            });
        });
    }

    group.finish();
}

fn bench_bm25_chinese_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_chinese_search");

    for size in [50, 200, 500] {
        let retriever = BM25Retriever::new();
        retriever.add_documents_sync(generate_chinese_documents(size));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| {
                let _results = retriever.search(black_box("编程语言"), 10);
            });
        });
    }

    group.finish();
}

fn bench_tokenizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer");

    let tokenizer = Tokenizer::new();

    let english_text = "Rust is a systems programming language focused on safety, \
        concurrency and speed. It provides memory safety without a garbage collector \
        and supports zero-cost abstractions, move semantics, and guaranteed memory safety.";

    let chinese_text = "Rust 是一门系统编程语言，注重安全性、并发性和速度。\
        它在没有垃圾回收器的情况下提供内存安全，并支持零成本抽象、移动语义和保证的内存安全。";

    group.bench_function("english_100_words", |b| {
        b.iter(|| {
            black_box(tokenizer.tokenize(black_box(english_text)));
        });
    });

    group.bench_function("chinese_mixed", |b| {
        b.iter(|| {
            black_box(tokenizer.tokenize(black_box(chinese_text)));
        });
    });

    // Benchmark tokenizing a large document
    let large_text: String = (0..100)
        .map(|i| {
            format!(
                "Paragraph {} about Rust programming and systems design. ",
                i
            )
        })
        .collect();

    group.bench_function("large_100_paragraphs", |b| {
        b.iter(|| {
            black_box(tokenizer.tokenize(black_box(&large_text)));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Vector search benchmarks
// ---------------------------------------------------------------------------

fn bench_vector_store_add_documents(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("vector_store_add_documents");

    for size in [50, 200, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                rt.block_on(async {
                    let store = InMemoryVectorStore::new();
                    let embedding_model: Arc<dyn Embeddings> = Arc::new(MockEmbeddings::new(128));

                    let docs: Vec<Document> = generate_english_documents(size);
                    let mut embeddings = Vec::with_capacity(size);
                    for doc in &docs {
                        let emb = embedding_model.embed_query(&doc.content).await.unwrap();
                        embeddings.push(emb);
                    }

                    store.add_documents(docs, embeddings).await.unwrap();
                });
            });
        });
    }

    group.finish();
}

fn bench_vector_store_similarity_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("vector_store_similarity_search");

    for size in [50, 200, 500] {
        // Pre-populate the store outside the benchmark loop
        let store = rt.block_on(async {
            let store = InMemoryVectorStore::new();
            let embedding_model: Arc<dyn Embeddings> = Arc::new(MockEmbeddings::new(128));

            let docs = generate_english_documents(size);
            let mut embeddings = Vec::with_capacity(size);
            for doc in &docs {
                let emb = embedding_model.embed_query(&doc.content).await.unwrap();
                embeddings.push(emb);
            }

            store.add_documents(docs, embeddings).await.unwrap();
            store
        });

        let query_embedding = rt.block_on(async {
            let embedding_model: Arc<dyn Embeddings> = Arc::new(MockEmbeddings::new(128));
            embedding_model
                .embed_query("programming language")
                .await
                .unwrap()
        });

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| {
                rt.block_on(async {
                    let _results = store
                        .similarity_search(black_box(&query_embedding), 10)
                        .await
                        .unwrap();
                });
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bm25_index_building,
    bench_bm25_search,
    bench_bm25_chinese_search,
    bench_tokenizer,
    bench_vector_store_add_documents,
    bench_vector_store_similarity_search,
);
criterion_main!(benches);
