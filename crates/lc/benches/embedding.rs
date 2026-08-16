// benches/embedding.rs
//! Benchmarks for embedding operations using MockEmbeddings.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use langchainrust::{cosine_similarity, Embeddings, MockEmbeddings};

// ---------------------------------------------------------------------------
// Single embedding generation benchmarks
// ---------------------------------------------------------------------------

fn bench_embed_query_single(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("embed_query_single");

    let texts = [
        ("short", "Hello world"),
        (
            "medium",
            "Rust is a systems programming language focused on safety and performance. \
             It provides memory safety without a garbage collector.",
        ),
        (
            "long",
            "Rust is a systems programming language that runs blazingly fast, prevents segfaults, \
             and guarantees thread safety. It achieves these goals without needing a garbage collector, \
             making it a practical choice for systems where deterministic memory management is required. \
             The language enforces memory safety at compile time through its ownership system, which \
             tracks the lifetime and borrowing of values throughout the program. The borrow checker \
             ensures that references follow strict rules about mutability and lifetime.",
        ),
    ];

    for (label, text) in texts {
        group.bench_function(label, |b| {
            b.iter(|| {
                rt.block_on(async {
                    let embeddings = MockEmbeddings::new(1536);
                    let _result = embeddings.embed_query(black_box(text)).await.unwrap();
                });
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Embedding dimension benchmarks
// ---------------------------------------------------------------------------

fn bench_embed_query_dimensions(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("embed_query_dimensions");

    let text = "Rust is a systems programming language focused on safety and performance.";

    for dim in [128, 512, 1536] {
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            b.iter(|| {
                rt.block_on(async {
                    let embeddings = MockEmbeddings::new(dim);
                    let _result = embeddings.embed_query(black_box(text)).await.unwrap();
                });
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Batch embedding benchmarks
// ---------------------------------------------------------------------------

fn bench_embed_documents_batch(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("embed_documents_batch");

    let base_texts: Vec<&str> = vec![
        "Rust is a systems programming language",
        "Python is a high-level programming language",
        "JavaScript is used for web development",
        "Go is designed for simplicity and efficiency",
        "C++ supports object-oriented and generic programming",
        "Java is a class-based object-oriented language",
        "TypeScript builds on JavaScript with types",
        "Kotlin is cross-platform and interoperable with Java",
        "Swift focuses on safety and performance",
        "Ruby supports multiple programming paradigms",
    ];

    for batch_size in [5, 20, 50] {
        let texts: Vec<&str> = (0..batch_size)
            .flat_map(|_| base_texts.iter().copied())
            .take(batch_size)
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &texts,
            |b, texts| {
                b.iter(|| {
                    rt.block_on(async {
                        let embeddings = MockEmbeddings::new(128);
                        let _result = embeddings.embed_documents(black_box(texts)).await.unwrap();
                    });
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Cosine similarity benchmarks
// ---------------------------------------------------------------------------

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in [128, 512, 1536] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).cos()).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            &(&a, &b),
            |bencher, (va, vb)| {
                bencher.iter(|| {
                    black_box(cosine_similarity(black_box(va), black_box(vb)).unwrap());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_embed_query_single,
    bench_embed_query_dimensions,
    bench_embed_documents_batch,
    bench_cosine_similarity,
);
criterion_main!(benches);
