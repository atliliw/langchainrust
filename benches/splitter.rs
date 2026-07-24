// benches/splitter.rs
//! Benchmarks for document splitting operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use langchainrust::{Document, RecursiveCharacterSplitter, TextSplitter};

// ---------------------------------------------------------------------------
// Helpers: generate text of various sizes
// ---------------------------------------------------------------------------

fn generate_paragraph_text(paragraph_count: usize) -> String {
    let paragraphs = [
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, \
         and guarantees thread safety. It achieves these goals without needing a garbage collector, \
         making it a practical choice for systems where deterministic memory management is required. \
         The language enforces memory safety at compile time through its ownership system, which \
         tracks the lifetime and borrowing of values throughout the program.",
        "The ownership model in Rust is one of its most distinctive features. Every value in Rust \
         has a single owner, and when the owner goes out of scope, the value is dropped. This \
         eliminates the need for manual memory management while also preventing common bugs like \
         dangling pointers, double frees, and use-after-free errors. The borrow checker ensures \
         that references to data follow strict rules about mutability and lifetime.",
        "Rust's type system is both expressive and strict. It supports generics, traits, and \
         associated types, enabling developers to write abstract and reusable code without \
         sacrificing performance. The trait system provides a powerful mechanism for ad-hoc \
         polymorphism, while the type inference engine reduces the need for explicit type \
         annotations in many common situations.",
        "Error handling in Rust is done through the Result and Option types rather than exceptions. \
         The Result type represents either a successful value or an error, while Option represents \
         either a value or nothing. This approach forces developers to handle errors explicitly, \
         reducing the likelihood of unhandled exceptions at runtime. The question mark operator \
         provides a convenient shorthand for propagating errors up the call stack.",
        "Concurrency in Rust is designed to be safe by default. The type system enforces that \
         data races cannot occur at compile time. The Send and Sync traits mark types that can \
         be safely transferred between threads or shared between threads respectively. This \
         means that many concurrency bugs are caught during compilation rather than at runtime, \
         making it easier to write correct concurrent programs.",
        "The Rust ecosystem includes Cargo, a build system and package manager that simplifies \
         project management, dependency resolution, and testing. Cargo provides a standard \
         project structure, automatic downloading and compilation of dependencies, and built-in \
         support for running tests and generating documentation. It has become an essential tool \
         for Rust developers worldwide.",
        "Pattern matching is a powerful feature in Rust that allows developers to destructure \
         complex data types and write concise conditional logic. The match expression provides \
         exhaustive checking, ensuring that all possible cases are handled. Combined with enums \
         and the Option type, pattern matching enables safe and expressive control flow that \
         eliminates many common sources of bugs found in other languages.",
        "Rust's macro system provides metaprogramming capabilities that go beyond what is possible \
         with regular functions or traits. Declarative macros allow pattern-based code generation, \
         while procedural macros enable custom derive implementations, attribute macros, and \
         function-like macros. These tools allow library authors to create ergonomic APIs that \
         reduce boilerplate and improve developer productivity.",
    ];

    (0..paragraph_count)
        .map(|i| paragraphs[i % paragraphs.len()])
        .collect::<Vec<&str>>()
        .join("\n\n")
}

fn generate_chinese_text(paragraph_count: usize) -> String {
    let paragraphs = [
        "Rust 是一门系统编程语言，运行速度极快，能够防止段错误并保证线程安全。\
         它无需垃圾回收器即可实现这些目标，使其成为需要确定性内存管理的系统的实用选择。\
         该语言通过其所有权系统在编译时强制执行内存安全，跟踪程序中值的生命周期和借用。",
        "Rust 中的所有权模型是其最显著的特征之一。Rust 中的每个值都有一个单一的所有者，\
         当所有者超出作用域时，该值将被丢弃。这消除了手动内存管理的需要，同时防止了悬空指针、\
         双重释放和释放后使用等常见错误。借用检查器确保对数据的引用遵循关于可变性和生命周期的严格规则。",
        "Rust 的类型系统既富有表现力又严格。它支持泛型、特征和关联类型，\
         使开发者能够编写抽象且可重用的代码而不牺牲性能。特征系统提供了一种强大的临时多态机制，\
         而类型推断引擎减少了许多常见情况下对显式类型注释的需求。",
        "Rust 中的错误处理通过 Result 和 Option 类型而非异常来完成。\
         Result 类型表示成功值或错误，而 Option 表示值或无。\
         这种方法强制开发者显式处理错误，降低了运行时未处理异常的可能性。\
         问号运算符为沿调用栈传播错误提供了便捷的简写。",
    ];

    (0..paragraph_count)
        .map(|i| paragraphs[i % paragraphs.len()])
        .collect::<Vec<&str>>()
        .join("\n\n")
}

// ---------------------------------------------------------------------------
// RecursiveCharacterSplitter benchmarks
// ---------------------------------------------------------------------------

fn bench_recursive_splitter_split_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("recursive_splitter_split_text");

    let splitter = RecursiveCharacterSplitter::new(500, 50);

    for para_count in [5, 20, 100] {
        let text = generate_paragraph_text(para_count);
        group.bench_with_input(BenchmarkId::new("english", para_count), &text, |b, text| {
            b.iter(|| {
                black_box(splitter.split_text(black_box(text)));
            });
        });
    }

    group.finish();
}

fn bench_recursive_splitter_chinese(c: &mut Criterion) {
    let mut group = c.benchmark_group("recursive_splitter_chinese");

    let splitter = RecursiveCharacterSplitter::new(500, 50);

    for para_count in [5, 20, 50] {
        let text = generate_chinese_text(para_count);
        group.bench_with_input(BenchmarkId::new("chinese", para_count), &text, |b, text| {
            b.iter(|| {
                black_box(splitter.split_text(black_box(text)));
            });
        });
    }

    group.finish();
}

fn bench_recursive_splitter_split_document(c: &mut Criterion) {
    let mut group = c.benchmark_group("recursive_splitter_split_document");

    let splitter = RecursiveCharacterSplitter::new(500, 50);

    for para_count in [5, 20, 100] {
        let text = generate_paragraph_text(para_count);
        let doc = Document::new(&text).with_metadata("source", "benchmark");

        group.bench_with_input(BenchmarkId::new("document", para_count), &doc, |b, doc| {
            b.iter(|| {
                black_box(splitter.split_document(black_box(doc)));
            });
        });
    }

    group.finish();
}

fn bench_recursive_splitter_chunk_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("recursive_splitter_chunk_sizes");

    let text = generate_paragraph_text(50);

    for chunk_size in [200, 500, 1000] {
        let splitter = RecursiveCharacterSplitter::new(chunk_size, chunk_size / 10);
        group.bench_with_input(
            BenchmarkId::new("chunk_size", chunk_size),
            &text,
            |b, text| {
                b.iter(|| {
                    black_box(splitter.split_text(black_box(text)));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_recursive_splitter_split_text,
    bench_recursive_splitter_chinese,
    bench_recursive_splitter_split_document,
    bench_recursive_splitter_chunk_sizes,
);
criterion_main!(benches);
