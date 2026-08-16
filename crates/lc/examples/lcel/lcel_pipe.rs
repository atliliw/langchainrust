// examples/lcel/lcel_pipe.rs
//! LCEL Pipeline Example
//!
//! Demonstrates the LangChain Expression Language (LCEL) pipe composition
//! using `RunnableLambda`, `RunnablePassthrough`, `RunnableParallel`,
//! `RunnableBranch`, and `RunnableSequence`.
//!
//! Run with: cargo run --example lcel_pipe

use langchainrust::{
    Runnable, RunnableBranch, RunnableConfig, RunnableExt, RunnableLambda, RunnableParallel,
    RunnablePassthrough,
};

#[tokio::main]
async fn main() {
    println!("=== LCEL Pipeline Example ===\n");

    // 1. Basic pipe: Double → AddOne
    let doubler = RunnableLambda::new_sync(|x: i32| x * 2);
    let add_one = RunnableLambda::new_sync(|x: i32| x + 1);

    let pipeline = doubler.pipe(add_one);
    let result = pipeline.invoke(5, None).await.unwrap();
    println!("5 → double → add_one = {} (expected 11)", result);

    // 2. Three-step pipeline: Double → AddOne → ToString
    let to_string = RunnableLambda::new_sync(|x: i32| format!("result: {}", x));
    let pipeline = RunnableLambda::new_sync(|x: i32| x * 2)
        .pipe(RunnableLambda::new_sync(|x: i32| x + 1))
        .pipe(to_string);

    let result = pipeline.invoke(3, None).await.unwrap();
    println!(
        "3 → double → add_one → to_string = {} (expected \"result: 7\")",
        result
    );

    // 3. Passthrough: input passes through unchanged
    let passthrough = RunnablePassthrough::<i32>::new();
    let result = passthrough.invoke(42, None).await.unwrap();
    println!("passthrough(42) = {} (expected 42)", result);

    // 4. Async lambda
    let async_lambda = RunnableLambda::new_async(|x: i32| async move {
        tokio::task::spawn_blocking(move || x * 10)
            .await
            .map_err(|e| langchainrust::LcelError::Other(e.to_string()))
    });
    let result = async_lambda.invoke(7, None).await.unwrap();
    println!("async_lambda(7) = {} (expected 70)", result);

    // 5. Branch: route based on input
    let branch = RunnableBranch::new(RunnableLambda::new_sync(|x: i32| format!("default: {}", x)))
        .when_fn(
            |x: &i32| *x > 10,
            RunnableLambda::new_sync(|x: i32| format!("big: {}", x)),
        )
        .when_fn(
            |x: &i32| *x < 0,
            RunnableLambda::new_sync(|x: i32| format!("negative: {}", x)),
        );

    let r1 = branch.invoke(5, None).await.unwrap();
    let r2 = branch.invoke(20, None).await.unwrap();
    let r3 = branch.invoke(-3, None).await.unwrap();
    println!("branch(5) = {} (expected \"default: 5\")", r1);
    println!("branch(20) = {} (expected \"big: 20\")", r2);
    println!("branch(-3) = {} (expected \"negative: -3\")", r3);

    // 6. Parallel: run multiple steps concurrently
    let parallel = RunnableParallel::<String>::new()
        .with("len", RunnableLambda::new_sync(|s: String| s.len() as i64))
        .with(
            "upper",
            RunnableLambda::new_sync(|s: String| s.to_uppercase()),
        );

    let result = parallel.invoke("hello".to_string(), None).await.unwrap();
    println!(
        "parallel(\"hello\") = {:?} (expected len=5, upper=HELLO)",
        result
    );

    // 7. Batch processing
    let doubler = RunnableLambda::new_sync(|x: i32| x * 2);
    let results = doubler.batch(vec![1, 2, 3, 4, 5], None).await.unwrap();
    println!(
        "batch([1,2,3,4,5]) → double = {:?} (expected [2,4,6,8,10])",
        results
    );

    // 8. Pipeline with config
    let pipeline = RunnableLambda::new_sync(|x: i32| x + 100);
    let config = RunnableConfig::default()
        .with_tag("production")
        .with_run_name("add_100");

    let result = pipeline.invoke(5, Some(config)).await.unwrap();
    println!("pipeline with config(5) = {} (expected 105)", result);

    println!("\n=== All LCEL examples completed! ===");
}
