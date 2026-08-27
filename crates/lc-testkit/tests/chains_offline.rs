//! A6: chains-layer offline record/replay — the six online cases f02-f07 run over zero network
//! using `ReplayProvider`.
//!
//! Covers lc-chains' skeleton shapes: streaming LLMChain / ConversationChain memory /
//! SequentialChain serial / RetrievalQA (RAG) / StuffDocumentsChain / callbacks threading.
//! Each case maps one-to-one to a same-named case in `crates/lc/tests/chains.rs`, only replacing
//! the real LLM with replay: the chain just consumes responses popped by `ReplayProvider` in FIFO
//! order, with no message matching.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures_util::StreamExt;
use lc_callbacks::{CallbackHandler, CallbackManager, RunTree};
use lc_chains::{
    BaseChain, ConversationChain, LLMChain, RetrievalQA, SequentialChain, StuffDocumentsChain,
};
use lc_core::language_models::LLMResult;
use lc_core::runnables::RunnableConfig;
use lc_embeddings::{Embeddings, MockEmbeddings};
use lc_memory::ConversationBufferMemory;
use lc_rag::SimilarityRetriever;
use lc_testkit::{RecordedExchange, ReplayProvider};
use lc_vector_stores::{Document, InMemoryVectorStore, VectorStore};

/// Hand-writes one deterministic recording (equivalent to a fixture line).
fn exchange(content: &str) -> RecordedExchange {
    RecordedExchange {
        messages: Vec::new(),
        response: LLMResult {
            content: content.to_string(),
            model: "replay".to_string(),
            ..Default::default()
        },
        tools: None,
    }
}

/// Convenience helper to assemble a single-key input.
fn input(key: &str, value: &str) -> HashMap<String, serde_json::Value> {
    HashMap::from([(
        key.to_string(),
        serde_json::Value::String(value.to_string()),
    )])
}

/// Takes the first string value from a chain result (same extraction as the online cases).
fn first_answer(result: &HashMap<String, serde_json::Value>) -> String {
    result
        .values()
        .next()
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

// ============================================================================
// F02 LLMChain streaming (→ single-chunk replay)
// ============================================================================

#[tokio::test]
async fn f02_llm_chain_stream_offline() {
    let replay = ReplayProvider::from_exchanges(vec![exchange("一句话介绍量子计算。")]);

    // 1. Build the chain: the input key is topic
    let chain = LLMChain::new(replay, "用一句话介绍:{topic}").with_input_key("topic");

    // 2. Start streaming, collect tokens chunk by chunk (replay streaming = one whole chunk)
    let mut stream = chain
        .stream(input("topic", "量子计算"))
        .await
        .expect("启动流式失败");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        let token = item.expect("流式块不应出错");
        chunks.push(token.token);
    }

    // 3. Concatenate all tokens, assert non-empty
    let full = chunks.concat();
    assert!(!full.trim().is_empty(), "流式拼接结果不能为空");
    assert_eq!(full, "一句话介绍量子计算。", "单块回放应原样落地");
}

// ============================================================================
// F03 ConversationChain multi-turn memory (→ two-line FIFO)
// ============================================================================

#[tokio::test]
async fn f03_conversation_chain_memory_offline() {
    // Two-line FIFO: first round self-introduces, second round "remembers Xiaoming"
    let replay =
        ReplayProvider::from_exchanges(vec![exchange("我叫小明,住在北京。"), exchange("小明")]);

    // 1. Build a conversation chain with buffer memory
    let chain = ConversationChain::new(replay, ConversationBufferMemory::new());

    // 2. First round: self-introduction
    chain
        .invoke(input("input", "我叫小明,住在北京。"))
        .await
        .expect("第一轮失败");

    // 3. Second round: memory should be injected into the prompt (the replay side does not verify messages, only pops in order)
    let result = chain
        .invoke(input("input", "我叫什么名字?只回答名字。"))
        .await
        .expect("第二轮失败");
    let answer = first_answer(&result);

    // 4. Assert the second round remembers "Xiaoming"
    assert!(
        answer.contains("小明"),
        "第二轮应记得第一轮的\"小明\",回答: {answer}"
    );
}

// ============================================================================
// F04 SequentialChain serial (→ shared queue, two-line FIFO)
// ============================================================================

#[tokio::test]
async fn f04_sequential_chain_offline() {
    // Both chains share the same ReplayProvider (Clone shares the Arc queue), popping in order
    let replay = ReplayProvider::from_exchanges(vec![
        exchange("扩写:人工智能是研究如何让机器表现出智能的学科。"),
        exchange("总结:AI"),
    ]);

    // 1. Chain A: input topic, output mapped to the global key text; chain B: input text, output to output
    let step1 =
        LLMChain::new(replay.clone(), "把\"{topic}\"扩写成一句话描述。").with_input_key("topic");
    let step2 = LLMChain::new(replay, "用最多10个字总结:{text}").with_input_key("text");

    // 2. Chain into two steps
    let chain = SequentialChain::new()
        .with_name("two_step")
        .add_chain_with_mapping(
            Arc::new(step1) as Arc<dyn BaseChain>,
            HashMap::from([("topic".to_string(), "topic".to_string())]),
            HashMap::from([("text".to_string(), "text".to_string())]),
        )
        .add_chain_with_mapping(
            Arc::new(step2) as Arc<dyn BaseChain>,
            HashMap::from([("text".to_string(), "text".to_string())]),
            HashMap::from([("text".to_string(), "output".to_string())]),
        );

    // 3. Invoke the serial chain, get the final output
    let result = chain
        .invoke(input("topic", "人工智能"))
        .await
        .expect("SequentialChain 失败");
    let answer = result
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    // 4. The final output comes from the second FIFO line
    assert_eq!(answer, "总结:AI", "第二步输出应为第二行录播");
}

// ============================================================================
// F05 RetrievalQA (RAG) (→ offline retrieval + single-line replay LLM)
// ============================================================================

#[tokio::test]
async fn f05_retrieval_qa_offline() {
    // 1. Build an in-memory vector store: 2 documents vectorized with MockEmbeddings (zero network throughout)
    let store: InMemoryVectorStore = InMemoryVectorStore::new();
    let embeddings: Arc<dyn Embeddings> = Arc::new(MockEmbeddings::new(8));
    let docs = vec![
        Document::new("langchainrust 是一个 Rust 的 LLM 框架。"),
        Document::new("langchainrust 支持 RAG、Agent、LangGraph。"),
    ];
    let texts: Vec<&str> = docs.iter().map(|d| d.page_content()).collect();
    let vectors = embeddings
        .embed_documents(&texts)
        .await
        .expect("向量化失败");
    VectorStore::add_documents(&store, docs, vectors)
        .await
        .expect("存文档失败");

    // 2. Wrap SimilarityRetriever + RetrievalQA (LLM uses replay)
    let retriever = SimilarityRetriever::new(Arc::new(store) as Arc<dyn VectorStore>, embeddings);
    let replay = ReplayProvider::from_exchanges(vec![exchange(
        "langchainrust 是一个 Rust 的 LLM 框架,支持 RAG。",
    )]);
    let chain = RetrievalQA::new(replay, Arc::new(retriever));

    // 3. Ask → retrieve + answer
    let result = chain
        .invoke(input("query", "langchainrust 是什么?"))
        .await
        .expect("RetrievalQA 失败");
    let answer = first_answer(&result);

    // 4. Assert the answer is non-empty
    assert!(!answer.trim().is_empty(), "RAG 回答不能为空");
}

// ============================================================================
// F06 StuffDocumentsChain (→ single-line replay + doc input)
// ============================================================================

#[tokio::test]
async fn f06_stuff_documents_offline() {
    let replay = ReplayProvider::from_exchanges(vec![exchange(
        "Rust 注重安全与性能,所有权系统避免了内存安全问题。",
    )]);
    let chain = StuffDocumentsChain::new(replay);

    // 1. Assemble the input: question + documents (2 Rust docs)
    let docs = vec![
        Document::new("Rust 是一门系统编程语言,注重安全和性能。"),
        Document::new("Rust 的所有权系统避免了内存安全问题。"),
    ];
    let mut inputs = input("input", "Rust 有什么特点?");
    inputs.insert("documents".to_string(), serde_json::to_value(docs).unwrap());

    // 2. Invoke the documents chain, get the answer
    let result = chain
        .invoke(inputs)
        .await
        .expect("StuffDocumentsChain 失败");
    let answer = first_answer(&result);

    // 3. Assert the answer is non-empty
    assert!(!answer.trim().is_empty(), "文档链回答不能为空");
}

// ============================================================================
// F07 callbacks threading (→ replay LLM + real callbacks)
// ============================================================================

#[tokio::test]
async fn f07_callbacks_propagate_offline() {
    // 1. Define a counting callback: increments on on_run_start
    struct CountHandler {
        count: Arc<AtomicUsize>,
    }
    #[async_trait::async_trait]
    impl CallbackHandler for CountHandler {
        async fn on_run_start(&self, _run: &RunTree) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
        async fn on_run_end(&self, _run: &RunTree) {}
        async fn on_run_error(&self, _run: &RunTree, _error: &str) {}
    }

    // 2. Build a callback manager: register the counting callback
    let count = Arc::new(AtomicUsize::new(0));
    let handler = CountHandler {
        count: count.clone(),
    };
    let manager = CallbackManager::new().add_handler(Arc::new(handler));

    // 3. Build the chain + input, config carries the callbacks
    let replay = ReplayProvider::from_exchanges(vec![exchange("你好!")]);
    let chain = LLMChain::new(replay, "用一句话回答:{question}");

    // 4. Run the chain with a config (containing the callbacks)
    let config = RunnableConfig::new().with_callbacks(Arc::new(manager));
    let _ = chain
        .invoke_with_config(input("question", "你好"), Some(config))
        .await
        .expect("带回调执行失败");

    // 5. Assert the callbacks actually fired (config threaded through)
    let events = count.load(Ordering::SeqCst);
    assert!(
        events > 0,
        "回调应收到 on_chain_start 事件,实际 {events} 次——config 没贯穿?"
    );
}
