//! A6:chains 层离线录播——f02-f07 六条在线用例改用 `ReplayProvider` 零网络跑通。
//!
//! 覆盖 lc-chains 的骨架形态:LLMChain 流式 / ConversationChain 记忆 /
//! SequentialChain 串行 / RetrievalQA(RAG)/ StuffDocumentsChain / 回调贯穿。
//! 每条用例与 `crates/lc/tests/chains.rs` 中同名单条一一对应,只把真实 LLM
//! 换成回放:链只消费 `ReplayProvider` 按 FIFO 弹出的响应,不做消息匹配。

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

/// 手写一条确定性录播(等价 fixture 行)。
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

/// 组装单键输入的快捷方式。
fn input(key: &str, value: &str) -> HashMap<String, serde_json::Value> {
    HashMap::from([(
        key.to_string(),
        serde_json::Value::String(value.to_string()),
    )])
}

/// 从链结果里取第一个字符串值(在线用例同款取法)。
fn first_answer(result: &HashMap<String, serde_json::Value>) -> String {
    result
        .values()
        .next()
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

// ============================================================================
// F02 LLMChain 流式(→ 单块回放)
// ============================================================================

#[tokio::test]
async fn f02_llm_chain_stream_offline() {
    let replay = ReplayProvider::from_exchanges(vec![exchange("一句话介绍量子计算。")]);

    // 1. 建链:输入键指定为 topic
    let chain = LLMChain::new(replay, "用一句话介绍:{topic}").with_input_key("topic");

    // 2. 启动流式,逐块收集 token(回放流式 = 单个整块)
    let mut stream = chain
        .stream(input("topic", "量子计算"))
        .await
        .expect("启动流式失败");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        let token = item.expect("流式块不应出错");
        chunks.push(token.token);
    }

    // 3. 拼接全部 token,断言非空
    let full = chunks.concat();
    assert!(!full.trim().is_empty(), "流式拼接结果不能为空");
    assert_eq!(full, "一句话介绍量子计算。", "单块回放应原样落地");
}

// ============================================================================
// F03 ConversationChain 多轮记忆(→ 两行 FIFO)
// ============================================================================

#[tokio::test]
async fn f03_conversation_chain_memory_offline() {
    // 两行 FIFO:第一轮自我介绍,第二轮"记得小明"
    let replay =
        ReplayProvider::from_exchanges(vec![exchange("我叫小明,住在北京。"), exchange("小明")]);

    // 1. 建带缓冲记忆的对话链
    let chain = ConversationChain::new(replay, ConversationBufferMemory::new());

    // 2. 第一轮:自我介绍
    chain
        .invoke(input("input", "我叫小明,住在北京。"))
        .await
        .expect("第一轮失败");

    // 3. 第二轮:记忆应注入 prompt(回放侧不校验消息,只按序弹出)
    let result = chain
        .invoke(input("input", "我叫什么名字?只回答名字。"))
        .await
        .expect("第二轮失败");
    let answer = first_answer(&result);

    // 4. 断言第二轮记得"小明"
    assert!(
        answer.contains("小明"),
        "第二轮应记得第一轮的\"小明\",回答: {answer}"
    );
}

// ============================================================================
// F04 SequentialChain 串行(→ 共享队列两行 FIFO)
// ============================================================================

#[tokio::test]
async fn f04_sequential_chain_offline() {
    // 两条链共享同一 ReplayProvider(Clone 共享 Arc 队列),按序弹出
    let replay = ReplayProvider::from_exchanges(vec![
        exchange("扩写:人工智能是研究如何让机器表现出智能的学科。"),
        exchange("总结:AI"),
    ]);

    // 1. 链 A:输入 topic,输出映射成全局键 text;链 B:输入 text,输出到 output
    let step1 =
        LLMChain::new(replay.clone(), "把\"{topic}\"扩写成一句话描述。").with_input_key("topic");
    let step2 = LLMChain::new(replay, "用最多10个字总结:{text}").with_input_key("text");

    // 2. 串成两步链
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

    // 3. invoke 串行链,取最终 output
    let result = chain
        .invoke(input("topic", "人工智能"))
        .await
        .expect("SequentialChain 失败");
    let answer = result
        .get("output")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    // 4. 最终输出来自第二行 FIFO
    assert_eq!(answer, "总结:AI", "第二步输出应为第二行录播");
}

// ============================================================================
// F05 RetrievalQA(RAG)(→ 离线检索 + 单行回放 LLM)
// ============================================================================

#[tokio::test]
async fn f05_retrieval_qa_offline() {
    // 1. 搭内存向量库:2 篇文档用 MockEmbeddings 向量化入库(全程零网络)
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

    // 2. 包 SimilarityRetriever + RetrievalQA(LLM 用回放)
    let retriever = SimilarityRetriever::new(Arc::new(store) as Arc<dyn VectorStore>, embeddings);
    let replay = ReplayProvider::from_exchanges(vec![exchange(
        "langchainrust 是一个 Rust 的 LLM 框架,支持 RAG。",
    )]);
    let chain = RetrievalQA::new(replay, Arc::new(retriever));

    // 3. 提问 → 检索 + 回答
    let result = chain
        .invoke(input("query", "langchainrust 是什么?"))
        .await
        .expect("RetrievalQA 失败");
    let answer = first_answer(&result);

    // 4. 断言回答非空
    assert!(!answer.trim().is_empty(), "RAG 回答不能为空");
}

// ============================================================================
// F06 StuffDocumentsChain(→ 单行回放 + doc 输入)
// ============================================================================

#[tokio::test]
async fn f06_stuff_documents_offline() {
    let replay = ReplayProvider::from_exchanges(vec![exchange(
        "Rust 注重安全与性能,所有权系统避免了内存安全问题。",
    )]);
    let chain = StuffDocumentsChain::new(replay);

    // 1. 组装输入:question + documents(2 篇 Rust 文档)
    let docs = vec![
        Document::new("Rust 是一门系统编程语言,注重安全和性能。"),
        Document::new("Rust 的所有权系统避免了内存安全问题。"),
    ];
    let mut inputs = input("input", "Rust 有什么特点?");
    inputs.insert("documents".to_string(), serde_json::to_value(docs).unwrap());

    // 2. invoke 文档链,取回答
    let result = chain
        .invoke(inputs)
        .await
        .expect("StuffDocumentsChain 失败");
    let answer = first_answer(&result);

    // 3. 断言回答非空
    assert!(!answer.trim().is_empty(), "文档链回答不能为空");
}

// ============================================================================
// F07 回调贯穿(→ 回放 LLM + 真实回调)
// ============================================================================

#[tokio::test]
async fn f07_callbacks_propagate_offline() {
    // 1. 定义计数回调:on_run_start 触发时计数 +1
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

    // 2. 建回调管理器:注册计数回调
    let count = Arc::new(AtomicUsize::new(0));
    let handler = CountHandler {
        count: count.clone(),
    };
    let manager = CallbackManager::new().add_handler(Arc::new(handler));

    // 3. 建链 + 输入,config 带上回调
    let replay = ReplayProvider::from_exchanges(vec![exchange("你好!")]);
    let chain = LLMChain::new(replay, "用一句话回答:{question}");

    // 4. 带 config(内含回调)执行链
    let config = RunnableConfig::new().with_callbacks(Arc::new(manager));
    let _ = chain
        .invoke_with_config(input("question", "你好"), Some(config))
        .await
        .expect("带回调执行失败");

    // 5. 断言回调确实触发(config 贯穿)
    let events = count.load(Ordering::SeqCst);
    assert!(
        events > 0,
        "回调应收到 on_chain_start 事件,实际 {events} 次——config 没贯穿?"
    );
}
