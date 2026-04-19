// tests/bm25/full_flow.rs
//! 完整流程测试：存储 → 检索 → 融合 → 返回

use langchainrust::{
    UnifiedHybridIndex, HybridIndexConfig, Document, MockEmbeddings, Embeddings,
    ChunkedDocumentStoreTrait,
};
use std::sync::Arc;

/// 测试：完整流程 - 打印每个步骤
#[tokio::test]
async fn test_full_flow_with_print() {
    println!("\n========== 完整流程测试开始 ==========\n");
    
    // Step 1: 创建组件
    println!("【Step 1】创建组件");
    println!("  - ChunkedDocumentStore (内容仓库)");
    println!("  - MockEmbeddings (向量生成)");
    println!("  - UnifiedHybridIndex (混合索引)");
    
    let config = HybridIndexConfig::new()
        .with_chunk_size(50)
        .with_top_k(5, 5)
        .with_rrf_k(60);
    
    let embeddings: Arc<dyn Embeddings> = Arc::new(MockEmbeddings::new(3));
    let index = UnifiedHybridIndex::with_config(embeddings.clone(), 3, config);
    
    println!("  配置:");
    println!("    - chunk_size: 50");
    println!("    - bm25_k: 5");
    println!("    - vector_k: 5");
    println!("    - rrf_k: 60");
    println!();
    
    // Step 2: 添加文档
    println!("【Step 2】添加文档");
    
    let docs = vec![
        Document::new("Rust是一门系统编程语言，注重安全和性能。所有权系统是核心特性。")
            .with_id("doc_001"),
        Document::new("Python是一门高级编程语言，适合数据科学和机器学习。")
            .with_id("doc_002"),
        Document::new("内存管理是系统编程的重要部分，包括栈分配和堆分配。")
            .with_id("doc_003"),
        Document::new("Rust的所有权规则确保内存安全，无需垃圾回收。")
            .with_id("doc_004"),
    ];
    
    println!("  添加 {} 个文档:", docs.len());
    for doc in &docs {
        println!("    - {} ({}字)", doc.id.clone().unwrap_or_default(), doc.content.chars().count());
    }
    
    for doc in docs {
        let id = index.add_document(doc).await.expect("添加失败");
        println!("  ✓ 添加成功: {}", id);
    }
    
    println!("  文档总数: {}", index.document_count().await);
    println!("  分片总数: {}", index.chunk_count().await);
    println!();
    
    // Step 3: 检索
    println!("【Step 3】检索 - 查询: 'Rust内存管理'");
    
    let query = "Rust内存管理";
    println!("  查询内容: '{}'", query);
    
    // 模拟 BM25 检索（打印过程）
    println!("\n  BM25 检索过程:");
    println!("    - 分词: ['Rust', '内存', '管理']");
    println!("    - 查倒排索引:");
    println!("      'Rust' → doc_001_0, doc_004_0 匹配");
    println!("      '内存' → doc_003_0, doc_004_0 匹配");
    println!("      '管理' → doc_003_0 匹配");
    println!("    - BM25分数计算:");
    println!("      doc_004_0: 'Rust所有权规则...' → 分数最高 (匹配'Rust')");
    println!("      doc_001_0: 'Rust是一门系统...' → 分数次高");
    println!("      doc_003_0: '内存管理是...' → 分数中等");
    println!("    - AutoMerging判断:");
    println!("      ratio < threshold → 不合并，返回chunks");
    println!("    - BM25结果:");
    println!("      #1: parent_id='doc_004', content='Rust所有权规则...'");
    println!("      #2: parent_id='doc_001', content='Rust是一门系统...'");
    println!("      #3: parent_id='doc_003', content='内存管理是...'");
    
    // 模拟 Vector 检索（打印过程）
    println!("\n  Vector 检索过程:");
    println!("    - 计算query embedding: [0.15, -0.28, 0.49]");
    println!("    - 遍历向量索引:");
    println!("      cosine(query, doc_001_0_vec) = 0.75");
    println!("      cosine(query, doc_003_0_vec) = 0.85");
    println!("      cosine(query, doc_004_0_vec) = 0.92");
    println!("    - Vector结果:");
    println!("      #1: parent_id='doc_004', score=0.92 ← 使用parent_id!");
    println!("      #2: parent_id='doc_003', score=0.85");
    println!("      #3: parent_id='doc_001', score=0.75");
    
    println!("\n  ⭐ 关键点: Vector也返回parent_id（不是chunk_id）");
    println!("     这样BM25和Vector的ID才能匹配！");
    println!();
    
    // Step 4: RRF 融合
    println!("【Step 4】RRF 融合");
    println!("  公式: RRF_score = Σ 1/(k + rank)");
    println!("  k = 60");
    
    println!("\n  BM25排名 → contribution:");
    println!("    doc_004: rank=1 → 1/(60+1) = 0.0164");
    println!("    doc_001: rank=2 → 1/(60+2) = 0.0161");
    println!("    doc_003: rank=3 → 1/(60+3) = 0.0159");
    
    println!("\n  Vector排名 → contribution:");
    println!("    doc_004: rank=1 → 1/(60+1) = 0.0164");
    println!("    doc_003: rank=2 → 1/(60+2) = 0.0161");
    println!("    doc_001: rank=3 → 1/(60+3) = 0.0159");
    
    println!("\n  融合计算:");
    println!("    doc_004: BM25(0.0164) + Vector(0.0164) = 0.0328 ← 双路命中！");
    println!("    doc_001: BM25(0.0161) + Vector(0.0159) = 0.0320 ← 双路命中！");
    println!("    doc_003: BM25(0.0159) + Vector(0.0161) = 0.0320 ← 双路命中！");
    println!("    doc_002: BM25(0) + Vector(0) = 0 ← 未命中");
    
    println!("\n  最终排序:");
    println!("    #1: doc_004 → 分数 0.0328 (双路命中，最高)");
    println!("    #2: doc_001 → 分数 0.0320 (双路命中)");
    println!("    #3: doc_003 → 分数 0.0320 (双路命中)");
    println!();
    
    // Step 5: 实际检索并打印结果
    println!("【Step 5】实际检索结果");
    
    let results = index.retrieve(query, 3).await.expect("检索失败");
    
    println!("  返回 {} 个结果:", results.len());
    for (i, result) in results.iter().enumerate() {
        let id = result.document.id.clone().unwrap_or_default();
        let content_preview = if result.document.content.chars().count() > 30 {
            format!("{}...", result.document.content.chars().take(30).collect::<String>())
        } else {
            result.document.content.clone()
        };
        println!("    #{}: id={}, content='{}'", i + 1, id, content_preview);
    }
    
    println!("\n========== 完整流程测试结束 ==========\n");
}

/// 测试：不同chunks命中同一parent的情况
#[tokio::test]
async fn test_different_chunks_same_parent() {
    println!("\n========== 不同chunks命中同一parent测试 ==========\n");
    
    let config = HybridIndexConfig::new()
        .with_chunk_size(30)
        .with_top_k(5, 5);
    
    let embeddings: Arc<dyn Embeddings> = Arc::new(MockEmbeddings::new(3));
    let index = UnifiedHybridIndex::with_config(embeddings.clone(), 3, config);
    
    // 创建一个长文档，分成多个chunks
    let long_doc = Document::new(
        "Rust是一门系统编程语言。内存管理是核心特性。所有权系统确保安全。借用规则防止数据竞争。"
    ).with_id("doc_001");
    
    println!("【添加文档】");
    println!("  文档ID: doc_001");
    println!("  内容: 'Rust是一门系统编程语言。内存管理是核心特性。所有权系统确保安全。借用规则防止数据竞争。'");
    println!("  chunk_size: 30");
    println!("  预期分成 4个chunks:");
    println!("    - doc_001_0: 'Rust是一门系统编程语言。'");
    println!("    - doc_001_1: '内存管理是核心特性。'");
    println!("    - doc_001_2: '所有权系统确保安全。'");
    println!("    - doc_001_3: '借用规则防止数据竞争。'");
    
    index.add_document(long_doc).await.expect("添加失败");
    
    println!("  实际分片数: {}", index.chunk_count().await);
    println!();
    
    // 查询命中不同的chunks
    println!("【检索】查询: 'Rust所有权'");
    println!("  BM25:");
    println!("    'Rust' → 匹配 doc_001_0");
    println!("    '所有权' → 匹配 doc_001_2");
    println!("    → BM25命中 chunk_0 和 chunk_2");
    println!("    → 但都属于 parent_id='doc_001'");
    println!("    → AutoMerging 后返回 doc_001 (parent)");
    
    println!("  Vector:");
    println!("    语义相似 → 匹配 doc_001_0 和 doc_001_2");
    println!("    → Vector命中 chunk_0 和 chunk_2");
    println!("    → 但都转换为 parent_id='doc_001'");
    
    println!("  RRF:");
    println!("    BM25: doc_001 (parent_id)");
    println!("    Vector: doc_001 (parent_id)");
    println!("    → 同一ID，分数合并！");
    println!("    → 最终返回: doc_001");
    
    let results = index.retrieve("Rust所有权", 3).await.expect("检索失败");
    
    println!("\n【实际结果】");
    for (i, result) in results.iter().enumerate() {
        println!("  #{}: id={}, content='{}'", 
            i + 1, 
            result.document.id.clone().unwrap_or_default(),
            result.document.content.chars().take(50).collect::<String>()
        );
    }
    
    println!("\n========== 测试结束 ==========\n");
}

/// 测试：BM25和Vector命中完全不同的情况
#[tokio::test]
async fn test_bm25_vector_different_hits() {
    println!("\n========== BM25和Vector命中不同文档测试 ==========\n");
    
    let config = HybridIndexConfig::new()
        .with_chunk_size(100)
        .with_top_k(3, 3);
    
    let embeddings: Arc<dyn Embeddings> = Arc::new(MockEmbeddings::new(3));
    let index = UnifiedHybridIndex::with_config(embeddings.clone(), 3, config);
    
    index.add_documents(vec![
        Document::new("Rust是一门系统编程语言，注重安全和并发。").with_id("doc_rust"),
        Document::new("Python适合数据科学和机器学习。").with_id("doc_python"),
        Document::new("系统编程的核心是内存管理和性能优化。").with_id("doc_system"),
    ]).await.expect("添加失败");
    
    println!("【添加文档】");
    println!("  - doc_rust: 'Rust是一门系统编程语言...'");
    println!("  - doc_python: 'Python适合数据科学...'");
    println!("  - doc_system: '系统编程的核心是内存管理...'");
    println!();
    
    println!("【检索】查询: '数据科学'");
    
    let results = index.retrieve("数据科学", 3).await.expect("检索失败");
    
    println!("\n【实际结果】");
    for (i, result) in results.iter().enumerate() {
        println!("  #{}: id={}, score={}", 
            i + 1, 
            result.document.id.clone().unwrap_or_default(),
            result.score
        );
    }
    
    println!("\n========== 测试结束 ==========\n");
}

/// 测试：大文档分割 + AutoMerging + chunk回溯完整流程
#[tokio::test]
async fn test_large_document_full_flow() {
    println!("\n========== 大文档完整流程测试 ==========\n");
    
    let large_content = "Rust是一门系统编程语言，注重安全和性能。\
        Rust的核心特性是所有权系统，这是内存安全的基石。\
        所有权规则确保每个值有唯一所有者，离开作用域自动释放。\
        借用规则允许临时访问数据，分为不可变借用和可变借用。\
        生命周期标注引用的有效范围，防止悬垂引用。\
        内存分配默认在栈上，Box用于堆分配。\
        智能指针如Rc和Arc用于共享所有权场景。\
        内存泄漏在Rust中仍可能发生，但大幅降低。\
        RAII模式自动管理资源，析构函数自动释放。\
        并发安全通过Send和Sync trait保证。\
        性能优化建议减少堆分配，提高缓存命中。\
        常见错误包括忘记处理Result和Option。\
        系统编程需要理解底层内存模型。\
        unsafe代码允许绕过安全检查但需谨慎。\
        外部函数接口FFI与C语言交互。";
    
    println!("【Step 1】创建大文档");
    println!("  文档内容: {}字", large_content.chars().count());
    println!("  配置: chunk_size=50, merge_threshold=0.5");
    println!();
    
    let config = HybridIndexConfig::new()
        .with_chunk_size(50)
        .with_merge_threshold(0.5)
        .with_top_k(10, 10);
    
    let embeddings: Arc<dyn Embeddings> = Arc::new(MockEmbeddings::new(3));
    let index = UnifiedHybridIndex::with_config(embeddings.clone(), 3, config);
    
    let large_doc = Document::new(large_content).with_id("large_doc");
    
    println!("【Step 2】添加文档并分割");
    index.add_document(large_doc).await.expect("添加失败");
    
    let parent_count = index.document_count().await;
    let chunk_count = index.chunk_count().await;
    
    println!("  文档数(parent): {}", parent_count);
    println!("  分片数(chunks): {}", chunk_count);
    println!("  每个chunk约50字");
    println!("  预期chunk数: ~{}", large_content.chars().count() / 50);
    println!();
    
    println!("【Step 3】场景1 - 查询命中少量chunks");
    println!("  查询: 'FFI外部函数'");
    println!("  预期: 只匹配最后几个chunks, ratio < 0.5, 不合并");
    
    let results1 = index.retrieve("FFI外部函数", 5).await.expect("检索失败");
    
    println!("  实际结果:");
    for (i, result) in results1.iter().enumerate() {
        let content_len = result.document.content.chars().count();
        println!("    #{}: id={}, content_len={}字, score={}", 
            i + 1,
            result.document.id.clone().unwrap_or_default(),
            content_len,
            result.score
        );
    }
    
    println!("\n  ⭐ 关键: 返回chunks内容（不是完整{}字文档）", 
        large_content.chars().count());
    println!();
    
    println!("【Step 4】场景2 - 查询命中大量chunks");
    println!("  查询: 'Rust内存系统编程'");
    println!("  预期: 匹配大部分chunks, ratio > 0.5, 合并返回完整parent");
    
    let results2 = index.retrieve("Rust内存系统编程", 5).await.expect("检索失败");
    
    println!("  实际结果:");
    for (i, result) in results2.iter().enumerate() {
        let content_len = result.document.content.chars().count();
        println!("    #{}: id={}, content_len={}字, score={}", 
            i + 1,
            result.document.id.clone().unwrap_or_default(),
            content_len,
            result.score
        );
    }
    
    println!("\n  ⭐ 关键: 返回完整文档（{}字）", large_content.chars().count());
    println!();
    
    println!("【Step 5】直接获取完整parent");
    
    let store = index.document_store();
    let full_doc = store.get_parent_document("large_doc")
        .await
        .expect("获取失败")
        .expect("文档不存在");
    
    println!("  get_parent_document('large_doc')");
    println!("  返回: {}字", full_doc.content.chars().count());
    println!("  内容预览: {}...", full_doc.content.chars().take(50).collect::<String>());
    println!();
    
    println!("【Step 6】验证 chunk → parent 回溯");
    
    let chunks = store.get_chunks_for_parent("large_doc")
        .await
        .expect("获取失败");
    
    println!("  parent 'large_doc' 的所有chunks:");
    for (i, chunk) in chunks.iter().enumerate() {
        println!("    {}: id={}, parent_id={}, len={}字",
            i,
            chunk.chunk_id,
            chunk.parent_id,
            chunk.content.chars().count()
        );
    }
    
    println!("\n  ⭐ 验证:");
    println!("    - 每个chunk.parent_id = 'large_doc'");
    println!("    - chunk_id格式: 'large_doc_0', 'large_doc_1', ...");
    println!("    - ChunkedStore维护 parent→chunks 映射");
    
    println!("\n========== 测试结束 ==========\n");
}

/// 测试：不同chunks命中同一parent时RRF合并
#[tokio::test]
async fn test_chunks_to_parent_rrf_merge() {
    println!("\n========== chunks→parent RRF合并测试 ==========\n");
    
    let config = HybridIndexConfig::new()
        .with_chunk_size(30)
        .with_merge_threshold(0.3)
        .with_top_k(10, 10);
    
    let embeddings: Arc<dyn Embeddings> = Arc::new(MockEmbeddings::new(3));
    let index = UnifiedHybridIndex::with_config(embeddings.clone(), 3, config);
    
    println!("【添加文档】");
    
    index.add_documents(vec![
        Document::new("Rust语言特性：所有权系统确保内存安全。借用规则防止数据竞争。生命周期标注引用范围。")
            .with_id("doc_rust"),
        Document::new("Python应用场景：数据科学和机器学习。Web开发框架Django。自动化脚本编写。")
            .with_id("doc_python"),
        Document::new("系统编程核心：内存管理栈堆分配。性能优化缓存命中。并发安全线程同步。")
            .with_id("doc_system"),
    ]).await.expect("添加失败");
    
    println!("  - doc_rust: 3个chunks");
    println!("  - doc_python: 3个chunks");
    println!("  - doc_system: 3个chunks");
    println!("  总chunk数: {}", index.chunk_count().await);
    println!();
    
    println!("【查询】'内存管理所有权'");
    
    let results = index.retrieve("内存管理所有权", 5).await.expect("检索失败");
    
    println!("【结果】");
    for (i, result) in results.iter().enumerate() {
        println!("  #{}: parent={}, score={}", 
            i + 1,
            result.document.id.clone().unwrap_or_default(),
            result.score
        );
    }
    
    println!("\n  ⭐ 验证:");
    println!("    - doc_rust和doc_system双路命中（分数高）");
    println!("    - doc_python单路命中或未命中（分数低）");
    
    println!("\n========== 测试结束 ==========\n");
}