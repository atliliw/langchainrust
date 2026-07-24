//! ConversationSummaryMemory 测试
//!
//! 测试智能摘要记忆的核心功能：
//! 1. 单轮对话摘要生成 - save_context 自动调用 LLM 生成摘要
//! 2. 多轮对话摘要累积 - 每轮更新摘要，包含所有关键信息
//! 3. 摘要清空功能 - clear() 清空 buffer 和 chat_memory
//! 4. Chain 集成 - load_memory_variables 返回摘要供 Chain 使用

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::{BaseMemory, ConversationSummaryMemory, MessageType};
use std::collections::HashMap;

/// 测试单轮对话摘要生成
///
/// 场景：技术咨询 - 用户询问 Rust 项目架构问题
#[tokio::test]
#[ignore = "需要 LLM API Key 且账户余额充足"]
async fn test_summary_memory_single_turn() {
    let llm = TestConfig::get().openai_chat();
    let mut memory = ConversationSummaryMemory::new(llm);

    println!("\n========================================");
    println!("测试：单轮对话摘要生成");
    println!("========================================");

    // 真实场景：用户咨询技术问题
    let inputs = HashMap::from([(
        "input".to_string(),
        "我们正在开发一个高并发的交易系统，用 Rust 实现。目前遇到的问题是：当订单量突然激增时，系统的消息队列会堵塞，导致订单处理延迟从 50ms 变成 500ms。我们的架构是：使用 tokio 异步运行时，消息通过 crossbeam channel 传递，worker 线程池有 8 个 worker。你有什么建议？".to_string()
    )]);
    let outputs = HashMap::from([(
        "output".to_string(),
        "这个问题很典型，高并发场景下 channel 容量不足会导致背压。几个建议：1) 增加 channel 容量或使用 unbounded channel（但要注意内存控制）；2) 实现动态 worker 扩缩容，用 ThreadPoolBuilder；3) 考虑用 mpsc channel 配合 async-stream 做 batch 处理；4) 加监控，用 prometheus 暴露队列深度指标。你当前的 worker 数量偏少，8 个 worker 对交易系统来说可能不够，建议至少 2x CPU 核数。".to_string()
    )]);

    println!("\n用户问题：{}", inputs.get("input").unwrap());
    println!("\nAI 回复：{}", outputs.get("output").unwrap());

    memory.save_context(&inputs, &outputs).await.unwrap();

    println!("\n>>> 调用 LLM 生成摘要...");
    let buffer = memory.buffer().await;
    println!("\n生成的摘要:\n{}", buffer);

    assert!(!buffer.is_empty(), "摘要应不为空");
    // 摘要应该包含关键信息：交易系统、高并发、队列堵塞、建议方案
    let has_key_info = buffer.contains("交易")
        || buffer.contains("并发")
        || buffer.contains("队列")
        || buffer.contains("worker")
        || buffer.contains("channel")
        || buffer.contains("延迟");
    assert!(has_key_info, "摘要应包含关键技术信息");

    let vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
    let history = vars.get("history").unwrap().as_str().unwrap();
    println!("\nChain 加载的历史摘要:\n{}", history);

    assert!(!history.is_empty(), "Chain 加载的摘要应不为空");
    println!("\n✅ 测试通过！");
}

/// 测试多轮对话摘要累积更新
///
/// 场景：产品需求讨论 - 多轮深入讨论一个产品的功能设计
#[tokio::test]
#[ignore = "需要 LLM API Key 且账户余额充足"]
async fn test_summary_memory_multi_turn() {
    let llm = TestConfig::get().openai_chat();
    let mut memory = ConversationSummaryMemory::new(llm);

    println!("\n========================================");
    println!("测试：多轮对话摘要累积");
    println!("场景：产品需求讨论 - 设计一个智能客服系统");
    println!("========================================");

    // 第一轮：产品背景介绍
    println!("\n--- 第一轮 ---");
    let inputs1 = HashMap::from([(
        "input".to_string(),
        "我们公司叫云智科技，主要做企业 SaaS 产品。现在要开发一个智能客服系统，目标客户是电商和金融行业。用户痛点是：客服团队成本高，夜间没人值班，常见问题重复回答效率低。我们希望用 AI 降低 60% 的人工客服工作量。预算大概 200 万，6 个月内上线。你觉得这个项目的关键风险是什么？".to_string()
    )]);
    let outputs1 = HashMap::from([(
        "output".to_string(),
        "智能客服项目有几个关键风险：1) 行业适配风险 - 电商和金融的业务逻辑差异大，知识库要分开设计；2) AI 准确率风险 - 金融行业对错误容忍度低，需要严格的答案审核机制；3) 夜间值班问题没完全解决 - AI 可以 24h 在线，但复杂问题仍需人工兜底；4) 预算可能不够 - 200 万做完整的知识库+多轮对话+人工转接系统，偏紧张；5) 6 个月上线时间紧 - 建议先做 MVP，覆盖 20% 高频问题，验证效果后再扩展。你们目前有现成的客服数据吗？".to_string()
    )]);

    println!("用户：{}", inputs1.get("input").unwrap());
    println!("AI：{}", outputs1.get("output").unwrap());

    memory.save_context(&inputs1, &outputs1).await.unwrap();

    let buffer1 = memory.buffer().await;
    let history_len = memory.chat_memory().len();
    let history_total_chars: usize = memory
        .chat_memory()
        .messages()
        .iter()
        .map(|m| m.content.len())
        .sum();

    println!("\n第一轮后:");
    println!("摘要长度: {} 字符", buffer1.len());
    println!(
        "历史消息: {} 条, 总 {} 字符",
        history_len, history_total_chars
    );
    println!(
        "历史还在吗? {}",
        if history_len == 2 {
            "✅ 在"
        } else {
            "❌ 不在"
        }
    );

    println!("\n--- 第二轮 ---");
    let inputs2 = HashMap::from([(
        "input".to_string(),
        "我们有过去两年的客服聊天记录，大概 50 万条对话。技术栈倾向于用你们的 langchainrust 框架做对话管理，向量数据库用 Milvus，前端用 Vue。关于知识库，我们想用 RAG 方案，把 FAQ 文档和聊天记录都向量化。你说先做 MVP 只覆盖高频问题，具体怎么定义高频？我们怎么快速验证效果？".to_string()
    )]);
    let outputs2 = HashMap::from([(
        "output".to_string(),
        "50 万条历史数据是很好的资产，可以做训练和验证。高频问题定义建议：对历史数据做聚类分析，找出占比超过 1% 的问题类别（通常能覆盖 60-70% 的咨询量）。验证方案：A/B 对照测试，选 1000 个真实咨询case，让 AI 和人工分别回答，对比准确率、响应时间、用户满意度。RAG 方案可行，但要注意：1) 文档 chunk 大小建议 500-800 字，overlap 100；2) 历史聊天记录要先清洗，去除无效对话；3) 用 hybrid search（向量+关键词）效果更好。langchainrust + Milvus + Vue 这个技术栈没问题，但建议加一个答案审核后台，让客服可以校验 AI 答案。".to_string()
    )]);

    println!("用户：{}", inputs2.get("input").unwrap());
    println!("AI：{}", outputs2.get("output").unwrap());

    memory.save_context(&inputs2, &outputs2).await.unwrap();

    let buffer2 = memory.buffer().await;
    let history_len = memory.chat_memory().len();
    let history_total_chars: usize = memory
        .chat_memory()
        .messages()
        .iter()
        .map(|m| m.content.len())
        .sum();

    println!("\n第二轮后:");
    println!("摘要长度: {} 字符", buffer2.len());
    println!(
        "历史消息: {} 条, 总 {} 字符",
        history_len, history_total_chars
    );
    println!(
        "历史还在吗? {}",
        if history_len == 4 {
            "✅ 在"
        } else {
            "❌ 不在"
        }
    );

    println!("\n--- 第三轮 ---");
    let inputs3 = HashMap::from([(
        "input".to_string(),
        "最后一个问题：部署架构怎么设计？我们考虑用 Kubernetes 部署，区域是华东和华南两个机房。预计峰值 QPS 是 500，平时 100 左右。数据库用 PostgreSQL 存用户信息和对话记录，Redis 做缓存。你建议怎么做灰度发布？".to_string()
    )]);
    let outputs3 = HashMap::from([(
        "output".to_string(),
        "Kubernetes 双机房部署是个好方案。建议架构：1) 前端 Vue 部署为独立 service，CDN 加速；2) 后端 API service 用 Deployment，配置 HPA（CPU > 70% 自动扩容）；3) langchainrust 对话引擎部署为单独 service，支持流式响应；4) Milvus 集群建议至少 3 节点，跨机房要做数据同步；5) PostgreSQL 用 Patroni 做高可用，主备复制。灰度方案：先在华南机房上线，开 10% 流量给新系统，观察准确率和延迟指标一周后逐步放量。Redis 缓存建议存会话上下文和最近答案，TTL 设 30 分钟。峰值 500 QPS 用 2 个 API pod + 1 个对话引擎 pod 应该够，但建议预留 3x buffer。".to_string()
    )]);

    println!("用户：{}", inputs3.get("input").unwrap());
    println!("AI：{}", outputs3.get("output").unwrap());

    memory.save_context(&inputs3, &outputs3).await.unwrap();

    let buffer3 = memory.buffer().await;
    let history_len = memory.chat_memory().len();
    let history_total_chars: usize = memory
        .chat_memory()
        .messages()
        .iter()
        .map(|m| m.content.len())
        .sum();

    println!("\n第三轮后:");
    println!("摘要长度: {} 字符", buffer3.len());
    println!(
        "历史消息: {} 条, 总 {} 字符",
        history_len, history_total_chars
    );
    println!(
        "历史还在吗? {}",
        if history_len == 6 {
            "✅ 在"
        } else {
            "❌ 不在"
        }
    );

    println!("\n========================================");
    println!("历史消息完整内容:");
    println!("========================================");
    for (i, msg) in memory.chat_memory().messages().iter().enumerate() {
        let role = match msg.message_type {
            MessageType::Human => "用户",
            MessageType::AI => "AI",
            MessageType::System => "系统",
            MessageType::Tool { .. } => "工具",
        };
        println!("\n[{}] {}: {}", i + 1, role, msg.content);
    }

    println!("\n========================================");
    println!("结论:");
    println!("========================================");
    println!(
        "摘要 {} 字符 vs 历史总 {} 字符",
        buffer3.len(),
        history_total_chars
    );
    println!(
        "历史被压缩了吗? {}",
        if buffer3.len() < history_total_chars {
            "✅ 摘要压缩了"
        } else {
            "❌ 没压缩"
        }
    );
    println!(
        "历史被删除了吗? {}",
        if history_len == 6 {
            "❌ 没删除，还在"
        } else {
            "✅ 删除了"
        }
    );

    assert_eq!(memory.chat_memory().len(), 6);
    println!("\n✅ 测试通过！");
}

/// 测试摘要清空功能
///
/// 场景：用户会话结束后清空记忆
#[tokio::test]
#[ignore = "需要 LLM API Key 且账户余额充足"]
async fn test_summary_memory_clear() {
    let llm = TestConfig::get().openai_chat();
    let mut memory = ConversationSummaryMemory::new(llm);

    println!("\n========================================");
    println!("测试：摘要清空功能");
    println!("========================================");

    let inputs = HashMap::from([(
        "input".to_string(),
        "我是字节跳动的工程师，想咨询一下大规模分布式系统的 trace 方案。我们有 2000+ 微服务，调用链平均深度 15 层，现在用 Jaeger 但查询慢，想做优化。".to_string()
    )]);
    let outputs = HashMap::from([(
        "output".to_string(),
        "2000+ 微服务的 trace 优化是典型的大规模场景问题。Jaeger 在这个量级确实会遇到查询瓶颈。建议：1) 分层采样 - 采样率根据服务重要性动态调整，核心服务 100%，边缘服务 1%；2) 预聚合 - 用 ClickHouse 或 Elasticsearch 替代 Cassandra 存储，查询性能提升 10x；3) trace ID 关联业务 ID，方便按业务查；4) 考虑用 OpenTelemetry 协议，兼容性更好。你们现在的采样率是多少？".to_string()
    )]);

    println!("用户：{}", inputs.get("input").unwrap());
    println!("AI：{}", outputs.get("output").unwrap());

    memory.save_context(&inputs, &outputs).await.unwrap();

    let buffer_before = memory.buffer().await;
    println!("\n清空前摘要:\n{}", buffer_before);
    assert!(!buffer_before.is_empty(), "清空前摘要应不为空");
    assert_eq!(memory.chat_memory().len(), 2, "清空前应有 2 条消息");

    println!("\n>>> 执行 clear() 清空记忆...");
    memory.clear().await.unwrap();

    let buffer_after = memory.buffer().await;
    println!("\n清空后摘要: \"{}\"", buffer_after);
    assert!(buffer_after.is_empty(), "清空后摘要应为空字符串");
    assert_eq!(memory.chat_memory().len(), 0, "清空后 chat_memory 应为空");

    println!("\n✅ 测试通过！");
}

/// 测试 Chain 集成场景
///
/// 场景：用户偏好设置，后续 Chain 可以读取这些偏好
#[tokio::test]
#[ignore = "需要 LLM API Key 且账户余额充足"]
async fn test_summary_memory_load_for_chain() {
    let llm = TestConfig::get().openai_chat();
    let mut memory = ConversationSummaryMemory::new(llm);

    println!("\n========================================");
    println!("测试：Chain 集成场景");
    println!("场景：用户设置个人偏好，后续对话可读取这些偏好");
    println!("========================================");

    // 用户设置多个偏好
    let inputs1 = HashMap::from([(
        "input".to_string(),
        "我偏好简洁的技术回答，不要啰嗦。回答用中文，专业术语可以保留英文。如果涉及代码，用 Rust 优先，其次 Python。".to_string()
    )]);
    let outputs1 = HashMap::from([(
        "output".to_string(),
        "好的，已记录你的偏好：1) 简洁回答风格；2) 中文为主，术语保留英文；3) 代码示例优先 Rust，其次 Python。后续回答会遵循这些规则。".to_string()
    )]);

    println!("用户偏好设置：{}", inputs1.get("input").unwrap());
    println!("AI 确认：{}", outputs1.get("output").unwrap());

    memory.save_context(&inputs1, &outputs1).await.unwrap();

    let inputs2 = HashMap::from([(
        "input".to_string(),
        "补充一点：我不喜欢用 ChatGPT 风格的 markdown 格式，直接用纯文本就好。代码块用简单的缩进表示，不要用 ``` 语法。".to_string()
    )]);
    let outputs2 = HashMap::from([(
        "output".to_string(),
        "补充偏好已记录：markdown 禁用，代码用缩进表示而非 ```语法。完整偏好现在包括：简洁风格、中文为主、Rust/Python 优先、纯文本格式。".to_string()
    )]);

    println!("\n补充偏好：{}", inputs2.get("input").unwrap());
    println!("AI 确认：{}", outputs2.get("output").unwrap());

    memory.save_context(&inputs2, &outputs2).await.unwrap();

    let vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
    let history = vars.get("history").unwrap().as_str().unwrap();

    println!("\n>>> Chain 读取的偏好摘要:");
    println!("{}", history);

    // 摘要应该包含用户的关键偏好
    let has_preference = history.contains("简洁")
        || history.contains("中文")
        || history.contains("Rust")
        || history.contains("Python")
        || history.contains("纯文本")
        || history.contains("偏好");
    assert!(has_preference, "Chain 加载的摘要应包含用户偏好信息");

    println!("\n✅ 测试通过！");
}
