// examples/advanced/full_pipeline.rs
//! 高级示例 3: 完整 AI 应用流程
//!
//! 运行: cargo run --example full_pipeline
//!
//! 功能: 演示一个完整的 AI 应用，结合 Agent、Memory、RAG 和工具

use langchainrust::{
    OpenAIChat, OpenAIConfig, BaseChatModel,
    ReActAgent, AgentExecutor, BaseAgent, BaseTool,
    ChatMessageHistory,
    Calculator, DateTimeTool, SimpleMathTool,
    Document, InMemoryVectorStore, VectorStore,
    MockEmbeddings,
    SimilarityRetriever, RetrieverTrait, TextSplitter, RecursiveCharacterSplitter,
    Message,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 高级示例 3: 完整 AI 应用流程 ===\n");
    
    // ========== 1. 初始化所有组件 ==========
    println!("--- 1. 初始化组件 ---\n");
    
    // LLM 配置
    let config = OpenAIConfig {
        api_key: "sk-l0YYMX65mCYRlTJYH0ptf4BFpqJwm8Xo9Z5IMqSZD0yOafl6".parse()?,
        base_url: "https://api.openai-proxy.org/v1".parse()?,
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        temperature: Some(0.5),
        max_tokens: Some(1000),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        organization: None,
    };
    
    let llm_for_agent = OpenAIChat::new(config.clone());
    let llm_for_chat = OpenAIChat::new(config);
    
    // Memory - 使用 ChatMessageHistory 代替 ConversationBufferMemory
    let mut chat_history = ChatMessageHistory::new();
    
    // Tools
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
        Arc::new(DateTimeTool::new()),
        Arc::new(SimpleMathTool::new()),
    ];
    
    // RAG 组件
    let embeddings = Arc::new(MockEmbeddings::new(128));
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());
    
    // Agent
    let agent: Arc<dyn BaseAgent> = Arc::new(ReActAgent::new(llm_for_agent, tools.clone(), None));
    let executor = AgentExecutor::new(agent, tools)
        .with_verbose(false)
        .with_max_iterations(5);
    
    println!("组件初始化完成\n");
    
    // ========== 2. 构建知识库 ==========
    println!("--- 2. 构建知识库 ---\n");
    
    let knowledge_docs = vec![
        Document::new("公司的退货政策: 商品购买后 30 天内可无条件退货。需要保持商品完好，附带购买凭证。"),
        Document::new("公司的配送服务: 订单确认后 1-3 个工作日发货。标准配送 3-5 天到达，加急配送 1-2 天到达。"),
        Document::new("公司的客服工作时间: 周一至周五 9:00-18:00。客服电话: 400-123-4567。邮箱: support@example.com。"),
    ];
    
    let splitter = RecursiveCharacterSplitter::new(200, 50);
    let mut chunks = Vec::new();
    for doc in &knowledge_docs {
        chunks.extend(splitter.split_document(doc));
    }
    
    retriever.add_documents(chunks).await?;
    
    println!("知识库构建完成，已添加 {} 个文档块\n", store.count().await);
    
    // ========== 3. 交互式问答系统 ==========
    println!("--- 3. 智能问答演示 ---\n");
    
    let questions = vec![
        "你们的退货政策是什么？",
        "配送需要多长时间？",
        "如果我今天下单，最快什么时候能收到？（请计算具体日期）",
        "客服电话是多少？",
        "这个电话的工作时间是什么时候？",
    ];
    
    for question in questions.iter() {
        println!("{}\n用户: {}\n", "-".repeat(50), question);
        
        // 1. 先从知识库检索
        let relevant_docs = retriever.retrieve(question, 2).await?;
        
        // 2. 构建上下文
        let context = if !relevant_docs.is_empty() {
            Some(relevant_docs.iter()
                .map(|d| d.content.as_str())
                .collect::<Vec<_>>()
                .join("\n"))
        } else {
            None
        };
        
        // 3. 判断问题类型并选择处理方式
        let needs_tools = question.contains("计算") || 
                         question.contains("什么时候") ||
                         question.contains("日期");
        
        let response = if needs_tools {
            println!("(使用 Agent + 工具处理)\n");
            
            let full_question = if let Some(ctx) = &context {
                format!("背景信息: {}\n\n问题: {}", ctx, question)
            } else {
                question.to_string()
            };
            
            executor.invoke(full_question).await?
        } else {
            println!("(直接回答)\n");
            
            let mut messages = vec![
                Message::system("你是一个友好的客服助手。请根据提供的信息回答问题。"),
            ];
            
            // 添加历史
            messages.extend(chat_history.messages().iter().cloned());
            
            if let Some(ctx) = &context {
                messages.push(Message::human(format!("参考资料:\n{}\n\n问题: {}", ctx, question)));
            } else {
                messages.push(Message::human(question.to_string()));
            }
            
            llm_for_chat.chat(messages, None).await?.content
        };
        
        println!("助手: {}\n", response);
        
        // 保存对话
        chat_history.add_message(Message::human(*question));
        chat_history.add_message(Message::ai(&response));
    }
    
    // ========== 4. 显示完整对话历史 ==========
    println!("--- 4. 完整对话历史 ---\n");
    
    for (i, msg) in chat_history.messages().iter().enumerate() {
        let role = match msg.message_type {
            langchainrust::schema::MessageType::Human => "用户",
            langchainrust::schema::MessageType::AI => "助手",
            langchainrust::schema::MessageType::System => "系统",
            langchainrust::schema::MessageType::Tool { .. } => "工具",
        };
        println!("[{}] {}: {}", i + 1, role, 
                msg.content.chars().take(80).collect::<String>());
        if msg.content.len() > 80 {
            println!("     ...");
        }
    }
    
    println!("\n=== 示例完成 ===");
    Ok(())
}