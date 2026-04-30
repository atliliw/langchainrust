// src/chains/conversation_retrieval.rs
//! ConversationRetrieval Chain
//!
//! 带记忆的检索增强生成 Chain，将对话历史与文档检索相结合。
//! 适用于需要同时利用对话上下文和外部知识的问答场景。

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use serde_json::Value;

use super::base::{BaseChain, ChainResult, ChainError};
use crate::language_models::OpenAIChat;
use crate::memory::{ConversationBufferMemory, BaseMemory};
use crate::retrieval::{RetrieverTrait, Document};
use crate::schema::Message;
use crate::Runnable;
use tokio::sync::Mutex;

/// 默认的检索增强对话提示词模板
const DEFAULT_QA_PROMPT: &str = "你是一个人工智能助手，请根据对话历史和参考信息回答用户的问题。

对话历史：
{history}

参考信息：
{context}

问题：{question}

回答：";

/// ConversationRetrievalChain
///
/// 带记忆的检索增强对话 Chain，自动完成：
/// 1. 加载对话历史
/// 2. 检索相关文档
/// 3. 组合历史 + 上下文 + 问题
/// 4. LLM 生成答案
/// 5. 保存到对话记忆
///
/// # 示例
/// ```ignore
/// use langchainrust::{ConversationRetrievalChain, OpenAIChat, SimilarityRetriever, ConversationBufferMemory};
///
/// let chain = ConversationRetrievalChain::new(llm, retriever, memory);
/// let answer = chain.query("什么是 Rust？").await?;
/// ```
pub struct ConversationRetrievalChain {
    llm: OpenAIChat,
    retriever: Arc<dyn RetrieverTrait>,
    memory: Arc<Mutex<ConversationBufferMemory>>,

    system_prompt: Option<String>,
    qa_prompt_template: String,
    input_key: String,
    output_key: String,
    memory_key: String,
    name: String,

    k: usize,
    verbose: bool,
    return_source_documents: bool,
    source_document_key: String,
}

impl ConversationRetrievalChain {
    pub fn new(
        llm: OpenAIChat,
        retriever: Arc<dyn RetrieverTrait>,
        memory: ConversationBufferMemory,
    ) -> Self {
        Self {
            llm,
            retriever,
            memory: Arc::new(Mutex::new(memory.with_return_messages(true))),
            system_prompt: None,
            qa_prompt_template: DEFAULT_QA_PROMPT.to_string(),
            input_key: "query".to_string(),
            output_key: "result".to_string(),
            memory_key: "history".to_string(),
            name: "conversation_retrieval".to_string(),
            k: 4,
            verbose: false,
            return_source_documents: false,
            source_document_key: "source_documents".to_string(),
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn with_qa_prompt(mut self, template: impl Into<String>) -> Self {
        self.qa_prompt_template = template.into();
        self
    }

    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_return_source_documents(mut self, return_source: bool) -> Self {
        self.return_source_documents = return_source;
        self
    }

    pub fn memory(&self) -> &Arc<Mutex<ConversationBufferMemory>> {
        &self.memory
    }

    pub async fn clear_memory(&self) -> Result<(), ChainError> {
        let mut memory = self.memory.lock().await;
        memory.clear().await.map_err(|e|
            ChainError::ExecutionError(format!("清空记忆失败: {}", e))
        )?;
        Ok(())
    }

    /// 简化的查询接口
    pub async fn query(&self, question: impl Into<String>) -> Result<String, ChainError> {
        let inputs = HashMap::from([
            (self.input_key.clone(), Value::String(question.into()))
        ]);
        let result = self.invoke(inputs).await?;
        result.get(&self.output_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ChainError::OutputError("缺少输出结果".to_string()))
    }

    fn format_context(&self, documents: &[Document]) -> String {
        documents.iter()
            .map(|doc| doc.content.clone())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n")
    }

    fn format_history(&self, messages: &[Message]) -> String {
        messages.iter()
            .map(|msg| {
                let role = match msg.message_type {
                    crate::schema::MessageType::Human => "用户",
                    crate::schema::MessageType::AI => "助手",
                    _ => "系统",
                };
                format!("{}: {}", role, msg.content)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn build_prompt(&self, history: &str, context: &str, question: &str) -> String {
        let mut prompt = String::new();

        if let Some(system) = &self.system_prompt {
            prompt.push_str(&format!("{}\n\n", system));
        }

        let template = self.qa_prompt_template
            .replace("{history}", history)
            .replace("{context}", context)
            .replace("{question}", question);

        prompt.push_str(&template);
        prompt
    }

    async fn load_history(&self) -> Result<Vec<Message>, ChainError> {
        let memory = self.memory.lock().await;
        Ok(memory.chat_memory().messages().to_vec())
    }

    async fn save_context(&self, input: &str, output: &str) -> Result<(), ChainError> {
        let mut memory = self.memory.lock().await;
        let inputs = HashMap::from([(self.input_key.clone(), input.to_string())]);
        let outputs = HashMap::from([(self.output_key.clone(), output.to_string())]);
        memory.save_context(&inputs, &outputs).await
            .map_err(|e| ChainError::ExecutionError(format!("保存上下文失败: {}", e)))?;
        Ok(())
    }
}

#[async_trait]
impl BaseChain for ConversationRetrievalChain {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key]
    }

    fn output_keys(&self) -> Vec<&str> {
        if self.return_source_documents {
            vec![&self.output_key, &self.source_document_key]
        } else {
            vec![&self.output_key]
        }
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.validate_inputs(&inputs)?;

        let question = inputs.get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        if self.verbose {
            println!("\n=== ConversationRetrievalChain 执行 ===");
            println!("问题: {}", question);
        }

        // 步骤 1: 加载对话历史
        let history_messages = self.load_history().await?;
        let history = self.format_history(&history_messages);

        if self.verbose {
            println!("历史消息: {} 条", history_messages.len());
        }

        // 步骤 2: 检索相关文档
        if self.verbose {
            println!("\n--- 步骤 2: 检索相关文档 ---");
        }

        let documents = self.retriever.retrieve(question, self.k).await
            .map_err(|e| ChainError::ExecutionError(format!("检索失败: {}", e)))?;

        if self.verbose {
            println!("检索到 {} 个文档", documents.len());
            for (i, doc) in documents.iter().enumerate() {
                let preview = if doc.content.len() > 100 {
                    &doc.content[..100]
                } else {
                    &doc.content
                };
                println!("文档 {}: {}", i + 1, preview);
            }
        }

        // 步骤 3: 组装 Prompt
        if self.verbose {
            println!("\n--- 步骤 3: 组装 Prompt ---");
        }

        let context = self.format_context(&documents);
        let prompt = self.build_prompt(&history, &context, question);

        if self.verbose {
            println!("历史长度: {} 字符", history.len());
            println!("上下文长度: {} 字符", context.len());
        }

        // 步骤 4: LLM 生成答案
        if self.verbose {
            println!("\n--- 步骤 4: LLM 生成答案 ---");
        }

        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("LLM 调用失败: {}", e)))?;

        let answer = response.content;

        if self.verbose {
            println!("答案: {}", answer);
        }

        // 步骤 5: 保存到记忆
        self.save_context(question, &answer).await?;

        if self.verbose {
            println!("=== ConversationRetrievalChain 完成 ===\n");
        }

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(answer));

        if self.return_source_documents {
            let sources: Vec<Value> = documents.iter()
                .map(|doc| serde_json::to_value(doc).unwrap_or(Value::Null))
                .collect();
            result.insert(self.source_document_key.clone(), Value::Array(sources));
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }
}
