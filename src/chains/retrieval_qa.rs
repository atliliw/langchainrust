// src/chains/retrieval_qa.rs
//! RetrievalQA Chain
//!
//! 一站式检索问答 Chain，封装完整的 RAG 流程。

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use serde_json::Value;

use super::base::{BaseChain, ChainResult, ChainError};
use crate::language_models::OpenAIChat;
use crate::retrieval::{RetrieverTrait, Document};
use crate::schema::Message;
use crate::Runnable;

/// 默认的 QA 提示词模板
const DEFAULT_QA_PROMPT: &str = "根据以下上下文信息回答问题。如果上下文中没有相关信息，请说'我不知道'。

上下文：
{context}

问题：{question}

回答：";

/// RetrievalQA Chain
///
/// 一站式检索问答 Chain，自动完成：
/// 1. 检索相关文档
/// 2. 组装 prompt（上下文 + 问题）
/// 3. LLM 生成答案
///
/// # 示例
/// ```ignore
/// use langchainrust::{RetrievalQA, OpenAIChat, SimilarityRetriever};
///
/// let llm = OpenAIChat::new(config);
/// let retriever = SimilarityRetriever::new(store, embeddings);
///
/// let qa = RetrievalQA::new(llm, retriever);
///
/// // 一行代码完成文档问答
/// let answer = qa.invoke("什么是 Rust？").await?;
/// ```
pub struct RetrievalQA {
    llm: OpenAIChat,
    retriever: Arc<dyn RetrieverTrait>,
    
    prompt_template: String,
    input_key: String,
    output_key: String,
    name: String,
    
    k: usize,
    verbose: bool,
    
    return_source_documents: bool,
    source_document_key: String,
}

impl RetrievalQA {
    pub fn new(llm: OpenAIChat, retriever: Arc<dyn RetrieverTrait>) -> Self {
        Self {
            llm,
            retriever,
            prompt_template: DEFAULT_QA_PROMPT.to_string(),
            input_key: "query".to_string(),
            output_key: "result".to_string(),
            name: "retrieval_qa".to_string(),
            k: 4,
            verbose: false,
            return_source_documents: false,
            source_document_key: "source_documents".to_string(),
        }
    }
    
    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
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
    
    pub fn with_source_document_key(mut self, key: impl Into<String>) -> Self {
        self.source_document_key = key.into();
        self
    }
    
    pub fn retriever(&self) -> &Arc<dyn RetrieverTrait> {
        &self.retriever
    }
    
    pub fn k(&self) -> usize {
        self.k
    }
    
    fn format_context(&self, documents: &[Document]) -> String {
        documents
            .iter()
            .map(|doc| doc.content.clone())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
    
    fn build_prompt(&self, context: &str, question: &str) -> String {
        self.prompt_template
            .replace("{context}", context)
            .replace("{question}", question)
    }
    
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
    
    pub async fn query_with_sources(&self, question: impl Into<String>) -> Result<(String, Vec<Document>), ChainError> {
        let inputs = HashMap::from([
            (self.input_key.clone(), Value::String(question.into()))
        ]);
        
        let qa = RetrievalQA::new(
            OpenAIChat::new(crate::OpenAIConfig::default()),
            self.retriever.clone(),
        )
            .with_k(self.k)
            .with_prompt_template(self.prompt_template.clone())
            .with_verbose(self.verbose)
            .with_return_source_documents(true);
        
        let result = qa.invoke(inputs).await?;
        
        let answer = result.get(&qa.output_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ChainError::OutputError("缺少输出结果".to_string()))?;
        
        let sources: Vec<Document> = result.get(&qa.source_document_key)
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                .collect())
            .unwrap_or_default();
        
        Ok((answer, sources))
    }
}

#[async_trait]
impl BaseChain for RetrievalQA {
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
            println!("\n=== RetrievalQA 执行 ===");
            println!("问题: {}", question);
            println!("检索数量 (k): {}", self.k);
        }
        
        if self.verbose {
            println!("\n--- 步骤 1: 检索相关文档 ---");
        }
        
        let documents = self.retriever.retrieve(question, self.k).await
            .map_err(|e| ChainError::ExecutionError(format!("检索失败: {}", e)))?;
        
        if self.verbose {
            println!("检索到 {} 个文档", documents.len());
            for (i, doc) in documents.iter().enumerate() {
                println!("文档 {}: {}", i + 1, 
                    if doc.content.len() > 100 {
                        &doc.content[..100]
                    } else {
                        &doc.content
                    }
                );
            }
        }
        
        if documents.is_empty() {
            if self.verbose {
                println!("警告: 没有检索到相关文档");
            }
        }
        
        if self.verbose {
            println!("\n--- 步骤 2: 组装 Prompt ---");
        }
        
        let context = self.format_context(&documents);
        let prompt = self.build_prompt(&context, question);
        
        if self.verbose {
            println!("上下文长度: {} 字符", context.len());
            println!("Prompt 长度: {} 字符", prompt.len());
        }
        
        if self.verbose {
            println!("\n--- 步骤 3: LLM 生成答案 ---");
        }
        
        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("LLM 调用失败: {}", e)))?;
        
        let answer = response.content;
        
        if self.verbose {
            println!("答案: {}", answer);
            println!("=== RetrievalQA 完成 ===\n");
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new() {
        let llm = crate::OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));
        
        let qa = RetrievalQA::new(llm, retriever);
        
        assert_eq!(qa.input_keys(), vec!["query"]);
        assert_eq!(qa.output_keys(), vec!["result"]);
        assert_eq!(qa.name(), "retrieval_qa");
        assert_eq!(qa.k(), 4);
    }
    
    #[test]
    fn test_with_options() {
        let llm = crate::OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));
        
        let qa = RetrievalQA::new(llm, retriever)
            .with_k(5)
            .with_input_key("question")
            .with_output_key("answer")
            .with_return_source_documents(true)
            .with_verbose(true);
        
        assert_eq!(qa.input_keys(), vec!["question"]);
        assert_eq!(qa.output_keys(), vec!["answer", "source_documents"]);
        assert_eq!(qa.k(), 5);
        assert!(qa.verbose);
    }
    
    #[test]
    fn test_format_context() {
        let llm = crate::OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));
        
        let qa = RetrievalQA::new(llm, retriever);
        
        let docs = vec![
            Document::new("文档1内容"),
            Document::new("文档2内容"),
        ];
        
        let context = qa.format_context(&docs);
        assert!(context.contains("文档1内容"));
        assert!(context.contains("文档2内容"));
    }
    
    #[test]
    fn test_build_prompt() {
        let llm = crate::OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));
        
        let qa = RetrievalQA::new(llm, retriever);
        
        let prompt = qa.build_prompt("这是上下文", "什么是 Rust?");
        
        assert!(prompt.contains("这是上下文"));
        assert!(prompt.contains("什么是 Rust?"));
    }
    
    #[test]
    fn test_custom_prompt_template() {
        let llm = crate::OpenAIChat::new(crate::OpenAIConfig::default());
        let retriever = Arc::new(crate::retrieval::SimilarityRetriever::new(
            Arc::new(crate::vector_stores::InMemoryVectorStore::new()),
            Arc::new(crate::embeddings::MockEmbeddings::new(64)),
        ));
        
        let custom_template = "背景信息：{context}\n请回答：{question}";
        
        let qa = RetrievalQA::new(llm, retriever)
            .with_prompt_template(custom_template);
        
        let prompt = qa.build_prompt("测试上下文", "测试问题");
        
        assert!(prompt.contains("背景信息"));
        assert!(prompt.contains("测试上下文"));
        assert!(prompt.contains("测试问题"));
    }
}