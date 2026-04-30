// src/chains/document_chains.rs
//! Document processing chains
//!
//! 文档处理 Chain，提供对多个文档进行 LLM 处理的能力：
//! - StuffDocumentsChain: 将所有文档一次性填充到 prompt 中
//! - RefineDocumentsChain: 逐文档迭代优化答案
//! - MapReduceDocumentsChain: 并行处理文档后合并结果
//! - MapRerankDocumentsChain: 并行处理文档后按相关性排序

use async_trait::async_trait;
use futures_util::future::try_join_all;
use std::collections::HashMap;
use serde_json::Value;


use super::base::{BaseChain, ChainResult, ChainError};
use crate::language_models::OpenAIChat;
use crate::retrieval::Document;
use crate::schema::Message;
use crate::Runnable;

// ============================================================
// StuffDocumentsChain
// ============================================================

/// 默认的 Stuff 处理提示词模板
const DEFAULT_STUFF_PROMPT: &str = "请根据以下参考信息回答用户的问题。

参考信息：
{context}

问题：{input}

回答：";

/// StuffDocumentsChain
///
/// 将所有文档一次性填充到 prompt 中交给 LLM 处理。
/// 适用于文档总量不超过 LLM 上下文窗口的场景。
///
/// # 示例
/// ```ignore
/// use langchainrust::{StuffDocumentsChain, OpenAIChat};
///
/// let chain = StuffDocumentsChain::new(llm);
/// let result = chain.invoke_with_documents(docs, "问题").await?;
/// ```
pub struct StuffDocumentsChain {
    llm: OpenAIChat,
    prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
    /// 文档最大字符数（超过则截断）
    max_doc_length: Option<usize>,
}

impl StuffDocumentsChain {
    pub fn new(llm: OpenAIChat) -> Self {
        Self {
            llm,
            prompt_template: DEFAULT_STUFF_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "stuff_documents".to_string(),
            verbose: false,
            max_doc_length: None,
        }
    }

    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
        self
    }

    pub fn with_document_variable(mut self, name: impl Into<String>) -> Self {
        self.document_variable_name = name.into();
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

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_max_doc_length(mut self, max: usize) -> Self {
        self.max_doc_length = Some(max);
        self
    }

    /// 格式化文档列表为上下文文本
    pub fn format_documents(&self, documents: &[Document]) -> String {
        let mut parts = Vec::new();
        for (i, doc) in documents.iter().enumerate() {
            let mut content = doc.content.clone();
            if let Some(max_len) = self.max_doc_length {
                let char_count: usize = content.chars().count();
                if char_count > max_len {
                    content = content.chars().take(max_len).collect::<String>();
                    content.push_str("...\n[文档已截断]");
                }
            }
            parts.push(format!("文档 {}:\n{}", i + 1, content));
        }
        parts.join("\n\n---\n\n")
    }

    /// 构建 prompt
    pub fn build_prompt(&self, context: &str, input: &str) -> String {
        let template = self.prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input);
        template
    }

    /// 直接使用文档和输入调用
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<String, ChainError> {
        let context = self.format_documents(&documents);

        if self.verbose {
            println!("\n=== StuffDocumentsChain ===");
            println!("文档数量: {}", documents.len());
            println!("上下文长度: {} 字符", context.len());
        }

        let prompt = self.build_prompt(&context, input);

        if self.verbose {
            println!("Prompt 长度: {} 字符", prompt.len());
        }

        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("LLM 调用失败: {}", e)))?;

        let output = response.content;

        if self.verbose {
            println!("输出: {}", output);
            println!("=== StuffDocumentsChain 完成 ===\n");
        }

        Ok(output)
    }
}

#[async_trait]
impl BaseChain for StuffDocumentsChain {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key, "documents"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs.get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents: Vec<Document> = inputs.get("documents")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect()
            })
            .ok_or_else(|| ChainError::MissingInput("documents".to_string()))?;

        let output = self.invoke_with_documents(documents, input).await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================
// RefineDocumentsChain
// ============================================================

/// 默认的初始处理提示词模板
const DEFAULT_REFINE_INITIAL_PROMPT: &str = "请根据以下参考信息回答问题。

参考信息：
{context}

问题：{input}

回答：";

/// 默认的迭代优化提示词模板
const DEFAULT_REFINE_PROMPT: &str = "你已基于部分信息给出了一个答案。以下是更多参考信息。

已有的答案：
{existing_answer}

新的参考信息：
{context}

请根据新的信息完善或修改你的答案。如果新信息与已有答案不冲突，请合并它们。如果新信息与已有答案冲突，请以新信息为准。

问题：{input}

完善后的答案：";

/// RefineDocumentsChain
///
/// 逐文档迭代优化答案。
/// 先用第一个文档生成初始答案，然后用后续文档不断优化。
/// 适用于文档总量较大的场景（逐个处理避免超出上下文窗口）。
///
/// # 示例
/// ```ignore
/// use langchainrust::{RefineDocumentsChain, OpenAIChat};
///
/// let chain = RefineDocumentsChain::new(llm);
/// let result = chain.invoke_with_documents(docs, "问题").await?;
/// ```
pub struct RefineDocumentsChain {
    llm: OpenAIChat,
    initial_prompt_template: String,
    refine_prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
}

impl RefineDocumentsChain {
    pub fn new(llm: OpenAIChat) -> Self {
        Self {
            llm,
            initial_prompt_template: DEFAULT_REFINE_INITIAL_PROMPT.to_string(),
            refine_prompt_template: DEFAULT_REFINE_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "refine_documents".to_string(),
            verbose: false,
        }
    }

    pub fn with_initial_prompt(mut self, template: impl Into<String>) -> Self {
        self.initial_prompt_template = template.into();
        self
    }

    pub fn with_refine_prompt(mut self, template: impl Into<String>) -> Self {
        self.refine_prompt_template = template.into();
        self
    }

    pub fn with_document_variable(mut self, name: impl Into<String>) -> Self {
        self.document_variable_name = name.into();
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

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn build_initial_prompt(&self, context: &str, input: &str) -> String {
        self.initial_prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
    }

    pub fn build_refine_prompt(&self, context: &str, input: &str, existing_answer: &str) -> String {
        self.refine_prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
            .replace("{existing_answer}", existing_answer)
    }

    /// 直接使用文档和输入调用（迭代优化）
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<String, ChainError> {
        if documents.is_empty() {
            return Err(ChainError::ExecutionError("文档列表为空".to_string()));
        }

        if self.verbose {
            println!("\n=== RefineDocumentsChain ===");
            println!("文档数量: {}", documents.len());
            println!("输入: {}", input);
        }

        // 第一步：用第一个文档生成初始答案
        let first_context = &documents[0].content;
        let initial_prompt = self.build_initial_prompt(first_context, input);

        if self.verbose {
            println!("\n--- 初始处理（文档 1）---");
        }

        let messages = vec![Message::human(&initial_prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("LLM 初始调用失败: {}", e)))?;
        let mut answer = response.content;

        if self.verbose {
            println!("初始答案: {}", answer);
        }

        // 后续步骤：用剩余文档迭代优化
        for (i, doc) in documents[1..].iter().enumerate() {
            if self.verbose {
                println!("\n--- 优化步骤 {}（文档 {}）---", i + 1, i + 2);
            }

            let refine_prompt = self.build_refine_prompt(&doc.content, input, &answer);

            let messages = vec![Message::human(&refine_prompt)];
            let response = self.llm.invoke(messages, None).await
                .map_err(|e| ChainError::ExecutionError(format!("LLM 优化调用失败: {}", e)))?;
            answer = response.content;

            if self.verbose {
                println!("优化后答案: {}", answer);
            }
        }

        if self.verbose {
            println!("=== RefineDocumentsChain 完成 ===\n");
        }

        Ok(answer)
    }
}

#[async_trait]
impl BaseChain for RefineDocumentsChain {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key, "documents"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs.get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents: Vec<Document> = inputs.get("documents")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect()
            })
            .ok_or_else(|| ChainError::MissingInput("documents".to_string()))?;

        let output = self.invoke_with_documents(documents, input).await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================
// MapRerankDocumentsChain
// ============================================================

/// 默认的 Map + Rerank 提示词模板
const DEFAULT_MAP_RERANK_PROMPT: &str = "请根据以下文档回答问题，并给出你对答案的相关性评分（0-100分，越高越相关）。

文档内容：
{context}

问题：{input}

请按以下格式输出：
相关性评分：<分数>
答案：<你的答案>";

/// MapRerankDocumentsChain
///
/// 先对每个文档独立调用 LLM 生成答案并评分，
/// 然后按相关性评分排序，返回最高分的答案。
///
/// 适用于需要从多个文档中选取最佳答案的场景。
///
/// # 示例
/// ```ignore
/// use langchainrust::{MapRerankDocumentsChain, OpenAIChat};
///
/// let chain = MapRerankDocumentsChain::new(llm);
/// let result = chain.invoke_with_documents(docs, "问题").await?;
/// ```
pub struct MapRerankDocumentsChain {
    llm: OpenAIChat,
    map_prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
    /// 返回前 k 个结果（默认 1，即只返回最高分）
    top_k: usize,
}

impl MapRerankDocumentsChain {
    pub fn new(llm: OpenAIChat) -> Self {
        Self {
            llm,
            map_prompt_template: DEFAULT_MAP_RERANK_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "map_rerank_documents".to_string(),
            verbose: false,
            top_k: 1,
        }
    }

    pub fn with_map_prompt(mut self, template: impl Into<String>) -> Self {
        self.map_prompt_template = template.into();
        self
    }

    pub fn with_document_variable(mut self, name: impl Into<String>) -> Self {
        self.document_variable_name = name.into();
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

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 设置返回前 k 个结果
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// 构建 Map 阶段的 prompt
    pub fn build_map_prompt(&self, context: &str, input: &str) -> String {
        self.map_prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
    }

    /// 从 LLM 输出中提取评分和答案
    pub fn extract_score(text: &str) -> (u32, String) {
        let score_re = regex::Regex::new(r"(?i)相关性评分\s*[:：]\s*(\d+)").unwrap();
        if let Some(caps) = score_re.captures(text) {
            if let Ok(score) = caps[1].parse::<u32>() {
                let cleaned = score_re.replace(text, "").trim().to_string();
                let cleaned = cleaned.trim_start_matches("答案").trim_start_matches(&[':', '：'][..]).trim().to_string();
                return (std::cmp::min(score, 100), if cleaned.is_empty() { text.to_string() } else { cleaned });
            }
        }

        let score_re2 = regex::Regex::new(r"(?i)score\s*[:：]\s*(\d+)").unwrap();
        if let Some(caps) = score_re2.captures(text) {
            if let Ok(score) = caps[1].parse::<u32>() {
                let cleaned = score_re2.replace(text, "").trim().to_string();
                return (std::cmp::min(score, 100), if cleaned.is_empty() { text.to_string() } else { cleaned });
            }
        }

        (50, text.to_string())
    }

    async fn map_document(
        &self,
        doc: &Document,
        input: &str,
        index: usize,
    ) -> Result<(u32, String), ChainError> {
        let prompt = self.build_map_prompt(&doc.content, input);
        if self.verbose {
            println!("\n--- Map 文档 {} ---", index + 1);
        }
        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("Map 调用失败（文档 {}）: {}", index + 1, e)))?;
        let (score, answer) = Self::extract_score(&response.content);
        if self.verbose {
            println!("文档 {} 评分: {}，答案: {}", index + 1, score,
                if answer.len() > 80 { &answer[..80] } else { &answer });
        }
        Ok((score, answer))
    }

    /// 直接使用文档和输入调用
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<Vec<(u32, String)>, ChainError> {
        if documents.is_empty() {
            return Err(ChainError::ExecutionError("文档列表为空".to_string()));
        }

        if self.verbose {
            println!("\n=== MapRerankDocumentsChain ===");
            println!("文档数量: {}, 输入: {}", documents.len(), input);
            println!("\n--- Map 阶段 ---");
        }

        let mut map_futures = Vec::new();
        for (i, doc) in documents.iter().enumerate() {
            map_futures.push(self.map_document(doc, input, i));
        }
        let mut results: Vec<(u32, String)> = try_join_all(map_futures).await?;

        results.sort_by(|a, b| b.0.cmp(&a.0));

        if self.verbose {
            println!("\n--- Rerank 阶段 ---");
            for (i, (score, answer)) in results.iter().enumerate() {
                println!("排名 {}: 评分={}, 答案={}", i + 1, score,
                    if answer.len() > 100 { &answer[..100] } else { &answer });
            }
        }

        let top_results: Vec<(u32, String)> = results.into_iter().take(self.top_k).collect();
        if self.verbose {
            println!("最终选取 {} 个最佳结果", top_results.len());
            println!("=== MapRerankDocumentsChain 完成 ===\n");
        }
        Ok(top_results)
    }
}

#[async_trait]
impl BaseChain for MapRerankDocumentsChain {
    fn input_keys(&self) -> Vec<&str> { vec![&self.input_key, "documents"] }
    fn output_keys(&self) -> Vec<&str> { vec![&self.output_key] }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs.get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents: Vec<Document> = inputs.get("documents")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| serde_json::from_value(v.clone()).ok()).collect())
            .ok_or_else(|| ChainError::MissingInput("documents".to_string()))?;

        let results = self.invoke_with_documents(documents, input).await?;
        let output_json: Vec<serde_json::Value> = results.iter()
            .map(|(score, answer)| serde_json::json!({"score": score, "answer": answer}))
            .collect();

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::Array(output_json));
        Ok(result)
    }

    fn name(&self) -> &str { &self.name }
}

// ============================================================
// MapReduceDocumentsChain
// ============================================================

/// 默认的 Map 处理提示词模板
const DEFAULT_MAP_PROMPT: &str = "请根据以下文档内容回答用户的问题。请基于文档内容给出简洁的答案。

文档内容：
{context}

问题：{input}

基于此文档的答案：";

/// 默认的 Reduce 合并提示词模板
const DEFAULT_REDUCE_PROMPT: &str = "以下是多个文档分别给出的答案，请将它们合并为一个完整、连贯的最终答案。

各文档的答案：
{summaries}

原始问题：{input}

最终综合答案：";

/// MapReduceDocumentsChain
///
/// 分两步处理文档：
/// 1. Map：对每个文档独立调用 LLM 生成答案
/// 2. Reduce：将所有独立答案合并为最终答案
///
/// 适用于文档量非常大的场景，map 阶段可以并行处理。
///
/// # 示例
/// ```ignore
/// use langchainrust::{MapReduceDocumentsChain, OpenAIChat};
///
/// let chain = MapReduceDocumentsChain::new(llm);
/// let result = chain.invoke_with_documents(docs, "问题").await?;
/// ```
pub struct MapReduceDocumentsChain {
    llm: OpenAIChat,
    map_prompt_template: String,
    reduce_prompt_template: String,
    document_variable_name: String,
    input_key: String,
    output_key: String,
    name: String,
    verbose: bool,
}

impl MapReduceDocumentsChain {
    pub fn new(llm: OpenAIChat) -> Self {
        Self {
            llm,
            map_prompt_template: DEFAULT_MAP_PROMPT.to_string(),
            reduce_prompt_template: DEFAULT_REDUCE_PROMPT.to_string(),
            document_variable_name: "context".to_string(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            name: "map_reduce_documents".to_string(),
            verbose: false,
        }
    }

    pub fn with_map_prompt(mut self, template: impl Into<String>) -> Self {
        self.map_prompt_template = template.into();
        self
    }

    pub fn with_reduce_prompt(mut self, template: impl Into<String>) -> Self {
        self.reduce_prompt_template = template.into();
        self
    }

    pub fn with_document_variable(mut self, name: impl Into<String>) -> Self {
        self.document_variable_name = name.into();
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

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn build_map_prompt(&self, context: &str, input: &str) -> String {
        self.map_prompt_template
            .replace(&format!("{{{}}}", self.document_variable_name), context)
            .replace("{input}", input)
    }

    pub fn build_reduce_prompt(&self, summaries: &[String], input: &str) -> String {
        let summaries_text = summaries.iter()
            .enumerate()
            .map(|(i, s)| format!("文档 {} 的答案：\n{}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n\n");

        self.reduce_prompt_template
            .replace("{summaries}", &summaries_text)
            .replace("{input}", input)
    }

    /// Map 阶段：对单个文档调用 LLM
    async fn map_document(
        &self,
        doc: &Document,
        input: &str,
        index: usize,
    ) -> Result<String, ChainError> {
        let prompt = self.build_map_prompt(&doc.content, input);

        if self.verbose {
            println!("\n--- Map 文档 {} ---", index + 1);
        }

        let messages = vec![Message::human(&prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("Map 调用失败（文档 {}）: {}", index + 1, e)))?;

        if self.verbose {
            println!("文档 {} 答案: {}", index + 1, response.content);
        }

        Ok(response.content)
    }

    /// 直接使用文档和输入调用
    pub async fn invoke_with_documents(
        &self,
        documents: Vec<Document>,
        input: &str,
    ) -> Result<String, ChainError> {
        if documents.is_empty() {
            return Err(ChainError::ExecutionError("文档列表为空".to_string()));
        }

        if self.verbose {
            println!("\n=== MapReduceDocumentsChain ===");
            println!("文档数量: {}", documents.len());
            println!("输入: {}", input);
        }

        // Map 阶段：并行处理每个文档
        if self.verbose {
            println!("\n--- Map 阶段 ---");
        }

        let mut map_futures = Vec::new();
        for (i, doc) in documents.iter().enumerate() {
            map_futures.push(self.map_document(doc, input, i));
        }
        let summaries: Vec<String> = try_join_all(map_futures).await?;

        if self.verbose {
            println!("\n--- Reduce 阶段 ---");
        }

        // Reduce 阶段：合并所有答案
        let reduce_prompt = self.build_reduce_prompt(&summaries, input);

        if self.verbose {
            println!("合并 {} 个文档的答案", summaries.len());
        }

        let messages = vec![Message::human(&reduce_prompt)];
        let response = self.llm.invoke(messages, None).await
            .map_err(|e| ChainError::ExecutionError(format!("Reduce 调用失败: {}", e)))?;

        let final_answer = response.content;

        if self.verbose {
            println!("最终答案: {}", final_answer);
            println!("=== MapReduceDocumentsChain 完成 ===\n");
        }

        Ok(final_answer)
    }
}

#[async_trait]
impl BaseChain for MapReduceDocumentsChain {
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.input_key, "documents"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec![&self.output_key]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs.get(&self.input_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChainError::MissingInput(self.input_key.clone()))?;

        let documents: Vec<Document> = inputs.get("documents")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| serde_json::from_value(v.clone()).ok())
                    .collect()
            })
            .ok_or_else(|| ChainError::MissingInput("documents".to_string()))?;

        let output = self.invoke_with_documents(documents, input).await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), Value::String(output));
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }
}
