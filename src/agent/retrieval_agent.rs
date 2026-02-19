use crate::agent::{Agent, AgentAction, AgentError};
use crate::llms::{LLM, LLMQwen, ModelConfig};
use crate::memory::Memory;
use crate::messages::Message;
use crate::prompts::ChatPromptTemplate;
use crate::retrieval::{Retriever, SearchResult};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
enum AnyLLM {
    OpenAI(LLM),
    Qwen(LLMQwen),
}

impl AnyLLM {
    async fn invoke_chat_template(
        &self,
        template: &ChatPromptTemplate,
        values: &HashMap<&str, &str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        match self {
            AnyLLM::OpenAI(llm) => llm.invoke_chat_template(template, values).await,
            AnyLLM::Qwen(llm) => llm.invoke_chat_template(template, values).await,
        }
    }
}

/// 带检索增强的 Agent
/// 
/// 在调用 LLM 之前，会先从向量数据库检索相关文档，
/// 然后将这些文档作为上下文注入到 prompt 中。
pub struct RetrievalAgent {
    llm: LLM,
    retriever: Arc<dyn Retriever>,
    memory: Option<Mutex<Box<dyn Memory>>>,
    user_template: Option<ChatPromptTemplate>,
    /// 检索的文档数量
    top_k: usize,
    /// 模型路由相关字段（可选）
    models: Option<Vec<ModelConfig>>,
    routing_state: Mutex<Option<RoutingState>>,
}

#[derive(Debug, Clone)]
struct RoutingState {
    key: u64,
    llm: AnyLLM,
}

impl RetrievalAgent {
    /// 创建新的 RetrievalAgent
    /// 
    /// # 参数
    /// - `llm`: LLM 实例
    /// - `retriever`: 检索器（从向量数据库检索文档）
    /// - `memory`: 可选的记忆模块
    /// - `top_k`: 每次检索返回的文档数量
    pub fn new(
        llm: LLM,
        retriever: Arc<dyn Retriever>,
        memory: Option<Box<dyn Memory>>,
        top_k: usize,
    ) -> Self {
        Self {
            llm,
            retriever,
            memory: memory.map(Mutex::new),
            user_template: None,
            top_k,
            models: None,
            routing_state: Mutex::new(None),
        }
    }

    /// 创建带自定义模板的 RetrievalAgent
    pub fn with_template(
        llm: LLM,
        retriever: Arc<dyn Retriever>,
        memory: Option<Box<dyn Memory>>,
        top_k: usize,
        template: ChatPromptTemplate,
    ) -> Self {
        Self {
            llm,
            retriever,
            memory: memory.map(Mutex::new),
            user_template: Some(template),
            top_k,
            models: None,
            routing_state: Mutex::new(None),
        }
    }

    /// 创建带模型路由的 RetrievalAgent
    pub fn with_models(
        llm: LLM,
        retriever: Arc<dyn Retriever>,
        memory: Option<Box<dyn Memory>>,
        top_k: usize,
        template: Option<ChatPromptTemplate>,
        models: Vec<ModelConfig>,
    ) -> Self {
        Self {
            llm,
            retriever,
            memory: memory.map(Mutex::new),
            user_template: template,
            top_k,
            models: Some(models),
            routing_state: Mutex::new(None),
        }
    }

    /// 检索相关文档
    async fn retrieve_context(&self, query: &str) -> Result<Vec<SearchResult>, AgentError> {
        self.retriever
            .retrieve(query, self.top_k)
            .await
            .map_err(|e| AgentError(format!("检索失败: {}", e)))
    }

    /// 将检索结果格式化为上下文字符串
    fn format_context(results: &[SearchResult]) -> String {
        if results.is_empty() {
            return "没有找到相关文档。".to_string();
        }

        let mut context = String::new();
        for (i, result) in results.iter().enumerate() {
            context.push_str(&format!("[文档{}]\n{}\n\n", i + 1, result.chunk.content));
        }
        context
    }

    pub fn memory_context(&self) -> String {
        if let Some(mem) = &self.memory {
            let m = mem.lock().unwrap();
            m.context()
        } else {
            String::new()
        }
    }

    fn parse_response(&self, response: &str) -> Result<AgentAction, AgentError> {
        // RetrievalAgent 直接返回最终答案，不解析工具调用
        Ok(AgentAction::FinalAnswer(response.trim().to_string()))
    }

    // ===== 模型路由相关方法 =====

    fn parse_difficulty(vars: &HashMap<String, String>) -> u8 {
        let raw = vars
            .get("difficulty")
            .or_else(|| vars.get("难度"))
            .or_else(|| vars.get("level"))
            .map(|s| s.trim())
            .unwrap_or("1");
        let parsed = raw.parse::<u8>().unwrap_or(1);
        parsed.clamp(1, 10)
    }

    fn routing_key(input: &str, difficulty: u8) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        input.hash(&mut hasher);
        difficulty.hash(&mut hasher);
        hasher.finish()
    }

    fn model_factor(model: &ModelConfig) -> u8 {
        match model {
            ModelConfig::OpenAI(cfg) => cfg.factor,
            ModelConfig::Qwen(cfg) => cfg.factor,
        }
    }

    fn model_id(model: &ModelConfig) -> (&'static str, &str) {
        match model {
            ModelConfig::OpenAI(cfg) => ("openai", cfg.model.as_str()),
            ModelConfig::Qwen(cfg) => ("qwen", cfg.model.as_str()),
        }
    }

    fn parse_model_response(raw: &str) -> (Option<String>, Option<String>) {
        #[derive(Deserialize)]
        struct ModelChoice {
            provider: String,
            model: String,
        }
        
        let raw = raw.trim();
        
        if raw.starts_with('{') &&
            let Ok(choice) = serde_json::from_str::<ModelChoice>(raw)
        {
            return (Some(choice.provider), Some(choice.model));
        }
        
        let mut provider = None;
        let mut model = None;
        
        for line in raw.lines() {
            let line = line.trim();
            if line.to_ascii_lowercase().starts_with("provider:") {
                provider = line[9..].trim().trim_matches(',').trim().to_string().into();
            } else if line.to_ascii_lowercase().starts_with("model:") {
                model = line[6..].trim().trim_matches(',').trim().to_string().into();
            }
        }
        
        if provider.is_none() && model.is_none() {
            let parts: Vec<&str> = raw.split(',').collect();
            for part in parts {
                let part = part.trim();
                if part.to_ascii_lowercase().starts_with("provider:") {
                    provider = part[9..].trim().to_string().into();
                } else if part.to_ascii_lowercase().starts_with("model:") {
                    model = part[6..].trim().to_string().into();
                }
            }
        }
        
        (provider, model)
    }

    fn models_as_text(models: &[ModelConfig]) -> String {
        let mut items = models.to_owned();
        items.sort_by(|a, b| {
            let (ap, am) = Self::model_id(a);
            let af = Self::model_factor(a);
            let (bp, bm) = Self::model_id(b);
            let bf = Self::model_factor(b);

            ap.cmp(bp).then(af.cmp(&bf)).then(am.cmp(bm))
        });

        items
            .into_iter()
            .map(|m| {
                let (provider, model) = Self::model_id(&m);
                let factor = Self::model_factor(&m);
                format!(
                    "{{\"provider\":\"{}\",\"model\":\"{}\",\"factor\":{}}}",
                    provider, model, factor
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn find_model_config(models: &[ModelConfig], provider: &str, model: &str) -> Option<ModelConfig> {
        let provider_lower = provider.to_ascii_lowercase();
        let model_clean = model.trim().to_ascii_lowercase();
        
        models.iter().find_map(|m| {
            let (p, name) = Self::model_id(m);
            if p.to_ascii_lowercase() == provider_lower && name.to_ascii_lowercase() == model_clean {
                Some(m.clone())
            } else {
                None
            }
        })
    }

    fn fallback_choose_model(models: &[ModelConfig], difficulty: u8) -> Result<ModelConfig, AgentError> {
        if models.is_empty() {
            return Err(AgentError("模型目录为空".into()));
        }

        let mut items = models.to_owned();
        items.sort_by_key(Self::model_factor);

        if let Some(best) = items.iter().find(|m| Self::model_factor(m) >= difficulty) {
            return Ok(best.clone());
        }

        Ok(items
            .into_iter()
            .max_by_key(Self::model_factor)
            .expect("models is not empty"))
    }

    fn build_llm(model: &ModelConfig) -> AnyLLM {
        match model {
            ModelConfig::OpenAI(cfg) => AnyLLM::OpenAI(LLM::new(cfg.clone())),
            ModelConfig::Qwen(cfg) => {
                AnyLLM::Qwen(LLMQwen::new(&cfg.api_key, &cfg.base_url, &cfg.model))
            }
        }
    }

    async fn choose_llm(
        &self,
        input: &str,
        vars: &HashMap<String, String>,
    ) -> Result<AnyLLM, AgentError> {
        let models = match &self.models {
            Some(m) => m,
            None => return Ok(AnyLLM::OpenAI(self.llm.clone())),
        };

        let difficulty = Self::parse_difficulty(vars);
        let key = Self::routing_key(input, difficulty);

        {
            let state = self.routing_state.lock().unwrap();
            if let Some(s) = &*state
                && s.key == key
            {
                return Ok(s.llm.clone());
            }
        }

        let difficulty_str = difficulty.to_string();
        let catalog_str = Self::models_as_text(models);

        let prompt = ChatPromptTemplate::new(vec![
            Message::system(
                "你是一个模型路由器。根据问题难度(1-10)与候选模型列表(含 factor 1-10，越高越贵/越强)，选择一个最合适的模型。",
            ),
            Message::human(
                "问题难度：{difficulty}\n用户问题：{input}\n候选模型（每行一条 JSON）：\n{catalog}\n\n请只输出一行纯JSON（不要markdown代码块），格式如下：\nprovider字段值: openai 或 qwen\nmodel字段值: 从候选列表中选择一个具体的模型名称\n\n示例输出：provider: openai, model: gpt-4",
            ),
        ]);

        let values_owned: HashMap<String, String> = [
            ("difficulty".to_string(), difficulty_str),
            ("input".to_string(), input.to_string()),
            ("catalog".to_string(), catalog_str),
        ]
        .into_iter()
        .collect();

        let values: HashMap<&str, &str> = values_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let raw = self
            .llm
            .invoke_chat_template(&prompt, &values)
            .await
            .map_err(|e| AgentError(format!("路由 LLM 调用失败: {}", e)))?;

        let (provider, model) = Self::parse_model_response(&raw);
        
        let chosen = if let (Some(p), Some(m)) = (provider, model) {
            Self::find_model_config(models, &p, &m)
        } else {
            None
        };

        let chosen = match chosen {
            Some(m) => m,
            None => Self::fallback_choose_model(models, difficulty)?,
        };

        let (provider, model) = Self::model_id(&chosen);
        let factor = Self::model_factor(&chosen);
        println!(
            "Routed model: provider={}, model={}, factor={}, difficulty={}",
            provider, model, factor, difficulty
        );

        let llm = Self::build_llm(&chosen);

        let mut state = self.routing_state.lock().unwrap();
        *state = Some(RoutingState {
            key,
            llm: llm.clone(),
        });

        Ok(llm)
    }
}

#[async_trait]
impl Agent for RetrievalAgent {
    async fn get_next_step(
        &self,
        input: &str,
        intermediate_steps: Option<&str>,
    ) -> Result<AgentAction, AgentError> {
        self.get_next_step_with_vars(input, intermediate_steps, &HashMap::new())
            .await
    }

    async fn get_next_step_with_vars(
        &self,
        input: &str,
        _intermediate_steps: Option<&str>,
        vars: &HashMap<String, String>,
    ) -> Result<AgentAction, AgentError> {
        let llm = self.choose_llm(input, vars).await?;

        // 1. 从向量数据库检索相关文档
        println!("[检索] 正在从向量数据库检索相关文档...");
        let search_results = self.retrieve_context(input).await?;
        
        // 2. 格式化检索到的文档为上下文
        let context = Self::format_context(&search_results);
        println!("[检索] 找到 {} 个相关文档", search_results.len());
        for (i, result) in search_results.iter().enumerate() {
            println!("  [{}] 相似度: {:.4}", i + 1, result.score);
        }

        // 3. 构建 prompt
        let mut chat_template = if let Some(t) = &self.user_template {
            t.clone()
        } else {
            ChatPromptTemplate::new(vec![
                Message::system(
                    "你是一个 AI 助手。请根据提供的参考文档回答用户问题。\n\
                    如果参考文档中没有相关信息，请说明并基于你的知识回答。\n\
                    回答时请标注信息来源。\n\n\
                    参考文档：\n{context}",
                ),
                Message::human("{input}"),
            ])
        };

        let mut merged: HashMap<String, String> = vars.clone();
        merged.insert("context".to_string(), context);
        merged.insert("input".to_string(), input.to_string());

        let merged_refs: HashMap<&str, &str> = merged
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        // 4. 添加记忆上下文
        if let Some(mem) = &self.memory {
            let m = mem.lock().unwrap();
            let history_entries = m.history();
            if !history_entries.is_empty() {
                let history_str = format!(
                    "以下是我们的历史对话，请根据上下文进行回答：\n\n{}",
                    history_entries.join("\n")
                );
                chat_template.add_to_front(Message::system(history_str));
            }
        }

        // 5. 调用 LLM
        let response = llm
            .invoke_chat_template(&chat_template, &merged_refs)
            .await
            .map_err(|e| AgentError(format!("LLM 调用失败: {}", e)))?;

        self.parse_response(&response)
    }

    fn add_memory(&self, input: &str, output: &str) {
        if let Some(mem) = &self.memory {
            let mut m = mem.lock().unwrap();
            m.add(input, output);
        }
    }
}

/// 简化版 Retrieval Agent（不使用 Agent trait，直接返回结果）
pub struct SimpleRetrievalAgent {
    llm: LLM,
    retriever: Arc<dyn Retriever>,
    top_k: usize,
}

impl SimpleRetrievalAgent {
    pub fn new(llm: LLM, retriever: Arc<dyn Retriever>, top_k: usize) -> Self {
        Self { llm, retriever, top_k }
    }

    /// 执行 RAG 查询
    pub async fn query(&self, question: &str) -> Result<String, Box<dyn std::error::Error>> {
        // 1. 检索相关文档
        println!("[RAG] 正在检索相关文档...");
        let results = self.retriever.retrieve(question, self.top_k).await?;
        
        if results.is_empty() {
            println!("[RAG] 未找到相关文档，直接使用 LLM 回答");
        } else {
            println!("[RAG] 找到 {} 个相关文档:", results.len());
            for (i, r) in results.iter().enumerate() {
                println!("  [{}] 相似度: {:.4}", i + 1, r.score);
            }
        }

        // 2. 构建上下文
        let context = if results.is_empty() {
            "无相关文档".to_string()
        } else {
            results
                .iter()
                .enumerate()
                .map(|(i, r)| format!("[文档{}]\n{}", i + 1, r.chunk.content))
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        // 3. 构建 prompt
        let template = ChatPromptTemplate::new(vec![
            Message::system(
                "你是一个 AI 助手。请根据提供的参考文档回答用户问题。\n\
                如果参考文档中没有相关信息，请说明。\n\n\
                参考文档：\n{context}",
            ),
            Message::human("{question}"),
        ]);

        let values = HashMap::from([
            ("context", context.as_str()),
            ("question", question),
        ]);

        // 4. 调用 LLM
        let response = self.llm.invoke_chat_template(&template, &values).await?;

        Ok(response)
    }
}
