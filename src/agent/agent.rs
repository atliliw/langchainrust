use crate::agent::{Agent, AgentAction, AgentError};
use crate::llms::{LLM, LLMQwen, ModelConfig};
use crate::memory::Memory;
use crate::messages::Message;
use crate::prompts::ChatPromptTemplate;
use crate::tools::Tool;
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

pub struct ReActAgent {
    llm: LLM,
    tools: Vec<Arc<dyn Tool>>,
    memory: Option<Mutex<Box<dyn Memory>>>,
    user_template: Option<ChatPromptTemplate>,
    // 模型路由相关字段（可选）
    models: Option<Vec<ModelConfig>>,
    routing_state: Mutex<Option<RoutingState>>,
}

#[derive(Debug, Clone)]
struct RoutingState {
    key: u64,
    llm: AnyLLM,
}

impl ReActAgent {
    pub fn new(llm: LLM, tools: Vec<Arc<dyn Tool>>, memory: Option<Box<dyn Memory>>) -> Self {
        let wrapped_memory = memory.map(Mutex::new);
        Self {
            llm,
            tools,
            memory: wrapped_memory,
            user_template: None,
            models: None,
            routing_state: Mutex::new(None),
        }
    }
    pub fn with_template(
        llm: LLM,
        tools: Vec<Arc<dyn Tool>>,
        memory: Option<Box<dyn Memory>>,
        template: ChatPromptTemplate,
    ) -> Self {
        let wrapped_memory = memory.map(Mutex::new);
        Self {
            llm,
            tools,
            memory: wrapped_memory,
            user_template: Some(template),
            models: None,
            routing_state: Mutex::new(None),
        }
    }

    /// 创建带模型路由功能的 Agent
    pub fn with_models(
        llm: LLM,
        models: Vec<ModelConfig>,
        tools: Vec<Arc<dyn Tool>>,
        memory: Option<Box<dyn Memory>>,
        template: Option<ChatPromptTemplate>,
    ) -> Self {
        let wrapped_memory = memory.map(Mutex::new);
        Self {
            llm,
            tools,
            memory: wrapped_memory,
            user_template: template,
            models: Some(models),
            routing_state: Mutex::new(None),
        }
    }

    fn tool_descriptions(&self) -> String {
        self.tools
            .iter()
            .map(|t| {
                let params = t.parameters();
                let param_str = if params.is_empty() {
                    "无参数".to_string()
                } else {
                    params
                        .iter()
                        .map(|(k, v)| format!("{}: {}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                format!("{} - {} (参数: {})", t.name(), t.description(), param_str)
            })
            .collect::<Vec<_>>()
            .join("\n")
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
        let response = response.trim();
        
        // 检查是否有工具调用标记 [TOOL: tool_name key=value ...]
        if response.contains("[TOOL:") {
            for line in response.lines() {
                if line.contains("[TOOL:")
                    && let Some(start) = line.find("[TOOL:")
                    && let Some(end) = line.find("]")
                {
                    let content = &line[start + 6..end].trim();
                    let parts: Vec<&str> = content.split_whitespace().collect();
                    
                    if parts.is_empty() {
                        continue;
                    }
                    
                    let tool_name = parts[0].to_string();
                    let mut params = HashMap::new();
                    
                    for part in &parts[1..] {
                        if let Some((k, v)) = part.split_once('=') {
                            params.insert(k.to_string(), v.to_string());
                        }
                    }
                    
                    return Ok(AgentAction::ToolCall(tool_name, params));
                }
            }
        }
        
        // 兼容旧格式 "行为：tool_name key=value"
        for line in response.lines() {
            if line.starts_with("行为：") {
                let rest = line.trim_start_matches("行为：").trim();
                let parts: Vec<&str> = rest.splitn(2, ' ').collect();

                if parts.is_empty() {
                    continue;
                }

                let tool_name = parts[0].to_string();
                let mut params = HashMap::new();

                if parts.len() == 2 {
                    for pair in parts[1].split_whitespace() {
                        if let Some((k, v)) = pair.split_once('=') {
                            params.insert(k.to_string(), v.to_string());
                        }
                    }
                }

                return Ok(AgentAction::ToolCall(tool_name, params));
            }
        }

        // 没有工具调用，直接作为最终答案
        Ok(AgentAction::FinalAnswer(response.to_string()))
    }

    // ===== 模型路由相关方法 =====

    fn parse_difficulty(vars: &HashMap<String, String>) -> u8 {
        let raw = vars
            .get("difficulty")
            .or_else(|| vars.get("难度"))
            .or_else(|| vars.get("level"))
            .map(|s| s.trim())
            .unwrap_or("1");  // 默认为 1
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
        
        // 首先尝试 JSON 格式
        if raw.starts_with('{') &&
            let Ok(choice) = serde_json::from_str::<ModelChoice>(raw)
        {
            return (Some(choice.provider), Some(choice.model));
        }
        
        // 然后尝试 "provider: xxx, model: yyy" 格式
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
        
        // 如果单行格式 "provider: xxx, model: yyy"
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
        // 如果没有配置模型路由，直接使用默认 LLM
        let models = match &self.models {
            Some(m) => m,
            None => {
                // 没有模型列表，使用默认的 LLM
                return Ok(AnyLLM::OpenAI(self.llm.clone()));
            }
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

        // 尝试解析 JSON 格式，如果失败则尝试解析 "provider: xxx, model: yyy" 格式
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
impl Agent for ReActAgent {
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
        intermediate_steps: Option<&str>,
        vars: &HashMap<String, String>,
    ) -> Result<AgentAction, AgentError> {
        // 选择 LLM（如果有模型路由则使用路由选择，否则使用默认 LLM）
        let llm = self.choose_llm(input, vars).await?;

        let tools_str = self.tool_descriptions();
        let input_str = input.to_string();
        let scratchpad_str = match intermediate_steps {
            Some(s) if !s.is_empty() => format!("\n上一步工具执行结果：{}\n", s),
            _ => String::new(),
        };

        let mut chat_template = if let Some(t) = &self.user_template {
            t.clone()
        } else {
            let system_msg = if self.tools.is_empty() {
                "你是一个 AI 助手。".to_string()
            } else {
                let tool_hint = if scratchpad_str.is_empty() {
                    format!(
                        "你可以使用以下工具：\n{}\n\n\
                        如果需要使用工具，只输出一行：[TOOL: 工具名 参数名=参数值]\n\
                        如果不需要工具，直接给出答案。",
                        tools_str
                    )
                } else {
                    "工具已执行完毕，请根据工具执行结果直接给出最终答案，不要再调用工具！".to_string()
                };

                format!("你是一个 AI 助手。\n\n{}", tool_hint)
            };

            ChatPromptTemplate::new(vec![
                Message::system(&system_msg),
                Message::human("{scratchpad}用户问题：{input}"),
            ])
        };

        let mut merged: HashMap<String, String> = vars.clone();
        merged.insert("input".to_string(), input_str);
        merged.insert("scratchpad".to_string(), scratchpad_str);

        let merged_refs: HashMap<&str, &str> = merged
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

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
