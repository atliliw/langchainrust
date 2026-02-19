use crate::agent::AgentError;
use crate::llms::{LLM, LLMQwen, ModelConfig};
use crate::messages::Message;
use crate::prompts::ChatPromptTemplate;
use serde::Deserialize;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use super::types::{AnyLLM, RoutingState};

/// 解析难度参数
pub fn parse_difficulty(vars: &HashMap<String, String>) -> u8 {
    let raw = vars
        .get("difficulty")
        .or_else(|| vars.get("难度"))
        .or_else(|| vars.get("level"))
        .map(|s| s.trim())
        .unwrap_or("1");
    let parsed = raw.parse::<u8>().unwrap_or(1);
    parsed.clamp(1, 10)
}

/// 生成路由缓存 key
pub fn routing_key(input: &str, difficulty: u8) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    input.hash(&mut hasher);
    difficulty.hash(&mut hasher);
    hasher.finish()
}

/// 获取模型的系数
pub fn model_factor(model: &ModelConfig) -> u8 {
    match model {
        ModelConfig::OpenAI(cfg) => cfg.factor,
        ModelConfig::Qwen(cfg) => cfg.factor,
    }
}

/// 获取模型标识
pub fn model_id(model: &ModelConfig) -> (&'static str, &str) {
    match model {
        ModelConfig::OpenAI(cfg) => ("openai", cfg.model.as_str()),
        ModelConfig::Qwen(cfg) => ("qwen", cfg.model.as_str()),
    }
}

/// 解析模型路由响应
pub fn parse_model_response(raw: &str) -> (Option<String>, Option<String>) {
    #[derive(Deserialize)]
    struct ModelChoice {
        provider: String,
        model: String,
    }

    let raw = raw.trim();

    if raw.starts_with('{') && let Ok(choice) = serde_json::from_str::<ModelChoice>(raw) {
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

/// 将模型列表格式化为文本
pub fn models_as_text(models: &[ModelConfig]) -> String {
    let mut items = models.to_owned();
    items.sort_by(|a, b| {
        let (ap, am) = model_id(a);
        let af = model_factor(a);
        let (bp, bm) = model_id(b);
        let bf = model_factor(b);

        ap.cmp(bp).then(af.cmp(&bf)).then(am.cmp(bm))
    });

    items
        .into_iter()
        .map(|m| {
            let (provider, model) = model_id(&m);
            let factor = model_factor(&m);
            format!(
                "{{\"provider\":\"{}\",\"model\":\"{}\",\"factor\":{}}}",
                provider, model, factor
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// 在模型列表中查找指定模型
pub fn find_model_config(
    models: &[ModelConfig],
    provider: &str,
    model: &str,
) -> Option<ModelConfig> {
    let provider_lower = provider.to_ascii_lowercase();
    let model_clean = model.trim().to_ascii_lowercase();

    models.iter().find_map(|m| {
        let (p, name) = model_id(m);
        if p.to_ascii_lowercase() == provider_lower
            && name.to_ascii_lowercase() == model_clean
        {
            Some(m.clone())
        } else {
            None
        }
    })
}

/// 备选模型选择
pub fn fallback_choose_model(
    models: &[ModelConfig],
    difficulty: u8,
) -> Result<ModelConfig, AgentError> {
    if models.is_empty() {
        return Err(AgentError("模型目录为空".into()));
    }

    let mut items = models.to_owned();
    items.sort_by_key(model_factor);

    if let Some(best) = items.iter().find(|m| model_factor(m) >= difficulty) {
        return Ok(best.clone());
    }

    Ok(items
        .into_iter()
        .max_by_key(model_factor)
        .expect("models is not empty"))
}

/// 构建 AnyLLM 实例
pub fn build_llm(model: &ModelConfig) -> AnyLLM {
    match model {
        ModelConfig::OpenAI(cfg) => AnyLLM::OpenAI(LLM::new(cfg.clone())),
        ModelConfig::Qwen(cfg) => AnyLLM::Qwen(LLMQwen::new(&cfg.api_key, &cfg.base_url, &cfg.model)),
    }
}

/// 选择 LLM（带模型路由）
pub async fn choose_llm(
    llm: &LLM,
    models: &Option<Vec<ModelConfig>>,
    routing_state: &Mutex<Option<RoutingState>>,
    input: &str,
    vars: &HashMap<String, String>,
) -> Result<AnyLLM, AgentError> {
    let models = match models {
        Some(m) => m,
        None => return Ok(AnyLLM::OpenAI(llm.clone())),
    };

    let difficulty = parse_difficulty(vars);
    let key = routing_key(input, difficulty);

    // 检查缓存
    {
        let state = routing_state.lock().unwrap();
        if let Some(s) = &*state && s.key == key {
            return Ok(s.llm.clone());
        }
    }

    let difficulty_str = difficulty.to_string();
    let catalog_str = models_as_text(models);

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

    let raw = llm
        .invoke_chat_template(&prompt, &values)
        .await
        .map_err(|e| AgentError(format!("路由 LLM 调用失败: {}", e)))?;

    let (provider, model) = parse_model_response(&raw);

    let chosen = if let (Some(p), Some(m)) = (provider, model) {
        find_model_config(models, &p, &m)
    } else {
        None
    };

    let chosen = match chosen {
        Some(m) => m,
        None => fallback_choose_model(models, difficulty)?,
    };

    let (provider, model) = model_id(&chosen);
    let factor = model_factor(&chosen);
    println!(
        "Routed model: provider={}, model={}, factor={}, difficulty={}",
        provider, model, factor, difficulty
    );

    let result_llm = build_llm(&chosen);

    // 更新缓存
    let mut state = routing_state.lock().unwrap();
    *state = Some(RoutingState {
        key,
        llm: result_llm.clone(),
    });

    Ok(result_llm)
}
