mod agent_impl;
mod parser;
mod retrieval;
mod routing;
mod types;

use crate::llms::{LLM, ModelConfig};
use crate::memory::Memory;
use crate::prompts::ChatPromptTemplate;
use crate::retrieval::Retriever;
use crate::tools::Tool;
use std::sync::{Arc, Mutex};

pub use types::{AnyLLM, RoutingState};

/// ReAct Agent - 支持工具调用、模型路由、RAG 检索
pub struct ReActAgent {
    pub llm: LLM,
    pub tools: Vec<Arc<dyn Tool>>,
    pub memory: Option<Mutex<Box<dyn Memory>>>,
    pub user_template: Option<ChatPromptTemplate>,
    pub models: Option<Vec<ModelConfig>>,
    pub routing_state: Mutex<Option<RoutingState>>,
    pub retriever: Option<Arc<dyn Retriever>>,
    pub top_k: usize,
}

impl ReActAgent {
    pub fn new(llm: LLM, tools: Vec<Arc<dyn Tool>>, memory: Option<Box<dyn Memory>>) -> Self {
        Self {
            llm,
            tools,
            memory: memory.map(Mutex::new),
            user_template: None,
            models: None,
            routing_state: Mutex::new(None),
            retriever: None,
            top_k: 3,
        }
    }

    pub fn with_template(
        llm: LLM,
        tools: Vec<Arc<dyn Tool>>,
        memory: Option<Box<dyn Memory>>,
        template: ChatPromptTemplate,
    ) -> Self {
        Self {
            llm,
            tools,
            memory: memory.map(Mutex::new),
            user_template: Some(template),
            models: None,
            routing_state: Mutex::new(None),
            retriever: None,
            top_k: 3,
        }
    }

    pub fn with_models(
        llm: LLM,
        models: Vec<ModelConfig>,
        tools: Vec<Arc<dyn Tool>>,
        memory: Option<Box<dyn Memory>>,
        template: Option<ChatPromptTemplate>,
    ) -> Self {
        Self {
            llm,
            tools,
            memory: memory.map(Mutex::new),
            user_template: template,
            models: Some(models),
            routing_state: Mutex::new(None),
            retriever: None,
            top_k: 3,
        }
    }

    pub fn with_retriever(
        llm: LLM,
        tools: Vec<Arc<dyn Tool>>,
        memory: Option<Box<dyn Memory>>,
        retriever: Arc<dyn Retriever>,
        top_k: usize,
    ) -> Self {
        Self {
            llm,
            tools,
            memory: memory.map(Mutex::new),
            user_template: None,
            models: None,
            routing_state: Mutex::new(None),
            retriever: Some(retriever),
            top_k,
        }
    }

    pub fn with_retriever_and_template(
        llm: LLM,
        tools: Vec<Arc<dyn Tool>>,
        memory: Option<Box<dyn Memory>>,
        retriever: Arc<dyn Retriever>,
        top_k: usize,
        template: ChatPromptTemplate,
    ) -> Self {
        Self {
            llm,
            tools,
            memory: memory.map(Mutex::new),
            user_template: Some(template),
            models: None,
            routing_state: Mutex::new(None),
            retriever: Some(retriever),
            top_k,
        }
    }

    pub fn with_all(
        llm: LLM,
        tools: Vec<Arc<dyn Tool>>,
        memory: Option<Box<dyn Memory>>,
        template: Option<ChatPromptTemplate>,
        models: Option<Vec<ModelConfig>>,
        retriever: Option<Arc<dyn Retriever>>,
        top_k: usize,
    ) -> Self {
        Self {
            llm,
            tools,
            memory: memory.map(Mutex::new),
            user_template: template,
            models,
            routing_state: Mutex::new(None),
            retriever,
            top_k,
        }
    }

    pub fn memory_context(&self) -> String {
        if let Some(mem) = &self.memory {
            let m = mem.lock().unwrap();
            m.context()
        } else {
            String::new()
        }
    }
}
