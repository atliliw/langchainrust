use crate::agent::{Agent, AgentAction, AgentError,ReActAgent};
use crate::llms::LLM;
use crate::messages::Message;
use crate::prompts::ChatPromptTemplate;
use crate::tools::Tool;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::memory::Memory;

impl ReActAgent {
    /// Create a ReAct agent with optional memory and no custom template.
    pub fn new(llm: LLM, tools: Vec<Arc<dyn Tool>>, memory: Option<Box<dyn Memory>>) -> Self {
        let wrapped_memory = memory.map(|m| Mutex::new(m));
        Self {
            llm,
            tools,
            memory: wrapped_memory,
            user_template: None,
            verbose: false,
        }
    }
    /// Create a ReAct agent that uses a user-provided chat prompt template.
    pub fn with_template(llm: LLM, tools: Vec<Arc<dyn Tool>>, memory: Option<Box<dyn Memory>>, template: ChatPromptTemplate) -> Self {
        let wrapped_memory = memory.map(|m| Mutex::new(m));
        Self {
            llm,
            tools,
            memory: wrapped_memory,
            user_template: Some(template),
            verbose: false,
        }
    }

    /// Enable or disable verbose logging of the final chat prompt.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build a human-readable description of the registered tools.
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

    /// Return the current memory context as a single string.
    pub fn memory_context(&self) -> String {
        if let Some(mem) = &self.memory {
            let m = mem.lock().unwrap();
            m.context()
        } else {
            String::new()
        }
    }

    /// Parse the LLM response into either a tool call or a final answer.
    fn parse_response(&self, response: &str) -> Result<AgentAction, AgentError> {
        for line in response.lines() {
            if line.starts_with("行为：") {
                let rest = line.trim_start_matches("行为：").trim();
                let parts: Vec<&str> = rest.splitn(2, ' ').collect();

                if parts.is_empty() {
                    return Err(AgentError("行为行格式无效".into()));
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

        Ok(AgentAction::FinalAnswer(response.trim().to_string()))
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
        let tools_str = self.tool_descriptions();
        let input_str = input.to_string();
        let scratchpad_str = intermediate_steps.unwrap_or("").to_string();

        let mut chat_template = if let Some(t) = &self.user_template {
            t.clone()
        } else {
            ChatPromptTemplate::new(vec![
                Message::system("你是一个 AI 助手，可以使用以下工具解决问题。\n\n可用工具：\n{tools}"),
                Message::human("用户问题：{input}\n上一步结果：{scratchpad}"),
            ])
        };

        let mut merged: HashMap<String, String> = vars.clone();
        merged.insert("tools".to_string(), tools_str);
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

        if self.verbose {
            match chat_template.format(&merged_refs) {
                Ok(messages) => {
                    println!("===== ReActAgent verbose prompt =====");
                    for (idx, msg) in messages.iter().enumerate() {
                        println!("[{}][{}] {}", idx, msg.role(), msg.content());
                    }
                    println!("=====================================");
                }
                Err(e) => {
                    eprintln!("ReActAgent verbose format error: {}", e);
                }
            }
        }

        let response = self
            .llm
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
