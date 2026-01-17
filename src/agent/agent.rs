use crate::agent::{Agent, AgentAction, AgentError,ReActAgent};
use crate::llms::LLM;
use crate::messages::Message;
use crate::prompts::{ChatPromptTemplate, PromptTemplate};
use crate::tools::Tool;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::memory::Memory;

impl ReActAgent {
    pub fn new(llm: LLM, tools: Vec<Arc<dyn Tool>>, memory: Option<Box<dyn Memory>>) -> Self {
        let wrapped_memory = memory.map(|m| Mutex::new(m));
        Self { llm, tools, memory: wrapped_memory }
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
        let tools_str = self.tool_descriptions();
        let memory_str = self.memory_context();
        let input_str = input.to_string();
        let scratchpad_str = intermediate_steps.unwrap_or("").to_string();

        let chat_template = ChatPromptTemplate::new(vec![
            Message::system(
                "你是一个 AI 助手，可以使用以下工具解决问题。\n\n可用工具：\n{tools}\n\n对话记忆：\n{memory}\n\n用户问题：{input}\n\n上一步结果：{scratchpad}"
            ),
            Message::human("请根据上述信息回答。"),
        ]);

        let mut values: HashMap<&str, &str> = HashMap::new();
        values.insert("tools", &tools_str);
        values.insert("memory", &memory_str);
        values.insert("input", &input_str);
        values.insert("scratchpad", &scratchpad_str);

        let response = self
            .llm
            .invoke_chat_template(&chat_template, &values)
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
