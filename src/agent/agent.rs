use crate::agent::{Agent, AgentAction, AgentError,ReActAgent};
use crate::llms::LLM;
use crate::messages::Message;
use crate::prompts::{ChatPromptTemplate, PromptTemplate};
use crate::tools::Tool;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;



impl ReActAgent {
    pub fn new(llm: LLM, tools: Vec<Arc<dyn Tool>>) -> Self {
        Self { llm, tools }
    }

    fn build_prompt(&self, input: &str, intermediate_steps: Option<&str>) -> String {
        let tool_descs: Vec<String> = self
            .tools
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
            .collect();

        let mut prompt = format!(
            "你是一个 AI 助手，可以使用以下工具解决问题。

响应格式要求：
- 如果需要使用工具，请严格按以下格式输出：
思考：你的推理过程
行为：工具名 参数1=值1 参数2=值2
- 如果无需工具，请直接给出最终答案。

可用工具：
{}
",
            tool_descs.join("\n")
        );

        prompt.push_str(&format!("\n用户问题：{}", input));

        if let Some(obs) = intermediate_steps {
            prompt.push_str(&format!("\n\n上一步结果：{}", obs));
        }

        prompt
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
        let prompt = self.build_prompt(input, intermediate_steps);
        let chat_template = ChatPromptTemplate::new(vec![
            Message::system(&prompt),
            Message::human("请根据上述信息回答。"),
        ]);

        let response = self
            .llm
            .invoke_chat_template(&chat_template, &HashMap::new())
            .await
            .map_err(|e| AgentError(format!("LLM 调用失败: {}", e)))?;

        self.parse_response(&response)
    }
}
