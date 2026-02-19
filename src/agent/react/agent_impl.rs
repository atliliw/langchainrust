use crate::agent::{Agent, AgentAction, AgentError};
use crate::messages::Message;
use crate::prompts::ChatPromptTemplate;
use async_trait::async_trait;
use std::collections::HashMap;

use super::parser::{parse_response, tool_descriptions};
use super::retrieval::retrieve_context;
use super::routing::choose_llm;

#[async_trait]
impl Agent for super::ReActAgent {
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
        let llm = choose_llm(
            &self.llm,
            &self.models,
            &self.routing_state,
            input,
            vars,
        )
        .await?;

        let tools_str = tool_descriptions(&self.tools);
        let input_str = input.to_string();
        let scratchpad_str = match intermediate_steps {
            Some(s) if !s.is_empty() => format!("\n上一步工具执行结果：{}\n", s),
            _ => String::new(),
        };

        // 如果有 retriever，先检索相关文档
        let retrieved_context = retrieve_context(&self.retriever, input, self.top_k).await;
        let has_retrieval = retrieved_context.is_some();

        let mut chat_template = if let Some(t) = &self.user_template {
            t.clone()
        } else {
            let system_msg = build_system_message(
                &tools_str,
                &scratchpad_str,
                has_retrieval,
                self.tools.is_empty(),
            );

            ChatPromptTemplate::new(vec![
                Message::system(&system_msg),
                Message::human("{scratchpad}用户问题：{input}"),
            ])
        };

        let mut merged: HashMap<String, String> = vars.clone();
        merged.insert("input".to_string(), input_str);
        merged.insert("scratchpad".to_string(), scratchpad_str);

        if let Some(context) = retrieved_context {
            merged.insert("context".to_string(), context);
        }

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

        parse_response(&response)
    }

    fn add_memory(&self, input: &str, output: &str) {
        if let Some(mem) = &self.memory {
            let mut m = mem.lock().unwrap();
            m.add(input, output);
        }
    }
}

/// 构建系统消息
fn build_system_message(
    tools_str: &str,
    scratchpad_str: &str,
    has_retrieval: bool,
    no_tools: bool,
) -> String {
    if has_retrieval {
        if no_tools {
            "你是一个 AI 助手。请根据提供的参考文档回答用户问题。\n\
            如果参考文档中没有相关信息，请说明。\n\n\
            参考文档：\n{context}"
                .to_string()
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

            format!(
                "你是一个 AI 助手。请根据提供的参考文档回答用户问题。\n\
                如果参考文档中没有相关信息，请说明。\n\n\
                参考文档：\n{{context}}\n\n{}",
                tool_hint
            )
        }
    } else if no_tools {
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
    }
}
