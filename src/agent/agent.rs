use crate::llms::LLM;
use crate::prompts::PromptTemplate;
use crate::messages::Message;
use crate::tools::{Tool, ToolInput};
use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
pub const MAX_ITERATIONS: usize = 10;

#[async_trait]
pub trait Agent: Send + Sync {
    async fn run(&self, input: &str) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct ReActAgent {
    llm: LLM,
    tools: Vec<Arc<dyn Tool>>,
    max_iterations: usize,
}

impl ReActAgent {
    pub fn new(
        llm: LLM,
        tools: Vec<Arc<dyn Tool>>,
    ) -> Self {
        Self {
            llm,
            tools,
            max_iterations: MAX_ITERATIONS,
        }
    }

    pub fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations;
    }

    fn get_system_prompt(&self) -> String {
        let tool_descriptions: Vec<String> = self.tools
            .iter()
            .map(|tool: &Arc<dyn Tool>| {
                let params = tool.parameters();
                let param_desc = if params.is_empty() {
                    "无参数".to_string()
                } else {
                    params.iter()
                        .map(|(name, desc)| format!("{}: {}", name, desc))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                format!("{} - {} (参数: {})", tool.name(), tool.description(), param_desc)
            })
            .collect();

        format!(
            "你是一个AI助手，可以使用工具来帮助用户解决问题。你的响应应该按照以下格式：
            
当您需要使用工具时，请按照以下格式回复：
```
思考：...思考过程...
行为：工具名 参数1=值1 参数2=值2
```

当您不需要使用工具时，直接给出最终答案。

可用工具：
{}

用户问题：{{input}}
使用工具时，请严格按照指定的行为格式。",
            tool_descriptions.join("\n"),
        )
    }

    fn parse_action(&self, response: &str) -> Option<(String, HashMap<String, String>)> {
        for line in response.lines() {
            if line.starts_with("行为：") {
                let rest = line.trim_start_matches("行为：").trim();

                // 分割工具名和参数部分（按第一个空格）
                let parts: Vec<&str> = rest.splitn(2, ' ').collect();
                if parts.is_empty() {
                    return None;
                }

                let tool_name = parts[0].to_string();
                let mut params = HashMap::new();

                if parts.len() == 2 {
                    for pair in parts[1].split_whitespace() {
                        if let Some((key, value)) = pair.split_once('=') {
                            params.insert(key.to_string(), value.to_string());
                        }
                    }
                }

                return Some((tool_name, params));
            }
        }
        None
    }

    async fn execute_tool(
        &self, 
        tool_name: &str, 
        params: HashMap<String, String>
    ) -> Result<String, Box<dyn std::error::Error>> {
        let tool = self.tools.iter()
            .find(|t: &&Arc<dyn Tool>| t.name() == tool_name)
            .ok_or_else(|| format!("未找到工具: {}", tool_name))?;

        let input = ToolInput {
            tool_name: tool_name.to_string(),
            parameters: params,
        };

        let output = tool.invoke(input).await?;
        
        if output.success {
            Ok(output.result)
        } else {
            Err(format!("工具执行失败: {}", output.result).into())
        }
    }
}

#[async_trait]
impl Agent for ReActAgent {
    async fn run(&self, input: &str) -> Result<String, Box<dyn std::error::Error>> {
        let mut iteration = 0;
        let mut final_answer = None;

        while iteration < self.max_iterations {
            iteration += 1;
            let system_prompt = self.get_system_prompt();
            let system_prompt = format!("{}", system_prompt);
            let system_template = PromptTemplate::new(&system_prompt);
            let system_prompt = system_template.format(&HashMap::from([("input", input)]))?;

            
            // 简化的消息系统，不支持记忆，只处理当前轮次
            let chat_template = crate::prompts::ChatPromptTemplate::new(vec![
                Message::system(&system_prompt),
                Message::human("请使用最佳可用工具来解决问题。如果你没有看到相关的工具，请直接回答。"),
            ]);
            
            let mut context: HashMap<String, String> = HashMap::new();
            
            let all_values: HashMap<&str, &str> = context
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            
            let response = self.llm.invoke_chat_template(&chat_template, &all_values).await?;
            
            if let Some((tool_name, params)) = self.parse_action(&response) {
                println!("工具调用: {}({:?})", tool_name, params);
                
                match self.execute_tool(&tool_name, params).await {
                    Ok(result) => {
                        println!("工具结果: {}", result);
                        context = HashMap::from([("tool_result".to_string(), result)]);
                        continue;
                    }
                    Err(e) => {
                        println!("工具执行出错: {}", e);
                        context = HashMap::from([("tool_error".to_string(), e.to_string())]);
                        continue;
                    }
                }
            } else {
                println!("最终答案: {}", response);
                final_answer = Some(response);
                break;
            }
        }

        if let Some(answer) = final_answer {
            Ok(answer)
        } else {
            Err("未找到解决方案".into())
        }
    }
}