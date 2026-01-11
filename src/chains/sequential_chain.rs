use crate::llms::LLM;
use crate::memory::Memory;
use crate::messages::Message;
use crate::prompts::ChatPromptTemplate;
use async_trait::async_trait;
use std::collections::HashMap;

#[async_trait]
pub trait Chain {
    fn input_keys(&self) -> Vec<&str>;
    fn output_key(&self) -> &str;

    async fn call(
        &self,
        input: &HashMap<String, String>,
        verbose: bool,
    ) -> Result<HashMap<String, String>, Box<dyn std::error::Error>>;
}

pub struct SequentialChain {
    chains: Vec<Box<dyn Chain>>,
    input_variables: Vec<String>,
    output_variables: Vec<String>,
    verbose: bool,
    memory: Option<Box<dyn Memory>>,
}

impl SequentialChain {
    pub fn new(
        chains: Vec<Box<dyn Chain>>,
        input_variables: Vec<&str>,
        output_variables: Vec<&str>,
        verbose: bool,
        memory: Option<Box<dyn Memory>>,
    ) -> Self {
        Self {
            chains,
            input_variables: input_variables.into_iter().map(|s| s.to_string()).collect(),
            output_variables: output_variables
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            verbose,
            memory,
        }
    }

    pub async fn call(
        &mut self,
        initial_input: &HashMap<&str, &str>,
    ) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        // 校验
        for var in &self.input_variables {
            if !initial_input.contains_key(var.as_str()) {
                return Err(format!("Missing input: {}", var).into());
            }
        }

        let main_key = &self.input_variables[0];

        // 构建 context
        let mut context: HashMap<String, String> = initial_input
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        // 注入历史到主输入
        if let Some(mem) = &self.memory {
            let history = mem.context();
            if !history.trim().is_empty() {
                if let Some(current) = context.get(main_key) {
                    context.insert(
                        main_key.clone(),
                        format!(
                            "【对话历史】\n{}\n\n【当前请求】\n{}",
                            history.trim(),
                            current
                        ),
                    );
                }
            }
        }
        let mut user_question1 = String::new();
        let mut step = 1; // 👈 步骤计数器
        // 执行 chains
        for chain in &self.chains {
            let chain_input: HashMap<String, String> = chain
                .input_keys()
                .into_iter()
                .filter_map(|k| context.get(k).map(|v| (k.to_string(), v.clone())))
                .collect();


            for (k) in self.memory {

            }


            let result = chain.call(&chain_input, self.verbose).await?;
            for (k, v) in result {
                if k == "question" {
                    user_question1 = v;
                }else {
                    context.insert(k, v);
                }
            }

            if let Some(ref mut mem) = self.memory {
                mem.add(&step.to_string(), &user_question1);
            }
            step += 1;
        }



        // 输出
        let mut final_output = HashMap::new();
        for var in &self.output_variables {
            final_output.insert(var.clone(), context.get(var).unwrap().clone());
        }
        Ok(final_output)
    }
}

pub struct PromptChain {
    llm: LLM,
    template: ChatPromptTemplate,
    input_keys: Vec<String>,
    output_key: String,
}

impl PromptChain {
    pub fn new(
        llm: LLM,
        template: ChatPromptTemplate,
        input_keys: Vec<&str>,
        output_key: &str,
    ) -> Self {
        Self {
            llm,
            template,
            input_keys: input_keys.into_iter().map(|s| s.to_string()).collect(),
            output_key: output_key.to_string(),
        }
    }
}

#[async_trait]
impl Chain for PromptChain {
    fn input_keys(&self) -> Vec<&str> {
        self.input_keys.iter().map(|s| s.as_str()).collect()
    }

    fn output_key(&self) -> &str {
        &self.output_key
    }

    async fn call(
        &self,
        input: &HashMap<String, String>,
        verbose: bool,
    ) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        // ✅ 校验输入是否齐全
        for key in &self.input_keys {
            if !input.contains_key(key) {
                return Err(format!("PromptChain 缺少输入变量: {}", key).into());
            }
        }
        if verbose {
            println!("Executing PromptChain with input: {:?}", input);
        }

        // 转为 &str 引用供模板使用
        let input_refs: HashMap<&str, &str> = input
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let question = &self
            .template
            .format(&input_refs)
            .map_err(|e| format!("Template formatting failed: {}", e))?;

        let questionstr = messages_to_string(question);

        let output = self
            .llm
            .invoke_chat_template(&self.template, &input_refs)
            .await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), output);

        result.insert("question".parse()?, questionstr);
        Ok(result)
    }
}

pub fn messages_to_string(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|msg| match msg {
            Message::System(sys) => format!("System: {}", sys.content),
            Message::Human(hum) => format!("Human: {}", hum.content),
            Message::AIMessage(ai) => format!("AI: {}", ai.content),
        })
        .collect::<Vec<_>>()
        .join("\n")
}
