use std::collections::HashMap;
use async_trait::async_trait;
use crate::llms::LLM;
use crate::prompts::ChatPromptTemplate;

#[async_trait]
pub trait Chain {
    /// 返回该链所需的输入变量名（用于校验）
    fn input_keys(&self) -> Vec<&str>;
    
    /// 返回该链的输出变量名
    fn output_key(&self) -> &str;

    /// 执行链：输入是上下文，输出是 {output_key: result}
    async fn call(&self, input: &HashMap<String, String>) -> Result<HashMap<String, String>, Box<dyn std::error::Error>>;
}



pub struct SequentialChain {
    chains: Vec<Box<dyn Chain>>,
    input_variables: Vec<String>,      // 整个链的初始输入
    output_variables: Vec<String>,     // 最终要返回的变量
}

impl SequentialChain {
    pub fn new(
        chains: Vec<Box<dyn Chain>>,
        input_variables: Vec<&str>,
        output_variables: Vec<&str>,
    ) -> Self {
        Self {
            chains,
            input_variables: input_variables.into_iter().map(|s| s.to_string()).collect(),
            output_variables: output_variables.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    pub async fn call(&self, initial_input: &HashMap<&str, &str>) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        // 1. 校验初始输入
        for var in &self.input_variables {
            if !initial_input.contains_key(var.as_str()) {
                return Err(format!("SequentialChain 缺少初始输入: {}", var).into());
            }
        }

        // 2. 初始化上下文
        let mut context: HashMap<String, String> = initial_input
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        // 3. 依次执行每个子链
        for (i, chain) in self.chains.iter().enumerate() {
            // 子链所需输入是否都在 context 中？
            for key in chain.input_keys() {
                if !context.contains_key(key) {
                    return Err(format!(
                        "第 {} 个链 ({}) 缺少输入变量: {}",
                        i, chain.output_key(), key
                    ).into());
                }
            }

            let result = chain.call(&context).await?;
            // 合并输出到上下文（key 不会冲突，因为每个链 output_key 唯一）
            for (k, v) in result {
                context.insert(k, v);
            }
        }

        // 4. 提取最终输出
        let mut final_output = HashMap::new();
        for var in &self.output_variables {
            if let Some(value) = context.get(var) {
                final_output.insert(var.clone(), value.clone());
            } else {
                return Err(format!("最终输出变量未生成: {}", var).into());
            }
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
    /// 创建一个 PromptChain
    /// - `input_keys`: 模板中用到的所有变量名（用于校验）
    /// - `output_key`: 本链输出的变量名
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

    async fn call(&self, input: &HashMap<String, String>) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        // ✅ 校验输入是否齐全
        for key in &self.input_keys {
            if !input.contains_key(key) {
                return Err(format!("PromptChain 缺少输入变量: {}", key).into());
            }
        }

        // 转为 &str 引用供模板使用
        let input_refs: HashMap<&str, &str> = input
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let output = self.llm.invoke_chat_template(&self.template, &input_refs).await?;

        let mut result = HashMap::new();
        result.insert(self.output_key.clone(), output);
        Ok(result)
    }
}