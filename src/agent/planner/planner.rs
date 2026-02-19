use crate::llms::LLM;
use crate::messages::Message;
use crate::prompts::ChatPromptTemplate;
use std::collections::HashMap;

use super::types::{Plan, SubTask, TaskResult};

/// 任务规划器 - 将复杂任务分解为子任务
pub struct TaskPlanner {
    llm: LLM,
    max_sub_tasks: usize,
    verbose: bool,
}

impl TaskPlanner {
    pub fn new(llm: LLM) -> Self {
        Self {
            llm,
            max_sub_tasks: 5,
            verbose: false,
        }
    }

    /// 设置最大子任务数量
    pub fn with_max_sub_tasks(mut self, max: usize) -> Self {
        self.max_sub_tasks = max;
        self
    }

    /// 设置是否打印详细日志
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 打印日志（仅在 verbose 模式下）
    fn log(&self, msg: &str) {
        if self.verbose {
            println!("{}", msg);
        }
    }

    /// 将复杂问题分解为子任务
    pub async fn plan(&self, question: &str) -> Result<Plan, Box<dyn std::error::Error>> {
        self.log("[TaskPlanner] 开始规划任务...");
        
        let system_prompt = format!(
            "你是一个任务规划专家。请将用户的复杂问题分解为 {} 个以内的子任务。\n\
            每个子任务应该是独立可执行的步骤。\n\
            如果问题很简单，不需要分解，直接返回一个子任务。\n\n\
            输出格式要求（严格按JSON数组格式输出，不要输出其他内容）：\n\
            输出一个JSON数组，每个元素包含三个字段：\n\
            - id: 任务序号(数字)\n\
            - description: 任务描述(字符串)\n\
            - depends_on_previous: 是否依赖前一个任务的结果(true/false)\n\n\
            示例输出格式：\n\
            一个包含id、description、depends_on_previous三个字段的JSON数组",
            self.max_sub_tasks
        );

        let template = ChatPromptTemplate::new(vec![
            Message::system(&system_prompt),
            Message::human("请分解以下任务：{question}"),
        ]);

        let values = HashMap::from([("question", question)]);
        let response = self.llm.invoke_chat_template(&template, &values).await?;

        let sub_tasks = self.parse_sub_tasks(&response)?;
        
        self.log(&format!("[TaskPlanner] 规划完成，共 {} 个子任务", sub_tasks.len()));

        Ok(Plan {
            original_question: question.to_string(),
            sub_tasks,
        })
    }

    /// 解析子任务列表
    fn parse_sub_tasks(&self, response: &str) -> Result<Vec<SubTask>, Box<dyn std::error::Error>> {
        // 尝试提取 JSON 数组
        let response = response.trim();
        
        // 查找 JSON 数组的起始和结束位置
        let start = response.find('[').ok_or("未找到 JSON 数组起始")?;
        let end = response.rfind(']').ok_or("未找到 JSON 数组结束")?;
        let json_str = &response[start..=end];

        // 解析 JSON
        let mut tasks: Vec<SubTask> = serde_json::from_str(json_str)
            .map_err(|e| format!("JSON 解析失败: {}", e))?;

        // 确保任务 ID 正确
        for (i, task) in tasks.iter_mut().enumerate() {
            task.id = i + 1;
        }

        // 限制任务数量
        if tasks.len() > self.max_sub_tasks {
            tasks.truncate(self.max_sub_tasks);
        }

        // 如果没有任务，创建一个默认任务
        if tasks.is_empty() {
            tasks.push(SubTask {
                id: 1,
                description: "直接回答用户问题".to_string(),
                depends_on_previous: false,
            });
        }

        Ok(tasks)
    }

    /// 汇总多个子任务的结果
    pub async fn summarize(
        &self,
        original_question: &str,
        results: &[TaskResult],
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.log("[TaskPlanner] 开始汇总结果...");
        
        if results.is_empty() {
            return Ok("没有可汇总的结果".to_string());
        }

        // 如果只有一个结果，直接返回
        if results.len() == 1 {
            self.log("[TaskPlanner] 只有一个结果，直接返回");
            return Ok(results[0].result.clone());
        }

        // 构建结果摘要
        let results_text = results
            .iter()
            .map(|r| {
                format!(
                    "【任务{}】{}\n结果：{}",
                    r.id, r.description, r.result
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let system_prompt = "你是一个结果汇总专家。请根据以下子任务的执行结果，\
            生成一个完整、连贯的最终答案。\n\
            要求：\n\
            1. 整合所有关键信息\n\
            2. 保持逻辑清晰\n\
            3. 不要遗漏重要内容\n\
            4. 用自然的语言表达，不要只是罗列";

        let template = ChatPromptTemplate::new(vec![
            Message::system(system_prompt),
            Message::human("原始问题：{question}\n\n子任务执行结果：\n{results}"),
        ]);

        let values = HashMap::from([
            ("question", original_question),
            ("results", &results_text),
        ]);

        let result = self.llm.invoke_chat_template(&template, &values).await;
        
        self.log("[TaskPlanner] 汇总完成");
        result
    }
}
