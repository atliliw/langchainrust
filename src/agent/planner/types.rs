use serde::Deserialize;

/// 子任务定义
#[derive(Debug, Clone, Deserialize)]
pub struct SubTask {
    /// 任务序号
    pub id: usize,
    /// 任务描述
    pub description: String,
    /// 是否依赖前一个任务的结果
    pub depends_on_previous: bool,
}

/// 任务执行结果
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// 任务序号
    pub id: usize,
    /// 任务描述
    pub description: String,
    /// 执行结果
    pub result: String,
    /// 是否成功
    pub success: bool,
}

/// 规划结果
#[derive(Debug, Clone)]
pub struct Plan {
    /// 原始问题
    pub original_question: String,
    /// 子任务列表
    pub sub_tasks: Vec<SubTask>,
}
