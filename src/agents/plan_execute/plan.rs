//! Plan / PlanStep 类型

use serde::{Deserialize, Serialize};

/// 步骤状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum StepStatus {
    #[default]
    Pending,
    Running,
    Completed,
    Failed { error: String },
}

/// 执行计划步骤
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: usize,
    pub description: String,
    #[serde(default)]
    pub status: StepStatus,
    #[serde(default)]
    pub result: Option<String>,
}

impl PlanStep {
    pub fn new(id: usize, description: impl Into<String>) -> Self {
        Self {
            id,
            description: description.into(),
            status: StepStatus::Pending,
            result: None,
        }
    }
}

/// 执行计划
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub objective: String,
    pub steps: Vec<PlanStep>,
}

impl Plan {
    pub fn new(objective: impl Into<String>, steps: Vec<PlanStep>) -> Self {
        Self {
            objective: objective.into(),
            steps,
        }
    }

    /// 从步骤描述列表构造(id 从 0 递增)
    pub fn from_descriptions(objective: impl Into<String>, descs: Vec<String>) -> Self {
        let steps = descs
            .into_iter()
            .enumerate()
            .map(|(i, d)| PlanStep::new(i, d))
            .collect();
        Self::new(objective, steps)
    }

    /// 下一个 pending 步骤
    pub fn next_pending(&self) -> Option<&PlanStep> {
        self.steps.iter().find(|s| s.status == StepStatus::Pending)
    }

    /// 是否全部完成
    pub fn is_complete(&self) -> bool {
        self.steps
            .iter()
            .all(|s| s.status == StepStatus::Completed)
    }

    pub fn mark_completed(&mut self, id: usize, result: String) {
        if let Some(s) = self.steps.iter_mut().find(|s| s.id == id) {
            s.status = StepStatus::Completed;
            s.result = Some(result);
        }
    }

    pub fn mark_failed(&mut self, id: usize, error: String) {
        if let Some(s) = self.steps.iter_mut().find(|s| s.id == id) {
            s.status = StepStatus::Failed { error };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_descriptions() {
        let plan = Plan::from_descriptions("目标", vec!["步骤1".to_string(), "步骤2".to_string()]);
        assert_eq!(plan.objective, "目标");
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].id, 0);
        assert_eq!(plan.steps[1].id, 1);
        assert_eq!(plan.steps[0].status, StepStatus::Pending);
    }

    #[test]
    fn test_next_pending() {
        let mut plan = Plan::from_descriptions("obj", vec!["a".to_string(), "b".to_string()]);
        assert_eq!(plan.next_pending().unwrap().id, 0);
        plan.mark_completed(0, "result".to_string());
        assert_eq!(plan.next_pending().unwrap().id, 1);
        plan.mark_completed(1, "result".to_string());
        assert!(plan.next_pending().is_none());
    }

    #[test]
    fn test_is_complete() {
        let mut plan = Plan::from_descriptions("obj", vec!["a".to_string(), "b".to_string()]);
        assert!(!plan.is_complete());
        plan.mark_completed(0, "r".to_string());
        assert!(!plan.is_complete());
        plan.mark_completed(1, "r".to_string());
        assert!(plan.is_complete());
    }

    #[test]
    fn test_mark_failed() {
        let mut plan = Plan::from_descriptions("obj", vec!["a".to_string()]);
        plan.mark_failed(0, "出错".to_string());
        assert_eq!(plan.steps[0].status, StepStatus::Failed { error: "出错".to_string() });
        assert!(!plan.is_complete());
    }

    #[test]
    fn test_empty_plan_complete() {
        let plan = Plan::from_descriptions("obj", vec![]);
        assert!(plan.is_complete());
    }
}
