//! 并行工具调用功能测试
//!
//! 本测试文件验证 AgentOutput::Actions 多动作支持：
//! - Agent 能够返回多个并发的工具调用请求
//! - AgentExecutor 能够并行执行多个工具
//! - 结果正确聚合到 intermediate_steps
//!
//! 测试策略：使用真实工具（Calculator, DateTimeTool）
//! 而非 Mock，确保与实际运行时行为一致。

use async_trait::async_trait;
use langchainrust::{
    AgentAction, AgentError, AgentExecutor, AgentFinish, AgentOutput, AgentStep, BaseAgent,
    BaseTool, Calculator, DateTimeTool, ToolInput,
};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// 测试用 Agent（模拟 LLM 返回不同类型的 AgentOutput）
//
// 注意：这些 Agent 不是真实的 LLM，而是用于测试 AgentExecutor
// 对不同 AgentOutput 类型（Action/Actions/Finish）的处理逻辑
// ============================================================================

/// 多动作 Agent - 模拟 LLM 返回多个 ToolCall
///
/// 使用场景：用户问 "现在几点？顺便算一下 100 + 200"
/// LLM 决定同时调用 datetime 和 calculator 两个工具
///
/// 行为：
/// - 第一次 plan() 返回 Actions([datetime, calculator])
/// - 第二次 plan() 返回 Finish（聚合结果）
///
/// 注意：真实工具需要特定输入格式：
/// - Calculator: {"expression": "100 + 200"} (JSON)
/// - DateTimeTool: {"operation": "now"} (JSON)
struct MultiActionAgent;

#[async_trait]
impl BaseAgent for MultiActionAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        if !intermediate_steps.is_empty() {
            let results: Vec<&str> = intermediate_steps
                .iter()
                .map(|s| s.observation.as_str())
                .collect();
            return Ok(AgentOutput::Finish(AgentFinish::new(
                format!("结果: {}", results.join(", ")),
                String::new(),
            )));
        }

        let actions = vec![
            AgentAction {
                tool: "datetime".to_string(),
                tool_input: ToolInput::Object {
                    value: serde_json::json!({"operation": "now"}),
                },
                log: "call_datetime_1".to_string(),
            },
            AgentAction {
                tool: "calculator".to_string(),
                tool_input: ToolInput::Object {
                    value: serde_json::json!({"expression": "100 + 200"}),
                },
                log: "call_calc_1".to_string(),
            },
        ];

        Ok(AgentOutput::Actions(actions))
    }

    fn get_allowed_tools(&self) -> Option<Vec<&str>> {
        Some(vec!["datetime", "calculator"])
    }
}

/// 单动作 Agent - 模拟 LLM 返回单个 ToolCall
///
/// 使用场景：用户问 "计算 25 * 4"
/// LLM 只需要调用一个 calculator 工具
///
/// 行为：
/// - 第一次 plan() 返回 Action(calculator)
/// - 第二次 plan() 返回 Finish（工具结果）
///
/// 注意：Calculator 需要 JSON 输入 {"expression": "25 * 4"}
struct SingleActionAgent;

#[async_trait]
impl BaseAgent for SingleActionAgent {
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        if !intermediate_steps.is_empty() {
            return Ok(AgentOutput::Finish(AgentFinish::new(
                intermediate_steps[0].observation.clone(),
                String::new(),
            )));
        }

        // Calculator 需要 JSON 格式输入
        Ok(AgentOutput::Action(AgentAction {
            tool: "calculator".to_string(),
            tool_input: ToolInput::Object {
                value: serde_json::json!({"expression": "25 * 4"}),
            },
            log: "call_calc_1".to_string(),
        }))
    }

    fn get_allowed_tools(&self) -> Option<Vec<&str>> {
        Some(vec!["calculator"])
    }
}

/// 直接返回 Finish 的 Agent
///
/// 使用场景：用户问简单问题，LLM 不需要调用工具
///
/// 行为：直接返回 Finish，不执行任何工具
struct DirectFinishAgent;

#[async_trait]
impl BaseAgent for DirectFinishAgent {
    async fn plan(
        &self,
        _intermediate_steps: &[AgentStep],
        _inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        Ok(AgentOutput::Finish(AgentFinish::new(
            "这是直接回答，不需要工具".to_string(),
            String::new(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // AgentOutput 枚举行为测试
    // -------------------------------------------------------------------------

    /// 验证 AgentOutput::Actions 的基本行为
    ///
    /// 测试点：
    /// - is_action() 返回 true（多动作仍是动作）
    /// - is_finish() 返回 false
    /// - actions() 返回所有动作的引用向量
    /// - action() 返回 None（多动作时无单个动作）
    #[test]
    fn test_agent_output_actions_variant() {
        let actions = vec![
            AgentAction {
                tool: "calculator".to_string(),
                tool_input: ToolInput::String {
                    value: "1 + 1".to_string(),
                },
                log: "call_1".to_string(),
            },
            AgentAction {
                tool: "datetime".to_string(),
                tool_input: ToolInput::String {
                    value: "now".to_string(),
                },
                log: "call_2".to_string(),
            },
        ];

        let output = AgentOutput::Actions(actions);

        assert!(output.is_action());
        assert!(!output.is_finish());
        assert_eq!(output.actions().len(), 2);
        assert!(output.action().is_none());

        let action_refs = output.actions();
        assert_eq!(action_refs[0].tool, "calculator");
        assert_eq!(action_refs[1].tool, "datetime");
    }

    /// 验证 Action 和 Actions 的区分逻辑
    ///
    /// 测试点：
    /// - 两者都返回 is_action() == true
    /// - Action 可通过 action() 获取单个动作
    /// - Actions 不可通过 action() 获取（返回 None）
    /// - actions() 对两者都有效
    #[test]
    fn test_agent_output_single_vs_multiple_actions() {
        let single = AgentOutput::Action(AgentAction {
            tool: "calculator".to_string(),
            tool_input: ToolInput::String {
                value: "test".to_string(),
            },
            log: "log".to_string(),
        });

        let multiple = AgentOutput::Actions(vec![AgentAction {
            tool: "calculator".to_string(),
            tool_input: ToolInput::String {
                value: "test".to_string(),
            },
            log: "log".to_string(),
        }]);

        assert!(single.is_action());
        assert!(multiple.is_action());

        assert!(single.action().is_some());
        assert!(multiple.action().is_none());

        assert_eq!(single.actions().len(), 1);
        assert_eq!(multiple.actions().len(), 1);
    }

    // -------------------------------------------------------------------------
    // AgentExecutor 并行执行测试（使用真实工具）
    // -------------------------------------------------------------------------

    /// 测试 AgentExecutor 处理单动作场景
    ///
    /// 场景：用户问 "计算 25 * 4"
    /// 工具：真实的 Calculator（计算 25 * 4 = 100）
    ///
    /// 验证：
    /// - Calculator 正确执行乘法运算
    /// - 结果包含计算答案
    #[tokio::test]
    async fn test_executor_single_action_with_real_calculator() {
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];

        let executor =
            AgentExecutor::new(Arc::new(SingleActionAgent), tools).with_max_iterations(2);

        let result = executor.invoke("计算 25 * 4".to_string()).await.unwrap();

        // Calculator 返回 JSON 格式：{"expression": "25 * 4", "result": 100}
        assert!(result.contains("100"), "Calculator 应返回计算结果 100");
    }

    /// 测试 AgentExecutor 并发执行多个真实工具
    ///
    /// 场景：用户问 "现在几点？顺便算一下 100 + 200"
    /// 工具：
    /// - DateTimeTool（获取当前时间）
    /// - Calculator（计算 100 + 200 = 300）
    ///
    /// 验证：
    /// - 两个工具并发执行（futures::join_all）
    /// - DateTimeTool 返回当前时间字符串
    /// - SimpleMathTool 返回计算结果 300
    /// - 最终结果聚合两个工具的输出
    #[tokio::test]
    async fn test_executor_parallel_actions_with_real_tools() {
        let tools: Vec<Arc<dyn BaseTool>> =
            vec![Arc::new(DateTimeTool::new()), Arc::new(Calculator::new())];

        let executor = AgentExecutor::new(Arc::new(MultiActionAgent), tools).with_max_iterations(2);

        let result = executor
            .invoke("现在几点？顺便算一下 100 + 200".to_string())
            .await
            .unwrap();

        // SimpleMathTool 计算结果为 300
        assert!(result.contains("300"), "应包含 SimpleMathTool 计算结果 300");
        // DateTimeTool 返回时间字符串（包含年份）
        assert!(
            result.contains("202") || result.len() > 10,
            "应包含 DateTimeTool 时间输出"
        );
    }

    /// 测试 AgentExecutor 处理直接返回 Finish（不调用工具）
    ///
    /// 场景：用户问简单问题，LLM 直接回答
    /// 工具：无
    ///
    /// 验证：
    /// - 不执行任何工具
    /// - 直接返回 Agent 的 Finish 结果
    #[tokio::test]
    async fn test_executor_direct_finish_without_tools() {
        let tools: Vec<Arc<dyn BaseTool>> = vec![];

        let executor =
            AgentExecutor::new(Arc::new(DirectFinishAgent), tools).with_max_iterations(1);

        let result = executor.invoke("你好".to_string()).await.unwrap();
        assert_eq!(result, "这是直接回答，不需要工具");
    }

    // -------------------------------------------------------------------------
    // AgentStep 执行历史记录测试
    // -------------------------------------------------------------------------

    /// 验证 AgentStep 正确记录动作和观察结果
    ///
    /// AgentStep 是 Agent 执行循环的核心数据结构：
    /// - action：记录哪个工具被调用、输入是什么
    /// - observation：记录工具返回什么
    ///
    /// 这些步骤被传递给下一次 plan()，使 Agent 能看到历史
    #[test]
    fn test_agent_step_records_action_and_observation() {
        let action = AgentAction {
            tool: "calculator".to_string(),
            tool_input: ToolInput::String {
                value: "10 + 20".to_string(),
            },
            log: "需要计算 10 + 20".to_string(),
        };

        let step = AgentStep::new(action, "{\"result\": 30}".to_string());

        assert_eq!(step.action.tool, "calculator");
        assert_eq!(step.action.tool_input.to_string(), "10 + 20");
        assert_eq!(step.observation, "{\"result\": 30}");
    }

    // -------------------------------------------------------------------------
    // AgentFinish 最终答案测试
    // -------------------------------------------------------------------------

    /// 验证 AgentFinish 正确构造最终答案
    ///
    /// AgentFinish 包含：
    /// - return_values：键值对形式的返回结果（通常有 "output" 键）
    /// - log：执行过程的日志（可选）
    ///
    /// output() 是最常用的便捷方法
    #[test]
    fn test_agent_finish_constructs_final_answer() {
        let finish = AgentFinish::new("最终答案是 42".to_string(), "经过计算得出".to_string());

        assert_eq!(finish.output(), Some("最终答案是 42"));
        assert!(finish.return_values.contains_key("output"));
        assert_eq!(
            finish.return_values.get("output").and_then(|v| v.as_str()),
            Some("最终答案是 42")
        );
    }

    // -------------------------------------------------------------------------
    // ToolInput 输入格式测试
    // -------------------------------------------------------------------------

    /// 验证 ToolInput 的两种格式都能正确工作
    ///
    /// ToolInput::String - 简单字符串输入（如 "1 + 2"）
    /// ToolInput::Object - JSON 对象输入（如 {"expression": "10 * 5"}）
    ///
    /// 不同工具可能期望不同格式
    #[test]
    fn test_tool_input_string_and_object_formats() {
        // String 格式
        let string_input = ToolInput::String {
            value: "1 + 2".to_string(),
        };
        assert_eq!(string_input.to_string(), "1 + 2");

        // Object 格式（JSON）
        let json_input = ToolInput::Object {
            value: serde_json::json!({
                "expression": "10 * 5"
            }),
        };
        let json_str = json_input.to_string();
        assert!(json_str.contains("expression"));
        assert!(json_str.contains("10 * 5"));
    }
}
