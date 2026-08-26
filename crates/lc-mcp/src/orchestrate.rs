//! 工具编排(P2-10):把一次任务的多个工具调用编排成依赖图,并行 + 串行执行。
//!
//! 复杂任务往往是一串互相依赖的工具调用——前一个工具的输出是后一个的输入。
//! [`ToolOrchestrator`] 把步骤声明成 DAG(`depends_on` 表达依赖),执行时:
//!
//! 1. Kahn 拓扑排序 + 环检测,环 / 未定义依赖在**执行前**报错;
//! 2. 按轮推进:每轮取出"依赖已全部满足"的步骤并发执行,用
//!    [`tokio::sync::Semaphore`] 封顶并发度(默认 4);
//! 3. 步骤参数支持 `${id}` / `${id.field}` 模板:执行前把前序步骤的输出
//!    代入参数(`${id}` 引用整个输出,`${id.field}` 提取对象字段)。
//!
//! [`ToolCaller`] trait 抽象"按名调用工具"的动作,[`MCPGateway`]
//! 已实现——把多工具编排直接挂在 Gateway 的 `server:tool` 命名空间上。

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use futures_util::future::join_all;
use serde_json::Value;
use tokio::sync::Semaphore;

use crate::gateway::MCPGateway;
use lc_core::tools::ToolError;

/// "按全名调用一个工具"的抽象(P2-10)。
///
/// [`MCPGateway`] 已实现(把 `server:tool` 字符串结果解析
/// 成 JSON,解析失败则退化为 `Value::String`);自定义 caller 可实现此 trait
/// 接入任意工具注册表。每次调用返回结构化输出,供 `${id.field}` 模板引用。
#[async_trait::async_trait]
pub trait ToolCaller: Send + Sync {
    /// 调用工具并返回其结构化输出。
    async fn call(&self, tool: &str, args: Value) -> Result<Value, ToolError>;
}

/// 编排的单个步骤(P2-10)。
///
/// `id` 是本步骤在编排内的唯一标识(供其他步骤 `depends_on` 与 `${id}` 引用);
/// `tool` 是工具全名(`server:tool`);`args` 可含 `${...}` 模板,执行时以
/// 前序步骤的输出代入。
#[derive(Debug, Clone)]
pub struct ToolStep {
    /// 步骤唯一标识。
    pub id: String,
    /// 工具全名(如 `fs:read_file`)。
    pub tool: String,
    /// 调用参数(支持 `${id}` / `${id.field}` 模板)。
    pub args: Value,
    /// 本步骤依赖的步骤 id;全部完成才执行。
    pub depends_on: Vec<String>,
}

impl ToolStep {
    /// 创建一个步骤,`depends_on` 默认空。
    pub fn new(id: impl Into<String>, tool: impl Into<String>, args: Value) -> Self {
        Self {
            id: id.into(),
            tool: tool.into(),
            args,
            depends_on: Vec::new(),
        }
    }

    /// 声明一个前置依赖(可链式多次调用)。
    pub fn after(mut self, dep: impl Into<String>) -> Self {
        self.depends_on.push(dep.into());
        self
    }
}

/// 工具编排的错误类别(P2-10)。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OrchestrateError {
    /// `depends_on` 引用了未定义的步骤 id。
    UnknownStep(String),
    /// 步骤 id 重复。
    DuplicateId(String),
    /// 依赖图存在环,无法拓扑排序。
    Cycle,
    /// 编排未收敛(依赖互等的死局;环检测通过后不应发生)。
    Deadlock,
    /// 参数模板引用了未执行的步骤。
    MissingStep(String),
    /// 参数模板 `id.field` 的字段不存在。
    MissingField(String, String),
    /// 底层工具调用失败。
    Tool(String),
    /// 并发闸门已关闭。
    SemaphoreClosed,
}

impl fmt::Display for OrchestrateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrchestrateError::UnknownStep(id) => {
                write!(f, "orchestration references an undefined step: {id}")
            }
            OrchestrateError::DuplicateId(id) => {
                write!(f, "orchestration step id is duplicated: {id}")
            }
            OrchestrateError::Cycle => write!(f, "orchestration has a circular dependency"),
            OrchestrateError::Deadlock => write!(
                f,
                "orchestration did not converge: steps waiting on each other"
            ),
            OrchestrateError::MissingStep(id) => {
                write!(f, "parameter template references an unexecuted step: {id}")
            }
            OrchestrateError::MissingField(id, field) => {
                write!(
                    f,
                    "parameter template field missing: step {id} has no field {field}"
                )
            }
            OrchestrateError::Tool(msg) => {
                write!(f, "tool call failed inside orchestration: {msg}")
            }
            OrchestrateError::SemaphoreClosed => {
                write!(f, "orchestration concurrency gate is closed")
            }
        }
    }
}

impl std::error::Error for OrchestrateError {}

impl From<ToolError> for OrchestrateError {
    fn from(e: ToolError) -> Self {
        OrchestrateError::Tool(e.to_string())
    }
}

/// 工具编排器(P2-10)。
///
/// 持有步骤 DAG 与并发上限,`execute` 以轮次推进:每轮并发执行所有
/// "依赖已满足"的步骤(并发度 ≤ `max_concurrency`),结果存入 map 供下一轮
/// `${id}` 引用。执行前先做环检测,环 / 未定义依赖直接拒绝执行。
#[derive(Debug)]
pub struct ToolOrchestrator {
    steps: Vec<ToolStep>,
    max_concurrency: usize,
}

impl Default for ToolOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolOrchestrator {
    /// 空编排器,默认并发上限 4。
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            max_concurrency: 4,
        }
    }

    /// 设置并发上限(同时执行的最大步骤数)。
    pub fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self
    }

    /// 添加一个步骤(链式)。
    pub fn add_step(mut self, step: ToolStep) -> Self {
        self.steps.push(step);
        self
    }

    /// 执行整个 DAG,返回 `步骤 id → 输出` 的映射。
    ///
    /// - 前置校验:步骤 id 唯一、依赖全部有定义、无环;
    /// - 每轮取"依赖已满足"的步骤并发执行(信号量封顶);
    /// - 步骤参数中的 `${id}` / `${id.field}` 模板以前序结果代入。
    pub async fn execute<C: ToolCaller>(
        &self,
        caller: &C,
    ) -> Result<HashMap<String, Value>, OrchestrateError> {
        self.validate()?;
        let semaphore = Semaphore::new(self.max_concurrency);
        let sem = &semaphore;
        let mut results: HashMap<String, Value> = HashMap::new();
        let mut pending: HashSet<&str> = self.steps.iter().map(|s| s.id.as_str()).collect();

        while !pending.is_empty() {
            let ready: Vec<&ToolStep> = self
                .steps
                .iter()
                .filter(|s| {
                    pending.contains(s.id.as_str())
                        && s.depends_on
                            .iter()
                            .all(|d| results.contains_key(d.as_str()))
                })
                .collect();
            if ready.is_empty() {
                // 拓扑检测通过后不应发生;防御性报错而非死循环。
                return Err(OrchestrateError::Deadlock);
            }

            let mut futures = Vec::with_capacity(ready.len());
            for step in ready {
                let args = resolve_template(&step.args, &results)?;
                let tool = step.tool.clone();
                let id = step.id.clone();
                futures.push(async move {
                    let _guard = sem
                        .acquire()
                        .await
                        .map_err(|_| OrchestrateError::SemaphoreClosed)?;
                    let out = caller.call(&tool, args).await?;
                    Ok::<(String, Value), OrchestrateError>((id, out))
                });
            }
            for result in join_all(futures).await {
                let (id, out) = result?;
                pending.remove(id.as_str());
                results.insert(id, out);
            }
        }
        Ok(results)
    }

    /// 前置校验:id 唯一、依赖有定义、依赖图无环。
    fn validate(&self) -> Result<(), OrchestrateError> {
        let mut seen = HashSet::with_capacity(self.steps.len());
        let mut by_id: HashMap<&str, &ToolStep> = HashMap::with_capacity(self.steps.len());
        for s in &self.steps {
            if !seen.insert(s.id.as_str()) {
                return Err(OrchestrateError::DuplicateId(s.id.clone()));
            }
            by_id.insert(s.id.as_str(), s);
        }
        for s in &self.steps {
            for d in &s.depends_on {
                if !by_id.contains_key(d.as_str()) {
                    return Err(OrchestrateError::UnknownStep(d.clone()));
                }
            }
        }
        // Kahn 拓扑:每步入度取去重后的依赖数;减到 0 才可执行。
        let mut in_degree: HashMap<&str, usize> = HashMap::with_capacity(self.steps.len());
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();
        for s in &self.steps {
            let distinct: HashSet<&str> = s.depends_on.iter().map(|d| d.as_str()).collect();
            in_degree.insert(s.id.as_str(), distinct.len());
            for d in distinct {
                dependents.entry(d).or_default().push(&s.id);
            }
        }
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &n)| n == 0)
            .map(|(&k, _)| k)
            .collect();
        let mut done = 0usize;
        while let Some(id) = queue.pop_front() {
            done += 1;
            if let Some(deps) = dependents.get(id) {
                for &dep in deps {
                    let e = in_degree
                        .get_mut(dep)
                        .ok_or_else(|| OrchestrateError::MissingStep(dep.to_string()))?;
                    *e -= 1;
                    if *e == 0 {
                        queue.push_back(dep);
                    }
                }
            }
        }
        if done != self.steps.len() {
            return Err(OrchestrateError::Cycle);
        }
        Ok(())
    }
}

/// 解析参数里的 `${id}` / `${id.field}` 模板。
///
/// 仅当整个字符串恰好是单个 `${...}` 引用时替换;普通文本原样保留。
/// 遍历 JSON 树:数组逐元素、对象逐字段,引用部分以其前序输出代入。
fn resolve_template(
    value: &Value,
    results: &HashMap<String, Value>,
) -> Result<Value, OrchestrateError> {
    match value {
        Value::String(s) => {
            if let Some(rest) = s.strip_prefix("${").and_then(|r| r.strip_suffix('}')) {
                let mut parts = rest.split('.');
                let Some(id) = parts.next() else {
                    return Err(OrchestrateError::MissingStep(rest.to_string()));
                };
                let cur = results
                    .get(id)
                    .ok_or_else(|| OrchestrateError::MissingStep(id.to_string()))?;
                let mut out = cur;
                for field in parts {
                    out = out.get(field).ok_or_else(|| {
                        OrchestrateError::MissingField(id.to_string(), field.to_string())
                    })?;
                }
                Ok(out.clone())
            } else {
                Ok(value.clone())
            }
        }
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(resolve_template(item, results)?);
            }
            Ok(Value::Array(out))
        }
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                out.insert(k.clone(), resolve_template(v, results)?);
            }
            Ok(Value::Object(out))
        }
        other => Ok(other.clone()),
    }
}

#[async_trait::async_trait]
impl ToolCaller for MCPGateway {
    async fn call(&self, tool: &str, args: Value) -> Result<Value, ToolError> {
        let raw = MCPGateway::call(self, tool, args).await?;
        // Gateway 返回的是序列化文本;能解析成 JSON 就给结构化输出,
        // 否则退化为纯文本值(供 `${id}` 整体引用)。
        match serde_json::from_str(&raw) {
            Ok(v) => Ok(v),
            Err(_) => Ok(Value::String(raw)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{start_fake_sse_server, PostMode};
    use crate::{GatewayServerSpec, MCPConfig};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// 脚本化 caller:按工具名返回结构化输出,便于断言模板代入。
    struct ScriptedCaller;

    #[async_trait::async_trait]
    impl ToolCaller for ScriptedCaller {
        async fn call(&self, tool: &str, args: Value) -> Result<Value, ToolError> {
            match tool {
                "sum" => Ok(json!({
                    "sum": args["a"].as_i64().unwrap_or(0) + args["b"].as_i64().unwrap_or(0)
                })),
                "upper" => Ok(json!({
                    "text": args["text"].as_str().unwrap_or("").to_uppercase()
                })),
                "echo" => Ok(args),
                _ => Err(ToolError::ExecutionFailed(format!("未知工具 {tool}"))),
            }
        }
    }

    /// 线性链:b 依赖 a,参数 `${a.sum}` 代入 a 的输出。
    #[tokio::test]
    async fn test_linear_chain_resolves_template() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("a", "sum", json!({ "a": 1, "b": 2 })))
            .add_step(ToolStep::new("b", "sum", json!({ "a": "${a.sum}", "b": 4 })).after("a"));

        let results = orch.execute(&ScriptedCaller).await.unwrap();
        assert_eq!(results["a"]["sum"], 3);
        assert_eq!(
            results["b"]["sum"], 7,
            "${{a.sum}} should be substituted with a's result 3"
        );
    }

    /// 并行依赖:x、y 互不依赖并行跑完,z 汇总两者。
    #[tokio::test]
    async fn test_parallel_deps_join() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("x", "sum", json!({ "a": 2, "b": 3 })))
            .add_step(ToolStep::new("y", "sum", json!({ "a": 4, "b": 1 })))
            .add_step(
                ToolStep::new("z", "sum", json!({ "a": "${x.sum}", "b": "${y.sum}" }))
                    .after("x")
                    .after("y"),
            );

        let results = orch.execute(&ScriptedCaller).await.unwrap();
        assert_eq!(results["z"]["sum"], 10, "x.sum=5 and y.sum=5 should sum");
    }

    /// 整结果引用:参数就是 `${id}`,替换为前序步骤的完整输出对象。
    #[tokio::test]
    async fn test_whole_result_reference() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("s", "upper", json!({ "text": "hi" })))
            .add_step(ToolStep::new("t", "echo", json!("${s}")).after("s"));

        let results = orch.execute(&ScriptedCaller).await.unwrap();
        assert_eq!(
            results["t"],
            json!({ "text": "HI" }),
            "whole result should be substituted"
        );
    }

    /// 环检测:a → b → a,执行前报 Cycle。
    #[tokio::test]
    async fn test_cycle_detected() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("a", "sum", json!({})).after("b"))
            .add_step(ToolStep::new("b", "sum", json!({})).after("a"));

        let err = orch.execute(&ScriptedCaller).await.unwrap_err();
        assert_eq!(err, OrchestrateError::Cycle);
    }

    /// 未定义的依赖:执行前报 UnknownStep。
    #[tokio::test]
    async fn test_unknown_step_rejected() {
        let orch =
            ToolOrchestrator::new().add_step(ToolStep::new("a", "sum", json!({})).after("nope"));

        let err = orch.execute(&ScriptedCaller).await.unwrap_err();
        assert_eq!(err, OrchestrateError::UnknownStep("nope".to_string()));
    }

    /// 重复 id:执行前报 DuplicateId。
    #[tokio::test]
    async fn test_duplicate_id_rejected() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("a", "sum", json!({})))
            .add_step(ToolStep::new("a", "sum", json!({})));

        let err = orch.execute(&ScriptedCaller).await.unwrap_err();
        assert_eq!(err, OrchestrateError::DuplicateId("a".to_string()));
    }

    /// 模板字段缺失:a 存在但无 `nope` 字段 → MissingField。
    #[tokio::test]
    async fn test_missing_template_field() {
        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("a", "sum", json!({ "a": 1, "b": 2 })))
            .add_step(ToolStep::new("b", "sum", json!({ "a": "${a.nope}", "b": 1 })).after("a"));

        let err = orch.execute(&ScriptedCaller).await.unwrap_err();
        assert_eq!(
            err,
            OrchestrateError::MissingField("a".to_string(), "nope".to_string())
        );
    }

    /// 并发上限:6 个独立步骤,`max_concurrency=2`,同时执行数 ≤ 2。
    #[tokio::test]
    async fn test_concurrency_capped() {
        struct GatedCaller {
            in_flight: AtomicUsize,
            max_observed: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl ToolCaller for GatedCaller {
            async fn call(&self, _tool: &str, _args: Value) -> Result<Value, ToolError> {
                let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_observed.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(15)).await;
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(Value::String("done".into()))
            }
        }

        let mut orch = ToolOrchestrator::new().with_max_concurrency(2);
        for i in 0..6 {
            orch = orch.add_step(ToolStep::new(format!("s{i}"), "sum", json!({})));
        }
        let caller = GatedCaller {
            in_flight: AtomicUsize::new(0),
            max_observed: AtomicUsize::new(0),
        };
        let results = orch.execute(&caller).await.unwrap();
        assert_eq!(results.len(), 6, "all steps should complete");
        assert!(
            caller.max_observed.load(Ordering::SeqCst) <= 2,
            "concurrent executions must not exceed the concurrency cap, actual {}",
            caller.max_observed.load(Ordering::SeqCst)
        );
    }

    /// 与真实 Gateway 集成:两个步骤都走假 SSE 服务器 `fs:echo`,
    /// 步骤间依赖成立,`impl ToolCaller for MCPGateway` 路径打通。
    #[tokio::test]
    async fn test_orchestrator_with_gateway() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let gateway = MCPGateway::new();
        gateway
            .register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
            .await
            .unwrap();

        let orch = ToolOrchestrator::new()
            .add_step(ToolStep::new("one", "fs:echo", json!({ "msg": "hi" })))
            .add_step(ToolStep::new("two", "fs:echo", json!({})).after("one"));

        let results = orch.execute(&gateway).await.unwrap();
        assert_eq!(results["one"], Value::String("echo".to_string()));
        assert_eq!(results["two"], Value::String("echo".to_string()));
        assert_eq!(results.len(), 2);
    }
}
