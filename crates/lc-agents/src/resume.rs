// lc-agents/src/resume.rs
//! 跨进程 resume(§4.2 人审/预算门):挂起状态落盘 + 恢复。
//!
//! 人审门 [`ApprovalHandler`](crate::approval::ApprovalHandler) 的 `approve`
//! 是纯 async —— 同进程内 future 挂起等待审批信号天然成立,但**进程死亡会
//! 丢失挂起点**:等待审批时进程被杀,审批信号永远不会到,agent 循环也无法
//! 继续。本模块把"等待审批的工具调用 + 恢复 agent loop 所需的上下文"序列化
//! 落盘,进程重启后从挂起点续跑,而不是从头重放整个对话。
//!
//! 分工:
//!
//! - [`PendingApproval`]:待审批工具 + 输入 / 中间步骤 / 迭代序号 / 预算累计
//!   快照,`Serialize + Deserialize`。
//! - [`ResumeStore`]:挂起点存取接口。
//! - [`FileResumeStore`]:磁盘实现(JSON + 原子写),真实跨进程恢复用。
//! - [`MemoryResumeStore`]:内存实现,测试 / 单进程演示用。
//!
//! 框架接入点(见 `executor/agent_loop.rs::execute_tool_inner`):每次工具调用
//! 进入人审门等待审批**之前**,把 [`PendingApproval`] 写入 store;审批决定
//! **落地之后**清除。崩溃时挂起点留在磁盘,新进程用 [`AgentExecutor::pending_approval`]
//! 查看、[`AgentExecutor::resume`] 续跑。
//!
//! 恢复(进程 B):
//!
//! ```rust,ignore
//! // 进程 B:重建与进程 A 相同的 executor(相同 agent / tools / store 目录)。
//! let store = Arc::new(FileResumeStore::new("/var/checkpoints/app")?);
//! let executor = AgentExecutor::new(agent, tools)
//!     .with_resume_store(store)
//!     .with_approval(handler);
//!
//! if let Some(pending) = executor.pending_approval().await? {
//!     // 向操作员展示 pending.tool_name / pending.arguments,收集审批决定。
//!     let answer = executor.resume(decision).await?;
//! }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::types::AgentStep;

/// 跨进程 resume 错误。
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ResumeError {
    /// 文件系统 I/O 错误(创建目录 / 读 / 写 / 删除)。
    #[error("resume store I/O error: {0}")]
    Io(String),
    /// 挂起点序列化 / 反序列化错误。
    #[error("resume store serialization error: {0}")]
    Serialize(String),
}

/// 待审批的工具调用 + 恢复 agent loop 所需的上下文快照。
///
/// 框架在 `execute_tool` 内、调用审批 handler **之前**落盘(此时同步 hook
/// 已跑完,`tool_name` / `arguments` 是审批真正看到的最终值);审批决定落地后
/// 清除。进程崩溃时挂起点留在磁盘,新进程加载后重新进入审批流程(或直接用
/// 给定决定续跑),而非从头重放已完成的中间步骤。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingApproval {
    /// 待审批工具名(同步 hook 修改后的最终值)。
    pub tool_name: String,
    /// 待审批工具参数(JSON;同步 hook 修改后的最终值)。
    pub arguments: serde_json::Value,
    /// 工具调用 id(function-calling 风格;无则空串)。
    pub tool_id: String,
    /// 恢复 agent loop 的输入变量(prompt 变量,含 `input`)。
    pub inputs: HashMap<String, String>,
    /// 已完成的中介步骤(此前所有工具 action + 观察)。恢复后从中继续,不重放。
    pub steps: Vec<AgentStep>,
    /// 挂起时的迭代序号。恢复后从该迭代继续,迭代预算不间断。
    pub iteration: usize,
    /// 预算:已消耗的工具调用次数(含本次待审批工具)。
    pub tool_calls_consumed: usize,
    /// 预算:已消耗的 LLM token(agent 不上报则为 None)。
    pub tokens_consumed: Option<usize>,
    /// 原运行 trace_id(恢复后沿用,保持追踪连续)。
    pub trace_id: Option<String>,
}

/// 挂起点存储接口。
///
/// 框架在审批 handler 前后调用 [`save_pending`](Self::save_pending) /
/// [`clear_pending`](Self::clear_pending);新进程用 [`load_pending`](Self::load_pending)
/// 读取挂起点。实现需 `Send + Sync`(跨任务 / 跨进程共享)。
#[async_trait]
pub trait ResumeStore: Send + Sync {
    /// 持久化一个待审批挂起点。
    async fn save_pending(&self, pending: &PendingApproval) -> Result<(), ResumeError>;
    /// 读取当前挂起点;无则返回 `None`。
    async fn load_pending(&self) -> Result<Option<PendingApproval>, ResumeError>;
    /// 清除挂起点(审批决定已落地 / 挂起点已被认领)。不存在时视为成功。
    async fn clear_pending(&self) -> Result<(), ResumeError>;
}

/// 磁盘挂起点存储:JSON 落盘 + 原子写。
///
/// 原子写:先写 `pending.json.tmp` 再 `rename`,崩溃不产生半截 checkpoint。
/// 目录由调用方指定,**并发 executor 须用各自独立目录**,避免互相覆盖挂起点。
pub struct FileResumeStore {
    dir: PathBuf,
}

impl FileResumeStore {
    /// 创建挂起点存储。目录不存在时自动创建(含父目录)。
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, ResumeError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| ResumeError::Io(format!("create dir {}: {}", dir.display(), e)))?;
        Ok(Self { dir })
    }

    fn pending_path(&self) -> PathBuf {
        self.dir.join("pending.json")
    }
}

#[async_trait]
impl ResumeStore for FileResumeStore {
    async fn save_pending(&self, pending: &PendingApproval) -> Result<(), ResumeError> {
        let bytes = serde_json::to_vec_pretty(pending)
            .map_err(|e| ResumeError::Serialize(e.to_string()))?;
        let tmp = self.dir.join("pending.json.tmp");
        tokio::fs::write(&tmp, &bytes)
            .await
            .map_err(|e| ResumeError::Io(format!("write {}: {}", tmp.display(), e)))?;
        tokio::fs::rename(&tmp, self.pending_path())
            .await
            .map_err(|e| ResumeError::Io(format!("rename {}: {}", tmp.display(), e)))?;
        Ok(())
    }

    async fn load_pending(&self) -> Result<Option<PendingApproval>, ResumeError> {
        let path = self.pending_path();
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(ResumeError::Io(format!("read {}: {}", path.display(), e)));
            }
        };
        let pending = serde_json::from_slice(&bytes)
            .map_err(|e| ResumeError::Serialize(format!("parse {}: {}", path.display(), e)))?;
        Ok(Some(pending))
    }

    async fn clear_pending(&self) -> Result<(), ResumeError> {
        let path = self.pending_path();
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            // 挂起点本就不存在:幂等清除,视为成功。
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(ResumeError::Io(format!("remove {}: {}", path.display(), e))),
        }
    }
}

/// 内存挂起点存储(测试 / 单进程演示用;进程死亡不持久)。
#[derive(Default)]
pub struct MemoryResumeStore {
    pending: tokio::sync::Mutex<Option<PendingApproval>>,
}

impl MemoryResumeStore {
    /// 新建空的内存挂起点存储。
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ResumeStore for MemoryResumeStore {
    async fn save_pending(&self, pending: &PendingApproval) -> Result<(), ResumeError> {
        *self.pending.lock().await = Some(pending.clone());
        Ok(())
    }

    async fn load_pending(&self) -> Result<Option<PendingApproval>, ResumeError> {
        Ok(self.pending.lock().await.clone())
    }

    async fn clear_pending(&self) -> Result<(), ResumeError> {
        *self.pending.lock().await = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentAction, ToolInput};

    fn sample_pending() -> PendingApproval {
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "compute".to_string());
        PendingApproval {
            tool_name: "calculator".to_string(),
            arguments: serde_json::json!({"expression": "2 + 3"}),
            tool_id: "call_1".to_string(),
            inputs,
            steps: vec![AgentStep::new(
                AgentAction {
                    tool: "other".to_string(),
                    tool_input: ToolInput::String {
                        value: "x".to_string(),
                    },
                    log: String::new(),
                },
                "obs".to_string(),
            )],
            iteration: 3,
            tool_calls_consumed: 4,
            tokens_consumed: Some(128),
            trace_id: Some("trace-1".to_string()),
        }
    }

    #[tokio::test]
    async fn test_file_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileResumeStore::new(dir.path()).unwrap();

        // 空目录:无挂起点。
        assert!(store.load_pending().await.unwrap().is_none());

        let pending = sample_pending();
        store.save_pending(&pending).await.unwrap();
        let loaded = store.load_pending().await.unwrap().unwrap();
        assert_eq!(loaded.tool_name, "calculator");
        assert_eq!(loaded.arguments, serde_json::json!({"expression": "2 + 3"}));
        assert_eq!(loaded.iteration, 3);
        assert_eq!(loaded.tool_calls_consumed, 4);
        assert_eq!(loaded.tokens_consumed, Some(128));
        assert_eq!(loaded.inputs.get("input").unwrap(), "compute");
        assert_eq!(loaded.steps.len(), 1);
        assert_eq!(loaded.trace_id.as_deref(), Some("trace-1"));

        store.clear_pending().await.unwrap();
        assert!(store.load_pending().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_file_store_clear_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileResumeStore::new(dir.path()).unwrap();
        // 无挂起点时 clear 不报错(幂等)。
        store.clear_pending().await.unwrap();
        store.clear_pending().await.unwrap();
    }

    #[tokio::test]
    async fn test_file_store_atomic_no_tmp_left() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileResumeStore::new(dir.path()).unwrap();
        store.save_pending(&sample_pending()).await.unwrap();

        // 原子写:保存完成后无 .tmp 残留。
        let entries: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            !entries.iter().any(|n| n == "pending.json.tmp"),
            "tmp file should be renamed away, got {entries:?}"
        );
        assert!(entries.contains(&"pending.json".to_string()));
    }

    #[tokio::test]
    async fn test_memory_store_roundtrip() {
        let store = MemoryResumeStore::new();
        assert!(store.load_pending().await.unwrap().is_none());

        store.save_pending(&sample_pending()).await.unwrap();
        assert!(store.load_pending().await.unwrap().is_some());

        store.clear_pending().await.unwrap();
        assert!(store.load_pending().await.unwrap().is_none());
    }

    #[test]
    fn test_pending_approval_serde_roundtrip() {
        let pending = sample_pending();
        let bytes = serde_json::to_vec(&pending).unwrap();
        let back: PendingApproval = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.tool_name, pending.tool_name);
        assert_eq!(back.arguments, pending.arguments);
        assert_eq!(back.steps.len(), pending.steps.len());
        assert_eq!(back.trace_id, pending.trace_id);
    }

    #[test]
    fn test_resume_error_display() {
        let e = ResumeError::Io("disk full".to_string());
        assert!(e.to_string().contains("disk full"));
        let e = ResumeError::Serialize("bad json".to_string());
        assert!(e.to_string().contains("bad json"));
    }

    /// 便捷断言:`FileResumeStore::new` 自动创建目录。
    #[tokio::test]
    async fn test_file_store_creates_dir() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a").join("b");
        let store = FileResumeStore::new(&nested).unwrap();
        assert!(nested.is_dir());
        assert!(store.load_pending().await.is_ok());
    }
}
