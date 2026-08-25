// lc-agents/src/approval.rs
//! 人审门(§4.2):工具执行前的**异步**审批闸。
//!
//! 与同步 hook 体系(`hooks::ApprovalHook` / `ToolCallAction`)并存、不冲突:
//! sync hook 在 `execute_tool` 内部跑,人审门也在 `execute_tool` 内部、同步
//! hook **之后**、实际执行**之前**跑 —— 顺序为
//! `预算门 → execute_tool(同步 hook → 人审门 → 工具执行)`。
//!
//! 框架只提供闸,审批策略由调用方实现 [`ApprovalHandler`](trait)。参考实现
//! [`AllowAll`] 供测试/演示。

use async_trait::async_trait;
use serde_json::Value;

use crate::hooks::ToolCallContext;

/// 工具执行前的审批决定。
#[derive(Debug, Clone)]
pub enum ApprovalDecision {
    /// 放行,原样执行。
    Allow,
    /// 拒绝:不执行该工具,把理由作为 observation 喂回循环,下一轮重新 plan。
    Deny {
        /// 拒绝理由(进入 observation)。
        reason: String,
    },
    /// 改参后执行:用 `arguments` 替换原参数。
    Modify {
        /// 替换后的工具参数。
        arguments: Value,
        /// 修改说明(记日志用)。
        note: String,
    },
}

/// 人审门接口。由调用方实现,`AgentExecutor::with_approval` 注入。
///
/// `approve` 是异步的:实现可以 `await` 一个审批信号(CLI 交互 / Webhook /
/// 消息通道)。同进程 resume 靠 async await 天然成立 —— future 挂起等待信号,
/// 信号到即从同一行继续,无需序列化/Checkpointer。
#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    /// 工具执行前被调用,返回审批决定。
    async fn approve(&self, ctx: &ToolCallContext) -> ApprovalDecision;
}

/// 参考实现:全部放行。测试/演示用。
#[derive(Debug, Default)]
pub struct AllowAll;

#[async_trait]
impl ApprovalHandler for AllowAll {
    async fn approve(&self, _ctx: &ToolCallContext) -> ApprovalDecision {
        ApprovalDecision::Allow
    }
}
