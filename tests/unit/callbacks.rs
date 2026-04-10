// tests/unit/callbacks.rs
//! 回调系统单元测试
//!
//! 测试追踪和回调系统的核心功能：
//! - RunType 枚举：不同运行类型的分类
//! - RunTree 数据结构：追踪层次结构表示
//! - CallbackManager：多处理器协调
//! - CallbackHandler trait：回调处理器实现

use langchainrust::callbacks::{RunTree, RunType, CallbackManager, CallbackHandler, StdOutHandler};
use langchainrust::schema::Message;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

// ============================================================================
// RunType 枚举测试
// ============================================================================

#[test]
fn test_run_type_as_str() {
    // 验证每种 RunType 变体返回正确的字符串表示
    // 用于 API 负载和日志记录
    assert_eq!(RunType::Llm.as_str(), "llm");
    assert_eq!(RunType::Chain.as_str(), "chain");
    assert_eq!(RunType::Tool.as_str(), "tool");
    assert_eq!(RunType::Retriever.as_str(), "retriever");
    assert_eq!(RunType::Embedding.as_str(), "embedding");
    assert_eq!(RunType::Prompt.as_str(), "prompt");
    assert_eq!(RunType::Parser.as_str(), "parser");
}

#[test]
fn test_run_type_emoji() {
    // 验证控制台输出的图标表示（用于调试可视化）
    assert_eq!(RunType::Llm.emoji(), "🤖");
    assert_eq!(RunType::Chain.emoji(), "🔗");
    assert_eq!(RunType::Tool.emoji(), "🔧");
    assert_eq!(RunType::Retriever.emoji(), "📚");
    assert_eq!(RunType::Embedding.emoji(), "📊");
    assert_eq!(RunType::Prompt.emoji(), "📝");
    assert_eq!(RunType::Parser.emoji(), "📄");
}

#[test]
fn test_run_type_display() {
    // 验证 Display trait 实现使用 as_str() 方法
    assert_eq!(format!("{}", RunType::Llm), "llm");
    assert_eq!(format!("{}", RunType::Chain), "chain");
}

// ============================================================================
// RunTree 创建测试
// ============================================================================

#[test]
fn test_run_tree_new() {
    // 验证 RunTree 基础创建，所有字段正确初始化
    let run = RunTree::new("Test Run", RunType::Chain, serde_json::json!({"input": "test"}));
    
    assert_eq!(run.name, "Test Run");
    assert_eq!(run.run_type, RunType::Chain);
    assert!(run.outputs.is_none(), "创建时 outputs 应为 None");
    assert!(run.error.is_none(), "创建时 error 应为 None");
    assert!(run.parent_run_id.is_none(), "根运行的 parent_run_id 应为 None");
    assert!(run.end_time.is_none(), "结束前 end_time 应为 None");
    assert!(run.tags.is_empty(), "默认 tags 应为空");
    assert!(run.metadata.is_empty(), "默认 metadata 应为空");
}

#[test]
fn test_run_tree_end() {
    // 验证 end() 正确设置 outputs 和 end_time
    let mut run = RunTree::new("Test", RunType::Llm, serde_json::json!({}));
    assert!(run.end_time.is_none());
    
    run.end(serde_json::json!({"output": "result"}));
    
    assert!(run.outputs.is_some(), "end() 后 outputs 应被设置");
    assert!(run.end_time.is_some(), "end() 后 end_time 应被设置");
    assert!(run.duration_ms().is_some(), "end() 后 duration_ms 应可用");
    assert!(run.duration_ms().unwrap() >= 0, "耗时应为非负数");
}

#[test]
fn test_run_tree_end_with_error() {
    // 验证 end_with_error() 正确设置 error 字段和 end_time
    let mut run = RunTree::new("Test", RunType::Tool, serde_json::json!({}));
    
    run.end_with_error("Something went wrong");
    
    assert!(run.error.is_some(), "error 应被设置");
    assert_eq!(run.error.unwrap(), "Something went wrong");
    assert!(run.end_time.is_some(), "错误时 end_time 也应被设置");
    assert!(run.outputs.is_none(), "错误时 outputs 应保持 None");
}

// ============================================================================
// RunTree 构建器模式测试
// ============================================================================

#[test]
fn test_run_tree_with_tag() {
    // 验证使用构建器模式添加标签
    let run = RunTree::new("Test", RunType::Chain, serde_json::json!({}))
        .with_tag("test-tag")
        .with_tag("another-tag");
    
    assert_eq!(run.tags.len(), 2);
    assert!(run.tags.contains(&"test-tag".to_string()));
    assert!(run.tags.contains(&"another-tag".to_string()));
}

#[test]
fn test_run_tree_with_metadata() {
    // 验证使用构建器模式添加元数据
    let run = RunTree::new("Test", RunType::Chain, serde_json::json!({}))
        .with_metadata("version", serde_json::json!("1.0"))
        .with_metadata("count", serde_json::json!(42));
    
    assert_eq!(run.metadata.len(), 2);
    assert_eq!(run.metadata.get("version").unwrap(), "1.0");
    assert_eq!(run.metadata.get("count").unwrap(), 42);
}

#[test]
fn test_run_tree_with_project() {
    // 验证可为 LangSmith 组织设置项目名称
    let run = RunTree::new("Test", RunType::Chain, serde_json::json!({}))
        .with_project("my-langsmith-project");
    
    assert_eq!(run.project_name, Some("my-langsmith-project".to_string()));
}

// ============================================================================
// RunTree 层次结构测试
// ============================================================================

#[test]
fn test_run_tree_create_child() {
    // 验证父子关系创建
    // 子运行应继承父运行的 trace_id 并设置 parent_run_id
    let parent = RunTree::new("Parent", RunType::Chain, serde_json::json!({"input": "test"}));
    let child = parent.create_child("Child", RunType::Tool, serde_json::json!({"action": "run"}));
    
    assert_eq!(child.name, "Child");
    assert_eq!(child.run_type, RunType::Tool);
    assert_eq!(child.parent_run_id, Some(parent.id), "子运行应引用父运行");
    assert_eq!(child.trace_id, Some(parent.id), "trace_id 应为父运行的 id");
    assert_eq!(child.project_name, parent.project_name, "子运行继承 project_name");
}

#[test]
fn test_run_tree_nested_children() {
    // 验证多级层次结构：父 -> 子 -> 孙
    // 所有后代应共享相同的 trace_id（根运行的 id）
    let parent = RunTree::new("Parent", RunType::Chain, serde_json::json!({}));
    let child1 = parent.create_child("Child1", RunType::Tool, serde_json::json!({}));
    let grandchild = child1.create_child("Grandchild", RunType::Llm, serde_json::json!({}));
    
    assert_eq!(grandchild.parent_run_id, Some(child1.id));
    assert_eq!(grandchild.trace_id, Some(parent.id), "所有后代共享根 trace_id");
}

#[test]
fn test_run_tree_chain() {
    // 模拟典型链结构：Chain -> Tool, Chain -> LLM
    let parent = RunTree::new("Chain", RunType::Chain, serde_json::json!({"input": "query"}));
    let tool_run = parent.create_child("Calculator", RunType::Tool, serde_json::json!({"expr": "1+1"}));
    let llm_run = parent.create_child("LLM", RunType::Llm, serde_json::json!({"prompt": "..."}));
    
    // 两个子运行都指向同一个父运行
    assert_eq!(tool_run.parent_run_id, Some(parent.id));
    assert_eq!(llm_run.parent_run_id, Some(parent.id));
    // 所有运行共享相同的 trace_id
    assert_eq!(tool_run.trace_id, Some(parent.id));
    assert_eq!(llm_run.trace_id, Some(parent.id));
}

// ============================================================================
// RunTree 工具方法测试
// ============================================================================

#[test]
fn test_run_tree_duration() {
    // 验证运行结束后计算耗时
    let mut run = RunTree::new("Test", RunType::Llm, serde_json::json!({}));
    assert!(run.duration_ms().is_none(), "结束前 duration 应为 None");
    
    run.end(serde_json::json!({"output": "done"}));
    assert!(run.duration_ms().unwrap() >= 0, "结束后 duration 应为非负数");
}

#[test]
fn test_run_tree_uuid_v7() {
    // 验证 UUID v7 生成唯一 ID 且包含时间戳排序
    let run1 = RunTree::new("First", RunType::Chain, serde_json::json!({}));
    let run2 = RunTree::new("Second", RunType::Chain, serde_json::json!({}));
    
    assert_ne!(run1.id, run2.id, "每个运行应有唯一 ID");
}

#[test]
fn test_run_tree_serialization() {
    // 验证 RunTree 可序列化为 JSON 并反序列化回来
    let run = RunTree::new("Test", RunType::Chain, serde_json::json!({"input": "test"}))
        .with_tag("test-tag")
        .with_metadata("key", serde_json::json!("value"));
    
    let json = serde_json::to_string(&run).unwrap();
    assert!(json.contains("Test"));
    assert!(json.contains("chain"));
    
    let deserialized: RunTree = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, "Test");
    assert_eq!(deserialized.run_type, RunType::Chain);
}

#[test]
fn test_run_tree_tags_and_metadata_combined() {
    // 验证标签和元数据在实际场景中的组合使用
    let run = RunTree::new("Test", RunType::Chain, serde_json::json!({}))
        .with_tag("production")
        .with_tag("v2")
        .with_metadata("user_id", serde_json::json!("123"))
        .with_metadata("session", serde_json::json!({"id": "abc", "start": 12345}));
    
    // 验证标签
    assert_eq!(run.tags.len(), 2);
    assert!(run.tags.contains(&"production".to_string()));
    assert!(run.tags.contains(&"v2".to_string()));
    
    // 验证元数据
    assert_eq!(run.metadata.len(), 2);
    assert_eq!(run.metadata.get("user_id").unwrap(), "123");
}

// ============================================================================
// CallbackManager 测试
// ============================================================================

#[test]
fn test_callback_manager_new() {
    // 验证空管理器创建
    let manager = CallbackManager::new();
    assert!(manager.is_empty());
    assert_eq!(manager.handlers().len(), 0);
}

#[test]
fn test_callback_manager_add_handler() {
    // 验证添加单个处理器
    let manager = CallbackManager::new()
        .add_handler(Arc::new(StdOutHandler::new()));
    
    assert!(!manager.is_empty());
    assert_eq!(manager.handlers().len(), 1);
}

#[test]
fn test_callback_manager_multiple_handlers() {
    // 验证可添加多个处理器
    let manager = CallbackManager::new()
        .add_handler(Arc::new(StdOutHandler::new()))
        .add_handler(Arc::new(StdOutHandler::new().with_verbose(false)));
    
    assert_eq!(manager.handlers().len(), 2);
}

#[test]
fn test_callback_manager_clone() {
    // 验证管理器可克隆（用于跨线程共享）
    let manager = CallbackManager::new()
        .add_handler(Arc::new(StdOutHandler::new()));
    
    let cloned = manager.clone();
    assert_eq!(cloned.handlers().len(), 1);
}

#[test]
fn test_callback_manager_debug() {
    // 验证 Debug 实现显示处理器数量
    let manager = CallbackManager::new()
        .add_handler(Arc::new(StdOutHandler::new()))
        .add_handler(Arc::new(StdOutHandler::new()));
    
    let debug_str = format!("{:?}", manager);
    assert!(debug_str.contains("CallbackManager"));
    assert!(debug_str.contains("handlers_count"));
}

// ============================================================================
// Mock 回调处理器（用于测试）
// ============================================================================

/// Mock 回调处理器，记录调用次数用于测试
struct MockCallbackHandler {
    start_count: Arc<Mutex<usize>>,
    end_count: Arc<Mutex<usize>>,
    error_count: Arc<Mutex<usize>>,
}

impl MockCallbackHandler {
    fn new() -> Self {
        Self {
            start_count: Arc::new(Mutex::new(0)),
            end_count: Arc::new(Mutex::new(0)),
            error_count: Arc::new(Mutex::new(0)),
        }
    }
}

#[async_trait]
impl CallbackHandler for MockCallbackHandler {
    async fn on_run_start(&self, _run: &RunTree) {
        let mut count = self.start_count.lock().unwrap();
        *count += 1;
    }
    
    async fn on_run_end(&self, _run: &RunTree) {
        let mut count = self.end_count.lock().unwrap();
        *count += 1;
    }
    
    async fn on_run_error(&self, _run: &RunTree, _error: &str) {
        let mut count = self.error_count.lock().unwrap();
        *count += 1;
    }
}

// ============================================================================
// 回调处理器调用测试
// ============================================================================

#[tokio::test]
async fn test_callback_handler_calls() {
    // 验证 on_run_start 和 on_run_end 被正确调用
    let handler = Arc::new(MockCallbackHandler::new());
    let start_count = Arc::clone(&handler.start_count);
    let end_count = Arc::clone(&handler.end_count);
    
    let manager = CallbackManager::new().add_handler(handler);
    let run = RunTree::new("Test", RunType::Chain, serde_json::json!({}));
    
    for h in manager.handlers() {
        h.on_run_start(&run).await;
    }
    assert_eq!(*start_count.lock().unwrap(), 1);
    
    for h in manager.handlers() {
        h.on_run_end(&run).await;
    }
    assert_eq!(*end_count.lock().unwrap(), 1);
}

#[tokio::test]
async fn test_callback_handler_error() {
    // 验证 on_run_error 被调用并传入错误消息
    let handler = Arc::new(MockCallbackHandler::new());
    let error_count = Arc::clone(&handler.error_count);
    
    let manager = CallbackManager::new().add_handler(handler);
    let run = RunTree::new("Test", RunType::Chain, serde_json::json!({}));
    
    for h in manager.handlers() {
        h.on_run_error(&run, "test error").await;
    }
    assert_eq!(*error_count.lock().unwrap(), 1);
}

#[tokio::test]
async fn test_llm_callbacks() {
    // 验证 LLM 专用回调（on_llm_start, on_llm_end）
    let handler = Arc::new(MockCallbackHandler::new());
    let start_count = Arc::clone(&handler.start_count);
    let end_count = Arc::clone(&handler.end_count);
    
    let manager = CallbackManager::new().add_handler(handler);
    let run = RunTree::new("LLM", RunType::Llm, serde_json::json!({}));
    let messages = vec![Message::human("test")];
    
    for h in manager.handlers() {
        h.on_llm_start(&run, &messages).await;
    }
    assert_eq!(*start_count.lock().unwrap(), 1);
    
    for h in manager.handlers() {
        h.on_llm_end(&run, "response").await;
    }
    assert_eq!(*end_count.lock().unwrap(), 1);
}

#[tokio::test]
async fn test_tool_callbacks() {
    // 验证 Tool 专用回调（on_tool_start, on_tool_end）
    let handler = Arc::new(MockCallbackHandler::new());
    let start_count = Arc::clone(&handler.start_count);
    let end_count = Arc::clone(&handler.end_count);
    
    let manager = CallbackManager::new().add_handler(handler);
    let run = RunTree::new("Tool", RunType::Tool, serde_json::json!({}));
    
    for h in manager.handlers() {
        h.on_tool_start(&run, "Calculator", "1 + 1").await;
    }
    assert_eq!(*start_count.lock().unwrap(), 1);
    
    for h in manager.handlers() {
        h.on_tool_end(&run, "2").await;
    }
    assert_eq!(*end_count.lock().unwrap(), 1);
}

#[tokio::test]
async fn test_retriever_callbacks() {
    // 验证 Retriever 专用回调（on_retriever_start, on_retriever_end）
    let handler = Arc::new(MockCallbackHandler::new());
    let start_count = Arc::clone(&handler.start_count);
    let end_count = Arc::clone(&handler.end_count);
    
    let manager = CallbackManager::new().add_handler(handler);
    let run = RunTree::new("Retriever", RunType::Retriever, serde_json::json!({}));
    
    for h in manager.handlers() {
        h.on_retriever_start(&run, "query").await;
    }
    assert_eq!(*start_count.lock().unwrap(), 1);
    
    for h in manager.handlers() {
        h.on_retriever_end(&run, &[serde_json::json!({"doc": "result"})]).await;
    }
    assert_eq!(*end_count.lock().unwrap(), 1);
}