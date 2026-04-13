//! FileCallbackHandler 功能测试
//!
//! 本测试文件验证文件日志回调处理器：
//! - 支持三种日志格式（Plain, Json, JsonLines）
//! - 正确写入所有回调事件
//! - 与 CallbackManager 集成
//!
//! 测试策略：使用真实的文件 I/O 操作，验证实际写入结果

use langchainrust::{
    FileCallbackHandler, LogFormat, CallbackHandler, CallbackManager,
    RunTree, RunType, Message,
};
use std::sync::Arc;
use tempfile::NamedTempFile;

#[cfg(test)]
mod tests {
    use super::*;
    
    // -------------------------------------------------------------------------
    // 文件创建和配置测试
    // -------------------------------------------------------------------------
    
    /// 验证 FileCallbackHandler 能成功创建日志文件
    ///
    /// FileCallbackHandler 使用 OpenOptions::append 模式：
    /// - create(true)：文件不存在时创建
    /// - append(true)：追加写入，不覆盖已有内容
    #[test]
    fn test_file_handler_creates_log_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path());
        
        assert!(handler.is_ok(), "应成功创建 FileCallbackHandler");
        assert!(temp_file.path().exists(), "日志文件应存在");
    }
    
    /// 验证三种日志格式都能正确配置
    ///
    /// - JsonLines：每行一个 JSON 对象（推荐，便于日志分析）
    /// - Json：格式化的 JSON（便于人类阅读）
    /// - Plain：纯文本格式（最简单）
    #[test]
    fn test_file_handler_format_configuration() {
        let temp_file = NamedTempFile::new().unwrap();
        
        // 默认是 JsonLines
        let handler_default = FileCallbackHandler::new(temp_file.path()).unwrap();
        // 可以切换为其他格式
        let handler_json = FileCallbackHandler::new(temp_file.path()).unwrap()
            .with_format(LogFormat::Json);
        let handler_plain = FileCallbackHandler::new(temp_file.path()).unwrap()
            .with_format(LogFormat::Plain);
        
        // 验证配置成功（通过 Debug 输出）
        assert!(format!("{:?}", handler_default).contains("JsonLines"));
        assert!(format!("{:?}", handler_json).contains("Json"));
        assert!(format!("{:?}", handler_plain).contains("Plain"));
    }
    
    // -------------------------------------------------------------------------
    // JsonLines 格式测试（推荐格式）
    // -------------------------------------------------------------------------
    
    /// 验证 JsonLines 格式正确写入 LLM 事件
    ///
    /// JsonLines 格式优势：
    /// - 每行独立 JSON，便于流式处理
    /// - 可用 jq、grep 等工具分析
    /// - 适合日志聚合系统（如 ELK）
    ///
    /// 写入内容包含：
    /// - timestamp：ISO 8601 格式时间戳
    /// - event：事件类型（llm_start, llm_end 等）
    /// - run_id, run_name, run_type：运行元数据
    /// - data：事件特定数据
    #[tokio::test]
    async fn test_json_lines_format_writes_llm_events() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path()).unwrap()
            .with_format(LogFormat::JsonLines);
        
        let run = RunTree::new("llm_run_001", RunType::Llm, serde_json::json!({
            "model": "gpt-4",
            "prompt": "Hello"
        }));
        
        let messages = vec![
            Message::system("You are helpful"),
            Message::human("Hello"),
        ];
        
        handler.on_llm_start(&run, &messages).await;
        handler.on_llm_end(&run, "Hi there!").await;
        
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        
        assert_eq!(lines.len(), 2, "应写入两行日志");
        
        // 验证第一行包含 llm_start 事件
        assert!(lines[0].contains("llm_start"));
        assert!(lines[0].contains("\"timestamp\""));
        assert!(lines[0].contains("\"run_name\":\"llm_run_001\""));
        
        // 验证第二行包含 llm_end 事件
        assert!(lines[1].contains("llm_end"));
        assert!(lines[1].contains("\"response_length\":9")); // "Hi there!" 长度
    }
    
    // -------------------------------------------------------------------------
    // Plain 格式测试
    // -------------------------------------------------------------------------
    
    /// 验证 Plain 格式输出人类可读的文本
    ///
    /// Plain 格式适用场景：
    /// - 快速调试
    /// - 简单日志查看
    /// - 不需要结构化分析
    ///
    /// 格式：[时间戳] 事件 - 名称 (类型)
    #[tokio::test]
    async fn test_plain_format_writes_human_readable_text() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path()).unwrap()
            .with_format(LogFormat::Plain);
        
        let run = RunTree::new("tool_run", RunType::Tool, serde_json::json!({}));
        
        handler.on_tool_start(&run, "calculator", "1 + 2").await;
        
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        
        // Plain 格式应包含时间戳和事件信息
        assert!(content.contains("tool_start"), "应包含事件名称");
        assert!(content.contains("tool_run"), "应包含运行名称");
        assert!(content.contains("["), "应包含时间戳括号");
    }
    
    // -------------------------------------------------------------------------
    // 所有回调事件测试
    // -------------------------------------------------------------------------
    
    /// 验证所有回调事件都能正确写入文件
    ///
    /// CallbackHandler 定义的事件：
    /// - 生命周期：run_start, run_end, run_error
    /// - LLM：llm_start, llm_end, llm_new_token, llm_error
    /// - Chain：chain_start, chain_end, chain_error
    /// - Tool：tool_start, tool_end, tool_error
    /// - Retriever：retriever_start, retriever_end, retriever_error
    #[tokio::test]
    async fn test_all_callback_events_written_to_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path()).unwrap();
        
        let run = RunTree::new("test_run", RunType::Chain, serde_json::json!({}));
        
        // 调用所有事件
        handler.on_run_start(&run).await;
        handler.on_run_end(&run).await;
        handler.on_run_error(&run, "test error").await;
        
        handler.on_llm_start(&run, &[]).await;
        handler.on_llm_end(&run, "response").await;
        handler.on_llm_new_token(&run, "token").await;
        handler.on_llm_error(&run, "llm error").await;
        
        handler.on_chain_start(&run, &serde_json::json!({})).await;
        handler.on_chain_end(&run, &serde_json::json!({})).await;
        handler.on_chain_error(&run, "chain error").await;
        
        handler.on_tool_start(&run, "tool", "input").await;
        handler.on_tool_end(&run, "output").await;
        handler.on_tool_error(&run, "tool error").await;
        
        handler.on_retriever_start(&run, "query").await;
        handler.on_retriever_end(&run, &[serde_json::json!({})]).await;
        handler.on_retriever_error(&run, "retriever error").await;
        
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        
        // 验证所有事件都有日志
        assert!(content.contains("run_start"));
        assert!(content.contains("run_end"));
        assert!(content.contains("run_error"));
        assert!(content.contains("llm_start"));
        assert!(content.contains("llm_end"));
        assert!(content.contains("llm_new_token"));
        assert!(content.contains("llm_error"));
        assert!(content.contains("chain_start"));
        assert!(content.contains("chain_end"));
        assert!(content.contains("chain_error"));
        assert!(content.contains("tool_start"));
        assert!(content.contains("tool_end"));
        assert!(content.contains("tool_error"));
        assert!(content.contains("retriever_start"));
        assert!(content.contains("retriever_end"));
        assert!(content.contains("retriever_error"));
    }
    
    // -------------------------------------------------------------------------
    // 追加写入测试
    // -------------------------------------------------------------------------
    
    /// 验证 FileCallbackHandler 正确追加写入（不覆盖）
    ///
    /// 多次调用应该累积日志，而非覆盖：
    /// - 第一次写入 run1
    /// - 第二次写入 run2
    /// - 文件应包含两者的日志
    #[tokio::test]
    async fn test_append_mode_accumulates_logs() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path()).unwrap();
        
        let run1 = RunTree::new("first_run", RunType::Llm, serde_json::json!({}));
        let run2 = RunTree::new("second_run", RunType::Llm, serde_json::json!({}));
        
        handler.on_llm_start(&run1, &[]).await;
        handler.on_llm_start(&run2, &[]).await;
        
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        
        assert!(content.contains("first_run"), "应包含第一次运行的日志");
        assert!(content.contains("second_run"), "应包含第二次运行的日志");
    }
    
    // -------------------------------------------------------------------------
    // CallbackManager 集成测试
    // -------------------------------------------------------------------------
    
    /// 验证 FileCallbackHandler 与 CallbackManager 正确集成
    ///
    /// CallbackManager 可以组合多个 Handler：
    /// - StdOutHandler（控制台输出）
    /// - LangSmithHandler（远程追踪）
    /// - FileCallbackHandler（本地文件）
    ///
    /// 三者可以同时工作，互不干扰
    #[test]
    fn test_file_handler_integrates_with_callback_manager() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_handler = Arc::new(FileCallbackHandler::new(temp_file.path()).unwrap());
        
        let manager = CallbackManager::new()
            .add_handler(file_handler);
        
        assert!(!manager.is_empty(), "CallbackManager 应有 handler");
        assert_eq!(manager.handlers().len(), 1, "应有 1 个 handler");
    }
    
    // -------------------------------------------------------------------------
    // 时间戳格式测试
    // -------------------------------------------------------------------------
    
    /// 验证时间戳使用 ISO 8601 格式
    ///
    /// ISO 8601 格式优势：
    /// - 国际标准，跨平台兼容
    /// - 包含时区信息
    /// - 便于排序和比较
    ///
    /// 格式示例：2026-04-13T12:34:56.789+08:00
    #[tokio::test]
    async fn test_timestamp_iso_8601_format() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path()).unwrap();
        
        let run = RunTree::new("time_test", RunType::Llm, serde_json::json!({}));
        
        handler.on_llm_start(&run, &[]).await;
        
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        
        // ISO 8601 格式特征
        assert!(content.contains("timestamp"), "应包含 timestamp 字段");
        assert!(content.contains("202"), "应包含年份");
        assert!(content.contains("-"), "应包含日期分隔符");
        assert!(content.contains("T"), "应包含日期时间分隔符");
    }
}