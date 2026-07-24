use langchainrust::{AgentState, MessageEntry, MessageRole, StateUpdate, StepEntry};

// 测试 AgentState 创建
// 验证: 新创建的 AgentState 包含初始输入和第一条 human 消息
#[test]
fn test_agent_state_creation() {
    let state = AgentState::new("Hello world".to_string());

    // 输入应保存
    assert_eq!(state.input, "Hello world");
    // 应有一条初始 human 消息
    assert_eq!(state.messages.len(), 1);
    assert_eq!(state.messages[0].role, MessageRole::Human);
    assert_eq!(state.messages[0].content, "Hello world");
    // steps 应为空
    assert!(state.steps.is_empty());
    // output 应为 None
    assert!(state.output.is_none());
}

// 测试 AgentState 添加消息
// 验证: add_message 方法正确追加消息到历史
#[test]
fn test_agent_state_add_message() {
    let mut state = AgentState::new("question".to_string());

    // 添加 AI 消息
    state.add_message(MessageEntry::ai("answer".to_string()));

    // 应有两条消息: human + ai
    assert_eq!(state.messages.len(), 2);
    assert_eq!(state.messages[0].role, MessageRole::Human);
    assert_eq!(state.messages[1].role, MessageRole::AI);
    assert_eq!(state.messages[1].content, "answer");
}

// 测试 AgentState 添加步骤
// 验证: add_step 方法正确记录执行步骤
#[test]
fn test_agent_state_add_step() {
    let mut state = AgentState::new("task".to_string());

    // 添加执行步骤
    state.add_step(StepEntry::new(
        "tool_call".to_string(),
        "result".to_string(),
    ));

    // 应有一条步骤记录
    assert_eq!(state.steps.len(), 1);
    assert_eq!(state.steps[0].action, "tool_call");
    assert_eq!(state.steps[0].observation, "result");
}

// 测试 AgentState 设置输出
// 验证: set_output 方法正确设置最终输出
#[test]
fn test_agent_state_set_output() {
    let mut state = AgentState::new("input".to_string());

    state.set_output("final answer".to_string());

    assert_eq!(state.output, Some("final answer".to_string()));
}

// 测试各种消息类型
// 验证: MessageEntry 支持 human, ai, system, tool 四种角色
#[test]
fn test_message_entry_types() {
    // Human 消息
    let human = MessageEntry::human("user message".to_string());
    assert_eq!(human.role, MessageRole::Human);
    assert_eq!(human.content, "user message");

    // AI 消息
    let ai = MessageEntry::ai("ai response".to_string());
    assert_eq!(ai.role, MessageRole::AI);
    assert_eq!(ai.content, "ai response");

    // System 消息
    let system = MessageEntry::system("system prompt".to_string());
    assert_eq!(system.role, MessageRole::System);
    assert_eq!(system.content, "system prompt");

    // Tool 消息
    let tool = MessageEntry::tool("tool output".to_string());
    assert_eq!(tool.role, MessageRole::Tool);
    assert_eq!(tool.content, "tool output");
}

// 测试 StateUpdate full 构造
// 验证: full() 创建包含完整状态的更新
#[test]
fn test_state_update_full() {
    let state = AgentState::new("test".to_string());
    let update = StateUpdate::full(state.clone());

    // 更新应包含状态
    assert!(update.update.is_some());
    // 元数据应为空
    assert!(update.metadata.is_empty());
}

// 测试 StateUpdate with_metadata 构造
// 验证: with_metadata() 创建带元数据的更新
#[test]
fn test_state_update_with_metadata() {
    let state = AgentState::new("test".to_string());
    let metadata = std::collections::HashMap::from([
        ("node".to_string(), serde_json::json!("step1")),
        ("timestamp".to_string(), serde_json::json!(12345)),
    ]);
    let update = StateUpdate::with_metadata(state, metadata.clone());

    assert!(update.update.is_some());
    // 验证元数据正确保存
    assert_eq!(update.metadata.get("node").unwrap(), "step1");
}

// 测试 StateUpdate unchanged 构造
// 验证: unchanged() 创建不修改状态的更新
#[test]
fn test_state_update_unchanged() {
    let update = StateUpdate::<AgentState>::unchanged();

    // 更新不应包含状态
    assert!(update.update.is_none());
    assert!(update.metadata.is_empty());
}

// 测试 AgentState clone 保留数据
// 验证: clone 后的状态独立, 修改不影响原始
#[test]
fn test_agent_state_clone_preserves_data() {
    let original = AgentState::new("original input".to_string());
    // clone 是浅拷贝, 原始不受影响
    original
        .clone()
        .add_message(MessageEntry::ai("response".to_string()));

    let cloned = original.clone();

    // clone 应保留原始数据
    assert_eq!(cloned.input, "original input");
    assert_eq!(cloned.messages.len(), 1);
}

// 测试 MessageRole 相等比较
// 验证: MessageRole 支持相等和不等比较
#[test]
fn test_message_role_equality() {
    assert_eq!(MessageRole::Human, MessageRole::Human);
    assert_ne!(MessageRole::Human, MessageRole::AI);
    assert_ne!(MessageRole::System, MessageRole::Tool);
}
