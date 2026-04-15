use langchainrust::{
    ThreadSafeMemoryCheckpointer,
    AgentState,
};
use langchainrust::Checkpointer;

// 测试 Checkpointer 保存功能
// 验证: save() 返回非空的 checkpoint ID
#[tokio::test]
async fn test_checkpointer_save() {
    let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
    
    let state = AgentState::new("test input".to_string());
    let id = checkpointer.save(&state).await.unwrap();
    
    // ID 应为有效的 UUID 字符串
    assert!(!id.is_empty());
}

// 测试 Checkpointer 加载功能
// 验证: load() 能正确恢复保存的状态
#[tokio::test]
async fn test_checkpointer_load() {
    let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
    
    // 保存状态
    let state = AgentState::new("saved state".to_string());
    let id = checkpointer.save(&state).await.unwrap();
    
    // 加载并验证
    let loaded = checkpointer.load(&id).await.unwrap();
    assert_eq!(loaded.input, "saved state");
}

// 测试 Checkpointer 列表功能
// 验证: list() 返回所有保存的 checkpoint ID
#[tokio::test]
async fn test_checkpointer_list() {
    let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
    
    // 保存两个状态
    let state1 = AgentState::new("first".to_string());
    let id1 = checkpointer.save(&state1).await.unwrap();
    
    let state2 = AgentState::new("second".to_string());
    let id2 = checkpointer.save(&state2).await.unwrap();
    
    // 获取列表
    let list = checkpointer.list().await.unwrap();
    assert_eq!(list.len(), 2);
    assert!(list.contains(&id1));
    assert!(list.contains(&id2));
}

// 测试 Checkpointer 删除功能
// 验证: delete() 能移除指定的 checkpoint
#[tokio::test]
async fn test_checkpointer_delete() {
    let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
    
    // 保存状态
    let state = AgentState::new("to_delete".to_string());
    let id = checkpointer.save(&state).await.unwrap();
    
    // 验证存在
    let list_before = checkpointer.list().await.unwrap();
    assert_eq!(list_before.len(), 1);
    
    // 删除
    checkpointer.delete(&id).await.unwrap();
    
    // 验证已删除
    let list_after = checkpointer.list().await.unwrap();
    assert!(list_after.is_empty());
}

// 测试加载不存在的 checkpoint 失败
// 验证: load() 对不存在的 ID 返回错误
#[tokio::test]
async fn test_checkpointer_load_nonexistent_fails() {
    let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
    
    // 加载不存在的 ID 应失败
    let result = checkpointer.load("nonexistent_id").await;
    assert!(result.is_err());
}

// 测试保存多个状态
// 验证: 可保存和加载多个独立状态
#[tokio::test]
async fn test_checkpointer_multiple_states() {
    let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
    
    // 保存5个状态
    let mut ids: Vec<String> = Vec::new();
    for i in 0..5 {
        let state = AgentState::new(format!("state_{}", i));
        let id = checkpointer.save(&state).await.unwrap();
        ids.push(id);
    }
    
    // 验证列表包含5个
    let list = checkpointer.list().await.unwrap();
    assert_eq!(list.len(), 5);
    
    // 验证每个都能正确加载
    for (i, id) in ids.iter().enumerate() {
        let loaded = checkpointer.load(id).await.unwrap();
        assert_eq!(loaded.input, format!("state_{}", i));
    }
}

// 测试 checkpoint ID 唯一性
// 验证: 即使状态相同, 每次保存生成不同 ID
#[tokio::test]
async fn test_checkpointer_id_uniqueness() {
    let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
    
    // 保存相同状态两次
    let state = AgentState::new("same input".to_string());
    let id1 = checkpointer.save(&state).await.unwrap();
    let id2 = checkpointer.save(&state).await.unwrap();
    
    // ID 应不同
    assert_ne!(id1, id2);
}
