// LangGraph 持久化模块测试
// 测试图的序列化、保存、加载和删除功能

use langchainrust::{
    langgraph::{
        EdgeDefinition, EdgeType, FilePersistence, GraphDefinition, GraphPersistence,
        MemoryPersistence, NodeDefinition, NodeType,
    },
    AgentState, GraphBuilder, StateUpdate, END, START,
};

#[cfg(feature = "mongodb-persistence")]
use langchainrust::langgraph::{MongoConfig, MongoPersistence};

use std::sync::Arc;
use tempfile::TempDir;

#[path = "../common/mod.rs"]
mod common;

#[cfg(feature = "mongodb-persistence")]
use common::MongoTestConfig;

// ============================================================================
// 内存持久化测试
// ============================================================================

/// 测试内存持久化的保存和加载功能
#[tokio::test]
async fn test_memory_persistence_save_load() {
    let persistence = MemoryPersistence::new();
    let definition = GraphDefinition::new("entry".to_string())
        .with_id("test-001".to_string())
        .with_name("Test Workflow".to_string());

    // 保存图定义
    persistence.save("test-001", &definition).await.unwrap();
    assert!(persistence.exists("test-001").await.unwrap());

    // 加载并验证
    let loaded = persistence.load("test-001").await.unwrap();
    assert_eq!(loaded.id, "test-001");
    assert_eq!(loaded.name, Some("Test Workflow".to_string()));
    assert_eq!(loaded.entry_point, "entry");
}

/// 测试内存持久化的删除功能
#[tokio::test]
async fn test_memory_persistence_delete() {
    let persistence = MemoryPersistence::new();
    let definition = GraphDefinition::new("entry".to_string()).with_id("test-002".to_string());

    // 保存后确认存在
    persistence.save("test-002", &definition).await.unwrap();
    assert!(persistence.exists("test-002").await.unwrap());

    // 删除后确认不存在
    persistence.delete("test-002").await.unwrap();
    assert!(!persistence.exists("test-002").await.unwrap());
}

/// 测试内存持久化的列表功能
#[tokio::test]
async fn test_memory_persistence_list() {
    let persistence = MemoryPersistence::new();

    // 保存3个图定义
    for i in 1..=3 {
        let def = GraphDefinition::new("entry".to_string()).with_id(format!("graph-{}", i));
        persistence
            .save(&format!("graph-{}", i), &def)
            .await
            .unwrap();
    }

    // 验证列表包含所有保存的图
    let list = persistence.list().await.unwrap();
    assert_eq!(list.len(), 3);
    assert!(list.contains(&"graph-1".to_string()));
    assert!(list.contains(&"graph-2".to_string()));
    assert!(list.contains(&"graph-3".to_string()));
}

/// 测试加载不存在的图时返回错误
#[tokio::test]
async fn test_memory_persistence_not_found() {
    let persistence = MemoryPersistence::new();

    // 加载不存在的图应该返回错误
    let result = persistence.load("nonexistent").await;
    assert!(result.is_err());
}

// ============================================================================
// 文件持久化测试
// ============================================================================

/// 测试文件持久化的保存和加载功能
#[tokio::test]
async fn test_file_persistence_save_load() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_str().unwrap();
    let persistence = FilePersistence::new(path).unwrap();

    let definition = GraphDefinition::new("process".to_string())
        .with_id("file-001".to_string())
        .with_name("File Test".to_string());

    // 保存到文件
    persistence.save("file-001", &definition).await.unwrap();
    assert!(persistence.exists("file-001").await.unwrap());

    // 从文件加载并验证
    let loaded = persistence.load("file-001").await.unwrap();
    assert_eq!(loaded.id, "file-001");
    assert_eq!(loaded.name, Some("File Test".to_string()));
}

/// 测试文件持久化生成的JSON格式正确
#[tokio::test]
async fn test_file_persistence_json_format() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_str().unwrap();
    let persistence = FilePersistence::new(path).unwrap();

    let definition = GraphDefinition::new("start".to_string())
        .with_id("json-test".to_string())
        .with_recursion_limit(100);

    persistence.save("json-test", &definition).await.unwrap();

    // 直接读取文件内容验证JSON格式
    let file_path = format!("{}/json-test.json", path);
    let content = std::fs::read_to_string(&file_path).unwrap();

    assert!(content.contains("\"id\": \"json-test\""));
    assert!(content.contains("\"recursion_limit\": 100"));
    assert!(content.contains("\"entry_point\": \"start\""));
}

/// 测试文件持久化的删除功能
#[tokio::test]
async fn test_file_persistence_delete() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_str().unwrap();
    let persistence = FilePersistence::new(path).unwrap();

    let definition = GraphDefinition::new("entry".to_string()).with_id("del-test".to_string());
    persistence.save("del-test", &definition).await.unwrap();

    // 确认文件存在
    let file_path = format!("{}/del-test.json", path);
    assert!(std::path::Path::new(&file_path).exists());

    // 删除后确认文件不存在
    persistence.delete("del-test").await.unwrap();
    assert!(!std::path::Path::new(&file_path).exists());
}

/// 测试文件持久化的列表功能
#[tokio::test]
async fn test_file_persistence_list() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().to_str().unwrap();
    let persistence = FilePersistence::new(path).unwrap();

    // 保存5个图定义
    for i in 1..=5 {
        let def = GraphDefinition::new("entry".to_string()).with_id(format!("wf-{}", i));
        persistence.save(&format!("wf-{}", i), &def).await.unwrap();
    }

    // 验证列表数量
    let list = persistence.list().await.unwrap();
    assert_eq!(list.len(), 5);
}

// ============================================================================
// 图定义构建器测试
// ============================================================================

/// 测试GraphDefinition的builder模式
#[test]
fn test_graph_definition_builder() {
    let def = GraphDefinition::new("entry_point".to_string())
        .with_id("custom-id".to_string())
        .with_name("My Workflow".to_string())
        .with_recursion_limit(50);

    assert_eq!(def.id, "custom-id");
    assert_eq!(def.name, Some("My Workflow".to_string()));
    assert_eq!(def.entry_point, "entry_point");
    assert_eq!(def.recursion_limit, 50);
}

/// 测试GraphDefinition自动生成UUID作为ID
#[test]
fn test_graph_definition_auto_id() {
    let def = GraphDefinition::new("entry".to_string());
    assert!(!def.id.is_empty());
    // 验证ID是有效的UUID格式
    assert!(uuid::Uuid::parse_str(&def.id).is_ok());
}

// ============================================================================
// 节点定义测试
// ============================================================================

/// 测试NodeDefinition的序列化和反序列化
#[test]
fn test_node_definition() {
    let node = NodeDefinition {
        name: "process".to_string(),
        node_type: NodeType::Sync,
        config: serde_json::json!({"key": "value"}),
    };

    // 序列化为JSON后再反序列化
    let json = serde_json::to_string(&node).unwrap();
    let parsed: NodeDefinition = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.name, "process");
    assert_eq!(parsed.node_type, NodeType::Sync);
}

/// 测试NodeType的所有类型都能正确序列化
#[test]
fn test_node_type_serialization() {
    let types = vec![
        NodeType::Sync,
        NodeType::Async,
        NodeType::Subgraph,
        NodeType::Custom,
    ];

    for t in types {
        let json = serde_json::to_string(&t).unwrap();
        let parsed: NodeType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
    }
}

// ============================================================================
// 边定义测试
// ============================================================================

/// 测试固定边的创建
#[test]
fn test_edge_definition_fixed() {
    let edge = EdgeDefinition::fixed("source".to_string(), "target".to_string());

    assert_eq!(edge.edge_type, EdgeType::Fixed);
    assert_eq!(edge.source, "source");
    assert_eq!(edge.target, Some("target".to_string()));
    assert!(edge.targets.is_none());
}

/// 测试条件边的创建
#[test]
fn test_edge_definition_conditional() {
    let mut targets = std::collections::HashMap::new();
    targets.insert("a".to_string(), "node_a".to_string());
    targets.insert("b".to_string(), "node_b".to_string());

    let edge = EdgeDefinition::conditional(
        "decision".to_string(),
        "router".to_string(),
        targets.clone(),
        Some("default".to_string()),
    );

    assert_eq!(edge.edge_type, EdgeType::Conditional);
    assert_eq!(edge.router_name, Some("router".to_string()));
    assert_eq!(edge.conditional_targets, Some(targets));
    assert_eq!(edge.default_target, Some("default".to_string()));
}

/// 测试扇出边（并行执行）的创建
#[test]
fn test_edge_definition_fan_out() {
    let edge = EdgeDefinition::fan_out(
        "source".to_string(),
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
    );

    assert_eq!(edge.edge_type, EdgeType::FanOut);
    assert_eq!(
        edge.targets,
        Some(vec!["a", "b", "c"].into_iter().map(String::from).collect())
    );
}

/// 测试扇入边（合并并行分支）的创建
#[test]
fn test_edge_definition_fan_in() {
    let edge = EdgeDefinition::fan_in(vec!["a".to_string(), "b".to_string()], "merge".to_string());

    assert_eq!(edge.edge_type, EdgeType::FanIn);
    assert_eq!(edge.target, Some("merge".to_string()));
}

/// 测试EdgeType的所有类型都能正确序列化
#[test]
fn test_edge_type_serialization() {
    let types = vec![
        EdgeType::Fixed,
        EdgeType::Conditional,
        EdgeType::FanOut,
        EdgeType::FanIn,
    ];

    for t in types {
        let json = serde_json::to_string(&t).unwrap();
        let parsed: EdgeType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
    }
}

// ============================================================================
// 编译图转换测试
// ============================================================================

/// 测试CompiledGraph转换为GraphDefinition
#[tokio::test]
async fn test_compiled_graph_to_definition() {
    // 构建3节点的线性图
    let graph = GraphBuilder::<AgentState>::new()
        .add_node_fn("step1", |s: &AgentState| Ok(StateUpdate::full(s.clone())))
        .add_node_fn("step2", |s: &AgentState| Ok(StateUpdate::full(s.clone())))
        .add_node_fn("step3", |s: &AgentState| Ok(StateUpdate::full(s.clone())))
        .add_edge(START, "step1")
        .add_edge("step1", "step2")
        .add_edge("step2", "step3")
        .add_edge("step3", END)
        .compile()
        .unwrap();

    // 转换为可序列化的定义
    let definition = graph.to_definition();

    assert_eq!(definition.nodes.len(), 3);
    assert_eq!(definition.edges.len(), 4);
    assert_eq!(definition.entry_point, "step1");
    assert_eq!(definition.recursion_limit, 25);
}

/// 测试完整的持久化往返：编译 -> 保存 -> 加载
#[tokio::test]
async fn test_persistence_roundtrip() {
    // 构建简单图
    let graph = GraphBuilder::<AgentState>::new()
        .add_node_fn("process", |s: &AgentState| {
            let mut state = s.clone();
            state.set_output("done".to_string());
            Ok(StateUpdate::full(state))
        })
        .add_edge(START, "process")
        .add_edge("process", END)
        .compile()
        .unwrap();

    let definition = graph.to_definition();

    // 保存到内存持久化
    let persistence = MemoryPersistence::new();
    persistence.save("roundtrip", &definition).await.unwrap();

    // 加载并验证数据一致
    let loaded = persistence.load("roundtrip").await.unwrap();

    assert_eq!(loaded.nodes.len(), definition.nodes.len());
    assert_eq!(loaded.edges.len(), definition.edges.len());
    assert_eq!(loaded.entry_point, definition.entry_point);
}

/// 测试GraphDefinition的时间戳自动设置
#[test]
fn test_definition_timestamps() {
    use chrono::Utc;

    let before = Utc::now();
    let def = GraphDefinition::new("entry".to_string());
    let after = Utc::now();

    // 创建时间应在调用前后之间
    assert!(def.created_at >= before);
    assert!(def.created_at <= after);
    // 更新时间初始等于创建时间
    assert_eq!(def.updated_at, def.created_at);
}

/// 测试GraphDefinition的metadata字段
#[test]
fn test_definition_metadata() {
    let mut def = GraphDefinition::new("entry".to_string());

    // 添加自定义元数据
    def.metadata
        .insert("version".to_string(), serde_json::json!("1.0"));
    def.metadata
        .insert("author".to_string(), serde_json::json!("test"));

    assert_eq!(def.metadata.get("version").unwrap(), "1.0");
}

// ============================================================================
// 并发测试
// ============================================================================

/// 测试并发保存多个图定义
#[tokio::test]
async fn test_concurrent_saves() {
    let persistence = Arc::new(MemoryPersistence::new());
    let mut handles = vec![];

    // 并发保存10个图定义
    for i in 0..10 {
        let p = persistence.clone();
        handles.push(tokio::spawn(async move {
            let def =
                GraphDefinition::new("entry".to_string()).with_id(format!("concurrent-{}", i));
            p.save(&format!("concurrent-{}", i), &def).await.unwrap();
        }));
    }

    // 等待所有任务完成
    for handle in handles {
        handle.await.unwrap();
    }

    // 验证所有图都已保存
    let list = persistence.list().await.unwrap();
    assert_eq!(list.len(), 10);
}

// ============================================================================
// MongoDB 持久化测试
// ============================================================================

#[cfg(feature = "mongodb-persistence")]
mod mongo_tests {
    use super::*;
    use langchainrust::langgraph::{MongoConfig, MongoPersistence};

    /// 测试MongoDB持久化的保存和加载功能
    #[tokio::test]
    async fn test_mongo_persistence_save_load() {
        let config = MongoTestConfig::get().to_mongo_config();
        let persistence = MongoPersistence::new(config).await.unwrap();

        let definition = GraphDefinition::new("entry".to_string())
            .with_id("mongo-test-001".to_string())
            .with_name("MongoDB Test Workflow".to_string());

        // 保存图定义到MongoDB
        persistence
            .save("mongo-test-001", &definition)
            .await
            .unwrap();
        assert!(persistence.exists("mongo-test-001").await.unwrap());

        // 从MongoDB加载并验证
        let loaded = persistence.load("mongo-test-001").await.unwrap();
        assert_eq!(loaded.id, "mongo-test-001");
        assert_eq!(loaded.name, Some("MongoDB Test Workflow".to_string()));
        assert_eq!(loaded.entry_point, "entry");

        // 清理测试数据
        persistence.delete("mongo-test-001").await.unwrap();
    }

    /// 测试MongoDB持久化的删除功能
    #[tokio::test]
    async fn test_mongo_persistence_delete() {
        let config = MongoTestConfig::get().to_mongo_config();
        let persistence = MongoPersistence::new(config).await.unwrap();
        let definition =
            GraphDefinition::new("entry".to_string()).with_id("mongo-test-del".to_string());

        // 保存后确认存在
        persistence
            .save("mongo-test-del", &definition)
            .await
            .unwrap();
        assert!(persistence.exists("mongo-test-del").await.unwrap());

        // 删除后确认不存在
        persistence.delete("mongo-test-del").await.unwrap();
        assert!(!persistence.exists("mongo-test-del").await.unwrap());
    }

    /// 测试MongoDB持久化的列表功能
    #[tokio::test]
    async fn test_mongo_persistence_list() {
        let config = MongoTestConfig::get().to_mongo_config();
        let persistence = MongoPersistence::new(config).await.unwrap();

        // 清理可能存在的旧测试数据
        for i in 1..=3 {
            let _ = persistence.delete(&format!("mongo-list-{}", i)).await;
        }

        // 保存3个图定义
        for i in 1..=3 {
            let def =
                GraphDefinition::new("entry".to_string()).with_id(format!("mongo-list-{}", i));
            persistence
                .save(&format!("mongo-list-{}", i), &def)
                .await
                .unwrap();
        }

        // 验证列表数量
        let list = persistence.list().await.unwrap();
        let mongo_items: Vec<_> = list
            .iter()
            .filter(|id| id.starts_with("mongo-list-"))
            .collect();
        assert_eq!(mongo_items.len(), 3);

        // 清理测试数据
        for i in 1..=3 {
            let _ = persistence.delete(&format!("mongo-list-{}", i)).await;
        }
    }

    /// 测试加载不存在的图时返回错误
    #[tokio::test]
    async fn test_mongo_persistence_not_found() {
        let config = MongoTestConfig::get().to_mongo_config();
        let persistence = MongoPersistence::new(config).await.unwrap();

        // 加载不存在的图应该返回错误
        let result = persistence.load("mongo-nonexistent").await;
        assert!(result.is_err());
    }

    /// 测试MongoDB的更新操作 (upsert)
    #[tokio::test]
    async fn test_mongo_persistence_upsert() {
        let config = MongoTestConfig::get().to_mongo_config();
        let persistence = MongoPersistence::new(config).await.unwrap();

        // 第一次保存
        let def1 = GraphDefinition::new("entry".to_string())
            .with_id("mongo-upsert".to_string())
            .with_name("Original Name".to_string())
            .with_recursion_limit(25);
        persistence.save("mongo-upsert", &def1).await.unwrap();

        // 第二次保存相同ID，但内容不同
        let def2 = GraphDefinition::new("entry".to_string())
            .with_id("mongo-upsert".to_string())
            .with_name("Updated Name".to_string())
            .with_recursion_limit(50);
        persistence.save("mongo-upsert", &def2).await.unwrap();

        // 验证内容已更新
        let loaded = persistence.load("mongo-upsert").await.unwrap();
        assert_eq!(loaded.name, Some("Updated Name".to_string()));
        assert_eq!(loaded.recursion_limit, 50);

        // 清理测试数据
        persistence.delete("mongo-upsert").await.unwrap();
    }

    /// 测试MongoDB持久化的并发保存
    #[tokio::test]
    async fn test_mongo_concurrent_saves() {
        let config = MongoTestConfig::get().to_mongo_config();
        let persistence = Arc::new(MongoPersistence::new(config).await.unwrap());
        let mut handles = vec![];

        // 并发保存10个图定义
        for i in 0..10 {
            let p = persistence.clone();
            handles.push(tokio::spawn(async move {
                let def = GraphDefinition::new("entry".to_string())
                    .with_id(format!("mongo-concurrent-{}", i));
                p.save(&format!("mongo-concurrent-{}", i), &def)
                    .await
                    .unwrap();
            }));
        }

        // 等待所有任务完成
        for handle in handles {
            handle.await.unwrap();
        }

        // 验证所有图都已保存
        let list = persistence.list().await.unwrap();
        let concurrent_items: Vec<_> = list
            .iter()
            .filter(|id| id.starts_with("mongo-concurrent-"))
            .collect();
        assert_eq!(concurrent_items.len(), 10);

        // 清理测试数据
        for i in 0..10 {
            let _ = persistence.delete(&format!("mongo-concurrent-{}", i)).await;
        }
    }

    /// 测试使用自定义配置创建MongoDB持久化实例
    #[tokio::test]
    async fn test_mongo_custom_config() {
        let config = MongoTestConfig::get().to_mongo_config();
        let persistence = MongoPersistence::new(config).await.unwrap();

        // 验证连接信息
        assert!(!persistence.database_name().is_empty());
        assert!(!persistence.collection_name().is_empty());

        // 测试基本操作
        let def =
            GraphDefinition::new("test".to_string()).with_id("custom-config-test".to_string());
        persistence.save("custom-config-test", &def).await.unwrap();
        assert!(persistence.exists("custom-config-test").await.unwrap());
        persistence.delete("custom-config-test").await.unwrap();
    }

    /// 测试MongoDB持久化完整往返：编译图 -> 保存 -> 加载
    #[tokio::test]
    async fn test_mongo_persistence_roundtrip() {
        let config = MongoTestConfig::get().to_mongo_config();
        let persistence = MongoPersistence::new(config).await.unwrap();

        // 构建简单图
        let graph = GraphBuilder::<AgentState>::new()
            .add_node_fn("process", |s: &AgentState| {
                let mut state = s.clone();
                state.set_output("done".to_string());
                Ok(StateUpdate::full(state))
            })
            .add_edge(START, "process")
            .add_edge("process", END)
            .compile()
            .unwrap();

        let definition = graph.to_definition();

        // 保存到MongoDB
        persistence
            .save("mongo-roundtrip", &definition)
            .await
            .unwrap();

        // 加载并验证数据一致
        let loaded = persistence.load("mongo-roundtrip").await.unwrap();

        assert_eq!(loaded.nodes.len(), definition.nodes.len());
        assert_eq!(loaded.edges.len(), definition.edges.len());
        assert_eq!(loaded.entry_point, definition.entry_point);

        // 清理测试数据
        persistence.delete("mongo-roundtrip").await.unwrap();
    }
}
