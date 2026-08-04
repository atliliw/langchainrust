// tests/unit/tracing.rs
//! Tracing system unit tests
//!
//! Tests for the agent observability / deep tracing system:
//! - Span creation and lifecycle
//! - Parent-child relationships
//! - InMemoryTracingBackend trace tree construction
//! - SpanGuard RAII behavior
//! - ConsoleTracingBackend
//! - Serialization roundtrip

use langchainrust::callbacks::tracing::{
    ConsoleTracingBackend, InMemoryTracingBackend, SpanKind, SpanStatus, SpanTokenUsage, TraceNode,
    TraceSpan, Tracer, TracingBackend,
};
use std::sync::Arc;

// ============================================================================
// SpanKind tests
// ============================================================================

#[test]
fn test_span_kind_variants() {
    // Verify all SpanKind variants can be created and displayed
    assert_eq!(format!("{}", SpanKind::Llm), "llm");
    assert_eq!(format!("{}", SpanKind::Chain), "chain");
    assert_eq!(format!("{}", SpanKind::Tool), "tool");
    assert_eq!(format!("{}", SpanKind::Retriever), "retriever");
    assert_eq!(format!("{}", SpanKind::Agent), "agent");
    assert_eq!(
        format!("{}", SpanKind::Custom("embedding".into())),
        "custom:embedding"
    );
}

#[test]
fn test_span_kind_serialization() {
    // Verify SpanKind serializes/deserializes correctly
    let kinds = vec![
        SpanKind::Llm,
        SpanKind::Chain,
        SpanKind::Tool,
        SpanKind::Retriever,
        SpanKind::Agent,
        SpanKind::Custom("my_kind".into()),
    ];
    for kind in kinds {
        let json = serde_json::to_string(&kind).unwrap();
        let deserialized: SpanKind = serde_json::from_str(&json).unwrap();
        assert_eq!(kind, deserialized);
    }
}

// ============================================================================
// SpanTokenUsage tests
// ============================================================================

#[test]
fn test_span_token_usage() {
    let usage = SpanTokenUsage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 50);
    assert_eq!(usage.total_tokens, 150);
}

#[test]
fn test_span_token_usage_serialization() {
    let usage = SpanTokenUsage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    let json = serde_json::to_string(&usage).unwrap();
    let deserialized: SpanTokenUsage = serde_json::from_str(&json).unwrap();
    assert_eq!(usage, deserialized);
}

// ============================================================================
// Span creation and lifecycle
// ============================================================================

#[test]
fn test_span_creation_root() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    {
        let guard = tracer.start("root_span", SpanKind::Chain);
        assert!(!guard.id().is_empty());
        assert!(guard.parent_id().is_none());
    }

    let spans = backend.spans();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert_eq!(span.name, "root_span");
    assert_eq!(span.kind, SpanKind::Chain);
    assert!(span.parent_id.is_none());
    assert!(span.start_time.is_some());
    assert!(span.end_time.is_some());
    assert!(span.latency_ms.is_some());
    assert_eq!(span.status, SpanStatus::Ok);
}

#[test]
fn test_span_lifecycle_with_explicit_end() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    let guard = tracer.start("explicit_end", SpanKind::Llm);
    let id = guard.id().to_string();
    guard.end();

    let spans = backend.spans();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].id, id);
    assert!(spans[0].end_time.is_some());
    assert!(spans[0].latency_ms.is_some());
}

#[test]
fn test_span_lifecycle_with_drop() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    {
        let _guard = tracer.start("auto_dropped", SpanKind::Tool);
        // Not calling .end() — relying on Drop
    }

    let spans = backend.spans();
    assert_eq!(spans.len(), 1);
    assert!(spans[0].end_time.is_some(), "Drop should end the span");
    assert!(
        spans[0].latency_ms.is_some(),
        "Drop should calculate latency"
    );
}

// ============================================================================
// Parent-child relationships
// ============================================================================

#[test]
fn test_parent_child_via_start_child() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    {
        let root = tracer.start("root", SpanKind::Agent);
        let root_id = root.id().to_string();

        {
            let child = tracer.start_child("child_tool", SpanKind::Tool);
            assert_eq!(child.parent_id(), Some(root_id.as_str()));
        }
    }

    let spans = backend.spans();
    assert_eq!(spans.len(), 2);

    let root = spans.iter().find(|s| s.name == "root").unwrap();
    let child = spans.iter().find(|s| s.name == "child_tool").unwrap();

    assert!(root.parent_id.is_none());
    assert_eq!(child.parent_id.as_deref(), Some(root.id.as_str()));
}

#[test]
fn test_parent_child_with_explicit_parent_id() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    let root = tracer.start("root", SpanKind::Chain);
    let root_id = root.id().to_string();

    let child = tracer.start_child_with_parent("child", SpanKind::Llm, root_id.clone());
    assert_eq!(child.parent_id(), Some(root_id.as_str()));
}

#[test]
fn test_start_child_without_active_span_creates_root() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    // Clear any leftover thread-local state
    langchainrust::callbacks::tracing::clear_span_stack();

    {
        let orphan = tracer.start_child("orphan", SpanKind::Tool);
        assert!(orphan.parent_id().is_none());
    }
}

#[test]
fn test_nested_parent_child() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    {
        let root = tracer.start("root", SpanKind::Agent);
        let root_id = root.id().to_string();

        let child = tracer.start_child("child_chain", SpanKind::Chain);
        let child_id = child.id().to_string();

        let grandchild = tracer.start_child("grandchild_llm", SpanKind::Llm);
        assert_eq!(grandchild.parent_id(), Some(child_id.as_str()));
        // The child should have root as parent
        assert_eq!(child.parent_id(), Some(root_id.as_str()));
    }

    let spans = backend.spans();
    assert_eq!(spans.len(), 3);

    let root = spans.iter().find(|s| s.name == "root").unwrap();
    let child = spans.iter().find(|s| s.name == "child_chain").unwrap();
    let grandchild = spans.iter().find(|s| s.name == "grandchild_llm").unwrap();

    assert!(root.parent_id.is_none());
    assert_eq!(child.parent_id.as_deref(), Some(root.id.as_str()));
    assert_eq!(grandchild.parent_id.as_deref(), Some(child.id.as_str()));
}

// ============================================================================
// SpanGuard builder methods
// ============================================================================

#[test]
fn test_span_guard_with_tokens() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    {
        let usage = SpanTokenUsage {
            prompt_tokens: 50,
            completion_tokens: 100,
            total_tokens: 150,
        };
        tracer
            .start("llm_call", SpanKind::Llm)
            .with_tokens(usage)
            .end();
    }

    let spans = backend.spans();
    assert_eq!(spans.len(), 1);
    let tokens = spans[0].tokens.as_ref().unwrap();
    assert_eq!(tokens.prompt_tokens, 50);
    assert_eq!(tokens.completion_tokens, 100);
    assert_eq!(tokens.total_tokens, 150);
}

#[test]
fn test_span_guard_with_cost() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    {
        tracer
            .start("llm_call", SpanKind::Llm)
            .with_cost(0.003)
            .end();
    }

    let spans = backend.spans();
    assert_eq!(spans.len(), 1);
    assert!((spans[0].cost.unwrap() - 0.003).abs() < f64::EPSILON);
}

#[test]
fn test_span_guard_with_metadata() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    {
        tracer
            .start("call", SpanKind::Chain)
            .with_metadata("model", serde_json::json!("gpt-4"))
            .with_metadata("temperature", serde_json::json!(0.7))
            .with_metadata("retry_count", serde_json::json!(3))
            .end();
    }

    let spans = backend.spans();
    assert_eq!(spans.len(), 1);
    let meta = &spans[0].metadata;
    assert_eq!(meta["model"], "gpt-4");
    assert_eq!(meta["temperature"], 0.7);
    assert_eq!(meta["retry_count"], 3);
}

#[test]
fn test_span_guard_set_error() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    {
        let mut guard = tracer.start("failing_tool", SpanKind::Tool);
        guard.set_error("tool execution failed: timeout");
        guard.end();
    }

    let spans = backend.spans();
    assert_eq!(spans.len(), 1);
    assert_eq!(
        spans[0].status,
        SpanStatus::Error("tool execution failed: timeout".to_string())
    );
}

#[test]
fn test_span_guard_combined_builder() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    {
        let usage = SpanTokenUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };
        tracer
            .start("full_span", SpanKind::Llm)
            .with_tokens(usage)
            .with_cost(0.001)
            .with_metadata("provider", serde_json::json!("openai"))
            .end();
    }

    let spans = backend.spans();
    assert_eq!(spans.len(), 1);
    let span = &spans[0];
    assert!(span.tokens.is_some());
    assert!(span.cost.is_some());
    assert_eq!(span.metadata["provider"], "openai");
    assert_eq!(span.status, SpanStatus::Ok);
}

// ============================================================================
// InMemoryTracingBackend trace tree
// ============================================================================

#[test]
fn test_trace_tree_simple() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    let root_id;
    {
        let root = tracer.start("root", SpanKind::Agent);
        root_id = root.id().to_string();

        let child1 = tracer.start_child("child1", SpanKind::Tool);
        drop(child1);

        let child2 = tracer.start_child("child2", SpanKind::Chain);
        drop(child2);

        drop(root);
    }

    let tree = backend.trace_tree(&root_id).unwrap();
    assert_eq!(tree.span.name, "root");
    assert_eq!(tree.children.len(), 2);
}

#[test]
fn test_trace_tree_nested() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    let root_id;
    {
        let root = tracer.start("root", SpanKind::Agent);
        root_id = root.id().to_string();

        let child = tracer.start_child("child", SpanKind::Chain);
        let _grandchild = tracer.start_child("grandchild", SpanKind::Llm);

        drop(child); // also drops grandchild via scope
        drop(root);
    }

    let tree = backend.trace_tree(&root_id).unwrap();
    assert_eq!(tree.span.name, "root");
    assert_eq!(tree.children.len(), 1);
    assert_eq!(tree.children[0].span.name, "child");
    assert_eq!(tree.children[0].children.len(), 1);
    assert_eq!(tree.children[0].children[0].span.name, "grandchild");
}

#[test]
fn test_trace_tree_not_found() {
    let backend = InMemoryTracingBackend::new();
    assert!(backend.trace_tree("nonexistent_id").is_none());
}

#[test]
fn test_in_memory_backend_clear() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    {
        let _guard = tracer.start("temp", SpanKind::Chain);
    }

    assert_eq!(backend.spans().len(), 1);
    backend.clear();
    assert!(backend.spans().is_empty());
}

#[test]
fn test_in_memory_backend_flush() {
    let backend = InMemoryTracingBackend::new();
    // flush should not panic
    backend.flush();
}

// ============================================================================
// Tracer current span tracking
// ============================================================================

#[test]
fn test_tracer_current_span_id() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    // Clear any leftover thread-local state
    langchainrust::callbacks::tracing::clear_span_stack();

    assert!(tracer.current_span_id().is_none());

    let root = tracer.start("root", SpanKind::Chain);
    let root_id = root.id().to_string();
    assert_eq!(tracer.current_span_id(), Some(root_id.clone()));

    let child = tracer.start_child("child", SpanKind::Tool);
    let child_id = child.id().to_string();
    assert_eq!(tracer.current_span_id(), Some(child_id));

    drop(child);
    assert_eq!(tracer.current_span_id(), Some(root_id));

    drop(root);
    assert!(tracer.current_span_id().is_none());
}

// ============================================================================
// SpanGuard RAII safety
// ============================================================================

#[test]
fn test_span_guard_end_then_drop_is_safe() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    {
        let guard = tracer.start("double_end", SpanKind::Chain);
        guard.end();
        // Drop will also fire but should be a no-op
    }

    // Should have exactly 1 span, not duplicated
    let spans = backend.spans();
    assert_eq!(spans.len(), 1);
}

// ============================================================================
// ConsoleTracingBackend
// ============================================================================

#[test]
fn test_console_backend_does_not_panic() {
    let backend = ConsoleTracingBackend;
    let span = TraceSpan {
        id: "test".to_string(),
        parent_id: None,
        name: "test_span".to_string(),
        kind: SpanKind::Chain,
        start_time: Some("2025-01-01T00:00:00Z".to_string()),
        end_time: None,
        tokens: None,
        cost: None,
        latency_ms: None,
        metadata: serde_json::Value::Null,
        status: SpanStatus::Ok,
    };

    // Should not panic
    backend.start_span(&span);
    backend.flush();
}

#[test]
fn test_console_backend_end_span_does_not_panic() {
    let backend = ConsoleTracingBackend;
    let span = TraceSpan {
        id: "test".to_string(),
        parent_id: None,
        name: "test_span".to_string(),
        kind: SpanKind::Llm,
        start_time: Some("2025-01-01T00:00:00Z".to_string()),
        end_time: Some("2025-01-01T00:00:01Z".to_string()),
        tokens: None,
        cost: None,
        latency_ms: Some(1000),
        metadata: serde_json::Value::Null,
        status: SpanStatus::Error("timeout".to_string()),
    };

    // Should not panic
    backend.end_span(&span);
}

// ============================================================================
// Serialization roundtrip
// ============================================================================

#[test]
fn test_trace_span_serialization_roundtrip() {
    let span = TraceSpan {
        id: "test-id-123".to_string(),
        parent_id: Some("parent-id-456".to_string()),
        name: "llm_call".to_string(),
        kind: SpanKind::Llm,
        start_time: Some("2025-01-01T00:00:00Z".to_string()),
        end_time: Some("2025-01-01T00:00:01Z".to_string()),
        tokens: Some(SpanTokenUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        }),
        cost: Some(0.005),
        latency_ms: Some(1000),
        metadata: serde_json::json!({"model": "gpt-4", "temperature": 0.7}),
        status: SpanStatus::Ok,
    };

    let json = serde_json::to_string(&span).unwrap();
    let deserialized: TraceSpan = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.id, span.id);
    assert_eq!(deserialized.name, span.name);
    assert_eq!(deserialized.kind, span.kind);
    assert_eq!(deserialized.tokens, span.tokens);
    assert_eq!(deserialized.cost, span.cost);
    assert_eq!(deserialized.latency_ms, span.latency_ms);
    assert_eq!(deserialized.status, span.status);
}

#[test]
fn test_trace_span_error_serialization() {
    let span = TraceSpan {
        id: "err-id".to_string(),
        parent_id: None,
        name: "failed_call".to_string(),
        kind: SpanKind::Tool,
        start_time: Some("2025-01-01T00:00:00Z".to_string()),
        end_time: Some("2025-01-01T00:00:00Z".to_string()),
        tokens: None,
        cost: None,
        latency_ms: Some(0),
        metadata: serde_json::Value::Null,
        status: SpanStatus::Error("connection refused".to_string()),
    };

    let json = serde_json::to_string(&span).unwrap();
    let deserialized: TraceSpan = serde_json::from_str(&json).unwrap();
    assert_eq!(
        deserialized.status,
        SpanStatus::Error("connection refused".to_string())
    );
}

#[test]
fn test_trace_node_serialization() {
    let node = TraceNode {
        span: TraceSpan {
            id: "root".to_string(),
            parent_id: None,
            name: "root".to_string(),
            kind: SpanKind::Agent,
            start_time: Some("2025-01-01T00:00:00Z".to_string()),
            end_time: Some("2025-01-01T00:00:01Z".to_string()),
            tokens: None,
            cost: None,
            latency_ms: Some(1000),
            metadata: serde_json::Value::Null,
            status: SpanStatus::Ok,
        },
        children: vec![TraceNode {
            span: TraceSpan {
                id: "child".to_string(),
                parent_id: Some("root".to_string()),
                name: "child".to_string(),
                kind: SpanKind::Tool,
                start_time: Some("2025-01-01T00:00:00Z".to_string()),
                end_time: Some("2025-01-01T00:00:00Z".to_string()),
                tokens: None,
                cost: None,
                latency_ms: Some(500),
                metadata: serde_json::Value::Null,
                status: SpanStatus::Ok,
            },
            children: vec![],
        }],
    };

    let json = serde_json::to_string(&node).unwrap();
    let deserialized: TraceNode = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.span.name, "root");
    assert_eq!(deserialized.children.len(), 1);
    assert_eq!(deserialized.children[0].span.name, "child");
}

// ============================================================================
// Tracer clone
// ============================================================================

#[test]
fn test_tracer_clone_shares_backend() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer1 = Tracer::new(backend.clone());
    let tracer2 = tracer1.clone();

    {
        let _guard = tracer1.start("from_tracer1", SpanKind::Chain);
    }
    {
        let _guard = tracer2.start("from_tracer2", SpanKind::Tool);
    }

    // Both tracers share the same backend
    let spans = backend.spans();
    assert_eq!(spans.len(), 2);
}

// ============================================================================
// Multi-level trace tree
// ============================================================================

#[test]
fn test_complex_trace_tree() {
    let backend = Arc::new(InMemoryTracingBackend::new());
    let tracer = Tracer::new(backend.clone());

    // Build: Agent -> Chain -> [Tool, LLM -> Retriever]
    // Spans must be dropped in LIFO order for correct thread-local stack management
    let root_id;
    {
        let root = tracer.start("agent", SpanKind::Agent);
        root_id = root.id().to_string();

        let chain = tracer.start_child("chain", SpanKind::Chain);
        let tool = tracer.start_child("calculator", SpanKind::Tool);
        drop(tool);

        let llm = tracer.start_child("llm", SpanKind::Llm);
        let retriever = tracer.start_child("retriever", SpanKind::Retriever);
        drop(retriever);
        drop(llm);

        drop(chain);
        drop(root);
    }

    let tree = backend.trace_tree(&root_id).unwrap();
    assert_eq!(tree.span.name, "agent");
    assert_eq!(tree.children.len(), 1); // chain
    let chain_node = &tree.children[0];
    assert_eq!(chain_node.span.name, "chain");
    assert_eq!(chain_node.children.len(), 2); // tool + llm

    let tool_node = chain_node
        .children
        .iter()
        .find(|c| c.span.name == "calculator")
        .unwrap();
    assert!(tool_node.children.is_empty());

    let llm_node = chain_node
        .children
        .iter()
        .find(|c| c.span.name == "llm")
        .unwrap();
    assert_eq!(llm_node.children.len(), 1);
    assert_eq!(llm_node.children[0].span.name, "retriever");
}
