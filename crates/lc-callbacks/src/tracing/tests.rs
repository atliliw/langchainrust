#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::backend::{ConsoleTracingBackend, InMemoryTracingBackend};
    use super::super::span::{SpanKind, SpanStatus, SpanTokenUsage, TraceSpan};
    use super::super::tracer::{clear_span_stack, Tracer};
    use super::super::TracingBackend;

    #[test]
    fn test_span_kind_display() {
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
    fn test_span_creation_lifecycle() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            let guard = tracer.start("test_span", SpanKind::Chain);
            assert!(!guard.id().is_empty());
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.name, "test_span");
        assert_eq!(span.kind, SpanKind::Chain);
        assert!(span.start_time.is_some());
        assert!(span.end_time.is_some());
        assert!(span.latency_ms.is_some());
        assert_eq!(span.status, SpanStatus::Ok);
    }

    #[test]
    fn test_parent_child_relationship() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            let root = tracer.start("root", SpanKind::Chain);
            let root_id = root.id().to_string();

            {
                let child = tracer.start_child("child", SpanKind::Tool);
                assert_eq!(child.parent_id(), Some(root_id.as_str()));
            }
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 2);

        // Find root and child
        let root = spans.iter().find(|s| s.name == "root").unwrap();
        let child = spans.iter().find(|s| s.name == "child").unwrap();

        assert!(root.parent_id.is_none());
        assert_eq!(child.parent_id.as_deref(), Some(root.id.as_str()));
    }

    #[test]
    fn test_start_child_without_parent_creates_root() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        // Clear any leftover state
        clear_span_stack();

        {
            let child = tracer.start_child("orphan", SpanKind::Tool);
            // No parent on the stack, so parent_id should be None
            assert!(child.parent_id().is_none());
        }
    }

    #[test]
    fn test_span_guard_with_tokens() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            let usage = SpanTokenUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            };
            let guard = tracer.start("llm_call", SpanKind::Llm).with_tokens(usage);
            guard.end();
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        let tokens = spans[0].tokens.as_ref().unwrap();
        assert_eq!(tokens.prompt_tokens, 10);
        assert_eq!(tokens.completion_tokens, 20);
        assert_eq!(tokens.total_tokens, 30);
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
                .end();
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        let meta = &spans[0].metadata;
        assert_eq!(meta["model"], "gpt-4");
        assert_eq!(meta["temperature"], 0.7);
    }

    #[test]
    fn test_span_guard_set_error() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            let mut guard = tracer.start("call", SpanKind::Tool);
            guard.set_error("tool failed");
            guard.end();
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(
            spans[0].status,
            SpanStatus::Error("tool failed".to_string())
        );
    }

    #[test]
    fn test_span_guard_raii_drop() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        // Do NOT call .end() — rely on Drop
        {
            let _guard = tracer.start("auto_dropped", SpanKind::Chain);
        }

        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
        assert!(spans[0].end_time.is_some(), "Drop should end the span");
        assert!(
            spans[0].latency_ms.is_some(),
            "Drop should calculate latency"
        );
    }

    #[test]
    fn test_in_memory_backend_trace_tree() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        // Build: root -> child1, root -> child2 -> grandchild
        let root_id;

        {
            let root = tracer.start("root", SpanKind::Agent);
            root_id = root.id().to_string();

            let child1 = tracer.start_child("child1", SpanKind::Tool);
            drop(child1);

            let child2 = tracer.start_child("child2", SpanKind::Chain);

            let grandchild = tracer.start_child("grandchild", SpanKind::Llm);
            drop(grandchild);

            drop(child2);
            drop(root);
        }

        let tree = backend.trace_tree(&root_id).unwrap();
        assert_eq!(tree.span.name, "root");
        assert_eq!(tree.children.len(), 2);

        // Find child1 and child2
        let child1_node = tree
            .children
            .iter()
            .find(|c| c.span.name == "child1")
            .unwrap();
        let child2_node = tree
            .children
            .iter()
            .find(|c| c.span.name == "child2")
            .unwrap();

        assert!(child1_node.children.is_empty());
        assert_eq!(child2_node.children.len(), 1);
        assert_eq!(child2_node.children[0].span.name, "grandchild");
    }

    #[test]
    fn test_in_memory_backend_trace_tree_not_found() {
        let backend = InMemoryTracingBackend::new();
        assert!(backend.trace_tree("nonexistent").is_none());
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
    fn test_tracer_current_span_id() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        // Clear any leftover state
        clear_span_stack();

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

    #[test]
    fn test_start_child_with_explicit_parent() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        // Clear any leftover state
        clear_span_stack();

        let root = tracer.start("root", SpanKind::Agent);
        let root_id = root.id().to_string();

        // Use explicit parent ID without relying on thread-local state
        let child = tracer.start_child_with_parent("child", SpanKind::Tool, root_id.clone());
        assert_eq!(child.parent_id(), Some(root_id.as_str()));
    }

    #[test]
    fn test_span_serialization_roundtrip() {
        let span = TraceSpan {
            id: "test-id".to_string(),
            parent_id: Some("parent-id".to_string()),
            name: "test_span".to_string(),
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
            metadata: serde_json::json!({"model": "gpt-4"}),
            status: SpanStatus::Ok,
        };

        let json = serde_json::to_string(&span).unwrap();
        let deserialized: TraceSpan = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, span.id);
        assert_eq!(deserialized.name, span.name);
        assert_eq!(deserialized.kind, span.kind);
        assert_eq!(deserialized.tokens, span.tokens);
    }

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
    fn test_tracer_flush() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());
        // flush should not panic
        tracer.flush();
    }

    #[test]
    fn test_span_guard_end_called_twice_is_safe() {
        let backend = Arc::new(InMemoryTracingBackend::new());
        let tracer = Tracer::new(backend.clone());

        {
            let guard = tracer.start("double_end", SpanKind::Chain);
            guard.end();
            // Drop will also fire but should be a no-op because `dropped` is set
        }

        // Should have exactly 1 span, not duplicated
        let spans = backend.spans();
        assert_eq!(spans.len(), 1);
    }
}
