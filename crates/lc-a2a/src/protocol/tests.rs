use super::*;

#[test]
fn agent_card_new() {
    let card = AgentCard::new("test-agent", "A test agent", "http://localhost:8080");
    assert_eq!(card.name, "test-agent");
    assert_eq!(card.description, "A test agent");
    assert_eq!(card.url, "http://localhost:8080");
    assert!(card.skills.is_empty());
    assert_eq!(card.protocol_version, "0.3.0");
}

#[test]
fn agent_card_with_skills() {
    let card = AgentCard::new("agent", "desc", "http://localhost")
        .with_skill(AgentSkill::new("s1", "text-generation", "Generates text"))
        .with_skill(AgentSkill::new("s2", "tool-use", "Uses tools"));
    assert_eq!(card.skills.len(), 2);
    assert_eq!(card.skills[0].id, "s1");
    assert_eq!(card.skills[0].name, "text-generation");
    assert_eq!(card.skills[1].description, "Uses tools");
}

#[test]
fn agent_card_serialization() {
    let card = AgentCard::new("agent", "desc", "http://localhost")
        .with_skill(AgentSkill::new("s1", "text-generation", "Generates text"))
        .with_security_schemes(serde_json::json!({ "bearerAuth": { "scheme": "bearer" } }))
        .with_interfaces(serde_json::json!({ "sse": false }));
    let json = serde_json::to_string(&card).unwrap();
    assert!(json.contains("\"name\":\"agent\""));
    assert!(json.contains("\"skills\""));
    assert!(json.contains("\"text-generation\""));
    assert!(json.contains("\"protocolVersion\":\"0.3.0\""));
    assert!(json.contains("\"securitySchemes\""));
    assert!(json.contains("\"bearerAuth\""));
    assert!(json.contains("\"interfaces\""));
}

#[test]
fn agent_card_deserialization() {
    let json = r#"{"name":"agent","description":"desc","url":"http://localhost","skills":[{"id":"s1","name":"text-generation","description":"Generates text"}],"protocolVersion":"0.3.0"}"#;
    let card: AgentCard = serde_json::from_str(json).unwrap();
    assert_eq!(card.name, "agent");
    assert_eq!(card.protocol_version, "0.3.0");
    assert_eq!(card.skills.len(), 1);
    assert_eq!(card.skills[0].id, "s1");
}

#[test]
fn task_status_serialization() {
    let statuses = vec![
        TaskStatus::Submitted,
        TaskStatus::Working,
        TaskStatus::InputRequired,
        TaskStatus::Completed,
        TaskStatus::Failed,
        TaskStatus::Cancelled,
        TaskStatus::Rejected,
        TaskStatus::AuthRequired,
        TaskStatus::Expired,
    ];
    let json = serde_json::to_string(&statuses).unwrap();
    assert!(json.contains("\"submitted\""));
    assert!(json.contains("\"working\""));
    assert!(json.contains("\"input-required\""));
    assert!(json.contains("\"completed\""));
    assert!(json.contains("\"failed\""));
    assert!(json.contains("\"cancelled\""));
    assert!(json.contains("\"rejected\""));
    assert!(json.contains("\"auth-required\""));
    assert!(json.contains("\"expired\""));
}

#[test]
fn task_status_display() {
    assert_eq!(TaskStatus::Submitted.to_string(), "submitted");
    assert_eq!(TaskStatus::Working.to_string(), "working");
    assert_eq!(TaskStatus::InputRequired.to_string(), "input-required");
    assert_eq!(TaskStatus::Completed.to_string(), "completed");
    assert_eq!(TaskStatus::Failed.to_string(), "failed");
    assert_eq!(TaskStatus::Cancelled.to_string(), "cancelled");
    assert_eq!(TaskStatus::Rejected.to_string(), "rejected");
    assert_eq!(TaskStatus::AuthRequired.to_string(), "auth-required");
    assert_eq!(TaskStatus::Expired.to_string(), "expired");
}

#[test]
fn task_status_legal_transitions() {
    use TaskStatus::*;
    // Plan-mandated transitions.
    assert!(Submitted.can_transition_to(&Working));
    assert!(Submitted.can_transition_to(&Rejected));
    assert!(Working.can_transition_to(&Completed));
    assert!(Working.can_transition_to(&Failed));
    assert!(Working.can_transition_to(&InputRequired));
    assert!(Working.can_transition_to(&Cancelled));
    assert!(InputRequired.can_transition_to(&Working));
    assert!(AuthRequired.can_transition_to(&Submitted));
    // Practical additions: cancel/expire from non-terminal states.
    assert!(Submitted.can_transition_to(&Cancelled));
    assert!(InputRequired.can_transition_to(&Cancelled));
    assert!(Submitted.can_transition_to(&Expired));
    assert!(Working.can_transition_to(&Expired));
}

#[test]
fn task_status_illegal_transitions() {
    use TaskStatus::*;
    // No backward / terminal / skipped transitions.
    assert!(!Working.can_transition_to(&Submitted));
    assert!(!Completed.can_transition_to(&Working));
    assert!(!Failed.can_transition_to(&Working));
    assert!(!Cancelled.can_transition_to(&Working));
    assert!(!Rejected.can_transition_to(&Working));
    assert!(!Expired.can_transition_to(&Working));
    assert!(!Submitted.can_transition_to(&Completed));
    assert!(!Working.can_transition_to(&Rejected));
}

#[test]
fn task_status_terminal() {
    use TaskStatus::*;
    for s in [Completed, Failed, Cancelled, Rejected, Expired] {
        assert!(s.is_terminal(), "{s:?} should be terminal");
    }
    for s in [Submitted, Working, InputRequired, AuthRequired] {
        assert!(!s.is_terminal(), "{s:?} should not be terminal");
    }
}

#[test]
fn a2a_task_details() {
    let details = A2ATaskDetails {
        task: A2ATask::new("t1", A2AMessage::user("hi")).with_status(TaskStatus::Completed),
        result: Some(A2ATaskResult::new("done")),
        error: None,
    };
    assert_eq!(details.task.status, TaskStatus::Completed);
    assert_eq!(details.result.unwrap().output, "done");
}

#[test]
fn a2a_message_user() {
    let msg = A2AMessage::user("hello");
    assert_eq!(msg.role, "user");
    assert_eq!(msg.content, "hello");
}

#[test]
fn a2a_message_agent() {
    let msg = A2AMessage::agent("response");
    assert_eq!(msg.role, "agent");
    assert_eq!(msg.content, "response");
}

#[test]
fn a2a_task_new() {
    let task = A2ATask::new("task-1", A2AMessage::user("hello"));
    assert_eq!(task.id, "task-1");
    assert_eq!(task.status, TaskStatus::Submitted);
    assert_eq!(task.message.content, "hello");
}

#[test]
fn a2a_task_with_status() {
    let task = A2ATask::new("task-1", A2AMessage::user("hello")).with_status(TaskStatus::Completed);
    assert_eq!(task.status, TaskStatus::Completed);
}

#[test]
fn a2a_task_result() {
    let result = A2ATaskResult::new("output text");
    assert_eq!(result.output, "output text");
}

#[test]
fn a2a_request_new() {
    let req = A2ARequest::new(1, "tasks/send", None);
    assert_eq!(req.jsonrpc, "2.0");
    assert_eq!(req.id, 1);
    assert_eq!(req.method, "tasks/send");
    assert!(req.params.is_none());
}

#[test]
fn a2a_request_send_task() {
    let msg = A2AMessage::user("hello");
    let req = A2ARequest::send_task(1, &msg);
    assert_eq!(req.method, "tasks/send");
    assert!(req.params.is_some());
    let params = req.params.unwrap();
    assert!(params.get("message").is_some());
}

#[test]
fn a2a_request_get_task() {
    let req = A2ARequest::get_task(2, "task-123");
    assert_eq!(req.method, "tasks/get");
    let params = req.params.unwrap();
    assert_eq!(params["taskId"], "task-123");
}

#[test]
fn a2a_request_cancel_task() {
    let req = A2ARequest::cancel_task(3, "task-456");
    assert_eq!(req.method, "tasks/cancel");
    let params = req.params.unwrap();
    assert_eq!(params["taskId"], "task-456");
}

#[test]
fn a2a_request_serialization_skips_none_params() {
    let req = A2ARequest::new(1, "tasks/send", None);
    let json = serde_json::to_string(&req).unwrap();
    assert!(!json.contains("params"));
}

#[test]
fn a2a_response_ok() {
    let resp = A2AResponse::ok(1, serde_json::json!({"status": "completed"}));
    assert!(!resp.is_error());
    assert!(resp.result.is_some());
    assert!(resp.error.is_none());
}

#[test]
fn a2a_response_error() {
    let resp = A2AResponse::error(1, -32601, "Method not found");
    assert!(resp.is_error());
    assert!(resp.result.is_none());
    let err = resp.error.unwrap();
    assert_eq!(err.code, -32601);
}

#[test]
fn a2a_response_into_result_ok() {
    let resp = A2AResponse::ok(1, serde_json::json!({"output": "done"}));
    let result = resp.into_result();
    assert!(result.is_ok());
    assert_eq!(result.unwrap()["output"], "done");
}

#[test]
fn a2a_response_into_result_err() {
    let resp = A2AResponse::error(1, -32601, "Method not found");
    let result = resp.into_result();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code, -32601);
}

#[test]
fn a2a_response_serialization() {
    let resp = A2AResponse::ok(1, serde_json::json!({"status": "completed"}));
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("\"jsonrpc\":\"2.0\""));
    assert!(json.contains("\"id\":1"));
    assert!(json.contains("\"result\""));
    assert!(!json.contains("\"error\""));
}

#[test]
fn a2a_error_data_display() {
    let err = A2AErrorData::new(-1, "boom");
    assert_eq!(format!("{}", err), "A2A Error [-1]: boom");
}

#[test]
fn a2a_error_data_standard_errors() {
    let err = A2AErrorData::method_not_found();
    assert_eq!(err.code, -32601);

    let err = A2AErrorData::invalid_params("bad input");
    assert_eq!(err.code, -32602);
    assert!(err.message.contains("bad input"));

    let err = A2AErrorData::internal_error("oops");
    assert_eq!(err.code, -32603);
}

#[test]
fn roundtrip_request_json() {
    let req = A2ARequest::send_task(42, &A2AMessage::user("test"));
    let json = serde_json::to_string(&req).unwrap();
    let parsed: A2ARequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.id, 42);
    assert_eq!(parsed.method, "tasks/send");
}

#[test]
fn roundtrip_response_json() {
    let resp = A2AResponse::ok(7, serde_json::json!({"task": {"id": "t1"}}));
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: A2AResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.id, 7);
    assert!(!parsed.is_error());
}

// ---- P1-3 / P2-8: AgentCard extensions ----

#[test]
fn agent_card_new_extension_fields_default() {
    let card = AgentCard::new("a", "d", "http://localhost");
    assert!(card.signature.is_none());
    assert!(card.data_class.is_none());
    assert!(card.jurisdiction.is_none());
    assert!(card.capabilities.is_empty());
}

#[test]
fn agent_card_with_signature_and_extensions() {
    let card = AgentCard::new("a", "d", "http://localhost")
        .with_signature("sig-v1")
        .with_data_class("confidential")
        .with_jurisdiction("EU")
        .with_capability("streaming-sse")
        .with_capability("input-required-resume");
    assert_eq!(card.signature.as_deref(), Some("sig-v1"));
    assert_eq!(card.data_class.as_deref(), Some("confidential"));
    assert_eq!(card.jurisdiction.as_deref(), Some("EU"));
    assert_eq!(card.capabilities.len(), 2);

    let json = serde_json::to_string(&card).unwrap();
    assert!(json.contains("\"signature\":\"sig-v1\""));
    assert!(json.contains("\"dataClass\":\"confidential\""));
    assert!(json.contains("\"jurisdiction\":\"EU\""));
    assert!(json.contains("\"capabilities\""));
}

#[test]
fn agent_card_deserialize_without_extensions() {
    // Backward compatibility: a card without the new fields must still parse.
    let json =
        r#"{"name":"a","description":"d","url":"http://localhost","protocolVersion":"0.3.0"}"#;
    let card: AgentCard = serde_json::from_str(json).unwrap();
    assert!(card.signature.is_none());
    assert!(card.capabilities.is_empty());
}

// ---- P1-4 / P2-2: A2ATask extensions ----

#[test]
fn a2a_task_new_populates_history() {
    let task = A2ATask::new("t1", A2AMessage::user("hello"));
    assert_eq!(task.message_history().len(), 1);
    assert_eq!(task.message_history()[0].content, "hello");
    assert!(task.owner.is_none());
}

#[test]
fn a2a_task_push_message_appends_history() {
    let mut task = A2ATask::new("t1", A2AMessage::user("hello"));
    task.push_message(A2AMessage::agent("hi there"));
    task.push_message(A2AMessage::user("continue"));
    let history = task.message_history();
    assert_eq!(history.len(), 3);
    assert_eq!(history[1].role, "agent");
    assert_eq!(history[2].content, "continue");
}

#[test]
fn a2a_task_with_owner() {
    let task = A2ATask::new("t1", A2AMessage::user("hi")).with_owner("org-a");
    assert_eq!(task.owner.as_deref(), Some("org-a"));
    let json = serde_json::to_string(&task).unwrap();
    assert!(json.contains("\"owner\":\"org-a\""));
}

#[test]
fn a2a_task_deserialize_without_owner_messages() {
    // Old single-message wire payload: history falls back to `message`.
    let json = r#"{"id":"t1","message":{"role":"user","content":"hi"},"status":"submitted"}"#;
    let task: A2ATask = serde_json::from_str(json).unwrap();
    assert!(task.owner.is_none());
    assert!(task.messages.is_empty());
    assert_eq!(task.message_history().len(), 1);
    assert_eq!(task.message_history()[0].content, "hi");
}

// ---- P1-5 / P1-6: A2ARequest metadata & idempotency ----

#[test]
fn a2a_request_with_trace_id() {
    let req = A2ARequest::send_task(1, &A2AMessage::user("hi")).with_trace_id("abc123");
    assert_eq!(req.trace_id(), Some("abc123"));
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("\"trace_id\":\"abc123\""));
}

#[test]
fn a2a_request_with_owner() {
    let req = A2ARequest::send_task(1, &A2AMessage::user("hi")).with_owner("org-a");
    assert_eq!(req.owner(), Some("org-a"));
}

#[test]
fn a2a_request_message_id_from_metadata() {
    let req = A2ARequest::send_task(1, &A2AMessage::user("hi")).with_message_id("msg-1");
    assert_eq!(req.message_id(), Some("msg-1"));
}

#[test]
fn a2a_request_message_id_from_params_fallback() {
    // A2A wire convention: messageId at top level of params.
    let mut req = A2ARequest::send_task(1, &A2AMessage::user("hi"));
    req.params = Some(serde_json::json!({ "messageId": "wire-1" }));
    assert_eq!(req.message_id(), Some("wire-1"));
}

#[test]
fn a2a_request_continue_task() {
    let req = A2ARequest::continue_task(5, "task-9", &A2AMessage::user("more"));
    assert_eq!(req.method, "tasks/send");
    assert_eq!(req.task_id(), Some("task-9"));
    let params = req.params.unwrap();
    assert!(params.get("message").is_some());
}

#[test]
fn a2a_request_metadata_skipped_when_none() {
    let req = A2ARequest::send_task(1, &A2AMessage::user("hi"));
    let json = serde_json::to_string(&req).unwrap();
    assert!(!json.contains("metadata"));
}

// ---- P2-1: TaskPushNotification ----

#[test]
fn push_notification_status_serialization() {
    let n = TaskPushNotification::status("t1", TaskStatus::Working);
    let json = serde_json::to_string(&n).unwrap();
    assert!(json.contains("\"kind\":\"status-update\""));
    assert!(json.contains("\"status\":\"working\""));
    assert_eq!(n.id(), "t1");
    assert_eq!(n.status_value(), Some(TaskStatus::Working));
}

#[test]
fn push_notification_artifact_serialization() {
    let n = TaskPushNotification::artifact("t1", A2ATaskResult::new("partial"));
    let json = serde_json::to_string(&n).unwrap();
    assert!(json.contains("\"kind\":\"artifact-update\""));
    assert!(json.contains("\"output\":\"partial\""));
    assert_eq!(n.status_value(), None);
}

#[test]
fn push_notification_roundtrip() {
    for original in [
        TaskPushNotification::status_with_error("t1", TaskStatus::Failed, "boom"),
        TaskPushNotification::artifact("t2", A2ATaskResult::new("chunk")),
        TaskPushNotification::status("t3", TaskStatus::Completed),
    ] {
        let json = serde_json::to_string(&original).unwrap();
        let parsed: TaskPushNotification = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id(), original.id());
    }
}

// ---- P2-8: TraceContext ----

#[test]
fn trace_context_roundtrip() {
    let tc = TraceContext::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7");
    let tp = tc.to_traceparent();
    assert_eq!(
        tp,
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
    );
    let parsed = TraceContext::parse(&tp).unwrap();
    assert_eq!(parsed, tc);
}

#[test]
fn trace_context_sampled() {
    let tc = TraceContext::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7").sampled();
    assert!(tc.is_sampled());
    assert!(tc.to_traceparent().ends_with("-01"));
}

#[test]
fn trace_context_parse_invalid() {
    assert!(TraceContext::parse("").is_none());
    // Wrong trace id length.
    assert!(TraceContext::parse("00-1234-00f067aa0ba902b7-00").is_none());
    // Non-hex chars.
    assert!(
        TraceContext::parse("00-zz92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00").is_none()
    );
    // Too many fields.
    assert!(
        TraceContext::parse("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00-extra")
            .is_none()
    );
}

// ---- P1-1: TaskFilter ----

#[test]
fn task_filter_empty_matches_all() {
    let f = TaskFilter::new();
    assert!(f.matches(&A2ATask::new("t1", A2AMessage::user("hi"))));
}

#[test]
fn task_filter_by_owner() {
    let f = TaskFilter::new().with_owner("org-a");
    let owned = A2ATask::new("t1", A2AMessage::user("hi")).with_owner("org-a");
    let foreign = A2ATask::new("t2", A2AMessage::user("hi")).with_owner("org-b");
    assert!(f.matches(&owned));
    assert!(!f.matches(&foreign));
    // Tasks with no owner never match an owner filter.
    assert!(!f.matches(&A2ATask::new("t3", A2AMessage::user("hi"))));
}

#[test]
fn task_filter_by_status() {
    let f = TaskFilter::new().with_statuses(vec![TaskStatus::Working]);
    assert!(f.matches(&A2ATask::new("t1", A2AMessage::user("hi")).with_status(TaskStatus::Working)));
    assert!(
        !f.matches(&A2ATask::new("t2", A2AMessage::user("hi")).with_status(TaskStatus::Completed))
    );
}

// ---- P2-8: MessageEnvelope (unified message model) ----

#[test]
fn message_envelope_roundtrips_through_json() {
    let trace = TraceContext::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7").sampled();
    let envelope = MessageEnvelope::new(A2AMessage::user("hello"))
        .with_trace(trace)
        .with_owner("alice")
        .with_header("x-region", "cn-east");

    let json = serde_json::to_string(&envelope).unwrap();
    let decoded: MessageEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.message.role, "user");
    assert_eq!(decoded.message.content, "hello");
    assert_eq!(decoded.owner.as_deref(), Some("alice"));
    assert_eq!(
        decoded.headers.get("x-region").map(String::as_str),
        Some("cn-east")
    );
    let t = decoded.trace.expect("trace present");
    assert_eq!(t.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
    assert!(t.is_sampled());
    assert_eq!(
        t.to_traceparent(),
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    );
}

#[test]
fn message_envelope_minimal_roundtrip() {
    let envelope = MessageEnvelope::new(A2AMessage::agent("hi"));
    let json = serde_json::to_string(&envelope).unwrap();
    let decoded: MessageEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.message.role, "agent");
    assert!(decoded.owner.is_none());
    assert!(decoded.trace.is_none());
    assert!(decoded.headers.is_empty());
    assert_eq!(decoded.into_message().content, "hi");
}

#[test]
fn send_envelope_propagates_owner_and_trace() {
    let trace = TraceContext::new("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7");
    let envelope = MessageEnvelope::new(A2AMessage::user("hi"))
        .with_trace(trace)
        .with_owner("alice");

    let req = A2ARequest::send_envelope(7, &envelope);
    assert_eq!(req.method, "tasks/send");
    assert_eq!(req.owner(), Some("alice"));
    assert_eq!(req.trace_id(), Some("4bf92f3577b34da6a3ce929d0e0e4736"));
}
