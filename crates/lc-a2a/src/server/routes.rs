//! A2A task-route handlers: `tasks/send|send_continue|get|cancel|list` + `workflow/run`.
//!
//! These methods are dispatched by `A2AServer::dispatch` and drive the A2A task
//! lifecycle; they work alongside `execution.rs` (task execution) and
//! `handlers.rs` (response assembly).

use super::*;

impl A2AServer {
    /// Handle `tasks/send`: create a new async task (or continue an existing
    /// one) and run the chain in the background.
    ///
    /// New tasks are acknowledged immediately with a `submitted` task and the
    /// chain runs in a spawned task. When the request carries a `taskId`
    /// (continuation, P2-2/P2-3), the message is appended to that task's
    /// history and it is re-run. A `message_id` makes the call idempotent
    /// (P1-6). A `skillId` param routes to a different chain (P2-4).
    pub(crate) async fn handle_tasks_send(
        &self,
        req: A2ARequest,
        owner: Option<&str>,
    ) -> A2AResponse {
        let params = match req.params.clone() {
            Some(p) => p,
            None => {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::invalid_params("Missing params for tasks/send"),
                )
            }
        };

        let message = extract_message(&params);

        // P1-6: idempotent re-send — a repeated message_id returns the
        // already-created task instead of running the chain a second time.
        // The reservation makes the check-then-act atomic: only the caller
        // that wins the claim proceeds, so concurrent retries with the same
        // message_id cannot both run the chain.
        let message_id = req.message_id().map(|s| s.to_string());
        let mut reserved = false;
        if let Some(mid) = &message_id {
            match self.reserve_message_id(mid).await {
                Ok(Some(existing_id)) => {
                    match self.store.get(&existing_id).await {
                        Ok(Some(stored)) => {
                            if !self.caller_owns(owner, &stored.task) {
                                return forbidden(req.id, "caller does not own the existing task");
                            }
                            return A2AResponse::ok(req.id, json!({ "task": stored.task }));
                        }
                        // Evicted between reserve and here: forget and treat
                        // this as a fresh send.
                        _ => self.abort_message_id(mid).await,
                    }
                }
                Ok(None) => reserved = true,
                Err(()) => {
                    return A2AResponse::from_error_data(
                        req.id,
                        A2AErrorData::new(-32000, "message_id is already being processed; retry"),
                    );
                }
            }
        }

        // P2-2/P2-3: continuation — append to an existing task and re-run.
        if let Some(task_id) = req.task_id().map(|s| s.to_string()) {
            return self
                .handle_tasks_send_continue(req, task_id, message, message_id, owner)
                .await;
        }

        // Fresh task.
        let task_id = uuid::Uuid::new_v4().to_string();
        let mut task = A2ATask::new(task_id.clone(), message);
        // C-A2A-1: the resolved caller identity is authoritative. The boundary
        // (`handle_a2a_request` / `handle_a2a_request_authenticated`) resolved it
        // from the authenticated principal (always) or from a self-reported
        // metadata `owner` (only after `with_trust_metadata_owner` opted in), so
        // no raw `req.owner()` read happens here.
        if let Some(owner) = owner {
            task = task.with_owner(owner);
        }
        let mut stored = StoredTask::new(task.clone());
        // B1 (0.26.0): persist the `skillId` so a later `tasks/send {taskId,
        // message}` continuation resolves the *same* skill route this task
        // started on (previously a multi-turn skill task silently switched to
        // the default chain on resume).
        let skill_id = params.get("skillId").and_then(Value::as_str);
        if let Some(sid) = skill_id {
            stored = stored.with_skill_id(Some(sid));
        }
        // P1-5: carry the request's trace id onto the task so the trace can be
        // correlated after creation.
        if let Some(trace_id) = req.trace_id() {
            stored = stored.with_trace_id(trace_id);
        }
        if self.store.upsert(stored).await.is_err() {
            if let Some(mid) = &message_id {
                if reserved {
                    self.abort_message_id(mid).await;
                }
            }
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::internal_error("task store write failed"),
            );
        }

        if let Some(mid) = &message_id {
            if reserved {
                self.finish_message_id(mid, &task_id).await;
            }
        }

        // P2-4: route by skill if a router is configured. `skill_id` was already
        // resolved above (and persisted onto the StoredTask for B1).
        let chain = self.resolve_chain(skill_id);

        let store = self.store.clone();
        let bus = self.event_bus.clone();
        let history = task.message_history().into_owned();
        let spawned_id = task_id.clone();
        // B7 point 6: register the task's cancellation handle so `tasks/cancel`
        // can terminate this in-flight execution, not just mark status.
        let cancellations = self.cancellations.clone();
        let cancel = Arc::new(tokio::sync::Notify::new());
        cancellations
            .write()
            .await
            .insert(spawned_id.clone(), cancel.clone());
        tokio::spawn(async move {
            run_task(&store, chain, &spawned_id, history, bus, cancel).await;
            cancellations.write().await.remove(&spawned_id);
        });

        A2AResponse::ok(req.id, json!({ "task": task }))
    }

    /// Handle `tasks/sendSubscribe`: create a task and acknowledge a streaming
    /// subscription (P2-9 subscribe flow).
    ///
    /// The request params are identical to `tasks/send` (a `message`). The task
    /// is created and acknowledged in the JSON-RPC result — that ack is the
    /// subscription confirmation — and the chain runs in the background the
    /// same way. Each subsequent status transition (and the final artifact) is
    /// then pushed as a [`TaskPushNotification`](crate::protocol::TaskPushNotification)
    /// over the SSE/broadcast bus ([`A2AServer::with_streaming`] +
    /// [`A2AServer::subscribe`]) instead of being polled by the client.
    ///
    /// Ownership (P1-4), idempotency (P1-6) and continuation (P2-2/P2-3)
    /// semantics are shared with `tasks/send`.
    pub(crate) async fn handle_tasks_send_subscribe(
        &self,
        req: A2ARequest,
        owner: Option<&str>,
    ) -> A2AResponse {
        self.handle_tasks_send(req, owner).await
    }

    /// Append a message to an existing task and re-run it (P2-2/P2-3).
    ///
    /// Only tasks in the `input-required` state can be resumed — this is the
    /// one A2A flow where the client sends another message to the same task
    /// (to supply the information the agent asked for). Continuing a
    /// `working` task would spawn a second background worker that races the
    /// first, and terminal tasks cannot change, so both are rejected with
    /// `-32004`.
    pub(crate) async fn handle_tasks_send_continue(
        &self,
        req: A2ARequest,
        task_id: String,
        message: A2AMessage,
        message_id: Option<String>,
        owner: Option<&str>,
    ) -> A2AResponse {
        // H3: serialize resumes per task so two concurrent resumes of the same
        // `input-required` task cannot both pass the state check and spawn
        // racing workers. The guard releases the claim on every exit path.
        let Some(_inflight) = self.begin_resume(&task_id) else {
            // G3 (0.20.0): this call was rejected before doing any work, so its
            // `message_id` reservation must be released — otherwise the key
            // leaks as permanently in-flight and a retry can never claim it.
            self.release_resume_id(&message_id).await;
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::new(-32000, "task is already being resumed; retry"),
            );
        };

        let mut stored = match self.store.get(&task_id).await {
            Ok(Some(s)) => s,
            Ok(None) | Err(_) => {
                self.release_resume_id(&message_id).await;
                return task_not_found(req.id, &task_id);
            }
        };

        // P1-4: ownership.
        if !self.caller_owns(owner, &stored.task) {
            self.release_resume_id(&message_id).await;
            return forbidden(req.id, "caller does not own this task");
        }

        // P2-3: an input-required or auth-required task can be resumed with new
        // input. B4 adds the `AuthRequired → Working` edge; supplying the missing
        // input / in-band credentials is the natural trigger, and
        // `can_transition_to(&Working)` below accepts it.
        if !matches!(
            stored.task.status,
            TaskStatus::InputRequired | TaskStatus::AuthRequired
        ) {
            self.release_resume_id(&message_id).await;
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::new(
                    -32004,
                    format!("Cannot continue task in state {}", stored.task.status),
                ),
            );
        }

        // B1: capture the originating skill route before `stored` is moved into
        // the upsert below so the resumed worker resolves the same chain.
        let stored_skill = stored.skill_id.clone();
        stored.task.push_message(message);
        if stored.task.status.can_transition_to(&TaskStatus::Working) {
            stored.task.status = TaskStatus::Working;
        }
        stored.touch();
        let task = stored.task.clone();
        if self.store.upsert(stored).await.is_err() {
            self.release_resume_id(&message_id).await;
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::internal_error("task store write failed"),
            );
        }

        // P1-6: idempotency — a retried resume with the same message_id must
        // not append the message a second time.
        if let Some(mid) = message_id {
            self.finish_message_id(&mid, &task_id).await;
        }

        let store = self.store.clone();
        // B1 (0.26.0): a continuation must return to the *same* skill route the
        // task started on. `handle_tasks_send` persists the `skillId` onto the
        // StoredTask; resolve from it instead of blindly falling back to the
        // default chain (which is what the previous `resolve_chain(None)` did, so
        // a multi-turn skill task silently switched models/chains mid-conversation).
        let chain = self.resolve_chain(stored_skill.as_deref());
        let bus = self.event_bus.clone();
        let history = task.message_history().into_owned();
        let spawned_id = task_id.clone();
        // B7 point 6: cancellation handle for the resumed run (see comment in
        // `handle_tasks_send`).
        let cancellations = self.cancellations.clone();
        let cancel = Arc::new(tokio::sync::Notify::new());
        cancellations
            .write()
            .await
            .insert(spawned_id.clone(), cancel.clone());
        tokio::spawn(async move {
            run_task(&store, chain, &spawned_id, history, bus, cancel).await;
            cancellations.write().await.remove(&spawned_id);
        });

        A2AResponse::ok(req.id, json!({ "task": task }))
    }

    /// Handle `tasks/get`: return a task by ID.
    ///
    /// Ownership (P1-4): tasks carrying an `owner` are only readable by the
    /// matching caller.
    pub(crate) async fn handle_tasks_get(
        &self,
        req: A2ARequest,
        owner: Option<&str>,
    ) -> A2AResponse {
        let task_id = req
            .params
            .as_ref()
            .and_then(|p| p.get("taskId"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if task_id.is_empty() {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::invalid_params("Missing taskId parameter"),
            );
        }

        self.cleanup_expired_tasks().await;

        match self.store.get(task_id).await {
            Ok(Some(stored)) => {
                if !self.caller_owns(owner, &stored.task) {
                    return forbidden(req.id, "caller does not own this task");
                }
                task_details_response(req.id, &stored)
            }
            Ok(None) => task_not_found(req.id, task_id),
            Err(_) => A2AResponse::from_error_data(
                req.id,
                A2AErrorData::internal_error("task store read failed"),
            ),
        }
    }

    /// Handle `tasks/cancel`: cancel a task by ID.
    ///
    /// Cancellation is only legal from a non-terminal state; cancelling an
    /// already-terminal task is an idempotent no-op that returns it unchanged.
    /// Ownership (P1-4) is enforced like `tasks/get`.
    pub(crate) async fn handle_tasks_cancel(
        &self,
        req: A2ARequest,
        owner: Option<&str>,
    ) -> A2AResponse {
        let task_id = req
            .params
            .as_ref()
            .and_then(|p| p.get("taskId"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if task_id.is_empty() {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::invalid_params("Missing taskId parameter"),
            );
        }

        let stored = match self.store.get(task_id).await {
            Ok(Some(s)) => s,
            Ok(None) | Err(_) => return task_not_found(req.id, task_id),
        };

        if !self.caller_owns(owner, &stored.task) {
            return forbidden(req.id, "caller does not own this task");
        }

        if stored.task.status.is_terminal() {
            // Idempotent: already finished, return it unchanged.
            return A2AResponse::ok(req.id, json!({ "task": stored.task }));
        }
        // 0.22.0 audit fix: perform the transition through a compare-and-update
        // so the check and the write are atomic — a concurrent terminal write
        // (e.g. the chain completing between our read and our write) can no
        // longer be clobbered by this cancel.
        let mut cancelled = stored.clone();
        cancelled.task.status = TaskStatus::Cancelled;
        cancelled.touch();
        let cancelled_task = cancelled.task.clone();
        match self.store.compare_and_update(task_id, cancelled).await {
            Ok(true) => {
                publish_status(&self.event_bus, task_id, TaskStatus::Cancelled, None);
                // B7 point 6: propagate the cancellation to the in-flight chain
                // so execution actually stops (select! drops the invoke future).
                if let Some(cancel) = self.cancellations.read().await.get(task_id) {
                    cancel.notify_one();
                }
                A2AResponse::ok(req.id, json!({ "task": cancelled_task }))
            }
            Ok(false) => {
                // Lost the race: report the current state of the task.
                match self.store.get(task_id).await {
                    Ok(Some(now)) if now.task.status.is_terminal() => {
                        A2AResponse::ok(req.id, json!({ "task": now.task }))
                    }
                    Ok(Some(now)) => A2AResponse::from_error_data(
                        req.id,
                        A2AErrorData::new(
                            -32002,
                            format!("Cannot cancel task in state {}", now.task.status),
                        ),
                    ),
                    _ => task_not_found(req.id, task_id),
                }
            }
            Err(_) => A2AResponse::from_error_data(
                req.id,
                A2AErrorData::internal_error("task store write failed"),
            ),
        }
    }

    /// Handle `tasks/list`: list stored tasks, optionally filtered by
    /// `owner` / `status` params (P1-1).
    ///
    /// Ownership (P1-4) is enforced per task, like `tasks/get` / `tasks/cancel`:
    /// a caller never sees tasks it does not own, and tasks without an owner
    /// are open to any caller. A caller that carries an `owner` identity and
    /// does not pass an explicit `owner` param additionally narrows the query
    /// to its own tasks.
    pub(crate) async fn handle_tasks_list(
        &self,
        req: A2ARequest,
        owner: Option<&str>,
    ) -> A2AResponse {
        self.cleanup_expired_tasks().await;

        let mut filter = TaskFilter::new();
        if let Some(params) = &req.params {
            if let Some(owner) = params.get("owner").and_then(Value::as_str) {
                filter = filter.with_owner(owner);
            }
            if let Some(status) = params.get("status").and_then(Value::as_str) {
                if let Ok(ts) =
                    serde_json::from_value::<TaskStatus>(Value::String(status.to_string()))
                {
                    filter = filter.with_statuses(vec![ts]);
                }
            }
        }
        if filter.owner.is_none() {
            // C-A2A-1: narrow to the resolved caller identity (authenticated
            // principal, or the opt-in metadata owner) rather than the raw
            // `req.owner()`.
            if let Some(o) = owner {
                filter = filter.with_owner(o);
            }
        }

        // L101: tasks/list pagination. `limit` clamps the page size; `cursor`
        // is the id of the last task on the previous page, so the next page
        // continues after it in the same stable (id-sorted) order.
        let limit = parse_pagination_limit(req.params.as_ref().and_then(|p| p.get("limit")));
        let cursor = req
            .params
            .as_ref()
            .and_then(|p| p.get("cursor"))
            .and_then(Value::as_str);

        match self.store.list(&filter).await {
            Ok(stored) => {
                // P1-4 ownership is enforced per task: `tasks/list` has no
                // single task to check against, so filter the result set.
                // Tasks without an owner remain open to any caller; owned
                // tasks are only visible to a matching owner identity.
                //
                // Pagination requires a deterministic order across calls, so
                // the candidates are sorted by task id (a monotonic, unique
                // key) and the cursor addresses position within that order.
                let mut visible: Vec<&A2ATask> = stored
                    .iter()
                    .filter(|s| self.caller_owns(owner, &s.task))
                    .map(|s| &s.task)
                    .collect();
                visible.sort_by(|a, b| a.id.cmp(&b.id));

                let start = cursor
                    .map(|c| {
                        visible
                            .iter()
                            .position(|t| t.id.as_str() == c)
                            .map(|i| i + 1)
                            .unwrap_or(0)
                    })
                    .unwrap_or(0);
                let end = (start + limit).min(visible.len());
                let page = &visible[start..end];
                let has_more = end < visible.len();
                let next_cursor = page.last().map(|t| t.id.as_str());

                let mut body = serde_json::Map::new();
                body.insert("tasks".into(), json!(page));
                body.insert("hasMore".into(), json!(has_more));
                if has_more {
                    if let Some(nc) = next_cursor {
                        body.insert("nextCursor".into(), json!(nc));
                    }
                }
                A2AResponse::ok(req.id, Value::Object(body))
            }
            Err(_) => A2AResponse::from_error_data(
                req.id,
                A2AErrorData::internal_error("task store read failed"),
            ),
        }
    }

    /// Handle `tasks/runWorkflow`: execute an ordered multi-step workflow and
    /// aggregate per-step results (P2-8).
    ///
    /// A workflow is backed by a single task (id = `workflow.workflow_id` or a
    /// fresh UUID). Steps run in order, each routed to the chain selected by
    /// its `skill_id` (falling back to the default chain). A step failure
    /// marks the workflow task `failed` and stops execution — the results
    /// aggregated up to that point are still returned. Ownership (P1-4) and
    /// trace propagation (P1-5) apply to the backing task like `tasks/send`.
    pub(crate) async fn handle_workflow_run(
        &self,
        req: A2ARequest,
        owner: Option<&str>,
    ) -> A2AResponse {
        let params = match req.params.clone() {
            Some(p) => p,
            None => {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::invalid_params("Missing params for tasks/runWorkflow"),
                );
            }
        };
        let workflow: A2AWorkflow = match params.get("workflow") {
            Some(w) => match serde_json::from_value(w.clone()) {
                Ok(wf) => wf,
                Err(_) => {
                    return A2AResponse::from_error_data(
                        req.id,
                        A2AErrorData::invalid_params("Malformed workflow"),
                    );
                }
            },
            None => {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::invalid_params("Missing workflow for tasks/runWorkflow"),
                );
            }
        };
        if workflow.steps.is_empty() {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::invalid_params("Workflow has no steps"),
            );
        }
        // G2 (0.20.0): a client-supplied workflow is unbounded input; cap the
        // number of steps so a single request cannot drive an arbitrarily long
        // serial chain run.
        if workflow.steps.len() > MAX_WORKFLOW_STEPS {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::invalid_params(format!(
                    "Workflow exceeds the maximum of {MAX_WORKFLOW_STEPS} steps"
                )),
            );
        }

        // Create the backing task, propagating owner and trace id.
        let task_id = workflow
            .workflow_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let mut task = A2ATask::new(
            task_id.clone(),
            A2AMessage::user(format!(
                "workflow: {}",
                workflow.name.as_deref().unwrap_or("unnamed")
            )),
        )
        .with_status(TaskStatus::Working);
        // C-A2A-1: see `handle_tasks_send` — the resolved caller identity is
        // authoritative; no raw `req.owner()` read happens here.
        if let Some(owner) = owner {
            task = task.with_owner(owner);
        }
        let mut stored = StoredTask::new(task.clone());
        if let Some(trace_id) = req.trace_id() {
            stored = stored.with_trace_id(trace_id);
        }

        // H2/F4: a client-controlled `workflow_id` must not overwrite a task
        // the caller does not own (P1-4). The CAS below atomically inserts the
        // backing task, or replaces it only when the existing task is
        // claimable by this caller — no owner → open to any caller; same owner
        // → re-run allowed; another caller's owner → rejected. This closes the
        // cross-tenant overwrite TOCTOU where a concurrent writer could slip
        // its task between a separate ownership check and the upsert
        // (0.22.0 audit fix).
        match self.store.upsert_claimed(&task_id, stored).await {
            Ok(true) => {}
            Ok(false) => return forbidden(req.id, "caller does not own the existing task"),
            Err(_) => {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::internal_error("task store write failed"),
                );
            }
        }
        publish_status(&self.event_bus, &task_id, TaskStatus::Working, None);

        // G2 (0.20.0): execute in the background (matching `tasks/send`), so the
        // handler is not blocked by a serial chain run. The client polls
        // `tasks/get` for the outcome.
        let steps = workflow.steps.clone();
        let chains: Vec<Arc<dyn BaseChain>> = steps
            .iter()
            .map(|s| self.resolve_chain(s.skill_id.as_deref()))
            .collect();
        let store = self.store.clone();
        let event_bus = self.event_bus.clone();
        let spawned_id = task_id.clone();
        tokio::spawn(async move {
            run_workflow(&store, steps, chains, &spawned_id, event_bus).await;
        });

        A2AResponse::ok(req.id, json!({ "task": task }))
    }
}

/// Parses the `limit` param for `tasks/list` pagination (L101).
///
/// Absent or unparseable values fall back to the default page size; everything
/// is clamped to `[1, MAX_PAGE_SIZE]` so a caller cannot force the store to
/// materialize an unbounded result set in one response.
fn parse_pagination_limit(value: Option<&Value>) -> usize {
    const DEFAULT_PAGE_SIZE: usize = 50;
    const MAX_PAGE_SIZE: usize = 1000;
    value
        .and_then(|v| v.as_u64())
        .map(|n| (n as usize).clamp(1, MAX_PAGE_SIZE))
        .unwrap_or(DEFAULT_PAGE_SIZE)
}
