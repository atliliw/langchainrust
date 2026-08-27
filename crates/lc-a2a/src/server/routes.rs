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
    pub(crate) async fn handle_tasks_send(&self, req: A2ARequest) -> A2AResponse {
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
                            if !self.caller_owns(&req, &stored.task) {
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
                .handle_tasks_send_continue(req, task_id, message, message_id)
                .await;
        }

        // Fresh task.
        let task_id = uuid::Uuid::new_v4().to_string();
        let mut task = A2ATask::new(task_id.clone(), message);
        if let Some(owner) = req.owner() {
            task = task.with_owner(owner);
        }
        let mut stored = StoredTask::new(task.clone());
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

        // P2-4: route by skill if a router is configured.
        let skill_id = params.get("skillId").and_then(Value::as_str);
        let chain = self.resolve_chain(skill_id);

        let store = self.store.clone();
        let bus = self.event_bus.clone();
        let history = task.message_history().into_owned();
        let spawned_id = task_id.clone();
        tokio::spawn(async move {
            run_task(&store, chain, &spawned_id, history, bus).await;
        });

        A2AResponse::ok(req.id, json!({ "task": task }))
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
    ) -> A2AResponse {
        // H3: serialize resumes per task so two concurrent resumes of the same
        // `input-required` task cannot both pass the state check and spawn
        // racing workers. The guard releases the claim on every exit path.
        let Some(_inflight) = self.begin_resume(&task_id) else {
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
        if !self.caller_owns(&req, &stored.task) {
            self.release_resume_id(&message_id).await;
            return forbidden(req.id, "caller does not own this task");
        }

        // P2-3: only an input-required task can be resumed with new input.
        if stored.task.status != TaskStatus::InputRequired {
            self.release_resume_id(&message_id).await;
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::new(
                    -32004,
                    format!("Cannot continue task in state {}", stored.task.status),
                ),
            );
        }

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
        let chain = self.resolve_chain(None);
        let bus = self.event_bus.clone();
        let history = task.message_history().into_owned();
        let spawned_id = task_id.clone();
        tokio::spawn(async move {
            run_task(&store, chain, &spawned_id, history, bus).await;
        });

        A2AResponse::ok(req.id, json!({ "task": task }))
    }

    /// Handle `tasks/get`: return a task by ID.
    ///
    /// Ownership (P1-4): tasks carrying an `owner` are only readable by the
    /// matching caller.
    pub(crate) async fn handle_tasks_get(&self, req: A2ARequest) -> A2AResponse {
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
                if !self.caller_owns(&req, &stored.task) {
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
    pub(crate) async fn handle_tasks_cancel(&self, req: A2ARequest) -> A2AResponse {
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

        let mut stored = match self.store.get(task_id).await {
            Ok(Some(s)) => s,
            Ok(None) | Err(_) => return task_not_found(req.id, task_id),
        };

        if !self.caller_owns(&req, &stored.task) {
            return forbidden(req.id, "caller does not own this task");
        }

        if stored.task.status.is_terminal() {
            // Idempotent: already finished, return it unchanged.
            return A2AResponse::ok(req.id, json!({ "task": stored.task }));
        }
        if stored.task.status.can_transition_to(&TaskStatus::Cancelled) {
            stored.task.status = TaskStatus::Cancelled;
            stored.touch();
            if self.store.upsert(stored.clone()).await.is_err() {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::internal_error("task store write failed"),
                );
            }
            publish_status(&self.event_bus, task_id, TaskStatus::Cancelled, None);
            A2AResponse::ok(req.id, json!({ "task": stored.task }))
        } else {
            A2AResponse::from_error_data(
                req.id,
                A2AErrorData::new(
                    -32002,
                    format!("Cannot cancel task in state {}", stored.task.status),
                ),
            )
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
    pub(crate) async fn handle_tasks_list(&self, req: A2ARequest) -> A2AResponse {
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
            if let Some(owner) = req.owner() {
                filter = filter.with_owner(owner);
            }
        }

        match self.store.list(&filter).await {
            Ok(stored) => {
                // P1-4 ownership is enforced per task: `tasks/list` has no
                // single task to check against, so filter the result set.
                // Tasks without an owner remain open to any caller; owned
                // tasks are only visible to a matching owner identity.
                let tasks: Vec<&A2ATask> = stored
                    .iter()
                    .filter(|s| self.caller_owns(&req, &s.task))
                    .map(|s| &s.task)
                    .collect();
                A2AResponse::ok(req.id, json!({ "tasks": tasks }))
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
    pub(crate) async fn handle_workflow_run(&self, req: A2ARequest) -> A2AResponse {
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

        // Create the backing task, propagating owner and trace id.
        let task_id = workflow
            .workflow_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // H2: a client-controlled `workflow_id` must not overwrite a task the
        // caller does not own (P1-4). An existing task with this id is only
        // re-runnable by its owner; anonymous callers may not clobber owned
        // tasks, and may only create fresh ids.
        if let Ok(Some(existing)) = self.store.get(&task_id).await {
            if !self.caller_owns(&req, &existing.task) {
                return forbidden(req.id, "caller does not own the existing task");
            }
        }
        let mut task = A2ATask::new(
            task_id.clone(),
            A2AMessage::user(format!(
                "workflow: {}",
                workflow.name.as_deref().unwrap_or("unnamed")
            )),
        )
        .with_status(TaskStatus::Working);
        if let Some(owner) = req.owner() {
            task = task.with_owner(owner);
        }
        let mut stored = StoredTask::new(task.clone());
        if let Some(trace_id) = req.trace_id() {
            stored = stored.with_trace_id(trace_id);
        }
        if self.store.upsert(stored).await.is_err() {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::internal_error("task store write failed"),
            );
        }
        publish_status(&self.event_bus, &task_id, TaskStatus::Working, None);

        // Execute steps in order, aggregating per-step outputs.
        let mut results = serde_json::Map::new();
        let mut failure: Option<(String, String)> = None; // (step_id, message)
        for step in &workflow.steps {
            let chain = self.resolve_chain(step.skill_id.as_deref());
            let input = build_chain_input(&step.message.content, chain.as_ref());
            match chain.invoke(input).await {
                Ok(result) => {
                    let output = extract_output(&result);
                    results.insert(step.id.clone(), Value::String(output));
                }
                Err(e) => {
                    failure = Some((step.id.clone(), e.to_string()));
                    break;
                }
            }
        }

        // Finalize the backing task with the aggregated outcome.
        let mut finalize = match self.store.get(&task_id).await {
            Ok(Some(s)) => s,
            _ => {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::internal_error("workflow task vanished"),
                );
            }
        };
        let aggregated = results
            .values()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        match failure {
            Some((step_id, message)) => {
                finalize.task.status = TaskStatus::Failed;
                finalize.error = Some(format!("step `{step_id}` failed: {message}"));
                let error = finalize.error.clone();
                if self.store.upsert(finalize.clone()).await.is_err() {
                    return A2AResponse::from_error_data(
                        req.id,
                        A2AErrorData::internal_error("task store write failed"),
                    );
                }
                publish_status(
                    &self.event_bus,
                    &task_id,
                    TaskStatus::Failed,
                    error.as_deref(),
                );
                A2AResponse::ok(
                    req.id,
                    json!({
                        "task": finalize.task,
                        "error": error,
                        "results": Value::Object(results),
                    }),
                )
            }
            None => {
                finalize.task.status = TaskStatus::Completed;
                finalize.result = Some(A2ATaskResult::new(aggregated));
                finalize.error = None;
                if self.store.upsert(finalize.clone()).await.is_err() {
                    return A2AResponse::from_error_data(
                        req.id,
                        A2AErrorData::internal_error("task store write failed"),
                    );
                }
                publish_status(&self.event_bus, &task_id, TaskStatus::Completed, None);
                if let Some(result) = finalize.result.clone() {
                    publish_artifact(&self.event_bus, &task_id, result);
                }
                A2AResponse::ok(
                    req.id,
                    json!({
                        "task": finalize.task,
                        "result": finalize.result,
                        "results": Value::Object(results),
                    }),
                )
            }
        }
    }
}
